// GPUForge custom CUDA kernels.
//
//   1. GEMM + fused bias + activation epilogue, three variants of increasing
//      sophistication so the benchmark can show *why* each optimisation helps:
//        naive    - one thread per output element, global-memory bound
//        smem     - 32x32 shared-memory tiling (data reuse across a block)
//        regtile  - 64x64 block tile, 4x4 register tile per thread
//                   (raises arithmetic intensity; the classic path toward cuBLAS)
//   2. Row-wise numerically-stable softmax using warp-shuffle reductions.
//
// All kernels are FP32 and row-major. They run on the current PyTorch CUDA
// stream so they compose correctly with the rest of a PyTorch graph.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace {

enum Act : int { ACT_NONE = 0, ACT_RELU = 1, ACT_GELU = 2 };

__device__ __forceinline__ float apply_act(float x, int act) {
  if (act == ACT_RELU) return fmaxf(x, 0.f);
  if (act == ACT_GELU) {
    // tanh approximation (matches torch.nn.functional.gelu(approximate="tanh"))
    const float k0 = 0.7978845608028654f;  // sqrt(2/pi)
    const float k1 = 0.044715f;
    return 0.5f * x * (1.f + tanhf(k0 * (x + k1 * x * x * x)));
  }
  return x;
}

// ---------------------------------------------------------------- naive ----
__global__ void gemm_naive_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  const float* __restrict__ bias, float* __restrict__ C,
                                  int M, int N, int K, int act) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc = fmaf(A[(size_t)row * K + k], B[(size_t)k * N + col], acc);
  if (bias) acc += bias[col];
  C[(size_t)row * N + col] = apply_act(acc, act);
}

// ----------------------------------------------------------------- smem ----
template <int TILE>
__global__ void gemm_smem_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                 const float* __restrict__ bias, float* __restrict__ C,
                                 int M, int N, int K, int act) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    const int ak = k0 + tx, bk = k0 + ty;
    As[ty][tx] = (row < M && ak < K) ? A[(size_t)row * K + ak] : 0.f;
    Bs[ty][tx] = (bk < K && col < N) ? B[(size_t)bk * N + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; ++k) acc = fmaf(As[ty][k], Bs[k][tx], acc);
    __syncthreads();
  }
  if (row < M && col < N) {
    if (bias) acc += bias[col];
    C[(size_t)row * N + col] = apply_act(acc, act);
  }
}

// -------------------------------------------------------------- regtile ----
// Block computes a BM x BN tile of C; each of the (BM/TM)*(BN/TN) threads owns
// a TM x TN sub-tile held in registers. A is stored transposed in shared memory
// so the inner loop reads contiguous TM values per thread.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
gemm_regtile_kernel(const float* __restrict__ A, const float* __restrict__ B,
                    const float* __restrict__ bias, float* __restrict__ C,
                    int M, int N, int K, int act) {
  constexpr int NTHREADS = (BM / TM) * (BN / TN);
  __shared__ float As[BK][BM];
  __shared__ float Bs[BK][BN];

  const int tid = threadIdx.x;
  const int tRow = tid / (BN / TN);
  const int tCol = tid % (BN / TN);
  const int bRow = blockIdx.y * BM;
  const int bCol = blockIdx.x * BN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;
  float regA[TM], regB[TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int i = tid; i < BM * BK; i += NTHREADS) {
      const int r = i / BK, c = i % BK;
      const int gr = bRow + r, gc = k0 + c;
      As[c][r] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.f;
    }
    for (int i = tid; i < BK * BN; i += NTHREADS) {
      const int r = i / BN, c = i % BN;
      const int gr = k0 + r, gc = bCol + c;
      Bs[r][c] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i) regA[i] = As[k][tRow * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) regB[j] = Bs[k][tCol * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(regA[i], regB[j], acc[i][j]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int r = bRow + tRow * TM + i;
    if (r >= M) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int c = bCol + tCol * TN + j;
      if (c >= N) continue;
      float v = acc[i][j];
      if (bias) v += bias[c];
      C[(size_t)r * N + c] = apply_act(v, act);
    }
  }
}

// -------------------------------------------------------------- softmax ----
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

// Block-wide reduction: warp reduce -> shared -> first warp reduce -> broadcast.
template <bool IS_MAX>
__device__ float block_reduce(float v, float* shm) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) >> 5;
  v = IS_MAX ? warp_max(v) : warp_sum(v);
  if (lane == 0) shm[wid] = v;
  __syncthreads();
  float r = (threadIdx.x < nwarps) ? shm[threadIdx.x] : (IS_MAX ? -INFINITY : 0.f);
  if (wid == 0) r = IS_MAX ? warp_max(r) : warp_sum(r);
  if (threadIdx.x == 0) shm[0] = r;
  __syncthreads();
  r = shm[0];
  __syncthreads();  // shm is reused by the next reduction
  return r;
}

// One block per row; subtract-max for numerical stability.
__global__ void softmax_rows_kernel(const float* __restrict__ X, float* __restrict__ Y, int cols) {
  __shared__ float shm[32];
  const float* x = X + (size_t)blockIdx.x * cols;
  float* y = Y + (size_t)blockIdx.x * cols;

  float m = -INFINITY;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) m = fmaxf(m, x[i]);
  m = block_reduce<true>(m, shm);

  float s = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) s += __expf(x[i] - m);
  s = block_reduce<false>(s, shm);

  const float inv = 1.f / s;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) y[i] = __expf(x[i] - m) * inv;
}

inline int cdiv(int a, int b) { return (a + b - 1) / b; }

}  // namespace

// ------------------------------------------------------------- launchers ---
void launch_gemm_bias_act(const torch::Tensor& A, const torch::Tensor& B,
                          const c10::optional<torch::Tensor>& bias, torch::Tensor& C,
                          int64_t act, int64_t variant) {
  const int M = A.size(0), K = A.size(1), N = B.size(1);
  const float* bptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  if (variant == 0) {
    dim3 block(16, 16), grid(cdiv(N, 16), cdiv(M, 16));
    gemm_naive_kernel<<<grid, block, 0, stream>>>(A.data_ptr<float>(), B.data_ptr<float>(), bptr,
                                                  C.data_ptr<float>(), M, N, K, (int)act);
  } else if (variant == 1) {
    constexpr int T = 32;
    dim3 block(T, T), grid(cdiv(N, T), cdiv(M, T));
    gemm_smem_kernel<T><<<grid, block, 0, stream>>>(A.data_ptr<float>(), B.data_ptr<float>(), bptr,
                                                    C.data_ptr<float>(), M, N, K, (int)act);
  } else {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
    dim3 block((BM / TM) * (BN / TN)), grid(cdiv(N, BN), cdiv(M, BM));
    gemm_regtile_kernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), bptr, C.data_ptr<float>(), M, N, K, (int)act);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_softmax_rows(const torch::Tensor& X, torch::Tensor& Y) {
  const int rows = X.size(0), cols = X.size(1);
  const int threads = cols >= 1024 ? 1024 : (cols >= 256 ? 256 : 128);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  softmax_rows_kernel<<<rows, threads, 0, stream>>>(X.data_ptr<float>(), Y.data_ptr<float>(), cols);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
