// PyTorch C++ bindings for the GPUForge CUDA kernels.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void launch_gemm_bias_act(const torch::Tensor& A, const torch::Tensor& B,
                          const c10::optional<torch::Tensor>& bias, torch::Tensor& C,
                          int64_t act, int64_t variant);
void launch_softmax_rows(const torch::Tensor& X, torch::Tensor& Y);

#define CHECK_INPUT(x)                                                        \
  TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");                    \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");                 \
  TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")

torch::Tensor gemm_bias_act(torch::Tensor A, torch::Tensor B,
                            c10::optional<torch::Tensor> bias, int64_t act, int64_t variant) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2-D");
  TORCH_CHECK(A.size(1) == B.size(0), "inner dimensions must match");
  TORCH_CHECK(act >= 0 && act <= 2, "act must be 0 (none), 1 (relu) or 2 (gelu)");
  TORCH_CHECK(variant >= 0 && variant <= 2, "variant must be 0 (naive), 1 (smem) or 2 (regtile)");
  if (bias.has_value()) {
    CHECK_INPUT((*bias));
    TORCH_CHECK(bias->dim() == 1 && bias->size(0) == B.size(1), "bias must have shape [N]");
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(A));
  auto C = torch::empty({A.size(0), B.size(1)}, A.options());
  launch_gemm_bias_act(A, B, bias, C, act, variant);
  return C;
}

torch::Tensor softmax_rows(torch::Tensor X) {
  CHECK_INPUT(X);
  TORCH_CHECK(X.dim() == 2, "X must be 2-D [rows, cols]");
  const at::cuda::OptionalCUDAGuard guard(device_of(X));
  auto Y = torch::empty_like(X);
  launch_softmax_rows(X, Y);
  return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "GPUForge custom CUDA kernels";
  m.def("gemm_bias_act", &gemm_bias_act, "C = act(A @ B + bias)",
        py::arg("A"), py::arg("B"), py::arg("bias") = py::none(), py::arg("act") = 0,
        py::arg("variant") = 2);
  m.def("softmax_rows", &softmax_rows, "Row-wise numerically stable softmax");
}
