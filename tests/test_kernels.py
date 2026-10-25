import pytest
import torch
import torch.nn.functional as F

from gpuforge import kernels
from tests.conftest import requires_cuda


@pytest.mark.parametrize("act", ["none", "relu", "gelu"])
def test_fallback_matches_torch(act):
    a, b, bias = torch.randn(17, 33), torch.randn(33, 9), torch.randn(9)
    ref = {"none": lambda t: t, "relu": F.relu, "gelu": lambda t: F.gelu(t, approximate="tanh")}[act](a @ b + bias)
    torch.testing.assert_close(kernels.gemm_bias_act(a, b, bias, act), ref)


def test_bad_args_rejected():
    with pytest.raises(ValueError):
        kernels.gemm_bias_act(torch.randn(2, 2), torch.randn(2, 2), act="tanh")
    with pytest.raises(ValueError):
        kernels.gemm_bias_act(torch.randn(2, 2), torch.randn(2, 2), variant="magic")


def test_fused_linear_matches_linear_in_eval():
    m = kernels.FusedLinear(16, 8, act="relu").eval()
    x = torch.randn(4, 16)
    with torch.no_grad():
        torch.testing.assert_close(m(x), F.relu(m.inner(x)))


@requires_cuda
@pytest.mark.skipif(not kernels.extension_available(), reason="CUDA extension not built")
@pytest.mark.parametrize("variant", list(kernels.VARIANTS))
@pytest.mark.parametrize("shape", [(1, 1, 1), (63, 65, 67), (256, 128, 512), (1000, 33, 7)])
def test_cuda_gemm_variants(variant, shape):
    m, n, k = shape
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(k, n, device="cuda")
    bias = torch.randn(n, device="cuda")
    ref = F.relu(a.double() @ b.double() + bias.double()).float()
    out = kernels.gemm_bias_act(a, b, bias, "relu", variant)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4 * k ** 0.5)


@requires_cuda
@pytest.mark.skipif(not kernels.extension_available(), reason="CUDA extension not built")
@pytest.mark.parametrize("shape", [(1, 1), (7, 31), (64, 1000), (3, 5000)])
def test_cuda_softmax(shape):
    x = torch.randn(*shape, device="cuda") * 20
    torch.testing.assert_close(kernels.softmax_rows(x), torch.softmax(x.double(), -1).float(), rtol=2e-5, atol=1e-6)
