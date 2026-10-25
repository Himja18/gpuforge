import numpy as np
import torch
import torch.nn.functional as F

from gpuforge.validation import numerics as nm
from gpuforge.validation import suite


def test_gamma_monotone_and_infinite_past_limit():
    assert nm.gamma(10, 2 ** -24) < nm.gamma(1000, 2 ** -24)
    assert nm.gamma(2 ** 12, 2 ** -11) == float("inf")


def test_conv_reference_matches_torch():
    rng = np.random.default_rng(0)
    x, w, b = rng.standard_normal((2, 4, 9, 9)), rng.standard_normal((5, 4, 3, 3)), rng.standard_normal(5)
    for stride, pad in [(1, 0), (1, 1), (2, 1)]:
        ref = nm.ref_conv2d(x, w, b, stride, pad)
        out = F.conv2d(torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(b), stride=stride, padding=pad)
        np.testing.assert_allclose(ref, out.numpy(), rtol=1e-10, atol=1e-10)


def test_fp32_gemm_inside_bound_and_bound_is_not_vacuous():
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal((64, 512)), rng.standard_normal((512, 32))
    a32, b32 = torch.from_numpy(a).float(), torch.from_numpy(b).float()
    aq, bq = a32.double().numpy(), b32.double().numpy()
    bound = nm.gemm_error_bound(aq, bq, nm.UNIT_ROUNDOFF[torch.float32])
    good = nm.check_against_bound("gemm", "cpu", a32 @ b32, aq @ bq, bound, "64x512x32")
    assert good.passed and good.bound_ratio < 1

    # an fp16-quality result must be rejected by the fp32 bound
    bad_out = (a32.half() @ b32.half()).float() if torch.cuda.is_available() else \
        (a32 @ b32).half().float()
    bad = nm.check_against_bound("gemm", "fake", bad_out, aq @ bq, bound, "64x512x32")
    assert not bad.passed


def test_suite_catches_a_buggy_kernel():
    """Inject an off-by-one-tile style bug (one K-slice dropped) and expect failure."""
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal((32, 96)), rng.standard_normal((96, 16))
    a32, b32 = torch.from_numpy(a).float(), torch.from_numpy(b).float()
    buggy = a32[:, :-1] @ b32[:-1, :]
    res = nm.check_against_bound("gemm", "buggy", buggy, a @ b,
                                 nm.gemm_error_bound(a, b, nm.UNIT_ROUNDOFF[torch.float32]), "32x96x16")
    assert not res.passed and res.bound_ratio > 10


def test_ulp_distance():
    x = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    y = np.nextafter(x, np.float32(np.inf))
    assert nm.ulp_distance_f32(x, y).tolist() == [1, 1, 1]


def test_softmax_stable_with_large_logits():
    x = torch.tensor([[1000.0, 1000.0, -1000.0]])
    r = nm.check_elementwise("softmax", "torch", torch.softmax(x, -1), nm.ref_softmax(x.double().numpy()), "1x3")
    assert r.passed


def test_full_suite_passes_on_this_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = suite.run(device)
    assert results and all(r.passed for r in results), [r for r in results if not r.passed]
