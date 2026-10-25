from gpuforge.analytics.store import AnalyticsStore


def _bench_run(store, value, name="gemm.regtile", hib=True):
    rid = store.start_run("benchmark", device="A100")
    store.log_benchmark(rid, name, {"M": 1024}, "tflops", value, higher_is_better=hib)
    return rid


def test_no_history_means_no_regression():
    s = AnalyticsStore(":memory:")
    assert s.detect_regressions(_bench_run(s, 10.0)) == []


def test_noise_is_not_flagged_but_real_drop_is():
    s = AnalyticsStore(":memory:")
    for v in [10.0, 10.2, 9.9, 10.1, 10.0]:
        _bench_run(s, v)
    ok = s.detect_regressions(_bench_run(s, 9.8))[0]
    assert not ok.regressed
    bad = s.detect_regressions(_bench_run(s, 8.0))[0]
    assert bad.regressed and bad.change_pct < -15


def test_lower_is_better_metrics():
    s = AnalyticsStore(":memory:")
    for v in [2.0, 2.05, 1.95, 2.0]:
        _bench_run(s, v, "infer.trt", hib=False)
    assert s.detect_regressions(_bench_run(s, 3.0, "infer.trt", hib=False))[0].regressed
    assert not s.detect_regressions(_bench_run(s, 1.5, "infer.trt", hib=False))[0].regressed  # faster is fine


def test_other_devices_are_not_baselines():
    s = AnalyticsStore(":memory:")
    rid = s.start_run("benchmark", device="H100")
    s.log_benchmark(rid, "gemm.regtile", {"M": 1024}, "tflops", 40.0, True)
    assert s.detect_regressions(_bench_run(s, 10.0)) == []


def test_metrics_roundtrip():
    s = AnalyticsStore(":memory:")
    rid = s.start_run("train", {"lr": 0.1})
    s.log_metrics(rid, 1, {"train/loss": 2.3})
    s.log_metrics(rid, 2, {"train/loss": 2.1})
    assert s.metric_series(rid, "train/loss") == [(1, 2.3), (2, 2.1)]


def test_results_markdown_only_reports_measured_numbers():
    from gpuforge.analytics.report import results_markdown

    s = AnalyticsStore(":memory:")
    assert "No results recorded yet" in results_markdown(s)
    rid = s.start_run("benchmark", {"suite": "kernels"}, device="Tesla T4")
    s.log_benchmark(rid, "gemm.cublas", {"M": 1024, "N": 1024, "K": 1024, "dtype": "fp32"}, "tflops", 4.0, True)
    s.log_benchmark(rid, "gemm.regtile", {"M": 1024, "N": 1024, "K": 1024, "dtype": "fp32"}, "tflops", 2.0, True)
    md = results_markdown(s)
    assert "Tesla T4" in md and "| regtile | 1024 |" in md and "50.0%" in md


def test_amp_auto_resolution():
    import pytest
    import torch

    from gpuforge.utils import resolve_amp_dtype

    assert resolve_amp_dtype("none", "cuda") is None
    assert resolve_amp_dtype("auto", "cpu") == torch.bfloat16
    assert resolve_amp_dtype("auto", "cuda") in (torch.bfloat16, torch.float16)
    with pytest.raises(ValueError):
        resolve_amp_dtype("fp8", "cuda")
