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
