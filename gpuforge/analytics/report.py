"""Analytics CLI.

    python -m gpuforge.analytics.report summary
    python -m gpuforge.analytics.report regress --fail-on-regression   # CI perf gate
    python -m gpuforge.analytics.report sql "SELECT * FROM runs"
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from gpuforge.analytics.store import AnalyticsStore


def cmd_summary(store: AnalyticsStore) -> int:
    for r in store.query("SELECT kind, COUNT(*) n, MAX(created_at) last FROM runs GROUP BY kind"):
        print(f"{r['kind']:<12} runs={r['n']}")
    bench = store.latest_run("benchmark")
    if bench:
        print(f"\nLatest benchmark run #{bench}:")
        for r in store.query("SELECT name, params, metric, value FROM benchmarks WHERE run_id=?", (bench,)):
            print(f"  {r['name']:<22} {r['metric']:<12} {r['value']:>12.4f}  {r['params']}")
    train = store.latest_run("train")
    if train:
        loss = store.metric_series(train, "train/loss")
        tput = store.metric_series(train, "train/images_per_sec")
        if loss:
            print(f"\nLatest train run #{train}: loss {loss[0][1]:.4f} -> {loss[-1][1]:.4f} over {loss[-1][0]} steps")
        if tput:
            vals = [v for _, v in tput[1:]] or [tput[0][1]]  # skip warm-up step
            print(f"  mean throughput {sum(vals) / len(vals):.1f} img/s")
    val = store.latest_run("validation")
    if val:
        r = store.query("SELECT SUM(passed) p, COUNT(*) n, MAX(bound_ratio) worst "
                        "FROM validations WHERE run_id=?", (val,))[0]
        print(f"\nLatest validation run #{val}: {r['p']}/{r['n']} passed, worst bound_ratio={r['worst']:.3f}")
    return 0


def _latest_suite_run(store: AnalyticsStore, suite: str) -> sqlite3.Row | None:
    rows = store.query("SELECT id, device, config FROM runs WHERE kind='benchmark' ORDER BY id DESC")
    return next((r for r in rows if json.loads(r["config"]).get("suite") == suite), None)


def results_markdown(store: AnalyticsStore) -> str:
    """Resume/README-ready tables built only from numbers that were actually measured."""
    out = ["# GPUForge measured results", ""]

    kr = _latest_suite_run(store, "kernels")
    if kr:
        out += [f"## GEMM kernels, FP32 (run #{kr['id']}, {kr['device']})", "",
                "| Kernel | Size (M=N=K) | Median ms | TFLOPS | % of vendor BLAS |", "|---|---|---|---|---|"]
        rows = store.query("SELECT name, params, metric, value FROM benchmarks WHERE run_id=? AND metric='tflops'",
                           (kr["id"],))
        by_size: dict[int, dict[str, float]] = {}
        for r in rows:
            by_size.setdefault(json.loads(r["params"])["M"], {})[r["name"]] = r["value"]
        for size in sorted(by_size):
            vals = by_size[size]
            vendor = vals.get("gemm.cublas") or vals.get("gemm.cpu_blas")
            for name, tf in vals.items():
                ms = 2 * size ** 3 / (tf * 1e12) * 1e3
                pct = f"{100 * tf / vendor:.1f}%" if vendor else "-"
                out.append(f"| {name.removeprefix('gemm.')} | {size} | {ms:.3f} | {tf:.2f} | {pct} |")
        out.append("")

    ir = _latest_suite_run(store, "inference")
    if ir:
        out += [f"## Inference (run #{ir['id']}, {ir['device']})", "",
                "| Backend | Batch | p50 latency (ms) | Throughput (img/s) |", "|---|---|---|---|"]
        rows = store.query("SELECT name, params, metric, value FROM benchmarks WHERE run_id=?", (ir["id"],))
        table: dict[tuple, dict[str, float]] = {}
        for r in rows:
            p = json.loads(r["params"])
            table.setdefault((r["name"], p["batch"], p.get("device", "")), {})[r["metric"]] = r["value"]
        for (name, batch, dev), m in sorted(table.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            out.append(f"| {name.removeprefix('infer.')} ({dev}) | {batch} | {m.get('p50_ms', 0):.3f} | "
                       f"{m.get('throughput', 0):,.0f} |")
        out.append("")

    tr = store.latest_run("train")
    if tr:
        cfg = json.loads(store.query("SELECT config FROM runs WHERE id=?", (tr,))[0]["config"])
        loss = store.metric_series(tr, "train/loss")
        tput = [v for _, v in store.metric_series(tr, "train/images_per_sec")[1:]]
        acc = store.metric_series(tr, "val/accuracy")
        amp = cfg.get("amp_resolved", cfg.get("amp_dtype"))
        out += [f"## Training (run #{tr})", "",
                f"- model `{cfg.get('model')}`, AMP `{amp}`, batch {cfg.get('batch_size')} "
                f"x {cfg.get('world_size', 1)} GPU(s), channels_last={cfg.get('channels_last')}"]
        if loss:
            out.append(f"- loss {loss[0][1]:.3f} -> {loss[-1][1]:.3f} over {loss[-1][0]} steps")
        if tput:
            out.append(f"- mean throughput {sum(tput) / len(tput):,.0f} images/s (first logging window excluded)")
        if acc:
            out.append(f"- final validation accuracy {acc[-1][1]:.3f}")
        out.append("")

    vr = store.latest_run("validation")
    if vr:
        out += [f"## Numerical validation (run #{vr})", "",
                "| Implementation | dtype | Checks passed | Worst bound ratio |", "|---|---|---|---|"]
        for r in store.query("""SELECT impl, dtype, SUM(passed) p, COUNT(*) n, MAX(bound_ratio) w
                                FROM validations WHERE run_id=? GROUP BY impl, dtype ORDER BY impl, dtype""", (vr,)):
            out.append(f"| {r['impl']} | {r['dtype']} | {r['p']}/{r['n']} | {r['w']:.3f} |")
        out += ["", "_Bound ratio = largest error / theoretical error bound; anything <= 1 passes._", ""]

    if len(out) == 2:
        out.append("No results recorded yet - run `make validate train bench` first.")
    return "\n".join(out)


def cmd_regress(store: AnalyticsStore, args) -> int:
    run_id = args.run or store.latest_run("benchmark")
    if run_id is None:
        print("no benchmark runs recorded")
        return 0
    regs = store.detect_regressions(run_id, args.window, args.threshold)
    if not regs:
        print(f"run #{run_id}: no baseline history yet")
        return 0
    bad = [r for r in regs if r.regressed]
    for r in regs:
        tag = "REGRESSION" if r.regressed else "ok"
        print(f"[{tag:>10}] {r.name:<22} {r.metric:<12} {r.value:.4f} vs baseline {r.baseline:.4f} "
              f"({r.change_pct:+.1f}%, z={r.robust_z:+.1f}) {r.params}")
    print(f"\n{len(bad)} regression(s) in run #{run_id}")
    return 1 if (bad and args.fail_on_regression) else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="reports/analytics.db")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("summary")
    rg = sub.add_parser("regress")
    rg.add_argument("--run", type=int)
    rg.add_argument("--window", type=int, default=5)
    rg.add_argument("--threshold", type=float, default=5.0, help="percent")
    rg.add_argument("--fail-on-regression", action="store_true")
    sq = sub.add_parser("sql")
    sq.add_argument("statement")
    rs = sub.add_parser("results", help="markdown tables of the latest measured results")
    rs.add_argument("--out", default="reports/RESULTS.md")
    args = p.parse_args(argv)

    store = AnalyticsStore(args.db)
    if args.cmd == "summary":
        return cmd_summary(store)
    if args.cmd == "results":
        md = results_markdown(store)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)
        print(md)
        return 0
    if args.cmd == "regress":
        return cmd_regress(store, args)
    for row in store.query(args.statement):
        print(dict(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
