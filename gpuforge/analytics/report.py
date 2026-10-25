"""Analytics CLI.

    python -m gpuforge.analytics.report summary
    python -m gpuforge.analytics.report regress --fail-on-regression   # CI perf gate
    python -m gpuforge.analytics.report sql "SELECT * FROM runs"
"""
from __future__ import annotations

import argparse
import sys

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
    args = p.parse_args(argv)

    store = AnalyticsStore(args.db)
    if args.cmd == "summary":
        return cmd_summary(store)
    if args.cmd == "regress":
        return cmd_regress(store, args)
    for row in store.query(args.statement):
        print(dict(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
