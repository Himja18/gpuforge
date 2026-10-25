"""Back-end analytics: one SQLite store for every training run, benchmark and validation.

Everything the platform produces lands in the same schema so you can answer
questions like "did throughput regress after this commit?" with SQL, and so the
CI pipeline can gate merges on performance, not just on tests passing.
"""
from __future__ import annotations

import json
import os
import socket
import sqlite3
import statistics
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,             -- train | benchmark | validation | serve
    git_sha     TEXT,
    host        TEXT,
    device      TEXT,
    config      TEXT,                      -- JSON
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id  INTEGER NOT NULL REFERENCES runs(id),
    step    INTEGER NOT NULL,
    name    TEXT NOT NULL,
    value   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS benchmarks (
    run_id  INTEGER NOT NULL REFERENCES runs(id),
    name    TEXT NOT NULL,                 -- e.g. gemm.regtile, infer.onnx
    params  TEXT NOT NULL,                 -- canonical JSON, used as the comparison key
    metric  TEXT NOT NULL,                 -- tflops | p50_ms | throughput ...
    value   REAL NOT NULL,
    higher_is_better INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS validations (
    run_id  INTEGER NOT NULL REFERENCES runs(id),
    op TEXT, impl TEXT, dtype TEXT, shape TEXT,
    passed INTEGER, max_abs_err REAL, max_rel_err REAL, bound_ratio REAL
);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id, name);
CREATE INDEX IF NOT EXISTS idx_bench_key ON benchmarks(name, params, metric);
"""


def _git_sha() -> str | None:
    sha = os.environ.get("GIT_COMMIT") or os.environ.get("GITHUB_SHA")
    if sha:
        return sha[:12]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


@dataclass
class Regression:
    name: str
    params: str
    metric: str
    value: float
    baseline: float
    change_pct: float
    robust_z: float
    regressed: bool


class AnalyticsStore:
    def __init__(self, path: str | os.PathLike = "reports/analytics.db"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(SCHEMA)

    @contextmanager
    def _tx(self):
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ---------------------------------------------------------------- write --
    def start_run(self, kind: str, config: dict | None = None, device: str | None = None) -> int:
        with self._tx() as c:
            cur = c.execute(
                "INSERT INTO runs(kind, git_sha, host, device, config, created_at) VALUES (?,?,?,?,?,?)",
                (kind, _git_sha(), socket.gethostname(), device or _device(),
                 json.dumps(config or {}, sort_keys=True, default=str), time.time()))
            return int(cur.lastrowid)

    def log_metrics(self, run_id: int, step: int, values: dict[str, float]) -> None:
        with self._tx() as c:
            c.executemany("INSERT INTO metrics VALUES (?,?,?,?)",
                          [(run_id, step, k, float(v)) for k, v in values.items()])

    def log_benchmark(self, run_id: int, name: str, params: dict, metric: str, value: float,
                      higher_is_better: bool) -> None:
        with self._tx() as c:
            c.execute("INSERT INTO benchmarks VALUES (?,?,?,?,?,?)",
                      (run_id, name, json.dumps(params, sort_keys=True), metric, float(value),
                       int(higher_is_better)))

    def log_validations(self, run_id: int, results) -> None:
        with self._tx() as c:
            c.executemany("INSERT INTO validations VALUES (?,?,?,?,?,?,?,?,?)",
                          [(run_id, r.op, r.impl, r.dtype, r.shape, int(r.passed), r.max_abs_err,
                            r.max_rel_err, r.bound_ratio) for r in results])

    # ----------------------------------------------------------------- read --
    def query(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        self._conn.row_factory = sqlite3.Row
        try:
            return list(self._conn.execute(sql, args))
        finally:
            self._conn.row_factory = None

    def latest_run(self, kind: str) -> int | None:
        row = self._conn.execute("SELECT MAX(id) FROM runs WHERE kind=?", (kind,)).fetchone()
        return row[0]

    def metric_series(self, run_id: int, name: str) -> list[tuple[int, float]]:
        return list(self._conn.execute(
            "SELECT step, value FROM metrics WHERE run_id=? AND name=? ORDER BY step", (run_id, name)))

    def detect_regressions(self, run_id: int, window: int = 5, threshold_pct: float = 5.0,
                           z_threshold: float = 3.0, noise_floor_pct: float = 3.0) -> list[Regression]:
        """Compare each benchmark in ``run_id`` with the same (name, params, metric)
        in the previous ``window`` runs on the same device.

        A result regresses only if it is *both* worse than the baseline median by
        more than ``threshold_pct`` *and* outside the run-to-run noise (robust
        z-score using the median absolute deviation). The two conditions together
        keep noisy GPU timings from producing false alarms. With only a few
        history points the MAD underestimates the real noise, so the noise scale
        never drops below ``noise_floor_pct`` of the baseline (typical run-to-run
        variation from GPU clock boost and thermal state).
        """
        device = self._conn.execute("SELECT device FROM runs WHERE id=?", (run_id,)).fetchone()[0]
        current = self._conn.execute(
            "SELECT name, params, metric, value, higher_is_better FROM benchmarks WHERE run_id=?",
            (run_id,)).fetchall()
        out: list[Regression] = []
        for name, params, metric, value, hib in current:
            hist = [r[0] for r in self._conn.execute(
                """SELECT b.value FROM benchmarks b JOIN runs r ON r.id=b.run_id
                   WHERE b.name=? AND b.params=? AND b.metric=? AND r.device=? AND b.run_id<?
                   ORDER BY b.run_id DESC LIMIT ?""", (name, params, metric, device, run_id, window))]
            if not hist:
                continue
            base = statistics.median(hist)
            mad = statistics.median([abs(h - base) for h in hist]) if len(hist) > 1 else 0.0
            # MAD -> sigma, floored at the expected run-to-run noise
            scale = max(1.4826 * mad, abs(base) * noise_floor_pct / 100.0, 1e-12)
            signed = (value - base) if hib else (base - value)  # >0 means better
            change_pct = 100.0 * signed / abs(base) if base else 0.0
            z = signed / scale
            out.append(Regression(name, params, metric, value, base, change_pct, z,
                                  regressed=(change_pct < -threshold_pct and z < -z_threshold)))
        return out
