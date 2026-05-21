"""Repeatable benchmark harness for Codex Hold'em.

Runs fixed-seed six-max and optional heads-up suites, then reports chip-delta
average, bust rate, positive-table rate, and per-opponent heads-up results.
Use this before enabling any generated model at runtime.
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sandbox.match import run_match  # noqa: E402


DEFAULT_TARGET = "codex_holdem"
DEFAULT_TABLE = {
    "codex_holdem": "bots/codex_holdem",
    "aggressor": "bots/aggressor",
    "mathematician": "bots/mathematician",
    "shark": "bots/shark",
    "template": "bots/template",
    "ref_bot_2": "bots/ref_bot_2",
}
DEFAULT_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
DEFAULT_HEADS_UP_GATE_OPPONENTS = ["mathematician", "ref_bot_2"]


def parse_seeds(raw):
    if not raw:
        return list(DEFAULT_SEEDS)
    seeds = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    return seeds


def parse_names(raw):
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def summarize(deltas):
    if not deltas:
        return {
            "n": 0,
            "avg": 0,
            "median": 0,
            "min": 0,
            "max": 0,
            "stdev": 0,
            "positive_rate": 0,
            "bust_rate": 0,
        }
    return {
        "n": len(deltas),
        "avg": round(sum(deltas) / len(deltas), 2),
        "median": round(statistics.median(deltas), 2),
        "min": min(deltas),
        "max": max(deltas),
        "stdev": round(statistics.pstdev(deltas), 2),
        "positive_rate": round(sum(1 for d in deltas if d > 0) / len(deltas), 3),
        "bust_rate": round(sum(1 for d in deltas if d <= -10000) / len(deltas), 3),
    }


def make_failure(scope, metric, observed, required):
    return {
        "scope": scope,
        "metric": metric,
        "observed": observed,
        "required": required,
    }


def evaluate_gates(report, args):
    thresholds = {
        "six_max_min_avg": args.min_avg,
        "six_max_min_positive_rate": args.min_positive_rate,
        "six_max_max_bust_rate": args.max_bust_rate,
        "heads_up_min_avg": args.min_heads_up_avg,
        "heads_up_opponents": args.heads_up_gate_opponents,
    }
    failures = []

    table = report["table"]["summary"]
    if table["avg"] < args.min_avg:
        failures.append(make_failure(
            "six-max",
            "avg",
            table["avg"],
            ">=" + str(args.min_avg),
        ))
    if table["positive_rate"] < args.min_positive_rate:
        failures.append(make_failure(
            "six-max",
            "positive_rate",
            table["positive_rate"],
            ">=" + str(args.min_positive_rate),
        ))
    if table["bust_rate"] > args.max_bust_rate:
        failures.append(make_failure(
            "six-max",
            "bust_rate",
            table["bust_rate"],
            "<=" + str(args.max_bust_rate),
        ))

    heads_up = report.get("heads_up")
    if heads_up:
        opponents = args.heads_up_gate_opponents
        if args.gate_all_heads_up:
            opponents = sorted(heads_up)
            thresholds["heads_up_opponents"] = opponents

        for opponent in opponents:
            if opponent not in heads_up:
                failures.append(make_failure(
                    "heads-up",
                    opponent,
                    "missing",
                    "opponent present in heads-up report",
                ))
                continue
            avg = heads_up[opponent]["summary"]["avg"]
            if avg < args.min_heads_up_avg:
                failures.append(make_failure(
                    "heads-up:" + opponent,
                    "avg",
                    avg,
                    ">=" + str(args.min_heads_up_avg),
                ))

    return {
        "passed": not failures,
        "thresholds": thresholds,
        "failures": failures,
    }


def run_table_suite(target, hands, seeds):
    rows = []
    for seed in seeds:
        result = seeded_run_match(
            "bench_table_" + str(seed),
            dict(DEFAULT_TABLE),
            hands,
            seed,
        )
        rows.append({
            "seed": seed,
            "delta": result["chip_delta"][target],
            "chip_delta": result["chip_delta"],
            "bot_errors": result["bot_errors"],
        })
    return rows


def run_heads_up_suite(target, hands, seeds):
    opponents = [bid for bid in DEFAULT_TABLE if bid != target]
    summary = {}
    for opponent in opponents:
        rows = []
        for seed in seeds:
            bot_paths = {
                target: DEFAULT_TABLE[target],
                opponent: DEFAULT_TABLE[opponent],
            }
            result = seeded_run_match(
                "bench_hu_" + opponent + "_" + str(seed),
                bot_paths,
                hands,
                seed,
            )
            rows.append({
                "seed": seed,
                "delta": result["chip_delta"][target],
                "chip_delta": result["chip_delta"],
                "bot_errors": result["bot_errors"],
            })
        summary[opponent] = {
            "rows": rows,
            "summary": summarize([r["delta"] for r in rows]),
        }
    return summary


def seeded_run_match(match_id, bot_paths, hands, seed):
    previous = os.environ.get("BOT_RANDOM_SEED")
    os.environ["BOT_RANDOM_SEED"] = str(seed)
    try:
        return run_match(match_id, bot_paths, n_hands=hands, seed=seed)
    finally:
        if previous is None:
            os.environ.pop("BOT_RANDOM_SEED", None)
        else:
            os.environ["BOT_RANDOM_SEED"] = previous


def print_human(report):
    table = report["table"]
    print("Six-max:", table["summary"])
    for row in table["rows"]:
        print("  seed", row["seed"], "delta", row["delta"], row["chip_delta"])

    if report.get("heads_up"):
        print("\nHeads-up:")
        for opponent, data in report["heads_up"].items():
            print(" ", opponent + ":", data["summary"])
            print("   deltas:", [r["delta"] for r in data["rows"]])

    if report.get("gate"):
        gate = report["gate"]
        print("\nGate:", "PASS" if gate["passed"] else "FAIL")
        if gate["failures"]:
            for failure in gate["failures"]:
                print(
                    " ",
                    failure["scope"],
                    failure["metric"],
                    "observed",
                    failure["observed"],
                    "required",
                    failure["required"],
                )


def main():
    parser = argparse.ArgumentParser(description="Benchmark Codex Hold'em")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--hands", type=int, default=400)
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--heads-up", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--min-avg", type=float, default=7000)
    parser.add_argument("--min-positive-rate", type=float, default=0.70)
    parser.add_argument("--max-bust-rate", type=float, default=0.10)
    parser.add_argument("--min-heads-up-avg", type=float, default=3000)
    parser.add_argument(
        "--heads-up-gate-opponents",
        default=",".join(DEFAULT_HEADS_UP_GATE_OPPONENTS),
        help="Comma-separated heads-up opponents checked by --gate.",
    )
    parser.add_argument(
        "--gate-all-heads-up",
        action="store_true",
        help="Check the heads-up average gate against every heads-up opponent.",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    args.heads_up_gate_opponents = parse_names(args.heads_up_gate_opponents)
    table_rows = run_table_suite(args.target, args.hands, seeds)
    report = {
        "target": args.target,
        "hands": args.hands,
        "seeds": seeds,
        "table": {
            "rows": table_rows,
            "summary": summarize([r["delta"] for r in table_rows]),
        },
    }
    if args.heads_up:
        report["heads_up"] = run_heads_up_suite(args.target, args.hands, seeds)

    if args.gate:
        report["gate"] = evaluate_gates(report, args)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)

    if args.gate and not report["gate"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
