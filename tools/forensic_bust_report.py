"""Classify target bust hands for Codex Hold'em benchmark forensics."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sandbox.match import run_match  # noqa: E402
from tools.benchmark_codex_holdem import DEFAULT_TABLE, DEFAULT_TARGET, parse_seeds  # noqa: E402


def last_action_for(action_log, seat):
    for action in reversed(action_log):
        if action.get("seat") == seat:
            return action
    return None


def last_aggression(action_log):
    for action in reversed(action_log):
        if action.get("action") in ("raise", "all_in"):
            return action
    return None


def target_commit_events(events, target):
    return [
        event for event in events
        if event.get("type") == "action"
        and event.get("bot_id") == target
        and event.get("action") in ("call", "raise", "all_in")
    ]


def target_seat(bot_ids, target):
    try:
        return bot_ids.index(target)
    except ValueError:
        return None


def classify_bust(seed, match, target):
    seat = target_seat(match.get("bot_ids", []), target)
    if seat is None:
        return None

    previous_stack = 10000
    for hand in match.get("hands", []):
        stacks = hand.get("final_stacks", {})
        current_stack = stacks.get(target, previous_stack)
        if previous_stack > 0 and current_stack <= 0:
            action_log = hand.get("action_log", [])
            target_actions = [a for a in action_log if a.get("seat") == seat]
            committed = [
                a for a in target_actions
                if a.get("action") in ("call", "raise", "all_in")
            ]
            commit_events = target_commit_events(hand.get("events", []), target)
            final_commit_event = commit_events[-1] if commit_events else None
            return {
                "seed": seed,
                "hand_num": hand.get("hand_num"),
                "hand_id": hand.get("hand_id"),
                "showdown_street": hand.get("street"),
                "commit_street": (final_commit_event or {}).get("street", hand.get("street")),
                "pot": hand.get("pot"),
                "community_cards": hand.get("community_cards", []),
                "showdown": hand.get("showdown"),
                "target_last_action": last_action_for(action_log, seat),
                "target_final_commit_action": committed[-1] if committed else None,
                "target_final_commit_event": final_commit_event,
                "last_aggression": last_aggression(action_log),
                "target_action_count": len(target_actions),
                "target_raise_count": sum(1 for a in target_actions if a.get("action") in ("raise", "all_in")),
                "target_call_count": sum(1 for a in target_actions if a.get("action") == "call"),
                "revealed_cards": hand.get("revealed_cards", {}).get(target, []),
                "hand_strength": hand.get("hand_strengths", {}).get(target),
                "final_stacks": stacks,
            }
        previous_stack = current_stack
    return None


def summarize(rows):
    by_street = {}
    by_commit = {}
    for row in rows:
        street = row.get("commit_street") or "unknown"
        by_street[street] = by_street.get(street, 0) + 1
        action = (row.get("target_final_commit_action") or {}).get("action", "none")
        by_commit[action] = by_commit.get(action, 0) + 1
    return {
        "busts": len(rows),
        "by_street": by_street,
        "by_final_commit_action": by_commit,
    }


def main():
    parser = argparse.ArgumentParser(description="Classify target bust hands across fixed seeds.")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--hands", type=int, default=400)
    parser.add_argument("--seeds", default="103,104,105,110,121,129,137,139,146,150")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    rows = []
    for seed in seeds:
        match = run_match(
            "forensic_bust_" + str(seed),
            dict(DEFAULT_TABLE),
            n_hands=args.hands,
            seed=seed,
        )
        row = classify_bust(seed, match, args.target)
        if row:
            rows.append(row)

    report = {
        "target": args.target,
        "hands": args.hands,
        "seeds": seeds,
        "summary": summarize(rows),
        "busts": rows,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("Busts:", report["summary"]["busts"])
        print("By street:", report["summary"]["by_street"])
        print("By final commit action:", report["summary"]["by_final_commit_action"])
        for row in rows:
            commit = row.get("target_final_commit_action") or {}
            print(
                "seed",
                row["seed"],
                "hand",
                row["hand_num"],
                "street",
                row["commit_street"],
                "commit",
                commit.get("action"),
                commit.get("amount"),
                "pot",
                row["pot"],
                "hand",
                row.get("revealed_cards"),
                row.get("hand_strength"),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
