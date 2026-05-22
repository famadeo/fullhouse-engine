"""Train the optional multi-head model used by bots/codex_holdem.

This is intentionally lightweight. It generates decision samples by running
local in-process matches, then fits small linear heads:

  chip_ev          tanh     best counterfactual action EV, scaled
  risk_adjusted_chip_ev
                   tanh     chip EV penalized by bust / downside risk
  showdown_equity sigmoid  Monte Carlo / heuristic showdown equity
  fold_pressure   sigmoid  estimated fold equity from betting
  danger          sigmoid  downside risk from continuing
  survival        sigmoid  reward for keeping the target bot alive
  stack_preservation
                   sigmoid  reward for preserving a playable stack
  ev_*            tanh     action-value heads for candidate actions

The exported model is plain JSON, so the runtime bot does not need sklearn.
"""

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.game import PokerEngine, STARTING_STACK  # noqa: E402


DEFAULT_BOTS = {
    "codex_holdem": "bots/codex_holdem",
    "aggressor": "bots/aggressor",
    "mathematician": "bots/mathematician",
    "shark": "bots/shark",
    "template": "bots/template",
    "ref_bot_2": "bots/ref_bot_2",
}
REPLAY_SCHEMA_VERSION = 1


def load_bot(bot_id, bot_path):
    path = ROOT / bot_path
    if path.is_dir():
        path = path / "bot.py"
    spec = importlib.util.spec_from_file_location("train_" + bot_id, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_decide(module, state):
    try:
        action = module.decide(state)
        if isinstance(action, dict) and "action" in action:
            return action
    except Exception:
        pass
    return {"action": "check"} if state.get("can_check") else {"action": "fold"}


def target_features(module, state):
    if state.get("street") == "preflop":
        equity = 0.5
    else:
        equity = module.estimate_equity(state)
    return module.extract_features(state, equity)


def target_public_belief_state(module, state):
    extractor = getattr(module, "extract_public_belief_state", None)
    if extractor is None:
        return {}
    return extractor(state)


def legal_context(state):
    stack = int(state.get("your_stack", 0) or 0)
    invested = int(state.get("your_bet_this_street", 0) or 0)
    owed = int(state.get("amount_owed", 0) or 0)
    min_raise_to = int(state.get("min_raise_to", 0) or 0)
    legal_actions = []
    if state.get("can_check"):
        legal_actions.append("check")
    else:
        legal_actions.extend(["fold", "call"])
    if stack > 0:
        legal_actions.append("all_in")
    if stack + invested > min_raise_to > 0:
        legal_actions.append("raise")

    return {
        "legal_actions": sorted(set(legal_actions)),
        "can_check": bool(state.get("can_check")),
        "amount_owed": owed,
        "current_bet": int(state.get("current_bet", 0) or 0),
        "min_raise_to": min_raise_to,
        "pot": int(state.get("pot", 0) or 0),
        "your_stack": stack,
        "your_bet_this_street": invested,
    }


def public_context(state):
    players = []
    for player in state.get("players", []):
        players.append({
            "seat": player.get("seat"),
            "bot_id": player.get("bot_id"),
            "stack": int(player.get("stack", 0) or 0),
            "bet": int(player.get("bet", 0) or 0),
            "state": player.get("state"),
            "is_folded": bool(player.get("is_folded")),
        })
    return {
        "street": state.get("street"),
        "seat_to_act": state.get("seat_to_act"),
        "button": state.get("button"),
        "community_cards": list(state.get("community_cards", [])),
        "pot": int(state.get("pot", 0) or 0),
        "current_bet": int(state.get("current_bet", 0) or 0),
        "players": players,
    }


def feature_equity(feats):
    return clamp(float(feats.get("equity", 0.5)), 0.0, 1.0)


def estimate_fold_probability(state, contribution, feats, jam=False):
    pot = max(int(state.get("pot", 0) or 0), 1)
    opponents = max(1, len([p for p in state.get("players", []) if not p.get("is_folded")]) - 1)
    size_ratio = contribution / max(pot + contribution, 1)
    pressure = float(feats.get("pressure", 0.0))
    pos = float(feats.get("position", 0.0))
    wet = float(feats.get("wet_board", 0.0))
    opp_call = float(feats.get("opp_call_rate", 0.33))
    opp_fold = float(feats.get("opp_fold_rate", 0.20))
    opp_raise = float(feats.get("opp_raise_rate", 0.10))

    fold_prob = (
        0.10
        + 0.38 * size_ratio
        + 0.16 * pos
        + 0.18 * opp_fold
        - 0.20 * opp_call
        - 0.08 * opp_raise
        - 0.08 * wet
        - 0.08 * max(0, opponents - 1)
        - 0.05 * pressure
    )
    if jam:
        fold_prob += 0.10
    if state.get("street") == "river":
        fold_prob += 0.04
    if contribution <= pot / 3:
        fold_prob -= 0.08
    return clamp(fold_prob, 0.03, 0.82 if opponents <= 2 else 0.55)


def counterfactual_values(state, feats):
    equity = feature_equity(feats)
    pot = max(int(state.get("pot", 0) or 0), 1)
    owed = int(state.get("amount_owed", 0) or 0)
    stack = int(state.get("your_stack", 0) or 0)
    invested = int(state.get("your_bet_this_street", 0) or 0)
    min_raise_to = int(state.get("min_raise_to", 0) or 0)
    current = int(state.get("current_bet", 0) or 0)

    fold_ev = 0.0 if owed > 0 else equity * pot
    check_call_ev = equity * (pot + owed) - owed

    def bet_ev(fraction):
        if stack <= 0:
            return -1.0, 0.0
        if owed > 0:
            target = max(min_raise_to, invested + owed + int((pot + owed) * fraction))
        else:
            target = max(min_raise_to, invested + int(pot * fraction))
        contribution = clamp(target - invested, 0, stack)
        if contribution <= owed and owed > 0:
            return check_call_ev, 0.0
        fold_prob = estimate_fold_probability(state, contribution, feats)
        called_ev = equity * (pot + contribution) - contribution
        raise_risk = float(feats.get("opp_raise_rate", 0.10)) * contribution * 0.25
        return fold_prob * pot + (1.0 - fold_prob) * called_ev - raise_risk, fold_prob

    def jam_ev():
        if stack <= 0:
            return -1.0, 0.0
        contribution = stack
        fold_prob = estimate_fold_probability(state, contribution, feats, jam=True)
        called_ev = equity * (pot + contribution) - contribution
        return fold_prob * pot + (1.0 - fold_prob) * called_ev, fold_prob

    ev33, fp33 = bet_ev(0.33)
    ev66, fp66 = bet_ev(0.66)
    evjam, fpjam = jam_ev()
    raw_values = {
        "ev_fold": fold_ev,
        "ev_check_call": check_call_ev,
        "ev_bet_33": ev33,
        "ev_bet_66": ev66,
        "ev_jam": evjam,
    }
    fold_pressure = max(fp33, fp66, fpjam)
    downside = max(0.0, -min(check_call_ev, ev33, ev66, evjam))
    return {
        name: clamp(value / 5000.0, -1.0, 1.0)
        for name, value in raw_values.items()
    }, fold_pressure, clamp(downside / max(stack, 1), 0.0, 1.0)


def opponent_responses_after(hand_action_log, decision_index, target_id):
    responses = []
    for row in hand_action_log[decision_index + 1:]:
        if row.get("bot_id") == target_id:
            break
        responses.append(row)
    return responses


def run_training_match(match_id, modules, bot_paths, target_id, hands, seed):
    random.seed(seed)
    bot_ids = list(bot_paths.keys())
    stacks = {bid: STARTING_STACK for bid in bot_ids}
    dealer = 0
    samples = []
    match_action_log = []

    for hand_num in range(hands):
        alive = [bid for bid in bot_ids if stacks[bid] > 0]
        if len(alive) < 2 or target_id not in alive:
            break

        hand_id = match_id + "_h" + str(hand_num).zfill(4)
        hand_seed = seed * 1000003 + hand_num
        starting = {bid: stacks[bid] for bid in alive}
        engine = PokerEngine(
            hand_id=hand_id,
            bot_ids=alive,
            dealer_seat=dealer % len(alive),
            starting_stacks=starting,
            seed=hand_seed,
        )

        pending = []
        hand_action_log = []
        state = engine.start_hand()
        steps = 0

        while state.get("type") == "action_request":
            seat = state["seat_to_act"]
            bot_id = alive[seat]
            module = modules[bot_id]
            state_for_bot = dict(state)
            state_for_bot["match_action_log"] = match_action_log[-200:]

            if bot_id == target_id:
                feats = target_features(module, state_for_bot)
                values, fold_pressure, danger = counterfactual_values(state_for_bot, feats)
                action = safe_decide(module, state_for_bot)
                pending.append({
                    "schema_version": REPLAY_SCHEMA_VERSION,
                    "match_id": match_id,
                    "hand_id": hand_id,
                    "hand_num": hand_num,
                    "hand_seed": hand_seed,
                    "seat": seat,
                    "decision_index": len(hand_action_log),
                    "features": feats,
                    "public_belief_state": target_public_belief_state(module, state_for_bot),
                    "public_context": public_context(state_for_bot),
                    "legal_context": legal_context(state_for_bot),
                    "action": action.get("action", "fold"),
                    "street": state_for_bot.get("street"),
                    "cf_values": values,
                    "cf_fold_pressure": fold_pressure,
                    "cf_danger": danger,
                })
            else:
                action = safe_decide(module, state_for_bot)

            action_row = {
                "hand_num": hand_num,
                "hand_id": hand_id,
                "street": state.get("street"),
                "seat": seat,
                "bot_id": bot_id,
                "action": action.get("action"),
                "amount": action.get("amount"),
            }
            match_action_log.append(action_row)
            hand_action_log.append(action_row)
            state = engine.apply_action(seat, action)
            steps += 1
            if steps > 1000:
                break

        for bid, value in state.get("final_stacks", {}).items():
            stacks[bid] = value

        target_delta = stacks.get(target_id, 0) - starting.get(target_id, 0)
        target_won = target_delta > 0
        target_stack = stacks.get(target_id, 0)
        target_survived = target_stack > 0
        stack_preservation = clamp(target_stack / STARTING_STACK, 0.0, 1.0)
        survival_reward = 1.0 if target_survived else 0.0
        showdown = bool(state.get("showdown"))
        for sample in pending:
            aggressive = sample["action"] in ("raise", "all_in")
            cf_values = dict(sample["cf_values"])
            best_cf = max(cf_values.values()) if cf_values else 0.0
            risk_adjusted = best_cf - 0.35 * sample["cf_danger"] - 0.80 * (1.0 - survival_reward)
            sample["opponent_responses"] = opponent_responses_after(
                hand_action_log,
                sample["decision_index"],
                target_id,
            )
            sample["terminal"] = {
                "target_delta": target_delta,
                "target_stack": target_stack,
                "target_survived": target_survived,
                "target_won": target_won,
                "showdown": showdown,
                "final_stacks": state.get("final_stacks", {}),
            }
            sample["labels"] = {
                **cf_values,
                "chip_ev": best_cf,
                "risk_adjusted_chip_ev": clamp(risk_adjusted, -1.0, 1.0),
                "showdown_equity": feature_equity(sample["features"]),
                "fold_pressure": sample["cf_fold_pressure"],
                "danger": sample["cf_danger"],
                "survival": survival_reward,
                "stack_preservation": stack_preservation,
                "realized_chip_ev": clamp(target_delta / 5000.0, -1.0, 1.0),
                "realized_win": 1.0 if target_won else 0.0,
                "realized_fold_pressure": 1.0 if aggressive and target_won and not showdown else 0.0,
            }
            samples.append(sample)

        dealer += 1

    return samples


def write_replay(path, samples, meta):
    output = ROOT / path
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write(json.dumps({
            "type": "metadata",
            "schema_version": REPLAY_SCHEMA_VERSION,
            "meta": meta,
        }, sort_keys=True) + "\n")
        for idx, sample in enumerate(samples):
            f.write(json.dumps({
                "type": "sample",
                "schema_version": REPLAY_SCHEMA_VERSION,
                "sample_id": idx,
                "sample": sample,
            }, sort_keys=True) + "\n")
    return output


def load_replay(paths):
    samples = []
    metadata = []
    for raw_path in paths:
        path = ROOT / raw_path
        with open(path, "r") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record_type = record.get("type")
                if record_type == "metadata":
                    metadata.append(record.get("meta", {}))
                elif record_type == "sample":
                    samples.append(record.get("sample", {}))
                else:
                    raise ValueError(str(path) + ":" + str(line_num) + " has unknown replay record type")
    return samples, metadata


def display_path(path):
    if path is None:
        return None
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def sigmoid(value):
    value = clamp(value, -30.0, 30.0)
    return 1.0 / (1.0 + pow(2.718281828459045, -value))


def tanh(value):
    ex = pow(2.718281828459045, clamp(2.0 * value, -30.0, 30.0))
    return (ex - 1.0) / (ex + 1.0)


def score(weights, feats, feature_names):
    return sum(weights.get(name, 0.0) * feats.get(name, 0.0) for name in feature_names)


def fit_head(samples, feature_names, label_name, activation, rng, epochs=18, lr=0.035, l2=0.0005):
    weights = {name: 0.0 for name in feature_names}
    rows = list(samples)

    for _ in range(epochs):
        rng.shuffle(rows)
        for sample in rows:
            feats = sample["features"]
            y = sample["labels"][label_name]
            raw = score(weights, feats, feature_names)
            if activation == "sigmoid":
                pred = sigmoid(raw)
                grad = pred - y
            elif activation == "tanh":
                pred = tanh(raw)
                grad = (pred - y) * (1.0 - pred * pred)
            else:
                pred = raw
                grad = pred - y

            for name in feature_names:
                x = feats.get(name, 0.0)
                weights[name] -= lr * (grad * x + l2 * weights[name])

    return weights


def summarize(samples):
    if not samples:
        return {}
    labels = {}
    for name in samples[0]["labels"]:
        values = [s["labels"][name] for s in samples]
        labels[name] = {
            "mean": round(sum(values) / len(values), 5),
            "min": round(min(values), 5),
            "max": round(max(values), 5),
        }
    by_street = {}
    for s in samples:
        by_street[s["street"]] = by_street.get(s["street"], 0) + 1
    return {"n_samples": len(samples), "labels": labels, "by_street": by_street}


def summarize_public_belief(samples, feature_names):
    if not samples or not feature_names:
        return {}
    summary = {}
    for name in feature_names:
        values = [
            float(s.get("public_belief_state", {}).get(name, 0.0))
            for s in samples
        ]
        summary[name] = {
            "mean": round(sum(values) / len(values), 5),
            "min": round(min(values), 5),
            "max": round(max(values), 5),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train Codex Hold'em JSON model")
    parser.add_argument("--hands", type=int, default=1200)
    parser.add_argument("--matches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--target", default="codex_holdem")
    parser.add_argument("--output", default="bots/codex_holdem/data/model.json")
    parser.add_argument(
        "--replay-input",
        action="append",
        default=[],
        help="Read training samples from a JSONL replay buffer. May be repeated.",
    )
    parser.add_argument(
        "--replay-output",
        help="Write generated or loaded training samples to a JSONL replay buffer.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate and optionally write replay samples without fitting a model.",
    )
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    bot_paths = dict(DEFAULT_BOTS)
    modules = {bid: load_bot(bid, path) for bid, path in bot_paths.items()}
    rng = random.Random(args.seed)
    replay_metadata = []
    if args.replay_input:
        samples, replay_metadata = load_replay(args.replay_input)
    else:
        hands_per_match = max(1, args.hands // max(1, args.matches))
        samples = []
        for match_idx in range(max(1, args.matches)):
            match_seed = args.seed + match_idx * 7919
            samples.extend(run_training_match(
                "train_" + str(args.seed) + "_m" + str(match_idx),
                modules,
                bot_paths,
                args.target,
                hands_per_match,
                match_seed,
            ))
    if not samples:
        raise RuntimeError("No samples generated")

    feature_names = list(modules[args.target].FEATURE_NAMES)
    public_belief_feature_names = list(getattr(modules[args.target], "PUBLIC_BELIEF_FEATURE_NAMES", []))
    train_samples = [s for s in samples if s["street"] != "preflop"] or samples
    replay_meta = {
        "target": args.target,
        "hands": args.hands,
        "matches": args.matches,
        "seed": args.seed,
        "n_samples": len(samples),
        "summary": summarize(samples),
        "public_belief_summary": summarize_public_belief(samples, public_belief_feature_names),
        "source_replay_metadata": replay_metadata,
    }
    replay_output = None
    if args.replay_output:
        replay_output = write_replay(args.replay_output, samples, replay_meta)

    if args.generate_only:
        print(json.dumps({
            "replay_output": display_path(replay_output),
            "summary": replay_meta["summary"],
        }, indent=2, sort_keys=True))
        return

    heads = {
        "chip_ev": {
            "activation": "tanh",
            "weights": fit_head(train_samples, feature_names, "chip_ev", "tanh", rng),
        },
        "risk_adjusted_chip_ev": {
            "activation": "tanh",
            "weights": fit_head(train_samples, feature_names, "risk_adjusted_chip_ev", "tanh", rng),
        },
        "showdown_equity": {
            "activation": "sigmoid",
            "weights": fit_head(train_samples, feature_names, "showdown_equity", "sigmoid", rng),
        },
        "fold_pressure": {
            "activation": "sigmoid",
            "weights": fit_head(train_samples, feature_names, "fold_pressure", "sigmoid", rng),
        },
        "danger": {
            "activation": "sigmoid",
            "weights": fit_head(train_samples, feature_names, "danger", "sigmoid", rng),
        },
        "survival": {
            "activation": "sigmoid",
            "weights": fit_head(train_samples, feature_names, "survival", "sigmoid", rng),
        },
        "stack_preservation": {
            "activation": "sigmoid",
            "weights": fit_head(train_samples, feature_names, "stack_preservation", "sigmoid", rng),
        },
    }
    for label_name in ("ev_fold", "ev_check_call", "ev_bet_33", "ev_bet_66", "ev_jam"):
        heads[label_name] = {
            "activation": "tanh",
            "weights": fit_head(train_samples, feature_names, label_name, "tanh", rng),
        }

    model = {
        "version": modules[args.target].MODEL_VERSION,
        "target": args.target,
        "runtime_enabled": False,
        "feature_names": feature_names,
        "public_belief_feature_names": public_belief_feature_names,
        "heads": heads,
        "meta": {
            "hands": args.hands,
            "matches": args.matches,
            "seed": args.seed,
            "summary": summarize(samples),
            "training_summary": summarize(train_samples),
            "public_belief_summary": summarize_public_belief(samples, public_belief_feature_names),
            "public_belief_training_summary": summarize_public_belief(train_samples, public_belief_feature_names),
            "replay_input": args.replay_input,
            "replay_output": display_path(replay_output),
        },
    }

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(model, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps({
        "output": display_path(output),
        "replay_output": display_path(replay_output),
        "summary": model["meta"]["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
