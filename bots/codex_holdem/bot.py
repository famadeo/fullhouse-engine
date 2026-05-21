"""Codex Hold'em: a sturdy chip-EV NLHE bot for the Fullhouse engine."""

import json
import math
import os
import random

try:
    import eval7
except Exception:
    eval7 = None


BOT_NAME = "Codex Holdem"
MODEL_VERSION = 1

RANKS = "23456789TJQKA"
SUITS = "shdc"
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}
CHEN_POINTS = {
    "A": 10.0, "K": 8.0, "Q": 7.0, "J": 6.0, "T": 5.0,
    "9": 4.5, "8": 4.0, "7": 3.5, "6": 3.0, "5": 2.5,
    "4": 2.0, "3": 1.5, "2": 1.0,
}

FEATURE_NAMES = [
    "bias",
    "street_preflop", "street_flop", "street_turn", "street_river",
    "pot_s", "owed_s", "stack_s", "spr_s", "pressure",
    "position", "opponents_s", "can_check", "current_bet_s",
    "chen_s", "equity", "required", "wet_board",
    "cat_premium", "cat_strong", "cat_medium", "cat_speculative", "cat_trash",
    "actions_s", "raises_s",
    "opp_call_rate", "opp_raise_rate", "opp_fold_rate", "history_s",
]


def load_model():
    data_dir = os.environ.get("BOT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
    path = os.path.join(data_dir, "model.json")
    try:
        with open(path, "r") as f:
            model = json.load(f)
        if model.get("version") != MODEL_VERSION:
            return None
        if not model.get("runtime_enabled", False):
            return None
        return model
    except Exception:
        return None


MODEL = load_model()


def decide(state):
    try:
        if state.get("type") == "warmup":
            return {"action": "check"}
        if state.get("street") == "preflop":
            return preflop_decision(state)
        return postflop_decision(state)
    except Exception:
        return safe_check_or_fold(state)


def safe_check_or_fold(state):
    return {"action": "check"} if state.get("can_check") else {"action": "fold"}


def safe_check_or_call(state):
    return {"action": "check"} if state.get("can_check") else {"action": "call"}


def safe_fold(state):
    return {"action": "check"} if state.get("can_check") else {"action": "fold"}


def raise_to(state, total):
    stack = int(state.get("your_stack", 0) or 0)
    already = int(state.get("your_bet_this_street", 0) or 0)
    max_total = stack + already
    if stack <= 0:
        return safe_check_or_fold(state)

    min_raise_to = int(state.get("min_raise_to", 0) or 0)
    total = int(max(total, min_raise_to))

    if max_total <= int(state.get("current_bet", 0) or 0):
        return {"action": "all_in"}
    if total >= max_total:
        return {"action": "all_in"}
    return {"action": "raise", "amount": max(min_raise_to, total)}


def active_opponents(state):
    hero = state.get("seat_to_act")
    opponents = []
    for p in state.get("players", []):
        if p.get("seat") == hero:
            continue
        if p.get("is_folded") or p.get("state") == "folded":
            continue
        opponents.append(p)
    return opponents


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def sigmoid(value):
    value = clamp(value, -30.0, 30.0)
    return 1.0 / (1.0 + math.exp(-value))


def blind_seats(state):
    sb = None
    bb = None
    for a in state.get("action_log", []):
        if a.get("action") == "small_blind":
            sb = a.get("seat")
        elif a.get("action") == "big_blind":
            bb = a.get("seat")
    return sb, bb


def acting_position(state):
    players = state.get("players", [])
    n = max(len(players), 1)
    seat = int(state.get("seat_to_act", 0) or 0)
    if n <= 1:
        return 1.0

    sb, bb = blind_seats(state)
    if sb is None or bb is None:
        return seat / max(n - 1, 1)

    if n == 2:
        dealer = sb
        first = sb if state.get("street") == "preflop" else bb
    else:
        dealer = (sb - 1) % n
        first = ((bb + 1) % n) if state.get("street") == "preflop" else ((dealer + 1) % n)

    order = [((first + i) % n) for i in range(n)]
    live = {
        int(p.get("seat", -1))
        for p in players
        if not p.get("is_folded") and p.get("state") != "folded"
    }
    order = [s for s in order if s in live or s == seat]
    if seat not in order or len(order) <= 1:
        return seat / max(n - 1, 1)
    return order.index(seat) / max(len(order) - 1, 1)


def card_ranks(cards):
    return sorted([c[0] for c in cards], key=lambda r: RANK_VALUE.get(r, 0), reverse=True)


def chen_score(cards):
    if len(cards) < 2:
        return 0.0
    ranks = card_ranks(cards)
    high, low = ranks[0], ranks[1]
    suited = cards[0][1] == cards[1][1]

    if high == low:
        return max(CHEN_POINTS[high] * 2.0, 5.0)

    score = CHEN_POINTS[high]
    if suited:
        score += 2.0

    gap = abs(RANK_VALUE[high] - RANK_VALUE[low]) - 1
    if gap == 1:
        score -= 1.0
    elif gap == 2:
        score -= 2.0
    elif gap == 3:
        score -= 4.0
    elif gap >= 4:
        score -= 5.0

    if gap == 0 and RANK_VALUE[high] < 12:
        score += 1.0
    return max(score, 0.0)


def preflop_category(cards):
    if len(cards) < 2:
        return "trash"
    ranks = card_ranks(cards)
    high, low = ranks[0], ranks[1]
    values = sorted([RANK_VALUE[high], RANK_VALUE[low]], reverse=True)
    suited = cards[0][1] == cards[1][1]

    if high == low:
        if values[0] >= RANK_VALUE["Q"]:
            return "premium"
        if values[0] >= RANK_VALUE["T"]:
            return "strong"
        if values[0] >= RANK_VALUE["7"]:
            return "medium"
        return "speculative"

    combo = high + low
    if combo in ("AK", "AQ"):
        return "premium" if suited or combo == "AK" else "strong"
    if combo in ("AJ", "KQ"):
        return "strong" if suited else "medium"
    if combo in ("AT", "KJ", "QJ", "KT"):
        return "medium" if suited else "speculative"
    if suited and values[0] >= RANK_VALUE["T"] and values[1] >= RANK_VALUE["8"]:
        return "medium"
    if suited and abs(values[0] - values[1]) <= 2 and values[0] >= RANK_VALUE["8"]:
        return "speculative"
    if abs(values[0] - values[1]) <= 1 and values[0] >= RANK_VALUE["9"]:
        return "speculative"
    return "trash"


def action_stats(state):
    actions = state.get("action_log", [])
    raises = 0
    for a in actions:
        if a.get("action") in ("raise", "all_in"):
            raises += 1
    return len(actions), raises


def opponent_tendencies(state):
    hero_seat = state.get("seat_to_act")
    hero_id = None
    for p in state.get("players", []):
        if p.get("seat") == hero_seat:
            hero_id = p.get("bot_id")
            break

    rows = []
    for action in state.get("action_log", []):
        if action.get("seat") != hero_seat:
            rows.append(action)
    for action in state.get("match_action_log", []):
        if hero_id is None or action.get("bot_id") != hero_id:
            rows.append(action)

    counted = [
        a for a in rows
        if a.get("action") in ("fold", "check", "call", "raise", "all_in")
    ]
    if not counted:
        return 0.33, 0.10, 0.20, 0

    total = len(counted)
    calls = sum(1 for a in counted if a.get("action") in ("call", "check"))
    raises = sum(1 for a in counted if a.get("action") in ("raise", "all_in"))
    folds = sum(1 for a in counted if a.get("action") == "fold")
    return calls / total, raises / total, folds / total, total


def extract_features(state, equity=None):
    cards = state.get("your_cards", [])
    category = preflop_category(cards)
    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(int(state.get("pot", 0) or 0), 1)
    stack = int(state.get("your_stack", 0) or 0)
    current = int(state.get("current_bet", 0) or 0)
    pos = acting_position(state)
    opponents = len(active_opponents(state))
    pressure = owed / max(pot + owed, 1)
    required = pressure
    spr = stack / pot
    actions, raises = action_stats(state)
    call_rate, raise_rate, fold_rate, history_count = opponent_tendencies(state)
    if equity is None:
        equity = 0.5 if state.get("street") == "preflop" else heuristic_equity(state)

    feats = {
        "bias": 1.0,
        "street_preflop": 1.0 if state.get("street") == "preflop" else 0.0,
        "street_flop": 1.0 if state.get("street") == "flop" else 0.0,
        "street_turn": 1.0 if state.get("street") == "turn" else 0.0,
        "street_river": 1.0 if state.get("street") == "river" else 0.0,
        "pot_s": clamp(pot / 10000.0, 0.0, 3.0),
        "owed_s": clamp(owed / 10000.0, 0.0, 2.0),
        "stack_s": clamp(stack / 10000.0, 0.0, 3.0),
        "spr_s": clamp(spr / 10.0, 0.0, 2.0),
        "pressure": clamp(pressure, 0.0, 1.0),
        "position": clamp(pos, 0.0, 1.0),
        "opponents_s": clamp(opponents / 8.0, 0.0, 1.0),
        "can_check": 1.0 if state.get("can_check") else 0.0,
        "current_bet_s": clamp(current / 10000.0, 0.0, 2.0),
        "chen_s": clamp(chen_score(cards) / 20.0, 0.0, 1.0),
        "equity": clamp(float(equity), 0.0, 1.0),
        "required": clamp(required, 0.0, 1.0),
        "wet_board": 1.0 if board_is_wet(state.get("community_cards", [])) else 0.0,
        "cat_premium": 1.0 if category == "premium" else 0.0,
        "cat_strong": 1.0 if category == "strong" else 0.0,
        "cat_medium": 1.0 if category == "medium" else 0.0,
        "cat_speculative": 1.0 if category == "speculative" else 0.0,
        "cat_trash": 1.0 if category == "trash" else 0.0,
        "actions_s": clamp(actions / 20.0, 0.0, 1.0),
        "raises_s": clamp(raises / 6.0, 0.0, 1.0),
        "opp_call_rate": clamp(call_rate, 0.0, 1.0),
        "opp_raise_rate": clamp(raise_rate, 0.0, 1.0),
        "opp_fold_rate": clamp(fold_rate, 0.0, 1.0),
        "history_s": clamp(history_count / 200.0, 0.0, 1.0),
    }
    return feats


def linear_score(weights, feats):
    score = 0.0
    for name, value in feats.items():
        score += float(weights.get(name, 0.0)) * value
    return score


def model_signals(state, equity):
    if not MODEL:
        return {}
    feats = extract_features(state, equity)
    heads = MODEL.get("heads", {})
    signals = {}
    for name, head in heads.items():
        raw = linear_score(head.get("weights", {}), feats)
        if head.get("activation") == "sigmoid":
            signals[name] = sigmoid(raw)
        elif head.get("activation") == "tanh":
            signals[name] = math.tanh(raw)
        else:
            signals[name] = raw
    return signals


def preflop_decision(state):
    cards = state.get("your_cards", [])
    category = preflop_category(cards)
    score = chen_score(cards)
    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(int(state.get("pot", 0) or 0), 1)
    stack = int(state.get("your_stack", 0) or 0)
    pos = acting_position(state)
    opponents = len(active_opponents(state))
    invested = int(state.get("your_bet_this_street", 0) or 0)
    current = int(state.get("current_bet", 0) or 0)
    min_raise_to = int(state.get("min_raise_to", current + 100) or current + 100)
    bb = max(100, min_raise_to - current)
    max_total = stack + invested
    heads_up = len(state.get("players", [])) == 2
    completing_small_blind = heads_up and invested > 0 and owed > 0 and owed <= bb

    if stack <= 0:
        return safe_check_or_fold(state)

    pressure = owed / max(pot + owed, 1)
    faced_large_raise = owed > max(bb * 2, pot * 0.45)

    if category == "premium":
        if max_total <= min_raise_to or max_total <= current * 2:
            return {"action": "all_in"}
        target = max(min_raise_to, current * 3 + bb, int(pot * 1.15) + owed)
        return raise_to(state, target)

    if category == "strong":
        if faced_large_raise and pos < 0.35 and score < 9.0:
            return safe_fold(state)
        if owed == 0 or pressure < 0.28:
            target = max(min_raise_to, current * 2 + bb, int(pot * 0.9) + owed)
            if pos >= 0.25 or opponents <= 3:
                return raise_to(state, target)
        return safe_check_or_call(state) if pressure < 0.34 else safe_fold(state)

    if category == "medium":
        if state.get("can_check"):
            if pos > 0.55 and opponents <= 3:
                return raise_to(state, max(min_raise_to, int(pot * 0.8)))
            return {"action": "check"}
        if pressure < (0.23 + pos * 0.08) and not faced_large_raise:
            return {"action": "call"}
        return safe_fold(state)

    if category == "speculative":
        if state.get("can_check"):
            return {"action": "check"}
        if completing_small_blind and not faced_large_raise:
            return {"action": "call"}
        implied_depth = max_total / max(owed, 1)
        if pos > 0.45 and pressure < 0.18 and implied_depth >= 18:
            return {"action": "call"}
        if pressure < 0.12 and owed <= bb:
            return {"action": "call"}
        return safe_fold(state)

    if state.get("can_check"):
        return {"action": "check"}
    if completing_small_blind and not faced_large_raise:
        return {"action": "call"}
    if pressure < 0.10 and owed <= bb:
        return {"action": "call"}
    return {"action": "fold"}


def stable_seed(state):
    text = "|".join([
        str(state.get("hand_id", "")),
        str(state.get("street", "")),
        "".join(state.get("your_cards", [])),
        "".join(state.get("community_cards", [])),
        str(len(state.get("action_log", []))),
    ])
    value = 1729
    for ch in text:
        value = (value * 131 + ord(ch)) % 2147483647
    return value


def estimate_equity(state):
    if eval7 is None:
        return heuristic_equity(state)

    hero_cards = [eval7.Card(c) for c in state.get("your_cards", [])]
    board_cards = [eval7.Card(c) for c in state.get("community_cards", [])]
    if len(hero_cards) != 2:
        return 0.0

    known = set([str(c) for c in hero_cards + board_cards])
    deck = [eval7.Card(r + s) for r in RANKS for s in SUITS if r + s not in known]
    opponents = max(1, len(active_opponents(state)))
    street = state.get("street")
    iters = 90 if street == "flop" else 120 if street == "turn" else 180
    if opponents >= 4:
        iters = max(45, iters // 2)
    elif opponents == 3:
        iters = max(60, int(iters * 0.7))

    rng = random.Random(stable_seed(state))
    wins = 0.0
    needed_board = 5 - len(board_cards)
    sample_size = needed_board + opponents * 2
    if sample_size > len(deck):
        return heuristic_equity(state)

    for _ in range(iters):
        draw = rng.sample(deck, sample_size)
        runout = board_cards + draw[:needed_board]
        cursor = needed_board
        hero_score = eval7.evaluate(hero_cards + runout)
        scores = [hero_score]
        for _opp in range(opponents):
            opp_cards = draw[cursor:cursor + 2]
            cursor += 2
            scores.append(eval7.evaluate(opp_cards + runout))
        best = max(scores)
        if hero_score == best:
            winners = sum(1 for s in scores if s == best)
            wins += 1.0 / winners
    return wins / max(iters, 1)


def heuristic_equity(state):
    cards = state.get("your_cards", [])
    board = state.get("community_cards", [])
    ranks = card_ranks(cards + board)
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    pairs = sorted(counts.values(), reverse=True)
    high = max([RANK_VALUE.get(c[0], 0) for c in cards] or [0])
    suited = len(cards) == 2 and cards[0][1] == cards[1][1]

    equity = 0.24 + high / 70.0
    if pairs and pairs[0] >= 3:
        equity += 0.28
    elif len([v for v in counts.values() if v >= 2]) >= 2:
        equity += 0.20
    elif any(c[0] in counts and counts[c[0]] >= 2 for c in cards):
        equity += 0.14
    if suited:
        equity += 0.03
    return min(max(equity, 0.05), 0.88)


def board_is_wet(board):
    if len(board) < 3:
        return False
    suits = {}
    values = []
    for c in board:
        suits[c[1]] = suits.get(c[1], 0) + 1
        values.append(RANK_VALUE.get(c[0], 0))
    values = sorted(set(values))
    flushy = max(suits.values()) >= 3
    connected = any(values[i + 2] - values[i] <= 4 for i in range(max(0, len(values) - 2)))
    paired = len(set(c[0] for c in board)) < len(board)
    return flushy or connected or paired


def postflop_decision(state):
    owed = int(state.get("amount_owed", 0) or 0)
    pot = max(int(state.get("pot", 0) or 0), 1)
    stack = int(state.get("your_stack", 0) or 0)
    invested = int(state.get("your_bet_this_street", 0) or 0)
    opponents = max(1, len(active_opponents(state)))
    pos = acting_position(state)
    equity = estimate_equity(state)
    signals = model_signals(state, equity)
    if "showdown_equity" in signals:
        equity = clamp(0.92 * equity + 0.08 * signals["showdown_equity"], 0.0, 1.0)
    fold_pressure = signals.get("fold_pressure", 0.0)
    danger = signals.get("danger", 0.5)
    chip_ev = signals.get("chip_ev", 0.0)
    required = owed / max(pot + owed, 1)
    spr = stack / max(pot, 1)
    wet = board_is_wet(state.get("community_cards", []))

    if stack <= 0:
        return safe_check_or_fold(state)

    if owed == 0 or state.get("can_check"):
        if equity >= 0.74 or (equity >= 0.66 and spr <= 1.2):
            if spr <= 1.1:
                return {"action": "all_in"}
            fraction = 0.72 if wet else 0.58
            return raise_to(state, invested + int(pot * fraction))
        if opponents <= 2 and pos >= 0.55 and equity >= 0.58:
            return raise_to(state, invested + int(pot * 0.62))
        if opponents <= 2 and pos >= 0.50 and equity >= 0.45:
            if fold_pressure >= 0.72 and danger <= 0.35 and chip_ev >= 0.10:
                return raise_to(state, invested + int(pot * 0.55))
        return {"action": "check"}

    margin = 0.035 + max(0, opponents - 1) * 0.035
    if state.get("street") == "river":
        margin += 0.035
    if signals:
        margin += 0.025 * danger
        if chip_ev > 0:
            margin -= min(0.015, chip_ev * 0.015)

    if equity >= 0.76 and spr <= 1.7:
        return {"action": "all_in"}

    if equity >= max(0.68, required + 0.20):
        fraction = 0.78 if wet else 0.62
        return raise_to(state, invested + owed + int((pot + owed) * fraction))

    if equity >= required + margin:
        return {"action": "call"}

    if signals and chip_ev < -0.55 and equity < required + 0.10:
        return {"action": "fold"}

    if owed <= max(100, pot * 0.08) and equity >= required - 0.025:
        return {"action": "call"}

    return {"action": "fold"}
