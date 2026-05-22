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
NOMINAL_STARTING_STACK = 10000

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
PUBLIC_BELIEF_FEATURE_NAMES = [
    "pbs_street_progress", "pbs_board_cards_s", "pbs_board_high_s",
    "pbs_board_pairing", "pbs_board_flushiness", "pbs_board_connectivity",
    "pbs_live_players_s", "pbs_heads_up", "pbs_multiway",
    "pbs_avg_opp_stack_s", "pbs_big_stack_pressure",
    "pbs_stack_at_risk", "pbs_hero_commitment", "pbs_pot_to_stack",
    "pbs_short_stack", "pbs_bet_size_ratio", "pbs_large_bet_pressure",
    "pbs_action_depth_s", "pbs_recent_raise_depth_s", "pbs_all_in_seen",
    "pbs_last_aggressor_hero", "pbs_range_narrowing",
    "pbs_field_looseness", "pbs_field_aggression",
]
FEATURE_NAMES += PUBLIC_BELIEF_FEATURE_NAMES


def load_bot_data():
    data_dir = os.environ.get("BOT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
    path = os.path.join(data_dir, "model.json")
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if data.get("version") != MODEL_VERSION:
            return None
        return data
    except Exception:
        return None


BOT_DATA = load_bot_data()
MODEL = BOT_DATA if BOT_DATA and BOT_DATA.get("runtime_enabled", False) else None
PREFLOP_STRATEGY = BOT_DATA.get("preflop_strategy", {}) if BOT_DATA else {}
POSTFLOP_EV = BOT_DATA.get("postflop_ev", {}) if BOT_DATA else {}
RANGE_EQUITY = BOT_DATA.get("range_equity", {}) if BOT_DATA else {}
RISK_GATES = BOT_DATA.get("risk_gates", {}) if BOT_DATA else {}
RIVER_BLUEPRINT = BOT_DATA.get("river_blueprint", {}) if BOT_DATA else {}
COMMITMENT_BLUEPRINT = BOT_DATA.get("commitment_blueprint", {}) if BOT_DATA else {}
RANGE_CANDIDATE_CACHE = {}


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
    return seat_position_value(state, seat, state.get("street"))


def seat_position_value(state, seat, street=None):
    players = state.get("players", [])
    n = max(len(players), 1)
    seat = int(seat or 0)
    if n <= 1:
        return 1.0

    sb, bb = blind_seats(state)
    if sb is None or bb is None:
        return seat / max(n - 1, 1)

    if n == 2:
        dealer = sb
        first = sb if (street or state.get("street")) == "preflop" else bb
    else:
        dealer = (sb - 1) % n
        first = ((bb + 1) % n) if (street or state.get("street")) == "preflop" else ((dealer + 1) % n)

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


def hand_key(cards):
    if len(cards) < 2:
        return ""
    ranks = card_ranks(cards)
    high, low = ranks[0], ranks[1]
    if high == low:
        return high + low
    return high + low + ("s" if cards[0][1] == cards[1][1] else "o")


def preflop_position_bucket(state, pos):
    if len(state.get("players", [])) <= 2:
        return "heads_up"
    if pos < 0.20:
        return "early"
    if pos < 0.45:
        return "middle"
    if pos < 0.75:
        return "late"
    return "blind"


def preflop_stack_bucket(stack, invested, bb):
    effective_bb = (stack + invested) / max(bb, 1)
    if effective_bb <= 15:
        return "short"
    if effective_bb <= 40:
        return "medium"
    return "deep"


def configured_hands(section, bucket=None):
    if not isinstance(section, dict):
        return set(section or [])
    hands = set(section.get("all", []))
    if bucket:
        hands.update(section.get(bucket, []))
    return hands


def preflop_table_plan(state, cards, pos, owed, pot, stack, invested, current, min_raise_to, bb, faced_large_raise):
    strategy = PREFLOP_STRATEGY
    if not strategy or not strategy.get("enabled", False):
        return None

    key = hand_key(cards)
    if not key:
        return None

    position = preflop_position_bucket(state, pos)
    depth = preflop_stack_bucket(stack, invested, bb)
    open_ranges = strategy.get("rfi", {})
    raise_ranges = strategy.get("vs_raise_continue", {})
    large_raise_ranges = strategy.get("vs_large_raise_continue", {})
    short_stack_ranges = strategy.get("short_stack_continue", {})

    max_total = stack + invested
    short_stack = depth == "short" or max_total <= bb * 16
    blind_price = owed > 0 and owed <= bb and current <= bb
    unopened = state.get("can_check") or current <= bb
    pressure = owed / max(pot + owed, 1)

    if unopened:
        if not strategy.get("rfi_enabled", False):
            return None
        if key in configured_hands(open_ranges, position):
            return "open_raise"
        if short_stack and key in configured_hands(short_stack_ranges, position):
            return "open_raise"
        if not state.get("can_check") and not blind_price and position not in ("blind", "heads_up"):
            return "open_fold"
        return None

    if faced_large_raise:
        if short_stack and key in configured_hands(short_stack_ranges, position):
            return "large_raise_raise"
        if key in configured_hands(large_raise_ranges, position):
            return "large_raise_raise"
        if pressure >= 0.26 or position in ("early", "middle"):
            return "large_raise_fold"
        return None

    if key in configured_hands(raise_ranges, position):
        return "raise_continue"
    if pressure >= 0.32 and not blind_price:
        return "raise_fold"
    return None


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


def preflop_pressure_control_gate(
    state,
    cards,
    owed,
    pot,
    stack,
    invested,
    current,
    max_total,
    pressure,
    faced_large_raise,
):
    config = RISK_GATES.get("preflop_pressure_control", {})
    if not config.get("enabled", False):
        return None
    if owed <= 0 or len(state.get("players", [])) < config.get("min_table_size", 5):
        return None

    key = hand_key(cards)
    pressure_hands = set(config.get("hands", ["AKo", "AQs", "AQo"]))
    pair_cap_hands = set(config.get("pair_cap_hands", []))
    if key not in pressure_hands and key not in pair_cap_hands:
        return None

    last_aggression = last_opponent_aggression(state)
    if not last_aggression:
        return None

    source_seat = last_aggression.get("seat")
    targeted_source = pressure_source_is_targeted(state, source_seat, config)
    extreme_pressure = preflop_pressure_is_extreme(
        state, source_seat, key, current, owed, stack, pressure, config
    )
    huge_all_in = (
        config.get("gate_huge_all_in", True)
        and last_aggression.get("action") == "all_in"
        and key in set(config.get("huge_all_in_hands", ["AKo", "AQs", "AQo"]))
        and pressure >= config.get("huge_all_in_min_pressure", 0.38)
    )
    if not targeted_source and not extreme_pressure and not huge_all_in:
        return None

    bb = big_blind_amount(state)
    depth_bb = max_total / max(bb, 1)
    if depth_bb <= config.get("allow_short_stack_bb", 18):
        return None
    if invested / max(max_total, 1) >= config.get("allow_committed_ratio", 0.42):
        return None

    large_total = current >= config.get("min_raise_to_bb", 18) * bb
    big_pressure = (
        faced_large_raise
        or pressure >= config.get("min_pressure", 0.30)
        or large_total
        or last_aggression.get("action") == "all_in"
    )
    if not big_pressure:
        return None

    if key in pair_cap_hands:
        left_after_call = stack - owed
        call_leaves_stack = left_after_call >= bb * config.get("pair_cap_min_left_bb", 2)
        can_cap_raise = (
            last_aggression.get("action") != "all_in"
            and owed <= stack * config.get("pair_cap_max_flat_stack_fraction", 0.38)
        )
        can_cap_all_in = (
            last_aggression.get("action") == "all_in"
            and call_leaves_stack
            and owed <= stack * config.get("pair_cap_max_all_in_call_fraction", 0.92)
        )
        if can_cap_raise or can_cap_all_in:
            return {"action": "call"}
        return None

    can_flat = (
        last_aggression.get("action") != "all_in"
        and owed <= stack * config.get("max_flat_stack_fraction", 0.34)
        and pressure <= config.get("max_flat_pressure", 0.50)
    )
    if can_flat:
        return {"action": "call"}
    return safe_fold(state)


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


def player_for_seat(state, seat):
    for player in state.get("players", []):
        if player.get("seat") == seat:
            return player
    return {}


def actions_for_seat(state, seat, include_current=True):
    player = player_for_seat(state, seat)
    bot_id = player.get("bot_id")
    rows = []
    if include_current:
        rows.extend([a for a in state.get("action_log", []) if a.get("seat") == seat])
    for action in state.get("match_action_log", []):
        if bot_id is not None and action.get("bot_id") == bot_id:
            rows.append(action)
        elif bot_id is None and action.get("seat") == seat:
            rows.append(action)
    return rows


def seat_action_profile(state, seat):
    counted = [
        a for a in actions_for_seat(state, seat)
        if a.get("action") in ("fold", "check", "call", "raise", "all_in")
    ]
    if not counted:
        return {"vpip": 0.33, "raise_rate": 0.10, "call_rate": 0.33, "fold_rate": 0.20, "count": 0}

    total = len(counted)
    raises = sum(1 for a in counted if a.get("action") in ("raise", "all_in"))
    calls = sum(1 for a in counted if a.get("action") in ("call", "check"))
    folds = sum(1 for a in counted if a.get("action") == "fold")
    vpip = sum(1 for a in counted if a.get("action") in ("call", "raise", "all_in")) / total
    return {
        "vpip": vpip,
        "raise_rate": raises / total,
        "call_rate": calls / total,
        "fold_rate": folds / total,
        "count": total,
    }


def action_profile_from_rows(rows):
    counted = [
        a for a in rows
        if a.get("action") in ("fold", "check", "call", "raise", "all_in")
    ]
    if not counted:
        return {"vpip": 0.33, "raise_rate": 0.10, "call_rate": 0.33, "fold_rate": 0.20, "count": 0}

    total = len(counted)
    raises = sum(1 for a in counted if a.get("action") in ("raise", "all_in"))
    calls = sum(1 for a in counted if a.get("action") in ("call", "check"))
    folds = sum(1 for a in counted if a.get("action") == "fold")
    vpip = sum(1 for a in counted if a.get("action") in ("call", "raise", "all_in")) / total
    return {
        "vpip": vpip,
        "raise_rate": raises / total,
        "call_rate": calls / total,
        "fold_rate": folds / total,
        "count": total,
    }


def match_actions_for_seat(state, seat):
    player = player_for_seat(state, seat)
    bot_id = player.get("bot_id")
    rows = []
    for action in state.get("match_action_log", []):
        if bot_id is not None and action.get("bot_id") == bot_id:
            rows.append(action)
        elif bot_id is None and action.get("seat") == seat:
            rows.append(action)
    return rows


def recent_action_profile(state, seat, limit):
    rows = match_actions_for_seat(state, seat)
    if not rows:
        rows = [a for a in state.get("action_log", []) if a.get("seat") == seat]
    counted = [
        a for a in rows
        if a.get("action") in ("fold", "check", "call", "raise", "all_in")
    ]
    return action_profile_from_rows(counted[-max(1, int(limit or 1)):])


def big_blind_amount(state):
    for action in state.get("action_log", []):
        if action.get("action") == "big_blind":
            return max(1, int(action.get("amount", 100) or 100))
    return 100


def opponent_archetype(state, seat):
    profile = seat_action_profile(state, seat)
    if profile["count"] < RANGE_EQUITY.get("min_profile_actions", 16):
        return "unknown"
    if profile["vpip"] >= 0.52 and profile["raise_rate"] < 0.12:
        return "loose_passive"
    if profile["vpip"] >= 0.44 and profile["raise_rate"] >= 0.16:
        return "lag"
    if profile["vpip"] <= 0.24 and profile["raise_rate"] <= 0.10:
        return "nitty"
    return "tag"


def last_opponent_aggression(state):
    hero = state.get("seat_to_act")
    for action in reversed(state.get("action_log", [])):
        if action.get("seat") == hero:
            continue
        if action.get("action") in ("raise", "all_in"):
            return action
    return None


def pressure_source_is_targeted(state, seat, config):
    if seat is None:
        return False

    profile = seat_action_profile(state, seat)
    if pressure_profile_matches(
        profile,
        config.get("min_profile_actions", 16),
        config.get("min_vpip", 0.36),
        config.get("min_raise_rate", 0.16),
    ):
        return True

    early_profile = pressure_profile_matches(
        profile,
        config.get("early_min_profile_actions", 6),
        config.get("early_min_vpip", 0.40),
        config.get("early_min_raise_rate", 0.40),
    )
    if early_profile:
        return True

    recent = recent_action_profile(state, seat, config.get("recent_profile_actions", 8))
    return pressure_profile_matches(
        recent,
        config.get("recent_min_profile_actions", 8),
        config.get("recent_min_vpip", 0.34),
        config.get("recent_min_raise_rate", 0.42),
    )


def pressure_profile_matches(profile, min_actions, min_vpip, min_raise_rate):
    if profile["count"] < min_actions:
        return False
    if profile["vpip"] < min_vpip:
        return False
    return profile["raise_rate"] >= min_raise_rate


def preflop_pressure_is_extreme(state, seat, key, current, owed, stack, pressure, config):
    if seat is None or key not in set(config.get("large_pressure_hands", config.get("hands", []))):
        return False

    bb = big_blind_amount(state)
    current_bb = current / max(bb, 1)
    owed_stack_fraction = owed / max(stack, 1)

    unprofiled_size = (
        current_bb >= config.get("unprofiled_min_raise_to_bb", 24)
        and pressure >= config.get("unprofiled_min_pressure", 0.36)
    )
    unprofiled_stack = (
        owed_stack_fraction >= config.get("unprofiled_min_owed_stack_fraction", 0.46)
        and pressure >= config.get("unprofiled_min_pressure", 0.36)
    )
    if unprofiled_size or unprofiled_stack:
        return True

    actions = [
        a for a in state.get("action_log", [])
        if a.get("seat") == seat and a.get("action") in ("raise", "all_in")
    ]
    if len(actions) < config.get("current_hand_min_raises", 2):
        return False
    return current_bb >= config.get("current_hand_min_raise_to_bb", 12)


def current_hand_actions_for_seat(state, seat):
    return [a for a in state.get("action_log", []) if a.get("seat") == seat]


def range_bucket_hands(bucket):
    buckets = RANGE_EQUITY.get("buckets", {})
    hands = buckets.get(bucket, [])
    return set(hands)


def legacy_inferred_range_bucket(state, seat):
    actions = current_hand_actions_for_seat(state, seat)
    archetype = opponent_archetype(state, seat)
    bucket = RANGE_EQUITY.get("archetype_buckets", {}).get(archetype, "unknown")

    raises = sum(1 for a in actions if a.get("action") in ("raise", "all_in"))
    calls = sum(1 for a in actions if a.get("action") == "call")
    all_ins = sum(1 for a in actions if a.get("action") == "all_in")
    player = player_for_seat(state, seat)
    current_bet = int(state.get("current_bet", 0) or 0)
    player_street_bet = int(player.get("bet_this_street", 0) or 0)
    postflop_pressure = (
        state.get("street") != "preflop"
        and current_bet > 0
        and player_street_bet >= current_bet
    )

    if all_ins or (postflop_pressure and player.get("state") == "all_in"):
        return "stackoff"
    if postflop_pressure and current_bet >= max(int(state.get("pot", 0) or 0) * 0.28, 300):
        return "postflop_raise"
    if raises >= 2:
        return "three_bet"
    if raises == 1:
        return "open_raise"
    if postflop_pressure:
        return "postflop_call"
    if calls:
        return "preflop_call"
    return bucket


def inferred_range_bucket(state, seat):
    if not RANGE_EQUITY.get("size_position_buckets_enabled", False):
        return legacy_inferred_range_bucket(state, seat)

    actions = current_hand_actions_for_seat(state, seat)
    archetype = opponent_archetype(state, seat)
    bucket = RANGE_EQUITY.get("archetype_buckets", {}).get(archetype, "unknown")

    raises = sum(1 for a in actions if a.get("action") in ("raise", "all_in"))
    calls = sum(1 for a in actions if a.get("action") == "call")
    all_ins = sum(1 for a in actions if a.get("action") == "all_in")
    player = player_for_seat(state, seat)
    street = state.get("street")
    bb = big_blind_amount(state)
    current_bet = int(state.get("current_bet", 0) or 0)
    player_street_bet = int(player.get("bet_this_street", 0) or 0)
    pot = max(int(state.get("pot", 0) or 0), 1)
    current_bb = current_bet / max(bb, 1)
    bet_ratio = current_bet / pot
    position = preflop_position_bucket(state, seat_position_value(state, seat, "preflop"))
    position_bucket = {
        "early": "early_open",
        "middle": "middle_open",
        "late": "late_open",
        "blind": "blind_open",
        "heads_up": "heads_up_open",
    }.get(position, "open_raise")

    postflop_pressure = (
        street != "preflop"
        and current_bet > 0
        and player_street_bet >= current_bet
    )

    if all_ins or (postflop_pressure and player.get("state") == "all_in"):
        return "stackoff"

    if postflop_pressure:
        if bet_ratio >= RANGE_EQUITY.get("postflop_stackoff_bet_ratio", 0.68):
            return "stackoff"
        if (
            raises >= RANGE_EQUITY.get("postflop_raise_war_count", 2)
            or bet_ratio >= RANGE_EQUITY.get("postflop_large_bet_ratio", 0.32)
            or current_bet >= RANGE_EQUITY.get("postflop_large_bet_min", 300)
        ):
            return "postflop_raise"
        return "postflop_call"

    if raises >= 2:
        if current_bb >= RANGE_EQUITY.get("huge_preflop_raise_bb", 24):
            return "stackoff"
        if current_bb >= RANGE_EQUITY.get("large_preflop_raise_bb", 12):
            return "large_preflop_raise"
        return "three_bet"

    if raises == 1:
        if current_bb >= RANGE_EQUITY.get("huge_preflop_raise_bb", 24):
            return "stackoff"
        if current_bb >= RANGE_EQUITY.get("large_preflop_raise_bb", 12):
            return "large_preflop_raise"
        return position_bucket

    if calls:
        return "preflop_call"
    return bucket


def add_bucket_weight(weights, bucket, amount):
    if not bucket or amount <= 0:
        return
    weights[bucket] = weights.get(bucket, 0.0) + float(amount)


def normalize_bucket_weights(weights):
    cleaned = {bucket: max(0.0, weight) for bucket, weight in weights.items() if weight > 0}
    total = sum(cleaned.values())
    if total <= 0:
        return {"unknown": 1.0}
    minimum = RANGE_EQUITY.get("bayesian_min_bucket_weight", 0.0)
    normalized = {bucket: weight / total for bucket, weight in cleaned.items()}
    if minimum <= 0:
        return normalized
    floored = {bucket: max(weight, minimum) for bucket, weight in normalized.items()}
    total = sum(floored.values())
    return {bucket: weight / total for bucket, weight in floored.items()}


def inferred_range_distribution(state, seat):
    if not RANGE_EQUITY.get("bayesian_enabled", False):
        return {inferred_range_bucket(state, seat): 1.0}

    actions = current_hand_actions_for_seat(state, seat)
    archetype = opponent_archetype(state, seat)
    archetype_bucket = RANGE_EQUITY.get("archetype_buckets", {}).get(archetype, "unknown")
    profile = seat_action_profile(state, seat)
    weights = {}

    add_bucket_weight(weights, archetype_bucket, 0.36)
    add_bucket_weight(weights, "unknown", 0.10 if profile["count"] < RANGE_EQUITY.get("min_profile_actions", 16) else 0.04)

    raises = sum(1 for a in actions if a.get("action") in ("raise", "all_in"))
    calls = sum(1 for a in actions if a.get("action") == "call")
    all_ins = sum(1 for a in actions if a.get("action") == "all_in")
    player = player_for_seat(state, seat)
    current_bet = int(state.get("current_bet", 0) or 0)
    player_street_bet = int(player.get("bet_this_street", 0) or 0)
    postflop_pressure = (
        state.get("street") != "preflop"
        and current_bet > 0
        and player_street_bet >= current_bet
    )

    if all_ins or (postflop_pressure and player.get("state") == "all_in"):
        add_bucket_weight(weights, "stackoff", 0.72)
        add_bucket_weight(weights, "postflop_raise", 0.18)
        add_bucket_weight(weights, "three_bet", 0.10)
    elif postflop_pressure and current_bet >= max(int(state.get("pot", 0) or 0) * 0.28, 300):
        add_bucket_weight(weights, "postflop_raise", 0.58)
        add_bucket_weight(weights, "stackoff", 0.20)
        add_bucket_weight(weights, "postflop_call", 0.10)
    elif raises >= 2:
        add_bucket_weight(weights, "three_bet", 0.54)
        add_bucket_weight(weights, "stackoff", 0.24)
        add_bucket_weight(weights, "open_raise", 0.12)
    elif raises == 1:
        add_bucket_weight(weights, "open_raise", 0.48)
        add_bucket_weight(weights, "three_bet", 0.14)
        add_bucket_weight(weights, "stackoff", 0.08)
    elif postflop_pressure:
        add_bucket_weight(weights, "postflop_call", 0.58)
        add_bucket_weight(weights, archetype_bucket, 0.18)
    elif calls:
        add_bucket_weight(weights, "preflop_call", 0.56)
        add_bucket_weight(weights, archetype_bucket, 0.18)

    if profile["count"] >= RANGE_EQUITY.get("min_profile_actions", 16):
        if profile["raise_rate"] >= 0.22:
            add_bucket_weight(weights, "postflop_raise", 0.12)
            add_bucket_weight(weights, "stackoff", 0.08)
        if profile["vpip"] >= 0.52 and profile["raise_rate"] < 0.14:
            add_bucket_weight(weights, "loose_passive", 0.20)
            add_bucket_weight(weights, "postflop_call", 0.12)
        if profile["vpip"] <= 0.24 and profile["raise_rate"] <= 0.10:
            add_bucket_weight(weights, "nitty", 0.20)

    return normalize_bucket_weights(weights)


def straight_draw_like(cards, board):
    values = sorted({RANK_VALUE.get(c[0], 0) for c in cards + board})
    if RANK_VALUE["A"] in values:
        values = sorted(set(values + [1]))
    for i in range(len(values)):
        window = [v for v in values if values[i] <= v <= values[i] + 4]
        if len(window) >= 4:
            return True
    return False


def combo_board_score(cards, board):
    if not board:
        return 0.0
    ranks = {}
    suits = {}
    for card in cards + board:
        ranks[card[0]] = ranks.get(card[0], 0) + 1
        suits[card[1]] = suits.get(card[1], 0) + 1

    counts = sorted(ranks.values(), reverse=True)
    board_high = max([RANK_VALUE.get(c[0], 0) for c in board] or [0])
    pair_cards = [
        RANK_VALUE.get(card[0], 0)
        for card in cards
        if ranks.get(card[0], 0) >= 2
    ]
    flush_draw = max(suits.values() or [0]) >= 4
    straight_draw = straight_draw_like(cards, board)

    if counts[0] >= 3:
        return 4.0
    if len([c for c in counts if c >= 2]) >= 2:
        return 3.2
    if pair_cards:
        best_pair = max(pair_cards)
        if best_pair > board_high:
            return 2.6
        if best_pair == board_high:
            return 2.2
        return 1.25
    if flush_draw and straight_draw:
        return 2.0
    if flush_draw or straight_draw:
        return 1.25
    if max([RANK_VALUE.get(c[0], 0) for c in cards] or [0]) >= RANK_VALUE["A"]:
        return 0.55
    return 0.0


def combo_passes_postflop_filter(cards, board, bucket):
    if not board:
        return True
    key = hand_key(cards)
    score = combo_board_score(cards, board)
    premium = key in range_bucket_hands("premium")
    if bucket == "stackoff":
        return premium or score >= RANGE_EQUITY.get("stackoff_min_board_score", 2.2)
    if bucket == "postflop_raise":
        return premium or score >= RANGE_EQUITY.get("raise_min_board_score", 1.2)
    if bucket == "postflop_call":
        return premium or score >= RANGE_EQUITY.get("call_min_board_score", 0.8)
    return True


def candidate_hands_for_bucket(bucket, known, board):
    cache_key = (bucket, tuple(sorted(known)), tuple(board))
    if cache_key in RANGE_CANDIDATE_CACHE:
        return RANGE_CANDIDATE_CACHE[cache_key]

    allowed = range_bucket_hands(bucket)
    if not allowed:
        allowed = range_bucket_hands("unknown")
    candidates = []
    deck_cards = [r + s for r in RANKS for s in SUITS if r + s not in known]
    for i, first in enumerate(deck_cards):
        for second in deck_cards[i + 1:]:
            cards = [first, second]
            if hand_key(cards) not in allowed:
                continue
            if not combo_passes_postflop_filter(cards, board, bucket):
                continue
            candidates.append((first, second))
    if len(RANGE_CANDIDATE_CACHE) >= RANGE_EQUITY.get("candidate_cache_limit", 256):
        RANGE_CANDIDATE_CACHE.clear()
    RANGE_CANDIDATE_CACHE[cache_key] = candidates
    return candidates


def weighted_candidates_for_distribution(distribution, known, board):
    weighted = []
    for bucket, bucket_weight in distribution.items():
        candidates = candidate_hands_for_bucket(bucket, known, board)
        if not candidates and bucket not in ("unknown", "loose_passive"):
            candidates = candidate_hands_for_bucket("unknown", known, board)
        if not candidates:
            continue
        combo_weight = bucket_weight / len(candidates)
        weighted.extend((first, second, combo_weight) for first, second in candidates)
    return weighted


def inferred_opponent_ranges(state, known, board):
    ranges = []
    for opponent in active_opponents(state):
        seat = opponent.get("seat")
        distribution = inferred_range_distribution(state, seat)
        if RANGE_EQUITY.get("bayesian_enabled", False):
            candidates = weighted_candidates_for_distribution(distribution, known, board)
        else:
            bucket = next(iter(distribution))
            candidates = candidate_hands_for_bucket(bucket, known, board)
            if not candidates and bucket not in ("unknown", "loose_passive"):
                candidates = candidate_hands_for_bucket("unknown", known, board)
        ranges.append(candidates)
    return ranges


def weighted_combo_choice(available, rng):
    if not available:
        return None
    if len(available[0]) < 3:
        return rng.choice(available)

    total = sum(max(0.0, combo[2]) for combo in available)
    if total <= 0:
        return rng.choice(available)
    roll = rng.random() * total
    cursor = 0.0
    for combo in available:
        cursor += max(0.0, combo[2])
        if cursor >= roll:
            return combo
    return available[-1]


def monte_carlo_equity(hero_cards, board_cards, deck_cards, opponents, iters, rng, range_candidates=None):
    wins = 0.0
    needed_board = 5 - len(board_cards)
    if needed_board < 0:
        return None

    card_objs = {card: eval7.Card(card) for card in set(deck_cards + hero_cards + board_cards)}
    hero_eval_cards = [card_objs[c] for c in hero_cards]
    known = set(hero_cards + board_cards)

    for _ in range(iters):
        used = set(known)
        opponent_cards = []

        if range_candidates:
            for candidates in range_candidates:
                available = [
                    combo for combo in candidates
                    if combo[0] not in used and combo[1] not in used
                ]
                if available:
                    chosen = weighted_combo_choice(available, rng)
                    first, second = chosen[0], chosen[1]
                else:
                    remaining = [card for card in deck_cards if card not in used]
                    if len(remaining) < 2:
                        return None
                    first, second = rng.sample(remaining, 2)
                used.add(first)
                used.add(second)
                opponent_cards.append([first, second])
        else:
            sample_size = needed_board + opponents * 2
            remaining = [card for card in deck_cards if card not in used]
            if sample_size > len(remaining):
                return None
            draw = rng.sample(remaining, sample_size)
            runout_cards = board_cards + draw[:needed_board]
            for index in range(needed_board, sample_size, 2):
                first, second = draw[index], draw[index + 1]
                used.add(first)
                used.add(second)
                opponent_cards.append([first, second])
            runout_eval = [card_objs[c] for c in runout_cards]
            hero_score = eval7.evaluate(hero_eval_cards + runout_eval)
            scores = [hero_score]
            for cards in opponent_cards:
                scores.append(eval7.evaluate([card_objs[c] for c in cards] + runout_eval))

            best = max(scores)
            if hero_score == best:
                winners = sum(1 for score in scores if score == best)
                wins += 1.0 / winners
            continue

        remaining_board = [card for card in deck_cards if card not in used]
        if needed_board > len(remaining_board):
            return None
        runout_cards = board_cards + rng.sample(remaining_board, needed_board)
        runout_eval = [card_objs[c] for c in runout_cards]
        hero_score = eval7.evaluate(hero_eval_cards + runout_eval)
        scores = [hero_score]
        for cards in opponent_cards:
            scores.append(eval7.evaluate([card_objs[c] for c in cards] + runout_eval))

        best = max(scores)
        if hero_score == best:
            winners = sum(1 for score in scores if score == best)
            wins += 1.0 / winners
    return wins / max(iters, 1)


def board_texture_features(board):
    if not board:
        return {
            "pbs_board_cards_s": 0.0,
            "pbs_board_high_s": 0.0,
            "pbs_board_pairing": 0.0,
            "pbs_board_flushiness": 0.0,
            "pbs_board_connectivity": 0.0,
        }

    values = sorted({RANK_VALUE.get(c[0], 0) for c in board})
    suits = {}
    for card in board:
        suits[card[1]] = suits.get(card[1], 0) + 1

    if len(values) > 1:
        gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        avg_gap = sum(gaps) / len(gaps)
        connectivity = 1.0 - clamp(avg_gap / 5.0, 0.0, 1.0)
    else:
        connectivity = 0.0

    return {
        "pbs_board_cards_s": clamp(len(board) / 5.0, 0.0, 1.0),
        "pbs_board_high_s": clamp(max(values or [0]) / 14.0, 0.0, 1.0),
        "pbs_board_pairing": 1.0 if len({c[0] for c in board}) < len(board) else 0.0,
        "pbs_board_flushiness": clamp(max(suits.values()) / max(len(board), 1), 0.0, 1.0),
        "pbs_board_connectivity": connectivity,
    }


def extract_public_belief_state(state):
    street = state.get("street")
    street_progress = {"preflop": 0.0, "flop": 0.33, "turn": 0.66, "river": 1.0}.get(street, 0.0)
    hero_seat = state.get("seat_to_act")
    players = state.get("players", [])
    live_players = [
        p for p in players
        if not p.get("is_folded") and p.get("state") != "folded"
    ]
    opponents = [p for p in live_players if p.get("seat") != hero_seat]
    pot = max(int(state.get("pot", 0) or 0), 1)
    owed = int(state.get("amount_owed", 0) or 0)
    stack = int(state.get("your_stack", 0) or 0)
    invested = int(state.get("your_bet_this_street", 0) or 0)
    current = int(state.get("current_bet", 0) or 0)
    min_raise_to = int(state.get("min_raise_to", 0) or 0)
    bb = max(100, min_raise_to - current)
    actions, raises = action_stats(state)
    call_rate, raise_rate, fold_rate, history_count = opponent_tendencies(state)

    opp_stacks = [
        int(p.get("stack", stack) or stack)
        for p in opponents
    ]
    avg_opp_stack = sum(opp_stacks) / len(opp_stacks) if opp_stacks else stack
    last_aggressor = None
    recent_raises = 0
    all_in_seen = False
    for action in state.get("action_log", [])[-12:]:
        if action.get("action") in ("raise", "all_in"):
            last_aggressor = action.get("seat")
            recent_raises += 1
        if action.get("action") == "all_in":
            all_in_seen = True

    pressure = owed / max(pot + owed, 1)
    bet_size_ratio = current / max(pot, 1)
    stack_at_risk = owed / max(stack, 1)
    hero_commitment = invested / max(stack + invested, 1)
    range_narrowing = (
        0.10
        + 0.26 * clamp(raises / 6.0, 0.0, 1.0)
        + 0.26 * pressure
        + 0.24 * raise_rate
        - 0.12 * call_rate
        - 0.08 * fold_rate
    )
    field_looseness = 0.45 * call_rate + 0.25 * (1.0 - fold_rate) + 0.15 * (1.0 - clamp(range_narrowing, 0.0, 1.0))

    belief = {
        "pbs_street_progress": street_progress,
        "pbs_live_players_s": clamp(len(live_players) / 8.0, 0.0, 1.0),
        "pbs_heads_up": 1.0 if len(opponents) == 1 else 0.0,
        "pbs_multiway": 1.0 if len(opponents) > 1 else 0.0,
        "pbs_avg_opp_stack_s": clamp(avg_opp_stack / NOMINAL_STARTING_STACK, 0.0, 3.0),
        "pbs_big_stack_pressure": 1.0 if opp_stacks and max(opp_stacks) > stack * 1.35 else 0.0,
        "pbs_stack_at_risk": clamp(stack_at_risk, 0.0, 1.0),
        "pbs_hero_commitment": clamp(hero_commitment, 0.0, 1.0),
        "pbs_pot_to_stack": clamp(pot / max(pot + stack, 1), 0.0, 1.0),
        "pbs_short_stack": 1.0 if stack <= bb * 18 else 0.0,
        "pbs_bet_size_ratio": clamp(bet_size_ratio, 0.0, 3.0),
        "pbs_large_bet_pressure": 1.0 if owed > max(pot * 0.45, bb * 3) else 0.0,
        "pbs_action_depth_s": clamp((actions + history_count * 0.1) / 30.0, 0.0, 1.0),
        "pbs_recent_raise_depth_s": clamp(recent_raises / 4.0, 0.0, 1.0),
        "pbs_all_in_seen": 1.0 if all_in_seen else 0.0,
        "pbs_last_aggressor_hero": 1.0 if last_aggressor == hero_seat else 0.0,
        "pbs_range_narrowing": clamp(range_narrowing, 0.0, 1.0),
        "pbs_field_looseness": clamp(field_looseness, 0.0, 1.0),
        "pbs_field_aggression": clamp(raise_rate, 0.0, 1.0),
    }
    belief.update(board_texture_features(state.get("community_cards", [])))
    return belief


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
    feats.update(extract_public_belief_state(state))
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
    pressure_gate = preflop_pressure_control_gate(
        state, cards, owed, pot, stack, invested, current, max_total, pressure, faced_large_raise
    )
    if pressure_gate:
        return pressure_gate

    table_plan = preflop_table_plan(
        state, cards, pos, owed, pot, stack, invested, current, min_raise_to, bb, faced_large_raise
    )

    if table_plan in ("open_raise", "raise_continue"):
        target = max(min_raise_to, current * 2 + bb, int(pot * 0.9) + owed)
        return raise_to(state, target)
    if table_plan == "large_raise_raise":
        if max_total <= min_raise_to or max_total <= current * 2:
            return {"action": "all_in"}
        target = max(min_raise_to, current * 3 + bb, int(pot * 1.15) + owed)
        return raise_to(state, target)
    if table_plan in ("open_fold", "large_raise_fold", "raise_fold"):
        return safe_fold(state)

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

    hero_cards = list(state.get("your_cards", []))
    board_cards = list(state.get("community_cards", []))
    if len(hero_cards) != 2:
        return 0.0

    known = set(hero_cards + board_cards)
    deck = [r + s for r in RANKS for s in SUITS if r + s not in known]
    opponent_players = active_opponents(state)
    opponents = max(1, len(opponent_players))
    street = state.get("street")
    iters = 90 if street == "flop" else 120 if street == "turn" else 180
    if opponents >= 4:
        iters = max(45, iters // 2)
    elif opponents == 3:
        iters = max(60, int(iters * 0.7))

    if 5 - len(board_cards) + opponents * 2 > len(deck):
        return heuristic_equity(state)

    pot = max(int(state.get("pot", 0) or 0), 1)
    stack = int(state.get("your_stack", 0) or 0)
    owed = int(state.get("amount_owed", 0) or 0)
    spr = stack / max(pot, 1)
    pressure = owed / max(pot + owed, 1)
    high_leverage = (
        spr <= RANGE_EQUITY.get("max_spr", 2.2)
        or pressure >= RANGE_EQUITY.get("min_pressure", 0.28)
        or opponents >= RANGE_EQUITY.get("always_opponents", 4)
    )

    use_ranges = (
        RANGE_EQUITY.get("enabled", False)
        and street != "preflop"
        and opponents >= RANGE_EQUITY.get("min_opponents", 2)
        and len(state.get("players", [])) >= RANGE_EQUITY.get("min_table_size", 5)
        and (high_leverage or not RANGE_EQUITY.get("high_leverage_only", True))
    )

    if use_ranges:
        ranges = inferred_opponent_ranges(state, known, board_cards)
        if ranges and any(ranges):
            range_rng = random.Random(stable_seed(state) + 7919)
            range_equity = monte_carlo_equity(
                hero_cards, board_cards, deck, opponents, iters, range_rng, ranges
            )
            if range_equity is not None:
                range_equity = bayesian_downside_equity(
                    state, range_equity, ranges, pressure, spr, opponents, high_leverage
                )
                blend = clamp(RANGE_EQUITY.get("blend", 0.70), 0.0, 1.0)
                if blend >= 0.999:
                    return range_equity
                uniform_rng = random.Random(stable_seed(state) + 104729)
                uniform_equity = monte_carlo_equity(
                    hero_cards, board_cards, deck, opponents, iters, uniform_rng
                )
                if uniform_equity is not None:
                    return clamp(blend * range_equity + (1.0 - blend) * uniform_equity, 0.0, 1.0)

    rng = random.Random(stable_seed(state))
    equity = monte_carlo_equity(hero_cards, board_cards, deck, opponents, iters, rng)
    if equity is None:
        return heuristic_equity(state)
    return equity


def bayesian_downside_equity(state, equity, ranges, pressure, spr, opponents, high_leverage):
    if not RANGE_EQUITY.get("bayesian_downside_enabled", False):
        return equity
    if not RANGE_EQUITY.get("bayesian_enabled", False):
        return equity
    if not high_leverage:
        return equity
    if state.get("amount_owed", 0) <= 0 and spr > RANGE_EQUITY.get("bayesian_downside_max_spr", 1.45):
        return equity

    weighted_total = 0.0
    top_heavy_total = 0.0
    risky_keys = range_bucket_hands("stackoff") | range_bucket_hands("postflop_raise") | range_bucket_hands("three_bet")
    for candidates in ranges:
        for combo in candidates:
            weight = combo[2] if len(combo) >= 3 else 1.0 / max(len(candidates), 1)
            weighted_total += weight
            if hand_key([combo[0], combo[1]]) in risky_keys:
                top_heavy_total += weight

    if weighted_total <= 0:
        return equity
    risk_mass = clamp(top_heavy_total / weighted_total, 0.0, 1.0)
    if risk_mass < RANGE_EQUITY.get("bayesian_downside_min_risk_mass", 0.28):
        return equity

    board_risk = board_stackoff_risk_score(state.get("community_cards", []))
    discount = RANGE_EQUITY.get("bayesian_downside_base_discount", 0.0)
    discount += risk_mass * RANGE_EQUITY.get("bayesian_downside_risk_weight", 0.030)
    discount += pressure * RANGE_EQUITY.get("bayesian_downside_pressure_weight", 0.025)
    discount += min(board_risk, 3.0) * RANGE_EQUITY.get("bayesian_downside_board_weight", 0.008)
    if opponents >= 3:
        discount += RANGE_EQUITY.get("bayesian_downside_multiway_discount", 0.010)
    discount = min(discount, RANGE_EQUITY.get("bayesian_downside_max_discount", 0.055))
    return clamp(equity - discount, 0.02, 0.98)


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


def board_is_scary_for_stackoff(board):
    if len(board) < 3:
        return False
    return board_is_wet(board) or any(card[0] == "A" for card in board)


def hero_hand_type(cards, board):
    if len(cards) < 2 or len(board) < 3:
        return "unknown"
    if eval7 is not None:
        try:
            score = eval7.evaluate([eval7.Card(card) for card in cards + board])
            return str(eval7.handtype(score))
        except Exception:
            pass

    ranks = {}
    for card in cards + board:
        ranks[card[0]] = ranks.get(card[0], 0) + 1
    counts = sorted(ranks.values(), reverse=True)
    if counts and counts[0] >= 3:
        return "Trips"
    if len([count for count in counts if count >= 2]) >= 2:
        return "Two Pair"
    if counts and counts[0] >= 2:
        return "Pair"
    return "High Card"


def has_strong_draw(cards, board):
    suits = {}
    for card in cards + board:
        suits[card[1]] = suits.get(card[1], 0) + 1

    flush_draw = False
    nut_flush_draw = False
    for suit, count in suits.items():
        if count >= 4 and any(card[1] == suit for card in cards):
            flush_draw = True
            if any(card == "A" + suit for card in cards):
                nut_flush_draw = True

    return nut_flush_draw or (flush_draw and straight_draw_like(cards, board))


def current_pressure_aggression(state):
    hero = state.get("seat_to_act")
    current = int(state.get("current_bet", 0) or 0)
    if current <= 0:
        return None

    for action in reversed(state.get("action_log", [])):
        seat = action.get("seat")
        if seat == hero or action.get("action") not in ("raise", "all_in"):
            continue
        player = player_for_seat(state, seat)
        street_bet = int(player.get("bet_this_street", 0) or 0)
        if street_bet >= current or action.get("action") == "all_in":
            return action
    return None


def board_stackoff_risk_score(board):
    if len(board) < 3:
        return 0.0

    suits = {}
    ranks = {}
    values = []
    for card in board:
        suits[card[1]] = suits.get(card[1], 0) + 1
        ranks[card[0]] = ranks.get(card[0], 0) + 1
        values.append(RANK_VALUE.get(card[0], 0))

    score = 0.0
    max_suit = max(suits.values() or [0])
    if max_suit >= 4:
        score += 2.0
    elif max_suit >= 3:
        score += 1.0

    max_rank_count = max(ranks.values() or [0])
    if max_rank_count >= 3:
        score += 1.6
    elif len(ranks) < len(board):
        score += 1.0

    unique_values = sorted(set(values))
    wheel_values = sorted(set(unique_values + ([1] if RANK_VALUE["A"] in unique_values else [])))
    if any(wheel_values[i + 3] - wheel_values[i] <= 5 for i in range(max(0, len(wheel_values) - 3))):
        score += 1.2
    elif any(wheel_values[i + 2] - wheel_values[i] <= 4 for i in range(max(0, len(wheel_values) - 2))):
        score += 0.8

    if max(unique_values or [0]) >= RANK_VALUE["A"]:
        score += 0.25
    return score


def hero_pair_quality(cards, board):
    if len(cards) < 2 or len(board) < 3:
        return "unknown"

    counts = {}
    for card in cards + board:
        counts[card[0]] = counts.get(card[0], 0) + 1
    board_values = sorted({RANK_VALUE.get(card[0], 0) for card in board}, reverse=True)
    board_high = board_values[0] if board_values else 0
    board_second = board_values[1] if len(board_values) > 1 else 0

    if cards[0][0] == cards[1][0]:
        pair_value = RANK_VALUE.get(cards[0][0], 0)
        if pair_value > board_high:
            return "overpair"
        if pair_value >= board_high:
            return "top_pair"
        return "underpair"

    paired_values = [
        RANK_VALUE.get(card[0], 0)
        for card in cards
        if counts.get(card[0], 0) >= 2
    ]
    if paired_values:
        pair_value = max(paired_values)
        if pair_value >= board_high:
            return "top_pair"
        if pair_value >= board_second:
            return "second_pair"
        return "weak_pair"

    if len({card[0] for card in board}) < len(board):
        return "board_pair"
    return "unknown"


def recent_pressure_raise_count(state):
    hero = state.get("seat_to_act")
    recent = state.get("action_log", [])[-10:]
    return sum(
        1 for action in recent
        if action.get("seat") != hero and action.get("action") in ("raise", "all_in")
    )


def postflop_wet_stackoff_veto(state, equity, bet_amount, owed, pot, stack, opponents, spr):
    config = RISK_GATES.get("postflop_wet_stackoff", {})
    if not config.get("enabled", False):
        return None
    if len(state.get("players", [])) < config.get("min_table_size", 5):
        return None
    if equity >= config.get("never_veto_equity", 0.90):
        return None

    cards = state.get("your_cards", [])
    board = state.get("community_cards", [])
    if not board_is_scary_for_stackoff(board):
        return None

    hand_type = hero_hand_type(cards, board)
    non_nut_types = set(config.get("non_nut_handtypes", ["High Card", "Pair", "Two Pair"]))
    if hand_type not in non_nut_types:
        return None
    if config.get("allow_strong_draws", True) and has_strong_draw(cards, board):
        return None

    risk = max(owed, min(stack, int(bet_amount or 0)))
    if stack <= 0:
        return None
    committed = risk >= stack * config.get("min_stackoff_fraction", 0.62)
    leaves_dust = stack - risk <= big_blind_amount(state) * config.get("dust_bb", 2.0)
    if not (committed or leaves_dust):
        return None

    pressure = owed / max(pot + owed, 1)
    pressure_action = current_pressure_aggression(state)
    pressure_targeted = (
        pressure_action is not None
        and pressure_source_is_targeted(state, pressure_action.get("seat"), config)
    )
    big_pressure = (
        owed > 0
        and (
            pressure >= config.get("min_pressure", 0.26)
            or owed >= stack * config.get("min_call_stack_fraction", 0.50)
            or (pressure_action is not None and pressure_action.get("action") == "all_in")
        )
    )
    self_stackoff = (
        owed == 0
        and config.get("veto_self_stackoff", True)
        and risk >= stack * config.get("self_stackoff_fraction", 0.78)
        and spr <= config.get("self_stackoff_max_spr", 1.35)
    )

    if big_pressure and config.get("profile_pressure_only", False):
        if not pressure_targeted:
            return None
    if not (big_pressure or self_stackoff):
        return None

    return {"action": "fold"} if owed > 0 else {"action": "check"}


def postflop_stackoff_ev_gate(state, equity, bet_amount, owed, pot, stack, opponents, spr):
    config = RISK_GATES.get("postflop_stackoff_ev", {})
    if not config.get("enabled", False):
        return None
    if config.get("river_only", False) and state.get("street") != "river":
        return None
    if len(state.get("players", [])) < config.get("min_table_size", 5):
        return None
    if equity >= config.get("never_veto_equity", 0.86):
        return None

    board = state.get("community_cards", [])
    if len(board) < config.get("min_board_cards", 3):
        return None

    cards = state.get("your_cards", [])
    hand_type = hero_hand_type(cards, board)
    vulnerable_types = set(config.get("vulnerable_handtypes", ["Pair", "Two Pair"]))
    if hand_type not in vulnerable_types:
        return None
    if config.get("allow_strong_draws", True) and has_strong_draw(cards, board):
        return None

    if stack <= 0:
        return None
    risk = max(int(owed or 0), min(int(stack), int(bet_amount or 0)))
    if risk <= 0:
        return None

    bb = big_blind_amount(state)
    risk_fraction = risk / max(stack, 1)
    leaves_dust = stack - risk <= bb * config.get("dust_bb", 2.0)
    stack_threat = risk_fraction >= config.get("min_stackoff_fraction", 0.58) or leaves_dust
    if not stack_threat:
        return None

    board_score = board_stackoff_risk_score(board)
    pair_quality = hero_pair_quality(cards, board) if hand_type == "Pair" else "two_pair"
    pressure = owed / max(pot + owed, 1)
    pressure_action = current_pressure_aggression(state)
    pressure_targeted = (
        pressure_action is not None
        and pressure_source_is_targeted(state, pressure_action.get("seat"), config)
    )
    facing_all_in = owed > 0 and (
        owed >= stack * config.get("all_in_owed_fraction", 0.92)
        or (pressure_action is not None and pressure_action.get("action") == "all_in")
    )
    raise_war = recent_pressure_raise_count(state) >= config.get("min_recent_raises", 2)
    self_stackoff = (
        owed == 0
        and config.get("veto_self_stackoff", False)
        and risk_fraction >= config.get("self_stackoff_fraction", 0.74)
    )
    proposed_raise = owed > 0 and risk > owed
    if owed <= 0 and not self_stackoff:
        return None
    if proposed_raise and pressure_action is None and not facing_all_in:
        return None

    pressure_context = (
        pressure >= config.get("min_pressure", 0.30)
        or pressure_targeted
        or facing_all_in
        or (raise_war and pressure >= config.get("min_raise_war_pressure", 0.20))
        or self_stackoff
    )
    if not pressure_context:
        return None

    marginal_pair = pair_quality in ("underpair", "second_pair", "weak_pair", "board_pair", "unknown")
    if hand_type == "Pair":
        if marginal_pair:
            min_board_score = config.get("marginal_pair_min_board_score", 0.0)
            min_realized = config.get("marginal_pair_min_realized_equity", 0.66)
        elif pair_quality == "top_pair":
            min_board_score = config.get("top_pair_min_board_score", 0.5)
            min_realized = config.get("top_pair_min_realized_equity", 0.70)
        else:
            min_board_score = config.get("overpair_min_board_score", 0.8)
            min_realized = config.get("overpair_min_realized_equity", 0.72)
    else:
        min_board_score = config.get("two_pair_min_board_score", 0.8)
        min_realized = config.get("two_pair_min_realized_equity", 0.70)

    pressure_override = (
        raise_war
        and risk_fraction >= config.get("raise_war_min_stackoff_fraction", 0.68)
    ) or facing_all_in
    self_stackoff_board_override = (
        self_stackoff
        and config.get("self_stackoff_ignores_board_score", False)
    )
    if board_score < min_board_score and not pressure_override and not self_stackoff_board_override:
        return None

    belief = extract_public_belief_state(state)
    realized = postflop_realized_equity(equity, risk, stack, opponents, board_score >= 0.8, belief)
    realized -= config.get("base_stackoff_discount", 0.035)
    realized -= board_score * config.get("board_score_discount", 0.018)
    realized -= risk_fraction * config.get("risk_fraction_discount", 0.035)
    if pressure_targeted or raise_war:
        realized -= config.get("pressure_discount", 0.025)
    if hand_type == "Pair":
        realized -= config.get("pair_discount", 0.045)
        if marginal_pair:
            realized -= config.get("marginal_pair_discount", 0.045)
        elif pair_quality == "overpair":
            realized -= config.get("overpair_discount", 0.025)
    else:
        realized -= config.get("two_pair_discount", 0.030)
    realized = clamp(realized, 0.02, 0.98)

    if realized >= min_realized:
        return None

    if proposed_raise and owed > 0:
        call_risk_fraction = owed / max(stack, 1)
        passive_equity = postflop_realized_equity(equity, owed, stack, opponents, board_score >= 0.8, belief, passive=True)
        call_ev = postflop_showdown_ev(passive_equity, pot + owed, owed)
        can_call = (
            owed > 0
            and call_risk_fraction <= config.get("max_fallback_call_stack_fraction", 0.44)
            and call_ev >= config.get("min_fallback_call_ev", -80)
        )
        if can_call:
            return {"action": "call"}

    return {"action": "fold"} if owed > 0 else {"action": "check"}


def postflop_stackoff_risk_veto(state, equity, bet_amount, owed, pot, stack, opponents, spr):
    veto = postflop_stackoff_ev_gate(state, equity, bet_amount, owed, pot, stack, opponents, spr)
    if veto:
        return veto
    veto = postflop_wet_stackoff_veto(state, equity, bet_amount, owed, pot, stack, opponents, spr)
    if veto:
        return veto
    wet = board_is_wet(state.get("community_cards", []))
    veto = commitment_blueprint_veto(state, equity, bet_amount, owed, pot, stack, opponents, spr, wet)
    if veto:
        return veto
    return river_blueprint_veto(state, equity, bet_amount, owed, pot, stack, opponents, spr, wet)


def commitment_hand_bucket(state):
    cards = state.get("your_cards", [])
    board = state.get("community_cards", [])
    hand_type = hero_hand_type(cards, board)
    if hand_type == "Pair":
        quality = hero_pair_quality(cards, board)
        if quality in ("overpair", "top_pair"):
            if has_strong_draw(cards, board):
                return quality + "_draw"
            return quality
        return "marginal_pair"
    if hand_type == "Two Pair":
        return "two_pair"
    if hand_type in ("Trips", "Straight", "Flush", "Full House", "Quads", "Straight Flush"):
        return "strong_made"
    if has_strong_draw(cards, board):
        return "strong_draw"
    return "weak_made"


def commitment_board_bucket(board):
    score = board_stackoff_risk_score(board)
    if score >= COMMITMENT_BLUEPRINT.get("very_scary_board_score", 2.2):
        return "very_scary"
    if board_is_wet(board) or score >= COMMITMENT_BLUEPRINT.get("wet_board_score", 1.0):
        return "wet"
    return "dry"


def commitment_spr_bucket(spr):
    if spr <= COMMITMENT_BLUEPRINT.get("low_spr", 1.15):
        return "low_spr"
    if spr <= COMMITMENT_BLUEPRINT.get("mid_spr", 2.5):
        return "mid_spr"
    return "high_spr"


def commitment_equity_bucket(equity):
    if equity >= COMMITMENT_BLUEPRINT.get("elite_equity", 0.84):
        return "elite"
    if equity >= COMMITMENT_BLUEPRINT.get("strong_equity", 0.74):
        return "strong"
    if equity >= COMMITMENT_BLUEPRINT.get("medium_equity", 0.64):
        return "medium"
    return "thin"


def commitment_action_bucket(state, bet_amount, owed, pot, stack):
    pressure = owed / max(pot + owed, 1)
    bb = big_blind_amount(state)
    risk = effective_stack_risk(state, bet_amount, owed, stack)
    if owed > 0:
        pressure_action = current_pressure_aggression(state)
        if (
            risk >= stack * COMMITMENT_BLUEPRINT.get("facing_stack_fraction", 0.72)
            or pressure >= COMMITMENT_BLUEPRINT.get("facing_stack_pressure", 0.38)
            or (pressure_action is not None and pressure_action.get("action") == "all_in")
        ):
            return "facing_stack_bet"
        if pressure >= COMMITMENT_BLUEPRINT.get("facing_large_pressure", 0.30):
            return "facing_large_bet"
        return "facing_bet"

    leaves_dust = stack - risk <= bb * COMMITMENT_BLUEPRINT.get("dust_bb", 2.0)
    if risk >= stack * COMMITMENT_BLUEPRINT.get("self_stack_fraction", 0.72) or leaves_dust:
        return "self_stackoff"
    if risk >= pot * COMMITMENT_BLUEPRINT.get("self_large_bet_pot_fraction", 0.68):
        return "self_large_bet"
    return "self_bet"


def commitment_lookup_keys(street, action_bucket, hand_bucket, board_bucket, spr_bucket, equity_bucket):
    return [
        street + "|" + action_bucket + "|" + hand_bucket + "|" + board_bucket + "|" + spr_bucket + "|" + equity_bucket,
        street + "|" + action_bucket + "|" + hand_bucket + "|" + board_bucket + "|" + spr_bucket + "|*",
        street + "|" + action_bucket + "|" + hand_bucket + "|" + board_bucket + "|*|*",
        street + "|" + action_bucket + "|" + hand_bucket + "|*|" + spr_bucket + "|" + equity_bucket,
        street + "|" + action_bucket + "|" + hand_bucket + "|*|" + spr_bucket + "|*",
        street + "|" + action_bucket + "|" + hand_bucket + "|*|*|*",
        street + "|" + action_bucket + "|*|" + board_bucket + "|" + spr_bucket + "|" + equity_bucket,
        street + "|" + action_bucket + "|*|" + board_bucket + "|" + spr_bucket + "|*",
        street + "|" + action_bucket + "|*|" + board_bucket + "|*|*",
        street + "|" + action_bucket + "|*|*|" + spr_bucket + "|*",
        street + "|" + action_bucket + "|*|*|*|*",
        "*|" + action_bucket + "|" + hand_bucket + "|" + board_bucket + "|" + spr_bucket + "|" + equity_bucket,
        "*|" + action_bucket + "|" + hand_bucket + "|" + board_bucket + "|" + spr_bucket + "|*",
        "*|" + action_bucket + "|" + hand_bucket + "|" + board_bucket + "|*|*",
        "*|" + action_bucket + "|" + hand_bucket + "|*|" + spr_bucket + "|*",
        "*|" + action_bucket + "|" + hand_bucket + "|*|*|*",
    ]


def commitment_blueprint_veto(state, equity, bet_amount, owed, pot, stack, opponents, spr, wet):
    if not COMMITMENT_BLUEPRINT.get("enabled", False):
        return None
    street = state.get("street")
    if street not in set(COMMITMENT_BLUEPRINT.get("streets", ["flop", "turn"])):
        return None
    if len(state.get("players", [])) < COMMITMENT_BLUEPRINT.get("min_table_size", 5):
        return None
    if equity >= COMMITMENT_BLUEPRINT.get("never_override_equity", 0.88):
        return None

    risk = effective_stack_risk(state, bet_amount, owed, stack)
    if stack <= 0 or risk <= 0:
        return None
    risk_fraction = risk / max(stack, 1)
    pressure = owed / max(pot + owed, 1)
    if risk_fraction < COMMITMENT_BLUEPRINT.get("min_risk_fraction", 0.58):
        return None
    if spr > COMMITMENT_BLUEPRINT.get("max_spr", 3.0):
        return None

    hand_bucket = commitment_hand_bucket(state)
    if hand_bucket in set(COMMITMENT_BLUEPRINT.get("protected_hand_buckets", ["strong_made", "strong_draw"])):
        return None
    board_bucket = commitment_board_bucket(state.get("community_cards", []))
    spr_bucket = commitment_spr_bucket(spr)
    equity_bucket = commitment_equity_bucket(equity)
    action_bucket = commitment_action_bucket(state, bet_amount, owed, pot, stack)

    table = COMMITMENT_BLUEPRINT.get("lookup", {})
    entry = None
    for key in commitment_lookup_keys(street, action_bucket, hand_bucket, board_bucket, spr_bucket, equity_bucket):
        if key in table:
            entry = table[key]
            break
    if not entry:
        return None

    if risk_fraction < entry.get("min_risk_fraction", COMMITMENT_BLUEPRINT.get("min_risk_fraction", 0.58)):
        return None
    if pressure < entry.get("min_pressure", 0.0):
        return None
    if opponents < entry.get("min_opponents", 1):
        return None
    if equity > entry.get("max_equity", COMMITMENT_BLUEPRINT.get("never_override_equity", 0.88)):
        return None
    if spr > entry.get("max_spr", COMMITMENT_BLUEPRINT.get("max_spr", 3.0)):
        return None
    if wet and entry.get("exclude_wet", False):
        return None

    action = entry.get("action")
    if action == "fold" and owed > 0:
        return {"action": "fold"}
    if action == "call" and owed > 0:
        return {"action": "call"}
    if action == "check" and (owed <= 0 or state.get("can_check")):
        return {"action": "check"}
    return None


def river_hand_bucket(state):
    cards = state.get("your_cards", [])
    board = state.get("community_cards", [])
    hand_type = hero_hand_type(cards, board)
    if hand_type == "Pair":
        quality = hero_pair_quality(cards, board)
        if quality in ("overpair", "top_pair"):
            return quality
        return "marginal_pair"
    if hand_type == "Two Pair":
        return "two_pair"
    if hand_type in ("Trips", "Straight", "Flush", "Full House", "Quads", "Straight Flush"):
        return "strong_made"
    return "weak_made"


def river_board_bucket(board):
    score = board_stackoff_risk_score(board)
    if score >= RIVER_BLUEPRINT.get("very_scary_board_score", 2.4):
        return "very_scary"
    if board_is_wet(board) or score >= RIVER_BLUEPRINT.get("wet_board_score", 1.0):
        return "wet"
    return "dry"


def river_equity_bucket(equity):
    if equity >= RIVER_BLUEPRINT.get("elite_equity", 0.88):
        return "elite"
    if equity >= RIVER_BLUEPRINT.get("strong_equity", 0.78):
        return "strong"
    if equity >= RIVER_BLUEPRINT.get("medium_equity", 0.66):
        return "medium"
    return "thin"


def effective_stack_risk(state, bet_amount, owed, stack):
    if owed > 0:
        return max(int(owed or 0), min(int(stack), int(bet_amount or 0)))

    invested = int(state.get("your_bet_this_street", 0) or 0)
    min_raise_to = int(state.get("min_raise_to", 0) or 0)
    max_total = stack + invested
    target_total = max(min_raise_to, invested + int(bet_amount or 0))
    if target_total >= max_total:
        return stack
    return max(0, min(stack, target_total - invested))


def effective_river_risk(state, bet_amount, owed, stack):
    return effective_stack_risk(state, bet_amount, owed, stack)


def river_action_bucket(state, bet_amount, owed, pot, stack):
    pressure = owed / max(pot + owed, 1)
    bb = big_blind_amount(state)
    risk = effective_river_risk(state, bet_amount, owed, stack)
    if owed > 0:
        pressure_action = current_pressure_aggression(state)
        if (
            owed >= stack * RIVER_BLUEPRINT.get("facing_stack_fraction", 0.82)
            or pressure >= RIVER_BLUEPRINT.get("facing_stack_pressure", 0.42)
            or (pressure_action is not None and pressure_action.get("action") == "all_in")
        ):
            return "facing_stack_bet"
        if pressure >= RIVER_BLUEPRINT.get("facing_large_pressure", 0.30):
            return "facing_large_bet"
        return "facing_bet"

    leaves_dust = stack - risk <= bb * RIVER_BLUEPRINT.get("dust_bb", 2.0)
    if risk >= stack * RIVER_BLUEPRINT.get("self_stack_fraction", 0.82) or leaves_dust:
        return "self_stackoff"
    if risk >= pot * RIVER_BLUEPRINT.get("self_large_bet_pot_fraction", 0.70):
        return "self_large_bet"
    return "self_bet"


def river_lookup_keys(action_bucket, hand_bucket, board_bucket, equity_bucket):
    return [
        action_bucket + "|" + hand_bucket + "|" + board_bucket + "|" + equity_bucket,
        action_bucket + "|" + hand_bucket + "|" + board_bucket + "|*",
        action_bucket + "|" + hand_bucket + "|*|" + equity_bucket,
        action_bucket + "|" + hand_bucket + "|*|*",
        action_bucket + "|*|" + board_bucket + "|" + equity_bucket,
        action_bucket + "|*|" + board_bucket + "|*",
        action_bucket + "|*|*|" + equity_bucket,
        action_bucket + "|*|*|*",
    ]


def river_blueprint_veto(state, equity, bet_amount, owed, pot, stack, opponents, spr, wet):
    if not RIVER_BLUEPRINT.get("enabled", False):
        return None
    if state.get("street") != "river":
        return None
    if len(state.get("players", [])) < RIVER_BLUEPRINT.get("min_table_size", 5):
        return None
    if equity >= RIVER_BLUEPRINT.get("never_override_equity", 0.90):
        return None

    hand_bucket = river_hand_bucket(state)
    if hand_bucket in set(RIVER_BLUEPRINT.get("protected_hand_buckets", ["strong_made"])):
        return None
    board_bucket = river_board_bucket(state.get("community_cards", []))
    equity_bucket = river_equity_bucket(equity)
    action_bucket = river_action_bucket(state, bet_amount, owed, pot, stack)
    risk = effective_river_risk(state, bet_amount, owed, stack)
    risk_fraction = risk / max(stack, 1)
    pressure = owed / max(pot + owed, 1)

    table = RIVER_BLUEPRINT.get("lookup", {})
    entry = None
    for key in river_lookup_keys(action_bucket, hand_bucket, board_bucket, equity_bucket):
        if key in table:
            entry = table[key]
            break
    if not entry:
        return None

    if risk_fraction < entry.get("min_risk_fraction", RIVER_BLUEPRINT.get("min_risk_fraction", 0.72)):
        return None
    if pressure < entry.get("min_pressure", 0.0):
        return None
    if opponents < entry.get("min_opponents", 1):
        return None
    if equity > entry.get("max_equity", RIVER_BLUEPRINT.get("never_override_equity", 0.90)):
        return None
    if spr > entry.get("max_spr", RIVER_BLUEPRINT.get("max_spr", 4.0)):
        return None
    if wet and entry.get("exclude_wet", False):
        return None

    action = entry.get("action")
    if action == "fold" and owed > 0:
        return {"action": "fold"}
    if action == "call" and owed > 0:
        return {"action": "call"}
    if action == "check" and (owed <= 0 or state.get("can_check")):
        return {"action": "check"}
    return None


def public_thin_stackoff_risk(state, equity, spr, opponents, wet):
    table_size = len(state.get("players", []))
    if table_size < 5 or equity >= 0.74 or spr > 1.2:
        return False

    belief = extract_public_belief_state(state)
    range_narrowing = belief.get("pbs_range_narrowing", 0.0)
    field_aggression = belief.get("pbs_field_aggression", 0.0)
    recent_raise_depth = belief.get("pbs_recent_raise_depth_s", 0.0)
    pot_to_stack = belief.get("pbs_pot_to_stack", 0.0)
    hero_commitment = belief.get("pbs_hero_commitment", 0.0)
    all_in_seen = belief.get("pbs_all_in_seen", 0.0) >= 0.5
    big_stack_pressure = belief.get("pbs_big_stack_pressure", 0.0) >= 0.5

    return (
        all_in_seen
        or hero_commitment >= 0.35
        or pot_to_stack >= 0.45
        or (opponents >= 2 and range_narrowing >= 0.28)
        or (field_aggression >= 0.22 and recent_raise_depth >= 0.25)
        or (big_stack_pressure and wet)
    )


def postflop_fold_probability(state, bet_amount, pot, opponents, wet, belief):
    call_rate, raise_rate, fold_rate, history_count = opponent_tendencies(state)
    bet_ratio = bet_amount / max(pot + bet_amount, 1)
    field_looseness = belief.get("pbs_field_looseness", 0.35)
    range_narrowing = belief.get("pbs_range_narrowing", 0.0)
    recent_raises = belief.get("pbs_recent_raise_depth_s", 0.0)

    probability = (
        0.18
        + 0.34 * fold_rate
        + 0.16 * bet_ratio
        + 0.08 * range_narrowing
        - 0.26 * call_rate
        - 0.18 * raise_rate
        - 0.18 * max(0, opponents - 1)
        - 0.10 * field_looseness
        - 0.08 * recent_raises
        - (0.08 if wet else 0.0)
    )
    if history_count < 25:
        probability = 0.65 * probability + 0.35 * (0.16 - 0.12 * max(0, opponents - 1))
    return clamp(probability, 0.02, 0.68 if opponents == 1 else 0.36)


def postflop_showdown_ev(equity, pot_after_call, cost):
    return equity * pot_after_call - (1.0 - equity) * cost


def postflop_realized_equity(equity, bet_amount, stack, opponents, wet, belief, passive=False):
    discount = 0.0
    discount += max(0, opponents - 1) * POSTFLOP_EV.get("multiway_equity_discount", 0.045)
    discount += belief.get("pbs_range_narrowing", 0.0) * POSTFLOP_EV.get("range_narrowing_discount", 0.10)
    discount += belief.get("pbs_recent_raise_depth_s", 0.0) * POSTFLOP_EV.get("recent_raise_discount", 0.05)
    if wet:
        discount += POSTFLOP_EV.get("wet_equity_discount", 0.025)
    if bet_amount >= stack * 0.75:
        discount += POSTFLOP_EV.get("stackoff_equity_discount", 0.04)
    if passive:
        discount *= POSTFLOP_EV.get("passive_discount_ratio", 0.45)
    return clamp(equity - discount, 0.02, 0.98)


def postflop_bet_ev(state, equity, bet_amount, pot, opponents, wet, belief):
    fold_probability = postflop_fold_probability(state, bet_amount, pot, opponents, wet, belief)
    realized_equity = postflop_realized_equity(
        equity, bet_amount, int(state.get("your_stack", 0) or 0), opponents, wet, belief
    )
    called_ev = postflop_showdown_ev(realized_equity, pot + bet_amount, bet_amount)
    return fold_probability * pot + (1.0 - fold_probability) * called_ev


def postflop_ev_veto(state, equity, bet_amount, owed, pot, stack, opponents, spr, wet):
    if not POSTFLOP_EV.get("enabled", False):
        return None
    if equity >= POSTFLOP_EV.get("never_veto_equity", 0.82):
        return None

    belief = extract_public_belief_state(state)
    bet_amount = max(0, min(int(bet_amount), int(stack + owed)))
    if bet_amount <= 0:
        return None

    bet_ev = postflop_bet_ev(state, equity, bet_amount, pot, opponents, wet, belief)
    passive_equity = postflop_realized_equity(equity, 0, stack, opponents, wet, belief, passive=True)
    if owed > 0:
        call_ev = postflop_showdown_ev(passive_equity, pot + owed, owed)
        fallback = {"action": "call"} if call_ev >= 0 else {"action": "fold"}
        passive_ev = call_ev
    else:
        fallback = {"action": "check"}
        passive_ev = postflop_showdown_ev(passive_equity, pot, 0)

    multiway_tax = max(0, opponents - 1) * POSTFLOP_EV.get("multiway_tax", 140)
    wet_tax = POSTFLOP_EV.get("wet_board_tax", 90) if wet else 0
    low_spr_tax = POSTFLOP_EV.get("low_spr_tax", 120) if spr <= 1.5 else 0
    required_edge = POSTFLOP_EV.get("min_bet_edge", 180) + multiway_tax + wet_tax + low_spr_tax

    if bet_ev + required_edge < passive_ev:
        return fallback
    return None


def postflop_call_veto(state, equity, owed, pot, stack, opponents, wet):
    if not POSTFLOP_EV.get("enabled", False):
        return None
    if owed <= 0 or equity >= POSTFLOP_EV.get("never_veto_call_equity", 0.74):
        return None

    belief = extract_public_belief_state(state)
    realized_equity = postflop_realized_equity(equity, 0, stack, opponents, wet, belief, passive=True)
    call_ev = postflop_showdown_ev(realized_equity, pot + owed, owed)
    call_edge = POSTFLOP_EV.get("min_call_edge", 90)
    pressure = owed / max(pot + owed, 1)
    extra_edge = max(0, opponents - 1) * POSTFLOP_EV.get("multiway_call_tax", 80)
    if pressure >= 0.32:
        extra_edge += POSTFLOP_EV.get("large_call_tax", 120)
    if call_ev + call_edge + extra_edge < 0:
        return {"action": "fold"}
    return None


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
    risk_ev = signals.get("risk_adjusted_chip_ev", chip_ev)
    survival = signals.get("survival", 1.0)
    stack_preservation = signals.get("stack_preservation", 1.0)
    required = owed / max(pot + owed, 1)
    spr = stack / max(pot, 1)
    wet = board_is_wet(state.get("community_cards", []))

    if stack <= 0:
        return safe_check_or_fold(state)

    if owed == 0 or state.get("can_check"):
        if equity >= 0.74 or (equity >= 0.66 and spr <= 1.2):
            if spr <= 1.1:
                if public_thin_stackoff_risk(state, equity, spr, opponents, wet):
                    return {"action": "check"}
                wet_veto = postflop_stackoff_risk_veto(state, equity, stack, owed, pot, stack, opponents, spr)
                if wet_veto:
                    return wet_veto
                veto = postflop_ev_veto(state, equity, stack, owed, pot, stack, opponents, spr, wet)
                if veto:
                    return veto
                return {"action": "all_in"}
            fraction = 0.72 if wet else 0.58
            bet_amount = int(pot * fraction)
            wet_veto = postflop_stackoff_risk_veto(state, equity, bet_amount, owed, pot, stack, opponents, spr)
            if wet_veto:
                return wet_veto
            veto = postflop_ev_veto(state, equity, bet_amount, owed, pot, stack, opponents, spr, wet)
            if veto:
                return veto
            return raise_to(state, invested + bet_amount)
        if opponents <= 2 and pos >= 0.55 and equity >= 0.58:
            bet_amount = int(pot * 0.62)
            wet_veto = postflop_stackoff_risk_veto(state, equity, bet_amount, owed, pot, stack, opponents, spr)
            if wet_veto:
                return wet_veto
            veto = postflop_ev_veto(state, equity, bet_amount, owed, pot, stack, opponents, spr, wet)
            if veto:
                return veto
            return raise_to(state, invested + bet_amount)
        if opponents <= 2 and pos >= 0.50 and equity >= 0.45:
            if fold_pressure >= 0.72 and danger <= 0.35 and risk_ev >= 0.10:
                bet_amount = int(pot * 0.55)
                wet_veto = postflop_stackoff_risk_veto(state, equity, bet_amount, owed, pot, stack, opponents, spr)
                if wet_veto:
                    return wet_veto
                veto = postflop_ev_veto(state, equity, bet_amount, owed, pot, stack, opponents, spr, wet)
                if veto:
                    return veto
                return raise_to(state, invested + bet_amount)
        return {"action": "check"}

    margin = 0.035 + max(0, opponents - 1) * 0.035
    if state.get("street") == "river":
        margin += 0.035
    if signals:
        margin += 0.025 * danger
        margin += max(0.0, 0.65 - survival) * 0.04
        margin += max(0.0, 0.45 - stack_preservation) * 0.03
        if risk_ev > 0:
            margin -= min(0.015, risk_ev * 0.015)

    if equity >= 0.76 and spr <= 1.7:
        wet_veto = postflop_stackoff_risk_veto(state, equity, stack, owed, pot, stack, opponents, spr)
        if wet_veto:
            return wet_veto
        veto = postflop_ev_veto(state, equity, stack, owed, pot, stack, opponents, spr, wet)
        if veto:
            return veto
        return {"action": "all_in"}

    if equity >= max(0.68, required + 0.20):
        fraction = 0.78 if wet else 0.62
        bet_amount = owed + int((pot + owed) * fraction)
        wet_veto = postflop_stackoff_risk_veto(state, equity, bet_amount, owed, pot, stack, opponents, spr)
        if wet_veto:
            return wet_veto
        veto = postflop_ev_veto(state, equity, bet_amount, owed, pot, stack, opponents, spr, wet)
        if veto:
            return veto
        return raise_to(state, invested + bet_amount)

    if equity >= required + margin:
        wet_veto = postflop_stackoff_risk_veto(state, equity, owed, owed, pot, stack, opponents, spr)
        if wet_veto:
            return wet_veto
        veto = postflop_call_veto(state, equity, owed, pot, stack, opponents, wet)
        if veto:
            return veto
        return {"action": "call"}

    if signals and risk_ev < -0.55 and equity < required + 0.10:
        return {"action": "fold"}

    if owed <= max(100, pot * 0.08) and equity >= required - 0.025:
        wet_veto = postflop_stackoff_risk_veto(state, equity, owed, owed, pot, stack, opponents, spr)
        if wet_veto:
            return wet_veto
        veto = postflop_call_veto(state, equity, owed, pot, stack, opponents, wet)
        if veto:
            return veto
        return {"action": "call"}

    return {"action": "fold"}
