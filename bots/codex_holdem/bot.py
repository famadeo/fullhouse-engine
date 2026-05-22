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


def current_hand_actions_for_seat(state, seat):
    return [a for a in state.get("action_log", []) if a.get("seat") == seat]


def range_bucket_hands(bucket):
    buckets = RANGE_EQUITY.get("buckets", {})
    hands = buckets.get(bucket, [])
    return set(hands)


def inferred_range_bucket(state, seat):
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


def inferred_opponent_ranges(state, known, board):
    ranges = []
    for opponent in active_opponents(state):
        seat = opponent.get("seat")
        bucket = inferred_range_bucket(state, seat)
        candidates = candidate_hands_for_bucket(bucket, known, board)
        if not candidates and bucket not in ("unknown", "loose_passive"):
            candidates = candidate_hands_for_bucket("unknown", known, board)
        ranges.append(candidates)
    return ranges


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
                    first, second = rng.choice(available)
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
                veto = postflop_ev_veto(state, equity, stack, owed, pot, stack, opponents, spr, wet)
                if veto:
                    return veto
                return {"action": "all_in"}
            fraction = 0.72 if wet else 0.58
            bet_amount = int(pot * fraction)
            veto = postflop_ev_veto(state, equity, bet_amount, owed, pot, stack, opponents, spr, wet)
            if veto:
                return veto
            return raise_to(state, invested + bet_amount)
        if opponents <= 2 and pos >= 0.55 and equity >= 0.58:
            bet_amount = int(pot * 0.62)
            veto = postflop_ev_veto(state, equity, bet_amount, owed, pot, stack, opponents, spr, wet)
            if veto:
                return veto
            return raise_to(state, invested + bet_amount)
        if opponents <= 2 and pos >= 0.50 and equity >= 0.45:
            if fold_pressure >= 0.72 and danger <= 0.35 and risk_ev >= 0.10:
                bet_amount = int(pot * 0.55)
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
        veto = postflop_ev_veto(state, equity, stack, owed, pot, stack, opponents, spr, wet)
        if veto:
            return veto
        return {"action": "all_in"}

    if equity >= max(0.68, required + 0.20):
        fraction = 0.78 if wet else 0.62
        bet_amount = owed + int((pot + owed) * fraction)
        veto = postflop_ev_veto(state, equity, bet_amount, owed, pot, stack, opponents, spr, wet)
        if veto:
            return veto
        return raise_to(state, invested + bet_amount)

    if equity >= required + margin:
        veto = postflop_call_veto(state, equity, owed, pot, stack, opponents, wet)
        if veto:
            return veto
        return {"action": "call"}

    if signals and risk_ev < -0.55 and equity < required + 0.10:
        return {"action": "fold"}

    if owed <= max(100, pot * 0.08) and equity >= required - 0.025:
        veto = postflop_call_veto(state, equity, owed, pot, stack, opponents, wet)
        if veto:
            return veto
        return {"action": "call"}

    return {"action": "fold"}
