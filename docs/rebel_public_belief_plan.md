# Public State and Range Modeling Plan

This project now treats ReBeL as a research direction, not as a label for the
runtime feature set. The current runtime uses public-state heuristics and
coarse opponent-range inference; it does not perform ReBeL value iteration or
subgame solving.

The practical near-term goal is to make equity and risk decisions conditional
on inferred opponent ranges, then reserve learned values or search for a later
offline CFR/self-play path.

## Current Foundation

- The bot exposes `extract_public_belief_state(state)`.
- Public belief features are included in the model feature vector.
- The trainer records public belief states and exports belief summaries.
- The trainer can write and reload JSONL replay buffers.
- Runtime learned behavior remains disabled until it clears benchmark gates.
- `estimate_equity(state)` supports range-conditioned multiway postflop Monte
  Carlo sampling from `model.json` range buckets.

## Milestone 1: Public State Quality

Improve and validate public-state features:

- public board texture
- stack-at-risk and commitment
- action depth and recent raise pressure
- field looseness and aggression
- inferred range narrowing
- heads-up vs multiway context

Success criterion: belief summaries are stable across repeated fixed-seed
training runs and correlate with known failure cases, especially maniac tables.

## Milestone 2: Range-Conditioned Equity

Replace equity versus random holdings with equity versus inferred ranges:

- classify opponent archetypes from VPIP/PFR/aggression proxies
- narrow ranges from current hand action pressure
- filter postflop raise/call/stackoff ranges by board interaction
- sample opponent hole cards from range buckets during Monte Carlo
- keep heads-up equity on the stable uniform path unless benchmark gates say otherwise

Success criterion: improve six-max average and/or bust rate without degrading
heads-up regression.

## Milestone 3: Public Belief Replay Buffer

Store training samples as public-state records:

- public belief vector
- private hand/equity vector
- legal action set
- chosen action
- opponent responses
- terminal stack deltas

Implementation commands:

```bash
/tmp/fullhouse-py310/bin/python tools/train_codex_holdem.py \
  --hands 12000 \
  --matches 60 \
  --seed 6161 \
  --replay-output replays/public_belief_6161.jsonl \
  --generate-only

/tmp/fullhouse-py310/bin/python tools/train_codex_holdem.py \
  --replay-input replays/public_belief_6161.jsonl \
  --output bots/codex_holdem/data/model.json
```

Success criterion: we can train and evaluate models from saved replay data
without rerunning full matches every time.

## Milestone 4: Collapse Learned Runtime Model

The current multi-head supervised model remains disabled. The next learned
runtime candidate should be one of:

- `V(s)` plus `pi(a|s)` trained from self-play returns
- a regret/average-strategy network trained from MCCFR or Deep CFR samples

Success criterion: learned runtime behavior beats the rule-based baseline under
fixed-seed gates before it is enabled by default.

## Milestone 5: Offline CFR Strategy Tables

Tournament constraints favor offline strategy computation over runtime search.
The clean path is:

- external-sampling MCCFR or Deep CFR over abstracted states
- 169 preflop classes and compact postflop buckets
- average strategy exported into `data/model.json`
- runtime uses O(1) lookup plus existing guards

Success criterion: learned runtime behavior beats the rule-based baseline under
fixed-seed gates before it is enabled by default.
