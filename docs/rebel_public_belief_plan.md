# ReBeL-Inspired Public Belief State Plan

This project should treat ReBeL as a research direction, not a single feature.
The practical goal is to make the bot reason from public game state plus inferred
range pressure, then use search or learned values only in high-leverage spots.

## Current Foundation

- The bot exposes `extract_public_belief_state(state)`.
- Public belief features are included in the model feature vector.
- The trainer records public belief states and exports belief summaries.
- Runtime learned behavior remains disabled until it clears benchmark gates.

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

## Milestone 2: Belief-Conditioned Value Heads

Train value heads that separate private-card value from public-state danger:

- showdown equity
- fold pressure
- danger
- survival
- stack preservation
- risk-adjusted chip EV
- action EVs for fold, call, small bet, large bet, jam

Success criterion: enabling a small high-risk override reduces bust rate without
dropping six-max average below the current baseline.

## Milestone 3: Public Belief Replay Buffer

Store training samples as public-state records:

- public belief vector
- private hand/equity vector
- legal action set
- chosen action
- opponent responses
- terminal stack deltas

Success criterion: we can train and evaluate models from saved replay data
without rerunning full matches every time.

## Milestone 4: Depth-Limited Search

For large pots and all-in decisions, run a small public-belief search:

- bucket opponent ranges from action pressure
- enumerate candidate actions
- simulate likely opponent responses
- use learned leaf values for unresolved future streets

Success criterion: seed 11 six-max bust is improved while existing heads-up
caller performance stays positive.

## Milestone 5: ReBeL-Like Self-Play Loop

Approximate the ReBeL shape:

- self-play generates public-belief states
- model predicts values and policy over the public state
- search improves action targets
- improved targets train the next model

Success criterion: learned runtime behavior beats the rule-based baseline under
fixed-seed gates before it is enabled by default.
