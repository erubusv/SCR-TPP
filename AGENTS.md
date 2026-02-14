# Autonomous Research Lab Rules (Self-improving)

## Roles and mandatory loop
Each round MUST follow:
1) Scientist-Explore: generate novel hypotheses / directions
2) Scientist-Exploit: refine best current direction
3) Critic: challenge both; pick minimal decisive experiments
4) Engineer: implement + run smallest experiments first
5) Forensics: diagnose failures; design re-check experiments
6) Meta-Critic: evaluate whether the round was useful; update strategy

## Outputs (MUST)
Write artifacts under autonomous_codex/:
- plans/: role outputs + selected plan
- runs/: commands + logs
- summaries/: round summaries + scoreboard + strategy

## Safety & reproducibility
- Stay inside this repository.
- Prefer small/smoke tests before full runs.
- Avoid destructive commands (rm -rf, mkfs, dd, etc.).
- No arbitrary network shell commands (curl/wget). If web search is needed, use Codex built-in search option.

## Budget
- Each round: max 12 shell commands total.
- Each round: max 6 experiments.
- If stuck, stop and write a diagnosis + what is needed next.
