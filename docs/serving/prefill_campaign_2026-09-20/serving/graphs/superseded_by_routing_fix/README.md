# Superseded on 2026-09-21: accepted under a routing check that was wrong for CUDA graphs

These 42 SCRATCH_FULL cases passed, and their numbers are not in question: their prefill
routing was verified token-for-token like every other case. They are set aside only so that
every graphs-mode case in `cases/` was accepted under the same rule.

The bug: `check_routing` required the decode counter to grow inside the measured window. With
CUDA graphs a step of nothing but single-token requests is replayed from a captured graph and
never reaches Python, so that counter cannot grow. It rejected all 42 SCRATCH_DECODE cases
(see `../failures.jsonl`) and left decode unverified for SCRATCH_FULL, which passed on its
prefill accounting alone. The fix checks the decode counter before the window, which covers
startup, graph capture and the warm-up run, and records that snapshot in each case file.

Raw client output for these attempts is untouched under `../sessions/`. Eager mode used the
in-window check throughout and is unaffected (126/126, no rejections).
