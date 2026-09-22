# Project Scope

The active project is D64 attention on RTX 4060 Ti (8 GB, sm_89).
Do not restart D128 optimization, new architecture experiments, or unattended
campaigns without an explicit user request. Existing D128 code is retained for
compatibility; it is not the optimization target or a headline result.

## Authoritative Files

- Prefill: `cuda/attention_forward.cu`.
- Serving decode: `cuda/attention_decode_paged.cu`, loaded by
  `integrations/vllm/scratch_vllm/loader.py`.
- Prefill+decode integration: `integrations/vllm_prefill/`.
- Current performance and reproduction entry points: root `README.md`.
- Latest kernel evidence: `docs/serving/backend_overview_2026-09-22/`.
- Historical serving evidence: `docs/serving/prefill_campaign_2026-09-20/`.

Historical reports describe their recorded source hashes, not the current
working tree. A "final" or "current" label inside a dated report is historical.
Do not attribute the older serving numbers to the newer prefill source.
Old shape sets named holdout have since been used for development.

Preserve measured raw data and the FP32 baseline in `experiments/cuda/fp32.cu`.
Do not change precision, add benchmark-shape whitelists, or replace scratch
computation with another attention implementation. Do not create AI-generated PRs.
Do not commit, push, use cloud resources, or start a long GPU run without a user request.
