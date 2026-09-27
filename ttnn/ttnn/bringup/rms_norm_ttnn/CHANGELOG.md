# rms_norm_ttnn (ttnn.bringup.rms_norm)

ai-generated perf optimized version

## Model-case tests (mimo_v2_6_d_p, O.1)
- What: added `tests/cases.py`, `tests/reference.py`, `tests/test_rms_norm_ttnn.py` (the bring-up model-case suite;
  the op's own suite stays in `tests/unit/`). First case: mimo_v2_6_d_p sig 8d29c1ff0f, [1,1,5120,4096] bf16 TILE,
  bf16 weight, eps 1e-6, HiFi4 + fp32 dest acc, 1x4 mesh FABRIC_2D.
- Why: the O.1 gate needs a random-input case for every ttnn.bringup call a model makes. No op code changed.
