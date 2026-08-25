# Milestone A Attention2D Readiness Work Log

## Scope

- Host-only audit; no TT hardware commands or tests.
- Read Milestone A requirements and inspect `Attention2D` implementation and tests.
- Audit functional completeness, selectors, decode/prefill cases, and KV-cache handling.
- Limit code changes to Attention2D implementation and Attention2D tests.

## Checkpoint 1: Baseline and ownership boundary

- Created a dedicated goal for the Attention2D readiness audit.
- Confirmed the worktree already contains broader Milestone A changes; those changes are treated as pre-existing and will not be modified by this audit.
- Located the implementation and host/hardware qualification tests:
  - `models/common/modules/attention/attention_2d.py`
  - `models/common/tests/modules/attention/test_attention_2d.py`
  - `models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py`
- Extracted the plan contract: fused QKV/WO lazy weights, optional QKV bias, optional head-local Q/K RMSNorm, explicit decode/prefill placement and program policy, externally bound paged KV caches, per-column batch offsets/prefix masks, injected prefetch/CCL collaborators, decode at physical batch 32, and prefill recipes through sequence length 2048.
- Hardware execution is explicitly excluded from this audit.

## Checkpoint 2: Baseline host suite and audit findings

- Baseline command: `pytest -q models/common/tests/modules/attention/test_attention_2d.py`
- Baseline result: `51 passed in 7.72s`.
- Python compilation of the implementation, host test, and hardware-test source passed.
- Functional gap: concat-32 prefill sliced K/V per user only for paged caches; contiguous cache fill reused the complete multi-row tensors for every user.
- Resource gap: Attention2D had no decode/prefill prefetch context fields and passed no resolved global-CB/subdevice arguments to its QKV or WO projections.
- Validation gap: optional Q/K norms checked only weight width, not the required head-local RMSNorm2D geometry.
- Existing selector coverage already spans the full row-mode, collective-mode, and attention-mode cross product at sequence length 128. A focused sequence-length 2048 selector case is still needed.

## Checkpoint 3: Initial fixes verified

- Added optional decode/prefill prefetch contexts, validated their mesh and mode during config resolution, and forwarded their global-CB/subdevice values to QKV and WO projections.
- Required optional Q/K norm configs to declare `RMSNorm2DGeometry.HEAD_LOCAL`.
- Fixed contiguous concat-32 cache fill to isolate each input row before writing its destination user slot.
- Added host coverage for prefetch selector rejection and propagation, explicit Q/K norm geometry, sequence-length 2048 recipe selection, and contiguous concat-32 cache isolation.
- Post-change command: `pytest -q models/common/tests/modules/attention/test_attention_2d.py`
- Post-change result: `57 passed in 8.61s`.
- `py_compile` and `git diff --check` passed for the bounded attention changes.
- Follow-up audit found paged concat-32 source-row selection used destination user IDs. The metadata allows ordered user mappings, so source rows must follow input order while page-table rows follow destination users.

## Checkpoint 4: Ordered paged-cache mapping verified

- Corrected paged concat-32 fill so source K/V rows follow the order of `user_ids`, while each page-table slice follows the corresponding destination user ID.
- Added a non-identity user-order reduction that distinguishes source-row and destination-row selection.
- Host result: `58 passed in 8.75s`.
- Python compilation, bounded `git diff --check`, and forbidden-import/API scans passed.
- Final coverage audit identified two branches that deserved explicit host evidence: contiguous decode cache operations and Q/K normalization in both decode and prefill.

## Checkpoint 5: Final host-only readiness result

- Added explicit host reductions for contiguous decode cache update/non-paged SDPA selection and optional Q/K norm execution in both decode and prefill.
- Final command: `pytest -q models/common/tests/modules/attention/test_attention_2d.py`
- Final result: `60 passed in 9.03s`.
- `python -m py_compile` passed for the implementation, host test, and hardware-test source.
- `git diff --check` passed for all bounded changes.
- `black --check models/common/modules/attention/attention_2d.py models/common/tests/modules/attention/test_attention_2d.py` passed with both files unchanged.
- A broader Black check reported that the pre-existing, unmodified `test_attention_2d_wh_galaxy.py` would be reformatted; this audit did not modify that file.
- Static scan found no `from_model_args`, legacy model imports, model-package imports, or Attention1D dependencies in `attention_2d.py`.
- No TT hardware command, fixture execution, or reset was performed.

## Final assessment

- Host-ready: static Galaxy geometry and placement validation, immutable composite prefill recipe selection, decode and prefill direct-TTNN stage ownership, decode physical batch 32, single-row and concat-32 prefill, regular and ring collective selectors, regular and prefix/chunked attention selectors, sequence lengths 128 and 2048, paged and contiguous cache branches, ordered-user cache isolation, optional QKV bias, optional head-local Q/K normalization, prefetch context propagation, borrowed cache ownership, cleanup, and exceptional-path draining.
- Remaining exit-gate evidence is hardware-only: representative Llama/Qwen PCC, KV-cache PCC, repeat invocation, and production Prefetcher2D/Galaxy CCL integration on WH `(8, 4)`. Those tests were intentionally not run in this goal.

## Changed files

- `models/common/modules/attention/attention_2d.py`
- `models/common/tests/modules/attention/test_attention_2d.py`
- `tttv2_attention2d_readiness_work_log_20260819.md`
