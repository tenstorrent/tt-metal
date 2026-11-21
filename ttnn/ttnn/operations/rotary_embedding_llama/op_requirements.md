# Operation Requirements: rotary_embedding_llama

## Definition
- **Formula**: `out = x * cos + (x @ T) * sin`, `T` (32×32) applied to every run of 32 channels of a head row; `cos` / `sin` (one row a user) broadcast over that user's head rows. Decode mode: `x [1, users, heads, head_dim]`, `cos`, `sin [1, users, 1, head_dim]`, `T [1, 1, 32, 32]`.
- **PyTorch Reference**:
  ```python
  def reference(x, cos, sin, trans_mat):
      rotated = (x.reshape(*x.shape[:-1], -1, 32) @ trans_mat.reshape(32, 32)).reshape(x.shape)
      return x * cos + rotated * sin
  ```
- **Import Path**: `from ttnn.operations.rotary_embedding_llama import rotary_embedding_llama`
- **Function Signature**:
  ```python
  rotary_embedding_llama(input_tensor, cos_cache, sin_cache, trans_mat, *,
                         is_decode_mode: bool = False,
                         compute_kernel_config: ttnn.ComputeConfigDescriptor = None) -> ttnn.Tensor
  ```
  Specification: `eval/specs/rotary_embedding_llama/spec.py`. Target: Quasar (`.claude/references/targets/quasar.md`), run on the ttsim simulator (`.claude/references/backends/ttsim.md`).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Perf gate on this backend**: ttsim has no clock (backend page, *Performance measurement*). Every perf refinement below is gated on a **structural count** read off the op by a test or probe, never on a simulator timing. The gain in time stays unconfirmed until a timed backend (`ttsim-perf`) or hardware measures it; say so in the changelog.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED placement**: memory_config=[L1], sharding=[height] (zero-copy: every tensor's shard backs a borrowed ring)
- **SUPPORTED shape-derived axes**: users=[1], alignment ∈ {tile_aligned, h_non_aligned}
- **SUPPORTED op-specific axes**: is_decode_mode=[True]
- **EXCLUSIONS**: `{dtype: float32, fp32_dest_acc_en: False}` (rule R1; inert until float32 is supported)
- **Cores**: the input shard's core set (one core per user; one core at users = 1). Thread plan `THREAD_PLAN = (1, 4, 1)`, C clamped to divide Wt: head_dim 128 → 4 NEOs, 64 → 2, 32 / 96 → 1.
- **Compute config**: `default_compute_kernel_config()` = HiFi4, `fp32_dest_acc_en=True`, `math_approx_mode=False`; any ComputeConfigDescriptor or device config accepted (R7)
- **Golden baseline**: 18 / 18 supported tests passing (12 cells + 6 numeric regression cases), 36 xfail_expected, 10 / 10 rejections; 0 supported_fail / xpass_drift / xfail_wrong_mode

### [ ] Refinement 1 — Numerical configurability: float32 inputs and a 16-bit accumulator

**Goal**: add `ttnn.float32` to `SUPPORTED["dtype"]` and `False` to `SUPPORTED["fp32_dest_acc_en"]`, in one refinement. This moves 24 golden cells from xfail to passing: 12 cells of `(bfloat16, fp32_dest_acc_en=False)`, graded by the spec's `(0.999, 0.05)`, and 12 of `(float32, True)`, graded by the shared `(0.999, 0.01)`. The 12 cells of `(float32, False)` must stay refused through the `EXCLUSIONS` entry already in place (rule R1). Along the way, re-decide the page format of the intermediate ring `rotated`; the L1-ledger finding below explains why.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- This is the only generality refinement: `TARGET − SUPPORTED` is exactly these two values. Every other TARGET axis is already at its single value.
- **Ledger audit-3 finding, folded in here.** `rotated` is packed as bfloat16 while `fp32_dest_acc_en=True`. That rounds `x @ T` to 16 bits at the phase boundary. The implementer did this deliberately. Measured cost: relative RMS 0.0024 against the exact reference versus 0.0017 against a "rotated rounded to bf16" reference, so this rounding is about a third of the error. PCC is still 0.999997 (see `verification_report.md`). Retry a Float32 `rotated` under a 32-bit DEST, for example by programming `hw_startup` / the srcA format from `rotated` or by ordering the phases so the format change is legal on Quasar. If that does not work, keep the input dtype, record the mechanism in `l1_ledger.md` (it is already there) and do not block the refinement on it. Under `fp32_dest_acc_en=False` the input dtype is the right page anyway.
- **float32 borrowed rings.** The Metal 2.0 host requires an explicit `unpack_modes` entry for every Float32 DFB under `enable_32_bit_dest` (`hardware_config.cpp`; implementer breadcrumb). `_compute_hw_config` sets one only for `rotated` today. All five borrowed rings (x, cos, sin, T, output side) will be Float32 in a float32 call. Each one feeds the FPU, so each needs `UnpackToSrc`.
- `DEST_AUTO_LIMIT` doubles under a 16-bit DEST. `section_tiles` and the chain already read it, so nothing to change, but keep the `(1, 1, 1)` × head_dim 128 plan case (4 tiles in one section) in the tests.
- Spec DECISIONS: a host simulation of a dense-matrix bf16 sum predicts PCC 0.9996–0.99997 at `fp32_dest_acc_en=False`. That is close to the 0.999 floor but above it. A numeric failure there is `numerical-precision` (leave it failing and name it), never a reason for `EXCLUSIONS`.
- Keep the thread plan and the session-order test (`test_rotary_embedding_llama_plan.py`) passing at the new dtype and accumulator. Extend `test_thread_clamp` with one float32 case and one `fp32_dest_acc_en=False` case at head_dim 128 under plans `(1, 4, 1)` and `(1, 2, 1)`.

**Done when**: the 24 cells named above pass in the golden suite, the 12 `(float32, False)` cells are `xfail_expected` (refused through `EXCLUSIONS`), `verify_supported` shows 0 in every loud category, and the plan tests pass at float32 and at `fp32_dest_acc_en=False`.

### [ ] Refinement 2 — Per-call fixed cost on the traced shapes: drop the data-format reconfigs that change nothing

**Type**: perf

**Goal**: speed up the two traced calls, Qwen3-VL queries `(1, 1, 32, 128)` and keys `(1, 1, 8, 128)`, at the configuration they run (bfloat16, the default `fp32_dest_acc_en=True`, HiFi4). Both already use 4 NEOs, one tile each, with zero bytes moved. Their cost is therefore the per-block fixed cost. Each NEO issues at least 6 data-format reconfigs per block: `reconfig_data_format(trans_mat, input)` and `pack_reconfig_data_format(rotated)` before the matmul, plus the `eltwise_chain`'s per-input reconfig (`DataFormatReconfig::Enabled` by default) on `rotated`, `sin`, `x` and `cos`. When every operand has one format, none of them changes anything. Use the second lever of `compute_block_size` in `ttnn/ttnn/operations/examples/master.md`: set `DataFormatReconfig::Disabled` on operands whose format equals the previous phase's, and skip the explicit reconfigs on those calls. Decide on the host from the ring formats, so that a call whose formats do differ (Refinement 1's Float32 `rotated`, if it lands) keeps exactly the reconfigs it needs. No SUPPORTED change.

**Done when**:
- the structural count *data-format reconfigs issued per NEO per call* is pinned before and after by a test or probe. It must drop to 0 on single-format calls (all-bf16 and all-float32) and to only the real format boundaries elsewhere;
- the golden suite is green, including the numeric regression cases;
- there is no regression on the guard set: one representative per kernel path (C = 4 / 2 / 1 → head_dim 128 / 64 / 32) × dtype × `fp32_dest_acc_en`, all still passing with unchanged PCC.

The changelog states that the time gain is unconfirmed on ttsim.

**Verifier notes**: this comes after Refinement 1 because the reconfig decision depends on which ring formats Refinement 1 settles on, and a single-format check written before float32 exists would have to be rewritten. This is a T2 lever on its own and a small phase. If the implementer has room, the design's other fixed-cost lamp may share the phase: the `rotated` L1 round trip against a DEST-fused combine. The catalog (`compute_fusion`) predicts a loss for an FPU consumer, so try it only as a measured alternative on `ttsim-perf`, never adopted on a structural count alone.

### [ ] Refinement 3 — Engage more than one NEO on the 3-tile heads (head_dim 96)

**Type**: perf

**Goal**: the head_dim 96 cells, `(1, 1, 12, 96)` and `(1, 1, 32, 96)`, run on 1 of the cluster's 4 NEOs. The design's C clamp forces this: a borrowed STRIDED ring's capacity is pinned to the shard, 3 entries, and C ∈ {1, 2, 4} must divide it. Raise the structural count *compute threads at work for head_dim 96* from 1 to 3, using the design's own perf lamp (`op_design.md`, Perf lamps, second row). That means non-borrowed STRIDED rings at C = 4 for x / cos / sin, filled by a core-local L1→L1 copy (implicit sync, tagged, with the trid hand-back), with the tiles dealt round-robin so NEO 3 takes none. The output ring is either drained back into the output shard or kept borrowed if the ring API allows it. This adds a new regime (`dealt_copy`, predicate `tensor_dim_tiles % C_plan != 0`). The `resident_shard_in_place` regime stays the path wherever C divides Wt. No SUPPORTED change.

**Done when**:
- `test_each_neo_computes_its_own_tiles` (or an equivalent tag test) shows head_dim 96 tiles 0, 1 and 2 computed by NEOs 0, 1 and 2;
- the added traffic is counted in `l1_ledger.md`'s data-movement budget: core-local copies only, 0 DRAM, 0 cross-core;
- head_dim 32 / 64 / 128 still take the borrowed zero-copy path, with 0 bytes moved and the same thread counts as Phase 0;
- the golden suite is green, and the session-order test passes on the new regime, since it now issues NoC transfers.

The changelog states that the time gain is unconfirmed: the design itself flags that 3 tiles of work may not repay 9–12 local tile copies.

**Verifier notes**: this is a scheme-change on the data-movement side (a new copy-in regime), so it stands alone. It is filed last because the traced model shapes (head_dim 64 / 128) do not take this path. Before adopting it as the default for Wt = 3, a timed comparison against the borrowed C = 1 path on `ttsim-perf` is advised. That comparison is a finding to record, not a Done-when clause, because this run cannot reach that backend.
