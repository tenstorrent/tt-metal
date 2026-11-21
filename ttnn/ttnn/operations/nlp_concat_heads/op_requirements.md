# Operation Requirements: nlp_concat_heads

## Definition
- **Formula**: `output[b, 0, s, h * head_dim + d] = input[b, h, s, d]` — input `[B, num_heads, S, head_dim]`, output `[B, 1, S, num_heads * head_dim]`; every element moved unchanged (spec R1, graded `exact`).
- **PyTorch Reference**: `def reference(x): b, h, s, d = x.shape; return x.permute(0, 2, 1, 3).reshape(b, 1, s, h * d)`
- **Import Path**: `from ttnn.operations.nlp_concat_heads import nlp_concat_heads`
- **Function Signature**: `nlp_concat_heads(input_tensor: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = None) -> ttnn.Tensor` (fixed by the specification's stub: no `compute_kernel_config`, no `runtime_rows`)
- **Specification**: `eval/specs/nlp_concat_heads/spec.py` (governs). Target quasar on ttsim (untimed): perf gates below are **structural counts**, and every gain in time is unconfirmed until a timed backend (`ttsim-perf`) or hardware measures it.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED placement**: memory_config=[DRAM], sharding=[interleaved], output_memory=[DRAM]
- **SUPPORTED shape-derived axes**: alignment=[tile_aligned], batch=[single]
- **Regime**: `tile_relay` (head_dim % 32 == 0): reader → one implicit-sync ring (`relay`, `RING_DEPTH_TILES × max(R, W)` tiles) → writer; no compute kernel (R1 holds bit for bit).
- **Cores**: flat output-page split over the whole 8×4 grid (`split_work_to_cores(..., row_wise=True)`), contiguous range per cluster; thread plan (R, C, W) = (1, 0, 1), data-movement bound.
- **Compute config**: none (the op computes nothing; the signature has no compute config).
- **Golden baseline**: 10 / 10 supported cells pass; 206 xfail_expected; 0 supported_fail / xpass_drift / xfail_wrong_mode; rejections X1–X3 pass.

### [ ] Refinement 1 — `subtile_assemble` regime: head_dim not a multiple of 32

**Goal**: add `"w_non_aligned"` to `SUPPORTED["alignment"]` by building the design's deferred `subtile_assemble` regime natively in the kernels (op_design.md, Regimes + the `subtile_assemble` contract under Dataflow Strategy). With `D % 32 != 0` a head ends inside a tile, so an output tile is assembled from parts of up to two input tiles per head boundary: input padding columns (`D .. 32·ceil(D/32)` of each head) must never reach the output, and output columns beyond `H·D` in the last output tile are zero-filled. Golden cells moved to passing: `(1,4,32,50)`, `(1,3,64,17)`, `(1,8,17,50)` at bfloat16 / DRAM → DRAM / single (the last also has S non-aligned: its padding rows travel within the assembled tiles).

**Implementation skill**: /memory-layouts

**Verifier notes**:
- **Scheme-change, stands alone**: a new pipeline stage, not a knob. Topology per the design: reader → `cb_head_tiles` (implicit sync, input tiles of the block's columns in (h, c) order) → **assembler** DM kernel (explicit sync on both rings, listed in its `disable_dfb_implicit_sync_for`, every NoC transfer of it tagged `{.trid = 0}` if it issues any; RISC-V stores of face-row element runs, data cache flushed before the writer's NoC read — `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp`) → `cb_output_tiles` → writer (implicit sync, the current writer unchanged). `/memory-layouts` covers the non-aligned pad/mask rules and the face layout; the cross-stage structure is the design's, not the skill's.
- Ordered first (hardest first): Refinement 2 is a `validate()` widening that then extends both regimes at once. There is no dependency either way, so build this on the small Phase 0 surface. Write the assembler with **element size from the page** (`page_bytes / 1024`), not a bfloat16 literal, so Refinement 2's float32 is a gate change, not a kernel change. Keep the walk per image (b in the counters) so `batched` also extends for free.
- Hold it to the performance bar from day 1: the same flat split over the whole grid (`B · ceil(S/32) · ceil(H·D/32)` output tiles), cut into blocks at token-row boundaries (a block never crosses a token row); each input tile read **at most once per block it feeds** (a sliding window inside the block — never once per output tile it contributes to); both NoC halves on implicit-sync streaming with no per-tile barrier. Expose the assembler's thread count as a named knob beside `THREAD_PLAN` (default 1, R + A + W ≤ 6, one source of truth). Refinement 4 turns it.
- Regime selection on the shape only (`input.shape[-1] % 32`), never on the grid. The program builder's current guard (`head_geometry` raises `NotImplementedError` on D % 32 != 0) is the switch point. Add the regime-pinned test pair the design requires (one `w_non_aligned` and one aligned cell, asserting which `ProgramSpec.name` was dispatched), and extend the one-device-session test with a w_non_aligned call: the explicit-sync assembler must not leave a sticky tag for `ttnn.to_layout`.
- Update `l1_ledger.md` with the two new rings (axis accounting: the block's window over the token row) and the traffic budget: the duplicated straddle reads, counted per block boundary.

**Done when**: the three w_non_aligned bf16 DRAM→DRAM single cells pass (exact) with `"w_non_aligned"` in SUPPORTED; every Phase 0 cell, the plan tests and the regime-pinned pair pass; and a test reads off the host split that input tile reads per block are ≤ the block's distinct input tiles (no per-output-tile re-read).

### [ ] Refinement 2 — Placement, float32, H non-alignment and batches (both regimes)

**Goal**: widen `validate()` to every remaining TARGET value. The `tile_relay` program already moves all of them bit-exact; a verifier probe that bypassed `validate()` ran each one exactly, and the precision baseline runs float32 exactly through the program:
- add `ttnn.float32` to `SUPPORTED["dtype"]` (R1: no element passes through another dtype — the DM relay moves raw pages; the ring's format is a label);
- add `"L1"` to `SUPPORTED["memory_config"]` (R5: an L1-interleaved input is read through the same `TensorAccessor`);
- add `None` and `"L1"` to `SUPPORTED["output_memory"]` (R3: `None` → the input's buffer, interleaved; already implemented in `_output_memory_config`, the probe confirmed placement);
- add `"h_non_aligned"` to `SUPPORTED["alignment"]` (`St = ceil(S/32)` per image already; padding rows travel with their tiles);
- add `"batched"` to `SUPPORTED["batch"]` (b is in the reader's walk from day 1).
After this refinement SUPPORTED equals TARGET: all 216 golden cells in the rectangle are `supported_pass`, `xfail_expected` is 0.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Knob-turns only, so one bundle (dtype + placement + alignment + batch). `/numeric-formats-metal` is attached for the dtype half only. This op has **no compute kernel and no compute config**, and the specification's signature forbids adding `compute_kernel_config`, so the skill's config / intermediate-CB parts do not apply. The pass condition is zero kernel changes for `tile_relay`. Placement has no skill: an interleaved L1 tensor is read and written natively through `TensorAccessor`, and no `ttnn.to_memory_config` wrapper may appear (one-dispatch rule).
- Lands after Refinement 1 so the widening covers the `subtile_assemble` cells too (float32 with 4-byte element runs, L1, batched — `(2,3,17,50)`). If the assembler turns out bfloat16-only, fix it in the kernel. Do not EXCLUDE those cells: they are structural capability, not precision.
- No new axis, no EXCLUSIONS expected. A sharded `memory_config` (input or output) stays refused by name (R4/R6): `sharding` is `["interleaved"]` only in TARGET.
- Keep the one-device-session test passing with an L1-placed input and output, and with float32 (the ring entry is 4096 B there; `RING_DEPTH_TILES × max(R, W) × 4096` must stay under the ring cap the host asserts).

**Done when**: the golden suite shows 216 / 216 cells `supported_pass` with 0 `xfail_expected`, 0 `xpass_drift`, 0 `xfail_wrong_mode`; every result exact; R3 placement checked by `HOOKS["check"]` on every cell.

### [ ] Refinement 3 — Stagger the DRAM bank each cluster starts on (tile_relay)

**Type**: perf

**Goal**: no loose case is perf-flagged (`LOOSE_CASES = []`), so this region is chosen by the verifier. The input and the output are interleaved over the device's DRAM banks (`device.dram_grid_size()` = 2×1 on quasar, page `p` in bank `p mod num_banks`). Each cluster walks its contiguous range from `start_page`, and on every shape whose per-cluster count is even, every engaged cluster starts on the **same bank**. The clusters then advance in lockstep, so at every step all 32 hit one bank while the other idles: on the writer side (contiguous output pages) and on the reader side alike (the source page's parity follows `c`). Measured off the host split (verifier probe): `[1,32,128,64]` → 32 / 32 clusters start on bank 0; `[1,64,32,128]` → 32 / 32; `[1,2,512,64]` → 32 / 32; the model's `[1,32,4096,128]` → 32 / 32. The traced calls `[1,4,32,64]` / `[1,4,32,128]` are already balanced (4/4, 8/8). Apply the `bank_stagger` pattern from `ttnn/ttnn/operations/examples/master.md` (⭐⭐ T2): rotate each cluster's walk start within its range by `core_index mod num_banks`, and wrap. Reader and writer must rotate **identically**, because the ring is FIFO and the writer stores in the reader's order. Make it a compile-time switch, so it can be compared on a timed backend. Cheap levers that may ride along in this phase: none with a structural signal. No SUPPORTED change.

**Done when**: a test reads the structural count off the program (host split + rotation offsets, the bank count queried from the device), the maximum number of engaged clusters whose first transfer in each direction lands on one DRAM bank. It must drop from 32 to ≤ `ceil(engaged / num_banks)` = 16 on `[1,32,128,64]`, `[1,64,32,128]` and `[1,2,512,64]`, without exceeding the current value on the traced shapes. The golden suite must stay green and exact, with no regression across the config-spanning guard set: one representative per distinct kernel path × placement × dtype (tile_relay bf16 DRAM→DRAM, tile_relay fp32 L1→L1, tile_relay batched + h_non_aligned, subtile_assemble). The changelog records the count before and after and states that the gain in time is unconfirmed on this untimed backend.

### [ ] Refinement 4 — Assembler threads for the `subtile_assemble` regime

**Type**: perf

**Goal**: in `subtile_assemble` (built by Refinement 1) every output tile is assembled element-run by element-run with RISC-V stores on one DM hart. That is far more per-tile work than the one NoC read and one NoC write that bound `tile_relay`, so the assembler, not DRAM, is expected to bound the w_non_aligned cells (`(1,4,32,50)`, `(1,3,64,17)`, `(1,8,17,50)`, `(2,3,17,50)`). Turn the assembler thread-count knob Refinement 1 exposes, within the 6 free DM harts (R + A + W ≤ 6, e.g. (1, A=4, 1) or (1, A=2, 2)), dealing blocks to assembler harts. A block is larger than a ring entry, so deal **whole rounds** of blocks and pad as the target page's *Thread plan* says (toy_row_passes shows the pattern; a hart without a block re-assembles a real one). See `ttnn/ttnn/operations/examples/master.md` (`split_reader`: split independent per-unit work across harts when one hart is the issue bottleneck). No SUPPORTED change.

**Done when**: assembler harts at work on the w_non_aligned cells go from 1 to the chosen A ≥ 2, shown by a test in which every assembler hart's share is observable (a missing hart leaves output undone), on a shape whose block count is not a multiple of A. Every w_non_aligned golden cell stays exact, the one-device-session test passes with the multi-hart assembler, and the guard set from Refinement 3 shows no regression. The changelog states the gain in time is unconfirmed on this untimed backend.
