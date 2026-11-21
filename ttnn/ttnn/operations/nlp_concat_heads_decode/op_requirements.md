# Operation Requirements: nlp_concat_heads_decode

## Definition
- **Formula**: `output[0, 0, u, h * head_dim + d] = input[0, u, h, d]` for `u < users`, `h < num_heads`, `d < head_dim`; output shape `[1, 1, 32, num_heads * head_dim]`, rows `users..31` are padding (spec R3). Every defined element is moved unchanged (R1).
- **PyTorch Reference**:
  ```python
  def reference(x, num_heads):
      _, users, _, head_dim = x.shape
      return x.double()[:, :, :num_heads, :].reshape(1, 1, users, num_heads * head_dim)
  ```
- **Import Path**: `from ttnn.operations.nlp_concat_heads_decode import nlp_concat_heads_decode`
- **Function Signature**:
  `nlp_concat_heads_decode(input_tensor: ttnn.Tensor, *, num_heads: int, sub_core_grids: ttnn.CoreRangeSet = None) -> ttnn.Tensor`
  (input `[1, users, H, head_dim]`, TILE, L1 height-sharded one `[32, head_dim]` shard per user; output L1 width-sharded one `[32, head_dim]` shard per head on `ttnn.num_cores_to_corerangeset(num_heads, grid, row_wise=True)`, R4)

Specification: `eval/specs/nlp_concat_heads_decode/spec.py` (governs). Target: quasar on ttsim (no timing model — perf gates below are structural counts).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED memory_config / sharding**: L1 / height (the input's one placement in TARGET)
- **SUPPORTED shape-derived axes**: users=1, alignment=tile_aligned only
- **Cores**: one output head per core on `num_cores_to_corerangeset(num_heads, grid, row_wise=True)` (1–32 clusters over INPUTS); thread plan (R, C, W) = (1, 0, 0) — a pure data-movement gather, no compute or writer kernel
- **Compute config**: none (no compute kernel)
- **Golden baseline**: 12 / 12 supported cells pass, 18 xfail_expected, 7 / 7 rejections pass, 0 loud categories (per verifier CLI)

### [ ] Refinement 1 — float32 and fewer-than-32 head rows (dtype + alignment)

**Goal**: add `ttnn.float32` to `SUPPORTED["dtype"]` and `"h_non_aligned"` to `SUPPORTED["alignment"]` — the whole remaining `TARGET − SUPPORTED`. Both are knob-turns in the Blocking Model ("TARGET − PHASE0" table of `op_design.md`): float32 only changes the host-derived CT geometry (`tile_bytes` 4096, `face_bytes` 1024, `run_bytes` 64), and H < 32 only changes which rows exist in the tile-padded input page (the kernel reads row `head_idx < num_heads ≤ H` and nothing else, R8). Moves the 18 `xfail_expected` cells (12 float32 tile-aligned, 3 bfloat16 + 3 float32 h_non_aligned) to `supported_pass`. Expected kernel delta: none.

**Implementation skill**: /numeric-formats-metal, /memory-layouts

**Verifier notes**:
- Verifier probe (validate gate relaxed in-process, kernel unchanged) already ran on ttsim: float32 × {(1,1,32,64) h2, (1,1,32,256) h32, (1,1,32,160) h8} and bfloat16/float32 × {(1,1,8,64) h8, (1,1,12,128) h2, (1,1,17,32) h17} were all bit-exact with zeroed padding rows; the golden runner's `auto_shard_config` builds a `[32, head_dim]` shard for H < 32, so the existing shard-shape check in `validate()` holds. So this refinement is mostly the SUPPORTED edit, plus proving it.
- `/numeric-formats-metal` applies only in part: the op has no compute kernel, no `compute_kernel_config` (the spec gives no such parameter: do not add one) and no intermediate buffer, so the dtype work is limited to SUPPORTED + R1 (float32 values must survive unchanged — they do, the data never leaves raw-byte NoC reads). `/memory-layouts` is attached for the alignment value's non-aligned rule only; no layout changes.
- Update `test_nlp_concat_heads_decode_plan.py::test_float32_refused_in_phase0` (it asserts float32 is refused) into a float32 positive case, and extend `test_every_reader_thread_ran` / `test_session_order` with one float32 and one H < 32 case each (the thread split and the tag hand-back must hold for the new geometry).
- Add float32 + one h_non_aligned shape to `test_nlp_concat_heads_decode_precision_baseline.py` (expect exact: PCC 1, rel-RMS 0, ULP 0).

**Done when**: `SUPPORTED["dtype"] == [bfloat16, float32]`, `SUPPORTED["alignment"] == ["tile_aligned", "h_non_aligned"]`, the verifier CLI shows 30 / 30 `supported_pass` with 0 `xfail_expected` and every loud category at 0, and the plan / precision tests above pass.

### [ ] Refinement 2 — Take the per-read setup and the zero-fill barrier off the gather's critical path

**Type**: perf

**Goal**: no `LOOSE_CASES` entry is perf-flagged, so the region is free-selected: the models' captured calls, `((1,1,32,128), 32)` (Qwen3-VL, 4096 wide) and `((1,1,32,64), 32)` (Llama, 2048 wide), on all 32 clusters, at bfloat16 and float32 (both SUPPORTED after Refinement 1). Each core issues `2·head_tiles·users` (8 / 4) reads of 32–64 B. That makes the op issue-latency bound, not byte bound: the whole op moves 8 KiB of NoC traffic, so the fixed costs per read and per barrier are the cost. Two catalog levers from `ttnn/ttnn/operations/examples/master.md` (`double_buffer` — keep bytes in flight and do not serialize on a barrier; `split_reader`'s point that per-transaction issue cost is what to cut) and from the design's Perf lamps:
1. **Stateful reads** (design lamp "Stateful reads"). For a fixed user, every read goes to the same source core with the same `run_bytes`. Program the read state once per (thread, user) with `noc.set_async_read_state<NocOptions::TXN_ID>(input, run_bytes, {.page_id = u·head_tiles}, {.trid = 0})`, then issue `noc.async_read_with_state<NocOptions::TXN_ID>(...)` per run (`tt_metal/hw/inc/api/dataflow/noc.h:283-370`). This drops the per-read TensorAccessor page→core resolution, the per-read trid write and the per-read command programming. The shard's pages are contiguous on its core, so tile `t` sits at `+t·tile_bytes` from page `u·head_tiles`; confirm this against the accessor before relying on it.
2. **Zero fill off the critical path** (design lamp "Zero fill (overlap)"). Today each thread zeroes its whole tile range (`(t_end−t_begin)·tile_bytes`, including the rows the reads then overwrite) and waits on `write_zeros_l1_barrier()` before the first read issues. Instead, zero only the padding ranges, rows `users..31` of each tile, which for users = 1 are two contiguous ranges per tile, `2·(t_end − t_begin)` per thread (`[t·tile_bytes + run_bytes, t·tile_bytes + face_bytes)` — face 0 below the defined row — and `[t·tile_bytes + face_bytes + run_bytes, (t+1)·tile_bytes)` — face 1 below the defined row plus faces 2–3; they do not merge across tiles because each tile's row 0 is defined; derive the general-`users` form from the same `row_run_offset` arithmetic). Issue those disjoint zero ranges concurrently with the reads and put one zero barrier at the end. Keep the padding zeroed: GUIDANCE notes the callers read into those rows, and zeros satisfy them always.

No SUPPORTED change.

**Structural gate (ttsim has no clock — backend page, Performance measurement)**, per thread at `((1,1,32,128), 32)`, read off the op by a probe or test (e.g. a DPRINT-counted or CT-switch-instrumented kernel, or a host-side count test over the CT args):
- TensorAccessor page resolutions + read-command full programs per thread: **8 → 1** (one `set_async_read_state` per user; `users = 1`).
- trid register writes per thread: **8 → 1**.
- barriers waited before the first read issues: **1 → 0**.
- bytes zero-filled per core: **`head_tiles·tile_bytes` (8 KiB bf16) → `head_tiles·tile_bytes − users·head_tiles·2·run_bytes`** (the defined row is no longer zeroed and then overwritten).

The gain in time is **unconfirmed** on this backend: say so in the changelog and outcome, and record the before/after counts, not any simulator timing.

**Verifier notes**: runs after Refinement 1 so that both dtypes (whose `run_bytes` differ, 32 vs 64) are covered by the same stateful path. Keep the CT switch pattern (`zero_padding_rows` already is one): add a CT switch for each lever (e.g. `stateful_reads`) so a timed backend can A/B them later without code changes. The thread-plan static_asserts and the trid hand-back (`noc_async_read_set_trid(0, …)` as the last statement) must survive. The iDMA zero borrows write command buffer 0 (`noc.h:832-836`); the reads use the read command buffer. Confirm on ttsim that concurrent zero + read does not corrupt rows (the session-order test plus an exact-match test at head_dim 256 cover it).

**Done when**: the four structural counts above reach their target values on the target shapes (recorded before/after in `changelog.md`), the golden suite is green (all `supported_pass`, 0 loud), `test_session_order` and `test_every_reader_thread_ran` pass for bfloat16 and float32, and nothing regresses across the config-spanning guard set: one representative per kernel path × dtype × alignment (e.g. `((1,1,32,32),1)` bf16, `((1,1,32,256),32)` fp32, `((1,1,17,32),17)` bf16 h_non_aligned, `READER_THREADS ∈ {1, 2}`), each still exact with the structural counts no worse than before.
