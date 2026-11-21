# Operation Requirements: nlp_create_qkv_heads_decode

## Definition
- **Formula**: with `head_dim = W / (num_heads + 2 * num_kv_heads)` (`num_kv_heads = None` → `num_heads`):
  `q[0,u,h,d] = x[0,0,u,h*head_dim+d]`, `k[0,u,h,d] = x[0,0,u,(num_heads+h)*head_dim+d]`,
  `v[0,u,h,d] = x[0,0,u,(num_heads+num_kv_heads+h)*head_dim+d]`. Every output element is one input element, unchanged (R1).
- **PyTorch Reference**:
  ```python
  def reference(x, num_heads, num_kv_heads=None):
      kv = num_heads if num_kv_heads is None else num_kv_heads
      _, _, users, width = x.shape
      hd = width // (num_heads + 2 * kv)
      rows = x[0, 0]
      return [rows[:, a * hd:(a + n) * hd].reshape(1, users, n, hd) for a, n in ((0, num_heads), (num_heads, kv), (num_heads + kv, kv))]
  ```
- **Import Path**: `from ttnn.operations.nlp_create_qkv_heads_decode import nlp_create_qkv_heads_decode`
- **Function Signature**:
  `nlp_create_qkv_heads_decode(input_tensor: ttnn.Tensor, *, num_heads: int, num_kv_heads: int = None, overlap_qk_coregrid: bool = True, memory_config: ttnn.MemoryConfig = None) -> list` (`[q, k, v]`, each height-sharded in L1, one shard `[heads rounded up to 32, head_dim]` per user on the first `users` cores, rule R5)
- **Specification**: `eval/specs/nlp_create_qkv_heads_decode/spec.py` (governs). Target: Quasar on ttsim.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Perf gates on this backend**: ttsim has no clock. A perf refinement here is gated on a **structural count** read off the op by a test (backend page, *Performance measurement*); its gain in time is unconfirmed until a timed backend (`ttsim-perf`) or hardware measures it.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED placement**: input `memory_config` = [L1], `sharding` = [interleaved]; `output_memory` = ["L1/height"]
- **SUPPORTED shape-derived axes**: `users` = [1], `head_rows` = [32]
- **Cores**: the output shard grid of rule R5 (`num_cores_to_corerangeset(users, grid, row_wise=True)`); users = 1 → core (0,0). One core is the placement the specification mandates, not an under-filled split.
- **Thread plan**: (R, C, W) = (2, 0, 0): one data-movement kernel `gather`, `num_threads = GATHER_THREADS = 2`; no compute config (the op computes nothing).
- **Golden baseline**: 8 / 8 supported cells passing, 72 xfail_expected, 8 / 8 rejections passing (per verifier CLI)

### [ ] Refinement 1 — Complete the TARGET rectangle: DRAM input regime, float32, head_rows 64, `memory_config=None`

**Goal**: bring every TARGET value into SUPPORTED. The core of the work is the DRAM input regime (design regime **A-DRAM**); the other three values ride along:

- add `"DRAM"` to `SUPPORTED["memory_config"]` — regime A-DRAM of `op_design.md`: the same direct gather, with the source accessor's DRAM-alignment branch. Where `face_row_bytes % ttnn.get_dram_alignment() != 0` (bfloat16: a 32 B run against Quasar's 64 B DRAM alignment), read each run's aligned window into a per-hart scratch slot, then move the run into its final place in the shard. Where the run is already aligned (float32: 64 B), read it directly into the shard as regime A does. Build a regime-selection function (`"A-DRAM" if memory_axes(input)["memory_config"] == "DRAM" else "A"`) and pin one test per regime by node id.
- add `ttnn.float32` to `SUPPORTED["dtype"]` — `face_row_bytes` is derived from the page size, so the run walk is already dtype-generic. Rule R1 (no element passes through another dtype) is met by construction: there is no intermediate buffer, and the bounce slot carries raw bytes in the input's format.
- add `64` to `SUPPORTED["head_rows"]` — the destination offset already uses `h // 32` for the shard's tile row.
- add `None` to `SUPPORTED["output_memory"]` — it selects the same R5 placement as `"L1/height"` (rule R6), with the same device program.

**Verifier notes**:
- Evidence: a verifier probe (`tests/ttnn/unit_tests/operations/nlp_create_qkv_heads_decode/probes/probe_003.py`) widened SUPPORTED in-process. Every combination {bf16, fp32} × {L1, DRAM} × {HS, None} × {32/8, 64/8, 40/40, 1/1 heads} was **bit-exact on ttsim with the Phase 0 kernel unchanged**. So float32, head_rows 64 and None are SUPPORTED changes plus tests.
- DRAM is the one value with real kernel work. Build the aligned path (the design and the GUIDANCE require it: tt-metal #43270 on Blackhole) and prove it structurally, as below. Do not ship the unaligned read.
- Build it performantly (performance-conformance), because Refinement 2 tunes this same path:
  - Keep one barrier per block per hart in both regimes.
  - Do not put a per-run barrier between the window read and the move into the shard. Batch the window reads of a block into per-hart scratch slots, drain them once, then do the moves.
  - Prefer a core-local copy (`local_copy_helpers_dataflow.hpp`, or a RISC-V copy of the 32 B) over a second NoC round trip per run.
  - Size the scratch from a named block knob (runs per drain), not from `input_width_tiles` unconditionally. `heads_per_block` and `GATHER_THREADS` must stay live, single-source knobs.
- Every NoC transfer stays tagged `TXN_ID {.trid = 0}`, including the local move if it is a NoC read. The last statement still resets the tag. Extend `test_one_device_session` with a DRAM input.
- Ledger: the scratch slot is the op's first op-allocated L1. Add its row to `l1_ledger.md` (capacity expression in the runs-per-drain knob, live set, axis accounting, `Shares with / why not`). Update the data-movement budget: a bf16 DRAM input crosses DRAM once, at 2× the run bytes through the 64 B windows, plus one core-local move per run.
- No skill in the inventory covers input placement in DRAM vs L1 for a direct-gather data-movement op. `/memory-layouts` is RM/TILE layout and `/numeric-formats-metal` is compute-config and CB precision for math ops; this op has neither a compute kernel nor CBs. So this entry is verifier-authored.
- Partial-tick: if the DRAM bounce stalls, land the other three values, mark `[~]`, and file the DRAM regime as `Refinement 1b`. Never gate DRAM behind EXCLUSIONS while the unaligned direct read is what runs.

**Done when**:
- SUPPORTED equals TARGET (dtype bf16+fp32, memory_config L1+DRAM, output_memory None+"L1/height", head_rows 32+64), and the golden suite shows 80 / 80 cells `supported_pass` with 0 `xfail_expected`, 0 `xpass_drift` and 0 `xfail_wrong_mode`.
- A test asserts on the built program (not on values) that every bf16 DRAM-source read has a size and source offset that are multiples of `ttnn.get_dram_alignment()`, and that fp32 runs go direct.
- `test_thread_plan.py` (every hart runs; block knob; one device session) passes for both regimes, L1 and DRAM.
- The precision baseline gains fp32 and DRAM shapes, all bit-exact.

### [ ] Refinement 2 — Raise read-issue parallelism on the traced decode calls

**Type**: perf

**Goal**: no `LOOSE_CASES` entry is perf-flagged, so the region is free-selected: the two traced calls, Llama `[1,1,1,3072]` 32/8 and Qwen `[1,1,1,6144]` 32/8. Their contract is bf16, an interleaved input in L1 (and in DRAM after Refinement 1), `memory_config = "L1/height"`, and every part of it is in SUPPORTED.

The op is bound by the number of transfers issued: 192 / 384 reads of 32 B on core (0,0). Today 2 of the 6 free DM harts issue them, 96 / 192 each, all on one NoC. Rule R5 pins the whole op to one core, so only in-core issue parallelism is left.

Turn the plan the design exposes (catalog `ttnn/ttnn/operations/examples/master.md`):
- `split_reader`: the `GATHER_THREADS` constant, R = 4 or 6, which `test_every_hart_runs` already shows exact.
- `noc_placement`: NoC selection. Deal half the harts' runs to the second NoC, if a Quasar DM thread can choose its NoC; if it cannot, record why.
- `trid_double_issue`: keep one barrier per hart per block.

Co-tune `heads_per_block` only if a barrier count moves. No SUPPORTED change.

**Done when**:
- The structural count, read off the built program by a test, improves on both traced shapes. The count is the reads issued by the busiest hart, `ceil(2 * input_width_tiles / R)`, and the target is Llama 96 → ≤ 48 and Qwen 192 → ≤ 96. If both NoCs are used, the test also records reads per NoC (192 → ≤ 96 for Llama).
- The reads per barrier per hart do not drop.
- The changelog records the counts before and after and states that the gain in time is **unconfirmed** on ttsim until `ttsim-perf` or hardware measures it. (On `ttsim-perf`, measure R = 2 / 4 / 6 once each on Qwen, as the design's perf lamp asks.)
- The golden suite stays green.
- The guard set is exact at the new plan. It has one representative per distinct kernel path × placement: L1 bf16 (regime A), DRAM bf16 (A-DRAM bounce), DRAM fp32 (A-DRAM direct), and head_rows 64.
- `test_every_hart_runs` and `test_one_device_session` still pass.
