# Operation Requirements: clone

## Definition
- **Formula**: `output[i] = input[i]` for every element (and every padding byte of every page). The output is a new interleaved tensor in the buffer `memory_config` names (DRAM or L1), or in the input's buffer when `memory_config` is None. Nothing is converted (spec R1, R2).
- **PyTorch Reference**: `torch.clone(input)` (spec `reference(input)` = `input.double()`)
- **Import Path**: `from ttnn.operations.clone import clone`
- **Function Signature**: `clone(input: ttnn.Tensor, *, dtype: ttnn.DataType = None, memory_config: ttnn.MemoryConfig = None) -> ttnn.Tensor`

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Perf on this backend (ttsim, untimed)**: a perf refinement is gated on a **structural count** read off the op by a test or probe (backend page, *Performance measurement*). Any gain in time is unconfirmed until a timed backend or hardware measures it; never report a simulator timing as a speedup.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE, ROW_MAJOR]
- **SUPPORTED memory_config / sharding (input)**: DRAM, L1 / interleaved
- **SUPPORTED dtype_argument**: [None, "same"]
- **SUPPORTED output_memory**: ["L1"]
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}, rank ∈ {2, 3, 4}
- **Regime**: `interleaved_page_stream`: per-cluster contiguous page range over the full grid (`split_work_to_cores(..., row_wise=True)`), reader → STRIDED DM→DM ring `staged_pages` → writer, implicit sync at both ends, trid handed back last. No compute kernel.
- **Cores**: full grid (32 clusters on quasar), `min(32, tensor_pages)` engaged
- **Thread plan**: (R, C, W) = (1, 0, 1), from one constant `THREAD_PLAN`; bound by data movement (design, Work Distribution)
- **Compute config**: none (the op computes nothing; spec: no compute config)
- **Golden baseline**: 200 / 200 supported cells pass, 1000 xfail_expected, 0 supported_fail / xpass_drift / xfail_wrong_mode, X1 and X2 rejections pass (per `verifier_report.json`)
- **Precision**: bit-exact on every measured shape (PCC 1.0, abs err 0, relative RMS 0, ULP 0)

### [ ] Refinement 1 — float32 and every output placement (the rest of TARGET)

**Goal**: add `ttnn.float32` to `SUPPORTED["dtype"]`, and add `None` and `"DRAM"` to `SUPPORTED["output_memory"]` (so `SUPPORTED["output_memory"]` = [None, "DRAM", "L1"]), which closes `TARGET − SUPPORTED` entirely. This is a knob-turn in the design (`op_design.md`, *TARGET placement axes against the blocking scheme*): `page_bytes` / `entry_bytes` are derived from the tensor, the program is placement-agnostic, and `_output_memory_config()` already maps `memory_config=None` to the input's buffer, interleaved (R3). Expected kernel diff: none.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Evidence that this is a knob-turn: a verifier probe that drove `create_program_artifacts` directly (bypassing `validate()`) copied 48/48 cells bit for bit: float32 and bfloat16 × TILE and ROW_MAJOR × `(1,1,17,50)`, `(2,1,100,47)`, `(1,32,4096)`, `(32,17)` × (DRAM→DRAM, L1→DRAM, L1→L1).
- Only the dtype half of `/numeric-formats-metal` applies. The spec gives clone **no compute config** (`TT_METAL["drop"]` drops `compute_kernel_config`; DEFINITION: "It therefore has no compute config"), and there is no compute kernel or intermediate CB. Do **not** expose `compute_kernel_config`, and do not add a compute stage: a float32 value through DEST breaks R1 (`compute_passthrough` is rejected in the design).
- Output placement has no implementation skill: it is a `validate()` / SUPPORTED change plus the review checks the golden suite cannot make (R3 and R5 are `review` rules). Extend `tests/.../clone/test_clone_plan.py`:
  - `test_placement_and_fresh_buffer`: assert `output.memory_config()` is `{DRAM, L1}` × INTERLEAVED for each `memory_config` ∈ {None, DRAM, L1} and each input buffer ∈ {DRAM, L1}. `None` must land in the input's buffer. Assert `output.buffer_address() != input.buffer_address()` whenever the output is in the input's buffer (R5: `memory_config=None` is now the common case of that).
  - `test_refusals`: the two `R8` asserts (`memory_config=DRAM`, `memory_config=None` refused) become accepted calls. Keep an R8 refusal with a value still outside SUPPORTED (for example `dtype=ttnn.int32` on an int32 input, or a bfloat8_b input, which the target page says to refuse).
  - Add float32 bit-exactness (R1's own check): a float32 input with values not representable in bf16 must come back bit-equal (compare `.view(torch.int32)`), in both layouts.
  - Rerun the device-session tests with a DRAM output and a float32 tensor (the trid hand-back is dtype- and placement-independent, but the session tests are what show it).
- The oversized-page cap still holds for TARGET: the widest float32 row is 8192 × 4 B = 32 KiB, under 65,536 B.
- Lands before Refinement 2 so `owner_local` is built against the complete placement surface (its selection predicate must fall back to the built stream for DRAM→DRAM, which this refinement makes reachable).

**Done when**: `SUPPORTED` equals `TARGET` on every axis. The golden suite has 1200 supported cells passing, 0 xfail_expected, and 0 supported_fail / xpass_drift / xfail_wrong_mode. The placement, fresh-buffer and float32 bit-exactness tests above pass, and the acceptance test and every earlier test stay green.

### [ ] Refinement 2 — `owner_local`: write each L1 page from the core that owns its bank

**Type**: perf

**Goal**: build the deferred cheapest-traffic regime `owner_local` (`op_design.md`, *Regimes* and *Traffic ranking*; `l1_ledger.md`, data-movement budget) for calls with an L1 endpoint. No loose case is perf-flagged, so the target is the traced YOLOv8 calls:
- `[1,1,1600,512]` bf16 TILE, DRAM → L1 (the largest; also `[1,1,400,512]`, `[1,1,400,256]`, `[1,1,1600,256]`);
- `[1,1,400,64]` bf16 ROW_MAJOR, L1 → L1.

The core that owns an output page's L1 bank reads the input page straight into that page's local slot. There is no ring and no NoC write. Where only the input is in L1, its owner reads the local page and writes it out. DRAM → DRAM stays on the built `interleaved_page_stream` (the design's selection predicate). The regime is selected on the host behind the same `validate()`, output allocation and **one** `generic_op` dispatch. Relevant catalog entries in `ttnn/ttnn/operations/examples/master.md`: `bank_stagger`, `noc_placement`, `read_coalescing`. This is a T3 scheme-change: a new page-to-core topology. No SUPPORTED change.

**Verifier notes**:
- The ledger's reason for deferring this ("no ttnn Python binding exposes the bank → core map") is weaker than stated. `ttnn.reports.get_buffer_pages(device)` returns `core_x`, `core_y`, `bank_id` and `page_index` per page (`ttnn/cpp/ttnn-nanobind/reports.cpp:38-52`). It has not been run on Quasar yet, so confirm it there first. A kernel can also recognise a local page from the NoC address `TensorAccessor` gives for it. Prove that every L1 bank of the output lies on a cluster of the compute grid (`ttnn.reports.get_device_info(device).l1_num_banks`, `num_banks_per_storage_core`). If that cannot be proved, the regime's predicate must fall back to the stream.
- Ring protocol: a read that lands directly in a tensor slot is not a ring transfer. Follow the target page's *Ring protocols* rules: a user trid and a trid barrier for untagged-ring work, and hand the tag back as the last statement. Keep the device-session tests (a tt-metal program before and after) for each regime.
- Thread plan: on this backend nothing can show that (2,2) beats (1,1) for small sticks. Keep `THREAD_PLAN` unless the new regime needs per-thread page sets; if so, keep it one constant and keep the sentinel every-thread-ran test for each regime.
- Ledger: when this lands, add the regime's row (no ring for L1→L1), change `owner_local` from `deferred` to `built` in the design's regime table, and update the data-movement budget.

**Done when**: a test or probe reads the structural counts off the built program and they move as follows (gain in time unconfirmed on ttsim):
- **cross-core NoC bytes written to remote L1 banks**: `[1,1,1600,512]` TILE DRAM→L1 goes from ≈1,638,400 B to 0 (every output page written by its bank's owner, checked page by page against `get_buffer_pages` or the accessor's NoC coordinates). `[1,1,400,64]` RM L1→L1 goes from 2 × 51,200 B of NoC traffic to ≤ 51,200 B.
- **ring entries staged per page** on those calls: from 1 to 0.

In addition: DRAM→DRAM and every other cell stay on the stream, with unchanged counts. The golden suite stays green (with Refinement 1 landed: 1200 supported cells pass), the precision baseline stays bit-exact, and there is no regression on the config-spanning guard set: one representative per regime × layout × (input buffer, output buffer), plus the every-thread-ran and device-session tests.

### [ ] Refinement 3 — Bank-run coalescing of small row-major sticks

**Type**: perf

**Goal**: on top of `owner_local` (Refinement 2), move a run of consecutive same-bank pages in **one** NoC transaction instead of one per page, where the input and output slots match (same page size, same slot alignment, same bank count: L1 → L1 interleaved). The design names this as `owner_local`'s successor (*Regimes*, "What a bigger block buys"). Target: the traced `[1,1,400,64]` bf16 ROW_MAJOR L1 → L1 call (128-B sticks, where cost is per transaction, not per byte), and `[2,1,100,47]` RM (96-B slots). Relevant catalog entries in `ttnn/ttnn/operations/examples/master.md`: `read_coalescing`, `double_buffer` (bytes in flight). No SUPPORTED change.

**Verifier notes**:
- This is a true dependency on Refinement 2: a bank's pages are contiguous only in that bank, and only the owner-local assignment hands one core a whole bank's run. On the stream regime, consecutive pages sit in different banks.
- A run is capped at one NoC packet (65,536 B, `NOC_MAX_BURST_BYTES`). Reuse that constant; do not restate it.
- DRAM → L1 sticks do not qualify: DRAM slot alignment (64 B) differs from L1 (16 B), and the bank counts differ. The predicate must check the slot geometry and fall back to per-page transfers otherwise.
- This is a block-size knob-turn on the new regime (the run length is the block on the `page` axis). It must be a host-side parameter with one source of truth, passed as an arg, never a literal in the kernel.

**Done when**: a test or probe reads off the program that the **NoC transactions per page** for `[1,1,400,64]` RM L1→L1 go from 1 to `≤ 1 / min(run_pages, floor(65,536 / 128))` per bank run (one transaction per bank run instead of one per page), with the run length taken from the host knob (gain in time unconfirmed on ttsim). In addition: the golden suite stays green, the precision baseline stays bit-exact, and there is no regression on the guard set (one representative per regime × layout × placement, plus the every-thread-ran and device-session tests).
