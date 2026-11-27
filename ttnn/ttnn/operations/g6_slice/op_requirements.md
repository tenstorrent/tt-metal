# Operation Requirements: g6_slice

## Definition
- **Formula**: `out[i_0..i_{r-1}] = x[b_0 + s_0·i_0, …, b_{r-1} + s_{r-1}·i_{r-1}]`, `M_d = ceil((e_d − b_d)/s_d)`, `b_d = clamp(starts[d] (+N_d if < 0), 0, N_d)`, `e_d = clamp(ends[d] (+N_d if < 0), b_d, N_d)`; bit copy, no arithmetic. Device-tensor form: `out = x[..., r0 : r0 + L, ...]` on dim `sd = slice_dim`, `r0 = floor(starts[sd]/32)·32` read on device, `L = N_sd / num_devices`; `ends` never read.
- **PyTorch Reference**: `x[starts[0]:ends[0]:steps[0], …, starts[r-1]:ends[r-1]:steps[r-1]]` (bit-identical); device form `x.narrow(sd, (starts[sd] // 32) * 32, N_sd // num_devices)` (`eval/golden_tests/g6_slice/helpers.py:pytorch_g6_slice`).
- **Import Path**: `from ttnn.operations.g6_slice import g6_slice`
- **Function Signature**: `g6_slice(input_tensor: ttnn.Tensor, starts, ends, steps=None, *, slice_dim: int = None, num_devices: int = None) -> ttnn.Tensor` — `starts`/`ends`: `list[int]` of length r, or `ttnn.Tensor` uint32 ROW_MAJOR DRAM `(r,)`; `out` takes `x`'s dtype, layout and memory config.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Golden runs (this op)**: run the suite with `EVAL_CAPTURE_L1=0 eval/eval_test_runner.sh …`. The default L1 capture disables the program cache per test, so the trace loose cases (`Qwen703_trace`, and `FLUX189_trace_begin` once device begins land) fail for any implementation and leave the device in trace capture for the rest of the run (see `verification_report.md`, Harness findings).
> **Perf measurements (this board)**: AICLK moved between 800 and 1350 MHz across sessions (±25 % device-ns on one build). Compare variants only inside one pytest session (interleaved arms, as in the verifier's A/B) and record the profiler's `CHIP_FREQ` line with every number.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE_LAYOUT]
- **SUPPORTED memory_config**: ["dram_interleaved"]
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}, rank ∈ {2, 3, 4}
- **SUPPORTED op-specific axes**: begins_source ∈ {host_list}, step_mode ∈ {unit_step}
- **Regimes built**: `page_copy` (whole tiles), `row_runs` (face-row runs, row begin inside a tile or partial last tile row), `col_realign` (column begin inside a tile; RISC-V column realign)
- **Cores**: full grid (110 on this Blackhole), `split_work_to_cores(row_wise=True)` over contiguous output-page ranges; reader NCRISC/NoC0, writer BRISC/NoC1
- **Compute config**: none — no compute kernel (pure data movement)
- **Golden baseline**: 881 supported_pass / 449 xfail_expected / 0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode (1349 collected; 19 registry-less regression tests pass)
- **Perf baseline (flagged case 703, 1350 MHz)**: 2669.6 µs = 1.078 × the 2475.8 µs roofline; 1.20 × faster than stock `ttnn.slice` in a same-session pair

### [ ] Refinement 1 — ROW_MAJOR layout (`stick_copy`)

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]` through the deferred `stick_copy` regime of `op_design.md`, natively in the reader and writer: pages are sticks; per output stick one read of exactly the selected `M[-1]·es` bytes at byte `b[-1]·es` of its input stick, landing at the same offset mod 64 in an aligned slot (DRAM read alignment); a RISC-V realign only when `(b[-1]·es) % 16 ≠ 0` (DRAM write alignment); one write per output stick. Same `out_page` axis (pages = sticks), same full-grid contiguous split, same `block_pages` / `cb_depth` knobs, same one-handshake-per-block reader and writer. Unlocks the 190 bfloat16 ROW_MAJOR DRAM cells (116 `h_non_aligned`, 74 `tile_aligned`); the 121 int32/uint32 ROW_MAJOR cells ride on this path once Refinement 4 declares their dtypes.

**Implementation skill**: /memory-layouts, /interleaved-parallel

**Verifier notes**:
- First because it is the largest structural addition (a new page geometry and regime); Refinements 2 and 4 extend it rather than the other way round.
- No `to_layout` / `tilize` / `untilize` wrapper, not even as a partial-tick fallback: the op spec forbids it outright (prompt rules 1–2), so a wrapped version is a correctness failure, not a partial.
- Read only `[b[-1], e[-1])` of each stick — reading whole sticks to drop columns is the forbidden full-row read (rule 5) on this layout.
- Write the stick path in bytes with the element size taken from the dtype, so Refinement 4 is a declaration on it, not a second rewrite.
- Sticks are 20 B – 17 KB in the production list: per-request and per-handshake costs dominate short sticks, so block many sticks per handshake and keep both halves batched (performance-conformance bar — Refinement 3's guard set measures this path).
- Keep the RM path a CT regime (compiled out of TILE programs, RB-038/BB-18). Add its CB row(s) to `l1_ledger.md` (the block cap uses the aligned stick size as `page_bytes`) and flip `stick_copy` to built in `op_design.md`.

**Done when**: all 190 bfloat16 ROW_MAJOR `dram_interleaved` golden cells pass bit-exact; every Phase 0 cell still passes; `verify_supported` loud categories are 0.

### [ ] Refinement 2 — L1 placements and device-resident begins

**Goal**: add `"l1_interleaved"` and `"l1_width_sharded"` to `SUPPORTED["memory_config"]` and `"device_tensor"` to `SUPPORTED["begins_source"]`:
- **`l1_interleaved`** — same accessors and regimes; replace the hard-coded `out_l1_bytes_per_core = 0` in `_plan()` with the placement-derived term (`ceil(total_out_pages / num_l1_banks) × aligned_page_bytes`) so `out`'s L1 bytes count against the 1347776-B BB-05 budget together with the CBs. 22 bfloat16 TILE cells (+ the bfloat16 ROW_MAJOR `l1_interleaved` cell on Refinement 1's path).
- **`l1_width_sharded`** (`shard_gather`) — `out` keeps `x`'s shard spec unchanged (grid, shard shape, orientation; it fills the first `ceil(M_w / shard_w)` shards); the core set is the output shard owners; one block = the whole output shard; the writer stores into a CB placed on `out`'s shard with `ttnn.cb_descriptor_from_sharded_tensor` (zero copy), and the reader takes the local input shard in place (remote shards only when the column begin crosses shards — plain NoC reads, no semaphores; the input is static). 2 production cells + loose case 1153 (`new_tensor`).
- **`device_tensor`** (`device_begin_copy`) — a reader prologue reads the `starts` page into `cb_slice_pages` slot 0 before the first fetch and applies `begin_page_offset = min(floor(idx/32), floor((N_sd − L)/32)) × page_stride_sd` to `page_copy`'s walk; the `starts` buffer address is a common RT arg patched on a plan-cache hit, `slice_dim` is CT, `ends` is never read. 2 production cells + 5 loose cases (`two_begins` ×2, `trace_begin`, `ends_unread`, `only_sd`).

**Verifier notes**:
- Tier 4 (simple, local sharding + placement), after the layout restructure and before the dtype bundle so Refinement 4 extends every placement once. The device-tensor form is bundled here because alone it is far smaller than a refinement (one prologue + one runtime offset on an existing walk).
- Native sharding means the core's own shard is consumed in place via the CB placement — re-reading the local shard through a `TensorAccessor` as if it were interleaved does not implement `l1_width_sharded` even if the golden cells pass. Never reshard, choose another grid, or fall back to interleaved for `out` (prompt rule).
- Rules that bind once `device_tensor` is in SUPPORTED: `starts[slice_dim]` is read on device on every call, never on the host, never in the program or its cache key; whatever value it holds, the op reads only inside `x` and writes only inside `out` (the clamp); the `ends` tensor's values are never read. The loose cases assert one program-cache entry across begins and a correct replay after an in-place begin write.
- Declaring `device_tensor` also claims it for ROW_MAJOR (Refinement 1 lands first): give the stick walk the same runtime offset. No production cell is device-form ROW_MAJOR, so if that path is out of reach an `EXCLUSIONS` entry `{"begins_source": "device_tensor", "layout": ttnn.ROW_MAJOR_LAYOUT}` is acceptable — never one touching `"h_non_aligned"`.
- Record `out`'s L1 term in `l1_ledger.md` (symbol table already names it) for both L1 placements.

**Done when**: the 22 bfloat16 TILE `l1_interleaved`, 3 `l1_width_sharded` (2 production + `new_tensor`) and 7 `device_tensor` cells pass bit-exact (plus the bfloat16 ROW_MAJOR `l1_interleaved` cell); loud categories are 0; no Phase 0 / Refinement 1 regression.

### [ ] Refinement 3 — Speed up the perf-flagged profile (case 703)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags Qwen case 703 as the mandatory perf target: `x` (16, 4096, 8704) bfloat16 TILE DRAM interleaved, columns 4352..8704 → `out` (16, 4096, 4352), host-list begins, no compute config (`math_fidelity` None), graded bit-exact; 278528 output tiles, 570,425,344 B read + 570,425,344 B written, DRAM-bound. Its goal: kernel time below stock (2930.52 µs) and ≤ 1.35 × roofline (3342.33 µs). Phase 0 already meets both on this board (2669.6 µs at 1350 MHz = 1.078 × the 2475.8 µs roofline); close the remaining ~7 % toward the roofline with the relevant levers in `ttnn/ttnn/operations/examples/master.md` — `bank_stagger` (each core's contiguous range starts on a bank correlated with its neighbours'), `trid_double_issue` (the per-block read barrier drains the reader's queue at every block), `noc_placement` (core ordering / NoC choice per stream). Block size and buffer depth are already swept (cap 8…329 × depth 2/4 all within ~2 %), so don't spend the phase on them. No SUPPORTED change.

**Verifier notes**:
- Measure with same-session interleaved A/B and log `CHIP_FREQ` (see the Phases header); a ≤ 3 % delta is noise (`eval/prompts/perf_rules.txt`).
- Guard set (one per kernel path × layout × placement, > 32 tiles): 795 (`page_copy`), 730 (`page_copy` + narrow last column), 587 and 1175 (`row_runs`, single row), plus one representative each of the Refinement 1 stick path, `l1_interleaved`, `l1_width_sharded` and the device-tensor form.
- The single-row `row_runs` class (587: 6.9 µs vs stock ≈ 2.7 µs) is the op's largest deficit against the incumbent but not the flagged shape — keep it in the guard set and do not trade it away; `verification_report.md` records its levers.

**Done when**: measured device-ns improves on 703 by more than the noise band in a same-session A/B, the golden suite is green (bit-exact), and no regression across the guard set.

### [ ] Refinement 4 — Element formats: float32, int32, uint32, bfloat8_b, bfloat4_b

**Goal**: add `ttnn.float32`, `ttnn.int32`, `ttnn.uint32`, `ttnn.bfloat8_b`, `ttnn.bfloat4_b` to `SUPPORTED["dtype"]`, values copied in the input's own storage format (no conversion anywhere; bit-exact, block formats within one quantization step):
- 32-bit TILE — verifier probe with SUPPORTED widened: float32 `page_copy` / `row_runs` / `col_realign` / narrow last column and int32 `row_runs` are already bit-exact (`page_bytes` 4096 → 64-B face rows, `copy_tile_column<…, uint32_t>`); the work is declaration and tests. 98 float32 TILE cells.
- 32-bit ROW_MAJOR on Refinement 1's stick path: 110 int32 + 11 uint32 cells.
- Block formats: today `page_copy` itself fails to compile for 1088/576-B pages — `TileGeometry`'s `static_assert` and the col-realign `static_assert` fire inside discarded `if constexpr` branches because `kernel_main` is not a template — so a `page_copy` block-format program must instantiate no row geometry. Shifted or partial rows need the deferred `block_format_runs` regime (exponent-aware row realign; reading the ≤ 2 input tiles an output tile draws from is allowed). 4 bfloat4_b DRAM cells, 2 bfloat8_b `l1_interleaved` cells (on Refinement 2).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Last generality refinement (tier 5): each dtype lands once on every layout and placement Refinements 1–2 built.
- The op has no compute kernel, so only the skill's dtype surface and precision-matrix test contract apply; there is no `compute_kernel_config` to expose and no intermediate-CB precision to set.
- `"h_non_aligned"` may not be excluded for block formats (prompt rule) — `block_format_runs` must be native. A column begin that is not a multiple of 16 splits a shared-exponent group (no production case has one); that cell class (`w_non_aligned` × block format) may go to `EXCLUSIONS` if requantizing is out of reach.
- int32/uint32 must stay off any float path: the golden inputs span the full 32-bit range precisely to catch it.

**Done when**: the 98 float32 TILE, 110 int32 ROW_MAJOR, 11 uint32 ROW_MAJOR, 4 bfloat4_b DRAM and 2 bfloat8_b `l1_interleaved` cells pass (bit-exact / `block_step`); loud categories are 0; no earlier cell regresses.

### [ ] Refinement 5 — Second perf pass on the flagged profile (case 703)

**Type**: perf

**Goal**: 703 again (same full config as Refinement 3), after Refinements 1, 2 and 4 have grown the kernels. Spend the lever class Refinement 3 did not: DRAM-bank-affine core assignment (`ttnn/ttnn/operations/examples/master.md` → `read_coalescing`, "bank affinity") — for 703 an output page and its input page sit on the same DRAM bank (`(in_page − out_page) mod 8 = 0`, since 136 tiles per row is a multiple of 8), so each core can take only pages of one bank, as a change of the core-assignment knob instead of contiguous ranges. One ⭐⭐⭐-sized lever is the whole phase. No SUPPORTED change.

**Verifier notes**: generality is exhausted after Refinement 4, so the remaining phases are perf. Re-measure 703 first: the new layout/placement/dtype paths must stay compiled out of 703's program (BB-18), so any regression against Refinement 3's number is fixed before the lever is tried. Same guard set and same-session A/B discipline as Refinement 3.

**Done when**: a measured device-ns win on 703 beyond the noise band with the golden suite green and no guard-set regression — or, if `/perf-ceiling-dm` places 703 within the noise band of its DRAM ceiling, that ceiling evidence and the guard-set re-measurement are recorded in `changelog.md`.
