# Verification Report: nomic_moe_dispatch

Verifier pass, 2026-11-04, Blackhole p300c (11×10 grid). The op is regime R1 `local_tilize_two_noc`. The output is ND-sharded in the op's own L1 layout. Each core gathers its shards' row pieces on two reader RISCs, one on each NoC, then fast-tilizes them into its own shard. There is no writer.

## Code Review

Fixed in place:

| # | Finding | Fix |
|---|---|---|
| 1 | **The k = 1 column split was slow** (perf lamp L4). The selection rule took the coarsest legal block, so in R ∈ [2816, 3520] ∪ [6336, 7040] it chose whole 24-tile rows. Each core then held 1 or 2 blocks, and its last 24-tile tilize could not overlap any read. R = 6368 is the model's 6×512 / 8×384 window: it ran at 15.0 us, barely under the incumbent's ~17 us. | Dropped k = 1 from `PIECES_PER_ROW_CANDIDATES`. k = 2 fits the L1 budget wherever k = 1 does (`ceil(2x)·12 ≤ ceil(x)·24`). Measured: **R = 3296 6.59 → 4.86 us (1.36×), R = 6368 14.97 → 7.47 us (2.0×)**. The other 46 R values are unchanged. This departs from the coarsest-block default, and the measurement justifies it. Recorded in `op_design.md` (amendment under the selection rule) and in `l1_ledger.md`. |
| 2 | **The ids scratch was sized to every block, not to its live set.** `cb_gather_ids_*` held `max_blocks_per_core + 1` pages. After the implementer's measured lookahead schedule, only `ids_lookahead + 1` blocks of ids are live at once. | It is now a ring of `ids_slots = min(max_blocks, ids_lookahead + 1)` pages plus 1 alignment page. That is 192 B instead of up to 896 B a reader. The slot that gets reused belongs to block j − ids_lookahead − 1, and all of that block's reads were issued before the fetch. `ids_slots` is a plan field and a CT arg, and a kernel `static_assert` bounds it. |
| 3 | **DRY: block-derived values were restated as literals.** The host passed `table_page_bytes = 768 * 2`. The kernel redefined `TILE_BYTES = 2048` (used for `slot_bytes`) and `IDS_ALIGN = 64` (a copy of the host's `IDS_ALIGN`). | The host now has `BF16_BYTES` and `ROW_BYTES = HIDDEN·BF16_BYTES`, and `piece_bytes` derives from `BF16_BYTES`. In the kernel, `slot_bytes = TILE_ROWS · piece_bytes` derives from the CT arg, `ids_align` is a CT arg from the host's single `IDS_ALIGN`, and `ID_BYTES = sizeof(uint32_t)`. |

Reviewed and found sound (no change):

- **Reader correctness.** One stateful one-packet read per row piece. The kernel polls `noc_cmd_buf_ready` explicitly before writing the coordinate, then issues with `skip_cmdbuf_chk`. `bank_to_l1_offset` is added on every read. The constant-divisor bank math is correct.
- **Trid race.** A block's trid is polled only after a read of a later block has left the command buffer (`pollable`). The most recent block uses the full read barrier, and the trid is cleared at exit.
- **CB credit balance.** Each reader pushes `w` pages per block, `num_blocks` times. Compute waits on and pops `w` pages on both aliased indices per block. The ids CBs are self-owned scratch and are never pushed.
- **Compute.** `compute_kernel_lib::tilize` (fast path) runs once per block with `WaitBlock` on head, and the caller waits on and pops tail. Init happens with the first block and uninit with the last. The design's zero-block InitOnly / UninitOnly calls would trip the helper's `ASSERT(num_blocks > 0)`, so the implementer folded them into the first and last blocks. That is correct; the friction note is filed in the breadcrumbs.
- **Kernel hygiene.** All kernels use `void kernel_main()`, `api/dataflow/dataflow_api.h` includes, and a `TensorAccessor` for gather_index. There is no `InterleavedAddrGen`.
- **Helper usage.** The tilize helper is used. The per-read path is raw NoC API, with the documented reason: every kernel_lib or `Noc` stateful-read helper fixes the source coordinate at set-state time. There is no multicast, so the multicast helper does not apply.
- **Host.** The plan is `lru_cache`d per R, and the CB, CT-arg and kernel parts are cached per (plan, accessor CT, config). Measured host time is about 60 us a call (50-call loop, no profiler). The ~0.9 ms that the perf harness's `op_window` shows is the real-time profiler's own overhead, not the op's.

### Prompt rules (`eval/prompts/nomic_moe_dispatch.txt` § Rules)

| Rule | Status |
|---|---|
| MUST be exactly one native `ttnn.generic_op` program, no ttnn op sequencing | ✓ one `generic_op`, nothing else on the path |
| MUST name kernels `nomic_moe_dispatch_{reader,compute}.cpp` (+ role suffix) | ✓ `_reader.cpp`, `_reader_noc1.cpp`, `_compute.cpp`; shared body in `_gather.hpp` |
| MUST pass io tensors in order `[table, gather_index, output]` | ✓ |
| MUST read gather_index on device; no host read; index values never CT args or cached state | ✓ only buffer addresses are (common) RT args; `test_new_inputs_each_call` / `test_new_tensors_between_calls` pass |
| A call allocates nothing but its output | ✓ (`allocate_tensor_on_device` only) |
| MUST write all R rows in `output_memory_config`'s layout | ✓ (`test_every_output_tile_is_written` with NaN poison passes) |
| MUST NOT change routed_combine / routing_metadata / grouped_matmul_w1 / other ops' kernels | ✓ the branch's op commits touch only op-local files |
| prefer: compute the per-shape plan once | ✓ followed |
| prefer: kernel-lib helpers over hand-written LLK | followed where expressible; the raw read path is justified (above) |

### Design conformance (`op_design.md`)

- **Algorithm and topology.** These match R1: a gather into an aliased two-producer slot, a fast tilize into the output-backed CB, and no writer.
- **Work distribution.** All 110 cores. Per-core work is derived on device from the logical core plus `num_shards` and `num_cores`, with zero per-core RT args. Both reader RISCs carry half the rows each, on separate NoCs. There are no output writes, so no half of the dataflow dribbles.
- **Blocking knobs.** `block_width_tiles`, `rows_noc0`, `row_slots`, `reads_in_flight`, `ids_lookahead`, `ids_slots`, `num_shards`, `num_cores` and `ids_align` are all host-plan CT args, each defined once. `ROW_TRIDS = 4` and `IDS_TRIDS_MAX = 8` are kernel-local hardware-trid ring caps (mechanism caps, not block knobs).
- **Split.** `tile_row` is split across cores in shard units (`SHARD_TILE_ROWS = 1`, fixed by the L1 budget), and each core's loop runs whole shards. The shard is the block, so there is no sub-shard re-chunking. Per-row reads sit inside `gather_rows_block`, with sync once per block. All three kernels are block-expressed.
- **Documented deviations.** Ids are fetched one block ahead (`ids_lookahead = 1`) instead of all at start. The implementer measured this: R = 8416 went 11.92 → 9.47 us. The tilize uses `WaitMode::WaitBlock` on head instead of `NoWait` after a caller wait, which is equivalent.

## Registry Conformance

- `INPUT_TAGGERS = {"window": tag_window}` uses the `(inputs, axes)` signature. `SUPPORTED` has `dtype`, `table_buffer`, `output_buffer` and `window`, which is every axis the op gates on. `EXCLUSIONS = []`.
- `validate()` checks SUPPORTED per axis first, then EXCLUSIONS, then the operand contract (shape, T and R range, layouts, 11×10 grid, CB budget). Every refusal is a `SupportRefusal`, and `validate()` is the first line of `nomic_moe_dispatch`.
- The op file does **not** declare `INVALID`.
- **INVALID audit** (`feature_spec.py`): `INVALID = []`, which is correct. Nothing in TARGET is structurally impossible. There is no layout axis, so the bf8b + ROW_MAJOR canonical entry does not apply. There is no weight tensor, so no canonicalization cells are needed.
- **XPASS / drift auto-fixes:** none needed.
- **L1 bank count.** The design listed "the L1 bank count == 110" as a validate() row. It is implied by the 11×10 grid check (one L1 bank a worker core), and the kernel's `NUM_L1_BANKS` is the build's own constant, so the table's interleave and the kernel always agree. No separate check was added.

## L1 Ledger Audit

- **Ledger currency.** The 5 CB rows match the 5 declared CB indices: rows head and tail in one aliased descriptor, the output-backed CB, and the two ids scratches. The ids rows, symbol table, total and budget rows were updated for fixes 1–2. The size expressions match `_rows_cb_bytes` / `_ids_cb_bytes`.
- **Capacity vs live set.** *Over:* the ids scratch was fixed in place (fix 2). The rows CB's capacity is `row_slots·w`, all slots resident. This is a stated pipelining decision (lamp L3) and stays. *Under:* none. The rows CB spans `tile_row` → `row_slots`, `column` → `w` and `row_in_block` → 32, and its capacity scales with all three.
- **Page format vs DEST.** Every tile page is Float16_b and `fp32_dest_acc_en = False`, so the formats agree. If a caller turns fp32 DEST on, the pages stay bf16, which is correct: rows arrive by dataflow (never through DEST) and the output ABI is bf16. Both non-default configs are measured bit-exact (below).
- **Disjoint lifetime with no justification.** None. Every CB lives for the whole kernel. The rows / output non-sharing is justified: an in-place tilize would overwrite rows still being unpacked.
- **Bounds.** R, k, w, `max_blocks_per_core`, `row_slots`, `ids_lookahead` and `ids_slots` are all bounded in the symbol table. The total is closed-form.
- **Block-size defaults.** The output is the op's own sharded layout, so the block is one whole shard, which holds. The coarsest legal shard width (k = 1) is now skipped, a measured departure (fix 1).
- **Per-core footprint.** `circular_buffer_bytes(R) = row_slots·w·2048 + 2·(ids_slots+1)·64` is ≤ 123,264 B in range (R ≥ 8096: 5·12 tiles). The rows term scales with `row_slots` and `w`, and the ids term only with `ids_lookahead`. The output-backed CB adds 0 B because it is the output tensor.
- **Data-movement budget.** It is present and matches the implemented split. Table pieces cross the NoC once (R·1536 B, half per NoC), the output crosses 0 times, and ids cost R·4·k B from one DRAM bank. The cheapest-traffic split is R1 itself. The one cheaper-traffic row, R7 `ids_sibling_forward`, is `deferred` with a measured reason and is now queued as perf **Refinement 3**.

## Precision Baseline

`tests/ttnn/unit_tests/operations/nomic_moe_dispatch/test_nomic_moe_dispatch_precision_baseline.py` uses uniform-random gather_index and a randn bf16 table. It checks with `assert_with_pcc` and `comp_allclose(rtol=0, atol=0)`, plus bit equality.

| Shape (T, R, k) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | max ULP | got/true ratio median [p5, p95] |
|---|---|---|---|---|---|---|
| T=1000, R=2240, k=3 (ragged) | 1.0 | 0 | 0 | 0 | 0 | 1.000000 [1.000000, 1.000000] |
| T=1536, R=3296, k=2 | 1.0 | 0 | 0 | 0 | 0 | 1.000000 [1.000000, 1.000000] |
| T=2048, R=4320, k=3 | 1.0 | 0 | 0 | 0 | 0 | 1.000000 [1.000000, 1.000000] |
| T=4096, R=8416, k=2 | 1.0 | 0 | 0 | 0 | 0 | 1.000000 [1.000000, 1.000000] |
| R=2272 / 8416, `dst_full_sync_en=True` (standard tilize) | 1.0 | 0 | 0 | 0 | 0 | 1.0 |
| R=2272 / 8416, `fp32_dest_acc_en=True`, HiFi4 | 1.0 | 0 | 0 | 0 | 0 | 1.0 |

**Assessment.** Bit-exact. As expected for a gather plus tilize with no arithmetic, the values do not depend on `compute_kernel_config`. There is no scale signature: the ratio is exactly 1.

**Recommended tolerances.** Bit equality (`view(int16)` compare). PCC = 1.0 and atol = rtol = 0 are implied.

## Verifier CLI Summary

Results: `ttnn/cpp/ttnn/operations/experimental/nomic_embed_text_v2_moe/nomic_moe_dispatch/verifier_results/verifier_report.json` (final run, after all fixes).

- supported_pass: 29 (11 `test_op` cells, 3 loose perf cases, 15 regression tests)
- xfail_expected: 33 (= 11 shapes × {table dram/output l1, table l1/output dram, table dram/output dram}, all properly refused by `validate()`)
- invalid_skipped: 0 (`INVALID = []`)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

Acceptance: 28/28. Precision baseline: 8/8. Both in default and `--dev` (watcher / NoC sanitizer / asserts) modes.

**TARGET − SUPPORTED → queue.** `table_buffer = dram` goes to Refinement 2. `output_buffer = dram` goes to Refinement 1. Refinement 2 covers the cell with both in DRAM. All 33 xfail cells map to a refinement.

## Device performance (device kernel ns per call)

Measured with the eval real-time profiler at the board's AICLK, not rescaled to 1350 MHz. Each value is the mean of 5 warm calls with 96 KB of every bank reserved, a real text's routing, table and Xg in L1. Harness: `tests/ttnn/unit_tests/operations/nomic_moe_dispatch/perf/test_nomic_moe_dispatch_perf_sweep.py`.

| R (window) | k / w / blocks | before verifier | after | incumbent | accept | stretch |
|---|---|---|---|---|---|---|
| 2240 | 3 / 8 / 2 | 4.50 | 4.54 | — | — | — |
| **2272 (2×512, guard)** | 3 / 8 / 2 | 4.60 | 4.58 | 8.97 | 6.5 ✓ | 4.5 ✗ (+0.08) |
| 2528 | 4 / 6 / 3 | 5.80 | 5.82 | — | — | — |
| 3296 | 1→2 / 24→12 / 1→2 | 6.32 | **4.87** | — | — | — |
| **4320 (4×512, guard)** | 3 / 8 / 4 | 6.99 | 6.99 | 13.03 | 9.0 ✓ | 6.5 ✗ (+0.49) |
| 5344 | 3 / 8 / 5 | 9.11 | 9.04 | — | — | — |
| 6368 | 1→2 / 24→12 / 2→4 | 15.01 | **7.51** | — | — | — |
| 7392 | 4 / 6 / 9 | 14.91 | 14.88 | — | — | — |
| 8064 | 3 / 8 / 7 | 12.37 | 12.30 | — | — | — |
| 8224 | 2 / 12 / 5 | 9.58 | 9.53 | — | — | — |
| **8416 (8×512, PERF FOCUS)** | 2 / 12 / 5 | 9.66 | 9.62 | 21.85 | 15.0 ✓ | 11.0 ✓ |

**Lever measurements** (knob sweep `perf/test_nomic_moe_dispatch_knob_sweep.py`; ablation by a temporary kernel stub, reverted):

| Lever | R = 2272 | R = 4320 | R = 8416 | Reading |
|---|---|---|---|---|
| `reads_in_flight` 4 / 8 / 12 / 16 | 5.17 / 4.62 / 4.61 / 4.54 | 8.01 / 6.95 / 7.01 / 6.95 | 10.36 / 9.64 / 9.63 / 9.67 | 8 is the knee. Lamp L2 is flat above it, so it is not a lever. |
| `ids_lookahead` 1 / 2 / 4 | 4.62 / 4.65 / 4.55 | 6.95 / 7.91 / 8.15 | 9.64 / 11.11 / 12.10 | More ids in flight queues the one gather_index DRAM bank harder. |
| ids fetch stubbed (local synthetic ids) | 2.88-2.96 | 4.44-4.52 | **12.76-14.34 (slower)** | At R ≤ 4320 the ids fetch is ~1.7-2.5 us of critical path. At R = 8416 it is effectively staggering the grid's first row reads: without it, 110 cores × 2 RISCs burst at once and the NoC / banks congest. |

## Recommendations

- **Perf queue (op_requirements.md):** two perf refinements, both on the flagged R = 8416 shape. Refinement 3 is the ids-fetch topology (R7 or a read-once broadcast) together with an explicit start stagger. Refinement 4 bundles the NoC-contention levers: lamp L1 path-aware row-to-NoC split, lamp L6 issue order, and lamp L3 / L7 co-tunes. The ablation shows why R7 alone is not enough at R = 8416: removing the ids latency without a stagger regressed by 3-5 us.
- **Guards above stretch.** The R = 2272 and 4320 guards are above their stretch targets (by 0.08 and 0.49 us) and well under acceptance. The ids ablation says the ids fetch is their gap, so Refinement 3 should close it. It is a no-regression guard there, not a target.
- **Budget-forced fine splits (no lever in scope, so this is a note, not a refinement).** Per-row cost tracks k: k = 2 ≈ 1.14 ns/row, k = 3 ≈ 1.5-1.7, k = 4 ≈ 2.0. At R = 7392 (7×512) the +8 KB L1 plan forces k = 4: 14.9 us against an interpolated incumbent of ~19.6 us. At R = 8064 it forces k = 3: 12.3 us. Both still beat the incumbent. Lifting them to k = 2 needs an output layout outside one uniform-round-robin ND shard spec: mixed shard widths, or a per-core shard count that tracks the budget exactly. That is a design change against the model's L1 plan, so it is recorded here only.
- **Hot-row pattern** (`test_hot_row`: every id = 0, one bank serves every read) takes 29 us. That is correctness-only by spec, not a perf case.
- **Acceptance-test comment drift.** The immutable acceptance test still labels `(1536, 0)` as "k = 1". R = 3296 now runs k = 2, and no R in range selects k = 1 any more. The test asserts only the layout's L1 bound, so it still passes. The comment needs no code change.
- **Tilize helper friction** (breadcrumbs): `InitOnly` / `UninitOnly` with 0 blocks trips `ASSERT(num_blocks > 0)`. Planners should stop prescribing the zero-block form.
