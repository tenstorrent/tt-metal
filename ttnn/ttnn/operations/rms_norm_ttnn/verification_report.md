# Verification Report: rms_norm_ttnn

Phase-0 verification of the derivative-of-seed implementation. Device:
**Blackhole p150b, 11x10 = 110-core compute grid, measured AICLK 1350 MHz**
(profiler preamble `ARCH: blackhole, CHIP_FREQ[MHz]: 1350`) — the same clock
`feature_spec.py`'s perf references are stamped at, so every ratio below has a
clock-scale factor of exactly 1.0000.

**Headline**: the op is in unusually good shape. `SUPPORTED` already equals
`TARGET` on all nine axes, `EXCLUSIONS` is empty, the full 120 960-cell golden
cartesian runs with **zero op-attributed failures**, and 16 of the 19 perf-group
targets already beat their reference latency. The refinement queue is therefore
all-perf, and it is built from measured device-ns rather than from the
`TARGET - SUPPORTED` gap (there is none).

---

## Code Review

### Fixed

| # | Finding | Fix |
|---|---|---|
| 1 | **`_check_per_channel` raised the wrong exception class.** A per-channel operand at a dtype outside the accepted set raised `UnsupportedAxisValue` (a `NotImplementedError` subclass). `eval/prompts/rms_norm_ttnn.txt` "## Validation" requires **ValueError or RuntimeError** for that case, and `test_validation.py::_REFUSED` is `(ValueError, RuntimeError)` — so `test_refuses_per_channel_dtype_outside_the_accepted_set` could not pass. | Changed to `ValueError`, with the two refusals' *different kinds* documented at the raise: a dtype outside the op's **kernel-capability** set is an input-contract violation (ValueError); a dtype inside it but outside `SUPPORTED["gamma_dtype"]` is a registry support refusal (`UnsupportedAxisValue`, for the xfail-strict gate). |
| 2 | **The two refusals could silently swap** if a future refinement narrowed `SUPPORTED["gamma_dtype"]`, because `_check_per_channel` runs *before* `validate()`'s SUPPORTED loop. That would surface as `xfail_wrong_mode`. | Added an import-time assertion that `PER_CHANNEL_DTYPES` is a **superset** of `SUPPORTED["gamma_dtype"] - {"none"}`, so any value the registry axis still lists must clear the capability gate to reach the loop that is supposed to refuse it. |
| 3 | **`l1_ledger.md`'s data-movement budget over-counted the reuse-shared operands by 16x.** Every per-channel figure in it (and in `op_design.md`'s `GAMMA_MCAST` row and D14: "gamma is 118 MB", "ROW_RESIDENT cuts it to 50 MB") is priced at **whole tiles**, but D23 trims a TILE-layout per-channel read to **two face-rows** — `2 * TILE_DIM * elem_bytes` = 128 B of a 2048 B bf16 tile. | Added a correction note to the traffic table changing the term to `Sg'` (bytes actually fetched) with the per-format factor spelled out. Consequence recorded below: it makes `GAMMA_MCAST`'s deferral *more* clearly right, not less. |
| 4 | **Stray `</content>` end-tags** left in `l1_ledger.md:244` and `op_design.md:515` — a copy artifact rendering as literal text. | Deleted. |
| 5 | **`PROPERTIES["multi_core"]` was tagged `"declared"`** although the evidence exists. | Upgraded to `"verified"`, naming the evidence: 23 340 golden cells carry a measured `device_num_cores` (the core count of the dominant program in the op's own profiler window — precisely what `eval/op_template.py` names as the "verified" standard), max 110 = the whole grid. |
| 6 | **`D3`'s reasoning for `ReduceFp32Mode::Fast` was an argument, not a measurement**, and the alternative it waves off is actively dangerous. | Replaced with the measured A/B (below) and an explicit warning that the failure mode at the op's own default is a *silent* inf. |

### Checked and correct — no change needed

- **Registry conformance.** `INPUT_TAGGERS` (both taggers take `(inputs, axes)`), `SUPPORTED` (every gated axis present), `EXCLUSIONS` (empty), `validate()` (SUPPORTED per-axis, then EXCLUSIONS, both raising from `ttnn.operations._op_contract`). `rms_norm_ttnn()`'s first statement is `validate(...)`. The op file does **not** declare `INVALID`. `PROPERTIES` is the sanctioned optional block (`eval/op_template.py:131`).
- **One dispatch per invocation.** Exactly one `ttnn.generic_op(tensors, program_descriptor)`; the zero-volume and rank-0 paths are alternative descriptors inside the same single dispatch, never a second op.
- **No host-side tensor manipulation.** Grepped for `ttnn.to_layout / tilize / untilize / pad / slice / to_memory_config / reshape / permute / typecast` across the op's Python — **none**. Both layouts are native.
- **Kernel hygiene.** `void kernel_main()` in all three kernels; `api/dataflow/dataflow_api.h` include form; `TensorAccessor` throughout (no `InterleavedAddrGen`); no raw `noc_async_write_multicast` + semaphore handshake — the combine's transport is `mcast_pipe.hpp`'s `SenderPipe`/`ReceiverPipe`.
- **Helper usage.** `ckl::reduce<>` with explicit `ReduceInputPolicy` / `ReduceAlgorithm` / `Accumulate`, `ckl::eltwise_chain` with `IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK)`, `ckl::tilize` / `untilize`, `transform_in_place`. The one raw-LLK path (`StatFinalize`, a user-defined element on the documented `UnaryOp<Derived, Slot>` CRTP surface, wrapping `_calculate_sqrt_body_`) is a sanctioned custom block helper built because no stock element exposes a `VectorMode` seam — mechanism choice, not a conformance issue.
- **Prompt `## Rules` (all MUST / MUST NOT).** Seed extension with A1–A13 recorded; operand-free builds byte-identical to the seed (asserted structurally by `test_program_is_structurally_the_seeds`, 72 cells over 14 scheme-spanning geometries — CB set *and* kernel args); one device program per call at every added capability; ROW_MAJOR native; `subblock_w` honoured and refused below 1; `block_h`/`block_w` checked as a restatement then geometry taken from the shard; config-variant mismatch refused naming both; `inplace` returns the input **object** and refuses a disagreeing `memory_config`; `packer_l1_acc` accepted and read by nothing (deliberately absent from `_COMPUTE_CONFIG_FIELDS`); `compute_with_storage_grid_size` range-checked but not a placement contract; host-resident input raises at the door; mixed per-channel dtype is numerically correct (each CB declares the dtype of the tensor it carries). **No violations.**

### Advisory (soft guidance, not blocking)

- **Reader/writer transaction granularity vs. the design.** `op_design.md`'s block-schedule table states the intent as "**one NoC barrier per (block, chunk) per stream** on the TILE path". The shipped reader issues one barrier per **tile-row** of the chunk (`WT_CHUNK` tiles), and the writer mirrors it. This is a documented decision at the code (">= 4 tiles per barrier whenever the block allows it") and it is a real trade — a coarser unit costs reader↔compute overlap *inside* a block — so it is not a defect. It is a measurable lever and is filed as such (Refinement 3, lever 3), not silently "fixed".
- **`_cb_block_mult` prices `cb_x_squared` at the full chunk width** even when D12's DEST square fold makes the CB `BLOCK_ROWS x 1`. The over-charge is conservative (it can only shrink `BLOCK_ROWS`, never overflow L1) and is inert in practice: the fold is only taken at `WT_CHUNK <= 8`, where the whole per-core slice fits many times over and `BLOCK_ROWS` is capped by the assignment rather than by L1. Folded into Refinement 3's scope rather than changed here, because it moves the blocking decision on real shapes and needs the same measurement the rest of that phase takes.

---

## Registry Conformance

- **Confirmed present and correctly wired**: `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `validate()`. Confirmed the op file does **not** declare `INVALID`.
- **No SUPPORTED auto-fixes were needed** — `xpass_drift` is 0, so there is no under-claim to promote, and `xfail_wrong_mode` is 0, so every refusal that fires is the right kind.
- `TARGET - SUPPORTED` is **empty on every axis** (dtype, fp32_dest_acc_en, layout, alignment, rank, gamma_mode, gamma_dtype, gamma_layout, memory_layout), so `xfail_expected` is 0 by construction. There is no xfail bucket and therefore no queue gap.

### INVALID audit (`eval/golden_tests/rms_norm_ttnn/feature_spec.py`)

The first 21 entries are **well-formed**: the two block-format entries (`bfloat8_b x ROW_MAJOR`, once for the activation and once for the per-channel operand) are single-tensor and pass the universe-must-change test; the 19 `gamma_mode` <-> `"none"`-sentinel entries are the sanctioned canonicalization-of-redundant-cells exception, coupled both ways so exactly one canonical `("none","none")` cell survives. The canonical bf8b+ROW_MAJOR activation entry is present. Norm-like no-weight canonicalization is present.

**Five entries are misclassified**, and `feature_spec.py` says so itself under the heading *"Author-scoped exclusions ("for now", NOT structural impossibility) ... deliberately parked in INVALID to keep them out of the refinement backlog"*:

```
{layout: ROW_MAJOR, memory_layout: HEIGHT_SHARDED, gamma_layout: TILE}
{layout: ROW_MAJOR, memory_layout: WIDTH_SHARDED,  gamma_layout: TILE}
{layout: ROW_MAJOR, memory_layout: BLOCK_SHARDED,  gamma_layout: TILE}
{dtype: bfloat8_b, alignment: w_non_aligned}
{dtype: bfloat8_b, alignment: h_non_aligned}
```

Three defects, in ascending order of consequence:

1. **They fail sanity rule 2 (universe-must-change).** By their own comment they encode "not supported for now", which is `EXCLUSIONS` (op file), not `INVALID` (test suite). The distinguishing question — "what would have to change?" — answers "a kernel improvement" for all five.
2. **The first three fail sanity rule 1 (single-tensor coupling).** They cross the *activation*'s `layout` and `memory_layout` with the *weight*'s `gamma_layout`. Those axes describe different tensors and there is no documented kernel-level coupling between them: the activation's placement selects the BAND / native-shard reader, while the per-channel operand's layout selects an entirely separate reader branch. This is the canonical authoring mistake the registry model names explicitly.
3. **All five hide capability the op already claims and already delivers.** `SUPPORTED` lists every one of those axis values, `validate()` accepts them, and `eval/prompts/rms_norm_ttnn.txt`'s Phase-0 list *requires* them ("dtype: float32, bfloat16, bfloat8_b"; "alignment: tile_aligned, w_non_aligned, h_non_aligned"; "layout: TILE and ROW_MAJOR, both native ... at every memory placement the op accepts"). Because `INVALID` cells are `skip`ped, the golden suite never calls the op for any of them, so the claim goes completely untested.

**Verified directly.** `tests/ttnn/unit_tests/operations/rms_norm_ttnn/test_rms_norm_ttnn_invalid_audit.py` runs all five regions and PCC-gates them: **16/16 pass**, including RM-activation x each of the three sharded schemes with a TILE weight, and bf8b at `(1,1,32,50)`, `(1,1,64,17)`, `(1,1,47,64)`, `(1,1,17,128)`, `(1,1,47,50)` with and without a weight. The bf8b cases compare against the *read-back* input so the input quantization is not charged to the op, and they pin the RMS denominator to the **logical** width.

**Requested `feature_spec.py` edit (not made by the verifier — please apply via `/golden-tests` or directly):** delete all five "author-scoped" entries. They are supported, they pass, and removing them adds real coverage. If any is later found genuinely unwanted, its home is `EXCLUSIONS` in the op file, where XPASS-strict keeps it honest.

**Two further `INVALID` candidates**, forwarded from `op_design.md`'s "Structural impossibilities" section and endorsed here — both genuinely structural, neither currently reachable, so adding them is optional housekeeping rather than a coverage fix: `{rank: 0, layout: TILE_LAYOUT}` / `{rank: 1, layout: TILE_LAYOUT}` (no second-to-last dimension means no tile grid), and a rank-0 blocked `(Wt, 32)` per-channel operand (indistinguishable from a flat rank-2 operand under the detection rule).

---

## Design Conformance

Checked against `op_design.md`'s Blocking Model, which is unusually complete (every axis carries a decision; every deferred regime carries a positive reason).

| Dimension | Verdict |
|---|---|
| **Algorithm** | Conformant. Two-pass reduce, `t = x + r` materialized once into `cb_x_sum` and read by both passes, `1/W` applied in fp32 by the finalize and never folded into the bf16 scaler, epsilon inside the denominator so a cancelled row yields zero rather than NaN. `use_welford` refused, as the contract requires. |
| **Data pipeline topology** | Conformant. Reader on NoC0 owns x / residual / per-channel staging; writer on NoC1 owns the whole cross-core combine (gather -> root -> multicast) so the two halves do not contend; compute owns tilize / add / square / reduce / finalize / normalize / scale / bias / untilize. |
| **Parallelization / grid fill** | Conformant, and **measured**. `split_work_to_cores(_core_range_set_full_grid(device), Rt, True)` — the full grid, `row_wise=True` (master.md's `noc_placement` reports the `False` default is 2.2–2.9x slower). Measured occupancy over 23 340 cells: max **110 = the whole grid**; `1x1x2048x256` median 64 / max 110; `4x1x512x512` median 64 / max 110; `1024x1024` median 32; `128x8192` median 44 / max 110. The 20.4% of cells at one core are the small shapes whose row axis is a single tile-row and whose width split is correctly gated off by `WIDTH_SPLIT_MIN_WT_PER_CORE`. No idle-silicon defect. |
| **Inter-core communication** | Conformant. 2-level slot tree gather + `mcast_pipe` broadcast, one arrival semaphore per level (a level-1 sender can legally arrive before one of the root's own level-0 members, so a single cumulative counter would be wrong — the code gets this right). |
| **Blocking-model fidelity** | Conformant, and exemplary. Every knob the planner named is a live parameter with a **single** definition: `BLOCK_ROWS` / `WT_CHUNK` / `NUM_W_CHUNKS` / `X_RESIDENT` / `x_hold_wt` all come out of one `_solve_blocking()` return tuple and are passed to the kernels as compile-time args, which never re-derive them. `_cb_block_mult()` and `_per_channel_bytes()` are the single source for "which CBs scale with the block, at what depth", called by **both** L1 solves so the fit predicate and the chunk-size solve cannot drift. Depth is *searched* over `CB_DEPTH_CANDIDATES` rather than fixed. No CB is sized to a whole-op dimension: every page count is a function of `BLOCK_ROWS`, `WT_CHUNK`, `x_hold_wt` or a depth constant. `WT_CHUNK == Wt` in the RESIDENT regime is the sanctioned residency fast-path (predicate-guarded, with ROW_RESIDENT and STREAM as the streaming fallbacks), not a collapsed knob. |
| **DRY** | No duplicated block literals found. `PASS_B_BLK` is the one knob computed kernel-side, and it is computed **once** (`pass_b_blk(WT_CHUNK, DEST_AUTO_LIMIT)` when the host CT arg is 0) and used everywhere from that one `constexpr`; no CB is sized by it, so no host/kernel duplication exists. |
| **Expression — does the schedule read as blocks?** | Compute: yes, unambiguously — every pass-B stage is one `eltwise_chain` over `IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK)`, so the block factor controls the live set, the command issue *and* the synchronization grouping. Reader/writer: the block factor controls the enclosing trip count and the CB live set, but the completion boundary is per **tile-row of the chunk**, not per block — see the advisory above. That is `WT_CHUNK` tiles per barrier (>= 4 on any real shape), well above master.md's whole-tile floor, so it is not the "handshaking per unit" failure the rule targets; it is a granularity choice, and it is filed as a measured lever. |
| **Half-turned split** | Not present. The per-core compute loop walks `BLOCK_ROWS x WT_CHUNK`, not one tile at a time; on a shard the block **is** the shard (`assert sw_t == wt_chunk` for the native-in CB), so there is no sub-shard re-chunking. |

---

## L1 Ledger Audit

`l1_ledger.md` is current and complete. Five checks:

1. **Ledger currency — PASS.** All 24 CBs the descriptor can declare have a row (the 19 seed CBs plus the five this op adds: `cb_residual_sticks`, `cb_residual_tiles`, `cb_x_sum`, `cb_bias_sticks`, `cb_bias_tiles`). Every size expression was checked against the allocation: `cb_input_tiles = DX * BR * (HAS_R ? WC : XH)`, `cb_x_sum = HAS_R * BR * XH`, `cb_normalized = ND * BR * WC`, `cb_row_stat = !CMB * SD * BR`, `cb_bank = CMP * BR`, `cb_gamma_sticks = PS` — all match the code exactly.
2. **Capacity vs live set, both directions — PASS.** Every over-capacity gap is named and justified (`DX`/`DR`/`DO`/`DS` double buffering; `FD` one-round-in-flight; `SD` on `cb_row_stat`/`cb_row_final` is D6's **correctness** ring-rotation floor, not overlap). No CB whose live set *spans* an axis has a capacity that fails to scale with it: the only apparent counterexample, `cb_partials_gathered`, genuinely *streams* over `row` since D27 replaced a `GS x BR` ring with one compact tile.
3. **Page format vs DEST width — PASS with a documented, correct override.** Three CBs carry `Float32` pages while `fp32_dest_acc_en` may be off (`cb_row_stat` / `cb_sum_handoff` / `cb_row_final`) and the reason is a *round-trip* property, not a DEST property: they are the cross-chunk and cross-core accumulators the reduce **reloads**, so a 16-bit page would make the reload lossy at exactly the widths this op exists for. That is the sanctioned "concrete mechanism reason". Every other CB carries the dtype of the tensor it holds, which is what makes the X-09 mixed-dtype cell correct rather than merely tolerated.
4. **Disjoint lifetime with no justification — PASS.** The one disjoint pair, `cb_x_squared` (pass A) and `cb_normalized` (pass B), is deliberately **not** shared, and the reason is recorded at the row: D25's cross-block pipeline issues block `b+1`'s pass A before block `b`'s combine, which makes them concurrent. That is the sanctioned "concurrent because of a pipelining decision".
5. **Bounds and closed form — PASS.** Every symbol in the footprint expression appears in the symbol table with a bound and its establishing predicate, and `arena_bytes` is closed-form in them. The op dimensions that appear (`XH = wt_per_core` in the RESIDENT regime) are predicate-guarded with a streaming fallback, which is the sanctioned residency fast-path rather than an unbounded term.

**Data-movement budget**: present, and consistent with the split the code implements (x/residual/output at 1 DRAM crossing under ROWS·RESIDENT/ROW_RESIDENT, 2 under STREAM with the residual doubling the penalty; per-channel at `C` crossings under the row split and `gh` under the width split). The cheapest-traffic split (`row` across the grid + `width` across a group, which ties on activations and *beats* the row split on the reuse-shared operands) is **implemented**, gated by a `WIDTH_SPLIT_MIN_GAIN = 4` threshold that was *measured* (`MIN_GAIN = 2` produced a 0.92x regression on `(1024,1024)`). Occupancy is explicitly not treated as sufficient — the ledger's own "Occupancy was not treated as sufficient" section addresses this directly.

**One ledger correction applied** (item 3 in Code Review): the traffic budget priced the reuse-shared per-channel term at whole tiles and therefore over-counted it by the D23 trim factor — 16x at bf16 TILE (128 B of a 2048 B tile), 2x at bf8b, 1x for a ROW_MAJOR operand. Corrected in place. **Consequence**: on `(1,1,8192,7168)` ROW_RESIDENT the operand term is ~3 MB against 234 MB of x+out (~1%), not the 50 MB the ledger claimed (~18%). This is what removes `GAMMA_MCAST` from the refinement queue — a broadcast that eliminates 1% of DRAM bytes cannot pay for an injector, so the design's deferral is right for a stronger reason than it states.

**Block-size defaults**: held. Interleaved spreads the split's work units across the full grid and then takes the coarsest block that fits (`min(max_rows, brmax)`); sharded takes the shard as one block, clamped to the declared mechanism caps (`BLOCK_ROWS <= TILE_DIM` on a combine path — D27's compact transpose packs a tile-row's stat into one *column* of one tile, and a 101-row block measured pcc 0.949). Departures from the coarsest block (ROW_RESIDENT, STREAM, the BAND depth/narrow-staging ladder) are taken only after the budget refuses the coarser option, and the inventory is minimized first — D30's narrow per-channel staging exists precisely because the wide form reserved 25 fp32 tiles (100 kB) to carry a 3 200 B stick.

**Per-core footprint**: closed-form in `l1_ledger.md` "Total per-core footprint". `BR` moves every activation CB plus the three fp32 accumulators; `WC` moves the stick and streaming CBs; `XH` moves the three held buffers; the combine ring is `O(G)` tiles and **flat in `BR`** since D27.

---

## Precision Baseline

`tests/ttnn/unit_tests/operations/rms_norm_ttnn/test_rms_norm_ttnn_precision_baseline.py` — 32 cells, all passing. Relative RMS is `||got - true||_2 / ||true||_2`; the ratio columns are `r = got/true` over finite, non-negligible reference elements (the scale-bug detector).

### No operands, `randn` input, HiFi4 / `math_approx_mode=True`

| Shape | dtype | `fp32_dest_acc_en` | PCC (gate) | Max abs | Mean abs | Rel RMS | r median | r std |
|---|---|---|---|---|---|---|---|---|
| (1,1,32,64) | float32 | True | pass (0.999) | 7.51e-03 | 5.86e-04 | **8.72e-04** | 1.000201 | 8.69e-04 |
| (1,1,128,512) | float32 | True | pass | 6.10e-03 | 5.08e-04 | 7.52e-04 | 1.000581 | 5.09e-04 |
| (1,2,256,1024) | float32 | True | pass | 6.95e-03 | 5.37e-04 | 7.90e-04 | 1.000631 | 5.08e-04 |
| (1,1,32,4096) | float32 | True | pass | 6.48e-03 | 4.85e-04 | 7.23e-04 | 1.000543 | 5.06e-04 |
| (1,1,32,64) | float32 | False | pass (0.995) | 2.67e-02 | 2.58e-03 | **4.06e-03** | 1.001480 | 3.73e-03 |
| (1,1,128,512) | float32 | False | pass | 3.45e-02 | 2.31e-03 | 3.60e-03 | 1.000815 | 3.51e-03 |
| (1,2,256,1024) | float32 | False | pass | 5.53e-02 | 2.24e-03 | 3.52e-03 | 1.000682 | 3.44e-03 |
| (1,1,32,4096) | float32 | False | pass | 3.54e-02 | 2.19e-03 | 3.43e-03 | 1.000301 | 3.40e-03 |
| (1,1,32,64) | bfloat16 | True | pass (0.995) | 1.56e-02 | 5.36e-04 | 1.90e-03 | 1.000000 | 1.82e-03 |
| (1,1,128,512) | bfloat16 | True | pass | 1.56e-02 | 4.98e-04 | 1.87e-03 | 1.000000 | 1.76e-03 |
| (1,2,256,1024) | bfloat16 | True | pass | 1.56e-02 | 5.05e-04 | 1.89e-03 | 1.000000 | 1.79e-03 |
| (1,1,32,4096) | bfloat16 | True | pass | 1.56e-02 | 6.63e-04 | 2.13e-03 | 1.000000 | 1.96e-03 |
| (1,1,32,64) | bfloat16 | False | pass | 3.13e-02 | 2.13e-03 | 4.07e-03 | 1.000000 | 3.45e-03 |
| (1,1,128,512) | bfloat16 | False | pass | 3.13e-02 | 1.69e-03 | 3.54e-03 | 1.000000 | 3.46e-03 |
| (1,2,256,1024) | bfloat16 | False | pass | 3.13e-02 | 1.55e-03 | 3.34e-03 | 1.000000 | 3.26e-03 |
| (1,1,32,64) | bfloat8_b | True | pass (0.99) | 3.91e-02 | 8.36e-03 | 1.09e-02 | 1.000000 | 7.23e-02 |
| (1,1,128,512) | bfloat8_b | True | pass | 3.91e-02 | 8.11e-03 | 1.07e-02 | 1.000000 | 8.11e-02 |
| (1,2,256,1024) | bfloat8_b | True | pass | 4.69e-02 | 7.99e-03 | 1.06e-02 | 1.000000 | 8.19e-02 |
| (1,1,32,4096) | bfloat8_b | True | pass | 3.91e-02 | 7.58e-03 | 1.00e-02 | 1.000000 | 8.02e-02 |

### With operands, bfloat16, the op's **default** compute config (HiFi4 / approx / 16-bit DEST)

| Shape | Mode | Max abs | Mean abs | Rel RMS | r median | r std |
|---|---|---|---|---|---|---|
| (1,1,128,512) | gamma | 9.38e-02 | 2.07e-03 | 4.59e-03 | 1.000000 | 4.25e-03 |
| (1,1,32,4096) | gamma | 6.25e-02 | 1.85e-03 | 4.37e-03 | 1.000000 | 4.22e-03 |
| (1,1,128,512) | gamma_bias | 6.25e-02 | 2.54e-03 | 3.98e-03 | 1.000000 | 1.92e-02 |
| (1,1,32,4096) | gamma_bias | 9.38e-02 | 2.26e-03 | 3.73e-03 | 1.000000 | 1.72e-02 |
| (1,1,128,512) | residual | 3.13e-02 | 1.98e-03 | 3.90e-03 | 1.000000 | 3.73e-03 |
| (1,1,32,4096) | residual | 3.13e-02 | 2.09e-03 | 4.04e-03 | 1.000000 | 3.92e-03 |
| (1,1,128,512) | gamma_bias_residual | 1.25e-01 | 2.64e-03 | 4.14e-03 | 1.000000 | 1.73e-02 |
| (1,1,32,4096) | gamma_bias_residual | 9.38e-02 | 2.49e-03 | 3.99e-03 | 1.000000 | 1.76e-02 |

**Assessment.** Precision is dtype-determined and shape-independent — rel-RMS is flat across a 64x width range at every dtype, which is what a correctly blocked reduce with an fp32 cross-chunk accumulator should look like. The precision surface is a clean **two-axis** model: `fp32_dest_acc_en` is worth ~4.7x on float32 (8.7e-04 -> 4.1e-03) and ~2.1x on bfloat16, exactly as `feature_spec`'s `TOLERANCE_OVERRIDES[(float32, False)]` anticipates. **No scale bug anywhere**: `r median` is 1.000000–1.00063 on every one of the 32 cells and `r std` is always the same order as the rel-RMS, i.e. a broad spread centred on 1.0 (rounding noise), never a tight cluster at a non-1.0 constant. bfloat8_b's larger `r std` (7–8e-02) with `r median` exactly 1.000000 is the expected shared-exponent block quantization, not a systematic shift.

**Recommended tolerances** — the shipped `helpers.TOLERANCES` / `TOLERANCE_OVERRIDES` bands are correct and should not be tightened:

| Cell | PCC | Rel RMS | Headroom observed |
|---|---|---|---|
| float32, `fp32_dest_acc_en=True` | >= 0.999 | <= 0.02 | 23x |
| float32, `fp32_dest_acc_en=False` | >= 0.995 | <= 0.04 | 9.8x |
| bfloat16 (either) | >= 0.995 | <= 0.04 | 9.4x |
| bfloat8_b (either) | >= 0.99 | <= 0.10 | 8.6x |

---

## Verifier CLI Summary

Full cartesian, all 35 `INPUTS` shapes plus every loose group. Run in 18 shards
(`scripts/verifier_golden_shards.sh`) and merged by
`scripts/verifier_merge_golden_shards.py` — the whole directory does not fit one
tool call, see "Harness findings" below. Merged into **121 438 unique tests =
the complete collected set**.

```
python3 -m eval.verify_supported generated/verifier_results ttnn.operations.rms_norm_ttnn
```

| Category | Count | |
|---|---:|---|
| `supported_pass` | **23 325** | ✓ |
| `invalid_skipped` | 97 840 | ✓ |
| `infeasible_skipped` | 231 | ✓ uncharged (shard geometry vs. this device's L1) |
| `no_axes_found` | 32 | the 21 `test_regression.py` numerics tests + the 11 `test_validation.py` refusal tests — outside the registry grid by design |
| `xfail_expected` | 0 | ✓ expected: `SUPPORTED == TARGET` and `EXCLUSIONS` is empty, so no cell is out of the rectangle |
| **`xpass_drift`** | **0** | ✓ |
| **`xfail_wrong_mode`** | **0** | ✓ |
| `supported_marked_xfail` | 0 | ✓ |
| **`supported_fail`** | **10** | ✗ — **all 10 are harness defects, zero op-attributed**; see below |

### The 10 `supported_fail` cells — every one is harness-side, independently verified

| Count | Cells | Failure | Root cause |
|---|---|---|---|
| 3 | `1x1x0x64`, `0x64`, `1x1x32x0` (the zero-volume loose group) | `RuntimeError: max(): Expected reduction dim to be specified for input.numel() == 0` | `eval/metrics.py::compute_metrics_torch` calls `torch.max()` on the readback. The op **already returned a correct empty tensor** — `check_output` validated its shape, dtype and layout first and they all passed; the crash is in the metric, after the op. Verified independently: `torch.max(torch.zeros(0))` raises this exact message. |
| 7 | the whole `program_config` loose group (`1x1x384x768`, `1x1x64x1536`, `1x1x256x1024`, `1x1x64x1024`) | `AttributeError: 'CoreRange' object has no attribute 'end_coord'` | `eval/golden_tests/rms_norm_ttnn/program_config.py:67-68` reads `bbox.end_coord` / `bbox.start_coord`; this ttnn build exposes `.start` / `.end`. Verified independently: `dir(ttnn.CoreRange(...))` = `['contains', 'end', 'from_json', 'grid_size', 'start', 'to_json']`. The stand-in config object is never built, so **the op is never called** — the entire config-consuming surface goes untested while reading as 7 op failures. |

The same `CoreRange` defect additionally fails **3 of the 11** `test_validation.py`
refusal tests (`test_refuses_subblock_w_below_one`,
`test_refuses_sharded_config_against_an_interleaved_input`,
`test_refuses_memory_config_disagreeing_under_inplace`) — all three use the
`sharded_input` fixture, and again the op is never reached. The op's own
equivalents of all three refusals are exercised and green in
`test_rms_norm_ttnn_debug.py`.

**Requested harness fixes** (not made by the verifier — these are shared eval
infrastructure, and both changes affect scoring for every op):

1. `eval/golden_tests/rms_norm_ttnn/program_config.py:67-68` — `bbox.end_coord.x` -> `bbox.end.x`, `bbox.start_coord.x` -> `bbox.start.x` (and `.y`). Recovers 7 loose cells + 3 validation tests.
2. `eval/metrics.py` `compute_metrics_torch` — guard the zero-element readback before `torch.max()`. Recovers 3 loose cells and fixes every op with a zero-volume case.

### The 6 red `test_regression.py` numerics cells — a scoring gap, not an op defect

`test_positive_only_input` and `test_negative_only_input`, at all three shapes,
fail with `severity=precision`:

| Test | PCC | rel RMS (as scored) | target |
|---|---|---|---|
| `test_positive_only_input[32x64]` | 0.999902 | 0.016671 | 0.01 |
| `test_positive_only_input[64x128]` | 0.999910 | 0.017828 | 0.01 |
| `test_positive_only_input[128x256]` | 0.999924 | 0.018925 | 0.01 |
| `test_negative_only_input[32x64]` | 0.999902 | 0.016491 | 0.01 |
| `test_negative_only_input[64x128]` | 0.999909 | 0.017824 | 0.01 |
| `test_negative_only_input[128x256]` | 0.999923 | 0.018904 | 0.01 |

Triaged per the scale-vs-precision protocol, and it is **precision, not scale**:

- **`r` is not a tight non-1.0 cluster.** Measured on the exact failing distributions: `r_median = 1.00263`, `r_std = 3.87e-03` at `(1,1,32,64)`. The spread is *larger* than the offset, which is rounding noise with a small bias, not a uniform scale error. (A uniform scale would also force `max_abs / max|y| == median_abs / median|y|`; measured they differ by ~4x.)
- **The cause is the op's own default 16-bit DEST accumulation, and it is quantified.** Re-running the identical cells at `fp32_dest_acc_en=True` gives a **6.7–7.1x** reduction — `(1,1,32,64)` positive-only 0.00459 -> 0.00068, `(1,1,128,256)` 0.00537 -> 0.00076.
- **Strictly-signed distributions are the worst case for it**, exactly as expected: with all-positive `x`, the `x^2` terms are same-signed so the rounding error accumulates monotonically instead of partially cancelling as it does for `randn`. That is also why `test_uniform_input`, `test_small_magnitude_input`, `test_large_magnitude_input` and the residual/bias regressions all pass.
- **The score is taken against the wrong band.** `test_regression.py` calls `check_output(...)` **without** `tolerance=`, so it falls back to `eval.metrics.DEFAULT_TOLERANCES[float32] = (0.999, 0.01)` — the full-precision band. The op's own suite declares the correct band for this exact cell: `helpers.TOLERANCE_OVERRIDES[(float32, False)] = (0.995, 0.04)`, because the prompt makes `{float32, fp32_dest_acc_en=False}` a **supported** cell precisely so the op's own default is reachable. Measured 0.0165–0.0189 is comfortably inside 0.04. `eval/metrics.py`'s own docstring says ops with a profile "should define a local `TOLERANCES` and pass `tolerance=TOLERANCES[dtype]` at the call site — the predominate path going forward"; `test_regression.py` is the one place in this op's suite that does not.
- **There is no in-scope kernel lever**, and that was tested rather than assumed (see the `ReduceFp32Mode` A/B below). The only lever that moves it is `fp32_dest_acc_en`, which is the **caller's** choice, and changing the default is forbidden — `eval/prompts/rms_norm_ttnn.txt` pins it: "The default is ... `fp32_dest_acc_en = False`. That is a real precision AND performance point, not a placeholder."

**Requested harness fix (3)**: `eval/golden_tests/rms_norm_ttnn/test_regression.py` should pass `tolerance=tolerance_for(ttnn.float32, fp32_dest_acc_en=False)` (already exported from `helpers.py`) at its `check_output` call sites, so the numerics tests score the op's default cell at the band the feature spec declares for it. Per the protocol these 6 are recorded as an observation, **not** filed as a refinement: no concrete lever is in scope.

#### Measured: `ReduceFp32Mode::Accurate` is not the lever, and is unsafe

The one plausible kernel-side precision lever was routing the float32 sum-of-squares
through the SFPU (`Fast` truncates the FPU/GMPOOL inputs to **tf32**, per
`reduce_helpers_common.hpp`). It was A/B'd by flipping the single template
argument at the `accumulate_reduce_block()` call site, each variant compiled into
its **own isolated `TT_METAL_CACHE`**:

| distribution / shape | Fast dest16 | **Accurate dest16** | Fast dest32 | Accurate dest32 |
|---|---|---|---|---|
| positive_only (1,1,32,64) | 0.00459 | **inf** (61 non-finite) | 0.00068 | 0.00068 |
| positive_only (1,1,128,256) | 0.00537 | **inf** (993 non-finite) | 0.00076 | 0.00076 |
| randn (1,1,32,64) | 0.00397 | **inf** (17 non-finite) | 0.00073 | 0.00073 |
| randn (1,1,128,256) | 0.00424 | **inf** (735 non-finite) | 0.00076 | 0.00076 |
| bfloat16 control | 0.00363 | 0.00363 | — | — |

Two independent facts kill it: at `fp32_dest_acc_en=False` — **the op's own
default** — the SFPU reduce path returns inf/NaN (it needs a 32-bit DEST), and at
`fp32_dest_acc_en=True`, where it works, it is identical to `Fast` to five
significant figures. The residual error at dest32 (`r std` ~5.3e-04) is tf32's
`2^-11 = 4.9e-04` mantissa step, but removing the input truncation buys nothing
because the *accumulation*, not the input rounding, sets the error. The kernel was
reverted; the finding is recorded at `D3` in the descriptor's design notes with
the numbers and a warning that the failure mode is a silent inf.

---

## Perf-Target Ranking

No `LOOSE_CASES` entry carries an `attention` note, so
`eval/prompts/perf_refinement_prompt.txt` step 1 falls through to its second
branch: rank the `perf` group by measured device-ns / the case's own clock-scaled
`achievable_ns`. `device_kernel_ns` is captured per test by the golden runner, so
this comes straight out of the run above via
`ttnn/ttnn/operations/rms_norm_ttnn/perf_target_ranking.py`. **Clock-scale factor
1.0000** (measured 1350 MHz == the reference clock).

**16 of 19 targets met. 3 missed, and all three are the same regime.**

| ratio | measured ns | ceiling ns | cores | case |
|---:|---:|---:|---:|---|
| **1.060** | 5 812 | 5 481 | 28 | `(1,1,32,7168)` WIDTH_SHARDED `[32,256]` (7,4), gamma, dest16 |
| **1.050** | 6 882 | 6 555 | 32 | `(1,1,32,5120)` WIDTH_SHARDED `[32,160]` (8,4), **gamma_bias_residual**, dest32 |
| **1.014** | 5 339 | 5 267 | 32 | `(1,1,32,5120)` WIDTH_SHARDED `[32,160]` (8,4), gamma, dest16 |
| 0.984 | 34 008 | 34 569 | 64 | `(1,1,7168,1024)` BLOCK_SHARDED, gamma_bias_residual |
| 0.978 | 4 517 | 4 617 | 9 | `(1,1,32,2304)` WIDTH_SHARDED |
| 0.939 | 3 860 | 4 110 | 8 | `(1,1,32,1024)` WIDTH_SHARDED |
| 0.930 | 89 992 | 96 744 | 110 | `(1,1,8192,1024)` INTERLEAVED |
| 0.921 | 194 606 | 211 345 | 110 | `(1,1,8192,2304)` INTERLEAVED |
| 0.886 | 1 628 773 | 1 837 678 | 110 | `(1,1,8192,7168)` INTERLEAVED, gamma_bias_residual, dest32 |
| 0.856 | 24 497 | 28 619 | 64 | `(1,1,8192,1024)` BLOCK_SHARDED |
| 0.612 | 9 118 | 14 894 | 22 | `(1,1,32,7168)` INTERLEAVED — the **>=7x** case |
| 0.573 | 423 285 | 738 307 | 110 | `(1,1,8192,5120)` INTERLEAVED |
| 0.571 | 589 591 | 1 032 281 | 110 | `(1,1,8192,7168)` INTERLEAVED |
| 0.539 | 4 932 | 9 149 | 8 | `(1,1,32,1024)` INTERLEAVED |
| 0.527 | 692 618 | 1 314 426 | 110 | `(1,1,8192,5120)` INTERLEAVED, gamma_bias_residual, dest32 |
| 0.336 | 5 715 | 17 003 | 22 | `(1,1,32,2304)` INTERLEAVED |
| 0.183 | 11 950 | 65 349 | 32 | `(1,1,128,4096)` INTERLEAVED, ROW_MAJOR weight |
| 0.104 | 7 870 | 75 825 | 22 | `(1,1,32,5120)` INTERLEAVED |
| 0.067 | 10 599 | 157 570 | 22 | `(1,1,32,5120)` INTERLEAVED, gamma_bias_residual, dest32 |

Notes that shaped the queue:

- **The hardest requirement in the spec is already met with room.** `(1,1,32,7168)` INTERLEAVED is the only case carrying a `minimum_expected_speedup` (7.0), i.e. a `104259 / 7 = 14 894 ns` ceiling. Measured **9 118 ns** — 0.612 of it, a 11.4x speedup over the reference. The interleaved AUTO width split is doing its job.
- **The three misses share one regime**: WIDTH-sharded decode at 28–32 cores, absolute durations of 5–7 µs where a per-round fixed cost is the whole story. That makes them one refinement, not three.
- **Several `op_design.md` perf lamps are already spent.** `L-SUBBLOCK` cites `feature_spec`'s own note that `(1,1,8192,1024)` BLOCK_SHARDED reaches 25 513 ns with `subblock_w = block_w` + `inplace` against 28 619 at `subblock_w = 1`; the op measures **24 497 ns** at its own default blocking, i.e. below the figure the lamp treats as the prize. Not filed.
- **The interleaved prefill cases are near — but not at — the DRAM roofline.** `(1,1,8192,1024)` moves ~34 MB in 89 992 ns = **378 GB/s** against the ~450 GB/s this part delivers (the figure `op_design.md` D14 measured); `(1,1,8192,7168)` runs 234 MB in 589 591 ns = **397 GB/s**. That is 13–19% of headroom, real but modest, and it is a compute-overhead story rather than a byte-count one because the byte count is already minimal (ROW_RESIDENT). Filed as Refinement 3.

---

## Harness findings (infrastructure, not the op)

1. **`CoreRange.start_coord` / `.end_coord`** (`eval/golden_tests/rms_norm_ttnn/program_config.py:67-68`) — see above. Deletes 7 loose cells + 3 validation tests silently.
2. **`torch.max()` on a zero-element readback** (`eval/metrics.py`) — see above. Deletes 3 loose cells, op-independent.
3. **`test_regression.py` omits the op's tolerance profile** — see above. Produces 6 red cells for behaviour the feature spec explicitly declares in-band.
4. **The golden cartesian does not fit one tool invocation.** 121 438 collected cells; the runner's up-front precompile pass alone spent >30 min on 15 620 unique programs and then a further 30+ min single-threaded before producing anything. Worked around with `scripts/verifier_golden_shards.sh` (18 `-k` shards, `--no-precompile`, ~4-7 min each) + `scripts/verifier_merge_golden_shards.py` (merge by `nodeid`, so overlapping `-k` selectors de-duplicate). Both scripts are committed so the next phase does not re-derive them. Merged coverage is the complete 121 438.
5. **`math_approx_mode` is a genuine coverage hole.** The prompt says it is "NOT gated — any value is accepted", and the golden runner pins `compute_kernel_config.math_approx_mode = False` for every cell. But `math_approx_mode=True` is the op's **own default** (A7), so the configuration a caller gets by omitting the config is never exercised by the 120 960-cell cartesian — only by the hand-written unit tests. Worth adding as a finite axis, or at least pinning one loose group at `True`.
6. **A kernel-source A/B can leave a poisoned JIT artifact that survives clearing `built/`.** This environment sets `TT_METAL_CACHE=<repo>/built`, `TT_METAL_CCACHE_KERNEL_SUPPORT=1` **and** `TT_METAL_JIT_SERVER_ENDPOINT=bgdepyc01:54778`. After the `ReduceFp32Mode::Accurate` experiment above was reverted, the *reverted* source kept being served the *Accurate* binary: `test_rms_norm_ttnn_debug.py` failed 2/19 with non-finite output on exactly the config the experiment had compiled (float32, `math_approx_mode=True`, `epsilon=1e-12`), while the same cells passed in the golden suite (which pins `math_approx_mode=False`) and in a fresh cache. `rm -rf built` did **not** clear it. Two things restore correctness: a fresh `TT_METAL_CACHE=<tmp>`, or `--no-jit-server`. Confirmed both ways, then verified clean: **455 passed / 1 skipped** across the whole unit directory. Any future kernel A/B (every perf refinement in the queue is one) **must** isolate the cache per variant or it will silently measure the previous binary — this is the single most expensive trap in this environment.

---

## Tests added by this phase

| File | Purpose |
|---|---|
| `tests/.../test_rms_norm_ttnn_precision_baseline.py` | 32 cells: PCC / max-abs / mean-abs / relative-RMS **and the got/true ratio spread** (the scale-bug detector) across 4 shapes x 3 dtypes x both DEST modes, plus the 4 operand modes at the op's default config. Asserts against a uniform scale error explicitly. |
| `tests/.../test_rms_norm_ttnn_invalid_audit.py` | 16 cells covering the five capability regions parked in `feature_spec.INVALID`, which the golden suite structurally cannot reach. The evidence behind the INVALID-audit recommendation and a standing regression test for those regions. |
| `ttnn/.../perf_target_ranking.py` | Ranks the `perf` loose group by measured device-ns / clock-scaled ceiling, from a golden run's own sidecars. The trailing perf pass reads the same ranking; re-run it at the start of every perf phase. |
| `scripts/verifier_golden_shards.sh`, `scripts/verifier_merge_golden_shards.py` | The sharded-golden-run harness (finding 4). |

---

## Recommendations

1. **Apply harness fixes 1–3.** They are 3 small edits and they recover 10 `supported_fail` cells + 3 validation tests + 6 numerics cells, turning the golden report fully green without touching the op. Until then, the honest reading of this run is **23 325 passing, 0 op-attributed failures, 19 harness-attributed**.
2. **Remove the five author-scoped `INVALID` entries.** They hide five working, prompt-required capability regions from the entire cartesian; `test_rms_norm_ttnn_invalid_audit.py` proves all five pass.
3. **Add `math_approx_mode` coverage** (finding 5) — the op's own default is currently untested by the cartesian.
4. **Isolate the JIT cache per variant in every perf phase** (finding 6). Every refinement in the queue is a kernel A/B; without `--no-jit-server` or a per-variant `TT_METAL_CACHE`, the measurement is of the previous binary.
5. **Do not tighten the tolerance bands.** Measured headroom is 8.6–23x on every dtype cell; the bands are correctly placed.
6. **L1 watch item, no action yet**: `_cb_block_mult` over-prices `cb_x_squared` under D12's DEST fold (Code Review advisory). It cannot cause an OOM — the error is conservative — and it only bites at `WT_CHUNK <= 8`, where L1 is not the binding constraint. Folded into Refinement 3's scope.
7. **The queue is all-perf, and that is the correct shape here.** The 2:1 generality/perf cadence degenerates because `TARGET - SUPPORTED` is empty on every axis and `EXCLUSIONS` is empty; per the protocol, once generality candidates are exhausted the remaining phases are all perf, and they continue until the headroom is gone or the roofline says there is none left. `op_requirements.md` carries four, ordered by measured miss first.
