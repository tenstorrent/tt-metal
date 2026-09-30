# Verification Report: chunk_gated_delta_rule_fwd

Verifier pass on the Phase 0 build (commits after `40f629dea5`: planner `e509b4f7f6`, implementer
`199fb657b5` … `9939c3e57f`). Hardware: Blackhole p300a, 110 worker cores (11×10 grid).

> **Environment note**: the shell exports `TT_MESH_GRAPH_DESC_PATH` pointing at the 1-chip p150
> descriptor on this 2-chip p300a box; every device open then fails ("Physical chip id 0 not found").
> All runs below were made with the variable unset (`unset TT_MESH_GRAPH_DESC_PATH`). The implementer
> logged the same friction.

## Code Review

### Fixed in place

1. **Constant-tile build was 47 µs of serial reader work on every core** (`build_constant_tiles`,
   reader). The diagonal `[32×32]` blocks of `EYE / LT / SL / SU` were written lane by lane through a
   non-inlined `store_word()` call (~3 × 1024 calls per diagonal tile), and the reader cannot start
   the first gather until it finishes — stage-marker zones put `R_P` at 47 µs on all 110 cores.
   Rewritten: of a diagonal tile's four 16×16 faces, faces 0 and 3 carry the triangle / identity
   pattern and face 2 (LT, SL) or face 1 (SU) is all ones. The CPU now writes three triangle faces
   and one identity face once (~400 inline stores) and everything else is a NoC copy (from the
   all-ones tile, or of diagonal tile 0). Measured, device kernel duration:
   tiny `(1,32,1,32,32)` 76 → 56 µs (1.35×); `(1,256,4,128,256)` bf16 387 → 344 µs (1.12×);
   `(1,1000,4,128,128)` bf16 373 → 331 µs (1.12×); `(1,2048,2,128,128)` 520 → 472 µs (1.10×);
   `(4,128,16,64,64)` 361 → 339 µs; `(1,256,32,128,128)` 852 → 813 µs; LOOSE fp32 5477 → 5341 µs.
   Acceptance 50/50 passes in both default and `--dev` mode (watcher, NoC sanitizer).
2. **Gate gather retired one read barrier per token tile** (`gather_gate`, reader). The `g` and `beta`
   page reads for an item (`2·Ct` pages) were each followed by their own `noc_async_read_barrier()`.
   Now all `2·Ct` reads go out back to back into the gather staging buffer and a single barrier
   retires them (`gather_gates`), guarded by a `static_assert` that they fit the staging buffer.
   Measured perf-neutral (within ±2% run-to-run noise): the item's face-row gather dominates. Kept
   because it removes a per-unit completion boundary.
3. **Knob guards**: `static_assert(32 % GATHER_TOKENS == 0)` (the window walk `WPT = 32 / GATHER_TOKENS`
   silently mis-staged otherwise) and `static_assert(1 ≤ GATHER_DEPTH ≤ 15)` (one 4-bit NoC transaction
   id per staging slot) — so a host-side knob turn fails at compile time instead of corrupting data.
4. **Validation gap: sharded I/O was silently accepted.** The design scopes I/O to interleaved DRAM or
   L1 (face-row gather/scatter and full-page state I/O through interleaved `TensorAccessor`s), and
   `TARGET` has no `memory_layout` axis, but `validate()` rejected neither a sharded input nor a
   sharded `memory_config`. It is now a documented mechanism-cap `ValueError` before any dispatch. The
   entry point passes `memory_config` to `validate()`. Test added:
   `test_chunk_gated_delta_rule_fwd_extended.py::test_rejects_sharded_output_memory_config`, plus the
   `float32 + fp32_dest_acc_en=False` refusal and the bf16 16-bit-DEST path (previously untested).
5. **L1 ledger currency**: the closed-form total and the quanta line omitted the `Vs` terms that
   `_quanta()` builds (`Qv = max(Ct·Vi, Kt·Vs, Kt·Vi)`, `Qf = Ct·max(Kt, Ct, Vi, Vs)`). Corrected. They
   give identical values at every INPUTS shape. Three disjoint-lifetime pairs had no recorded
   justification; those are now recorded (see L1 Ledger Audit).

### Reviewed, no change needed

- **Prompt Rules (MUST / MUST NOT)** — all hold:
  - The entry point calls no TTNN compute or data-movement op. It only does `allocate_tensor_on_device` for the outputs and scratch, plus exactly one `ttnn.generic_op`.
  - The reader zero-fills the ragged tail, and the writers emit exactly `T` rows.
  - `no_h0` reads no tensor (`HAS_H0 = 0` compiles the read out), and `h[:,0]` is packed zeros, checked with `== 0.0` exactly.
  - `h` is the ENTERING state.
- **Soft rules** — all followed:
  - `D = (LT@diag(g))@SL` with `exp` in DEST; `w = exp(SU@g)`. `decay` never reaches an FPU source register on its way to `L`.
  - The V-block extent knob (`Vi`) is host-solved from the closed-form ledger.
  - `(bh)` and V blocks are spread across cores.
- **Helper usage**: elementwise phases use `compute_kernel_lib::eltwise_chain` (`BinaryFpu` Mul/Sub with
  COL broadcast, `CopyTile`, `MulUnary`, `FillScalar`, `PackTile`). The raw-API block ops are
  justified by missing helpers, which I confirmed. `ttnn/cpp/ttnn/kernel_lib/` has no matmul helper and
  no `mcast_pipe.hpp`, and the chain has no transpose element. The `Γ·S` carry is SFPU
  `mul_binary_tile` on fp32 DEST, per the precision contract. There is no reduce-helper candidate:
  the op has no reduction.
- **Correctness hygiene**: `void kernel_main()`, `api/...` include paths, and `TensorAccessor` are used
  everywhere. There is no `InterleavedAddrGen`. Every CB has one push quantum and matching wait/pop
  counts (walked per stage in all three kernels). Every handoff `noc_semaphore_inc` is preceded by
  `noc_async_write_barrier()`.
- **Broadcast efficiency**: the gate tiles carry column 0 only and are consumed via COL broadcast or
  as a matmul column. The one full-width need (`Γ_full`) is rebuilt in compute (`g @ E_ROW0`, then
  `ONES @ ·`) rather than CPU-filled. There is no redundant full-tile fill.

### Deferred (needs architectural rework; filed as refinements)

- The face-row gather/scatter transaction count is the measured critical path (Refinement 1).
- The `NV×` DRAM re-read of the scan's reuse-shared operands (Refinement 2).

## Registry Conformance

- `INPUT_TAGGERS` (`chunk_size` first, then `seq_alignment`, `head_dims`), `SUPPORTED`,
  `EXCLUSIONS = []` and `validate()` are present and correctly wired. Every tagger has the
  `(inputs, axes)` signature.
- `validate()` checks shape, rank and mixed dtype/layout first, then `SUPPORTED` per axis
  (`UnsupportedAxisValue`), then `EXCLUSIONS` (`ExcludedCell`), then the mechanism caps (`ValueError`).
  The public entry point calls it as its first statement.
- The op file does **not** declare `INVALID`.
- No auto-fixes to `SUPPORTED` were needed: `xpass_drift = 0`. SUPPORTED equals TARGET exactly.
- **INVALID audit** (`feature_spec.py`): `INVALID = []`, and this is well-formed. TARGET layout is
  TILE-only and bfloat8_b is outside the dtype universe (documented there), so the canonical
  bf8b + ROW_MAJOR entry is vacuous. There are no cross-tensor couplings. It is not a norm-like op, so
  no no-weight canonicalization applies. Nothing to report to the user.

## Design Conformance

- **Algorithm**: matches `op_design.md`: Neumann-doubling UT inverse with
  `neumann_steps = ceil(log2 C)`, `o` taken off the scan into stage E, and `nkcd` / `Pᵀ` materialized
  in stage P.
- **Pipeline / RISC ownership**: P → S → E in all three kernels. P never waits, S waits only on P, and
  E waits only on S.
- **Work distribution**: matches the design and fills the machine. At the LOOSE shape all 110 cores
  are active in P and E (`min(G, NI)`), with 64 scan units. Both dataflow halves are batched: the
  reader keeps windows in flight with trids, and the writer does block writes with one flush per block.
- **Blocking-model fidelity**: every planner knob is a host constant or a host solve, passed to the
  kernels as a CT arg from one source: `BLOCK_HEADS`, `BLOCK_CHUNKS`, `GATHER_STAGE_TOKENS`,
  `GATHER_DEPTH`, `SCAN_STREAM_DEPTH`, `EGRESS_DEPTH`, `ACCUM_DEPTH`, `BLOCK_DEPTH`,
  `READY_SEGMENTS_MAX`, `Vi`, `Vs`.
  - Push quanta come from `_quanta()`. Page counts come from `_cb_pages()`, the single source for both
    the L1 solve and the descriptor. `cb_const` / `cb_vec` layouts are `static_assert`-checked against
    host page counts.
  - No CB scales with an op dimension (`T`, `NC`, `BH`).
  - `BLOCK_HEADS` / `BLOCK_CHUNKS` are exposed but the kernels `static_assert(== 1)`. This is loud and
    documented. Turning them is kernel work, filed in Refinements 1 and 3.
- **Expression check**: the scheduling boundary is the block in all three kernels, not the tile.
  - Reader: one push per gathered block, a pipelined window stream inside it, and one barrier per scan chunk for its 4 operand CBs.
  - Compute: block ops.
  - Writer: one flush per egress block.
  - The only per-unit boundary found (the gate reads) is fixed (item 2 above).
- **Deviations the implementer recorded** (all accepted, all in `l1_ledger.md`):
  - Column-0 gate tiles instead of full-width.
  - `cb_vec` 2·Ct+1.
  - `E_ROW0` constant.
  - `h[:,0]` copied through by the reader, bit-exact.
  - No `UnpackToDestFp32`, because `cb_state` is also a matmul `in1` (the conflict the design's risk
    table anticipated). The carry `S` therefore enters DEST at tf32 precision. It is measured inside
    the fp32 band everywhere, including `g_scale = 8` and the 32-chunk `g_scale = 0.05` LOOSE case.

## L1 Ledger Audit

- **Currency**: all 27 CBs have rows, and the rows match `_cb_pages()` / `_page_bytes()`. The closed
  form and quanta line are now corrected to the as-built `_quanta()` (fix 5).
- **Capacity vs live set**:
  - Over-capacity rows are each justified. `ACCUM_DEPTH = 2` in-place CBs are required (1 hangs). The
    uniform-quantum tails of `cb_vblock_in` and the egress CBs are also accounted for.
  - `cb_kw` is a recorded but untaken alias candidate: 32 KB, with the reason recorded.
  - No collapsed extent: every spanned axis scales capacity, and streamed axes appear in no capacity
    expression.
- **Page format vs DEST**: Float32 pages with `fp32_dest_acc_en = True` (default). A caller-chosen
  16-bit DEST on bfloat16 makes them an *over* case. That is accepted by the prompt mandate (internal
  CBs Float32 regardless of config) and is non-default.
- **Disjoint lifetime, no justification**: three pairs were found, and their reasons are now recorded
  in the ledger rows:
  - `cb_vmat` ↔ `cb_scan_vnew`
  - `cb_T` / `cb_pow` / `cb_kb` ↔ `cb_state`
  - `cb_L` / `cb_cc_*` ↔ `cb_intra_in`

  The reasons are either different push quanta or different producer kernels. None is fixed in place,
  because L1 is not binding: the worst INPUTS footprint is 1181 KB against about 1416 KB of budget.
  The `cb_state` ↔ `cb_T` sharing opportunity (≤ 32 KB) is **folded into Refinement 3**, whose
  depth-2 input buffers are what would need the headroom.
- **Bounds / closed form**: every capacity symbol (`Ct, Kt, Vt, Vi, Vs, NS, gather_stage_tokens,
  row_span_stride`) is bounded by a validated predicate or the host solve. Op dimensions appear in no
  capacity expression.
- **Data-movement budget**: present, and consistent with the built R1 split. The `NV×` scan-operand
  re-read and the `h` re-read are counted. The cheapest-traffic split (R3 page-harvest) and R4 are
  `deferred` with a reachability argument but **no positive reason**. R3 serves the LOOSE / `H ≥ 4`
  shapes that R1 serves slowly, and occupancy is not a reason. This is discharged by filing them as
  Refinements 1 and 2.
- **Block-size defaults**:
  - Interleaved `Vi` takes the coarsest block (`Vi = Vt` at every INPUTS shape).
  - The item split spreads over the full grid.
  - The scan's `Vs` departs toward the finest split to fill the grid (occupancy-first `NV`). That is
    argued, not measured; the `NV/2` co-tune is in Refinement 2.
- **Per-core footprint** (closed form in `l1_ledger.md`): the `Vi` terms are `cb_vmat`, `cb_vnew_in`,
  `cb_vblock_in` and the egress quanta. The `Vs` terms are the scan stream and `cb_state`. Everything
  else is in `Ct`, `Kt`. The peak measured `device_l1_peak_bytes` is 1 200 768 B at
  `(1,256,4,128,256)` fp32.

## Precision Baseline

`tests/ttnn/unit_tests/operations/chunk_gated_delta_rule_fwd/test_chunk_gated_delta_rule_fwd_precision_baseline.py`.
The oracle is float64 on dtype-rounded inputs, and all six outputs are measured. The table shows the
outputs that matter most: the rest are in the test's printout. `h` is shown for `with_h0`. For
`no_h0`, `h` is exact zero at `NC = 1`.

| Shape | dtype | output | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP p99 | got/true median [p5, p95] |
|---|---|---|---|---|---|---|---|---|
| (1,64,1,64,64) c64 | fp32 | o | 0.9999994 | 3.94e-04 | 3.02e-05 | 3.64e-03 | 448 210 | 0.9964 [0.9922, 1.0000] |
| (1,64,1,64,64) c64 | fp32 | final_state | 0.9999993 | 2.16e-03 | 1.91e-04 | 3.46e-03 | 304 657 | 0.9969 [0.9938, 0.9998] |
| (1,64,1,64,64) c64 | fp32 | g_cumsum | 1.0000000 | 3.68e-02 | 1.79e-02 | 1.31e-03 | 10 867 | **0.9994 [0.9993, 0.9995]** |
| (1,64,1,64,64) c64 | fp32 | A | 1.0000000 | 7.36e-04 | 1.62e-06 | 1.32e-04 | 571 758 | 0.9990 [0.9958, 1.0011] |
| (1,64,1,64,64) c64 | bf16 | o | 0.9999982 | 2.84e-04 | 1.73e-05 | 2.37e-03 | 3 | 0.9982 [0.9944, 1.0018] |
| (1,64,1,64,64) c64 | bf16 | g_cumsum | 0.9999946 | 1.23e-01 | 3.93e-02 | 3.29e-03 | 1 | 1.0001 [0.9973, 1.0024] |
| (1,100,2,64,64) c64 | fp32 | o | 0.9999993 | 4.14e-04 | 2.84e-05 | 3.98e-03 | 748 937 | 0.9963 [0.9911, 1.0019] |
| (1,100,2,64,64) c64 | fp32 | h (with_h0) | 0.9999980 | 3.76e-03 | 1.23e-04 | 2.72e-03 | 265 040 | 1.0000 [0.9941, 1.0000] |
| (1,100,2,64,64) c64 | bf16 | o | 0.9999981 | 3.24e-04 | 1.70e-05 | 2.72e-03 | 6 | 0.9980 [0.9936, 1.0023] |
| (1,256,4,128,256) c64 | fp32 | o | 0.9999993 | 3.33e-04 | 1.42e-05 | 4.36e-03 | 644 377 | 0.9960 [0.9914, 1.0004] |
| (1,256,4,128,256) c64 | fp32 | final_state | 0.9999994 | 2.04e-03 | 1.32e-04 | 3.43e-03 | 398 633 | 0.9968 [0.9933, 0.9999] |
| (1,256,4,128,256) c64 | bf16 | o | 0.9999982 | 2.90e-04 | 8.29e-06 | 2.85e-03 | 7 | 0.9978 [0.9936, 1.0018] |
| (1,1000,4,128,128) c64 | fp32 | o | 0.9999993 | 3.47e-04 | 1.45e-05 | 4.35e-03 | 668 191 | 0.9960 [0.9912, 1.0006] |
| (1,1000,4,128,128) c64 | fp32 | h (with_h0) | 0.9999987 | 4.62e-03 | 1.32e-04 | 3.35e-03 | 393 536 | 0.9967 [0.9934, 1.0000] |
| (1,1000,4,128,128) c64 | bf16 | o | 0.9999981 | 2.79e-04 | 8.66e-06 | 2.92e-03 | 7 | 0.9977 [0.9935, 1.0019] |
| (1,1000,4,128,128) c64 | bf16 | final_state | 0.9999982 | 1.91e-03 | 5.81e-05 | 2.82e-03 | 7 | 0.9982 [0.9939, 1.0024] |

Worst over all 96 (shape × dtype × state × output) rows:

| dtype | min PCC | max rel-RMS | Golden band |
|---|---|---|---|
| fp32 | 0.99999796 | 4.38e-03 | PCC ≥ 0.999, rel-RMS ≤ 0.02 |
| bf16 | 0.99999456 | 3.29e-03 | PCC ≥ 0.99, rel-RMS ≤ 0.12 |

Across the golden suite, the recorded (worst-output) rows show min PCC 0.9999962 and max rel-RMS
4.9e-3 in fp32 (LOOSE 32-chunk weak-gate case). In bf16 they show min PCC 0.9999980 and max rel-RMS
3.0e-3.

**Assessment**: every cell passes with ≥ 4× margin on rel-RMS and ~3 orders of magnitude on PCC. The
bfloat16 path is accurate to a few bf16 ULP (p99 ≤ 7).

**Scale-vs-precision triage**: the fp32 path shows a **small, systematic sub-unity scale**:
- `o` median ratio 0.996, `final_state` 0.997.
- `g_cumsum` sits in a tight [0.9991, 0.9995] band around 0.9993.

It is **not** a structural or scale bug:
- PCC ≥ 0.99999 and rel-RMS 4e-3 is the "genuine precision" row of the triage table, not the
  "high-PCC + rel-RMS ≳ 0.1" bug signature.
- `g_cumsum` is `LT @ g`, where LT is exact 0/1. Its bias is exactly what truncating fp32 operands to
  the FPU's ~tf32 source-register width produces. bf16-exact inputs give a median of **1.0000** for
  the same matmul.

So the fp32 path is tf32-limited, with a slight truncation bias that compounds through the
chunk-level matmul chain to ~0.4% on `o`. Its rel-RMS (4.3e-3) is therefore *worse* than bf16's
(2.9e-3). A split-precision (hi/lo) fp32 matmul is the only lever, and no cell is failing, so this is
a recommendation, not a refinement.

**Recommended tolerances**: fp32 PCC ≥ 0.9999, rel-RMS ≤ 0.01; bf16 PCC ≥ 0.999, rel-RMS ≤ 0.01.
These are tighter than the golden bands, with ≥ 2× margin over the worst measured cell. The golden
bands (0.999/0.02, 0.99/0.12) stay as they are.

## Verifier CLI Summary

Results dir: `/tmp/cgdr_verify/golden2`. The report is copied to `verifier_report.json` in this
directory. The golden suite passed 90/90 (0 failed, 0 hangs), both before and after the fixes.

- supported_pass: 76 (72 TARGET × INPUTS cells + 4 LOOSE cases)
- xfail_expected: 0 (TARGET − SUPPORTED is empty)
- invalid_skipped: 0 (`INVALID = []`)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 14 (the `test_regression.py` numerics tests — gate strength, `beta = 0`
  pass-through, chunk chaining, `h[:,0]` semantics, explicit scale / config, L1 output — all passing)

Other suites:
- Acceptance: 50/50 in default and `--dev` mode.
- Extended: 3/3.
- Precision baseline: 16/16.

## Recommendations

- **Refinement queue**: `TARGET − SUPPORTED` is empty, so every (axis, value) of TARGET is already
  supported and nothing needs a refinement or a documented omission. The queue in
  `op_requirements.md` is therefore three measured perf refinements:
  1. Face-row transaction count (multi-head items / R3 page-harvest) — the measured critical path.
  2. R4 scan-operand multicast + scan placement.
  3. The block-depth / segment / chunk-block / fixed-cost knob co-tune.

  No LOOSE case is perf-flagged; the target is the LOOSE Qwen3.5 profile in both of its configs.
- **Perf picture** (stage-marker zones, LOOSE bf16, Phase 0 build):
  - Constant build was 47 µs on every core; it is now fixed.
  - Stage P ends between 1.22 and 3.66 ms per core for 9–10 items each, a 3× spread. The low-row
    cores are slowest.
  - The scan cores finish their own P at ~1.2 ms and then wait on the slowest P cores' segment
    releases.
  - Stage E is the last ~0.5 ms.
  - The gather knobs (`GATHER_DEPTH` 2/4 × `GATHER_STAGE_TOKENS` 16/32) are flat: the gather is
    transaction-bound, not latency-bound.

  The perf harness is `test_chunk_gated_delta_rule_fwd_perf.py` (target + guard set, run under
  `--profile`).
- **Mechanism caps, documented, not axes**:
  - `H ≤ 32` (page index assumes `ceil(H/32) == 1`; every INPUTS / LOOSE shape has `H ≤ 32`).
  - Interleaved I/O only.
  - `float32` needs `fp32_dest_acc_en = True`.

  None is a TARGET value, so none is an omission from the queue. If a future `H > 32` model shape is
  added to INPUTS, the page index becomes `((b·T + t)·Ht + h/32)·Dt + dt`, which is a reader/writer
  address-map change.
- **fp32 precision (no lever in scope)**: the fp32 path is tf32-limited and slightly biased low
  (triage above). If an fp32 consumer (e.g. a backward gated at a tighter band) ever needs more, the
  lever is a hi/lo split fp32 matmul on the chunk-level products, or at least an exact (SFPU or
  scalar) `g_cumsum`.
- **L1 headroom**: the largest INPUTS footprint is 1181 KB of about 1416 KB. Depth-2 input buffers
  (Refinement 3) at `K = 128, V = 256` will eat most of the rest. The `cb_state` ↔ `cb_T` sharing and
  the `cb_kw` alias are the recorded reclaim options.
- **Master catalog absent**: `ttnn/ttnn/operations/examples/master.md` is not on this branch, so the
  perf levers are argued from this op's own measurements and the planner's Regimes / Perf-lamps
  tables.
