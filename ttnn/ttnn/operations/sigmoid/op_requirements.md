# Operation Requirements: sigmoid

## Definition
- **Formula**: `output[i] = 1 / (1 + exp(-input[i]))`, element by element; output has the input's shape, dtype and layout, allocated in the input's buffer (spec R6)
- **PyTorch Reference**: `torch.sigmoid(input_tensor.double())` (the specification's `reference`)
- **Import Path**: `from ttnn.operations.sigmoid import sigmoid`
- **Function Signature**: `sigmoid(input_tensor: ttnn.Tensor, *, compute_kernel_config: ttnn.ComputeConfigDescriptor = None) -> ttnn.Tensor` (`None` → `default_compute_kernel_config()` = HiFi4, `fp32_dest_acc_en=True`, `math_approx_mode=False`)
- **Specification**: `eval/specs/sigmoid/spec.py` (governs). Target: Quasar on ttsim (no clock: perf gates are structural counts, see Refinement 3).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases (acceptance, `test_sigmoid_threads.py`, `test_sigmoid_precision_baseline.py`, golden suite).
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **One dispatch (mandatory, every phase)**: each public call is exactly one `ttnn.generic_op` of one `ttnn.ProgramSpec`. No refinement may sequence another TTNN op (`ttnn.to_layout`, `ttnn.typecast`, `ttnn.tilize`, …) around the kernel — not even as a partial-tick fallback.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE_LAYOUT]
- **SUPPORTED memory_config / sharding**: [DRAM, L1] / [interleaved]
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}, rank ∈ {2, 3, 4}
- **Cores**: full grid (8 × 4 = 32 clusters on quasar), flat page axis dealt in whole rounds of C pages (`split_rounds`, `row_wise=True`); thread plan (R, C, W) = (2, 4, 2) from `THREAD_PLAN`
- **Compute config**: exposed; default HiFi4 + `fp32_dest_acc_en=True`; `math_fidelity`, `math_approx_mode`, `dst_full_sync_en` passed through, never gated (R3)
- **Accuracy**: 0 bf16 ULP on every element of 12 baseline cases (4 shapes × normal / normal×10 / uniform[0,1)), PCC ≥ 0.99986
- **Golden baseline**: 80 / 599 passing (74 PHASE0 cells + 6 NUMERIC regression cases), 518 xfail_expected, 0 loud

### [ ] Refinement 1 — ROW_MAJOR layout, in-kernel (layout-blind stream)

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]` (ROW_MAJOR_LAYOUT), for every Phase 0 cell's other axes — bfloat16, DRAM and L1 interleaved, ranks 2–4, all three alignment buckets — by a native ROW_MAJOR reader/writer pair inside the same single program. The output keeps the input's layout (ROW_MAJOR in → ROW_MAJOR out, in the input's buffer, R6). 74 golden cells move from `xfail_expected` to `supported_pass` (`dtype=BFLOAT16, fp32_dest_acc_en=True, layout=ROW_MAJOR`).

**Implementation skill**: /memory-layouts

**Verifier notes**:
- Ordered first by difficulty (a new dataflow path: tier 3) so that Refinement 2's precision axes extend both layouts once, rather than the RM path being built against a dtype surface that then changes.
- The design already names the regime: **R-rm-stream** (`op_design.md`, Regimes). sigmoid is element-wise, so the compute kernel is layout-blind: an entry of 1,024 elements of the flat row-major stream goes through unpack → DEST → pack as a permutation and its inverse and lands in place. R1's compute kernel (`eltwise_chain` + `SigmoidAccurate`) stays unchanged; only the reader/writer and the host split change. The skill's default for math-bearing ops is a tilize-wrapped reader; the design rejected that regime (R-tilize: two extra compute phases and two intermediate rings for the same bytes) and the layout-blind stream is the in-kernel path to build. Either is native; the tilize wrapper is the fallback only if the layout-blind path hits a blocker you can name.
- Ring protocols (target page): an entry assembled from several stick reads is **explicit sync** — list the rings in `disable_dfb_implicit_sync_for`, tag every NoC transfer of that kernel with `NocOptions::TXN_ID` / `{.trid = 0}` and barrier on the same id. The writer that scatters an entry back across sticks is the same. Keep the tag hand-back at kernel end, and extend `test_one_device_session` with a ROW_MAJOR input (op between two tt-metal programs in one session) — a one-process-per-test golden run cannot see a tag left set.
- RM pages are sticks of `W` elements; read the page size from the buffer (`buffer_page_size()`, which may exceed `W * elem_size` by alignment) and move only the logical bytes. The last entry of a core's run is partial: compute on whatever the ring holds and write back only the valid elements. Non-aligned W/H are not special in RM (no tile padding).
- Keep the Phase 0 bar on the new path: full grid, dealt in whole entries, the (2, 4, 2) plan with every thread busy, the block knobs (`BUFFER_DEPTH`, `block_tiles`, entry size) defined once and carried as parameters, per-cluster work balanced to within one entry. Extend `test_sigmoid_threads.py`'s every-NEO-ran and plan tests to a ROW_MAJOR input. Write the reader/writer element-size generic (page size from the tensor) so float32 in Refinement 2 is a host-side change only.
- **No `ttnn.to_layout` / `ttnn.tilize` wrapper**: the run's one-dispatch rule forbids it outright, so there is no wrapper partial-tick escape hatch for this refinement.

**Done when**: every `layout=ROW_MAJOR, dtype=BFLOAT16, fp32_dest_acc_en=True` golden cell passes (74), the 80 Phase 0 passes still pass, the thread and session tests pass on a ROW_MAJOR input, and `verify_supported` shows no loud category.

### [ ] Refinement 2 — Numerical configurability: float32 and 16-bit DEST on both layouts

**Goal**: add `ttnn.float32` to `SUPPORTED["dtype"]` (float32) and `False` to `SUPPORTED["fp32_dest_acc_en"]` (False), on both TILE and ROW_MAJOR, DRAM and L1, every rank and alignment. Add the permanent `EXCLUSIONS` entry `{"dtype": ttnn.float32, "fp32_dest_acc_en": False}` (spec rule R1: refused natively, legal but lossy — it is a rule, not a gap, and never leaves EXCLUSIONS). Moves 296 golden cells to `supported_pass` (bf16 × 16-bit DEST × 2 layouts = 148, float32 × fp32 DEST × 2 layouts = 148); the 148 float32 × 16-bit DEST cells stay `xfail_expected` through EXCLUSIONS. After it: 450 pass, 148 refused by R1.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Depends on Refinement 1 (the float32 RM cells need the RM path). Much of the descriptor work is already in place: `compute_kernel_config` is exposed and mapped field for field (`enable_32_bit_dest`, `double_buffer_dest`, fidelity, SFPU precision), ring entry size is read from the tensor, `DEST_TILES[(fp32_dest_acc_en, dst_full_sync_en)]` already sizes `block_tiles` for 16-bit DEST, and the SFPU body is templated on `DST_ACCUM_MODE` (`_sfpu_sigmoid_<false>` = `exp_21f` + one Newton step at 16-bit DEST). Expect mostly a SUPPORTED/EXCLUSIONS change plus precision verification.
- **Grading is ULP, not PCC** (`NUMERIC_RESOLVED`): bfloat16 cells ≤ 2 bf16 ULP everywhere (≤ 1 on `[0, 1]` inputs), regardless of DEST width; float32 cells ≤ **2 float32 ULP** on bf16-representable inputs (the golden inputs are). The 16-bit-DEST bf16 cells risk an extra rounding in DEST before the pack; the float32 cells are the hard bar: Phase 0's body (`_sfpu_exp_fp32_accurate_` + LUT reciprocal + 2 Newton steps) has never been measured against a 2-fp32-ULP gate. Measure ULP histograms on float32 first, and if the bound fails, strengthen the reciprocal / final rounding (e.g. an FMA residual correction) rather than loosening anything. Cells left failing on precision stay failing as `numerical-precision` with a follow-up `Refinement 2b` — never silenced through EXCLUSIONS.
- float32 unpack: the default srcA path is lossless for bf16-representable inputs; `UnpackToDest` is available for 32-bit pages on ttsim (not for bf16 pages), and with it the chain needs `DataFormatReconfig::Disabled`. Ring capacity doubles with the 4,096 B page (ledger worst case 524,288 B with full-sync DEST, under the ring cap).
- `bfloat8_b` / `bfloat4_b` are not in TARGET and have no path on Quasar: no work for them.

**Done when**: every golden cell with `dtype ∈ {BFLOAT16, FLOAT32}` and `fp32_dest_acc_en ∈ {True, False}` passes on both layouts except the R1 cells (float32 × `fp32_dest_acc_en=False`), which are refused with `NotImplementedError` (`xfail_expected`); `test_sigmoid_precision_baseline.py` is extended with float32 and 16-bit-DEST rows and gates on the numeric entry's ULP bounds; prior phases still green; no loud category.

### [ ] Refinement 3 — Cut the SFPU cost per tile on the YOLOv8 class-score profile

**Type**: perf

**Goal**: no loose case is perf-flagged, so the region is chosen here: the op's real caller, YOLOv8's class scores `(1, 80, 8400)` bfloat16 TILE in DRAM at the default config (`fp32_dest_acc_en=True`, HiFi4; 789 pages, 6–7 per NEO over all 32 clusters × 4 NEOs). The op is SFPU-bound by design (`op_design.md`, Work Distribution: Bound), and the work split is already at its structural minimum (max pages per NEO = ⌈num_tiles / 128⌉ on every corpus shape), so the remaining lever is the per-tile compute. The sigmoid element's per-row loop issues **57 SFPU instructions** today (compiled math TRISC, `sfpload` → `sfpstore`; 23 of them `sfploadi` constant reloads). Use the compute patterns in `ttnn/ttnn/operations/examples/master.md` (compute fusion / work per tile) and the levers this body offers: hoist loop-invariant constants into LREGs, and — because the output is bfloat16 — a cheaper exp/reciprocal combination than the fp32-accurate one (e.g. the 16-bit-DEST body's `exp_21f` + one Newton step, or one Newton step fewer), kept only if it holds the numeric entry. No SUPPORTED change.

**Gate (structural — this backend has no clock)**: the count is *SFPU instructions per DEST row in the sigmoid element's loop body*, read from the compiled math binary (TRISC1 ELF in the JIT cache, `riscv-tt-elf-objdump -d`, instructions between the loop's `sfpload` and `sfpstore`) by a test or probe that the refinement adds. Before: 57 at `fp32_dest_acc_en=True`. The gain in time is **unconfirmed** on ttsim and must be stated so in the changelog; never report a simulator timing as a speedup. Whether the op is still SFPU-bound at C = 4 is a question for `ttsim-perf` / hardware, not this backend.

**Done when**: the count is ≤ 45 (a ≥ 20 % cut) for the bfloat16-output path at `fp32_dest_acc_en=True`; an **exhaustive** bfloat16 test (all 65,536 bf16 input patterns minus NaN/Inf, through the device) shows ≤ 1 ULP on `[0, 1]` and ≤ 2 ULP elsewhere (the spec's `bf16-exhaustive` domain, tiny-reference band honoured); the float32 path keeps its own (accurate) body and its Refinement 2 result; the golden suite and all prior tests stay green; and no regression of the count or of correctness across the config-spanning guard set (one representative per kernel path × layout × placement: TILE/RM × DRAM/L1 × bf16/fp32 × both DEST widths).
