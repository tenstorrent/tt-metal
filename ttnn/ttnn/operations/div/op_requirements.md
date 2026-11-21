# Operation Requirements: div

## Definition
- **Formula**: `out = a / b`; with `rounding_mode="floor"` `floor(a / b)`, with `"trunc"` `trunc(a / b)`. b is a tensor (torch broadcasting) or a Python number; output shape = broadcast shape, layout = a's, dtype = `dtype` or the operands'.
- **PyTorch Reference**: `torch.div(a.double(), b.double(), rounding_mode=rounding_mode)` (`reference` in `eval/specs/div/spec.py`)
- **Import Path**: `from ttnn.operations.div import div`
- **Function Signature**: `div(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor | float, *, rounding_mode: str = None, dtype: ttnn.DataType = None, compute_kernel_config: ttnn.ComputeConfigDescriptor = None) -> ttnn.Tensor`
- **Target / backend**: Quasar (`ttnn.ProgramSpec` only), ttsim (untimed: perf gates are structural counts, see Refinements 3 and 5).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **One dispatch per call (run rule)**: every refinement keeps `div` at exactly one `ttnn.generic_op` dispatch per call. A wrapper of tensor-manipulation ops (`ttnn.to_layout`, `ttnn.tilize`, `ttnn.untilize`, `ttnn.repeat`, `ttnn.to_memory_config`, …) around the kernel is therefore **not** an available escape hatch in this run, not even for a `[~]` partial: a partial ships an in-kernel subset and files the rest as a lettered follow-up.
> **Thread plan**: every new path keeps the design's plan (`THREAD_PLAN = (4, 4, 2)`, whole rounds of C units over the full grid) and extends `test_div_thread_plan.py` with it: every NEO ran (the `tag_threads` probe), and one device session with another program before and `ttnn.to_layout` after.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED memory_config / sharding**: [DRAM, L1] / [interleaved]
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}, rank ∈ {2, 3, 4}, broadcast ∈ {none, number}
- **SUPPORTED op-specific axes**: rounding_mode ∈ {None, floor, trunc}, output_dtype ∈ {None, bfloat16, float32}
- **Cores**: full grid, whole rounds of C tiles per cluster (`split_work_to_cores(..., row_wise=True)`), thread plan (R, C, W) = (4, 4, 2)
- **Compute config**: `default_compute_kernel_config()` = HiFi4, fp32_dest_acc_en=True, math_approx_mode=False; any fidelity / approx mode accepted (R4)
- **Golden baseline**: 435 / 435 supported cells passing (432 PHASE0 cells + 3 numeric regression cases), 2 / 2 rejections; 4608 xfail_expected (1680 of them refused for good by rules R1/R2); 0 loud (per `verifier_report.json`)

### [ ] Refinement 1 — Broadcast operands (both orders)

**Goal**: add `over_w`, `over_h`, `over_hw`, `over_leading` and `first_operand` to SUPPORTED["broadcast"] — every TARGET broadcast value — on the TILE path, for a tensor divisor of lower rank or with size-1 dimensions and for a first operand that is the expanded one (rule R9). Phase 0's cells stay green; output shape is `torch.broadcast_shapes(a, b)` and the flat split runs over the **output's** tiles. Unlocks the 198 TILE/bfloat16 broadcast cells now in `xfail_expected` (the RM and float32 broadcast cells follow with Refinements 2 and 4).

**Verifier notes**:
- Anchor: this refinement lands the full contract of perf Refinement 3 (the broadcast cells it optimizes run at bfloat16, TILE, DRAM, `fp32_dest_acc_en=True`, the default config — everything else of that contract is already in SUPPORTED). It goes first for that reason.
- Classification (Blocking Model): a knob-turn on the built structure (design regimes `bcast_b_leading`, `bcast_b_inner`, `bcast_a_first_operand`): each reader maps an output page id `p → (n, ht, wt)` to the expanded operand's page with broadcast strides (RT args, per operand: size-1 dims stride 0), and only the DEST load of the expanded operand changes. `over_leading` needs no compute change; `over_h` / `over_w` / `over_hw` need a row / column / scalar broadcast load of the expanded operand's tile into its DEST slot; `first_operand` is the same on a (D0) — and an INPUTS entry may expand a in one dim and b in another, so give **both** readers the general mapping.
- The expanded operand is re-read once per output tile here (the design's stepping stone). That is accepted for this refinement only because Refinement 3 is the scheme-change that removes it; do not build the resident/read-once topology here.
- Quasar check first: whether `UnaryBcast` (`ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp`) builds and is correct on Quasar is unverified (several `bcast.h` entry points are `#ifndef ARCH_QUASAR`). Prove it at C = 1 and C = 4 on one cell of each of over_h / over_w / over_hw before building on it. If it does not work, a fallback that keeps one dispatch is the SFPU row/col broadcast primitives (`tt_metal/hw/inc/api/compute/sfpu_binary_bcast.h`) or a reader that assembles the broadcast tile itself — the latter makes that ring explicit-sync, so follow the target page's protocol (ring in `disable_dfb_implicit_sync_for`, every transfer of that kernel tagged).
- Compute binary ≤ 24 KiB: select the broadcast variant per operand by CT arg / define (`if constexpr`), as Phase 0 selects the regime, so one binary holds one chain.
- Padding: a broadcast b tile with H or W = 1 holds zeros in its padding; a broadcast load replicates only the valid row/column, so padding never reaches a logical output element. Check `(1,1,32,1)/(1,1,32,64)` and `(1,512)/(128,512)` (first_operand) explicitly.

**Done when**: every golden cell with `broadcast` in {over_w, over_h, over_hw, over_leading, first_operand} and the rest of its axes in SUPPORTED passes; SUPPORTED["broadcast"] lists them; `verify_supported` shows 0 loud; the thread-plan tests cover a broadcast cell (every NEO ran, one device session with `ttnn.to_layout` after).

### [ ] Refinement 2 — ROW_MAJOR layout, in-kernel

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to SUPPORTED["layout"] for both tensor operands and the output (the output takes a's layout), every alignment, rank, broadcast value of Refinement 1, number divisor and rounding mode: a tilize-wrapped reader/compute front (sticks → tiles in the compute kernel), the unchanged divide/round chain on tiles, and an untilize back to sticks for the writer. Unlocks the 432 RM cells of the Phase 0 broadcast values plus the 198 RM broadcast cells (bfloat16).

**Implementation skill**: /memory-layouts, /interleaved-parallel

**Verifier notes**:
- Ordered after Refinement 1 because the RM path must carry the broadcast mappings too (RM × broadcast cells); RM is the hardest remaining generality axis (layout tier), so it precedes the numeric bundle, which then extends RM and broadcast once.
- Native only (one-dispatch rule above): no `ttnn.to_layout` / `ttnn.tilize` around the kernel. If the non-aligned RM shapes (W not a multiple of 32: stick widths of 50 / 17 elements, odd byte sizes against the 16 B L1 / 64 B DRAM alignment) are out of reach in one pass, ship the tile-aligned RM cells as `[~]` and file `Refinement 2b` for the non-aligned ones.
- The work unit becomes a tile-row of sticks (32 rows × W), larger than a ring entry: use the target page's padded-round rule (`toy_row_passes`): whole rounds of C rows per cluster, a NEO without a row in the last round recomputes a real one. Keep the full grid and plan (4, 4, 2).
- Quasar gap that applies directly: the compute kernel now packs to more than one ring (tilize → divide → untilize), and the eltwise helpers keep the previous pack target when the output ring changes — call `pack_init(<ring>)` before each helper call whose output ring differs (`.claude/references/quasar-library-gaps.md`). Fast untilize is disabled on Quasar (`untilize_helpers.inl`); verify the tilize/untilize helpers at C = 1 and C = 4 first.
- L1: the stick rings and the tilized intermediates are new ledger rows; size them from `BLOCK_TILES` / `BUFFER_DEPTH` (or a row-block knob defined once), never from W or the tensor; update `l1_ledger.md`.

**Done when**: every golden cell with `layout=ROW_MAJOR` and the rest of its axes in SUPPORTED passes (0 loud); one dispatch per call; the thread-plan tests cover an RM cell (every NEO ran; one device session with another program before and `ttnn.to_layout` after).

### [ ] Refinement 3 — Read a broadcast operand once per cluster

**Type**: perf

**Goal**: after Refinement 1 the expanded operand of a broadcast cell crosses DRAM once per **output** tile. Build the design's `bcast_b_resident` regime (op_design.md, Regimes): split the output so each cluster's range spans the broadcast axes, read the expanded operand's tiles that range needs **once** into a resident ring that every NEO of the cluster sees, and reuse them across the range. Same rule for the first operand in `first_operand` cells. Region: the TILE/bfloat16/DRAM broadcast cells with reuse, e.g. `(4,8,32,256)/(1,1,32,256)` (over_leading), `(2,4,128,512)/(2,4,1,512)` (over_h), `(4,128,512)/(4,1,1)` (over_hw), `(1,1,64,128)/(1,1,1,1)`, `(1,512)/(128,512)` (first_operand). Relevant catalog entries in `ttnn/ttnn/operations/examples/master.md`: `shared_input_reuse` (the read-once principle; here cluster-local, no multicast), `width_split` / `noc_placement` (keep the grid full while re-splitting). No SUPPORTED change. **No LOOSE_CASE is perf-flagged** in the specification, so this region is the verifier's selection: div at plan (4, 4, 2) is most likely bound by data movement (the toy's timed figures show 16 SFPU passes per tile already at the DRAM floor with 4 NEOs), and this is the lever that removes DRAM bytes.

**Structural gate (ttsim has no clock)**: the count is **pages of the expanded operand read from DRAM/L1 banks per call**, read off the op by a test (count the reader's page reads per regime — e.g. a host-side derivation from the RT args the program is built with, asserted equal to what the kernel loops over). Before → done: `(4,8,32,256)/(1,1,32,256)`: 256 → ≤ 32; `(2,4,128,512)/(2,4,1,512)`: 512 → ≤ 128; `(4,128,512)/(4,1,1)`: 256 → ≤ 32; `(1,1,64,128)/(1,1,1,1)`: 8 → 1 per engaged cluster (2 at the current split of 8 tiles). Clusters engaged must not drop below the Refinement 1 split's count by more than the reuse requires (state the count before and after). The gain in time is unconfirmed until a timed backend (`ttsim-perf`) or hardware measures it — say so in the changelog; never report a ttsim timing as a speedup.

**Verifier notes**: a scheme-change (new split + resident ring), so it stands alone. The resident ring is read by every NEO: verify the ring's `ALL` consumer access pattern on Quasar under C = 4 (the design names it; it is not exercised anywhere in this op yet), else hold one copy per NEO slice. The design's `bcast_b_mcast` stays deferred with its positive reason (after this split each b tile's readers are disjoint for every broadcast entry of INPUTS); keep that row honest in op_design.md. Fold the L1 ledger update for the resident ring into this refinement.

**Done when**: the structural count above reaches its done value on the named shapes; no regression in the same count or in clusters engaged on the guard set (one representative per path: same-shape tensor, number, each broadcast kind, TILE and RM, DRAM and L1); golden suite green (0 loud); changelog records counts before/after and that the time gain is unconfirmed.

### [ ] Refinement 4 — Numerical configurability: float32 operands and 16-bit DEST

**Goal**: add `ttnn.float32` to SUPPORTED["dtype"] and `False` to SUPPORTED["fp32_dest_acc_en"], across every layout, broadcast value, rounding mode and output dtype the op then supports. Rules R1 and R2 stay as explicit refusals in `validate()` (float32 operands or a float32 output with `fp32_dest_acc_en=False`: 1680 cells that stay `xfail_expected` by rule, not by gap). Unlocks 2100 cells (720 TILE + 720 RM same-shape/number, 330 + 330 broadcast).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Ordered after Refinements 1–2 (precision tier, cheapest): it extends the TILE, RM and broadcast paths once instead of being reworked under them.
- **The float32 quotient is graded at 1 ulp, elementwise max** (`NUMERIC_RESOLVED[('float32','default')]`, applied by `eval/numeric_entry.py`). Measured on Phase 0 (bfloat16 operands, float32 output, `test_div_precision_baseline.py`): Quasar's `DIV` is `a · reciprocal₂(b)` with no residual step (`tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h:145-147`) and reaches **2 fp32 ulp** on ~0.5–1 % of elements (42 % at ≥ 1 ulp). Expect the float32/float32 cells to fail by that margin unless the quotient gets one residual correction (`q += r · (a − b·q)`, a custom SFPU element in the chain, keeping x/x = 1 and the zero / infinity cases). Measure before choosing.
- Float32 pages must be unpacked exactly: `UnpackToDest` for the 32-bit input rings (the srcA path is tf32). Quasar gaps that apply (`.claude/references/quasar-library-gaps.md`): the chain's operand reconfig asserts on a ring read with `UnpackToDest` → `DataFormatReconfig::Disabled` on those inputs; every operand of a DEST section lands before the first SFPU op (this chain's order already does). bfloat16 pages keep the default unpack mode in both DEST widths (`UnpackToDest` of bf16 into 32-bit DEST aborts ttsim).
- `fp32_dest_acc_en=False` (bfloat16 operands and output only, by R1/R2): Quasar's `DIV` then rounds RNE to bfloat16 in the SFPU itself (`round_div_to_bf16`); confirm the 0-ulp bfloat16 grade still holds and that floor/trunc of a bf16-rounded quotient stays within NUMERIC freedoms on the suite's inputs. DEST holds twice the tiles; the chain clamps its section itself.
- Ring cap: float32 pages are 4096 B — `BUFFER_DEPTH · BLOCK_TILES · C · 4096` = 131,072 B at the Phase 0 knobs, inside the 1,048,560 B cap the host already asserts; update `l1_ledger.md` page formats.

**Done when**: every golden cell with `dtype=float32` or `fp32_dest_acc_en=False` and the rest of its axes in SUPPORTED passes, except the R1/R2 refusals which raise `NotImplementedError`; 0 loud; the precision baseline gains float32-operand and 16-bit-DEST rows.

### [ ] Refinement 5 — Scope the SFPU to the valid rows of a padded tile-row

**Type**: perf

**Goal**: the traced YOLOv8 calls (`(1,2,8400)` and `(1,2,33600)` divided by `2`, and `(1,2,8400)/(1,2,8400)`; bfloat16, TILE, DRAM, default config, output `None` / `float32`, every rounding mode — all in SUPPORTED since Phase 0) have H = 2 in a 32-row tile: 30/32 of every tile's SFPU work (fill, divide, floor/trunc) is on padding. Scope the SFPU elements of the last tile-row to its valid rows (catalog `sfpu_tile_scope` in `ttnn/ttnn/operations/examples/master.md`: `VectorMode::R` + `ITERATIONS`; needs the underlying calls or a custom chain element). After Refinement 4 adds a residual-correction step, the SFPU cost per tile rises, which is when this lever matters most. No SUPPORTED change; no LOOSE_CASE is perf-flagged, so the region is the verifier's selection (the shapes the model calls).

**Structural gate (ttsim has no clock)**: **SFPU vector operations issued per output tile of a tile-row with `H mod 32 = 2`**, per SFPU element (fill, divide, floor/trunc): 32 now (4 faces × 4 row-groups × 2 column parities) → ≤ 4 (the first row-group of the two top faces: `VectorMode::R` + `ITERATIONS=2`, the catalog's `r_iter2`; `VectorMode::R` alone gives 16, an acceptable `[~]`), read off the op by a test (e.g. a CT probe, or the output's padding rows left undivided where today they hold 0/0 = NaN). Full tile-rows keep 32. The gain in time is unconfirmed until a timed backend measures it — say so in the changelog.

**Verifier notes**: confirm headroom first with `/perf-ceiling-dm` (pass `--aiclk-mhz` explicitly; ttsim has no clock) on `(1,2,33600)`: if the op is bound by data movement at plan (4, 4, 2) even with Refinement 4's SFPU work, record that and close this refinement with the count reached but no claimed gain. Valid-row count per tile-row is a function of H only: pass it as an RT arg (`H mod 32`, 0 = full), not a CT arg (CT args that vary per shape recompile on ttsim; op_design.md Key Risks).

**Done when**: the structural count reaches its done value on the three traced shapes; full tile-rows unchanged; golden green (0 loud), in particular every h_non_aligned cell; no regression in the count on the guard set (one representative per path × layout × placement).
