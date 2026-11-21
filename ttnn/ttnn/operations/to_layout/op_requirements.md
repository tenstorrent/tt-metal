# Operation Requirements: to_layout

## Definition
- **Formula**: `output[i] = input[i]` for every logical index `i`; only the storage order changes (`TILE_LAYOUT` ↔ `ROW_MAJOR_LAYOUT`). Every cell is graded `exact` (rule R1).
- **PyTorch Reference**: `def reference(input_tensor): return input_tensor.double()` (no torch equivalent; the values are the input's).
- **Import Path**: `from ttnn.operations.to_layout import to_layout`
- **Function Signature**: `to_layout(input_tensor: ttnn.Tensor, layout: ttnn.Layout) -> ttnn.Tensor` — output in the input's buffer type, interleaved (R5); exactly one `ttnn.generic_op` dispatch per call.
- **Target / backend**: Quasar on ttsim. Only `ttnn.ProgramSpec` programs run; no time can be measured, so every perf refinement below is gated on a **structural count** (backend page, *Performance measurement*), and its gain in time stays unconfirmed until a timed backend or hardware measures it.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

Omitted: SUPPORTED["dtype"] bfloat8_b — Quasar has no path for block-float formats (target page, *Family facts*); its golden cells skip as unbuildable, validate() refuses it, and R2 refuses a block-float ROW_MAJOR target in every phase. No refinement is queued for it.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout / target_layout**: [TILE, ROW_MAJOR] × [TILE, ROW_MAJOR] (regimes `tilize_stream`, `untilize_stream`, `identity_copy`)
- **SUPPORTED placement**: memory_config ∈ {DRAM, L1}, sharding = interleaved
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, h_non_aligned}, rank = 4
- **Cores**: full grid — 2-D independent split (tile-row × column block of `bw` tiles), `split_work_to_cores(row_wise=True)`; thread plan (1, 1, 1) (data-movement bound, stated in the design)
- **Compute config**: none (pure data movement; compute kernel is a fixed LoFi / 16-bit DEST datacopy)
- **Golden baseline**: 120 / 120 supported cells passing; 1656 xfail_expected, 296 invalid_skipped, 0 loud (per verifier CLI)

### [x] Refinement 0 — Rank 3 (Llama's tilize calls)

**Goal**: add `3` to SUPPORTED["rank"], at the values of the other axes supported by now.

**Origin**: the model-level test of the Llama job. tt-metal's `tilize`, `untilize` and `untilize_with_unpadding` are this op with the layout fixed and are routed to it; 65 of the three jobs' 67 calls are Phase 0 cells and are served. The two that are not are `tilize` of `[1, 32, 64]` and `[1, 32, 2048]` (bfloat16, row-major, DRAM), which the op refuses: `rank=3 not in SUPPORTED [4]`. Both shapes are INPUTS entries now.

**Operator notes** (queued by the operator of the run, not by the verifier):
- This takes over the rank-3 part of Refinement 4; rank 2 and W not a multiple of 32 stay with it.
- Probe before you build: the kernels may not care how many leading dimensions there are.

**Done when**: every golden cell with `rank=3` inside the new SUPPORTED passes exactly, in both directions; earlier cells unchanged; `verify_supported` loud categories are 0.
**Outcome**: host-only knob-turn, as the operator note predicted: `block_geometry` already folds leading dims into `num_images = prod(shape[:-2])`, so the only change is `SUPPORTED["rank"] = [3, 4]`. All 64 bf16 rank-3 golden cells inside SUPPORTED (8 shapes × 2×2 layouts × DRAM/L1, tile_aligned + h_non_aligned) pass exact; a rank-3 shape gets the same block geometry (bw, split) as its rank-4 `[1, ...]` twin.

### [ ] Refinement 1 — Bit-exact narrow-integer dtypes (uint16, uint8)

**Goal**: add `ttnn.uint16` and `ttnn.uint8` to `SUPPORTED["dtype"]`, in every regime (tilize, untilize, identity copy), both placements, every currently supported alignment and rank, with every element moved bit-exactly (R1: `exact`). The op today has one datapath, bfloat16 through SrcA and 16-bit DEST. These dtypes need their own exact path, which is the design's deferred `exact16_8_stream` regime.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- First because it is the hardest generality item: no bit-exact unpack path is known for these dtypes on this part. uint16 cannot ride the bfloat16 SrcA path. Golden inputs are integers in [0, 1000] for uint16, and 1–127 have a zero exponent field when read as bfloat16, so they are denormals that SrcA flushes. Quasar's ring formats accept `UInt8`, `Int16`, `RawUInt8`, `RawUInt16` but **not** `UInt16` (`is_supported_quasar`, `tt_metal/common/tt_backend_api_types.cpp:97`). Whether an unpacker/packer path exists that keeps these bit patterns, or a data-movement-only shuffle is needed (the design's `dm_only_layout_change` row, rejected for bfloat16 only), is the implementer's call. Prove it with the golden `exact` check, not by assumption.
- The identity-copy regime is format-agnostic (DM → DM ring with no metadata), so it should take both dtypes unchanged.
- 1-byte elements change the stick-chunk alignment: `select_block_width_tiles` step 3 already requires `bw·32·elem_size % dram_alignment == 0` (`bw` even for uint8). Keep that derivation the single source.
- Build it to the Phase 0 standard: full-grid split, block-batched reader and writer. Refinement 4's ragged-W work is written later against this path.
- Any cell that fails in an interpretable way and is out of reach goes to EXCLUSIONS with a follow-up `Refinement 1b`. Never use a host-side `ttnn.typecast` / multi-dispatch workaround: that breaks the one-dispatch rule.

**Done when**: every golden cell with `dtype ∈ {uint16, uint8}` inside the resulting SUPPORTED passes `exact`; Phase 0 cells still pass; `test_one_session_with_tt_metal_programs` (or an equivalent session test) passes for the new dtypes' regimes.

### [ ] Refinement 2 — Bit-exact 32-bit dtypes (float32, int32, uint32)

**Goal**: add `ttnn.float32`, `ttnn.int32`, `ttnn.uint32` to `SUPPORTED["dtype"]`, in every regime, both placements, every currently supported alignment and rank, bit-exact (R1: "an int32 or float32 value does not survive a bfloat16 buffer"). This is the design's deferred `exact32_stream` regime. Only the ring format, the compute `KernelSpec` (32-bit DEST, unpack mode) and the helper template arguments change; the blocking, the split and the ring topology stay as they are. `tile_bytes` doubles, so `bw_max` halves through step 2 of `select_block_width_tiles`.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- The SrcA path truncates fp32 to tf32 (`tilize_helpers.hpp:49-73`), so the plain path breaks R1. The exact route is `UnpackToDest` into 32-bit DEST (`enable_32_bit_dest=True`, `Fp32Mode::Lossless` on the tilize helper).
- uint32 is not a ring format on Quasar: a byte-moving ring carries it as `Int32`. Keep the dtype → ring-format mapping in one host function shared with Refinement 1's.
- After Refinement 1, because the per-dtype datapath selection that Refinement 1 introduces is extended here, not rebuilt. That is ordering by difficulty, not a hard dependency.
- `R1`'s check is `Check("cells", {"dtype": int32})`: the int32 cells are the rule's evidence and must pass `exact`, not merely be declared.

**Done when**: every golden cell with `dtype ∈ {float32, int32, uint32}` inside the resulting SUPPORTED passes `exact`; prior cells still pass; the session test passes for the 32-bit regimes.

### [ ] Refinement 3 — Speed up the YOLOv8 traced calls

**Type**: perf

**Goal**: no loose case is perf-flagged, so the region is the specification's traced callers: the four YOLOv8 calls, `[1, 1, 400, 512]` and `[1, 1, 1600, 256]`, bfloat16, DRAM, TILE → ROW_MAJOR and ROW_MAJOR → TILE. All are in Phase 0 SUPPORTED, so nothing needs unlocking first. Two levers from `ttnn/ttnn/operations/examples/master.md` match this op's profile (a data-movement-bound permutation over interleaved DRAM, 1–2 units per cluster on these shapes):
1. **Bank stagger on the stick side** (`bank_stagger`, ⭐⭐ T2). Clusters that share a tile-row (the 2-D split puts `n_col_blocks` = 4 or 1 of them on each) read and write the *same* stick page at step `r`. Every tile-row starts at stick `32·h`, so whenever the DRAM bank count divides 32, every engaged cluster hits the same bank at every step of the tilize reader and the untilize writer. Rotate each cluster's row order (`stagger_rows`: start at row `core_index % 32` of its block and wrap) as a compile-time switch.
2. **Untilize padding reads** (design lamp *Untilize padding reads*). An image's last tile-row with `valid_rows ≤ 16` holds only padding in faces 2–3, so read faces 0–1 (the first half page) of those tiles.

No SUPPORTED change.

**Done when** (structural counts, read off the op by a host test from the built program's runtime args and the kernels' documented issue order, before and after recorded in the changelog):
- (a) the peak number of engaged clusters whose step-`r` stick transfer targets one DRAM bank, on both YOLO shapes in both directions, drops from *all engaged clusters* (32) to ≤ `ceil(engaged / num_banks)`;
- (b) input bytes read by the untilize of `[1, 1, 400, 512]` drop from 425,984 B (`S_tile`) to 409,600 B (`S_rm`, the minimum);
- golden suite green, all four YOLO calls still `exact`, the session test green;
- no regression across the config-spanning guard set: one representative per regime (tilize, untilize, copy-tile, copy-RM) × placement (DRAM, L1). On this backend "regression" means a structural count (bytes moved, transfers per barrier, clusters engaged) getting worse.

State in the changelog that the time gain is unconfirmed on ttsim.

### [ ] Refinement 4 — Ranks 2 and 3, and W not a multiple of 32

**Goal**: add `2` and `3` to `SUPPORTED["rank"]` and `w_non_aligned` to `SUPPORTED["alignment"]`, in every regime and placement and for every dtype SUPPORTED holds by then. The reader zeroes (or masks) the tail columns `[W, 32·ceil(W/32))` of each tilized row, and every stick-side NoC transfer moves exactly the valid bytes of its stick: the last column block's transfer is `W·elem_size − col_block·chunk_bytes` bytes. This is the design's *TARGET axes → refinement class* row for `w_non_aligned`, a reader/writer knob-turn.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- **Latent bug to fix, not just a value to declare.** Ranks 2/3 and ragged W both came back bit-exact in a verifier probe that bypassed `validate()` (DRAM only), but ragged W passed only by luck. The stick reader and writer always move a whole `chunk_bytes = bw·32·elem_size`, which is more than the stick holds: W = 50 bf16 is a 100 B stick moved as 128 B. In DRAM the extra bytes land inside the page's 64 B-aligned slot. In **L1 (16 B alignment) the untilize writer would write past the stick into the neighbouring page** of that bank, which another cluster owns. Exact-size transfers are required, and an L1 `w_non_aligned` untilize cell must be in the evidence.
- The identity-copy ring's `entry_size` is `buffer_page_size()`. A ragged RM stick (W = 17 bf16 = 34 B; uint8 17 B) is not a multiple of the 16 B L1 alignment, so check what the ring and the NoC accept, and pad the entry if needed.
- Ranks 2/3 are host-only (`num_images = prod(shape[:-2])` already folds them). Their golden cells mix with `w_non_aligned` shapes, which is why the two ship together.
- Last among the generality items because ragged W touches every regime's stick side, and after Refinements 1–2 it is written once, for every element size (2-byte, 4-byte, 1-byte sticks).
- The partial last column block must not become a ragged *tilize* block: the Quasar tilize MOP width must equal the init width (Mechanism caps). Keep `bw | tensor_w_tiles` and shorten only the stick transfers and the zeroing.

**Done when**: every golden cell with `rank ∈ {2, 3}` or `alignment = w_non_aligned` inside the resulting SUPPORTED passes `exact`, including L1 cells; prior cells still pass.

### [ ] Refinement 5 — The explicit-sync stick side (thread plan)

**Type**: perf

**Goal**: each conversion has one explicit-sync side: the tilize reader (assembles a block from 32 stick reads plus zeroing) and the untilize writer (scatters a block into 32 stick writes). Build the design's lamp *Thread plan*: a plan with two threads on the explicit side (e.g. (2, 2, 2)) on **BLOCKED** rings of `block_size = bw` (a tile-row block must not be dealt per entry), dealing whole rounds of C blocks per cluster. The alternative is the catalog's `trid_double_issue` (keep the next block's transfers outstanding across the barrier). Pick from master.md; this is ⭐⭐⭐ T3, the whole phase. No SUPPORTED change.

**Verifier notes**: this deliberately re-opens the design's (1, 1, 1) choice, for the explicit side only. The target page's rule still holds: a cluster needs at least C units for C threads to help, and on the YOLO shapes clusters hold 1–2 units. So the host must engage fewer clusters rather than pad rounds where units are few, and state that choice in `THREAD_PLAN`'s derivation. The plan stays one named host constant, read off the `ProgramSpec` by `test_thread_plan_on_every_kernel`.

**Done when** (structural counts; time gain unconfirmed on ttsim, stated in the changelog):
- (a) DM threads issuing the explicit side's transfers: 1 → 2 on clusters with ≥ 2 units, or barriers that drain the stick side with no next-block transfer outstanding: 1 per block → ≤ 1 per kernel, depending on the lever chosen;
- (b) padded (recomputed) rounds per op ≤ 1;
- a test shows every thread ran (a missing thread leaves missing work, not just a correct-looking result);
- the session test (after another program, before tt-metal's `ttnn.to_layout`) passes at the new plan;
- golden green and no structural regression across the guard set (regime × placement).

### [ ] Refinement 6 — Size the identity copy's in-flight window in bytes

**Type**: perf

**Goal**: `identity_copy` keeps `BUFFER_DEPTH · COPY_BLOCK_PAGES` = 8 pages in flight per cluster, counted in **pages**. A row-major page is one stick, so a narrow tensor keeps very few bytes on the NoC: W = 64 bf16 gives 128 B pages and 1 KiB in flight, below the catalog's 256 B-per-transfer floor times the 4–8 outstanding transfers of `double_buffer`. Derive the window from a byte target, from one host constant, clamped by the ring cap, so small pages get proportionally more entries. ⭐ T1 knob-turn (`double_buffer` in master.md). No SUPPORTED change.

**Done when** (structural count; time gain unconfirmed on ttsim): bytes in flight per cluster for an RM identity copy of `[32, 64]` and `[1, 1, 32, 64]` rise from 1 KiB to ≥ the chosen byte target (e.g. 8 KiB), with tile-page copies unchanged or larger; golden green; no structural regression across the guard set; the session test passes.
