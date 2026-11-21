# Operation Requirements: nlp_create_qkv_heads

## Definition
- **Formula**: input `[B, 1, S, (nh + 2·nkv)·hd]`, `nkv = nh` when not given, `hd = W / (nh + 2·nkv)`;
  `q[b,h,s,d] = x[b,0,s,h·hd+d]` `[B, nh, S, hd]`; `k[b,h,s,d] = x[b,0,s,(nh+h)·hd+d]` `[B, nkv, S, hd]`
  (with `transpose_k_heads=True`, the default: `[B, nkv, hd, S]`, last two dims exchanged);
  `v[b,h,s,d] = x[b,0,s,(nh+nkv+h)·hd+d]` `[B, nkv, S, hd]`. Pure movement: every output element is one
  input element, bit-exact (rule R1).
- **PyTorch Reference**: `reference()` in `eval/specs/nlp_create_qkv_heads/spec.py`:
  `[s.reshape(B, S, n, hd).permute(0, 2, 1, 3) for s, n in zip(torch.split(x[:, 0], [nh*hd, nkv*hd, nkv*hd], -1), (nh, nkv, nkv))]`, k transposed on its last two dims when `transpose_k_heads`.
- **Import Path**: `from ttnn.operations.nlp_create_qkv_heads import nlp_create_qkv_heads`
- **Function Signature**:
  `nlp_create_qkv_heads(input: ttnn.Tensor, *, num_heads: int, num_kv_heads: int = None, transpose_k_heads: bool = True, memory_config: ttnn.MemoryConfig = None) -> list  # [q, k, v]`

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Backend (ttsim, no clock)**: every perf refinement below is gated on a **structural count** read off the op by a test, not on device-ns (`.claude/references/backends/ttsim.md`, *Performance measurement*). Its gain in time stays unconfirmed until a timed backend or hardware measures it; never report a simulator timing as a speedup.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED placement**: memory_config=[DRAM], sharding=[interleaved], output_memory=[DRAM]
- **SUPPORTED shape-derived axes**: alignment=[tile_aligned], batch=[single]
- **SUPPORTED op-specific axes**: transpose_k_heads=[False]
- **Regime**: `tile_relocate` — DM-only whole-tile permutation, reader → STRIDED implicit-sync ring (8 pages) → writer, no compute kernel; folded input-page split over the full 8×4 grid (`split_work_to_cores(row_wise=True)`); thread plan (R, C, W) = (1, –, 1) (DM-bound, stated in the design).
- **Compute config**: none (the op computes nothing; the specification gives it no compute config)
- **Golden baseline**: 9 / 9 supported cells pass (15 / 439 tests passed incl. 6 rejections; 423 xfail_expected; 0 loud), per `verifier_report.json`

### [ ] Refinement 1 — Sub-tile head assembly (head_dim not a multiple of 32)

**Goal**: add `"w_non_aligned"` to `SUPPORTED["alignment"]` by building the design's deferred `subtile_assemble` regime natively in the kernels: a head starts and ends inside an input tile, so each output tile is assembled in L1 from face-row segments of the (up to two) input tiles its head spans, its own new padding columns filled, and written whole. `tile_relocate` stays the regime for `head_dim % 32 == 0`. After this refinement every SUPPORTED cell with `alignment = w_non_aligned` passes exactly — including `((1, 1, 17, 150), 1, 1)`, which is S-non-aligned as well (its tile-padding rows travel with their tiles).

**Verifier notes**:
- **Scheme-change — stands alone** (op_design.md Blocking Model: `head_dim_tile` row, "scheme-change when `hd % 32 ≠ 0`"; regime table row `subtile_assemble`). The topology is the work: the unit becomes (part of) an input tile-row resident in L1, and the writer (or a DM assembly stage) builds output tiles with explicit-sync rings (target page *Ring protocols*: list the ring in `disable_dfb_implicit_sync_for`, tag every transfer of that kernel with `{.trid = 0}`, hand the tag back at exit). `tile_column_dataflow.hpp` and `l1_helpers.hpp` (`fill_l1_range` / `zero_tile`) are the kernel-library building blocks the design named.
- **Fill the machine.** The w_non_aligned INPUTS have 1–2 tile-rows per sequence, so a tile-row split engages 1–2 of 32 clusters. Split finer (output tiles or (row, head) units), each cluster reading only the input tiles its outputs touch; record the resulting DRAM crossings per input page (≤ 2 where a tile is shared by two heads on different clusters) in `l1_ledger.md`'s traffic budget and regime table, with the cheapest-traffic alternative and why it was or was not taken.
- **Expose the block knobs as single-source parameters** (rows/units per block, assembly-ring depth, assembling threads): Refinement 3 turns them. Size face-row copies from `page_bytes` / element size, not a bf16 literal — Refinement 4 adds float32 onto this regime.
- Regime-pinned tests (one cell per regime, asserted by the regime the host selects) and the two device-session tests (op after another program; `ttnn.to_layout` after the op) for the new kernels, which use explicit sync.
- Lands before Refinement 2 (difficulty order, hardest first): the k-transpose there is a per-tile step that composes onto an assembled k tile, so it is built once over both regimes.

**Done when**: every golden cell with `alignment = w_non_aligned` inside SUPPORTED passes exact; `tile_relocate` cells unchanged and green; regime-pinned and device-session tests pass; `l1_ledger.md` and `op_design.md`'s regime table updated to the built regime.

### [ ] Refinement 2 — Transposed k heads (tt-metal's default)

**Goal**: add `True` to `SUPPORTED["transpose_k_heads"]`: return k as `[B, nkv, hd, S]`. Tile `(dt, st)` of kᵀ is the 32×32 transpose of tile `(st, dt)` of the untransposed k (padding positions map onto padding positions), so the transpose is one per-tile step applied to every k-section tile of either regime (`tile_relocate` and Refinement 1's `subtile_assemble`), written at kᵀ page `((b·nkv + h)·hd_t + dt)·St + st`. q and v are untouched. This also makes the default call (`transpose_k_heads` omitted) a supported call.

**Verifier notes**:
- **Mechanism / exactness — read before choosing**: the design's deferred-row topology (compute `transpose_tile` through DEST, `fp32_dest_acc_en` for float32) collides with Quasar facts: the unpack-to-DEST path cannot transpose (`tile_move_copy.h:53-58`, `llk_unpack_unary_operand_to_dest.h:49`; only SrcA transposes), and nothing on record shows the SrcA path bit-exact for float32. Rule R1 (MUST, exact for float32) binds the moment Refinement 4 adds float32. A face-aware 32×32 transpose done by a data-movement thread in L1 is exact for every dtype; a NEO path is acceptable only with a test proving it bit-exact on float32 bit patterns. If a compute kernel is added: set `fpu_math_fidelity` explicitly and keep bf16 pages off `UnpackToDest` with 32-bit DEST.
- New stage → new thread-plan decision (target page *Choosing the plan*): state the stage's bound, put the plan in one host constant, and add the test that shows each thread transposed its own tiles. Refinement 3 may turn the count.
- k pages need a second ring (reader → transposing stage → writer); each ring keeps one producer and one consumer.
- Regime-pinned tests per (regime × transpose) path; device-session tests again if the protocol mix changes.

**Done when**: every golden cell with `transpose_k_heads = True` inside SUPPORTED (both alignments then supported) passes exact; the acceptance test's default call passes as a supported call; all earlier cells still green.

### [ ] Refinement 3 — Co-tune the block and thread knobs of the two new paths

**Type**: perf

**Goal**: the paths Refinements 1 and 2 add do real per-tile work on data-movement or compute threads (face-row assembly; per-tile transpose), unlike `tile_relocate`. Turn their exposed knobs — units per block, assembly/transpose ring depth, and the threads working each stage — guided by `ttnn/ttnn/operations/examples/master.md` (`double_buffer`: whole blocks per handshake and a `2·block` ring; `trid_double_issue`; `split_reader`: splitting issue-bound DM work over harts). Target cells: `((1, 1, 64, 300), 2, 2)` and `((1, 1, 32, 150), 1, 1)` (w_non_aligned, bf16, DRAM) and the traced text-model shape `((1, 1, 128, 6144), 32, 8)` with `transpose_k_heads=True`. No SUPPORTED change.

**Gate (structural, ttsim)**: (a) explicit-sync handshakes (reserve/push pairs) on the assembly ring per input tile-row — from Refinement 1's value to at most one per block; (b) threads at work on the assembly stage and on the transpose stage per cluster — from Refinement 1/2's value to the plan chosen here (≥ 2 where the stage is per-thread work), each thread's share shown disjoint and non-empty by a test; (c) DRAM crossings per page and NoC write transactions per output page unchanged. Record before/after in `changelog.md`; the time gain is unconfirmed on this backend.

**Done when**: counts (a)–(c) reach the values above on the target cells, the golden suite is green, every result exact, and no regression across the guard set (one representative per kernel path: `tile_relocate`, `subtile_assemble`, each with and without k transpose).

### [ ] Refinement 4 — float32, L1 placement, results in the input's buffer, S non-aligned, batched

**Goal**: add `ttnn.float32` to `SUPPORTED["dtype"]`; `"L1"` to `SUPPORTED["memory_config"]`; `None` and `"L1"` to `SUPPORTED["output_memory"]`; `"h_non_aligned"` to `SUPPORTED["alignment"]`; `"batched"` to `SUPPORTED["batch"]` — across all three regimes then built. After this SUPPORTED equals TARGET.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- **Knob-turn bundle** (Blocking Model: `batch` and `token_row` rows "knob-turn"; Dataflow Strategy: DRAM ↔ L1 "knob-turn"). Evidence: `probes/probe_003.py` ran today's `tile_relocate` program, validate() bypassed, on float32, L1 input, L1 output, L1 both (float32), S = 17 and 50, B = 2 and 4 — **all bit-exact with zero kernel change**. For `tile_relocate` this refinement is SUPPORTED + tests; the work is on the Refinement 1/2 paths: element size from the dtype in the assembly copies, exact float32 transposes (rule R1 — see Refinement 2's notes), the batch index in their walks, and S tile-padding rows in kᵀ's tiles (they become padding columns).
- The skill's dtype + intermediate-format parts apply; its `compute_kernel_config` part does not — the specification gives this op no compute config, so expose none. Every ring's format is the input dtype; no buffer may hold a float32 value in a narrower format (R1). `bfloat8_b` is not in TARGET and is unbuildable on Quasar: refuse it, queue nothing for it.
- `memory_config=None` puts the results in the input's buffer, interleaved (R5): already implemented in the entry point; the cells check it.
- Ordered after the structural refinements on purpose (difficulty order): the hard paths are built and tested on the small surface first, then extended here once.

**Done when**: all 432 golden cells are supported and pass exact (0 xfail_expected, 0 loud); earlier tests green.

### [ ] Refinement 5 — Stagger the clusters' DRAM bank order on the traced calls

**Type**: perf

**Goal**: on the callers' configuration (bf16, DRAM → DRAM, `transpose_k_heads=False`, the traced `((1, 1, 128, 3072), 16, 16)` and `((1, 1, 128, 6144), 32, 8)`), every cluster's page range has an even length (12 and 24 pages), so all 32 clusters start their read stream on the same one of the 2 DRAM banks and walk the banks in lockstep. Rotate each cluster's issue order within its range (catalog `bank_stagger`, ⭐⭐ T2, `ttnn/ttnn/operations/examples/master.md`; it measured 1.04–1.07× on interleaved sources) so concurrent first requests spread over the banks; apply the same reading to the writer's first output pages. The ring order must stay the page order both kernels walk. No SUPPORTED change.

**Gate (structural, ttsim)**: the largest number of engaged clusters whose first read targets one DRAM bank — 32 today on both traced calls — down to `ceil(32 / num_dram_banks)` (16 on this part), computed by a test from the run args and the input's bank mapping; the same count for first writes no worse than today. Time gain unconfirmed on this backend.

**Done when**: the count reaches 16 on both traced calls, every result exact, the golden suite green, and no regression on the guard set (one representative per kernel path × dtype × placement).
