# Operation Requirements: padded_slice

## Definition
- **Formula**: `output[0, 0, r, j] = input[s0 + n_r, s1 + h_r, s2 + w_r, s3 + j]` for `j < c`, and `0` for
  `c <= j < width`, where `(n_r, h_r, w_r)` is the `r`-th position of the block `[n, h, w]` in row-major
  order and `width` is the shard width of `memory_config`. The result lives in `memory_config`
  (height-sharded L1): row `r` is in shard `r // shard_h`.
- **PyTorch Reference**:
  ```python
  def padded_slice_ref(x, s, e, shard_width):
      rows = x[s[0]:e[0], s[1]:e[1], s[2]:e[2], s[3]:e[3]].reshape(1, 1, -1, e[3] - s[3])
      return torch.nn.functional.pad(rows, (0, shard_width - (e[3] - s[3])))
  ```
- **Import Path**: `from ttnn.operations.padded_slice import padded_slice`
- **Function Signature**:
  `padded_slice(input_tensor: ttnn.Tensor, padded_slice_start, padded_slice_end, padded_slice_step=None, *, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor`
- **Specification**: `eval/specs/padded_slice/spec.py` (the only source of requirements).
- **Design**: `op_design.md` (regimes A–H, Blocking Model); `l1_ledger.md`.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Backend (this run: Quasar on ttsim, untimed)**: a perf refinement is gated on the **structural count** its entry names, read off the op by a test or a probe, with the value before and after in the changelog. The time gain is unconfirmed on this backend until a timed backend (`ttsim-perf`) or hardware measures it; never report a simulator timing as a speedup.
> **Thread plan**: every refinement keeps the plan stated by one host constant (`GATHER_THREADS`, and any new kernel's count beside it), sized loops from CT counts + `get_my_thread_id()`, and keeps the three plan tests green: every hart writes its rows (sentinel), the built kernel's `num_threads`, and the one-device-session test (tt-metal program before and after the op). A new kernel or ring gets the same three.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [ROW_MAJOR]
- **SUPPORTED placement (input)**: memory_config=[DRAM], sharding=[interleaved]
- **SUPPORTED destination**: to=[L1/height] (shards exactly as wide as the block's rows)
- **SUPPORTED op-specific axes**: step=[ones], sliced_region=[rows], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[4]
- **Regime**: A — each core of the caller's shard grid gathers its own rows from DRAM straight into its own shard, one tagged NoC read per row, one barrier per hart per block (block = whole shard). No DFB, no compute, no writer; op-owned L1 = 0 B.
- **Thread plan**: (R, C, W) = (`GATHER_THREADS` = 2, 0, 0)
- **Cores**: the caller's shard grid (R10: 1, 2 or up to 32 cores)
- **Compute config**: none (the op computes nothing; the spec gives it no compute config)
- **Golden baseline**: 12 / 12 supported cells passing (11 PHASE0 cells + 1 loose case), 533 xfail_expected, 6/6 rejections, 0 loud (per verifier CLI)

### [ ] Refinement 1 — TILE-layout input (regime C, native segment gather)

**Goal**: add `ttnn.TILE_LAYOUT` to SUPPORTED["layout"] by reading a tiled input natively in the gather kernel (design regime C): for each output row, read only its logical elements from the input tile(s) — the face-row segments (16 elements each) of the faces that hold `(n, h, w)` and channels `[s3, e3)` — straight into the row's place in the shard. Tile indices per image use `ceil(W/32)` and `ceil(C/32)`; the tile padding of a non-aligned W or C is never read (rule R4). The result stays row-major in the caller's config (R5). One dispatch, no `ttnn.to_layout` / `ttnn.untilize` in front of the kernel.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- Hardest generality value first (layout tier), so the light axes of Refinement 2 extend the finished TILE path once. Build the TILE address map over an **arbitrary channel range `[s3, e3)` and with `elem_bytes` from the dtype** (`input_tensor.element_size()`, already the host's source since verification) — a segment may start mid-face (`s3 % 16 != 0`) — so Refinement 2's `channels` and float32 are SUPPORTED-only on the TILE side too. Selected by a CT `input_is_tiled` (or a second kernel source) under the existing regime switch; regime A's RM path must stay byte-for-byte unchanged.
- No skill covers the NoC side: on Quasar reads are 1-B aligned (`noc_parameters.h:486-491`), so a mid-face segment needs no staging. Segments of one row are independent tagged reads under the same one-barrier-per-hart-per-block schedule — do **not** add a barrier per row or per tile.
- Wide rows (c = 4096/8192) issue `~2·ceil(c/32)` reads per row on this regime; that is accepted here and is the target of Refinement 4 (regime D), not a reason to stage tiles now.
- The golden cells of every alignment bucket (tile_aligned, w_non_aligned, h_non_aligned) at `layout=TILE` exercise the padding rule.

**Done when**: every golden cell with `layout=TILE` inside the then-current SUPPORTED passes (bfloat16, DRAM, rows, L1/height, step ones), all prior tests green, the plan tests extended with a TILE case (sentinel + session), and the kernel's DRAM bytes read on a TILE cell equal the block's logical bytes (no tile-padding over-read; a probe or test asserts the read count/bytes).

### [ ] Refinement 2 — Rest of TARGET on the gather: padded shards, cut rows, float32, L1 input, step None

**Goal**: one bundle that finishes TARGET on regimes A and C:
- add `"L1/height/padded"` to SUPPORTED["to"]: build design regime B — common RT `pad_bytes = out_row_stride - row_bytes`-driven `zero_block` (`Noc::async_write_zeros` over the hart's rows + `write_zeros_l1_barrier()`) before `gather_rows_block`; the shard memory is stale, so the zeros must be written (rule R3);
- add `"channels"` to SUPPORTED["sliced_region"] (common RT `src_offset_bytes` is already threaded; verified exact on RM by probe, including a 2-byte offset);
- add `ttnn.float32` to SUPPORTED["dtype"] (bytes are copied, never converted — rule R1; verified exact on RM by probe);
- add `"L1"` to SUPPORTED["memory_config"] (an L1-interleaved input: the TensorAccessor args already describe it; verified exact on RM by probe);
- add `None` to SUPPORTED["step"] (host only: `canonical_step` maps it; same result as "ones", rule R7).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- The skill applies to the dtype only: the op has **no compute kernel and no `compute_kernel_config`** (spec DEFINITION: "It therefore has no compute config"); do not expose one. Its pass condition (zero kernel changes) holds for float32 here.
- `memory_config="L1"` is an input-placement value with no skill; it is a SUPPORTED widening (interleaved TensorAccessor). `"L1/height/padded"` is the only kernel change in this bundle (regime B) and must hold on both the RM (A) and TILE (C) paths from Refinement 1.
- Keep the zero and the gather on the same hart over the same rows: `async_write_zeros` reprograms write cmd buffer 0 until its barrier, and the gather issues only reads — no NoC write may sit between them (design *Mechanism caps*). Build B as designed (zero, its barrier, then the gather); the overlapped per-row-tail variant is a perf lever of Refinement 3.
- Padded cells need the zero shown, not assumed: extend the sentinel plan test to a padded config (shard memory pre-filled with a non-zero sentinel) so a hart that skipped its zero range fails.
- After this refinement SUPPORTED equals TARGET; INVALID is empty, so there is no structurally impossible cell left.

**Done when**: every golden cell passes (0 xfail_expected left), the verifier CLI reports 0 loud categories, all prior tests green, and the padded sentinel test passes at R ∈ {1, 2, 6}.

### [ ] Refinement 3 — Issue parallelism of the gather on few-core configs

**Type**: perf

**Goal**: no loose case is perf-flagged, so the region is free-selected: the traced ResNet-50 calls pin the result to **1 core** (stem: `(1,115,115,32)` → 6,785 rows × 64 B on one core) and **2 cores** (`(1,64,32,32)` → 512 rows each; `(1,32,16,16)` → 128 rows of 32 B each). Each row is one 32–64 B NoC read, so the op is bound by transaction issue on the caller's cores, and today 2 harts issue them on one NoC (`Noc()` takes the kernel's single `noc_index`). Levers, all cheap (⭐ T1–T2), from the design's perf lamps and `ttnn/ttnn/operations/examples/master.md` (`split_reader`, `noc_placement`, `trid_double_issue`):
1. turn `GATHER_THREADS` (lamp 1) — possibly per call from rows per core (more harts where a core holds many rows; fewer where 32 cores hold ≤ 8 rows each), the plan still stated by one host rule;
2. hart `t` issues on NoC `t % 2` (lamp 2), if the part's DM harts can issue on both NoCs (`Noc(id)`; check the Quasar NoC parameters first — drop this lever with a note if not);
3. padded cells (regime B from Refinement 2): zero only each row's tail (`pad_bytes`, disjoint from the data bytes) interleaved with the reads, removing the zero→gather barrier (lamp 3).
No SUPPORTED change.

**Done when** (structural counts — the time gain is unconfirmed on this untimed backend; record before/after in the changelog):
- on the single-core stem config, the largest number of rows issued by one hart drops from 3,393 (2 harts) to ≤ 1,697 (≥ 4 harts at work, shown by the sentinel test at that R), and, if lever 2 holds on this part, the harts issue on both NoCs (asserted from the kernel's per-hart NoC choice by a test or probe);
- on padded cells, barriers per hart per block drop from 2 (zero, then gather) to 1, with the padded sentinel test green;
- DRAM bytes read and reads per output row unchanged (1 per row on RM), op-owned L1 still 0 B;
- golden suite green, and no regression across the config-spanning guard set: one representative per kernel path × layout × placement (RM/TILE × DRAM/L1 input × L1/height/padded), each still exact with its structural counts no worse.

### [ ] Refinement 4 — Whole-tile reads for TILE input (regime D)

**Type**: perf

**Goal**: regime C (Refinement 1) reads a tiled input as face-row segments: `~2·ceil(c/32)` DRAM transactions of 32–64 B per output row (c = 32 → 2 per row; c = 4096 → 256 per row). Build design regime D for the TILE cells where it does not over-read: each tile covering the block is read **once, whole** (one transaction per tile) into an L1 staging ring and reordered into shard rows — by the kernel library's untilize helper (`untilize_helpers.hpp`, adding a compute kernel to the plan) or by core-local DM copies; the implementer picks the realization. Predicate-gated (design: `untilize_min_c` / block shape), with regime C kept as the fallback where whole tiles would over-read (block `w`-range much shorter than a tile, e.g. W = 8). This is a ⭐⭐⭐ T3 restructure (a ring, possibly a compute kernel, a new schedule) and stands alone. No SUPPORTED change.

**Done when** (structural counts — the time gain is unconfirmed on this untimed backend; record before/after in the changelog):
- on the TILE cell `(1,64,32,32)` → `(1,32,32,32)` (W and C tile-aligned: no over-read), DRAM read transactions per output row drop from 2 to ≤ 1/16, and DRAM bytes read are not more than regime C's on that cell;
- on the wide TILE cell `(2,4,8,4096)` the predicate's choice and its counts (transactions per row, bytes read) are recorded for both regimes, and the regime with fewer bytes read is chosen when they differ;
- the new ring is ledgered in `l1_ledger.md` (size expression in the block knob, axis accounting, lifetime) and every new kernel carries the plan tests (sentinel, `num_threads`, one-session);
- golden suite green, and no regression across the config-spanning guard set (one representative per kernel path × layout × placement).
