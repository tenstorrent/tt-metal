# Operation Design: max_pool2d

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement-heavy compute (a gather of window taps + an elementwise max fold) |
| Goal | 2-D max pooling of N channels-last images given flattened as `[1, 1, N*H*W, C]`, returning `[1, 1, N*H_out*W_out, C]`, row-major, the input's dtype, sharded in L1 (R9, R11), in **one** `ttnn.generic_op` dispatch. |
| Math | `out[n, i, j, c] = max_{a<kh, b<kw} in[n, i*sh - ph + a*dh, j*sw - pw + b*dw, c]`, padded positions never win (R1). Flattened rows: input row `(n*H + r)*W + col`, output row `(n*H_out + i)*W_out + j`. `H_out = (H + 2ph - dh(kh-1) - 1)//sh + 1`, `W_out` alike. |
| Specification | `eval/specs/max_pool2d/spec.py` (governs over this document) |
| References | `.claude/references/targets/quasar.md`, `.claude/references/backends/ttsim.md`, `.claude/references/quasar_op_examples/` (toy_row_passes: rows dealt to NEOs, padded rounds, explicit/implicit protocols), `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` Part A + C2, `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, catalog `ttnn/ttnn/operations/examples/{read_coalescing,double_buffer,tile_reorder,split_reader,width_split}` |

### The two ideas the whole design rests on

1. **A window tap is a row-major block, and the max is elementwise across taps.** For a fixed tap `k = (a, b)`, the input sticks that the core's consecutive output positions need, laid end to end (position-major, `stick_pitch` elements per position), form a *tap stream* with the same layout as the core's output stream. Cut both streams into **stream tiles** of 1024 consecutive elements (one tile page, 2048 B in bf16). Then output stream tile `u` = elementwise max over `k` of tap stream tile `(u, k)`. The unpacker and packer move a page as a fixed permutation of its 1024 elements, the same one in and out, so an elementwise op on row-major bytes treated as a tile is exact. Consequences: no tilize, no untilize, no within-tile reduce, every element of every tile is useful, and the output tile **is** 1024 consecutive elements of the row-major output.
2. **The input is read into L1 once, and the window is gathered locally.** The core's receptive field (the input rows its output positions need) is staged into a per-core scratchpad once (or used in place when it already *is* the core's own shard). The `K` taps are then gathered by core-local L1→L1 copies. That costs one DRAM crossing instead of `K/(sh·sw)`. A padded position is replaced by a real tap of the same window (max is idempotent), so padding is never stored and never wins (R1).

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on the ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | `quasar` | target page |
| Compute grid | 8 × 4 = 32 clusters (probe on this backend: `compute_with_storage_grid_size()` = 8-4); queried at run time | target page; probe 2026-11-21 |
| L1 per core | 4,194,304 B | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` = 3,963,328 − 65,536 = 3,897,792 B on this target (query value from a probe on this backend, 2026-11-21) | probe |
| Alignment | L1 16 B (`ttnn.get_l1_alignment()`), DRAM 64 B (`ttnn.get_dram_alignment()`); both confirmed by the probe | target page; probe |
| Host program | `ttnn.ProgramSpec` (Metal 2.0: `ttnn.experimental.{ProgramSpec, KernelSpec, DataflowBufferSpec, ScratchpadSpec, TensorParameter, WorkUnitSpec, ProgramRunArgs}`) + `ttnn.generic_op`. Read `.claude/references/quasar_op_examples/` first (README, then `toy_row_passes`). | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1,2,4}); 6 user DM harts (R + W ≤ 6); on STRIDED rings R \| C and W \| C. DM threads of one kernel share a barrier `sync_threads()` (`tt_metal/hw/inc/api/kernel_thread_globals.h:281`). | target page *Thread plan*; header |
| Ring limits | 32 DFBs per cluster, ≤ 24 per NEO, each ring ≤ 1,048,560 B; one producer kernel and one consumer kernel per ring; 16 semaphores; kernel binaries ≤ 48 KiB (DM) / 24 KiB (compute) | target page *Cluster model* |
| Unavailable here | **Not available on this target:** `ttnn.ProgramDescriptor` and the legacy C++ program factory (Metal 1 kernels refused at creation). `bfloat8_b` / `bfloat4_b`: no path here (refuse in validate, queue no refinement). `eltwise/binary/sfpu/minmax.hpp` (the chain's `BinaryMax` element): not used; the fold is written with LLK calls. `UnpackToDest`: not used; the default unpack is used. `ReduceFp32Mode::Accurate` is guarded out. MAX+REDUCE_ROW Accumulate is refused. uint16/uint32 are not buffer formats. `fpu_math_fidelity` is set on every compute kernel. | target page |

Every worked number below is for this target (32 clusters, 4 NEOs, 3,963,328 B unreserved). Every predicate the op evaluates queries the device.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | ttnn.Tensor | yes | `[1, 1, N*H*W, C]` (X2); PHASE0: bf16, TILE or ROW_MAJOR, DRAM/L1 interleaved or L1 height/width-sharded | — | tensor binding |
| `batch_size` (N) | int | yes | ≥ 1; PHASE0 `batch = single` (N = 1) | — | host → RT (position→image map) |
| `input_h` (H), `input_w` (W) | int | yes | ≥ 1; N·H·W = shape[2] (X2) | — | CT (`H`, `W`) |
| `channels` (C) | int | yes | = shape[3]; PHASE0 C % 32 == 0 | — | host only (C_core / pitch derive from it) |
| `kernel_size` (kh, kw) | list/tuple of 2 ints (R5) | yes | ≥ 1 (X4); dilated kernel fits the padded image (X3) | — | CT (`kh`, `kw`; `K = kh·kw`) |
| `stride` (sh, sw) | list/tuple of 2 ints | yes | ≥ 1 (X4) | — | CT |
| `padding` (ph, pw) | list/tuple of 2 ints | yes | 0 ≤ 2·p ≤ k (X1); 4-value form refused (X4) | — | CT |
| `dilation` (dh, dw) | list/tuple of 2 ints | yes | ≥ 1 (X4); PHASE0 `dilate = none` | — | CT |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` (keyword-only) | no | any `math_fidelity` / `math_approx_mode` (R4); `fp32_dest_acc_en`: PHASE0 True; float32 + False refused (R2) | `None` → `default_compute_kernel_config()` (R3): HiFi4, `fp32_dest_acc_en=True`, `math_approx_mode=False` | maps to `ComputeHardwareConfig(fpu_math_fidelity=math_fidelity, enable_32_bit_dest=fp32_dest_acc_en)`. The fidelity is always set explicitly. |

Entry-point canonicalization: the four window arguments become 2-tuples of ints before any check (R5). There is no index axis. `validate()` derives `memory_config` / `sharding` with `eval.memory.memory_axes(input_tensor)` (R8; block/nd-sharded is refused by name). It computes the derived axes with the specification's `AXES` (`INPUT_TAGGERS = AXES`, never redefined) on the entry `(tuple(shape), N, H, W, C, k, s, p, d)`. It reads `fp32_dest_acc_en` from the resolved config and `dtype` / `layout` from the tensor, then raises `NotImplementedError` outside `SUPPORTED` (R12). It raises `ValueError` for X1–X4. It never reads `math_fidelity` / `math_approx_mode` (R4).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `[1, 1, N*H*W, C]` |
| Dtype | PHASE0 bfloat16 (TARGET adds float32) |
| Layout | TILE or ROW_MAJOR (both PHASE0). In TILE, rows ≥ N·H·W and columns ≥ C are padding and are never addressed (R7). |
| Memory | DRAM interleaved, L1 interleaved, L1 height-sharded, L1 width-sharded (row-major shard orientation); any shard-grid size, one core included (R10) |

### Output

| Property | Value |
|----------|-------|
| Shape | `[1, 1, P, C]`, `P = N·H_out·W_out` |
| Dtype | the input's |
| Layout | ROW_MAJOR always (R11) |
| Memory | L1, row-major orientation, allocated with `ttnn.allocate_tensor_on_device(spec, device)` from a TensorSpec carrying the sharded memory config (R9). **Height-sharded input:** HEIGHT_SHARDED on the input's shard grid (same cores, same order), shard `[ceil(P / n_in_cores), C]`, rows dealt in order. **Width-sharded input:** WIDTH_SHARDED on the input's grid, shard `[P, Cs_in]` (Cs_in = the input's shard width). **Interleaved input:** WIDTH_SHARDED on `n_w` cores `ttnn.num_cores_to_corerangeset(n_w, grid, row_wise=True)`, shard `[P, C / n_w]`, where `n_w` = the largest divisor of `C/32` that is ≤ the grid's core count (worked: C=256 → 8 cores × 32; C=64 → 2 × 32; C=32 → 1 × 32; C=128 → 4 × 32). |

**Structural impossibilities:** none beyond the specification's `INVALID`.

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Scope names: `tensor_*`, `shard_*`, `core_*`, `block_*` (§3).

### Axes

| Axis | Character (+ reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|----------------------|-------------|---------------|-------------|-----------------|--------------|
| **Output positions** `p ∈ [0, P)` (flattened `n, i, j`; batch and both spatial output axes fold in, because the position→coordinate map is per-image) | **independent**: each output position's window is self-contained; its *input footprint* overlaps its neighbours' (halo), which makes a cut cost bytes, not a combine | `core_positions` (positions per assignment); `block_tiles` (stream tiles per NEO per staging pass) | `core_positions`: the output shard's rows (height-sharded) or all P (width/interleaved), each cut into `position_splits` whole-round sub-ranges. `block_tiles = Q` (the NEO's whole share: one staging pass) | host: `position_splits` and `block_tiles` computed once in the program builder, passed as RT args (`pos_start`, `pos_count`, `block_tiles`) | Phase 0 spreads the positions over the grid: an output shard's positions are cut into `position_splits` sub-ranges of whole rounds (C stream tiles). Sub-range 0 goes to the shard's owner, the others to spare cores in row-major order. | knob-turn (`position_splits`, `min_rounds_per_core`) |
| **Channels** `c ∈ [0, C)` | **independent**, no footprint overlap: channels never mix | `core_channels` (C_core) → `stick_pitch = align_up(C_core·esize, l1_align)/esize` | C (height-sharded); the input shard width (width-sharded); `C / n_w` (interleaved) | host, from the output shard spec (one source: the output memory config). RT/CT `stick_pitch`, `chan_start` | one channel run per output shard. This is the **primary split** for width-sharded and interleaved inputs. | knob-turn (narrower runs = more shards) |
| **Window taps** `k = a·kw + b ∈ [0, K)` | **dependent**: the max reduces over it | `taps_per_batch` (taps the reader lands per barrier and per tap-ring window) | `K` (the whole window; worked: 9 or 25) clamped by the ring cap (Mechanism caps) | host constant derived from K, CT arg | **not split across cores.** The fold is sequential in DEST within one NEO. | scheme-change (rejected: see Regimes) |
| **Channel elements within a stream tile** (1024 per tile) | independent (elementwise) | `tile_elems` = 1024 (the tile shape's element count, from `ttnn.TILE_SIZE²`; not tunable) | 1024 | host constant | — | — |

Intermediates have axes too, and are blocked and assigned:

| Intermediate | Axes | Block / assignment | Why |
|---|---|---|---|
| Staged receptive field (scratchpad `staging_window`) | input sticks `s ∈ [s_lo, s_hi)` × `C_core` channels | the whole receptive field of one block (`C·block_tiles` stream tiles). Its **copy is split across the R reader threads** (thread t copies the t-th contiguous quarter of the sticks / tile pages), then one `sync_threads()`. | no stage runs on one hart while the others wait. The copy is spread, and the barrier is a rendezvous, not a work assignment. |
| Tap stream tiles (`cb_window_taps`) | (output stream tile u, tap k) | NEO t's tiles `u ∈ [b·C·block_tiles + t·block_tiles, +block_tiles)`, `K` taps each, gathered by reader thread t | one reader feeds one NEO (R = C), so the gathers are spread four ways |
| Output stream tiles (`cb_pooled_tiles`) | u | written by writer thread w for NEOs w, w+2 | spread two ways |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_window_taps` | `tap_ring_depth` (tap windows per NEO in flight) | 2 | the reader lands tile u+1's K taps while the NEO folds tile u's (catalog `double_buffer`: a batch of reads per barrier, CB = 2 × batch) |
| `cb_pooled_tiles` | `out_ring_depth` (entries per NEO) | 2 | the NEO packs tile u+1 while the writer stores tile u |
| `staging_window` | `staging_depth` | 1 | Phase 0 stages once per block, and with `block_tiles = Q` there is one block. Depth 2 (stage block b+1 during block b's gathers) is the overlap lamp. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DFB ring size ≤ 1,048,560 B (target page) | `taps_per_batch` | `taps_per_batch` = the largest divisor of K that is ≤ `floor(1048560 / (tap_ring_depth · C · tile_bytes))` (= 63 for bf16 at C=4). A divisor, so every reserve/push is the same size and the ring cycle closes exactly. | ring creation refused, or a window straddles the ring end |
| Fold accumulators in DEST | output tiles folded at once | each output tile holds 2 DEST slots (accumulator + operand). Tiles in flight ≤ `DEST_AUTO_LIMIT / 2` (4 slots with fp32 DEST in half-sync → 1–2 tiles). The subblocking is the implementer's. | DEST overrun, silently wrong maxima |
| NoC alignment: L1 16 B, DRAM 64 B | `stick_pitch`, segment offsets | `stick_pitch` = `align_up(C_core·esize, 16)` bytes; every staging, tap and output offset is a multiple of the pitch or of a 16 B face-row chunk. For a DRAM-interleaved source, `C_core` is a multiple of 32 channels (64 B), guaranteed by choosing `n_w` as a divisor of `C/32`. | misaligned reads |
| Row-major width shards narrower than 16 B (the runner builds `[P, 1..8]` shards for C ≤ 256 in ROW_MAJOR width-sharding: `eval/memory.py:width_shards`, step 1) | `stick_pitch` vs `C_core` | `stick_pitch` = the shard's aligned page (16 B = 8 bf16). The `pitch − C_core` padding elements ride along through gather, fold and write. The write lands them in the output page's alignment padding, which holds no data. | writing only `C_core·esize` bytes per stick breaks NoC alignment |
| TILE face geometry | TILE gather segment | a stick's 32 channels in one tile are two 16-element (32 B) face-row chunks, at face `2·((s%32)/16) + 0/1`, row `s%16` | reading 64 B straight across a face boundary mixes rows |
| Stream tile = 1024 elements | tile tail | the last stream tile of an assignment is partial: the reader may leave its tail unread, the fold computes don't-care values there, and the writer stores only `P_core·stick_pitch − 1024·(T_core−1)` elements | overrun into the next shard / the next assignment's rows |
| NoC burst (`NOC_MAX_BURST_SIZE`) | coalesced segment length | plain `noc.async_read` / `async_write` split bursts themselves. The stateful single-packet path (`set_read_state`) is only for segments ≤ the burst. | — |
| Equal entries per thread on a STRIDED ring (target page) | per-NEO tile count | every NEO runs `block_tiles` tiles per block. A NEO without a real tile recomputes the assignment's last real tile, and the writer stores the same bytes again (toy_row_passes). | ring credits unbalanced |
| Staging scratchpad fits L1 | `block_tiles` | `staging_bytes(block_tiles) + ring_bytes ≤ get_max_worker_l1_unreserved_size() − l1_reserve_bytes`. `block_tiles = Q` when it fits, else the largest that fits (streamed regime). | allocation failure |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum (boundary: each input byte crosses its home tier once, each output byte is written once) | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **staged-resident** | built | the assignment's core does **not** hold its whole receptive field in its own input shard (input interleaved in DRAM/L1, or height-sharded, or a spare core of a width-sharded input) **and** `staging_bytes(Q) + ring_bytes ≤ budget` | one block per core: `C × block_tiles` stream tiles, `block_tiles = Q = ceil(T_core / C)`. Staging spans the receptive field `[s_lo, s_hi) × C_core`. | input: its receptive field crosses the home tier once per assignment, minimum + halo. The halo is `(kh_eff − sh)·W` sticks per position cut; for sharded inputs the own rows are a core-local copy and the halo is cross-core. Then K core-local gathers per output element, and 1 output write (local on owners, cross-core on spare cores) | one staging pass and one `sync_threads()` per core. A bigger block = fewer halo re-reads at cut boundaries. Inits: once per kernel. |
| **staged-streamed** | built (same code, `num_blocks > 1`) | as above but `staging_bytes(Q) + ring_bytes > budget`. **No PHASE0 or TARGET input reaches it on this target** (largest staging is SPPF-class ≈ 14 KB). Pinned by a test that lowers the module-level `_STAGING_BUDGET_OVERRIDE`. | `block_tiles` = the largest whose staging fits. `num_blocks = ceil(T_core / (C·block_tiles))`. Block b covers core tiles `[b·C·block_tiles, (b+1)·C·block_tiles)`. | staged-resident + the halo re-read once per block boundary (overlapping footprint: the extent ranking) | each extra tile per block removes a share of `(kh_eff − sh)·W` halo sticks per boundary, and two `sync_threads()` per block (staging landed / gathers done) |
| **in-place-shard** | built | input **width-sharded** and the assignment's core is the owner of its input shard (sub-range 0). Its receptive field is a subset of the own shard (all rows, own channels). | as staged-resident, with **no staging copy**: the gather reads the shard through `LocalTensorAccessor` (staging base = shard base, `s_lo = 0`, pitch = the shard's aligned page) | minimum: the input crosses nothing (already resident), K core-local gathers, output written locally (the output shard is on the same core) | per-block fixed costs only (no staging, no barrier) |
| direct-source-gather | **rejected**: dead end, superseded by staged-* everywhere | — | — | each tap re-reads its sticks from the source: `K / (sh·sw)` input crossings of DRAM (2.25× for 3×3/s2, **25×** for 5×5/s1), and ≤ kh× the halo cross-core for sharded inputs. No good scheme passes through it. | — |
| untilized-staging (TILE) | **deferred**: the built face-row gather covers every TILE shape, and the TILE callers (ResNet, C ≤ 128) are tiny | TILE input | staged tile pages → a compute untilize (`untilize_helpers.hpp`) into row-major staging, then the row-major gather | the same DRAM bytes. It halves the gather transactions (one 64 B+ stick read instead of two 32 B face-row reads per 32 channels) at the cost of one compute pass and one ring | reachable: the staging fill becomes a compute stage and the gather's address map switches to the row-major form already built for ROW_MAJOR inputs |
| separable-window | **deferred**: the built direct window covers every shape | 2-D window with `sh = sw = 1` and K ≥ 9 (SPPF) | pass 1: horizontal max over kw taps into a staged row-max field (kh_eff + rows − 1 rows); pass 2: vertical max over kh taps | gathers and folds per output tile drop from `kh·kw` to `≈ kh + kw` (25 → 10 for 5×5/s1). The input bytes are unchanged. | reachable: the same gather and fold run twice, with a `1×kw` then a `kh×1` window. It relies on max being associative and commutative. |
| tap-split-combine | **rejected**: superseded by positions × channels | — | taps split across cores, partial maxima combined over the NoC | adds `(splits − 1) ×` the output bytes cross-core and buys no residency: every tap subset still needs nearly the whole receptive field | — |
| zero-copy-output-pack | **deferred**: positive reason below | output shard bytes are a whole number of tiles and the assignment = the owner's whole shard | compute packs into a DFB `borrowed_from` the output shard | saves one core-local write of the output. It is unreachable in general: R9 fixes the shard shape exactly (`[400, 32]` = 12.5 tiles), so the last packed tile would overrun the shard allocation, and padded rounds would rewrite tiles. | reachable: the writer is the only consumer of `cb_pooled_tiles`; replacing it with a borrowed ring changes no other stage |
| in-place-own-rows (height-sharded) | **deferred**: staged-resident covers it | height-sharded input | gather own rows from the shard in place, halo rows from a halo staging, with a piecewise address map | saves one core-local copy of the own rows (1 of K+2 core-local passes). The staged copy also moves each halo row cross-core **once**, versus up to kh times for direct halo taps. | reachable: the address map gains a third region |

Regime-selection function (host, per assignment, a runtime arg `staging_source` ∈ {STAGED, IN_PLACE}): `IN_PLACE` iff the input is width-sharded **and** the assignment runs on the core holding input shard g. Otherwise `STAGED`, with `block_tiles` = Q if `staging_bytes(Q) + ring_bytes ≤ budget`, else the largest fitting value. Regime-pinned tests (required): width-sharded (in-place, plus spare-core staged sub-ranges); height-sharded and interleaved (staged-resident); streamed forced via `_STAGING_BUDGET_OVERRIDE`; one cell per layout of each.

### Traffic ranking

Counted per tensor per tier, for the candidate splits (no time estimates):

| Split | Input DRAM / home-tier crossings | Cross-core | Core-local | Verdict |
|---|---|---|---|---|
| channels only (each core: all positions × its channel run) | 1× (no footprint overlap across channels) | 0. The output is written on its owner. | K gathers + 1 write per output element | **cheapest**. Primary split wherever the output is width-sharded (width-sharded and interleaved inputs). |
| positions (each core: a position range × all its channels) | 1× + halo `(kh_eff − sh)·W·C_core` per cut | the output is remote on non-owner cores; for height-sharded input the halo rows are cross-core | same | used (a) as the **given** split for height-sharded inputs (the shard fixes it), and (b) as the occupancy step under channels: sub-ranges of whole rounds onto spare cores |
| taps (dependent) + NoC combine | 1× per core × nearly the whole receptive field | `(splits − 1)·P·C` partial maxima | same | rejected |
| no staging (direct gather) | `K/(sh·sw)`× | ≤ kh× halo | — | rejected (dead end) |

**Chosen:** channels (primary) × positions (secondary, whole rounds). The occupancy step trades bytes for engaged cores, and per blocking-model §4 that trade is stated, not hidden. Worked (SPPF 20×20×256, 5×5/s1/p2, DRAM): channels alone engage 8 clusters with input crossing DRAM 1.0×. The position sub-split (`position_splits = 4`) engages 32 clusters at 1.65× input DRAM bytes (staged rows per group 9 + 11 + 10 + 3 = 33 vs 20). The output gains 0.75× cross-core writes. With K = 25 folds per output tile the op is compute-bound per NEO (below), so the 4× NEO count is bought for 0.65× extra input reads of a 200 KB tensor. The grid-synchronization lamp measures `position_splits = 1`.

### Block schedule

Logical schedule per core. Reader, compute and writer are separate asynchronous kernels.

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    stage_receptive_field(block_idx);   // reader ×R: copy [s_lo, s_hi) × C_core into staging_window (skipped IN_PLACE); sync_threads()
    gather_window_taps(block_idx);      // reader thread t → NEO t: block_tiles stream tiles × K taps into cb_window_taps
    fold_window_max(block_idx);         // compute NEO t: block_tiles output tiles, each = max over K taps, in DEST
    store_pooled_rows(block_idx);       // writer ×W: block_tiles tiles per NEO → output rows (exact length)
    // streamed only: sync_threads() before the next stage_receptive_field overwrites staging_window
}
```

| Block operation | Block shape | Resident across it | Intended fixed-cost frequency |
|---|---|---|---|
| `stage_receptive_field` | sticks `[s_lo, s_hi)` × `C_core` (ROW_MAJOR: one segment per stick, coalesced over runs from `TensorAccessor::num_contiguous_pages`; TILE: whole tile pages for tile rows `[s_lo/32, (s_hi−1)/32]` × `C_core/32` channel tiles), split into R contiguous quarters by thread | staging_window, for the whole block | 1 trid barrier + 1 `sync_threads()` per block (Phase 0: per core). Zero for IN_PLACE. |
| `gather_window_taps` | per NEO: `block_tiles` × K stream tiles. Each tap tile = 1024 elements = the positions it covers × `stick_pitch`, read as core-local segments, consecutive positions coalesced when their source sticks are contiguous (`sw = 1`, same image row, no padding substitution: one read per run, catalog `read_coalescing`) | staging_window (read-only) | 1 `reserve_back(taps_per_batch)`, 1 trid barrier, 1 `push_back(taps_per_batch)` per `taps_per_batch` taps (Phase 0: per output tile) |
| `fold_window_max` | per NEO: `block_tiles` output tiles. Each: acquire DEST; copy tap 0 → acc; for k ≥ 1: copy tap k → operand, `binary_max(acc, operand) → acc`; pack acc | the accumulator in DEST, across the K taps of one output tile | `compute_kernel_hw_startup` once. `copy_init` + `binary_max_tile_init` + `pack_init(cb_pooled_tiles)` intended **once per kernel**: re-arm per tap only if the Quasar LLK requires it after an SFPU op (verify on non-monotonic data). DEST acquire/release once per output tile (or per DEST subblock). Ring wait/pop 1 per tap, push 1 per output tile. |
| `store_pooled_rows` | per entry: one stream tile → the output rows it covers. One write per contiguous run (within an output shard rows are contiguous at the page pitch, so normally one 2048 B write), the last tile partial | — | 1 `wait_front(1)`, 1 trid barrier, 1 `pop_front(1)` per tile |

Stall-shadow check: `gather_window_taps` waits on `sync_threads()` after staging. A tile's taps span every thread's quarter, so no gather is independent of the barrier. The streamed regime's remedy is the overlap lamp (stage block b+1 during block b). `fold_window_max` waits on taps, and max being commutative and associative lets it fold taps in arrival order, which is the order they land. `store_pooled_rows` has nothing independent. Max is exact in any order, so no reorder touches precision.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization: `position_splits` fills the grid in whole rounds | PHASE0 outputs are 0.5–100 stream tiles, so a core may get 1 round (or less, padded). Halo re-reads (1.65× on SPPF) and remote writes may outweigh the extra NEOs. | `position_splits = 1` (channels only, owners only); `min_rounds_per_core = 2` |
| Grid synchronization: thread plan (4, 4, 2) on tiny assignments | a height-sharded input spread over 32 cores gives T_core = 1. Three NEOs then recompute a pad tile and three readers duplicate its gathers. | plan (1, 1, 1) when `max T_core < C` (the plan is one host constant) |
| Overlap: one staging pass per block, `staging_depth = 1` | for the streamed regime, gathers stall on each block's staging | `staging_depth = 2` (stage block b+1 while gathering b) |
| Gather transaction size | ROW_MAJOR `sw > 1` and TILE gathers issue one read per position (TILE: two 32 B face rows per 32 channels). This is issue-bound on the reader (catalog `split_reader`: confirm the hart is issue-bound first). | stateful single-packet reads (`dataflow_kernel_lib::set_read_state` / `read_with_state`) for fixed-size segments. For TILE: the untilized-staging regime row. |
| Reader barrier batch `taps_per_batch = K` | K = 25 means 25 tiles of reads per barrier. A smaller batch may overlap better with the fold. | `taps_per_batch` = a divisor of K near 4–8 (catalog `double_buffer`) |
| Writer per-tile barrier | 1 barrier per 2 KB write | `wait_front(2)` / one barrier per 2 tiles |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Input home → `staging_window` (STAGED) | ROW_MAJOR sticks of `stick_pitch` (channels `[chan_start, chan_start + C_core)` of each input row), or whole TILE pages | reader ×R, `noc.async_read<NocOptions::TXN_ID>` with `{.trid = 0}` from `TensorAccessor(tensor::input)` addresses (`get_noc_addr(page, offset)`) into the scratchpad. Runs coalesced with `num_contiguous_pages` where the source is sharded. Barrier on trid 0, then `sync_threads()`. | one data path for every placement: a DRAM bank, an L1 bank, a neighbour's shard, or the own shard (spec GUIDANCE) |
| Own input shard (IN_PLACE) | the shard as stored | `LocalTensorAccessor(tensor::input)` gives the shard's local base. No copy. | the physical shard is consumed in place |
| staging → `cb_window_taps` | stream tiles: position-major, `stick_pitch` per position, 1024 elements per tile | reader thread t → NEO t: self-aimed L1 reads (`dataflow_kernel_lib::local_addr`, tagged trid 0) into the reserved window. Explicit sync (`reserve_back` / `push_back`). The ring is listed in `disable_dfb_implicit_sync_for`. | ROW_MAJOR source: 1 segment per position (or per coalesced run). TILE source: 2 × 32 B face-row segments per 32 channels per position. Padding taps take the nearest in-image tap of the same window: for dilation 1 that is the clamp of the row/column into `[0, H−1]` / `[0, W−1]`. In general it is the first/last in-image `a`/`b` of that window. |
| `cb_window_taps` → DEST → `cb_pooled_tiles` | bf16 pages, fp32 DEST | compute: `copy_tile` (default unpack, **not** UnpackToDest) + `binary_max_tile` fold + `pack_tile` | the max selects an input value exactly, so a bf16 output loses nothing from fp32 DEST |
| `cb_pooled_tiles` → output | row-major output rows, `stick_pitch` per row | writer ×W, explicit sync (`wait_front` / `pop_front`), `noc.async_write<NocOptions::TXN_ID>` `{.trid = 0}` to `TensorAccessor(tensor::output)` row addresses, exact length. Local on owners, cross-core on spare cores. | the output is sharded in L1, so every write is to a core's L1 |

Every NoC transfer in the reader and writer carries the user id (`NocOptions::TXN_ID`, `{.trid = 0}`), and barriers wait on the same id (target page: explicit-sync rule). Each DM kernel's last statement hands the tag back (`noc_async_read_set_trid(0, noc.get_noc_id())` in the reader, the write form in the writer).

TARGET placements as regime rows: `memory_config ∈ {DRAM, L1}` × `sharding = interleaved` → staged-resident, a knob-turn (placement only: the accessor resolves the bank). `sharding = height` → staged-resident with halo, a knob-turn: the position axis is independent, and the halo is footprint overlap, not a combine. `sharding = width` → in-place-shard, a knob-turn: the channel axis is independent with no footprint overlap. No TARGET placement needs a scheme-change. Block sharding is outside TARGET (refused by name, R8).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | an **assignment** `(pos_start, pos_count, chan_start, C_core, output owner)`. Inside it, **stream tiles** of 1024 elements: `T_core = ceil(pos_count · stick_pitch / 1024)` |
| Grid | `device.compute_with_storage_grid_size()` (8×4 here). Owner cores: the output shard grid (R9). Spare cores: the rest of the grid, row-major. |
| Per-core work | **Output shards:** height-sharded → `n_in` shards of `ceil(P/n_in)` positions × C. Width-sharded → `n_in` shards of all P × `Cs_in`. Interleaved → `n_w` shards of all P × `C/n_w`. **Sub-ranges:** `position_splits = min(floor(grid_cores / n_shards), ceil(T_shard_max / (C · min_rounds_per_core)))`, ≥ 1. Shard s's positions are cut in quanta of `C·1024 / gcd(C·1024, stick_pitch)` positions (whole rounds), sub-range 0 on the owner, the rest on spare cores. **Inside a core:** `Q = ceil(T_core / C)`. Block b, NEO t → tiles `[b·C·block_tiles + t·block_tiles, +block_tiles)`. |
| Remainder | every count uses `ceil` (`H_out`, `W_out` by the floor formula of the spec; `T_core`, `Q`, shard height `ceil(P/n_in)`). The last sub-range / last stream tile is partial. A NEO past the last real tile recomputes the core's last real tile, so each NEO takes the same `block_tiles` per block, and the writer re-stores identical bytes. A core with `pos_count = 0` gets no assignment (it is left out of the work unit). |
| Bound | **compute**: each output tile costs K (9–25) unpack + datacopy + SFPU max passes against one 2 KB write. The K tap tiles it reads are core-local copies, not DRAM traffic. "Several passes, SFPU work per tile" (target page). The op reads K tiles per tile written. |
| Thread plan | **(R, C, W) = (4, 4, 2)**, host constant `THREAD_PLAN` in the program builder, passed to the kernels as CT args `reader_threads`, `compute_threads`, `writer_threads`. Reader thread t feeds NEO t (R = C on a STRIDED ring); writer thread w drains NEOs w, w+2. R + W = 6 ≤ 6, 4 \| 4, 2 \| 4. |
| Unit dealt to compute threads | one **output stream tile**, which is larger than a ring entry: K `cb_window_taps` entries in, 1 `cb_pooled_tiles` entry out. So every NEO takes `block_tiles` tiles per block, padded as above. The host deals **whole rounds** (C tiles) to sub-range cores, so padding happens once per output shard, not once per core. |
| Tests the plan needs | (1) A cell where every NEO of every engaged core owns a distinct real tile, checked exactly (SPPF interleaved: 3 of 4 sub-ranges per group have exactly 4 tiles): a missing NEO or reader leaves its tile unwritten. (2) In one device session: the op after another program (e.g. `ttnn.to_layout` of the input), and the op followed by `ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)` / a second `max_pool2d` on its own output (R10). (3) R10: height-sharded and width-sharded inputs on **one** core, and pooling a pool's result (height- and width-sharded). (4) The regime-pinned tests listed under Regimes. |

Worked Phase 0 values (quasar, 32 clusters):

| Input (spec INPUTS) | Placement | Shards × C_core | position_splits | Engaged | T_core (max) | Q | K |
|---|---|---|---|---|---|---|---|
| 1×20×20×256, 5×5/s1/p2 | DRAM interleaved | 8 × 32 | 4 | 32 | 4 | 1 | 25 |
| 1×8×8×64, 5×5/s1/p2 | L1 interleaved | 2 × 32 | 1 (T_shard = 2 < C) | 2 | 2 | 1 | 25 |
| 1×32×32×64, 3×3/s2/p1 | TILE height-sharded (runner: 32 shards of 32 rows) | 32 × 64 | 1 | 32 | 1 | 1 | 9 |
| 1×32×32×64, 3×3/s2/p1 | TILE height-sharded, one core (R10) | 1 × 64 | 4 | 4 | 4 | 1 | 9 |
| 1×32×32×32, 3×3/s2/p1 | ROW_MAJOR width-sharded (runner: 32 shards of 1 channel, pitch 8) | 32 × 1 (pitch 8) | 1 | 32 | 2 | 1 | 9 |
| 1×32×64×32, (3,5)/(2,1)/(1,2) | DRAM interleaved | 1 × 32 | 8 | 8 | 4 | 1 | 15 |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `cb_window_taps` (DFB) | 0 | `tile_bytes` (2048 bf16) | `tap_ring_depth · taps_per_batch · C` (Phase 0: 2·K·4) | spans taps (one window per batch, double-buffered). Streams over positions/tiles and channels (within the 1024-element entry). Counted per NEO × C. | input dtype (bf16) | reader | compute | reader: explicit (`reserve_back(taps_per_batch)` / `push_back`, ring in `disable_dfb_implicit_sync_for`, all transfers tagged trid 0). Compute: explicit `wait_front(1)` / `pop_front(1)` per tap. | whole kernel |
| `cb_pooled_tiles` (DFB) | 1 | `tile_bytes` | `out_ring_depth · C` (Phase 0: 8) | one output tile per NEO, double-buffered. Streams over everything. | output dtype (= input, bf16) | compute | writer | compute: explicit `reserve_back(1)` / `push_back(1)`. Writer: explicit `wait_front(1)` / `pop_front(1)` (exact-length segment writes; ring in `disable_dfb_implicit_sync_for`, tagged trid 0). | whole kernel |
| `staging_window` (ScratchpadSpec, DM-only, no format metadata) | — | bytes | `staging_bytes(block_tiles)` = ROW_MAJOR `(s_hi − s_lo)·stick_pitch·esize`; TILE `(tr_hi − tr_lo + 1)·(C_core/32)·tile_bytes`, maximised over the core's blocks. Allocated on every engaged core; unused on IN_PLACE cores. | spans the block's receptive-field sticks and the channel run | n/a | reader (all R threads write disjoint quarters) | reader (all R threads read, after `sync_threads()`) | intra-kernel: trid-0 barrier then `sync_threads()` before any read. Streamed: a second `sync_threads()` before the next overwrite. | one block |

Push = wait per NEO: `cb_window_taps` Q·K pushed (in `push_back(taps_per_batch)` units) and Q·K waited/popped (in units of 1). `cb_pooled_tiles` Q pushed, and each writer thread pops Q per NEO it serves. Ring cycles close: capacities are whole multiples of the push units.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `stage_receptive_field` | `[s_lo, s_hi) × C_core`, R quarters | raw dataflow (NoC reads via TensorAccessor) | input tensor (accessor) | `staging_window` (block bytes) | staging valid for the block after `sync_threads()` |
| 2 | `gather_window_taps` | per NEO `block_tiles × K` stream tiles | `dataflow_kernel_lib::local_addr` (+ trid tagging) | `staging_window` or the own shard (read-only) | `cb_window_taps`, K pages per tile | K entries pushed per tile |
| 3 | `fold_window_max` | per NEO `block_tiles` output tiles; DEST: 2 slots per tile | raw_api (justified below) | `cb_window_taps`, 1 page waited/popped per tap | `cb_pooled_tiles`, 1 page per tile | taps consumed |
| 4 | `store_pooled_rows` | 1 stream tile per entry | raw dataflow (NoC writes via TensorAccessor) | `cb_pooled_tiles`, 1 page | output tensor rows | popped |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| host | helper | `ttnn.experimental.ProgramSpec`, `KernelSpec(num_threads=…)`, `DataflowBufferSpec`, `ScratchpadSpec`, `producer_of` / `consumer_of`, `DataMovement2XXConfig(disable_dfb_implicit_sync_for=[...])` | `ttnn/cpp/ttnn-nanobind/program_spec.cpp:300-347` (DFB), `:364-398` (Scratchpad), `:439-450` (implicit-sync opt-out), `:745-783` (KernelSpec), `:798-829` (binding helpers) | as `toy_row_passes_program_artifacts.py` | — | — | `num_threads` = THREAD_PLAN; `num_entries` from depth knobs |
| host | helper | `ttnn.generic_op([input, output], spec, run_args, {"input": 0, "output": 1})` | `.claude/references/quasar_op_examples/toy_row_passes/toy_row_passes.py` | one dispatch | — | — | — |
| 1 | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)`, `num_contiguous_pages` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165, 186-190` | page = input row (ROW_MAJOR) or tile (TILE) | input | staging_window | segment = stick / run / tile page |
| 1 | raw_api | `Noc::async_read<NocOptions::TXN_ID>(…, {.trid = 0})`, `Noc::async_read_barrier({.trid = 0})` | `tt_metal/hw/inc/api/dataflow/noc.h:252, 739-742` | trid 0 | input | staging_window | — |
| 1, 2 | raw_api | `Scratchpad<T>` (staging base), `LocalTensorAccessor<T>` (own shard base, IN_PLACE) | `tt_metal/hw/inc/api/scratchpad.h:42`; `tt_metal/hw/inc/api/tensor/local_tensor_accessor.h:47-60` | — | — | — | — |
| 1, 2 | raw_api | `sync_threads()`, `get_my_thread_id()` | `tt_metal/hw/inc/api/kernel_thread_globals.h:281, 66` | — | — | — | — |
| 2 | helper | `dataflow_kernel_lib::local_addr(addr, noc.get_noc_id())` (self-aimed source) | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57` | — | staging | cb_window_taps | segment length (coalesced run) |
| 2 | raw_api | `DataflowBuffer::reserve_back` / `push_back` / `get_write_ptr` | `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:243-244, 407` | `taps_per_batch` | — | cb_window_taps | `taps_per_batch` |
| 3 | raw_api | `compute_kernel_hw_startup(cb_window_taps, cb_pooled_tiles)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:113` | — | cb_window_taps | cb_pooled_tiles | — |
| 3 | raw_api | `copy_init(cb_window_taps)`, `copy_tile(cb_window_taps, 0, dst)` | `tt_metal/hw/inc/api/compute/tile_move_copy.h:47, 126` (Quasar path `:55-60`: no transpose) | front tile only (index 0, so no strided indexing) | cb_window_taps | DEST | — |
| 3 | raw_api | `binary_max_tile_init()`, `binary_max_tile(acc, operand, acc)` | `tt_metal/hw/inc/api/compute/binary_max_min.h:145, 126` | `VectorMode::RC` | DEST | DEST | — |
| 3 | raw_api | `tile_regs_acquire/commit/wait/release` | `tt_metal/hw/inc/api/compute/reg_api.h:50, 96, 63, 106` | — | — | — | tiles per DEST window ≤ `DEST_AUTO_LIMIT/2` |
| 3 | raw_api | `pack_init(cb_pooled_tiles)`, `pack_tile(acc, cb_pooled_tiles)` | `tt_metal/hw/inc/api/compute/pack.h:32, 87` | — | DEST | cb_pooled_tiles | — |
| 3 | raw_api | `DataflowBuffer::wait_front(1)` / `pop_front(1)` / `reserve_back(1)` / `push_back(1)` | `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:243-246` | — | both | both | — |
| 4 | raw_api | `TensorAccessor::get_noc_addr`, `num_contiguous_pages`; `Noc::async_write<NocOptions::TXN_ID>` + barrier `{.trid = 0}` | `tensor_accessor.h:165, 186`; `noc.h:411` | page = output row (shard row) | cb_pooled_tiles | output | write length (run) |

**Helpers considered and rejected, op 3 (`fold_window_max`):**

| Candidate | File:line of the mismatch | Why it cannot express the shape |
|---|---|---|
| `eltwise_chain` / `binary_sfpu<BinaryMax>` (`eltwise/binary/sfpu/minmax.hpp`) | — | not available on Quasar. |
| `eltwise_chain` with a kernel-local max element on the `BinaryOp` base (`eltwise/core/chain.inl:652`) | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:16-17` (the chain owns one dst-sync window per call); `:302-313` (`DestAccumulation` is "Scope of an **FPU binary**'s persistent DEST accumulation"); `:516` (`BinaryFpuOp { Add, Sub, Mul }`, no max) | a fold of K inputs into one held DEST slot needs either a persistent DEST accumulation for an SFPU op (absent) or 2K−1 compile-time chain elements per call (K up to 25). Neither is a block operation the chain provides. |
| `reduce<PoolType::MAX, …>` (`reduce_helpers_compute.hpp`) | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:20` (REDUCE_ROW reduces the W dimension within/across tiles of a row) | reduce collapses elements *within* tiles (row/col/scalar). This op needs an elementwise max *across* K tiles with every element kept. MAX+REDUCE_ROW Accumulate is also refused on Quasar. |

What would close the gap: a `BinaryMax` chain element available on Quasar plus an SFPU persistent-DEST fold scope in the chain. The implementer may wrap the raw fold in a kernel-local `fold_window_max(block_tiles)` function. The anti-gaming clause applies: inits are not repeated per tap unless the LLK requires it.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| A STRIDED ring's per-thread entries are not adjacent in L1 | with C = 4, NEO t's entries are slots t, t+4, … (DataflowBuffer.md C2: `stride_in_entries = max(P, C)`). A reader that fills its `reserve_back(K)` window assuming contiguity corrupts the other NEOs' taps. | address entry k of the reserved window as `get_write_ptr() + k · 4 · tile_bytes`. The window never straddles the ring end because capacity per NEO is a whole number of windows. Test with K ≥ 2 on several NEOs. Fallback if uncertain: `taps_per_batch = 1` (one entry per reserve). |
| The DEST fold reloads slot `operand` every tap | this design uses the default unpack (srcA → datacopy) | never enable UnpackToDest. The golden suite's normal-distributed data is non-monotonic, so a reload failure would show as wrong maxima at K = 9/25. |
| Padding must lose, also to negatives (R1) | zero-fill would win on all-negative windows | padding taps re-read a real tap of the same window; no padded stick is ever stored. Covered by the `all_negative` regression. |
| Row-major width shards of 1–8 channels | the runner deals C=32..256 over 32 cores, 1..8 channels each (`eval/memory.py:width_shards`) | `stick_pitch` = aligned page (16 B). Gather/fold/write carry the pitch padding, and the output page has the same pitch. |
| Explicit-sync rings after implicit-sync programs | an untagged transfer inherits a ring's tag | every transfer is tagged trid 0. The tag is handed back as the last statement. The one-session tests in Work Distribution. |
| Output written by several cores into one shard | spare-core sub-ranges write into the owner's shard while the owner writes its own rows | sub-ranges are disjoint row ranges. Pad tiles rewrite identical bytes of the same core's own range only. |
| Halo on height-sharded inputs spans several neighbour shards | shards can be shorter than one image row (400 rows on 25 cores = 16 rows; W = 20) | staging reads every stick through the accessor, page by page or in `num_contiguous_pages` runs, so no shard count is assumed (R10) |
| Fidelity | set on every compute kernel | map `math_fidelity` from the resolved config into `ComputeHardwareConfig.fpu_math_fidelity` always |
| `pack_init` target | raw pack, one output ring | call `pack_init(cb_pooled_tiles)` once before the loop |
| Ragged per-core tile counts | sharded placements fix T_core per core (T_core may be 1 with C = 4) | pad rounds as specified. Nominal per-NEO counts are identical (`block_tiles`). |
