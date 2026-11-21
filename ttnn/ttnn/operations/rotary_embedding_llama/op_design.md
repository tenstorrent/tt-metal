# Operation Design: rotary_embedding_llama

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (one matmul + two row-broadcast products + one add per tile), on L1-resident shards: zero data movement |
| Goal | Decode-mode Llama rotary embedding, `out = x * cos + (x @ T) * sin`, `T` applied to every run of 32 channels. All four inputs and the output are height-sharded in L1 on one core. The op works on them where they are, in one `ttnn.generic_op` dispatch. |
| Math | `out[0,u,h,32j+c] = x[0,u,h,32j+c]·cos[0,u,0,32j+c] + (Σ_{k<32} x[0,u,h,32j+k]·T[0,0,k,c])·sin[0,u,0,32j+c]` |
| References | `eval/specs/rotary_embedding_llama/spec.py` (authoritative); `.claude/references/targets/quasar.md`; `.claude/references/backends/ttsim.md`; `.claude/references/quasar_op_examples/` (README, `toy_thread_plan`); `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` (Part A); zero-copy sharded pattern: `ttnn/cpp/ttnn/operations/experimental/quasar/untilize/device/factories/untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical_program_factory.cpp:22-140` + `.../untilize/device/kernels/dataflow/reader_unary_sharded_metal2.cpp`; perf catalog `ttnn/ttnn/operations/examples/master.md` (`compute_fusion`, `compute_block_size`) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8×4 = 32 clusters (worked numbers only). The op engages the input shard's core set, read from `input_tensor.memory_config().shard_spec.grid`, never assumed to be (0,0). | target page; run-time: `device.compute_with_storage_grid_size()` |
| L1 per core | 4,194,304 B (4 MiB) | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` (value read at run time; the page gives no number). The op adds at most 16 KiB of its own L1 (see ledger) and has **no** budget predicate. | target page (query); value `unknown` |
| Host program | `ttnn.ProgramSpec` (`ttnn.experimental.ProgramSpec` / `KernelSpec` / `DataflowBufferSpec` / `WorkUnitSpec`) dispatched by one `ttnn.generic_op(tensors, spec, run_args, tensor_indices)`. Reference: `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py` | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1,2,4}), 6 user DM harts (R + W ≤ 6). On STRIDED rings R and W each divide C. An ALL ring fans out to at most 4 consumers (`DataflowBuffer.md:147`). | target page, *Thread plan* |
| Unavailable here | **(1)** `ttnn.ProgramDescriptor` / Metal 1 kernels / legacy program factory: refused at kernel creation (target page). **(2)** `matmul_block_helpers.hpp`: not used; the matmul is written with LLK calls. **(3)** `UnpackToDest` of a bfloat16 page into 32-bit DEST: not used; every ring of this op uses the default unpack mode. **(4)** `bfloat8_b` / `bfloat4_b`: no path on this part (target page). | target page |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | `[1, users, heads, head_dim]`, rank 4 (X10), dim0 = 1 (X5), heads ≤ 32 (X6), head_dim % 32 == 0 (X4), head_dim ≤ 128 (X7) | — | host |
| `cos_cache` | `ttnn.Tensor` | yes | `[1, users, 1, head_dim]` (X2), same shape as `sin_cache` (X1) | — | host |
| `sin_cache` | `ttnn.Tensor` | yes | `[1, users, 1, head_dim]` | — | host |
| `trans_mat` | `ttnn.Tensor` | yes | `[1, 1, 32, 32]` (X3) | — | host |
| `is_decode_mode` | bool, keyword-only | no | `True` only (R4); `False` is refused by name (X8) | `False` | host (axis `is_decode_mode`) |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` or device config (`WormholeComputeKernelConfig`), keyword-only | no | Any. Read `fp32_dest_acc_en`, `math_fidelity`, `math_approx_mode` as attributes (R7), never by type. | `None` → `default_compute_kernel_config()` = HiFi4, `fp32_dest_acc_en=True`, `math_approx_mode=False` (R5) | `fpu_math_fidelity`, `enable_32_bit_dest`, `sfpu_precision_mode` on the compute `KernelSpec` hw_config |

Host-side rules:

- **R5:** `default_compute_kernel_config()` is exported from the op module and is the only place the default lives.
- **R7:** normalize any accepted config to a `ttnn.ComputeConfigDescriptor` from its three fields, ignoring `packer_l1_acc` / `throttle_level`. Map it onto `ComputeHardwareConfig(fpu_math_fidelity=…, enable_32_bit_dest=…, sfpu_precision_mode=…)`, which is the Metal 2.0 kernel's config.
- **R6:** `validate()` gates on neither fidelity nor approx mode.
- **Support check:** `INPUT_TAGGERS = AXES` (imported from the spec). The axes `memory_config` / `sharding` come from `eval.memory.memory_axes(input_tensor)` (R8), and `fp32_dest_acc_en` from the resolved config. There are no index axes to canonicalize.

## Tensors

### Input

| Property | `input_tensor` (x) | `cos_cache`, `sin_cache` | `trans_mat` (T) |
|----------|-------------------|--------------------------|-----------------|
| Shape | `[1, users, heads, head_dim]` | `[1, users, 1, head_dim]` | `[1, 1, 32, 32]` |
| Dtype | bfloat16 (Phase 0); float32 in TARGET | same as x (R8) | same as x |
| Layout | TILE | TILE | TILE |
| Memory | L1 HEIGHT_SHARDED, one shard per user = `[32, head_dim]` (one tile row) | L1 HEIGHT_SHARDED on **the same core set** as x, shard `[32, head_dim]`; row 0 is data, rows 1..31 are tile padding | L1 HEIGHT_SHARDED on the same core set, shard `[32, 32]` |

R8: anything placed otherwise, or differing from x in dtype or layout, is refused with `NotImplementedError`. Shapes are checked before placement so that X1–X7 and X10 raise on shape.

### Output

| Property | Value |
|----------|-------|
| Shape | = `input_tensor.shape` |
| Dtype | = input dtype |
| Layout | TILE |
| Memory | = `input_tensor.memory_config()` exactly: shard shape, grid, orientation (R9). Allocated with `ttnn.allocate_tensor_on_device(input_tensor.spec, device)`. |

## Blocking Model

**Note on the numbers.** Worked values are for `quasar`. `tensor_dim_tiles` (Wt) = `head_dim / 32` ∈ {1,2,3,4} (X4, X7). One user's head rows are one tile row (X6). `users = 1` in all of TARGET.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `user` (x dim 1) | independent: no result spans users; each user has its own cos/sin row | `users_per_core` | 1 (the shard holds one user; TARGET `users = [1]`) | host: `shard_spec.shape[0] / 32` (one tile row per user) | the shard's core: caller-pinned by the height shard; one core per user | knob-turn (users > 1 is one shard per user per core; see regime `multi_user_shards`) |
| `head_row` (x dim 2, padded to one tile row) | independent: rows are rotated independently; cos/sin are row-broadcast within the tile | `head_row_tiles` | 1 = the whole extent (≤ 32 heads = one tile row, X6). Not 1 by choice: it is the full axis. | host: `ceil(heads / 32)`, bounded to 1 by validate (X6) | not split: a tile row is the indivisible unit of the FPU | knob-turn (heads > 32, regime `multi_tile_row_heads`) |
| `head_dim_tile` j (x dim 3 / 32) | independent: `T` acts within one 32-run; cos/sin vary per column, so tile j needs only x_j, cos_j, sin_j, T | `tensor_dim_tiles` (per core), `neo_dim_tiles` (per NEO) | per core: Wt (the whole shard); per NEO: Wt / C | host: `tensor_dim_tiles = head_dim // 32`; `neo_dim_tiles = tensor_dim_tiles // compute_threads` (CT arg) | not split across cores: the shard pins it, and moving it would add NoC traffic to resident data (regime `split_dim_tiles_across_cores`, rejected). **Dealt across the core's C NEOs** by the STRIDED rings (tile j → NEO j mod C). | — (already spread at the finest legal level) |
| `channel_k` (the 32 channels contracted by `T`) | dependent: every output column sums over all 32 k | `contraction_tiles` | 1 (one 32×32 tile = the whole contraction; the op's definition fixes the run at 32) | fixed by the op (`T` is 32×32, X3); a named constant, not a literal in loops | not split: the contraction is inside one tile and one `matmul_tiles` call | — (no wider contraction exists) |
| `trans_mat` (operand, all axes) | reuse-shared: identical for every j, every head, every user | — (1 tile) | 1 tile, resident | shard | one L1 copy per core, fanned out to all C NEOs by an **ALL** ring: no copy, no re-read from outside L1 | multi-user: caller-replicated per core (stepping stone) → mcast (deferred) |

**Intermediate data, axes re-run.** `rotated = x @ T` has x's axes (`head_row` × `head_dim_tile`), and it stays independent along j. Each NEO computes the rotated tiles of its own j share and consumes them itself. No core or NEO collects, coordinates or hands out work, so there is no idle-while-one-works stage.

**Operand-reuse check** (split = j over NEOs; later users over cores):

| Operand | Varies along j? | Varies along user? | Consequence |
|---------|-----------------|--------------------|-------------|
| x | yes | yes | dealt (STRIDED) |
| cos, sin | yes | yes | dealt (STRIDED), aligned with x by construction (same ring geometry, same dealing) |
| T | **no** | **no** | reuse-shared: ALL ring inside the core (built). Across cores: caller-replicated shard per core, then mcast (deferred row). |

**Stall-shadow check.** No stage waits on anything outside the core. The reader only posts credits for data that is already resident, and the writer only drains credits. Inside a NEO, `rotate_combine_block` waits on `cb_rotated`, which the same NEO's `rotate_block` packs. The `x·cos` half of the combine does not depend on that wait, but it already runs in the same DEST section as `rotated·sin`, so there is no stall to fill. No reorder is made, and the floating-point order is the reference's: two products, then one add.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles`, `cb_cos_tiles`, `cb_sin_tiles`, `cb_output_tiles` | none: borrowed rings, depth = the whole shard (`tensor_dim_tiles` entries) | Wt | nothing to tune. The ring *is* the tensor's storage, and all of it is resident before launch. |
| `cb_trans_mat` | none: borrowed, 1 entry | 1 | — |
| `cb_rotated` | `rotated_depth_blocks` | 1 block (= `neo_dim_tiles` entries per NEO) | Between two sequential phases of the same compute kernel, which cannot overlap (`ttnn-cb-memory-fundamentals.md`: sequential helpers need a full block). A deeper ring buys nothing. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| STRIDED ring capacity per NEO = `num_entries / C` (`DataflowBuffer.md:251-255`). A borrowed ring's `num_entries` is fixed at the shard's page count, Wt (`ttnn/core/tensor/tensor_utils.cpp:131-136`). | `compute_threads` C | `compute_threads = max(c for c in (4, 2, 1) if c <= THREAD_PLAN.compute and tensor_dim_tiles % c == 0)`. Worked: Wt=4 → 4, Wt=2 → 2, Wt∈{1,3} → 1. | the host rejects the ring |
| `llk_unpack_AB_matmul` adds `tile_index` to the ring cursor (`tt_metal/hw/ckernels/quasar/metal/llk_api/llk_unpack_AB_matmul_api.h:191-194`). | in-block tile index of `matmul_tiles` on `cb_input_tiles` | pass `k * dfb_input.get_consumer_stride_tiles()` (`api/dataflow/dataflow_buffer.h:240`) for the k-th tile of the NEO's share. `cb_trans_mat` is ALL (stride 1, index 0). | — |
| DEST capacity, matmul phase: one DEST slot per output tile held at once. `DEST_AUTO_LIMIT` = 4 (fp32, half sync) / 8 (16-bit) (`ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:89-103`). | tiles per DEST section in `rotate_block` | `section_tiles = min(neo_dim_tiles, DEST_AUTO_LIMIT)`. With X7 (Wt ≤ 4) the share always fits one section, but the clamp is written anyway. | — |
| DEST capacity, combine chain: lane width 2 (D0, D1) | chain block size | the chain clamps `block_size * lane_width ≤ DEST_AUTO_LIMIT` itself (`chain.hpp` §3). The caller passes `neo_dim_tiles`. | — (clamped by the helper) |
| ALL ring: ≤ 4 consumers (`DataflowBuffer.md:147`) | C on `cb_trans_mat` | C ≤ 4 by the plan | host rejection |
| DM STRIDED side share = 1 entry per `push_back` / `pop_front` (`tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl:390-396`, `ASSERT(num_entries == get_producer_share())`) | entries per credit call in the reader / writer | one `reserve_back(1)`/`push_back(1)` (reader) or `wait_front(1)`/`pop_front(1)` (writer) per entry, Wt times per ring | watcher assert |
| Head rows ≤ 32 (X6), head_dim ≤ 128 (X7) | `head_row_tiles = 1`, Wt ≤ 4 | validate() refuses beyond | — |
| Compute kernel binary ≤ 24 KiB (target page) | number of distinct LLK op inits in one kernel (matmul + 2× bcast mul + SFPU add + pack) | none needed at this inventory. Watch the build. | build failure |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `resident_shard_in_place` | **built** | `users == 1 and heads <= 32 and head_dim % 32 == 0 and head_dim <= 128`, all four tensors L1 height-sharded on x's core set (exactly what validate() admits) | per core: `head_row_tiles(1) × tensor_dim_tiles(Wt)`, the whole shard. Per NEO: `1 × neo_dim_tiles`. | **Boundary DRAM: 0 crossings for every tensor; NoC: 0.** The minimum against DRAM ("each input crosses once") is beaten because nothing crosses at all. Against core-local L1 (minimum = read each input once, write the output once): x is unpacked twice (matmul + x·cos), T once per x tile, and `rotated` makes one pack + one unpack round trip. All are local. | Each NEO pays, once per block: the matmul init + `mm_uninit`, the chain's per-element inits (2 bcast-mul inits, SFPU add init), the srcA/srcB/pack reconfigs, one `pack_init` retarget, and one wait/pop/reserve/push per ring. The block is the whole NEO share, so these are paid **once per NEO per call**, the least possible. |
| `multi_user_shards` | deferred: TARGET has `users = [1]` (spec DECISIONS: "One user only"), so no cell can select it. | `users > 1`, one user shard per core | same per-core block; core-assignment = the shard grid | T per core: caller replicates it (tt-metal: `[1,1,32·users,32]`; stepping stone, 0 extra NoC), then mcast from one core (successor: −(users−1) T copies of L1, +1 mcast of 2 KiB) | same as built |
| | Reachable: the program already takes `target_nodes` = the shard grid and every core's kernels are self-contained. Adding users changes run-time args per core only. | | | | |
| `multi_tile_row_heads` | deferred: X6 refuses > 32 heads and no TARGET cell has them | `heads > 32` | `head_row_tiles × Wt` per core | same as built (resident) | same. The unit dealt to NEOs stays the tile; cos/sin are re-read per tile row (row broadcast), a **reuse** the `head_row_tiles` knob covers. |
| `split_dim_tiles_across_cores` | rejected: superseded by dealing j over the core's NEOs (built) | — | Wt split over cores | +2·Wt·(3 inputs + output) tiles over the NoC (copy out, copy back to the pinned output shard, R9), plus T replicated: strictly more than 0 | — |
| `fold_cos_sin_into_matrix` (`out_j = x_j @ (T·diag(sin_j) + diag(cos_j))`) | rejected: each `M_j` is consumed by exactly one x tile (one tile row of heads), so there is no reuse to amortize. It adds a diagonal-mask build and changes the floating-point order. | — | — | same bytes; more compute | — |

No other regime applies in the domain validate() admits. The data is placed by the caller on one core (R8, R9), it is at most 4 tiles, and a split smaller than a tile does not exist. Single predicate; regime-pinned tests are not needed for regime selection. Tests pinned to the **thread-count clamp** are required (Work Distribution).

### Traffic ranking

| Rank | Candidate split | DRAM crossings | Cross-core NoC | Core-local L1 | Verdict |
|------|-----------------|----------------|----------------|---------------|---------|
| 1 | j over the shard core's NEOs (in place, zero-copy rings) | 0 | 0 | x ×2, cos ×1, sin ×1, T ×Wt (ALL), rotated pack+unpack ×1, out pack ×1 | **chosen**: cheapest on every tier |
| 2 | j over cores | 0 | 4·Wt tiles + T per core | same | rejected (row above) |
| 3 | dependent `channel_k` split + combine | — | — | — | not a candidate: the contraction is inside one 32×32 tile and one FPU matmul |
| 4 | `head_row` split | — | — | — | not a candidate: heads share one tile row, and a sub-tile split does not exist |

The extent ranking is separate. Bytes are constant in the block extent here (no overlap, and T is resident and read through an ALL ring with no fetch). So the extent is set by fixed cost alone: take the whole NEO share as one block (catalog `compute_block_size`: one pass over the whole block, 1.27–1.65× over row-at-a-time on WH).

### Block schedule

```cpp
// per core (num_blocks_this_core = users_per_core = 1); NEO t of C works its share j = t, t+C, ...
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    publish_shards(block_idx);        // reader:  credit the resident x / cos / sin tiles (dealt) and T (ALL)
    rotate_block(block_idx);          // compute: rotated[neo share] = x[neo share] @ T
    rotate_combine_block(block_idx);  // compute: out[neo share] = rotated * sin_row + x * cos_row
    release_output(block_idx);        // writer:  drain the output credits (the tiles are already in the output shard)
}
```

| Operation | Block shape | Resident across it | Intended fixed-cost frequency |
|-----------|-------------|--------------------|-------------------------------|
| `publish_shards` | Wt entries each on x, cos, sin; 1 on T | everything (the shards) | no NoC transfer, no barrier. One `reserve_back(1)`+`push_back(1)` per entry (DM share = 1), once per call. |
| `rotate_block` | `1 × neo_dim_tiles` | x share, T (ALL), cos/sin share | `compute_kernel_hw_startup<SrcOrder::Reverse>` once per kernel; `matmul_init` once per block; one DEST section per `section_tiles` (one, given X7); `mm_uninit` once; one reserve/push of `cb_rotated` per block |
| `rotate_combine_block` | `1 × neo_dim_tiles` | x share (not re-fetched), cos/sin share, rotated share | `pack_init(cb_output_tiles)` once; one chain call per block. The chain does each element's init once per block (uniform chain), one wait (Upfront) / pop (AtEnd) per input ring and one reserve/push per block on the output. |
| `release_output` | Wt entries | output shard | one `wait_front(1)`+`pop_front(1)` per entry, once per call |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Thread plan on a ≤ 4-tile op | (1,4,1) is chosen from the bound (compute). But the whole op is ≤ 4 tiles, so engaging 4 NEOs may cost more fixed setup than the ~3 tiles of work it takes off NEO 0. | (1,1,1) against (1,C,1), counted in model cycles on `ttsim-perf`. Turning it is a one-constant edit (`THREAD_PLAN`). |
| Wt ∈ {3} (and 1) runs at C = 1 | The borrowed STRIDED ring forces C to divide Wt, so a 3-tile head runs on one NEO. | non-borrowed STRIDED rings with C = 4 filled by a local L1→L1 copy (implicit sync, 3 tiles × 3 inputs) plus a padded round, against borrowed C = 1 |
| `cb_rotated` L1 round trip | The intermediate is packed and unpacked once per tile. Catalog `compute_fusion`: when the consumer is an **FPU** op, the L1 round trip beats DEST reuse (0.82× for dest-reuse), so the default is the pack. | a single DEST section per tile: `matmul_tiles` → D0; `copy_tile(sin)` → D1; SFPU row-broadcast multiply (`sfpu_binary_bcast.h`) D0·=D1; FPU `mul_tiles_bcast_rows(x, cos)` → D1; SFPU add; pack. The catalog predicts a loss (SFPU multiply 0.58×), so measure before adopting. |
| Data-format reconfig at chain entry | Only `cb_rotated` (Float32) differs from the bf16 inputs. The chain's fold may emit reconfigs that change nothing. | chain with `DataFormatReconfig::Disabled` on the bf16 operands (catalog `compute_block_size`, second lever) |
| Overlap / grid-sync lamps | not attached. There is no data movement to overlap with compute (all resident), and only one core participates (no collective). | — |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x, cos, sin, T arrive | TILE, bf16, L1 shards on x's core | caller placement | nothing is fetched |
| Shards → rings | same bytes | `ttnn.experimental.dfb_spec_from_sharded_tensor(name, tensor, borrowed_from=<TensorParameter name>)` (`ttnn/cpp/ttnn-nanobind/program_specs.cpp:29-82`): one entry per tile of the shard, zero-copy | the reader posts credits only (pattern: quasar untilize `reader_unary_sharded_metal2.cpp`) |
| Rings → NEOs | dealt | x / cos / sin: **STRIDED** consumer (tile j → NEO j mod C). T: **ALL** consumer (`ttnn.experimental.all_consumer_of`, every NEO sees the one entry). | ring geometry is identical for x, cos, sin, so NEO t's k-th tile of each is the same j |
| Compute | DEST fp32 (Phase 0) | `rotate_block` (raw LLK matmul) → `cb_rotated` (Float32) → `rotate_combine_block` (eltwise chain) → `cb_output_tiles` | |
| Output | bf16 tiles written by the packer straight into the output shard | `cb_output_tiles` borrowed from the output TensorParameter | the writer drains credits only; no NoC write |
| Cross-core | none | — | the deferred `multi_user_shards` regime keeps each core self-contained. Its T successor is a 1-sender mcast of one tile (`ttnn.Mcast` + `McastArgs`) into a per-core `cb_trans_mat` that is not borrowed. |

TARGET axes against this scheme:

| TARGET value not in PHASE0 | Class | What changes |
|----------------------------|-------|--------------|
| `dtype = float32` | knob-turn | Ring formats follow the tensor (`dfb_spec_from_sharded_tensor` derives them). `cb_rotated` is already Float32. FPU src registers take fp32 as tf32, which meets the shared (0.999, 0.01). `fp32_dest_acc_en=False` with float32 is refused (R1). |
| `fp32_dest_acc_en = False` | knob-turn | `enable_32_bit_dest=False`. The `cb_rotated` page becomes the input dtype (ledger audit 2). `DEST_AUTO_LIMIT` doubles; the clamps already read it. Graded (0.999, 0.05). |
| `memory_config`, `sharding` | single value each (L1, height): this is the built scheme's physical shard | — |
| `users`, `is_decode_mode`, `layout` | single value each | — |

Structural impossibilities: none to add (INVALID = [] is consistent with this design).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block = one user's shard: `head_row_tiles (1) × tensor_dim_tiles (Wt)` tiles |
| Grid | the input shard's core set (`shard_spec.grid`), one core with `users = 1`. `WorkUnitSpec(target_nodes=shard_spec.grid)`. `split_work_to_cores` is not used: the shard is the split. |
| Per-core work | Wt tiles; per NEO `neo_dim_tiles = Wt / C` (exact by the C clamp) |
| Remainder | none across NEOs (C divides Wt by construction). Tile geometry: `tensor_dim_tiles = ceil(head_dim / 32)` (= head_dim/32 by X4); `head_row_tiles = ceil(heads / 32)` per user image (= 1 by X6). h_non_aligned (heads < 32): the tile's padding rows are computed and never read back (spec GUIDANCE). |
| Bound | **compute.** Per tile the core does a 32×32×32 matmul at HiFi4, two FPU row-broadcast multiplies, an SFPU add and two packs, and it moves **zero** bytes (all operands resident, zero-copy rings). |
| Thread plan | Host constant `THREAD_PLAN = (1, 4, 1)` (R, C, W); effective `compute_threads` from the C clamp (Mechanism caps). C = 4 because the op is compute-bound. R = W = 1 because the DM threads move no bytes (they only post / drain credits), and more harts would only split a loop of ≤ 4 credit calls. R and W divide every C. The counts are compile-time args (`reader_threads`, `compute_threads`, `writer_threads`) and `KernelSpec(num_threads=…)`. |
| Unit dealt to compute threads | the ring entry itself (one tile j), so no round padding is needed. The host deals whole rounds trivially: C divides Wt on the single core. |
| Tests the plan needs | **(a)** `test_each_neo_computes_its_own_tiles`: at Wt = 4 (C = 4), a compile-time probe (e.g. `tag_threads`: NEO t adds t to its result, as in `toy_thread_plan/compute_llk.cpp`) shows tile j was produced by NEO j mod 4. **(b)** `test_thread_clamp`: one case per effective C (head_dim 32 → 1, 64 → 2, 96 → 1, 128 → 4), asserting the host's `compute_threads` and the result. Add head_dim 128 under plan (1, 2, 1), so a NEO holds two tiles and the matmul stride index is exercised. The plan is a host argument of the artifacts builder, defaulting to `THREAD_PLAN`. **(c)** `test_session_order`: in one device session, the op after another program (another call of the op, then `ttnn.add`) and before `ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)`. |

## Circular Buffers

Under the plan, every page count below is the ring's **total**. STRIDED capacity is that total / C per NEO.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `cb_input_tiles` | 0 | input tile (2048 B bf16) | `tensor_dim_tiles` (borrowed: whole x shard) | spans head_dim_tile (whole core extent); head_row = 1 tile row | input dtype | reader | compute (STRIDED, C) | explicit, credits only (`disable_dfb_implicit_sync_for`), no NoC transfer | whole program |
| `cb_cos_tiles` | 1 | input tile | `tensor_dim_tiles` (borrowed: cos shard) | spans head_dim_tile | input dtype | reader | compute (STRIDED, C) | explicit, credits only | whole program |
| `cb_sin_tiles` | 2 | input tile | `tensor_dim_tiles` (borrowed: sin shard) | spans head_dim_tile | input dtype | reader | compute (STRIDED, C) | explicit, credits only | whole program |
| `cb_trans_mat` | 3 | input tile | 1 (borrowed: T shard) | constant: one tile | input dtype | reader | compute (**ALL**, C) | explicit, credits only | whole program |
| `cb_rotated` | 4 | intermediate tile (4096 B Float32) | `neo_dim_tiles × compute_threads` (= Wt) | spans head_dim_tile (NEO share), one block | Float32 if `fp32_dest_acc_en` else input dtype | compute | compute (same NEO; intra-tensix packer → unpacker) | compute-internal (always explicit) | `rotate_block` → `rotate_combine_block` |
| `cb_output_tiles` | 5 | output tile | `tensor_dim_tiles` (borrowed: output shard) | spans head_dim_tile | output dtype | compute (STRIDED, C) | writer | explicit, credits only | whole program |

Every ring has exactly one producer kernel and one consumer kernel. `cb_input_tiles` is read by two compute phases of the **same** kernel, which is one consumer. Push = wait for every ring:

- reader pushes Wt on each of the three STRIDED rings; each NEO waits for `neo_dim_tiles`, and C × `neo_dim_tiles` = Wt;
- T: 1 pushed, 1 waited per NEO (ALL);
- `cb_rotated`: each NEO pushes `neo_dim_tiles` and waits for `neo_dim_tiles`;
- output: the NEOs push C × `neo_dim_tiles` = Wt, and the writer waits for Wt.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `publish_shards` | Wt / Wt / Wt / 1 entries | no (credit calls only) | — (data already resident) | `cb_input_tiles`, `cb_cos_tiles`, `cb_sin_tiles` (Wt each, dealt RR over C), `cb_trans_mat` (1) | all entries visible to the NEOs |
| 2 | `rotate_block` | `1 × neo_dim_tiles` | raw LLK (matmul) | `cb_input_tiles` (wait `neo_dim_tiles`, **not popped**), `cb_trans_mat` (wait 1, popped at the end of the block) | `cb_rotated` (reserve/push `neo_dim_tiles`) | x share still held for #3 |
| 3 | `rotate_combine_block` | `1 × neo_dim_tiles` | helper: `eltwise_chain` | `cb_rotated`, `cb_sin_tiles`, `cb_input_tiles`, `cb_cos_tiles`: each `WaitPolicy::Upfront` / `PopPolicy::AtEnd` / `InputTileMapping::Block` (`neo_dim_tiles`) | `cb_output_tiles` (`ReservePolicy::Upfront` / `PushPolicy::AtEnd`, `neo_dim_tiles`) | all inputs popped; output share published |
| 4 | `release_output` | Wt entries | no (credit calls only) | `cb_output_tiles` (wait/pop 1 × Wt) | — | output shard holds the result |

Chain for #3 (one DEST section per chain block; lane width 2):

```
eltwise_chain(IterationShape::tiles(neo_dim_tiles).block_size(neo_dim_tiles),
    BinaryFpu<Mul, input(cb_rotated, Upfront, AtEnd, Block),
                   input(cb_sin_tiles, BroadcastDim::Row, Upfront, AtEnd, Block), Dst::D0>{rot, sin},
    BinaryFpu<Mul, input(cb_input_tiles, Upfront, AtEnd, Block),
                   input(cb_cos_tiles, BroadcastDim::Row, Upfront, AtEnd, Block), Dst::D1>{x, cos},
    AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
    PackTile<output(cb_output_tiles, ReservePolicy::Upfront, PushPolicy::AtEnd), Dst::D0>{out});
```

Before the chain:

- `mm_uninit(cb_input_tiles, cb_trans_mat)`;
- `pack_init(cb_output_tiles)`.

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| host: shard → ring | host API | `ttnn.experimental.dfb_spec_from_sharded_tensor` | `ttnn/cpp/ttnn-nanobind/program_specs.cpp:29-82`; impl `ttnn/core/tensor/tensor_utils.cpp:86-136` | `(name, tensor, borrowed_from=<param>)`; `num_entries` left at 0 (whole shard) | — | the 5 borrowed rings | none (the shard fixes depth) |
| host: T fan-out | host API | `ttnn.experimental.all_consumer_of` | `ttnn/cpp/ttnn-nanobind/program_spec.cpp:823` | compute binding of `cb_trans_mat` | — | — | — |
| host: program | host API | `ttnn.experimental.ProgramSpec`, `KernelSpec(num_threads=…)`, `WorkUnitSpec`, `ttnn.generic_op` | `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py:95-170` | `ComputeHardwareConfig(fpu_math_fidelity, enable_32_bit_dest, sfpu_precision_mode)` (`compute_hardware_config.hpp:43-70`); DM `config_2xx.disable_dfb_implicit_sync_for` = all of that kernel's rings | — | — | `THREAD_PLAN` |
| `publish_shards` | raw_api (dataflow) | `DataflowBuffer::reserve_back` / `push_back` | `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:243-244`; share rule `internal/tt-2xx/dataflow_buffer.inl:390-396` | 1 entry per call | — | x, cos, sin, T rings | `tensor_dim_tiles` (RT/CT arg) |
| `rotate_block` | raw_api | `compute_kernel_hw_startup<SrcOrder::Reverse>`, `matmul_init`, `matmul_tiles`, `pack_tile`, `mm_uninit` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:59-60`; `api/compute/matmul.h:113`, `:172`, `:147`; `api/compute/pack.h:87` | `matmul_tiles(cb_input_tiles, cb_trans_mat, k * x.get_consumer_stride_tiles(), 0, k)` for k < `section_tiles`; `pack_tile(k, cb_rotated, k)` (the pack side steps by stride itself) | `cb_input_tiles`, `cb_trans_mat` | `cb_rotated` | `neo_dim_tiles`, `section_tiles = min(neo_dim_tiles, DEST_AUTO_LIMIT)` |
| `rotate_combine_block` | helper | `eltwise_chain` + `BinaryFpu` (Row broadcast on srcB) + `AddBinary` + `PackTile` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:619-620`, `:568`, `:588`, `:330` (`BroadcastDim::Row`); `eltwise/binary/sfpu/basic.hpp:20`; Quasar row-bcast mul path `api/compute/bcast.h:192-205`, `:452`; SFPU add `api/compute/eltwise_binary_sfpu.h:103` | shown under Block Operation Realization | `cb_rotated`, `cb_sin_tiles`, `cb_input_tiles`, `cb_cos_tiles` | `cb_output_tiles` | `IterationShape::tiles(neo_dim_tiles).block_size(neo_dim_tiles)` |
| `rotate_combine_block` (prefix) | raw_api | `pack_init(cb_output_tiles)` | `api/compute/pack.h:32` | — | — | — | — |
| `release_output` | raw_api (dataflow) | `wait_front` / `pop_front` | `dataflow_buffer.h:245-246` | 1 entry per call | `cb_output_tiles` | — | `tensor_dim_tiles` |

Helpers considered and rejected:

| Raw entry | Candidate helper | Mismatch (file:line) | Reason |
|-----------|------------------|----------------------|--------|
| `rotate_block` | `matmul_block` (`ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp`) | — | The block is 1×K(=1)×`neo_dim_tiles`, so per-tile `matmul_tiles` *is* the block's matmul. |
| `rotate_block` | eltwise chain (fuse matmul into the chain) | `chain.hpp:9-11`: the chain's element set is FPU binary / SFPU / copy / pack / dest-reuse; there is no matmul element, and the chain owns the DEST window (`chain.hpp:16`), so a matmul cannot sit inside it | no covering element |
| `publish_shards` / `release_output` | `dfb_helpers_dataflow.hpp`, `local_copy_helpers_dataflow.hpp` | those move bytes; here no byte moves, only credits | nothing to cover |
| `pack_init` | — | no helper | — |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| `rotate_combine_block` D0 | Mul | `cb_rotated`: All (rows 0..heads-1 real, the rest padding-derived) | `cb_sin_tiles`: Row0 (`[1,…,head_dim]` padded to a tile) | Row (srcB row 0 to all 32 rows) |
| `rotate_combine_block` D1 | Mul | `cb_input_tiles`: All | `cb_cos_tiles`: Row0 | Row |
| `rotate_combine_block` | SFPU Add | D0: All | D1: All | none (DEST–DEST) |
| `rotate_block` | Matmul | `cb_input_tiles`: All (in0, rows) | `cb_trans_mat`: All (in1) | none |

The padding rows of cos/sin never reach the output: a Row broadcast reads only row 0 (R3, spec GUIDANCE).

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Matmul tile index | `llk_unpack_AB_matmul` adds the tile index to the ring cursor (`llk_unpack_AB_matmul_api.h:191-194`) | index `k * get_consumer_stride_tiles()`. The default plan never gives a NEO more than one tile (Wt=4 → C=4, Wt=2 → C=2). Test (b) also runs head_dim 128 with the plan overridden to (1, 2, 1): share 2, so k = 1 is exercised. |
| C does not divide Wt on a borrowed STRIDED ring | `num_entries` is pinned to the shard | the C clamp; test (b) pins each of C ∈ {1, 2, 4} |
| Pack target | Quasar programs the pack target at `pack_init` (`reconfig_data_format.h:454-456`) | `pack_init(cb_output_tiles)` between `rotate_block` and the chain. |
| srcA/srcB formats after the matmul | `hw_startup<SrcOrder::Reverse>` maps in0→SrcB. `matmul_init` latches an unpacker format override that only `mm_uninit` undoes (`matmul.h:131-153`, a no-op for bf16 but the documented exit), and `cb_rotated` is Float32 while the rest are bf16. | `mm_uninit` before the chain; the chain's reconfig is left **Enabled** for every operand. If the first combine is wrong, add an explicit `reconfig_data_format(cb_rotated, cb_sin_tiles)` before the chain. |
| A ring with no NoC traffic under implicit sync | an implicit-sync ring gets its credits from the DM0 ISR on tagged transfers, and none happen here | every DM ring is listed in `disable_dfb_implicit_sync_for`, and the reader/writer use explicit credit calls. No NoC transfer means no transaction tag is left behind, so no trid reset is needed. Test (c) proves the session ordering anyway. |
| Output memory config ≠ input's | R9 is checked with `==` by HOOKS["check"] | allocate from `input_tensor.spec` (`ttnn.allocate_tensor_on_device`). Never rebuild the memory config. |
| T not on x's core | R8: the four tensors must be placed alike | validate() compares the shard grids of all four against x's and refuses with `NotImplementedError` |
| DM share assert | a DM STRIDED side's share is 1 entry (`dataflow_buffer.inl:390-396`) | one credit call per entry, never `push_back(Wt)` |
| `is_decode_mode` default | the signature's default `False` must be refused (R4, X8), and the axis read from the argument | validate() derives the `is_decode_mode` axis from the argument and refuses `False` by name |
