# L1 Ledger: genify_matmul

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes, from the Blocking Model in `op_design.md`:
- **b**: `block_batches`
- **m**: `block_m_tiles`
- **n**: `block_n_tiles`; the output width is `out_block_n_tiles = block_n_tiles / 2` for swiglu, else `block_n_tiles`
- **k**: `block_k_tiles`, with `num_k_blocks` blocks

Phase 0 (regime R1, epilogue none) allocates the first four rows. The later rows are listed so that the extent solve and the sharing decisions are already priced.

## CB table

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_act_tiles` | `depth_act · block_m_tiles · block_k_tiles` | `block_m_tiles · block_k_tiles` (one K-block in compute) + one in flight from the reader | `{b: streams, m: spans → block_m_tiles, n: streams (re-read per N-block), k: spans → block_k_tiles (window of one K-block)}` | A dtype (bf16 / bf8b / fp32 since Refinement 1) | reader | compute | whole kernel | **None.** `depth_act = 2` is double buffering (the read of K-block kb+1 overlaps the multiply of kb). It cannot alias `cb_weight_tiles`: both are live concurrently, as in0 and in1 of the same matmul (the helper requires distinct CBs, `matmul_block_helpers.hpp:242-246`). |
| `cb_weight_tiles` | `depth_weight · block_k_tiles · block_n_tiles` | `block_k_tiles · block_n_tiles` + one in flight | `{b: streams, m: streams (re-read per M-block), n: spans → block_n_tiles, k: spans → block_k_tiles}` | B dtype (bf16 / bf8b / bf4b since Refinement 1) | reader | compute | whole kernel | **None.** `depth_weight = 2` is double buffering. Same concurrency as `cb_act_tiles`. |
| `cb_partials` | `block_m_tiles · block_n_tiles` (exact, a mechanism cap: in-place packer L1 accumulation needs the ring to return to the block base). Allocated **only if** `num_k_blocks > 1 or epilogue != none`. | the whole block: every output tile's running sum is live across all K-blocks | `{b: streams (per batch slice), m: spans → block_m_tiles, n: spans → block_n_tiles, k: streams (accumulated in place)}` | `Float32` when `fp32_dest_acc_en` (the K-accumulator must keep fp32 DEST precision across K-blocks), else `Float16_b` | compute | compute | one block | **None.** It cannot share with `cb_output_tiles`: the page formats differ (Float32 vs the output dtype), and on the last K-block both are live (the partials are reloaded while output row groups are packed). It cannot fold into DEST: the block (up to ~200 tiles) exceeds DEST (4 tiles), which is the reason the accumulator exists. When `num_k_blocks == 1` and epilogue none, the CB is omitted entirely and `cb_output_tiles` is the helper's placeholder. |
| `cb_output_tiles` | `depth_out · sb_h · out_block_n_tiles` | one row group `sb_h · out_block_n_tiles` being packed + one draining | `{b: streams, m: streams (row groups of sb_h), n: spans → out_block_n_tiles, k: streams (written once, after the last K-block)}` | output dtype (the `dtype` override, else A dtype) | compute | writer | whole kernel | **None.** `depth_out = 2` is double buffering against the writer. Packing straight into this destination is already the reuse (pattern 1). It cannot alias `cb_partials` (different format and concurrent use, see above). |
| `cb_bias_row` (bias, bias_relu; built in Refinement 2) | `block_n_tiles` | `block_n_tiles` (the whole N-block row, read for every row group) | `{b: streams, m: streams (broadcast over m), n: spans → block_n_tiles, k: streams (unused)}` | bias dtype | reader | compute | one block | **None.** It is live across the whole epilogue while `cb_partials` and `cb_output_tiles` are both in use. Single-buffered: one fetch per block, so a second slot buys almost nothing (a lamp if the per-block bias read ever shows up in a profile). |
| `cb_gate_row` (addcmul; built in Refinement 2) | `block_n_tiles` | `block_n_tiles` | `{b: streams, m: streams (broadcast), n: spans → block_n_tiles, k: streams}` | gate dtype (= output dtype) | reader | compute | one block | **None.** Concurrent with `cb_partials` and `cb_residual_tiles` in the same chain step. Only one of `cb_bias_row` / `cb_gate_row` exists per program (the epilogues are exclusive), so they **alias by construction**: the same index range is never allocated twice. |
| `cb_residual_tiles` (addcmul; built in Refinement 2) | `depth_residual · sb_h · block_n_tiles` | one row group + one in flight | `{b: streams, m: streams (row groups), n: spans → block_n_tiles, k: streams}` | output dtype | reader | compute | whole kernel | **None.** It streams concurrently with `cb_output_tiles`, so the reader and the writer would both touch one CB, which breaks the ownership rule. `depth_residual = 2` is double buffering. |
| `cb_act_shard` (index 5; R3 built in Refinement 5; later R2) | the A shard's tiles (`cb_descriptor_from_sharded_tensor`) | the whole shard (resident) | `{b: streams, m: spans → shard_m_tiles (= block_m_tiles), n: streams, k: spans → shard_k_tiles}` | A dtype | reader (one `cb_push_back`, no NoC) | compute | kernel | **Backed on the A tensor's own L1**, so it adds 0 bytes to the CB budget. It replaces `cb_act_tiles` in R2 (that CB is then not allocated). R3 keeps `cb_act_tiles` for the peer K-slices; same format, so `In0SourceFn` can select either. |
| `cb_output_shard` (R4, R3 when `block_n_tiles == out shard width`; later R2) | the output shard's tiles | the whole shard | `{b: streams, m: spans → block_m_tiles, n: spans → block_n_tiles, k: streams}` | output dtype | compute | writer (wait/pop only) | kernel | **Backed on the output tensor's L1**, 0 extra bytes. It replaces `cb_output_tiles`. |

**Refinement 2 (epilogues) as built.**
- `cb_partials` is allocated whenever epilogue ≠ none, still exactly one block. The matmul packs its last K-block there (`LastBlockTarget::Interm`), and the epilogue drains it into `cb_output_tiles`.
- `_footprint()` now carries every epilogue term below (`E_bias` / `E_addcmul`, and `out_block_n = block_n/2` for swiglu), so the S1–S3 solve prices them. `_block_k_max()` derives its fixed part from `_footprint(…, bk=0)`, which keeps the formula in one place.
- `cb_bias_row` / `cb_gate_row` stay single-buffered. The reader pushes the row **after the first batch's K-walk of the block**, not before it. With one slot, a push ahead of the K-blocks would park the reader on `cb_reserve_back` until the previous block's epilogue popped the row, which would stall the next block's K-block prefetch for a whole epilogue.
- `cb_residual_tiles` row groups are pushed after each batch's K-walk, in epilogue order.
- Ablation, PERF FOCUS 1: the Interm/partials plumbing adds no measurable time. With the chain compute stubbed the run takes 1564 µs, against 1572 µs for the epilogue-none twin. The fused swiglu costs only its SFPU compute.

**Refinement 3 (R1a `interleaved_2d_mcast`, single-walk K split) as built.**
- **No new CB.** Under R1a the line injector reads its K-block into its own `cb_act_tiles` / `cb_weight_tiles` slot and multicasts it to the same slot address on the line's receivers (the grid-uniform walk keeps every core's write pointer in lockstep). The receivers' slot is the landing zone; the handshake is "slot reserved". Producer and consumer per CB are unchanged (reader → compute). Two `ttnn.Mcast` channels add 2 + 2 L1 semaphores per core, no CB bytes.
- **Folded finding resolved (dead second slot / no read-compute overlap).** When a core's whole walk would be ONE K-block (S1, `core_batches == 1`), the solve now splits K into `SINGLE_WALK_K_BLOCKS = 2` K-blocks (when `tensor_k_tiles ≥ 2·block_k_floor` and the split fits with `cb_partials`). The two halves occupy exactly the old single-slot bytes, so the depth-2 operand CBs now hold two live K-blocks and compute starts on the first while the second lands. Measured: 384×1536×1024 43.4 → 34.2 µs, 32×1024×2048 26.6 → 24.5 µs, neutral elsewhere. Shapes with `tensor_k_tiles < 16` keep the unsplit S1 block.

**Refinement 4 (R4 `dram_sharded`) as built.**
- **Per bank worker** (one per DRAM bank holding a valid B column; `get_optimal_dram_bank_to_logical_worker_assignment(NOC_0)`): `cb_act_tiles`, `cb_weight_tiles`, `cb_partials`, and the epilogue rows as in R1, with `block_n_tiles = sw` (the bank shard width), `block_k_tiles = kw / s` (`kw` = A shard width; `s` = the smallest divisor of `kw` that fits L1 and the 16-bit-DEST chain cap; `s > 1` only with `block_m_tiles == 1`), `block_m_tiles` = the largest divisor of the padded rows that fits.
- **`cb_output_shard` (index 17) replaces `cb_output_tiles`**: backed on the worker's own L1 output shard (`cb_descriptor_from_sharded_tensor`), so it adds 0 CB bytes (`_footprint(..., out_in_place=True)` drops the `depth_out` term). The op builds that shard spec (shard w on bank w's worker, `[padded rows, out_sw·32]`), so `out_block_n == out shard width` always holds and the compute packs in place; the writer only retires row groups.
- **`cb_act_tiles` is also allocated on every non-worker core the A broadcast writes** (A's shard cores and the rest of the workers' bounding rectangle), at the same address: it is the multicast landing zone there, never pushed or popped. A sender-only core addresses the workers' slot as `base + (K-block % depth_act) · slot bytes` (CT `act_cb_blocks = DEPTH_ACT`).
- One `ttnn.Mcast` channel (`act`) adds 2 L1 semaphores per participating core; the B line channel is absent.

**Refinement 6 (R4 `cores_per_bank` split, writer-side A rounds, trid B ring) as built.**
- **Workers = C per bank** (`DS_CORES_PER_BANK`, auto: the largest C dividing the bank width `sw` into slices of ≥ `DS_CORE_N_TILES_MIN = 4` columns (whole swiglu pairs) with `C · banks ≤ grid`): bank w's optimal worker plus its C − 1 nearest free cores. Every per-worker CB above is the same with `block_n_tiles = sw / C` (PERF FOCUS 2: 20 → 4 columns, 40 workers), so `cb_weight_tiles` / `cb_partials` / epilogue rows shrink C×.
- **Refinement 17:** `block_n_tiles = cw = ceil(sw / C)` (a bank's last slice may be narrower; its reads fetch only the valid columns, the CB slot keeps the `cw` row stride). C comes from the compute-hiding rule (`DS_COMPUTE_SHARE`), not a divisor of `sw`; no CB added, removed or resized beyond the `cw` term (qkvg 14 → 4 columns, 32 workers; LM head 64 → 22).
- **C > 1: `cb_output_tiles` (index 16, `depth_out · sb_h · out_block_n`) replaces `cb_output_shard`** on every worker: a slice is not the whole output shard, so it cannot be packed in place. The writer lands it in the bank's output shard (on the bank's primary worker) through the output's sharded accessor. C = 1 keeps `cb_output_shard` (0 bytes). Shares with / why not: the output shard itself only exists on the primary worker and holds all C slices' columns interleaved per row, so a slice core has no in-place target.
- **`cb_weight_tiles` depth** = `DS_DEPTH_WEIGHT` (default 2 = `DEPTH_WEIGHT`) when it fits (`ds_weight_depth`), else `DEPTH_WEIGHT`; the B ring keeps `min(DS_B_AHEAD, depth)` K-blocks in flight on transaction ids (default 1: depth 3–4 measured no gain once 5 slice readers share each bank — the bank is already saturated — so the knob is parked at its trivial value and costs no L1).
- **`cb_act_tiles`** unchanged in size; under `DS_ACT_ON_WRITER` (default) its producer on the workers is the **writer** (BRISC runs the A rounds), not the reader — still a single producer (the reader no longer touches it) and a single consumer (compute). The non-worker cores of the broadcast run only the writer kernel (sender-only / landing zone).

**Refinement 5 (R3 `block_sharded`) as built.** Per core of A's block-shard grid (column c = K shard, row r = M shard):
- **`cb_act_shard` (index 5)** is backed on the core's own A shard (`cb_descriptor_from_sharded_tensor`, 0 CB bytes). The reader pushes `block_m · shard_k` tiles of it on the core's own K turn (no NoC); compute selects it for K-block `own_kb = c` through `In0SourceFn`. With `nb_m > 1` the pushes walk the shard row-block by row-block; with `nb_n > 1` (only when `nb_m == 1`) each N-block pass wraps the whole shard, so the ring always lines up with the shard rows.
- **`cb_act_tiles`** (`depth_act · block_m · shard_k`) is the landing zone of the row peers' K-slices. Under the rotating row multicast (`BS_ROW_MCAST`, default) every core of a grid row reserves / pushes round kb's slot (the sender too, receiving the loopback copy), so the row's write pointers stay in lockstep and each multicast lands at the sender's own slot address; compute retires the own round's slot unread (`PostKBlockFn`). Under the unicast stepping stone the own round pushes nothing there.
- **`cb_weight_tiles`, `cb_partials`, epilogue rows**: as R1, with `block_k = shard_k` (one peer shard per K-block, `num_k_blocks` = grid columns), `block_n` = the output shard's B columns (×2 for swiglu).
- **Output**: `cb_output_shard` (in place) when `block_n` is the whole output shard width, with `block_m` the largest divisor of the shard rows that fits. Fallback when no such block fits: `block_m = shard_m`, the N range split, `cb_output_tiles` streamed and written by the writer through the output's sharded accessor (every page is local).
- `_footprint(..., out_in_place)` prices it; `l1_cb_budget` already deducts A's and the output's resident shards. The 16-bit-DEST chain cap is NOT applied to R3 (`block_k` is the shard width; every golden cell has `shard_k ≤ 8`).

**Refinement 7 (R2 `height_sharded`, batched weights) as built.** R2 is R3's single-K-shard case (`cols = 1`, `block_k = tensor_k_tiles`) on A's shard cores in shard order (`ttnn.corerange_to_cores` with the shard orientation; any core set, not only a rectangle):
- **`cb_act_shard` (index 5)** is the compute kernel's A operand outright (compute CT arg 0 = `CB_ACT_SHARD`), backed on the core's resident A shard (0 CB bytes). The reader pushes `block_m · K` tiles per block: with `nb_m > 1` the pushes walk the shard row-block by row-block; with `nb_n > 1` (only when `nb_m == 1`) each N-block pass wraps the whole shard.
- **`cb_act_tiles` is not allocated** whenever the walk has a single K shard (R2, and R3 with a one-column grid): there are no peer K-slices to land. `_footprint(..., act_cb=False)` drops its `depth_act · block_m · block_k` term. Shares with / why not: nothing left to share — it no longer exists.
- **`cb_weight_tiles` depth** = `DEPTH_WEIGHT` when the core's walk has more than one K-block (`nb_m · nb_n · num_k_blocks · block_batches > 1`: the second slot prefetches the next block's B), else **1** (`_weight_depth`): a whole-shard, whole-K walk is ONE K-block, whose second slot could never fill (same finding as Refinement 3's folded one). Applies to R3 too; R3's multi-shard walks keep depth 2.
- **Output**: `cb_output_shard` in place when the block spans the whole output shard width (the default; R2's output shard is the whole output width), with `block_m` the largest divisor of the shard's batch-slice rows that fits; fallback `block_m` = those rows, N split, `cb_output_tiles` streamed and written through the output's sharded accessor (local pages).
- **Batched B on R2 / R3** (`ShardRows`): a shard either lies inside one image (`img_m % shard_m == 0`: `block_batches = 1`, b0 / row0 per shard) or holds whole images (`shard_m % img_m == 0`: `block_batches = shard_m / img_m`, `block_m = img_m`, `nb_m = 1`, so the pushes walk the shard image by image = sequentially). No CB changes: the batch slices reuse the per-batch CBs (the matmul helper's batch loop; the epilogue path calls it per slice). Any other shard height is refused (no block may straddle two batches).
- **R1d (batched B, interleaved A)**: no CB change; `block_batches = core_batches` as the S1 row already says (`grid_b > 1`, flat placement).

**Refinement 8 (operand residency, L1-discounted split ranking, write batching) as built.** R1 / R1a only; no new CB.
- **Operand residency** (`RESIDENT_SPLIT_BLOCKS = 4`): a one-block, one-K-block S1 walk that K cannot split (`tensor_k_tiles < SINGLE_WALK_K_BLOCKS · block_k_floor`, `core_batches == 1`) is split along ONE axis instead of running as one block. The smaller per-core operand stays fronted in its CB across the blocks (matmul_block `WaitAndRetainOnLastBlock` on it; compute pops it after its last use):
  - `act_resident` (A smaller): N-blocks. `cb_act_tiles` holds the M-block's A (`block_m · K`) from its first N-block to its last; its second slot prefetches the next M-block's A. `cb_weight_tiles` streams `K · block_n` B blocks (double-buffered). Axis accounting: `{n: streams (A NOT re-read per N-block), m: spans}`.
  - `weight_resident` (B smaller): M-blocks (multiples of the whole region's `sb_h`). `cb_weight_tiles` holds the core's only B block (`K · block_n`, `nb_n = 1`) for the whole walk; its second slot is unused (there is no next B block — `core_batches == 1`). Axis accounting: `{m: streams (B NOT re-read per M-block), n: spans}`.
  - Capacities are the R1 formulas at the smaller split extent, so F only shrinks. Shares with / why not: unchanged — the resident operand is still in0 / in1 of the same matmul.
  - Gate: the split is taken only if it keeps the R1a multicast decision (a smaller K-block can fall under `MCAST_MIN_KBLOCK_BYTES`).
- **`cb_output_tiles` capacity** is `depth_out · write_batch · sb_h · out_block_n` (`OUT_WRITE_BATCH_ROW_GROUPS`, parked at 1 = the Refinement 7 size; a larger batch is granted only if it fits the budget). The writer waits / flushes / pops `write_batch` row groups at a time; the ring is a whole number of batches, so a batch never wraps.

**Refinement 9 (R1b `interleaved_stream_mcast_a`) as built.** R1's CB inventory on a 1-D N split (`grid_m' = 1`); no new CB.
- **`cb_act_tiles`: producer = the writer** (BRISC), not the reader. The writer's A stream produces it (the injector reads each A K-block into its own slot and multicasts it to the same slot on every core; the others reserve, ack, wait, push). The reader no longer touches it, so single producer / single consumer holds. Every core walks the same nominal block sequence (grid-uniform `nb_n`), so the write pointers stay in lockstep and the slot is the multicast landing zone.
- **Operand depths**: `cb_act_tiles` / `cb_weight_tiles` hold `ACT_STREAM_DEPTH = 4` K-blocks each on an R1b K-walk (S2 / S3), carried in `Blocking.depth_act` / `depth_weight`, which both the solve (`_footprint`, `_block_k_max`) and the CB sizes read. The solve shrinks `block_k` to pay for it (PERF FOCUS 4: `block_k` 20 → 10, 10 → 20 K-blocks; same bytes per CB). What depth buys: every core waits for each K-block's A round, so the slowest core's B read gates the next broadcast; four slots give the coupling three K-blocks of slack and let B run ahead of compute (683 → 633 µs). The one-K-block S1 walk keeps depth 2. Shares with / why not: unchanged (in0 / in1 of the same matmul, both live).
- **Injector stall-shadow** (`ACT_STREAM_AHEAD = 2`): the injector reserves K-block kb+1's slot (`(j − kb + 1)` blocks ahead, address `ring base + (slot % depth_act) · slot bytes`, the CB is exactly `depth_act` slots) and issues its reads (trid `(j % ahead) + 1`) before multicasting kb. No extra L1: the slot is one of the ring's own.
- One `ttnn.Mcast` channel (`act`, NoC1, a fixed in-group sender, one group = the whole split; a flat split's partial last row is a second rectangle) adds 2 L1 semaphores per core; the R1a line channels are absent.

## Symbol table (non-block parameters and their bounds)

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `depth_act`, `depth_weight`, `depth_out`, `depth_residual` | = 2 (Phase 0 constants, knobs) | host constants |
| `sb_h` | `1 ≤ sb_h ≤ dest_limit`, `sb_h ∣ block_m_tiles` | host subblock derivation (`dest_limit = 4` fp32 DEST / 8) |
| `a_tile_bytes`, `b_tile_bytes`, `o_tile_bytes`, `bias_tile_bytes` | ≤ 4096 (fp32 tile) | `tensor.buffer_page_size()` / `ttnn.tile_size(dtype)` |
| `p_tile_bytes` | 4096 (fp32 DEST) or 2048 | `fp32_dest_acc_en` |
| `block_m_tiles` | ≤ `core_m_tiles` = `ceil(tensor_m_tiles / grid_m)` | the L1 solve (S1–S3) keeps `F ≤ l1_cb_budget` |
| `block_n_tiles` | ≤ `core_n_tiles` (multiple of `q_n`) | the L1 solve |
| `block_k_tiles` | `min(tensor_k_tiles, 8, block_k_cap) ≤ block_k_tiles ≤ block_k_cap` | the L1 solve (S2 floor `block_k_floor`), capped by `block_k_cap` |
| `block_k_cap` | `tensor_k_tiles` with fp32 DEST; `min(tensor_k_tiles, DEST16_BLOCK_K_MAX = 8)` with 16-bit DEST | Refinement 1 precision knob: a bf16 DEST running sum swamps small addends, so the in-DEST K chain is capped and the cross-K-block sum goes to the packer (`cb_partials`, Float16_b). With 16-bit DEST and K > 8 tiles, `cb_partials` therefore always exists (S1 "one resident K-block" applies only when `tensor_k_tiles ≤ block_k_cap`). |
| `tensor_k_tiles`, `tensor_m_tiles`, `tensor_n_tiles` | **not** in any capacity expression | — (the op dimensions enter only through the bounded block extents) |
| `l1_cb_budget` | `min(lowest allocated L1 block offset in ttnn.get_memory_view(device, L1).block_table (bank size if none), total_bytes_per_bank − l1_tensor_bytes_per_core) − l1_reserve_bytes`. Verifier fix: the earlier `total − allocated` ignored fragmentation (a freed tensor above a live one) and let CBs clash with L1 buffers (implementer: `get_max_worker_l1_unreserved_size()` counts from the kernel-config ring buffer, below the CB base `l1_unreserved_base`, and over-stated the budget by ~70 KB on BH → CB overflow on the large loose cases; 1460288 B on p300c) | host `l1_cb_budget()`; `l1_reserve_bytes = 0` in Phase 0 |

## Total per-core footprint (closed form)

```
F(block_m, block_n, block_k) =
      depth_act    · block_m · block_k · a_tile_bytes                       # scales with m·k
    + depth_weight · block_k · block_n · b_tile_bytes                       # scales with k·n
    + P            · block_m · block_n · p_tile_bytes                       # scales with m·n;  P = [num_k_blocks>1 or epilogue!=none]
    + depth_out    · sb_h · out_block_n · o_tile_bytes                      # scales with n
    + E_bias       · block_n · bias_tile_bytes                              # scales with n  (bias epilogues)
    + E_addcmul    · (block_n · o_tile_bytes + depth_residual · sb_h · block_n · o_tile_bytes)   # scales with n
  ≤ l1_cb_budget
```

`F` is affine in `block_k` for fixed (m, n). So S2's `block_k_tiles_max` is a closed-form division:
`(budget − terms without k) / (depth_act·block_m·a + depth_weight·block_n·b)`.

Worked values, 11×10 grid, budget ≈ 1.4 MB (illustrative; the host computes the real number):

| Shape (A × B, dtypes) | Region / block | Terms | `block_k` |
|---|---|---|---|
| prefill gate_up 2048×5120 @ 5120×8704 (bf16 × bf4b, swiglu, regime R1 → R1a later) | grid 11×10, block 6×28 | partials 168 × 4 KB = 672 KB; out 2·1·14·2 KB = 56 KB; per `bk` 2·(6·2 KB + 28·576 B) = 56.8 KB | `num_k_blocks = 15`, `block_k = 11`, `last = 6` |
| 1×6144 @ 6144×6144 (bf16, 1-D grid) | `grid_n' = 96`, block 1×2 | partials 8 KB; out 8 KB; per `bk` 2·(2 KB + 4 KB) | `num_k_blocks = 2`, `block_k = 96` |
| (1,1,2048,256) @ 256×512 (bf16) | grid 11×8, block 6×2 | S1 fits: A 2·6·8·2 KB + B 2·8·2·2 KB + out 2·2·2 KB ≈ 264 KB | `num_k_blocks = 1`, no `cb_partials` |

**Refinement 11 (split cost model, R1a re-split / injector prefetch, epilogue-none raw walk) as built.** R1 / R1a; no new CB.
- **`cb_act_tiles` / `cb_weight_tiles` depth 3** (`R1A_PREFETCH_DEPTH`, carried in `Blocking.depth_act` / `depth_weight`) on a plain R1a helper walk whose line injector binds (`R1A_INJECTOR_AHEAD = 2`, gated by `split_cost_terms`: busiest-core reads ≥ `R1A_PREFETCH_MIN_RATIO = 2` × the K-walk). The injector keeps K-block kb+1's reads in flight (trid `(seq % 2) + 1`, slot `ring base + (seq % 3) · slot bytes`, the CB is exactly 3 slots) while kb is multicast; compute holds kb−1. The solve shrinks `block_k` (`_block_k_max` at depth 3) when the third slot does not fit, and drops the prefetch if the walk would leave R1a. Shares with / why not: unchanged — both rings are in0 / in1 of the same matmul, concurrently live. Every other walk keeps depth 2.
- **`MIN_K_BLOCKS = 8`**: a plain R1a helper walk (no head ramp) of fewer K-blocks is re-split into ≥ 8 K-blocks of ≥ `BLOCK_K_FLOOR` (only while the K-block stays above `MCAST_MIN_KBLOCK_BYTES`). Every CB term only shrinks (smaller `block_k`).
- **Epilogue-none raw walk** (`NONE_RAW_WALK`, compute-bound splits only): `cb_partials` (already allocated for every multi-K-block walk, one block) is now the raw walk's Interm target and an identity copy drains it to `cb_output_tiles` row by row — the same lifecycle as under an epilogue. No size change.
- **Split ranking** (`SPLIT_COST_MODEL`): the chosen grid / blocking per shape changes (every candidate ranked by `split_cost`), not the CB formulas; the S3 block search prices padded blocks and may step `block_n` by `q_n_min`, so subblocks 1×2 / 1×3 / 3×1 occur where the 4-tile quantum would pad (FLUX single out: 7×10 blocks, 1×2 subblock).
- **Data-movement budget**: unchanged per regime (the tables below). What changed is which regime / split a shape lands in: FLUX proj_out moves from R1 64×1 (B ×64 = 50 MB) to R1a 11×4 (A ×1, B ×1 = 13.4 MB); text-encoder qkv from R1b (A through one injector) to R1a 8×11 (A ×1, B ×1).

## Data-movement budget (chosen split R1, interleaved)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| A | `grid_n' · nb_n` | Each core reads its own `block_m` rows per N-block; nothing is shared between the cores of a grid row (stepping stone) | none |
| B | `grid_m' · nb_m` | Each core reads its own `block_n` columns per M-block | none |
| output | 1 | Written once, after the last K-block (the K-sum stays in `cb_partials`, so there is no read-modify-write in DRAM) | none |
| bias / gate | `grid_m' · nb_m` (one row per block, ≤ `block_n` tiles) | reuse-shared row, re-read per block | none |
| residual | 1 | streamed once alongside the output | none |

**Total bytes per tier, prefill gate_up example** (A bf16 21.0 MB, B bf4b 25.1 MB, out bf16 17.8 MB, grid 11×10 with `grid_m' = 11`, `grid_n' = 10`, `nb = 1`):
- DRAM: A ×10 = 210 MB, B ×11 = 276 MB, out 17.8 MB, total ≈ 504 MB
- cross-core: 0
- core-local: each K-block is read once from its CB; the partials are read and written once per K-block per tile (L1 accumulation)

**1-D (decode-shaped) example** (1×6144 @ 6144×6144 bf16, `grid_n' = 96`):
- DRAM: A ×96 = 37.7 MB, B ×1 = 75.5 MB, out 12 KB, total ≈ 113 MB

**R1a as built (Refinement 3)** — DRAM crossings per tensor when `uses_operand_mcast()` holds:

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| A | `nb_n` | one injector per grid row-line reads each A K-block of its rows once per N-block | multicast of every A K-block to the `grid_n' − 1` other cores of the line |
| B | `nb_m` | one injector per column-line reads each B K-block of its columns once per M-block | multicast of every B K-block to the `grid_m' − 1` other cores of the line |
| output, bias / gate, residual | as R1 | unchanged (per-core) | none |

Prefill gate_up (grid 11×10, block 6×28, `nb = 1`): DRAM ≈ 21 MB (A) + 25 MB (B) + 17.8 MB (out) ≈ 64 MB, down from ≈ 504 MB.

> Cheapest-traffic split considered: `R1a interleaved_2d_mcast` (and `R1b interleaved_stream_mcast_a` for 1-D grids). Delta for prefill: DRAM 504 MB → 64 MB (−440 MB), adding NoC multicasts of each A K-block to `grid_n'−1` receivers per row and each B K-block to `grid_m'−1` per column. Delta for the 1-D example: DRAM 113 MB → 75.5 MB (−37.7 MB), adding a broadcast of A (≤ 0.4 MB) to 96 cores.
> Implemented: `R1 interleaved_out_split`.
> Deferred because R1 is the stepping stone both successors are built on. They keep R1's per-core region, CB inventory (no new CB: the multicast lands in `cb_act_tiles` / `cb_weight_tiles`), compute kernel and block walk; only the reader's source for one operand changes from DRAM to an injector. R1's factorization already maps the logical `(grid_m', grid_n')` onto a physical rectangle, so the multicast groups (grid rows and columns) exist unchanged.

**R4 `dram_sharded` as built (Refinement 4)** — DRAM / NoC crossings per tensor:

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| A | 0 | resident L1 width shards, read where they live (no reshard, no interleaved copy) | each A K-block multicast once from its owning core to the workers' bounding rectangle, `nb_m` times (1 for decode) |
| B | `nb_m` (1 for decode) | each bank worker streams its own bank's column shard, one coalesced read per K-block (`k_valid · sw` contiguous tiles) | none |
| output | 0 | packed in place into the worker's own L1 shard | none |
| bias / gate | 1 per worker block (`sw` tiles) | per-block row read | none |
| residual | 1 | streamed with the epilogue | none |

PERF FOCUS 2 (1×4352 @ 4352×5120 bf4b): DRAM = B once = 12.5 MB (the roofline's traffic); NoC = 8 A shards × 34 KB multicast.

**Refinement 6 delta (`cores_per_bank` split):** DRAM crossings unchanged (B once: each slice worker reads its `sw / C` columns of every K row — `k_valid` row reads of `sw/C` tiles per K-block instead of one coalesced run; a slice wholly in the last bank's padding reads nothing). The A multicast now lands on the C·banks workers' bounding rectangle (PERF FOCUS 2: 100 cores) and runs on NoC1 from the writer RISC. Output: C > 1 adds one NoC write per slice tile into the bank's output shard (on its primary worker), `M · sw/C` tiles per worker.

**R3 `block_sharded` as built (Refinement 5)** — DRAM / NoC crossings per tensor (grid `rows × cols`):

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| A | 0 | resident L1 block shards, read where they live; the own shard is consumed zero-copy | each A shard multicast once along its grid row (`cols − 1` receivers), `nb_n` times (1 when the output is in place) |
| B | `rows · nb_m` | each core reads its column slice from DRAM (Refinement 10: 1 under the R3a column multicast, below) | none |
| output | 0 | packed in place into the core's own L1 shard | none |
| bias / gate | 1 per core block | per-block row read | none |
| residual | 1 | streamed with the epilogue | none |

PDL res5 expand (2048×512 @ 512×2048 bf8b, grid 8×10): DRAM = B ×10 ≈ 11 MB (the bound: 45.6 µs vs stock 24.2 µs); NoC = each 7×2-tile A shard multicast to 7 row peers. The next lever is R3a's B column multicast (B once from DRAM).

**R2 `height_sharded` as built (Refinement 7)** — DRAM / NoC crossings per tensor (`C_A` = A's shard cores):

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| A | 0 | resident L1 height shards holding all of K, consumed in place (`cb_act_shard` is compute's A operand; no NoC read at all) | none |
| B | `C_A · nb_m` (`nb_m = 1` when the whole shard block fits) | each core reads its whole `K × N` B range from DRAM (Refinement 10: 1 under the R2a multicast, below) | none |
| output | 0 | packed in place into the core's own L1 height shard | none (fallback: local writes through the sharded accessor) |
| bias / gate | 1 per core block | per-block row read | none |
| residual | 1 | streamed with the epilogue | none |

PDL res3 reduce (8192×512 @ 512×128 bf8b, 86 cores): DRAM = B × 86 ≈ 6.0 MB (≈ 13 µs at 460 GB/s — the bound: 20.2 µs measured vs stock 6.9 µs). The next lever is the deferred R2a `height_sharded_mcast_b`: B once from DRAM, multicast to the shard cores.

**R2a / R3a weight multicast as built (Refinement 10)** — `SHARDED_B_MCAST` (single B; groups of > 1 core; R2 core sets of ≤ 3 rectangles; R3 only when the saved B re-reads `(rows − 1)·K·N·b` ≥ `BS_B_MCAST_MIN_SAVED_BYTES` = 3 MiB). Crossings per tensor (R3 grid `rows × cols`; R2: `C_A` shard cores = one group):

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| A | 0 | unchanged (resident shards; R3 row multicast) | unchanged |
| B | **1** (× `nb_m`) | one fixed injector per B group (R3: grid column, staggered over rows; R2: the shard set) reads each K-block once | multicast of each B round (`g` K-blocks) to the group's other `rows − 1` cores (R2: `C_A − 1`, two rectangles on PDL's 86 / 103-core sets) |
| bias / gate | **1** for the walk's first block (then 1 per core block) | the injector reads the first row beside B and multicasts it in one row round (every core reading the same row tiles was a ~2.5 µs bank hotspot) | one row multicast per group |
| output, residual | as R2 / R3 | unchanged | none |

PDL res3 reduce (HS, 86 cores): B DRAM 6.0 MB → 70 KB; 20.6 → 8.4 µs. res5 reduce 2 (BS 8×10): B 11 MB → 1.1 MB; 40.9 → 22.9 µs. CB delta: **no new CB.** `cb_weight_tiles` holds whole B rounds: depth `2·g` K-blocks (`g` = `sharded_b_round`: the smallest divisor of `num_k_blocks` with a ≥ 32 KB round, shrunk until `_footprint` fits; `g` when the walk is one round; `g = 1` is the old `_weight_depth`). The receivers' slot is the landing zone (lockstep pointers; handshake = slot reserved). The `weight` channel adds 2 L1 semaphores per core. Shares with / why not: unchanged (in1 of the matmul, live beside `cb_act_tiles` / `cb_act_shard`).

**R1d batched (Refinement 7)**: DRAM crossings as R1 per batch (A and B of batch b never cross batches); PERF FOCUS 3 (32 × 512×128 @ 128×1024 bf8b, B in L1): 96 cores, 70.8 µs (stock 123.0 µs).

**Refinement 8 delta (residency + `L1_READ_COST = 0.25`)** — the split ranking weighs an L1-sourced operand byte at 0.25 of a DRAM byte (R1 runs out of DRAM bandwidth; on write-bound shapes DRAM operand reads compete with the output stream, L1 reads only load the NoC), and a resident operand counts once per core:

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| A (`act_resident`) | `grid_n' · 1` | each core reads each of its M-blocks' A once, held across its N-blocks | none |
| B (`weight_resident`) | `grid_m' · 1` | each core reads its B block once, held across its M-blocks | none |
| output | 1 | unchanged | none |

PERF FOCUS 3 (32 × 512×128 @ 128×1024 bf8b, B in L1): split 32×1×3 → **32×3×1** (96 cores, `act_resident`, 8-column B blocks). DRAM: A ×3 (6.6 MB) → **A ×1 (2.2 MB)** + out 17.8 MB = 20.0 MB (the roofline's traffic). L1 → NoC: B re-read ×3 (13.2 MB, NoC0, overlapped with the write stream). 70.3 → 65.2 µs; the DRAM write stream binds (write-only 59–61 µs at any core count from 32 to 96, ≈ 300 GB/s for 1088 B interleaved tile writes).

**R1b `interleaved_stream_mcast_a` as built (Refinement 9)** — DRAM / NoC crossings per tensor when `operand_mode() == 2` (1-D N split over `grid_n'` cores, weight-streaming shape):

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| A | `nb_n` (1 when the whole region is one block) | one injector (the writer RISC of `ACT_STREAM_INJECTOR`) reads each A K-block once and broadcasts it | multicast of every A K-block to the other `grid_n' − 1` cores (NoC1) |
| B | 1 (× `nb_m`) | each core streams its own column slice (NoC0) | none |
| output, bias / gate, residual | as R1 | unchanged (per-core) | none |

PERF FOCUS 4 (256×6144 @ 6144×18432 bf16, swiglu; 96 cores, block 8×6): DRAM = A 3.1 MB + B 226 MB + out 4.7 MB ≈ 234 MB (the roofline's traffic), NoC = 3.1 MB A multicast to 95 cores. Before (R1a 8×9, 72 cores): the same DRAM bytes, but B streamed by only the 9 column injectors (25 MB each), 1125 µs; now 633–646 µs, B-stream-bound (B alone 550 µs ≈ 411 GB/s for interleaved 2 KB tile reads; with the A lockstep 578 µs). The single injector caps R1b at ~23 GB/s of A, so the split ranking counts R1b at `max(A + B, ACT_STREAM_INJECTOR_COST · A)` (= 18): text-encoder down 512×16384×5120 (A = 19% of B) keeps R1a.

**R1b bank-local split as built (Refinement 15)** (`grid.placement == "bank"`, `operand_mode() == 2`). The CB inventory and sizes are R1b's; no new CB. The core's region is `core_m × core_n`, where `core_n` = the longest unit run × unit columns, and its columns sit 8 apart in B. Idle cores (w = 0) keep the nominal CBs and walk; they read and write nothing. The rotating injectors reserve through their next turn in the existing `cb_act_tiles` ring: no extra L1, `injectors < depth_act` (the ring holds the next turn while compute holds the current one).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| A | 1 | 3 rotating injectors each read every third A K-block once | multicast of every A K-block to the whole grid rectangle, once (the flat split sent each byte twice over its two rectangles) |
| B | exactly 1 | each core streams its own bank run per K row, from the bank of its home row | none |
| output, bias / gate, residual | as R1 | per-core, columns 8 apart (per-tile writes / reads) | none |

Qwen3.8 M = 128 down (4352×5120 bfp4, A in L1): DRAM = B 12.5 MB + out 1.3 MB; NoC = 1.1 MB of A multicast once to 109 cores. 41.5 µs: the A stream (~36 µs) binds, and B alone is 34 µs.

## Refinement 13 — split output (`chunks`)
- **No CB change.** `cb_output_tiles` keeps its size and its single producer (compute) / single consumer (writer). The reader still writes its tail share straight out of that CB's L1 under the existing post / done handshake, without popping it.
- **Data-movement budget unchanged.** The output is written once, tile by tile, as before; only the destination tensor of each tile column changes (`kernels/out_chunks.hpp`). There is no full-width intermediate and no extra op. The `chunks` tensors are DRAM (or L1) interleaved like the unsplit output: `l1_cb_budget` counts all of them when they live in L1.

## Refinement 14 — A as K segments (list `input_tensor`)
- **No CB change.** `cb_act_tiles` keeps its size, its producer (the reader; R1b: the writer) and its consumer (compute). Only the source address of each A tile changes (`kernels/a_segments.hpp`).
- **Data-movement budget unchanged.** A is read exactly as often as the single-tensor call reads the concatenation (same split, blocking and A residency); there is no concat intermediate and no extra op. When the segments live in L1, `l1_cb_budget` counts every one of them.

## Refinement 16 — R1a split lines, R1b ring fallback, operand-aware subblocks

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_act_stage` (index 6; R1a split lines only) | `block_m_tiles · block_k_tiles` (one A K-block) | this core's next A-line turn, from its first chunk read until its send | `{b: streams, m: spans → block_m_tiles, n: streams, k: spans → block_k_tiles}` | A dtype | writer (raw scratch: the A-line injector reads its turn here) | writer (multicast source, its own slot included by loopback) | whole kernel | **None.** It cannot be a `cb_act_tiles` ring slot: a ring slot frees only when compute pops the round `depth_act` rounds earlier, so reading the next turn into it left the read on the critical path every round (measured: round issue blocking the line, Qwen3.8 M = 512 q512_in 84 µs vs 75 µs with staging + paced issue). Not shared with `cb_weight_stage`: different RISCs, concurrent, different formats. |
| `cb_weight_stage` (index 7; R1a split lines only) | `block_k_tiles · block_n_tiles` (one B K-block) | this core's next B-line turn | `{b: streams, m: streams, n: spans → block_n_tiles, k: spans → block_k_tiles}` | B dtype | reader (raw scratch) | reader (multicast source) | whole kernel | **None.** Same reason, the B line's twin. |

- **`cb_act_tiles` producer under the split lines = the writer** (BRISC runs the A row line, as R1b's A stream): single producer (the reader no longer touches it), single consumer (compute). Capacity unchanged (`depth_act = 2`).
- **Solve**: a split-lines walk is solved with `depth + 1` K-blocks of each operand (the staging buffers; `_choose_blocking(..., (da + 1, dw + 1))`), and kept only if the output block is unchanged and the footprint `F + staging_bytes` fits (`staging_bytes = block_k · (block_m · a_tile + block_n · b_tile)`); the write-batch grant counts the staging too. Text encoder down 512×16384×5120: `block_k` 31 → 20 (26 K-blocks), output block 2×15 unchanged.
- **R1b ring fallback** (`ACT_STREAM_MIN_DEPTH = 3`): when the depth-4 rings would split the core's rows into M-blocks (B streamed once per M-block), the solve takes the deepest ring ≥ 3 that keeps all rows in one block. Text encoder gate 512×5120×16384: block 8×5 (2 M-blocks, `depth 4`) → 16×5 (`depth 3`, `block_k` 9).
- **Subblock choice** (R1 family, `SUBBLOCK_UNPACK_AWARE`): the cheapest `max(FPU, unpack(A) / sb_w, unpack(B) / sb_h)` first, then the larger area; no capacity changes except `cb_output_tiles` (`depth_out · sb_h · out_block_n`, now from the chosen `sb_h`; `_footprint` prices the taller of the operand-aware and the area-first `sb_h`, since it is shared with R2 / R3 / R4).

**Data-movement budget, R1a split lines** (`r1a_split()`):

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| A | `nb_n` (as R1a) | each A K-block is read once per row line, by the line core whose turn it is (round r from core r % grid_n', on the writer RISC / NoC1) | multicast of every A K-block to the `grid_n' − 1` other cores of the line (NoC1) |
| B | `nb_m` (as R1a) | each B K-block is read once per column line, by its turn's core (r % grid_m', reader / NoC0) | multicast of every B K-block to the `grid_m' − 1` other cores (NoC0) |
| output, bias / gate, residual | as R1 | unchanged | none |

DRAM bytes are R1a's; what changes is who issues them: a line's reads are spread over its cores (Qwen3.8 M = 512 in_proj: a column line's 1920 bfp4 tile reads, ~50 µs on one core's RISC, now ~1/8 per core) and over the rounds (chunks over the `R1A_LINE_WINDOW = 2` rounds before each turn).

## Refinement 18 — PDL sharded 1×1 convs: one-shot B channel, B groups, split roles

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_weight_tiles` (R2 / R3, one-shot walks with a row epilogue) | `depth_weight · block_k_tiles · block_n_tiles` **+ `row_tail` pages** (`row_tail = ceil(block_n · row_tile_bytes / b_tile_bytes)`, `sharded_b_one_shot`) | the walk's only B round, plus the first bias / gate row in the tail past it (until every core copied it into `cb_row`) | `{b: streams, m: streams, n: spans → block_n_tiles, k: spans → K}` (one round = the whole walk); the tail holds one N-block row | B dtype (the tail is raw bytes of the row's dtype; it is never pushed) | reader (the group's injector reads B + the row into it; the multicast lands both on every core) | compute (the round); the reader itself (the tail, L1 → L1 into `cb_row`) | kernel | **Shares the allocation with the row's landing zone.** One delivery per walk is what makes the channel handshake-free and race-free (one flag set, one clear), so the row must ride in B's delivery; a separate `cb_row` landing would need a second flag delivery. Priced in `sharded_b_one_shot` (falls back to the handshaked rounds when the tail does not fit). |
| `cb_act_tiles` (R3a, parked `BS_A_ONE_SHOT`) | `num_k_blocks · block_m_tiles · block_k_tiles` when on (default `DEPTH_ACT = 2`) | every peer shard of the walk's only block | `{k: spans → K}` | A dtype | reader | compute | kernel | **None** (in0 of the matmul). Parked off: measured slower (concurrent row multicasts serialize). |

- **Producers under the R3 split roles** (`BS_B_ON_WRITER`, in-place R3 walks): `cb_weight_tiles` and `cb_row` are produced by the reader-kernel instance on BRISC (`GM_BS_ROLE 2`); `cb_act_tiles` / `cb_act_shard` and `cb_residual_tiles` by the NCRISC instance (`GM_BS_ROLE 1`). Each CB keeps exactly one producer and one consumer (compute). The output CB is the output's own shard (packed in place, sized to the whole shard), so it has no consumer kernel: the writer kernel is not placed (compute never waits on its space).
- **R2a B-read helper** (`SHARDED_B_READ_HELPER`): on each multicasting group's injector, a reader instance on BRISC (`GM_BS_ROLE 3`) issues the last `floor(K / 2)` K rows of the walk's only B round into the same `cb_weight_tiles` slot. It never reserves or pushes; the NCRISC reader stays the CB's only producer and pushes after the helper's post. One extra L1 semaphore per core (`GM_BS_B_HELPER_SEM`, the post word). The writer kernel is not placed (in-place output).
- **Semaphores**: R2 now attaches one `weight` channel per B group (disjoint core sets, each at its own ids). The one-shot channel is handshake-free (1 flag semaphore instead of 2).
- **No other CB change.** `BS_SPLIT_B_DEPTH` (split, per-core B, parked at 2 = `DEPTH_WEIGHT`) and `BS_SPLIT_B_LOOKAHEAD` (parked at 0) are live knobs that deepen `cb_weight_tiles` to the whole walk; measured flat.

**Data-movement budget, Refinement 18:**

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| B (R2) | **one per B group** (= rectangle of the HS core set: 2 on PDL's 86 / 103 / 64-core sets) | each group's injector reads B once (`SHARDED_B_RECT_GROUPS`); before, one injector sent B to the rectangles one after another | multicast of B to each group (the groups multicast concurrently) |
| bias / gate (R2 one-shot) | 1 per group | read by the injector into `cb_weight_tiles`' tail, delivered with B | rides B's multicast (+`block_n` tiles per delivery) |
| B (R3 split) | 1 per grid column (now also below the old 3 MiB traffic gate: `BS_SPLIT_B_MCAST_MIN_SAVED_BYTES = 0`) | the column injector's BRISC reads each K-block once, one K-block per round (`BS_SPLIT_B_ROUND_BYTES`) | multicast down the column on NoC1, concurrent with the A row exchange on NoC0 |
| A, output, residual | unchanged | | |

## Refinement 19 — tall, narrow M-only splits (R1e): B multicast per rectangle, A in-flight cap

| CB | Size (pages) | Holds | Axis accounting | Dtype | Producer | Consumer | Lifetime | Shares with / why not |
|----|--------------|-------|-----------------|-------|----------|----------|----------|-----------------------|
| `cb_weight_tiles` (R1e: `R1_B_MCAST` on an M-only, B-resident walk) | `1 · block_k_tiles · block_n_tiles` **+ `row_tail` pages** (`row_tail = ceil(block_n · row_tile_bytes / b_tile_bytes)` with a bias / gate row, else 0) | the core's only B block (all of K × all of N), plus the walk's bias / gate row in the tail past it | `{b: 1, m: streams (B NOT re-read per M-block), n: spans → N, k: spans → K}`; the tail holds one N-block row | B dtype (the tail is raw bytes of the row's dtype; never pushed) | reader (the rectangle's injector reads B + row into it; the one-shot multicast lands both on every core of the rectangle) | compute (B, retained across the M-blocks); the reader itself (the tail, L1 → L1 into `cb_row` once) | kernel | **Depth 1, not `DEPTH_WEIGHT`**: B is loaded once (weight-resident), so the second slot the solve priced could never fill; the tail rides in the freed bytes. The row must ride in B's delivery: one delivery per walk is what keeps the channel handshake-free. A one-core rectangle (no channel) reads its own B and row into the same layout. |

- **`cb_row` (R1 family, `nb_n == 1`)**: unchanged size (one N-block row). The reader now re-reserves / re-pushes it on every block after the first **without re-reading**: the CB holds exactly one row, so the slot it re-reserves is the one compute just popped, still holding the same row (the core's column range is one N-block). Producer / consumer unchanged.
- **No other CB change.** The A in-flight cap (`R1_A_INFLIGHT_TILES`, trids 13 / 14) and the bank stagger change only the reader's issue order within `cb_act_tiles`' reserved slot. The distributed B delivery (`R1_B_DISTRIBUTED`, parked) lands in the same `cb_weight_tiles` layout.
- The `weight` channel adds 2 L1 semaphores per rectangle (one-shot, fixed sender).

**Data-movement budget, Refinement 19 (R1e):**

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| A | 1 (DRAM A) / 0 DRAM (L1 A: every core reads its rows from the L1 banks, all-to-all on NoC0) | each core reads its own rows once | none |
| B | **one per rectangle** of the split (2 on a flat split: the full grid rows + the partial last row; a lone core reads its own) | the rectangle's injector reads it once; before, every core read all of B (res3.0 conv1 on 64 cores: 64 × 35 KB) | one multicast of B (+ row) per rectangle |
| bias / gate | one per rectangle | rides B's delivery (was: every core, once per M-block — a bank-serialized hot spot) | rides B's multicast |
| output | once (L1 interleaved for the PDL convs: all-to-all on NoC1 / the tail share on NoC0) | unchanged | none |
