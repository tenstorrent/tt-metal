# MoE fused SwiGLU kernel: core placement, dataflow, M-blocks, NoCs, and overlap

This document explains the current C++ implementation of
`ttnn.experimental.deepseek_prefill.moe_fused_swiglu`. The source of truth is the C++ program
factory and the three device kernels in `device/kernels/`. The Python geometry and descriptor under
`ttnn/ttnn/operations/moe_fused_swiglu/` mirror the implementation for diagnostics and tuning.

The kernel computes one routed expert:

```text
G   = X @ W_gate
U   = X @ W_up
H   = SiLU(G) * U
Out = H @ W_down
```

`H` is never materialized in DRAM. The operation is one device program containing:

* one reader data-movement kernel on NCRISC/NoC0;
* one writer data-movement kernel on BRISC/NoC1;
* one compute kernel running on the three compute TRISCs;
* all three kernels on every core in the selected rectangular worker grid.

The optional SiTU-GLU activation changes the compute epilogue but not the placement or transport
topology described here.

---

## 1. The most important answer: what uses both NoCs?

There are two meanings of "uses two NoCs":

1. Two independent streams run concurrently, one owned completely by NoC0 and one by NoC1.
2. One logical operand is deliberately divided between the two NoCs.

The kernel uses both forms, but it is very careful never to give two RISC-Vs ownership of the same
circular-buffer FIFO state.

### 1.1 Shipped NoC ownership matrix

| Movement | NoC0 / reader | NoC1 / writer | Uses both at once? | Main overlap purpose |
|---|---|---|---|---|
| Count, expert index, region offset | All reads | None | No | Dispatch setup |
| Activation `X` from DRAM | Reads its injector rows | None | No | Feeds the latency-critical first matmul |
| Horizontal `X` multicast | Sends/receives | None | No | Shares one activation read across hidden columns |
| Gate/up weights | `W_gate` | `W_up` | **Yes** | Two independent weight streams overlap each other and X/compute |
| Gate/up reduce-scatter | UP payload | GATE payload | **Yes** | Both accumulator blocks scatter concurrently |
| `W_down` DRAM load | Head rows not assigned to the writer | Tail `floor(rows×3/8)` | **Yes** | One logical W-down block is filled by both NoCs |
| Finished local `H` slice | None | Sends slice to column root or row aggregator | No | Assembly is performed by the transfers themselves |
| Grid-wide `H` broadcast | Normally all rounds | Optional complete rounds selected by a mask | Normally no | Delivers H to every output-shard core |
| Output writeback | None | Writes output rows | No | Can remain in flight across an M-block boundary |

The three principal dual-NoC mechanisms are therefore:

```text
NoC0                             NoC1
----------------------------     ----------------------------
W_gate chunks                    W_up chunks
UP reduce-scatter payload        GATE reduce-scatter payload
W_down head rows                 W_down tail `floor(rows*3/8)`
```

### 1.2 What is deliberately not byte-split across NoCs

The `H` multicast is normally owned by NoC0 as a complete round. A dual-RISC byte split of one H
broadcast was implemented and measured, but dropped. The NoC1 portion could not share the reader's
linked data-plus-VALID chain and needed an additional acknowledged completion barrier. That cost
more than the second lane saved.

The retained experimental control transfers **whole H rounds** to NoC1 through
`H_ROUND_NOC1_MASK`. A round is entirely reader-owned or entirely writer-owned. The shipped mask is
zero, so the reader owns all H broadcasts by default.

This whole-round ownership is an important general rule in the kernel: splitting independent
payloads is cheap; splitting one protocol across two RISC-Vs often introduces a cross-RISC
completion problem that erases the gain.

---

## 2. Which chunkings exist to create data-movement/compute overlap?

Not every block or slice is an overlap mechanism. Some are mathematical sharding, some are
flow-control units, and some exist specifically to expose pipeline overlap.

### 2.1 Chunking and streaming classification

| Mechanism | Grain | Primary reason | Is it primarily an overlap optimization? |
|---|---|---|---|
| `GU_CHUNKS` | Hidden-N chunks of W-gate/W-up | Compute chunk `c` while reading chunk `c+1` | **Yes** |
| Full-M progressive X publication | One M tile-row at a time | Start W-up matmul before X multicast completes | **Yes** |
| Next-M-block X prefetch | One injector row for block `b+1` | Hide activation DRAM under block `b` reduce/down | **Yes** |
| Deferred W-down barrier | One hidden K-block ahead | Land W-down under the following H collective | **Yes** |
| Deferred output barrier | One whole output block | Keep output DMA in flight while later work starts | **Yes** |
| Per-row output publication in full-M mode | One output tile-row | Writer emits row `r` while compute produces `r+1` | **Yes** |
| `M_BLOCK = 8` | Up to eight M tile-rows | CB sizing, runtime work granularity, and round-count tradeoff | Partly; not purely |
| Gate/up reduce-scatter slices | A contiguous flattened M×H slice | Distribute reduction and SwiGLU over KGROUPS workers | Primarily load distribution |
| Ordinary W-down K-blocks | One hidden-column block | Mathematical contraction and H ownership | No; scheduling is pipelined |
| Full-M row rounds | One complete H token tile-row | Reduce rounds and eliminate intermediate accumulation | Primarily topology/compute efficiency |
| ND-shard page runs | Consecutive pages within a DRAM shard | Reduce NoC command count | No; request coalescing |
| W-down reader/writer row split | Hidden rows inside each W-down K-block | Give the writer a nominal `3/8` tail and use both data-movement RISC-Vs/NoCs | Bandwidth and issue overlap, not compute chunking |
| H acknowledgement lookahead | Two H slots by default | Allow consecutive H rounds to overlap | Protocol overlap, not data chunking |

### 2.2 Gate/up N-chunking: the clearest compute/dataflow overlap

The hidden block owned by one grid column has physical width `HN_PAD`. It is split into:

```text
GU_CHUNKS = 3
GU_CHUNK_W = HN_PAD / GU_CHUNKS
```

For the common 11x8, hidden-2048 geometry:

```text
HN_PAD = 6 tiles
GU_CHUNKS = 3
GU_CHUNK_W = 2 hidden tiles
```

Each weight chunk contains:

```text
KR_PAD x GU_CHUNK_W tiles
```

The intended pipeline is:

```text
read chunk 0
    compute chunk 0  ||  read chunk 1
    compute chunk 1  ||  read chunk 2
    compute chunk 2
```

Chunking is on the output-N axis rather than contraction-K. N chunks are independent output
columns and can be placed directly into their final M-major accumulator offsets. K chunking would
require an additional accumulating pack for each extra K chunk and was measured as a regression.

Issuing all N chunks at once is also counterproductive. The X staging prologue contains read
barriers; a blanket barrier drains all outstanding reads, turning an apparent prefetch into a
fully paid read before compute starts. The reader therefore issues/publishes chunks in a staggered
sequence so only the immediately needed chunk is behind each barrier.

### 2.3 Full-M row streaming

For `m_eff == M_BLOCK`, the reader reserves the whole resident-X slot but publishes one
`KR_PAD`-tile M row after each horizontal multicast round. Compute runs W-up first and waits for
cumulative prefixes:

```text
row 0 published -> compute may start M row 0
row 1 published -> compute may continue through row 1
...
row 7 published -> the full slot is available
```

W-gate then reuses the same resident X slot.

For smaller `m_eff`, the reader uses one whole-block push. The short multicast does not provide
enough time to repay eight individual CB bookkeeping operations. This is a runtime scheduling
choice; it does not require a separate compiled program for the device-resident count.

### 2.4 Cross-M-block X prefetch

During block `b` phase 2, each core can fetch the one X tile-row it will inject for block `b+1`.
On the supported grids `HGROUPS >= M_BLOCK`, so one core injects at most one row per block and the
one-row BF16 stick CB is sufficient.

This only works because the reader separates two read classes by transaction ID:

```text
transaction ID 14: current block W-down and phase-2 local reads
transaction ID 15: next block X read
```

Every phase-2 barrier is scoped to ID 14. A blanket `noc_async_read_barrier()` would also drain ID
15 and serialize the supposed prefetch. At the end of the block, the reader waits for ID 15 and
publishes the prefetched BF16 stick row or marks the tiled X row ready.

### 2.5 W-down one-block-ahead scheduling

The ordinary phase-2 path has one H multicast and one W-down K-block per hidden-column round. The
read is issued after one round's synchronization and completed/published during the next round:

```text
round r H traffic/compute       || W_down block r+1 lands
round r+1 consumes W_down r+1  || W_down block r+2 lands
```

`WD_AHEAD` remains one. Deeper prefetch was measured and did not help. The critical optimization
was not greater depth; it was moving the barrier away from the issue site so the read had useful
work under which to land.

### 2.6 Deferred output completion

The writer issues block `b` output writes and leaves the output CB occupied. At the top of block
`b+1`, it performs the write barrier and only then pops block `b`'s output pages. `DEPTH_OUT = 2`
makes one outstanding output block legal.

The last block has no successor, so its barrier is paid in the writer epilogue.

In full-M row mode, compute additionally pushes output one row at a time. This overlaps the output
writer with the remaining down matmuls inside the same M-block.

---

## 3. Four meanings of "row"

The code uses row-oriented language at several levels:

| Name | Meaning |
|---|---|
| Token row | One activation vector `X[token, :]` |
| M tile-row | 32 token rows; one row of activation tiles across embedding K |
| Grid row | One logical row of Tensix cores, coordinate `y` |
| Hidden/W-down row | One hidden tile on the W-down contraction axis |

`M_BLOCK = 8` means eight **M tile-rows**, or at most 256 token rows. It does not mean eight
individual tokens and does not mean eight physical grid rows, although the optimized full-M path
intentionally chooses `KGROUPS == M_BLOCK == 8` so one reduce worker can own one M tile-row.

---

## 4. Core placement

Let the selected logical grid be:

```text
HGROUPS columns x KGROUPS rows
```

and let core `C(x,y)` have:

```text
x = 0 .. HGROUPS-1
y = 0 .. KGROUPS-1
```

The host translates every logical coordinate to the worker's virtual NoC coordinate and passes
column peer tables and multicast rectangles as runtime arguments. The blocking model remains in
logical coordinates even if physical worker coordinates contain architecture-specific gaps.

### 4.1 Gate/up placement

For:

```text
X[M,K] @ W_gate[K,H]
X[M,K] @ W_up[K,H]
```

* grid row `y` owns embedding contraction range `K_y`;
* grid column `x` owns hidden output range `H_x`.

Core `C(x,y)` owns:

```text
X[:, K_y]
W_gate[K_y, H_x]
W_up[K_y, H_x]
```

and computes:

```text
G_partial(x,y) = X[:,K_y] @ W_gate[K_y,H_x]
U_partial(x,y) = X[:,K_y] @ W_up[K_y,H_x]
```

Because K is split across rows, the `KGROUPS` partials in one grid column must be reduced.

### 4.2 Down placement

For:

```text
H[M,H] @ W_down[H,K]
```

the output embedding axis is split over all cores. Core `C(x,y)` has row-major linear index:

```text
core_index = y * HGROUPS + x
```

and owns:

```text
W_down[:, EC_core]
Out[:, EC_core]
```

All cores need complete H, but each core produces a disjoint output-column shard. There is no
cross-core output reduction.

This is the fundamental axis rotation:

```text
Gate/up:
    rows split contraction K
    columns split hidden H
    vertical reduction required

Down:
    all cores split output K
    complete H delivered to every core
    no output reduction
```

Moving the hidden activation is cheaper than splitting the hidden contraction and reducing a much
wider `[M, embedding]` output partial.

### 4.3 Example: 11x8, embedding 7168, hidden 2048

```text
EMB_T = 7168 / 32 = 224 tiles
HID_T = 2048 / 32 = 64 tiles
```

The placement becomes:

```text
Gate/up K split:
    224 / 8 = 28 K tiles per grid row

Hidden split:
    HN_PAD = 6
    columns 0..9 own 6 real hidden tiles
    column 10 owns 4 real tiles in a 6-tile physical slot

Down output split:
    224 output tiles / 88 cores
    48 cores own 3 tiles
    40 cores own 2 tiles
    EC_MAX = 3

Gate/up N chunks:
    6 hidden tiles / 3 chunks = 2 tiles per chunk
```

`KR_PAD`, `HN_PAD`, `EC_MAX`, and `WD_EC_MAX` are physical CB strides. Ragged cores execute only
their real prefix while retaining uniform reserve/push/pop amounts where collective agreement
requires them.

---

## 5. Dispatch startup and the runtime mailbox

The actual expert token count remains in device memory. Every reader computes:

```text
global_expert_id = global_expert_idx_table[local_expert_id]
count            = counts[global_expert_id]
M_t              = ceil(count / 32)
m_blocks         = ceil(M_t / M_BLOCK)
```

If shared-buffer extract/insert fusion is active, it also reads:

```text
start_row = expert_region_offsets[global_expert_id]
```

The reader publishes `{count, M_t, m_blocks, start_row}` into a local raw-L1 mailbox and writes a
magic ready word after a fence. The writer and compute consume independent logical CB views of the
same 64-byte allocation.

Compute's UNPACK thread reads the mailbox and explicitly broadcasts `M_t` and `m_blocks` to MATH
and PACK using the hardware inter-TRISC mailbox. A compute CB wait is UNPACK-only, so this explicit
broadcast is necessary to keep all three TRISCs on the same loop count.

If `count == 0`, every core derives `m_blocks == 0`; no M-block CB traffic, collective, or output
write occurs.

---

## 6. M-block arithmetic and why it is a correctness constraint

`M_BLOCK = 8` is the maximum physical M scheduling unit. For block `b`, the kernel derives two
different row counts.

### 6.1 `m_eff`: page, communication, and FIFO rows

Let:

```text
remaining = M_t - b * M_BLOCK
```

`m_eff` is the next power of two covering `remaining`, capped at `M_BLOCK`:

| Remaining tile rows | `m_eff` |
|---:|---:|
| 8 or more | 8 |
| 5-7 | 8 |
| 3-4 | 4 |
| 2 | 2 |
| 1 | 1 |

`m_eff` controls:

* CB reserve, push, and pop counts;
* X multicast rounds;
* reduce-scatter slice construction;
* H transfer sizes;
* down-matmul input consumption;
* physical circular-buffer pointer movement.

The power-of-two rule is load-bearing. M-scaled CBs are sized as a multiple of `M_BLOCK`; making
every runtime block size divide `M_BLOCK` prevents a reservation from beginning near the end of a
CB, crossing its physical limit, and overwriting the next allocation.

### 6.2 `m_rows`: actual gate/up arithmetic rows

`m_rows` is the real number of remaining tile rows, capped at `M_BLOCK`:

```text
remaining 5 -> m_eff 8, m_rows 5
remaining 3 -> m_eff 4, m_rows 3
```

Gate/up computes only `m_rows`. Rows in `[m_rows, m_eff)` are undefined padding.

The down matmul deliberately stays on `m_eff`. Its helper pops `cb_h` according to the matmul
shape. If down were shrunk to `m_rows`, every hidden K-block would under-pop the H CB, its pointer
would drift, and a later round would hang. Page arithmetic and arithmetic work are therefore
allowed to differ only where the input lifecycle makes it safe.

### 6.3 Why `m_blocks` matters

`m_blocks` controls more than an outer-loop trip count:

* block zero fills resident weights;
* later blocks reuse them;
* a second resident-X slot is allocated when the host maximum can exceed one block;
* block `b+1` X can be prefetched during block `b`;
* block `b` output DMA can be drained during block `b+1`;
* L1 alias reuse needs a cross-block completion edge;
* grouped full-M scheduling requires enough complete blocks;
* only the final block can shrink below `M_BLOCK`.

Examples:

```text
count <= 256  -> one M-block
count = 512   -> two full M-blocks
count = 1024  -> four full M-blocks
count = 5120  -> twenty full M-blocks
```

`input_m_tiles` is a host-provided upper bound used to size the program. The actual `M_t` and
`m_blocks` still come from the device-resident count.

---

## 7. Per-M-block timeline

A useful conceptual timeline is:

```text
Reader / NoC0
  X stage -> X row multicast -> W_gate chunks -> W_down head/prefetch
                                      -> UP scatter -> H broadcasts
                                                        || next-block X prefetch

Writer / NoC1
  wait local X staged -> W_up chunks -> W_down tail batch
                                      -> GATE scatter -> H-slice gather -> output issue

Compute
  BF16 tilize -> progressive UP/GATE matmuls -> distributed reduce/SwiGLU -> down matmul
```

Arrows show local program order. Vertical alignment is approximate: CB waits and NoC operations
allow one processor to advance while another is still working.

### 7.1 Stage X and multicast horizontally

For M tile-row `t`, the injector column is:

```text
injector_x = t % HGROUPS
```

Core `C(t % HGROUPS, y)` reads the activation row for grid row `y`'s K shard and multicasts it to
all other columns in grid row `y`.

For a full 11x8 block, columns 0 through 7 inject one row each. Columns 8 through 10 read no X from
DRAM for that block; they receive all eight rows.

Input-format behavior:

* BF16 row-major: reader fetches 32 token sticks for its K slice; compute tilizes directly into the
  reader-reserved final resident-X row.
* BFP8 tile: reader fetches `kr_rows` tiles directly into the resident-X row.

The BF16 stick walk starts at `(my_col + my_row) % 32` and wraps. This staggers simultaneous cores
across DRAM banks without changing which M row each core injects.

The horizontal multicast uses a rotating sender and a consumer-ready/data-ready handshake. The
sender waits until every receiver has reserved its identical resident-X address before sending.

### 7.2 Gate and up matmuls

Every core computes two partial blocks over the same resident X:

```text
[m_rows, kr_rows] @ [kr_rows, hn_cols]
```

The physical output allocation is `[m_eff, HN_PAD]`; only the real prefix is arithmetic work.

W-gate and W-up are streamed on different NoCs. Full blocks run UP first with progressive X
prefix waits, then GATE reuses X. Short blocks use gate-first bulk waits because their multicast
is too short to benefit from row streaming.

### 7.3 Column reduce-scatter and distributed SwiGLU

For a fixed hidden column `x`, every grid row holds a partial `[m_eff,HN_PAD]` GATE block and UP
block. Flatten one block in M-major order:

```text
tile = m * HN_PAD + h
total_tiles = m_eff * HN_PAD
```

Choose the largest worker count `W <= KGROUPS` that divides `total_tiles`. Worker row `r < W`
owns a contiguous slice of `total_tiles / W` tiles. All `KGROUPS` contributors send that slice to
worker `r`:

```text
UP slice   over NoC0
GATE slice over NoC1
```

Each worker folds the `KGROUPS` contributions for its own slice and computes:

```text
H_slice = SiLU(sum(GATE contributions)) * sum(UP contributions)
```

The SiLU-heavy epilogue is distributed over several workers instead of being serialized on one
column root.

For 11x8 with `HN_PAD = 6`:

| `m_eff` | Total tiles | Workers | Slice tiles per worker |
|---:|---:|---:|---:|
| 8 | 48 | 8 | 6 |
| 4 | 24 | 8 | 3 |
| 2 | 12 | 6 | 2 |
| 1 | 6 | 6 | 1 |

Rows beyond `W` still contribute their partials; they simply own no final slice.

### 7.4 One readiness signal for two concurrent scatter payloads

With `SCATTER_ONE_SIGNAL` enabled:

1. Reader scatters UP on NoC0 and waits for its NoC0 payload barrier.
2. Reader writes a same-core mailbox word saying UP is resident remotely.
3. Writer scatters GATE on NoC1 and waits for its NoC1 payload barrier.
4. Writer waits for the reader's UP-complete mailbox word.
5. Writer sends one `SEM_DATA` notification to each destination.

The destination's one arrival therefore proves that **both** independent NoC payloads have landed.
This halves atomic fan-in while preserving concurrent transfers and exclusive CB ownership.

### 7.5 Assemble H

Each active reduce worker has one finished `H_slice`. The writer sends it on NoC1 to an assembly
core. The transfer destination offset is the final position in `cb_h_local`, so the network writes
perform the assembly; the root does not copy/concatenate the slices afterward.

The following phase differs substantially for full and short M blocks.

---

## 8. Ordinary or short-block down schedule

This schedule is used when `m_eff != M_BLOCK` or the full-row optimization is unavailable.

For hidden column `x`, choose root:

```text
root_row = x % KGROUPS
root = C(x, root_row)
```

All H slices in that column land in the root's `cb_h_local`, assembling the complete
`[m_eff,HN_PAD]` hidden block.

Then run one phase-2 round per hidden column:

```text
for hidden block r in 0 .. HGROUPS-1:
    root for column r broadcasts H[:, H_r] to the grid
    each core consumes W_down[H_r, EC_core]
    each core accumulates its output-column shard
```

An 11-column grid therefore performs 11 H broadcast rounds and 11 down K-blocks.

The reader pipelines the W-down head reads across those rounds. Non-final down K-blocks
packer-accumulate into a fixed unpushed BF16 scratch allocation. The final K-block reloads that
partial, adds its contribution, and packs directly into the output-format CB.

---

## 9. Full-M row down schedule

The optimized schedule is available when:

```text
m_eff == M_BLOCK
KGROUPS == M_BLOCK
W_down is resident
WD_MROW_ROUNDS is enabled
```

For the common 11x8 geometry:

```text
m_eff = 8
KGROUPS = 8
```

The `[8,6]` block contains 48 tiles and the eight reduce workers own six tiles each. Because the
flattened layout is M-major, worker row `r` owns exactly one complete HN_PAD-wide fragment for M
tile-row `r`.

Across all hidden columns, row `r`'s fragments are gathered horizontally onto diagonal core:

```text
C(r,r), r = 0..7
```

The diagonal core owns one complete `[1,HID_T]` activation row. The phase then performs:

```text
8 rounds:
    broadcast one complete H token tile-row
    every core computes [1,HID_T] @ [HID_T,EC_core]
```

instead of:

```text
11 rounds:
    broadcast one [m_eff,HN_PAD] hidden block
    accumulate one of 11 W-down K-blocks
```

Consequences:

* eight rounds replace eleven;
* each row matmul has one full-hidden K-block;
* no BF16 accumulation spill is needed between hidden blocks;
* compute pushes one output row at a time;
* writer can issue row `r` while compute produces row `r+1`.

A block with `m_rows = 7` and `m_eff = 8` can still use this physical schedule. The eighth row is
undefined padding, and the writer refuses to write rows at or beyond `M_t`.

### 9.1 Optional grouped schedule

For tuned 12x8 shapes and enough complete M-blocks, the eight grid rows may be divided into two
four-row groups. Each 12x4 group:

* assembles and broadcasts four M rows;
* uses 48 cores to split the complete output embedding;
* communicates over half the grid rectangle;
* produces one half of the M-block's rows.

It is enabled only when every M-block is full. A ragged tail must not change output ownership while
the dispatch continues to reuse one resident W-down layout.

---

## 10. How block zero differs from later blocks

| Property | Block zero | Blocks `b > 0` |
|---|---|---|
| Gate/up weights | Actually read W-gate and W-up from DRAM | Normally skip DRAM, replay the same CB protocol over resident bytes |
| W-down weights | Reader and writer fill the resident W-down payload | Reuse resident bytes |
| Activation prefetch | Cannot have been prefetched | May have one injector row prefetched during block `b-1` |
| Output completion | No previous output exists | Drain block `b-1`'s deferred output DMA at the top |
| Phase-aliased SRAM | No preceding output user | Wait until block `b-1` output DMA releases the shared storage |

Resident-weight reuse preserves every reserve, push, wait, and pop. Only the DRAM issue loops are
skipped. Compute therefore consumes the same logical stream on every M-block.

Block zero also fills W-down cooperatively. `WD_SPLIT = 3` denotes a nominal three-eighths writer
share, but the writer row count is integer-floored independently for each real hidden block:

```text
writer_rows = floor(real_hidden_rows * 3 / 8)
reader/NoC0: all preceding head rows
writer/NoC1: writer_rows contiguous tail rows
```

If the entire dispatch contains exactly one M-block, there is an additional X-priority barrier:
all cores in a grid row finish local X staging before W-gate traffic is released. For a multi-block
dispatch, preserving the steady-state cross-block pipeline is more valuable than that one-block
protection.

The final block is special in the opposite direction:

* it has no next-X prefetch;
* it may have a smaller `m_eff`;
* it pays the final output write barrier in the writer epilogue.

---

## 11. Streaming grain summary

| Data or operation | Streaming grain |
|---|---|
| Expert index/count | Once per dispatch, one page each |
| BF16 X DRAM | Per injector M tile-row: 32 token sticks |
| Tiled BFP8 X DRAM | Per injector M tile-row: `kr_rows` tiles |
| X multicast | Per M tile-row, `KR_PAD` tiles horizontally |
| X CB publication | Per row for full M; whole block for short M |
| W-gate/W-up | Per hidden N chunk, `KR_PAD × GU_CHUNK_W` tiles |
| Gate/up matmul | Per hidden N chunk |
| Reduce-scatter | Per M-block, one contiguous slice per source/destination |
| Finished H transfer | Per reduce worker per M-block |
| Ordinary H broadcast | Per hidden-column block |
| Full-M H broadcast | Per M tile-row |
| Ordinary W-down | Per hidden K-block |
| Full-M W-down | Complete resident shard, reused for every M row |
| Output compute publication | Per row in full-M mode, whole block otherwise |
| Output DRAM write | Per output tile-row, coalesced over the core's output columns |
| Output barrier | At the next M-block, or epilogue for the last block |

In one sentence:

> X is streamed per M row, gate/up weights per hidden-N chunk, reduction per M-block, ordinary H
> per hidden block, optimized full-M H per M row, and output per row.

---

## 12. The synchronization tricks that make the overlap safe

### 12.1 Exclusive CB ownership across the two data-movement RISC-Vs

Independent streams are assigned by complete payload:

```text
reader exclusively owns UP accumulator consumption
writer exclusively owns GATE accumulator consumption
```

Two RISC-Vs must not pop one CB. CB push/pop updates shared page counters such as
`tiles_received`/`tiles_acked`; two independent software views would corrupt those counters even if
the byte ranges did not overlap.

### 12.2 Cross-RISC W-down completion

`noc_async_read_barrier()` proves completion only for the calling RISC-V. Since one W-down K-block
is filled by NoC0 and NoC1, the reader cannot publish it after only its own barrier.

The writer publishes a monotone per-K-block completion count through `SEM_WDSPLIT`. The reader:

1. drains its NoC0 head reads;
2. waits until the writer's counter includes the same K-blocks;
3. pushes the complete W-down CB pages to compute.

The writer tags W-down K-block `r` with transaction ID `r+1` and publishes completion block by
block. This lets the reader use early W-down blocks before the writer's entire batch has drained.

### 12.3 Reserve before invite

Before reduce-scatter contributors are allowed to write a destination core's landing CB, that
destination:

1. reserves the complete landing capacity;
2. waits for phase-aliased physical storage to become free;
3. sends `SEM_GO` invites to the whole column.

A contributor waits for every destination's invite before sending. This prevents block `b+1`
from overwriting a block-`b` landing slot that compute has not consumed.

### 12.4 Whole-capacity gather pushes

The gather CB is pushed at its complete logical capacity even when only part is live. That returns
the write pointer to the CB base on every M-block. Contributors can then use their own identical CB
cursor to derive the remote destination address.

The scatter plan also ensures every runtime slice size divides the physical CB capacity. Without
that property, a FIFO operation can run past its end into the next CB rather than wrap at the
intended block boundary.

### 12.5 Per-slot H flags and linked ordering

An H sender performs:

```text
payload multicast, linked
VALID semaphore multicast, non-posted and terminating the link
flush
reset the local rotating-sender flag
```

The linked virtual-channel chain guarantees that a receiver cannot observe VALID before the
payload. The payload itself may be posted; the terminating flag remains tracked and ordered.

There are `DEPTH_H` independent flag cells. Consecutive H rounds therefore do not serialize on one
flag reset. A receiver reserves several future H slots and acknowledges up to `HACK_AHEAD = 2`
senders at once. Round `r+DEPTH_H` cannot reuse a flag until the corresponding old slot has been
consumed.

### 12.6 Exclude-source multicast after a self-copy

The sender first copies its locally assembled H into the exact CB address used on every receiver.
It then performs an exclude-source multicast with `src == dst`.

An earlier include-source/loopback form raced the rotating sender's local flag reset against its own
in-flight loopback VALID write. The result was nondeterministic numerical corruption rather than a
clean hang. Self-copy plus exclude-source removes that race.

### 12.7 Transaction-ID-scoped barriers

Next-block X prefetch and current-block phase-2 reads share NoC0 but use different transaction IDs.
Every current-block barrier is scoped so it cannot drain next-block X.

This is an example of overlap on **one NoC**, rather than two-NoC parallelism: multiple classes of
read remain outstanding on NoC0, and scoped completion waits preserve the intended ordering.

### 12.8 Phase-alias release

For BFP8 output, these logical CBs can share one physical allocation because their same-block
lifetimes do not overlap:

```text
GATHER_GATE -> H_SLICE -> OUT_TILES
```

Their cross-block lifetimes can overlap: a peer might start block `b+1` gather while this core's
writer still has block `b` output DMA reading the same SRAM. After the writer drains output DMA, it
publishes `SEM_PHASE_FREE = b+1`; the reader waits for this before issuing the next gather invites.

### 12.9 Resident data keeps the protocol alive

When weights are resident, later blocks do not simply bypass the CB operations. They reserve and
push the old bytes again. This maintains all producer/consumer credit edges and identical compute
trip counts while eliminating redundant DRAM traffic.

### 12.10 Direct BF16 tilize without becoming a second CB producer

The reader reserves the whole resident-X slot. Compute packs the BF16 row directly into an explicit
offset within that reservation, but it does not push `cb_x_tiles`. It pushes a one-page completion
CB instead. The reader remains the only owner of resident-X FIFO state and pushes the actual row
after seeing the completion event.

The explicit physical offset includes:

```text
(block_index % DEPTH_X) * M_BLOCK * KR_PAD
```

because compute never advances the resident-X CB write pointer itself. Omitting the physical slot
term would make later M-blocks overwrite slot zero.

---

## 13. Circular buffers and L1 placement

The major logical pipeline is:

```text
CB_X_IN -> CB_X_TILES
CB_W_GATE / CB_W_UP
CB_GATE_ACC / CB_UP_ACC
CB_GATHER_GATE / CB_GATHER_UP
CB_SLICE_GATE / CB_SLICE_UP / CB_GATE_SILU
CB_H_SLICE -> CB_H_LOCAL -> CB_H
CB_W_DOWN
CB_OUT_INTERM -> CB_OUT_TILES
```

Important capacities scale as:

```text
resident X:   DEPTH_X * M_BLOCK * KR_PAD
gate/up acc:  M_BLOCK * HN_PAD
W_down:       DEPTH_WD * HN_PAD * WD_EC_MAX
H stream:     DEPTH_H * (M_BLOCK*HN_PAD or HID_T in full-row mode)
output:       DEPTH_OUT * M_BLOCK * EC_MAX
```

Logical alias groups include:

* `CB_X_STAGE`, compute mailbox, and writer mailbox over one 64-byte allocation;
* `CB_GATHER_GATE`, `CB_H_SLICE`, and `CB_OUT_TILES` when their BFP8 page sizes agree;
* `CB_GATE_SILU` and `CB_OUT_INTERM` when their BF16 capacity LCM saves space.

The alias allocation uses the least common multiple of logical page capacities, not merely their
maximum. This preserves each logical view's whole-capacity wrap behavior.

Geometry has a fallback ladder when L1 does not fit. It gives up optional grouped/full-row depth,
then W-down residency/depth and other pipeline depth before failing. Those concessions affect both
performance and which NoC optimizations remain legal: the W-down two-NoC split, for example,
requires a complete resident W-down layout with stable absolute K-block slots.

---

## 14. Weight placement and NoC request coalescing

Weights may be DRAM interleaved or DRAM ND-sharded.

For a recognized ND shard with N width `SHARD_W`, consecutive N pages inside one shard row are
physically contiguous. `WeightRuns<SHARD_W>` issues the longest run ending at the next shard
boundary as one NoC request.

For interleaved placement, consecutive N pages land in different banks, so the safe run length is
one page. An earlier implementation remapped the N axis to coalesce stride-bank runs inside
interleaved DRAM; it was measured as a net regression and removed.

This run coalescing is different from `GU_CHUNKS`:

* `WeightRuns` changes the number and size of NoC requests without changing compute visibility.
* `GU_CHUNKS` changes when partial weight blocks become visible so compute can overlap later reads.

---

## 15. Optimization history and the lesson from each step

The current structure accumulated through measured changes rather than one initial design.

1. **Two-dimensional gate/up blocking and axis rotation.** Split gate/up K across grid rows and
   hidden N across columns; distribute H rather than reduce wide output partials.
2. **Runtime `m_eff`.** Stop computing all eight M rows for small counts. This materially improved
   low-count performance and exposed a previously hidden multicast race.
3. **Deferred read/write barriers.** A prefetch followed immediately by a blanket barrier is not a
   prefetch. Move completion waits under useful work.
4. **Cross-block weight residency.** Weight addresses do not depend on M-block index, so later
   blocks reuse the same L1 bytes while replaying CB flow control.
5. **Reduce-scatter plus distributed epilogue.** Replace root-heavy reduction/SiLU with distributed
   slice work across KGROUPS workers.
6. **Per-slot H flags and ack lookahead.** Remove the round-to-round reset chain without paying an
   acknowledged payload barrier.
7. **N-chunked gate/up weights.** Expose compute/read overlap without K-accumulation packs.
8. **Activation priority and DRAM-bank stagger.** Stage X before allowing the larger W-up stream and
   spread stick reads across banks.
9. **ND-sharded weight runs.** Increase request size using true physical shard contiguity.
10. **Next-M-block X prefetch.** Use transaction IDs and scoped barriers to hide the exposed
    activation read under the preceding block.
11. **Progressive full-M X publication.** Start W-up before the horizontal multicast completes.
12. **Direct BF16 tilize into resident X.** Remove a redundant local NoC copy while preserving one
    logical CB producer.
13. **Direct final down packing.** Keep intermediate BF16 accumulation at a fixed address and pack
    the final K-block directly to output.
14. **Phase-disjoint CB aliasing.** Recover L1 for residency and deeper pipelines, with a new
    cross-block phase-release edge.
15. **Eight-row full-M down schedule.** Replace hidden-column rounds with complete-H-row rounds and
    remove intermediate down accumulation.
16. **Balanced hidden splits.** Keep wide grids active without changing fixed CB slot sizes.
17. **Grouped full-M schedules for measured 12x8 shapes.** Reduce multicast rectangle and assign
    output shards separately within each M-row group.
18. **Shared-buffer extract/insert fusion.** Rebase X reads and output writes through one start-row
    mailbox value, eliminating surrounding DRAM copies.

Several plausible ideas were built and rejected:

* byte-splitting H over both NoCs: completion cost exceeded the bandwidth gain;
* deeper W-down prefetch: no improvement beyond one block ahead;
* K-chunked gate/up: extra accumulation packs exceeded the overlap gain;
* interleaved DRAM bank-run remap: net regression;
* deeper H CB alone: not the bottleneck after per-slot flags;
* single-NoC reduce-scatter: lost useful gate/up payload concurrency;
* smaller M-blocks to create more phase overlap: doubled collective round overhead and regressed;
* retaining multiple W-down K chunks in DEST: unsafe/hanging with the existing block-major H
  stream and therefore removed rather than left as a disabled path.

The recurring lesson is that overlap comes less from adding queue depth than from placing the
completion edge correctly. Most successful changes either:

* let compute see a useful prefix sooner;
* move a barrier below independent work;
* separate traffic classes by NoC or transaction ID;
* or preserve enough buffer ownership to let independent processors proceed concurrently.

---

## 16. Source map

| Topic | Source |
|---|---|
| Public operation and grid/defaults | `moe_fused_swiglu.cpp`, `moe_fused_swiglu.hpp` |
| Core placement and runtime arguments | `device/moe_fused_swiglu_program_factory.cpp` |
| Blocking, CB sizes, aliases, L1 fallbacks | `device/moe_fused_swiglu_geometry.cpp`, `.hpp` |
| Runtime M arithmetic and scatter plan | `device/kernels/moe_fused_swiglu_common.hpp` |
| Shared NoC transport helpers | `device/kernels/moe_fused_swiglu_dataflow.hpp` |
| ND-sharded page coalescing | `device/kernels/moe_fused_swiglu_bank_runs.hpp` |
| NoC0 path, X, W-gate, UP scatter, H broadcast | `device/kernels/moe_fused_swiglu_reader.cpp` |
| NoC1 path, W-up, GATE scatter, H slices, output | `device/kernels/moe_fused_swiglu_writer.cpp` |
| Tilize, matmuls, reduce, SwiGLU, down | `device/kernels/moe_fused_swiglu_compute.cpp` |

---

## 17. Compact mental model

For each M-block, think of the operation as four transformations:

```text
1. Horizontal reuse
   One core per M row reads X; NoC0 multicasts it across a grid row.

2. Vertical partial reduction
   Every core computes its K-shard partial. NoC0 scatters UP and NoC1 scatters GATE.
   Workers reduce and compute SwiGLU on disjoint slices.

3. Axis rotation
   NoC1 gathers finished H slices into an owner. The owner broadcasts H so every output-shard
   core sees the hidden contraction operand.

4. Independent output
   Every core multiplies complete H by its private W-down/output-column shard and NoC1 writes it.
```

The dual-NoC strategy is not "use both NoCs everywhere." It is:

> Use both NoCs when two streams can have clean, exclusive ownership; keep a transport on one NoC
> when splitting it would require a more expensive cross-RISC completion protocol.
