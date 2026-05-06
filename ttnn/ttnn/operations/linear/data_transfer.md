# Data Transfer Analysis: linear (Phase 0)

Single-core 2D matmul + optional row-broadcast bias add. Static analysis based on
the entry point, program descriptor, and three kernels at
`ttnn/ttnn/operations/linear/`.

```
no-bias:    output = input @ weight
with bias:  output = input @ weight + bias[0, :]
```

Shapes — `input [1,1,M,K]`, `weight [1,1,K,N]`, optional `bias [1,1,32,N]`,
`output [1,1,M,N]`. M, K, N divisible by 32. `Mt = M/32`, `Kt = K/32`,
`Nt = N/32`. All tensors are bf16, TILE-layout, DRAM-interleaved.

---

## 1. Tensor Inventory

| Tensor | Role | Pages (tiles) | Page size | Total bytes |
|--------|------|---------------|-----------|-------------|
| `input` | LHS matrix | `Mt × Kt` | 2048 B (bf16 tile) | `Mt × Kt × 2048` |
| `weight` | RHS matrix | `Kt × Nt` | 2048 B | `Kt × Nt × 2048` |
| `bias` (optional) | row-0 broadcast bias, padded to a single tile-row | `Nt` | 2048 B | `Nt × 2048` |
| `output` | result | `Mt × Nt` | 2048 B | `Mt × Nt × 2048` |

> Tile pages count is exact (no header inflation here): the descriptor uses
> `tensor.buffer_page_size()` and the descriptor sizes every CB to integer
> multiples of one tile (`linear_program_descriptor.py:60-69, 84-159`).

---

## 2. DRAM Read Profile (per program execution)

Single core (0,0) reads everything itself; no senders/receivers, no multicast.

| Tensor | Tiles read (single core) | Transactions | Bytes per call |
|--------|--------------------------|--------------|----------------|
| `input` | `Mt × Kt` | `Mt × Kt` `noc_async_read_tile` calls | `2048` B |
| `weight` | `Kt × Nt` | `Kt × Nt` `noc_async_read_tile` calls | `2048` B |
| `bias` (when present) | `Nt` | `Nt` `noc_async_read_tile` calls | `2048` B |

Each read is followed by an immediate per-tile `noc_async_read_barrier()` and a
single-tile `cb_push_back` (`linear_reader.cpp:52-58, 65-71, 78-84`).

**Total DRAM read bytes per program (no bias)**: `(Mt × Kt + Kt × Nt) × 2048`
**Total DRAM read bytes per program (with bias)**: `(Mt × Kt + Kt × Nt + Nt) × 2048`

Concrete examples (single-core baseline shapes):

| Shape (M,K,N) | input B | weight B | bias B | total reads (bias) |
|---------------|---------|----------|--------|--------------------|
| 32, 32, 32    | 2 KiB   | 2 KiB    | 2 KiB  | 6 KiB              |
| 128, 128, 128 | 32 KiB  | 32 KiB   | 8 KiB  | 72 KiB             |
| 256, 256, 256 | 128 KiB | 128 KiB  | 16 KiB | 272 KiB            |

---

## 3. DRAM Write Profile (per program execution)

| Tensor | Tiles written | Transactions | Bytes per call |
|--------|---------------|--------------|----------------|
| `output` | `Mt × Nt` | `Mt × Nt` `noc_async_write_tile` calls | `2048` B |

Each write is followed by an immediate per-tile `noc_async_write_barrier()` and
a single-tile `cb_pop_front` (`linear_writer.cpp:27-33`).

**Total DRAM write bytes per program**: `Mt × Nt × 2048`.

---

## 4. Cross-Core Data Duplication

| Tensor | Pattern | Notes |
|--------|---------|-------|
| `input`, `weight`, `bias`, `output` | **None** | Single core handles the entire `[Mt, Nt]` output grid (`linear_program_descriptor.py:74-75`). No multicast; no second core that could duplicate reads. |

Phase 0 is single-core by design, so the canonical 2D-multicast matmul
duplication pattern (`num_blocks_x ×` for A, `num_blocks_y ×` for B) does not
apply yet. Refinement to multi-core will introduce the standard duplication
profile: A duplicated `num_blocks_x ×` along the column axis, B duplicated
`num_blocks_y ×` along the row axis — multicast is the standard mitigation.

---

## 5. NoC Channel Balance

| Channel | Driver | Bytes per program (no bias) | Bytes per program (with bias) |
|---------|--------|-----------------------------|-------------------------------|
| **NoC0** (reads, NCRISC reader) | `linear_reader.cpp` | `(Mt × Kt + Kt × Nt) × 2048` | `(Mt × Kt + Kt × Nt + Nt) × 2048` |
| **NoC1** (writes, BRISC writer) | `linear_writer.cpp` | `Mt × Nt × 2048` | `Mt × Nt × 2048` |

Both kernels use the default NoC assignment (`ReaderConfigDescriptor` /
`WriterConfigDescriptor` — `linear_program_descriptor.py:199, 222`). No dual-NoC
reads, no NoC0 writes.

**Balance ratio (read / write, no bias)**:
```
NoC0/NoC1 = (Mt*Kt + Kt*Nt) / (Mt*Nt)
          = Kt × (Mt + Nt) / (Mt × Nt)
```

For square `M = K = N` this collapses to `2 × Nt / Nt² = 2 / Nt`, i.e. NoC0 is
slightly heavier than NoC1 only at small N and tends toward read-light at
large N. A few representative shapes:

| Shape (M,K,N) | NoC0 read (KiB) | NoC1 write (KiB) | NoC0 / NoC1 |
|---------------|-----------------|------------------|-------------|
| 32, 32, 32    | 4.0             | 2.0              | 2.0× read-heavy |
| 128, 128, 128 | 64              | 32               | 2.0× read-heavy |
| 32, 128, 64   | 12              | 4.0              | 3.0× read-heavy |
| 128, 96, 64   | 36              | 16               | 2.25× read-heavy |
| 256, 256, 256 | 256             | 128              | 2.0× read-heavy |

Mild read-heavy bias is the matmul norm: every output tile is the product of
one row-block of A and one column-block of B, so the read volume per output
tile is `Mt × Kt + Kt × Nt` divided by `Mt × Nt`, i.e. `Kt × (1/Nt + 1/Mt)`.
For Phase 0 sizes (single-core, all CBs sized to a full block) this is well
within balanced.

---

## 6. Work Distribution

| Field | Value |
|-------|-------|
| Cores used | **1** — `CoreRangeSet([CoreRange((0,0),(0,0))])` (`linear_program_descriptor.py:74-75`) |
| Tiles per core | `Mt × Kt` input + `Kt × Nt` weight + `Nt` bias + `Mt × Nt` output |
| Load imbalance | n/a (single core) |
| Remainder handling | n/a — Python validation rejects non-32-multiple shapes (`linear.py:96-97`) |

There is no work distribution. The full `[Mt, Nt]` output tile grid lands on a
single Tensix.

---

## 7. Read Amplification

Minimum bytes that **must** be moved from DRAM to L1:

```
input  + weight  [+ bias]
= (Mt*Kt + Kt*Nt + [Nt]) × 2048
```

Bytes actually read by the program:

```
= (Mt*Kt + Kt*Nt + [Nt]) × 2048   (each tile read exactly once)
```

**Amplification: 1.0×.** Every input tile, every weight tile, and (when
present) every bias tile is read from DRAM exactly once. This is optimal for a
single-core layout — there is no second core that could share or duplicate
reads.

---

## 8. Per-Tile vs Bulk Read Tradeoff

The reader uses a strict per-tile pattern (`linear_reader.cpp:52-58, 65-71,
78-84`):

```
for each tile:
    cb_reserve_back(cb, 1);
    noc_async_read_tile(...);
    noc_async_read_barrier();   // ← per-tile round-trip
    cb_push_back(cb, 1);
```

**Implications**:

- **Pros**: Tiles become available to the matmul helper as soon as they land
  in L1, allowing compute-side `matmul_block` to start on the first K-tile
  pair while later tiles are still in flight. The CBs are sized to the full
  block (`Mt*Kt` for input, `Kt*Nt` for weight) so backpressure never stalls
  the reader.
- **Cons**: Every tile incurs a NoC round-trip stall at the barrier instead
  of overlapping the second-half of one read with the dispatch of the next.
  For Phase 0 sizes (Mt*Kt up to ~Kt × Mt = 8x8 = 64 tiles for 256×256×256)
  the cumulative barrier overhead is noticeable but not dominant relative to
  matmul compute time.

A bulk pattern (`reserve K tiles, issue K reads, single barrier, push K`)
would amortize barrier overhead but is unnecessary at Phase 0's scales — the
matmul helper waits on the **entire** `Mt*Kt` block before consuming
(`matmul_block_helpers.inl:206-207`), so streaming reads earlier do not
unlock earlier compute work for `num_k_blocks=1`.

The same reasoning applies to the writer: the helper packs one tile per
subblock (`out_subblock_h=out_subblock_w=1`), and the writer drains one tile
per `cb_pop_front`. The per-tile barrier matches the producer cadence.

---

## 9. Refinement Implications

| Refinement | Expected DRAM-traffic change |
|------------|------------------------------|
| **Multi-core (2D mcast)** | Total DRAM reads drop from `O(N_cores × Mt × Kt)` (naive duplicated) to `1.0×` of the matrices when sender/receiver multicast is wired up (`reader_bmm_tile_layout_in0_sender_*` / `reader_bmm_tile_layout_in1_sender_*`). Without multicast the reads scale with grid_y × grid_x — that is the optimization target. |
| **Sharded inputs** | Resident-shard reads drop to **0** (data is already in the local L1 shard); the reader becomes a pure CB hand-off for the non-resident operand. Halves total DRAM traffic if both inputs are sharded. |
| **`packer_l1_acc`** | No DRAM-traffic change. Eliminates the bf16 spill-and-reload through `cb_partials`, but at Phase 0 with `num_k_blocks=1` there is no spill anyway, so the only benefit is in future K-blocked refinements. |
| **K-blocking (`num_k_blocks > 1`)** | No DRAM-traffic change — input/weight are still each read once. CB sizing changes (input/weight CBs shrink to `in0_block_w * Mt` / `in0_block_w * Nt` per block), DRAM read cadence changes (per-block streaming instead of one-shot fill). |
| **bf16 → bf8 / bfp4 inputs** | Linear reduction in NoC0 bytes — bf8 halves the page size to 1024 B, bfp4 quarters it. Tile counts and transaction counts unchanged; total bytes drop proportional to dtype width. |

---

## 10. Key Observations

- **Read amplification is already optimal at 1.0×.** Each input/weight/bias
  tile is read exactly once. There is no cross-core duplication to fix
  because there is only one core. The next bandwidth lever is multi-core
  with sender/receiver multicast — which preserves the 1.0× amplification
  by design but distributes the work across N cores.

- **Per-tile barriers are the dominant reader inefficiency at Phase 0.**
  Each `noc_async_read_tile` is followed by a same-iteration
  `noc_async_read_barrier()`. Since the matmul helper waits on the full
  block before starting, batching reads (one barrier at the end of each CB
  fill loop) is a strict win. This is a refinement target — see notes on
  multi-core, where the standard sender pattern issues a batched
  `noc_async_read_tile`-then-barrier across the whole block.

- **NoC channel balance is mildly read-heavy (typical matmul shape).** For
  square shapes the ratio is `2 × Nt / Nt² = 2 / Nt`, dropping below
  read-heavy as N grows. NoC1 is not a bottleneck at Phase 0; both NoCs
  remain on default reader/writer assignment.

- **Bias adds modest read traffic (`Nt × 2048` B).** With `Nt` capped at
  `N / 32` and bias being a single tile-row, the bias path adds ~3% extra
  reads at 256×256×256 and rapidly becomes a rounding error at larger N.
  The reader pushes bias **first** (`linear_reader.cpp:49-59`), but since
  the matmul helper does not wait on `cb_bias_tiles` and the bias CB is
  sized to `Nt` tiles, the ordering choice has no functional effect on
  throughput at Phase 0.

- **Write side is the simpler half.** The writer issues exactly `Mt × Nt`
  per-tile writes, one barrier each. The compute helper's
  `OutputLayout::SubblockMajor` pack with `1×1` subblocks emits tiles in
  natural row-major DRAM page order, so the writer's `i = m*Nt + n`
  indexing is a direct read off the natural tile-linear index — zero
  reordering, zero scatter.
