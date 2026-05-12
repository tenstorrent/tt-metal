# Data Transfer Profile: multigammaln_lanczos

> Static analysis of the elementwise unary kernel under `ttnn/ttnn/operations/multigammaln_lanczos/`. Methodology follows `tt_metal/third_party/tt_ops_code_gen/references/data_transfer_analysis_reference.md`.

---

## Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Op classification | Elementwise unary | One output tile per input tile; no broadcast, no reduce, no inter-core comm. |
| Total DRAM read | `N_in × 4096 B` | One fp32 tile = 4096 B (32×32×4 + header). |
| Total DRAM write | `N_in × 4096 B` | Output shape = input shape. |
| Read/write balance | 1:1 | DRAM traffic is symmetric. |
| Cross-core duplication | **None** | `split_work_to_cores` partitions the tile-id range; every core reads a disjoint slice. |
| NoC channel use | **Balanced** | Reader on NoC0 (default `ReaderConfigDescriptor`), writer on NoC1 (default `WriterConfigDescriptor`). |
| Per-core L1 CB footprint | **24 KB** | 3 CBs × 2 pages × 4096 B. |
| Multicast / semaphores | **None** | Pure per-tile streaming. |

---

## 1. Per-Tile Transaction Inventory

Per output tile produced on a given core (`multigammaln_lanczos_reader.cpp` / `multigammaln_lanczos_writer.cpp` / `multigammaln_lanczos_compute.cpp`):

| Direction | Endpoint | Transactions | Bytes | Source |
|-----------|----------|--------------|-------|--------|
| DRAM → L1 | reader → `cb_input_tiles` | **1** `noc_async_read_tile` + barrier | 4096 | `multigammaln_lanczos_reader.cpp:34-35` |
| L1 → DRAM | writer ← `cb_output_tiles` | **1** `noc_async_write_tile` + barrier | 4096 | `multigammaln_lanczos_writer.cpp:32-33` |
| L1 ↔ L1 (intra-core) | compute ↔ `cb_accumulator` | **5** pack + **4** unpack reload | 5×4096 + 4×4096 = **36 KB** | `multigammaln_lanczos_compute.cpp:95, 183, 191` |

The 36 KB of intra-core L1 traffic per output tile is the algorithmic cost of the 4-lgamma accumulator round-trip; it never leaves the Tensix and contributes zero DRAM bandwidth.

### Why one tile per read

The reader calls `cb_reserve_back(cb_input_tiles, 1)` → `noc_async_read_tile(tile_id, ...)` → `noc_async_read_barrier()` → `cb_push_back(cb_input_tiles, 1)` (`multigammaln_lanczos_reader.cpp:32-36`). This **synchronous** per-tile loop blocks on the barrier before pushing — the next read cannot start until the previous one is L1-visible. Each tile crosses the NoC as a single transaction. **Implication**: read latency is exposed serially, not amortised across multiple inflight reads.

---

## 2. Aggregate DRAM Bandwidth

| Quantity | Formula | At `(2, 4, 64, 128)` shape |
|----------|---------|----------------------------|
| `N_in` (tile count) | `N · C · ceil(H/32) · ceil(W/32)` | `2·4·2·4 = 64` tiles |
| Total DRAM read | `N_in × 4096 B` | 256 KB |
| Total DRAM write | `N_in × 4096 B` | 256 KB |
| Reuse factor | 1.0× | No tile is read twice from DRAM; the 4× per-tile re-reads happen entirely from `cb_input_tiles` in L1 (compute does 4 `copy_tile(cb_input_tiles, 0, 1)` calls before `cb_pop_front` — see `multigammaln_lanczos_compute.cpp:110, 195`). |

The 4× re-use of the input tile is L1-resident, not DRAM-resident — this is the central efficiency property of the multivariate fusion.

---

## 3. Per-Core Work Distribution

From `multigammaln_lanczos_program_descriptor.py:48-55`:

```python
(num_cores, all_cores, core_group_1, core_group_2,
 tiles_per_core_g1, tiles_per_core_g2) = ttnn.split_work_to_cores(grid_size, total_tiles)
```

| Case | `core_group_1` | `core_group_2` | DRAM bandwidth per core |
|------|----------------|----------------|-------------------------|
| `total_tiles ≤ num_cores` | `total_tiles` cores × 1 tile | 0 cores | 1 tile read + 1 tile write per core |
| `total_tiles = k·num_cores` | All cores × k tiles | 0 cores | k tile reads + k tile writes per core |
| Uneven (most realistic) | `total_tiles mod num_cores` cores × `⌈total_tiles/num_cores⌉` | rest × `⌊total_tiles/num_cores⌋` | differs by ±1 tile across the grid |

Per-core RT args carry `(buffer_address, num_tiles, start_tile_id)` → cores walk **disjoint** tile-id intervals (`multigammaln_lanczos_program_descriptor.py:122-133`). No DRAM page is read or written by more than one core in a given launch.

### NoC topology — addressing scatter

Pages are interleaved round-robin across DRAM banks (`page_id mod num_banks`). The reader iterates `tile_id = start_tile_id .. end_tile_id - 1` in order; sequential tile IDs hit **different banks** on each NoC transaction, which is the standard interleaved access pattern. `TensorAccessor` resolves the bank + offset for each tile (`multigammaln_lanczos_reader.cpp:34`).

---

## 4. NoC Channel Balance

| Channel | Carries | Bytes per output tile |
|---------|---------|----------------------|
| **NoC0** (reader) | DRAM → L1 input reads | 4096 |
| **NoC1** (writer) | L1 → DRAM output writes | 4096 |

The two NoCs traverse the chip in opposite directions and are fully decoupled. Because each NoC carries exactly the same byte count per output tile and they belong to different RISCV processors (`ReaderConfigDescriptor` → NCRISC/NoC0, `WriterConfigDescriptor` → BRISC/NoC1), the read and write phases pipeline perfectly on every core — there is **no NoC contention** between the reader and writer for the same tile boundary.

`cb_input_tiles` and `cb_output_tiles` are both **double-buffered** (2 pages each, `multigammaln_lanczos_program_descriptor.py:69, 81`), so the reader can prefetch tile `N+1` while the compute consumes tile `N`, and the writer can drain tile `N` while the compute produces tile `N+1`. This is the standard reader/compute/writer triple-overlap pattern.

---

## 5. L1 Footprint

| CB | Pages | Page size | Total | Notes |
|----|-------|-----------|-------|-------|
| `cb_input_tiles` (idx 0) | 2 | 4096 B | 8192 B | fp32, reader↔compute double-buffer |
| `cb_output_tiles` (idx 16) | 2 | 4096 B | 8192 B | fp32, compute↔writer double-buffer |
| `cb_accumulator` (idx 24) | 2 | 4096 B | 8192 B | fp32 intra-tile RMW; 2-page minimum for the front+back ping-pong (see `op_design.md` K6 #3) |
| **Total per core** |  |  | **24 576 B (24 KB)** | Well below Wormhole's ~1.5 MB L1. |

Doubling the input or output CB sizes would buy throughput room but is not bandwidth-bound on Wormhole today; the kernel does ~24 SFPU ops × 4 lgammas per tile, which dominates the per-tile cycle count.

---

## 6. Cross-Core Communication

**None.** The kernel performs no `noc_async_write_multicast`, no semaphore exchange, no ring topology. Every Tensix executes its slice independently; the only inter-core synchronisation is the global launch barrier and host-side completion.

This is the simplest and most scalable communication pattern for an elementwise op — adding cores scales linearly until per-tile work becomes negligible vs. per-launch overhead.

---

## 7. Bandwidth Bottleneck Assessment

- **Compute-bound**, not bandwidth-bound. Each output tile requires roughly 24 SFPU vector ops × 4 lgammas + 6 inner Lanczos terms + accumulator round-trips ≈ on the order of 100–150 SFPU operations per element. The DRAM round-trip is 2 × 4 KB = 8 KB per tile, well within Wormhole's per-core inbound DRAM budget when running 64 cores in parallel.
- **No replicated reads** — therefore no easy bandwidth wins from caching. The 4× tile re-use is already in L1 (`cb_input_tiles` 2-page buffer).
- **Future optimisation (not done at Phase 0)**: an issue-many-then-barrier-once reader pattern would reduce per-read latency exposure if the kernel ever becomes bandwidth-bound (e.g., at larger fp16 dtypes that complete compute faster). For Phase 0 it is unnecessary.

---

## Cross-References

- Reader: `kernels/multigammaln_lanczos_reader.cpp`
- Writer: `kernels/multigammaln_lanczos_writer.cpp`
- Compute: `kernels/multigammaln_lanczos_compute.cpp`
- Program descriptor: `multigammaln_lanczos_program_descriptor.py`
- Companion analysis: `numerical_stability.md` (for the L1-resident accumulator precision implications)
