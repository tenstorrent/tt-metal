# Data Transfer Analysis: layer_norm_rm

## Summary

Per-tile-row, single-core implementation that **reads the input tensor three times**
from DRAM (one full sweep per pass: mean → variance → normalize-and-drain).
Gamma/beta are read once. Output is written once.

| Metric | Value |
|--------|-------|
| Core grid | 1 (single core at `(0, 0)`) |
| DRAM reads of input | **3 × (B·C·H · W · sizeof(dtype))** bytes (Pass 1 / Pass 2 / Pass 3) |
| DRAM reads of gamma/beta | `1 × (W · sizeof(dtype))` per tensor (single stick), pre-Pass-1 |
| DRAM writes of output | `1 × (B·C·H · W · sizeof(dtype))` bytes |
| Per-pass read granularity (RM input) | 1 stick per `noc_async_read` via `read_sticks_for_tilize<ROW>` |
| Per-pass read granularity (TILE input) | 1 tile per `noc_async_read_tile` |
| Output write granularity (RM out) | 1 stick per `noc_async_write` via `write_sticks_after_untilize` |
| Output write granularity (TILE out) | 1 tile per `noc_async_write_tile` |
| Inter-core communication | **None** (single-core, no semaphores / mcast / rings) |

The kernel's bandwidth profile is **read-dominated**: total DRAM read = 3× input + 2× small affine; total DRAM write = 1× output.

## Per-pass DRAM transactions

For input shape `(..., H, W)` with `total_tile_rows = ceil(prod(shape[:-1]) / 32)`
and `Wt = ceil(W / 32)`:

| Tensor | Direction | Bytes | Read/write count |
|--------|-----------|-------|------------------|
| input (RM) | DRAM → cb_input_sticks | `(prod(shape[:-1])) × W × sizeof(dtype)` per pass × 3 passes | `total_tile_rows × 32 × 3` sticks of size `input_row_bytes` |
| input (TILE) | DRAM → cb_input_tiles | `total_tile_rows × Wt × tile_size` per pass × 3 passes | `total_tile_rows × Wt × 3` tiles |
| gamma (RM) | DRAM → cb_gamma_sticks | `W × sizeof(dtype)` once | 1 stick |
| beta (RM) | DRAM → cb_beta_sticks | `W × sizeof(dtype)` once | 1 stick |
| scaler | (compute-side fill, no DRAM I/O) | — | — |
| output (RM) | cb_output → DRAM | `prod(shape[:-1]) × W × sizeof(dtype)` once | `total_tile_rows × 32` sticks |
| output (TILE) | cb_output → DRAM | `total_tile_rows × Wt × tile_size` once | `total_tile_rows × Wt` tiles |

**Effective bandwidth cost**: `3 × input_bytes + 1 × output_bytes ≈ 4 × tensor_bytes` (gamma/beta are negligible).

## NoC balance

- **NoC0 (reader, NCRISC)** handles every DRAM → L1 transfer. No NoC1 reads.
- **NoC1 (writer, BRISC)** handles every L1 → DRAM transfer. No NoC0 writes.
- Reader and writer never cross-issue. Both NoCs are utilized but **NoC0 carries ~4× more traffic** because the read-three-times sweep dominates.

## L1 footprint (per-core CB budget)

CB sizes scale with `Wt` (sized to hold one full row's worth of tiles per buffer slot).
The op uses **single-core** so the budget is the full 1.5 MB L1 minus runtime overhead.

| CB | Pages | Page size | Bytes (bf16) | Bytes (fp32) | Notes |
|----|-------|-----------|--------------|--------------|-------|
| `cb_input_tiles` | `2·Wt` (TILE in) or `Wt` (RM in) | `tile_size` | `2·Wt · 2 KiB` | `2·Wt · 4 KiB` | Holds one row's worth of tiles, double-buffered for TILE input |
| `cb_input_sticks` (RM in only) | `32` | `padded_row_bytes` | `32 · padded_row_bytes` | `32 · padded_row_bytes` | One tile-row of sticks |
| `cb_scaler` | 1 or 2 | `tile_size(bfloat16) = 2 KiB` | 2–4 KiB | 2–4 KiB | Always bfloat16 |
| `cb_output` | `2·Wt` | `tile_size` | `2·Wt · 2 KiB` | `2·Wt · 4 KiB` | Double-buffered drain |
| `cb_gamma_tiles` (if present) | `Wt` | `tile_size` | `Wt · 2 KiB` | `Wt · 4 KiB` | Held WaitUpfrontNoPop for all rows |
| `cb_beta_tiles` (if present) | `Wt` | `tile_size` | `Wt · 2 KiB` | `Wt · 4 KiB` | Same |
| `cb_gamma_sticks` / `cb_beta_sticks` | `32` each | `padded_row_bytes` | small | small | Used only during the pre-Pass-1 tilize |
| `cb_mean` | 2 | `tile_size` | 4 KiB | 8 KiB | Holds 1 mean tile (per tile-row), double-buffered |
| `cb_inv_std` | 2 | `tile_size` | 4 KiB | 8 KiB | Same |
| `cb_centered` | `Wt` | `tile_size` | `Wt · 2 KiB` | `Wt · 4 KiB` | Holds one row of normalized output before drain |

**Hot-path L1 (TILE input + gamma/beta):**
`cb_input_tiles + cb_output + cb_centered + cb_gamma_tiles + cb_beta_tiles + cb_mean + cb_inv_std`
`= (2·Wt + 2·Wt + Wt + Wt + Wt) · tile_size + 4·tile_size`
`= (7·Wt + 4) · tile_size` (RM input adds the sticks CB).

This caps `Wt` before L1 overflows:

| dtype | L1 budget | Max `Wt` (approx) | Max `W` |
|-------|-----------|-------------------|---------|
| bf16  | 1.5 MB | ~96 | ~3072 |
| fp32  | 1.5 MB | ~48 | ~1536 |

Observed at `W = 4096`:
- `bf16 + TILE + gamma_only`: total CB size ~1.9 MB → **fails** with "Statically allocated circular buffers grow to 1956352 B which is beyond max L1 size of 1572864 B".
- `fp32 + TILE + gamma_only`: even worse.

This is the **dominant scaling bottleneck** for Phase 0. Refinement candidate: streaming reduce over W (chunk `cb_input_tiles` into `BLOCK_SIZE`-tile blocks) — the `accumulate_reduce` / `accumulate_reduce_block` helpers in `streaming_reduce_helpers.hpp` are designed for exactly this.

## Latency / pipelining considerations

- Reader and compute are pipelined via the double-buffered `cb_input_tiles` (TILE in). Single-buffer for RM in is acceptable because the in-kernel tilize step amortizes the read cost.
- All three passes serialize on the reader: a tile-row's Pass-2 can't start until its Pass-1 mean is in DEST. Reading the input three times is the fundamental three-pass cost — no overlap is possible across passes.
- Hidden behind DRAM read latency: tile manipulation (tilize/untilize), reduce LLKs, eltwise broadcasts. These are compute-bound at small `Wt` and memory-bound at large `Wt`.

## Performance characterization (Phase 0)

| Shape | dtype | Status |
|-------|-------|--------|
| `(B≤32, H≤512, W≤512)` | bf16 / fp32 | Works; bandwidth-light, single-core latency-bound |
| `(B≤4, H, W≤3000)` | bf16 | Works; CB pressure rising but OK |
| `W ≥ 4096` | bf16 / fp32 | **OOM** (CB allocation) |

## Recommendations (mapped to refinements)

1. **Streaming reduce for wide W.** Replace one-shot `reduce<>` with
   `accumulate_reduce<>` / `accumulate_reduce_block<>` from
   `streaming_reduce_helpers.hpp`. Chunks `cb_input_tiles` into
   `BLOCK_SIZE ≤ 8` blocks; resizes the input CB to
   `2 · BLOCK_SIZE · tile_size` (constant in `Wt`). Removes the L1 cap on
   `Wt`. Requires the same change in Pass 2 (variance reduce, on `cb_centered`).
   ➜ unlocks the 86 `OOM`-categorized cells in `verifier_report.json`.
2. **Multi-core distribution.** Embarrassingly parallel over `total_tile_rows`
   (each core processes a disjoint slice). No inter-core dependencies; fold
   into the streaming-reduce refinement since the per-core kernel barely changes.
3. **Avoid the read-three-times cost when L1 has headroom.** When
   `Wt × tile_size < L1_budget / 3`, an alternative is a **single-pass** layout
   that caches the input in L1 and re-reads from L1 for Passes 2-3.
   Saves 2× DRAM bandwidth in the small-W regime. Lower priority — the
   single-core latency at small `Wt` is dominated by compute, not DRAM.
