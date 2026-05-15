# Data Transfer Profile: rms_norm

Phase 0 (single-core) data movement analysis. Numbers are per-row-chunk
(32 rows × Wt tiles wide). Repeat ×num_chunks for total per launch.

## DRAM read / write summary

| Stream | Granularity | Per chunk | Per launch (NC×Ht chunks) | Notes |
|---|---|---|---|---|
| input read (TILE) | tile | `Wt` tiles | `NC × Ht × Wt` tiles | Single pass — `cb_input_tiles` holds Wt across pass 1 (NoWaitNoPop) and pass 2 (WaitAndPop). |
| input read (RM) | stick | up to 32 sticks of `W * elem_size` bytes | `NC × H` sticks | Each chunk reads 32 sticks (final chunk may be partial via `rows_this_chunk`). |
| gamma read | stick | 1 stick of `padded_W * gamma_elem_size` | `NC × Ht` sticks | **Re-read every chunk**. Optimization: hoist outside the chunk loop (see risk 12, op_design.md). |
| output write (TILE) | tile | `Wt` tiles | `NC × Ht × Wt` tiles | Sequential `noc_async_write_tile` + per-tile `cb_wait_front/cb_pop_front`. |
| output write (RM) | stick | 32 sticks of `W * elem_size` (last chunk partial) | `NC × H` sticks | `write_sticks_after_untilize` helper handles partial-H. |

## NoC balance

| NoC | Owner | Traffic | Pressure |
|---|---|---|---|
| NoC 0 | reader (NCRISC) | input read + gamma read + scaler-tile init (once) | input dominates; gamma is a single stick per chunk so negligible on bandwidth but a fixed setup-latency hit. |
| NoC 1 | writer (BRISC) | output write | symmetric to input traffic. |

Reads and writes use opposite NoCs by default; no cross-NoC contention.
No Tensix↔Tensix traffic at v1 (single-core).

## L1 budget per core

CB allocation reservoir per kernel launch — all CBs live simultaneously
inside the row-chunk inner loop (CB sizes per `rms_norm_program_descriptor.py:124-285`):

| CB | Pages | Page size | Bytes (fp32 / bf16) | Notes |
|---|---|---|---|---|
| `cb_input_raw_rm` (RM only) | `Wt` | `tile_size(input_dtype)` | 4 KB / 2 KB × Wt | TILE granularity reader → tile-sized pages. |
| `cb_input_tiles` | `Wt` | `tile_size(input_dtype)` | 4 KB / 2 KB × Wt | Stage A NoWaitNoPop + Stage D WaitAndPop. |
| `cb_gamma_rm` (gamma only) | 1 | `padded_gamma_row_bytes` | `W * gamma_elem_size` | Single stick. |
| `cb_gamma_tiled` (gamma only) | `Wt` | `tile_size(gamma_dtype)` | 4 KB / 2 KB × Wt | Compute tilizes (asymmetric) 1 row → Wt tiles. |
| `cb_scaler` | 1 or 2 | `tile_size(bfloat16)` | 2 KB × (1 or 2) | Always bf16. Held persistently for all reductions. |
| `cb_output_tiles` | `Wt` | `tile_size(output_dtype)` | 4 KB / 2 KB × Wt | TILE writer pops Wt; RM path untilize pops Wt. |
| `cb_output_rm` (RM only) | `Wt` | `tile_size(output_dtype)` | 4 KB / 2 KB × Wt | Output of in-kernel untilize. |
| `cb_x_sq` | `Wt` | `tile_size(input_dtype)` | 4 KB / 2 KB × Wt | Pass-1 stage A → stage B reduce. |
| `cb_mean_sq` | 2 | `tile_size(input_dtype)` | 4 KB / 2 KB × 2 | 1 tile in active use + 1 spare for `transform_in_place`. |
| `cb_x_norm` (gamma only) | `Wt` | `tile_size(input_dtype)` | 4 KB / 2 KB × Wt | Pass-2 stage D → stage E. |

Worst-case totals (gamma + RM I/O):
- bf16 + Wt=32 (W=1024): ≈ 7 × 32 × 2 KB + 4 KB + 2 KB ≈ 454 KB.
- fp32 + Wt=32 (W=1024): ≈ 7 × 32 × 4 KB + 8 KB + 2 KB ≈ 906 KB.
- fp32 + Wt=128 (W=4096): ≈ 7 × 128 × 4 KB + 8 KB + 2 KB ≈ 3.6 MB > 1.5 MB L1 cap.

This is why `SUPPORTED["shape_size"]=["small"]` for v1: Wt > 32 blows the
budget. The unblocking refinement is W-blocking (op_design.md risk 13)
via `accumulate_reduce_block` from `streaming_reduce_helpers.hpp:53-61`.

## Inefficiencies & optimization candidates

1. **Re-tilizing gamma every row chunk** — gamma is read from DRAM and
   tilized once per chunk despite being a single 1×W stick. For LLM-shape
   workloads (H large, W large), this is `NC*Ht` redundant DRAM reads
   plus `NC*Ht` redundant tilizes. v1 accepts the cost; future
   refinement: tilize gamma once at kernel start, keep `cb_gamma_tiled`
   persistent (op_design.md risk 12).

2. **Re-tilizing input twice per row chunk on the RM path** — `cb_input_tiles`
   has Wt pages but the reader / compute logic was originally planned
   for a two-pass tilize pattern. The current implementation pushes input
   ONCE per chunk, then `cb_wait_front(cb_input_tiles, Wt)` + NoWaitNoPop
   stage A + WaitAndPop stage D — so the tilize happens once. ✓ Good.
   No further work needed here.

3. **Scaler CB is bf16 regardless of input dtype** — reduce LLK requires
   bf16 scaler tiles. This adds 2 KB to every launch and forces an
   fp32→bf16 rounding event on the scaler value. For fp32 inputs, that's
   a known precision cost. (See `numerical_stability.md` "Divide-then-sum"
   for a way to skip the scaler entirely.)

4. **Output write loop is per-tile** (TILE output path). Each tile takes a
   `cb_wait_front(1)` + `noc_async_write_tile` + barrier + `cb_pop_front(1)`.
   For Wt small (≤ 4) this is fine; for wider future shapes a
   `write_tiles_for_*` aggregator would batch the wait/pop and amortize
   the barrier cost. Out of scope at v1.

5. **Single-core only** — `device.compute_with_storage_grid_size()` is not
   consulted; one core processes everything. For NC large or H large
   this leaves the rest of the chip idle. Distribution across cores is
   embarrassingly parallel (each row chunk is independent). Bundle this
   into another refinement; not its own entry per the verifier's
   guidance (no inter-core communication needed).

6. **RM input + non-tile-aligned H + NC>1** required a Phase-0 descriptor
   fix during this verification pass: `num_chunks` is now computed as
   `ceil(NC*H/32)` (uniformly for both TILE and RM paths) rather than
   `NC*Ht`, eliminating a class of hangs where the reader pushed data for
   `NC*H` rows but compute iterated over `NC*Ht*32 > NC*H` rows. Mentioned
   here for completeness; it's not a transfer-efficiency concern but the
   fix lived in the descriptor's chunk-count plumbing.

## Verifier takeaway

For refinement queueing:
- **Wide-W path (W > 1024)** is the only data-transfer-relevant
  refinement: introducing `accumulate_reduce_block` W-blocking
  proportionally shrinks every Wt-sized CB and unlocks
  `SUPPORTED["shape_size"]=["small", "large"]`.
- **Multi-core distribution** is L1-pressure-neutral (each core owns
  fewer chunks but each chunk still costs the same L1). Bundle.
- **Gamma hoist** (cache `cb_gamma_tiled` across the chunk loop) is a
  pure perf refinement — doesn't unlock new SUPPORTED cells, so it
  belongs in `verification_report.md` as a future note, not in
  `op_requirements.md`.
