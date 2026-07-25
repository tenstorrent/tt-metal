# triangle_solve (experimental)

Per-tile forward-substitution solve of `L X = RHS` on the SFPU, for a single 32×32 tile.
Standalone op used to develop and validate the triangle-solve LLK that the gated-delta
prefill op (`gated_delta_prefill_query`) will use.

## Interface
`ttnn.experimental.triangle_solve(l_neg, rhs) -> x`

- `l_neg`: `[1,1,32,32]` TILE bf16 — the unit lower-triangular matrix `L`, supplied **negated**
  on the strict-lower part (diagonal is an implicit 1, upper triangle ignored). Pre-negating lets
  the per-column update be an accumulate: `X[i] = RHS[i] + Σ_{j<i} L_neg[i][j]·X[j]`, which equals
  the forward-substitution subtraction `X[i] = RHS[i] − Σ_{j<i} L[i][j]·X[j]`.
- `rhs`: `[1,1,32,32]` TILE bf16 — right-hand side.
- returns `x`: `[1,1,32,32]` TILE bf16 — solution `X`.

## Kernel / LLK
- Compute API: `tt_metal/hw/inc/api/compute/triangle_solve.h` — `triangle_solve_tile(cb_l, l_tile_idx, idst_in, idst_out)`.
- SFPU LLK: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_triangle_solve.h`.
- `L` is read element-by-element straight from L1 (not a DST reg); only `RHS` occupies a DST input.
- DEST addressing uses the face/parity `{0,2,16,18}` block pattern (unit-stride would alias). The
  solved rows are stashed row-oriented in `dst_out`, then an in-place `SFPTRANSP` fix-up writes the
  standard tile layout. Blackhole-only (relies on the HW scoreboard for the `SFPMAD` chain).

## Validation
Test: `tests/ttnn/unit_tests/operations/experimental/test_triangle_solve.py`

Compares against `torch.linalg.solve_triangular(L, RHS, upper=False, unitriangular=True)` on the
**non-negated** `L`. On Blackhole (P300, single-chip):

- **PCC: 0.9999963**
- **max abs error: 0.0216** (bf16 rounding; grows slightly down the rows as substitution compounds)

### Running on this box (P300 reports 1 chip → CUSTOM cluster)
```
export TT_METAL_HOME=/localdev/vsuresh/tt-metal
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_p300_mesh_graph_descriptor.textproto
python -m pytest tests/ttnn/unit_tests/operations/experimental/test_triangle_solve.py -x -s
```
