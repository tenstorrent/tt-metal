# Operation Requirements: tilize

## Definition
- **Formula**: `output[i] = input[i]` — identity of element VALUES; only the byte
  positions change (ROW_MAJOR → TILE layout). Optional value-preserving cast when
  `dtype=` narrows/widens the storage format (bf16↔fp32↔bf8b).
- **PyTorch Reference**: none — the oracle is identity. `to_torch(tilize(from_torch(x,
  ROW_MAJOR))) == x` (PCC for bf8b / value-preserving-cast tolerance when `dtype=`
  narrows).
- **Import Path**: `from ttnn.operations.tilize import tilize`
- **Function Signature**:
  ```python
  tilize(
      input_tensor: ttnn.Tensor,                       # ROW_MAJOR_LAYOUT, on device, last two dims % 32 == 0
      memory_config: ttnn.MemoryConfig | None = None,  # output mem config (default: input's)
      *,
      dtype: ttnn.DataType | None = None,              # output dtype (default: input's); value-preserving cast only
      use_multicore: bool = True,                      # distribute tile-rows across the grid
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses `Refinement \d+[a-z]?`)**: primary refinements are `Refinement N`; a partial's follow-up appends a lowercase letter (`Refinement 1b`), ordered immediately after its parent.

### [x] Phase 0 — Core Implementation (already delivered, verified this pass)

- **SUPPORTED dtype**: [bfloat16, float32]
- **SUPPORTED output_dtype**: [bfloat16, float32, bfloat8_b]
- **SUPPORTED use_multicore**: [False, True]
- **SUPPORTED shard_api**: [none]
- **SUPPORTED out_scheme**: [interleaved]
- **SUPPORTED buffer**: [dram_to_dram, dram_to_l1, l1_to_dram, l1_to_l1]
- **SUPPORTED rank**: [2, 3, 4]
- **Cores**: single + multi (tile-row split, no inter-core sync)
- **Compute config**: bf16 = default `Fast`; fp32 = `Lossless` + `UnpackToDestFp32` +
  `fp32_dest_acc_en` (bit-exact terminal fp32); wide-W CB bounded by a constant chunk.
- **Golden baseline**: 36 / 36 in-scope cells passing (verifier CLI); bf16/fp32 identity
  exact, bf16→bf8b PCC ≥ 0.99.

### [x] Refinement 1 — uint32 integer passthrough

**Goal**: add `ttnn.uint32` to `SUPPORTED["dtype"]` and `SUPPORTED["output_dtype"]` so the
integer-passthrough family works (only `uint32 → uint32` is valid — int↔float crosses are
pruned by `INVALID` in `feature_spec.py`). Moves the 6 pure-interleaved `uint32→uint32`
cells from `xfail_expected` to `supported_pass`. The tilize LLK reorders integer bytes
with no arithmetic and no cast, so the work is: derive the output CB `data_format` and
`tile_size` from the uint32 dtype, and ensure the compute path does **not** take the
fp32 branch (no `Fp32Mode::Lossless` / `UnpackToDestFp32` / `fp32_dest_acc_en` for
integers — those apply only to float precision).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: this is integer passthrough, **not** float precision configuration —
the skill's ComputeKernelConfig float knobs (`math_fidelity`, `fp32_dest_acc_en`,
`UnpackToDestFp32`) do **not** apply; the only levers are the CB `data_format`/`tile_size`
(already dtype-derived in `tilize_program_descriptor.py`) and keeping the integer path off
the fp32 branch in `tilize_compute.cpp` (`is_fp32_in` is already 0 for uint32, so verify
the default `Fast` tilize handles integer formats — the helper falls back to standard
tilize for non-Float32/Float16_b formats). Independent of Refinement 2 and can land in
either order; the 4 `uint32 + sharded` crossed cells need **both** refinements. Land this
first — it is interleaved-only and the smaller change.

**Done when**: `SUPPORTED["dtype"]` and `SUPPORTED["output_dtype"]` include `uint32`; the
6 pure `uint32→uint32` interleaved golden cells pass (`comp_equal` exact); no regression
in the bf16/fp32/bf8b cells.

### [ ] Refinement 2 — Sharded I/O (legacy_2d HEIGHT/WIDTH/BLOCK + nd)

**Goal**: add `"legacy_2d"` and `"nd"` to `SUPPORTED["shard_api"]`, and
`HEIGHT_SHARDED`, `WIDTH_SHARDED`, `BLOCK_SHARDED`, `"nd"` to `SUPPORTED["out_scheme"]`.
Support RM-sharded L1 input → TILE-sharded L1 output on the same grid/shard spec, via
in-kernel sharded data access: the output CB is aliased directly onto the local L1 shard
buffer (zero-copy — `tilize_block` packs tiles straight into the shard, **no DRAM
writes**), and the reader consumes the local RM shard. Moves the ~30 sharded golden cells
(12 nd + 18 legacy_2d HEIGHT/WIDTH/BLOCK) from `xfail_expected` to `supported_pass`.

**Implementation skill**: /memory-layouts

**Verifier notes**: no dedicated sharding skill exists; `/memory-layouts` is the closest —
its native-support policy covers the in-kernel "sharded reader for sharded I/O" data-access
change, and `/interleaved-parallel` explicitly defers sharded tensors to `/memory-layouts`
territory. Two op-specific fixes this refinement must also make (the skill can't see them):
(1) `validate()._shard_api_of` currently returns `"legacy_2d"` for *any* sharded config,
including nd — fix it to return `"nd"` for the NdShardSpec path so the validated axis
matches the tagger; (2) add the design's `EXCLUSIONS` for single-core + sharded
(`{"use_multicore": False, "shard_api": "legacy_2d"}` and `... "nd"`) — sharding is
inherently multi-core. This carries the design's zero-copy write perf lever (R3): when it
lands, re-target the roofline — the write side becomes an L1 loopback, not a DRAM transfer
(`/perf-roofline-dm`, `/perf-measure` for the perf gate). Land after Refinement 1 (memory
config stress goes last; the 4 `uint32 + sharded` cells depend on both). If the full
legacy + nd surface is more than one focused pass, ship legacy_2d first as `[~]` and file
the nd remainder as `Refinement 2b` — do **not** partial-tick by wrapping the op in
`ttnn.to_memory_config` / host round-trips (the prompt forbids extra DRAM passes; the
whole point is the zero-copy shard write).

**Done when**: `SUPPORTED["shard_api"]` includes `legacy_2d` + `nd` and
`SUPPORTED["out_scheme"]` includes the three legacy schemes + `nd`; the sharded golden
cells pass identity per scheme; `test_golden_main_tests.py` sharded tests
(`test_tilize_row_major_to_width_sharded`, `test_tilize_nd_sharded`) pass; no DRAM write
on the sharded output path (tt-npe shows zero output-side DRAM traffic).
