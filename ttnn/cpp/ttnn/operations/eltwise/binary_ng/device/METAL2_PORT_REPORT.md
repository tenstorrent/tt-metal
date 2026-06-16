# binary_ng — Metal 2.0 port report

## STATUS: PARTIAL (coverage widened; remaining cases deliberately on legacy)

`BinaryNgDeviceOperation::ProgramSpecFactory::create_program_spec` (in
`binary_ng_program_factory_m2.cpp`) is the single Metal 2.0 factory for the op. It now models a
widened set of interleaved cases. `select_program_factory` routes ONLY the verified-ported cases to
the spec factory (via the shared predicate `binary_ng_m2_routes_to_spec`); every other case stays on
the legacy `ProgramFactory::create_descriptor`. The custom `compute_program_hash` is **kept**
(unchanged) — it is shared by both factories.

This pass extends the previously-routed 3 cases (NONE × tile × interleaved × {tensor-b FPU,
tensor-b SFPU, scalar-b FPU}) with:
1. **ROW_MAJOR × NONE-broadcast × tensor-b × FPU** — own reader/writer RM forks; reuses the FPU
   no-bcast compute (RM forces freq=1/counter=0, so the no-bcast compute is correct).
2. **scalar-b × SFPU (tile)** — adds the SFPU scalar compute fork.
3. **Simple broadcast on tile (SCALAR / ROW / COL) × tensor-b × FPU** — adds bcast readers + the
   LLK-bcast compute kernels, extra bcast DFBs (legacy c_5 / c_6), compute freq/counter RTAs, and
   the `BCAST_INPUT` / `SRC_BCAST(_B)` defines. Routed **only on the LLK-bcast subset** (see below).

## Routed-cases matrix (broadcast × layout × compute × operand → spec vs legacy)

Common gate for any spec routing: interleaved (not sharded), no activations, no typecast
(`a.dtype() == output dtype`), plain op ∈ {ADD, SUB, MUL}, not where-op, not quant-op, operand
present (tensor-b or scalar value).

| Broadcast | Layout | Compute | Operand | Routed to |
|-----------|--------|---------|---------|-----------|
| NONE | TILE | FPU | tensor-b | **SPEC** |
| NONE | TILE | SFPU | tensor-b | **SPEC** |
| NONE | TILE | FPU | scalar-b | **SPEC** |
| NONE | TILE | SFPU | scalar-b | **SPEC** (new) |
| NONE | ROW_MAJOR | FPU | tensor-b | **SPEC** (new) |
| NONE | ROW_MAJOR | SFPU | tensor-b | legacy |
| NONE | ROW_MAJOR | any | scalar-b (RM scalar-op) | legacy |
| SCALAR_A / SCALAR_B | TILE | FPU | tensor-b | **SPEC iff `use_llk_bcast`** (new); else legacy |
| ROW_A / ROW_B | TILE | FPU | tensor-b | **SPEC iff `use_llk_bcast`** (new); else legacy |
| COL_A / COL_B | TILE | FPU | tensor-b | **SPEC iff `use_llk_bcast`** (new); else legacy |
| SCALAR / ROW / COL | TILE | SFPU | tensor-b | legacy |
| SCALAR / ROW / COL | TILE | any | scalar-b | legacy |
| ROW_A_COL_B / ROW_B_COL_A (mixed) | any | any | any | legacy |
| any broadcast | ROW_MAJOR | any | any | legacy |
| any (sharded) | any | any | any | legacy |
| activations / typecast / where / quant / non-plain op | any | any | any | legacy |

`use_llk_bcast` is the legacy factory's decision (reproduced verbatim in
`binary_ng_m2_use_llk_bcast`): true for {bf16, bf8, bf4} all-matching on any arch, and for
{fp32, int32, uint32, uint16} all-matching only on Wormhole_B0; with the legacy fallbacks subtracted
(BH col bf16→fp32 hang; BH MOVB2D fp32 for SCALAR/ROW; UInt16 scalar relational — inert for the
routed plain-arithmetic subset). When `use_llk_bcast` is false, the software-fallback (`BCAST_LLK=0`)
compute kernels would run — those are **not ported**, so those tuples stay on legacy. This honors the
rule "never route a case whose kernels you didn't convert."

## Kernels (ported vs forked)

All kernels live inside the op directory and are referenced by path. None is shared by another op
(`grep -rl` across `ttnn/cpp/ttnn/operations` confirmed), so each was **forked** to a `*_m2.cpp`
copy (the legacy factory still binds the originals by `KernelName` path). Newly forked this pass:

| New `*_m2` kernel | Forked from | Used by |
|---|---|---|
| `kernels/compute/eltwise_binary_sfpu_scalar_m2.cpp` | `eltwise_binary_sfpu_scalar.cpp` | scalar-b × SFPU |
| `kernels_ng/dataflow/reader_interleaved_rm_no_bcast_m2.cpp` | `reader_interleaved_rm_no_bcast.cpp` | RM no-bcast |
| `kernels_ng/dataflow/writer_interleaved_rm_no_bcast_m2.cpp` | `writer_interleaved_rm_no_bcast.cpp` | RM no-bcast |
| `kernels_ng/dataflow/reader_interleaved_scalar_bcast_m2.cpp` | `reader_interleaved_scalar_bcast.cpp` | SCALAR bcast |
| `kernels_ng/dataflow/reader_interleaved_row_bcast_m2.cpp` | `reader_interleaved_row_bcast.cpp` | ROW bcast |
| `kernels_ng/dataflow/reader_interleaved_col_bcast_m2.cpp` | `reader_interleaved_col_bcast.cpp` | COL bcast |
| `kernels_ng/compute/eltwise_binary_scalar_bcast_m2.cpp` | `eltwise_binary_scalar_bcast.cpp` | SCALAR bcast (LLK) |
| `kernels_ng/compute/eltwise_binary_row_bcast_m2.cpp` | `eltwise_binary_row_bcast.cpp` | ROW bcast (LLK) |
| `kernels_ng/compute/eltwise_binary_col_bcast_m2.cpp` | `eltwise_binary_col_bcast.cpp` | COL bcast (LLK) |

Pre-existing forks (prior pass, reused): `reader_interleaved_no_bcast_m2.cpp` (both kernels/ and
kernels_ng/), `writer_interleaved_no_bcast_m2.cpp`, `writer_interleaved_scalar_m2.cpp`,
`eltwise_binary_no_bcast_m2.cpp`, `eltwise_binary_sfpu_no_bcast_m2.cpp`, `eltwise_binary_scalar_m2.cpp`.

Per-kernel conversions were mechanical (CB id → `dfb::`, `TensorAccessorArgs`/addr → `ta::`,
positional `get_arg_val`/`get_compile_time_arg_val` → `get_arg(args::)`); logic, `#ifdef`s, loop
bounds, numeric paths and comments preserved. The `HAS_ACTIVATIONS(LHS/RHS) ? c_3/c_4 : …` ternaries
were rewritten as `#if/#else` preprocessor gates so the conditionally-bound `dfb::post_lhs/post_rhs`
tokens never enter name lookup when their activation is absent (Conditional-DFB-binding pattern).

## Spec-construction notes

- **DFB ← CB**: c_0=`src`, c_1=`src_b`, c_2=`dst`. The LLK-bcast tile path adds the legacy
  intermediate c_5 (bcast operand = A → `llk_post_a`) or c_6 (bcast operand = B → `llk_post_b`),
  bound on the compute kernel as a **self-loop** (PRODUCER+CONSUMER, shared accessor name) — a genuine
  self-loop (the compute packs into it then consumes it), not a fake-CB workaround.
- **Conditional activation DFBs** (c_3/c_4 = `post_lhs`/`post_rhs`) are NOT bound for the routed
  cases (no activations route), and the kernels `#ifdef`-gate their tokens accordingly.
- **RM page-size override removed**: the legacy RM reader/writer passed
  `TensorAccessor(args, addr, page_size_override)` "to override TensorAccessorArgs::AlignedPageSize
  which may be stale on program cache hits." The Metal 2.0 framework rebuilds the interleaved
  TensorAccessor CTA payload (incl. `aligned_page_size = align(compute_page_size_bytes, alignment)`)
  from the live `TensorSpec` on every cache miss (`program_spec.cpp` `ResolveTensorParameterStaticCTAs`),
  so the staleness the override guarded against does not exist on the m2 path. The override (and the
  `page_size_*` RTAs feeding it) were dropped; the `alignment_*` RTAs were KEPT (still used for
  `align(current_chunk_bytes, alignment)` read/write-length rounding). This is the only behavioral
  reasoning beyond pure mechanical translation; flagged here for reviewer confirmation.
- **Work split**: tile path parallelizes across output tiles (`physical_volume/tile_hw`); RM path
  across row blocks (`total_row_blocks`), reproducing the legacy RM geometry
  (`num_rows_per_tile`, `tiles_per_row_width`, reader/writer stride bytes). Multiplicity preserved:
  one compute `KernelSpec` + `WorkUnitSpec` per core group.
- **Compute RTAs**: no-bcast/scalar/RM read `num_tiles` only. The simple-broadcast computes read
  `num_tiles, tile_freq, tile_start` (`calculate_compute_kernel_args`); the legacy 4th compute arg
  (`compute_scalar_value` = quant zero-point) is dead for plain ADD/SUB/MUL and dropped (a
  superfluous named arg is an m2 validation error).
- **dynamic_tensor_shape = true** on all tensor parameters mirrors the legacy
  `RuntimeTensorShape` config; for interleaved tensors it is a pure host-side validation loosening.

## Open items / findings (routed to owners, not changed here)

- The software-fallback (`BCAST_LLK=0`) simple-broadcast compute kernels (`eltwise_binary.cpp`,
  `eltwise_binary_no_bcast.cpp` software-bcast use) are not ported; porting them would let the
  remaining SCALAR/ROW/COL tuples (and Blackhole fp32 / non-WH non-bf16) route to spec.
- SFPU simple-broadcast, scalar-b simple-broadcast, mixed ROW+COL broadcast, RM broadcast, and RM
  scalar-op remain on legacy (their kernels were not converted this pass).
- Sharded paths remain on legacy (the m2 factory models interleaved only).
- `binary_ng_m2_use_llk_bcast` duplicates the legacy `is_llk_bcast` gate (kept in sync by comment).
  A future consolidation could hoist the single source of truth into `binary_ng_utils`.

## Not built / not committed
Per instructions: no build, no measurement, no commit. The patch is at
`/tmp/port_binary_ng_more2.patch`.
