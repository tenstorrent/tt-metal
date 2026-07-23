# Metal 2.0 Port Report — `data_movement/fill_pad`

## Outcome

**PORTED** — both `FillPadProgramFactory` (DRAM interleaved/sharded) and `FillPadL1ShardedProgramFactory` (HEIGHT/WIDTH/BLOCK L1-sharded) converted to `MetalV2FactoryConcept` (`create_program_artifacts`). The full confirmed test set `tests/ttnn/unit_tests/operations/data_movement/test_fill_pad.py` passes: **280 passed** on blackhole (all 5 test functions × dtype/layout/shape sweeps).

## Provenance

- **Recipe docs (this port):** `c1349c0d941 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `c1349c0d941 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` on both factories, as the audit specified. `program_factory_t` remains a two-alternative variant (`FillPadProgramFactory`, `FillPadL1ShardedProgramFactory`); both now expose `create_program_artifacts` (was `create_descriptor`). `select_program_factory` and the device-op class (`validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`, the `fill_pad(...)` launcher) are unchanged.

### Device-op-class edits
- Custom `compute_program_hash` deleted: none (op never had one).
- Pybind entry points removed: none (nanobind binds only the host function `fill_implicit_tile_padding`, never `create_descriptor`).

### Open items
- **Relaxation candidates:** none identified. Strict `TensorSpec` matching left in place (in-place op; the single `input` TensorParameter is bound by both reader and writer).
- **Dead kernel args preserved:** reader `N_slices`/`elem_size`/`fill_bits`, compute `W_tiles`/`H_tiles`/`elem_size`, sharded reader `elem_size` are declared as named CTAs but unread by the kernel (faithful to the legacy positional layout; the audit's Misc-anomalies note routes their removal to the ops team). Harmless (unused namespace-scope `constexpr` in the generated header).

## Handoff points
None. The port stayed entirely within the op directory — no out-of-op kernel edits, no kernel-lib/LLK gaps, no `sem::`/`tensor::` boundary-crossing call sites, no pybind surface removed, no capitulation.

## Successes
- **Case 2 raw-pointer binding (`get_bank_base_address`) worked exactly as the recipe describes.** The sharded reader/writer read local L1 via `UnicastEndpoint` arithmetic on a raw base; replacing `shard_l1_base = get_arg_val<uint32_t>(0)` with `TensorAccessor(tensor::src).get_bank_base_address()` and leaving the arithmetic untouched ([kernel-side whitelist rule 5](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md)) produced correct results across all L1-sharded configs. `fill_pad_sharded_reader.cpp:60-62`, `fill_pad_sharded_writer.cpp:54-56`.
- **Conditional-DFB-binding pattern caught a real per-node validity trap.** Following [Pattern: Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md) forced the recognition that the sharded writer had to be split by `has_bottom_pad_core` (the `BOT_MASK` producer must be per-node consistent with the compute consumer, which keys `has_bottom_pad` at compile time). Binding `BOT_MASK` on an unsplit writer would have shipped a producer-only DFB on interior/right-only nodes — a spec-validator failure the pattern's "the define and the binding share one condition" guidance anticipated. See `METAL2_PORT_PLAN.md` → "Sharded writer split — rationale."
- **The audit's Case 1/Case 2 split was exactly right.** DRAM factory Case 1 (`TensorAccessor(tensor::src/dst)`, 3rd page-size arg dropped) and sharded factory Case 2 both matched what the kernels needed; no surprises.

## Friction

### Gaps
- **`./build_metal.sh --build-tests` returned exit 0 despite a `-Werror` compile failure in the ported factory.** An unused-variable error (`is_float_type`, `-Werror,-Wunused-variable`) failed the ttnn compile, but the wrapper reported success and left the *previous* `.so` in place. The stale `.so` still ran the legacy `create_descriptor`, which emitted legacy positional `-DKERNEL_COMPILE_TIME_ARGS=...` against the newly-Metal-2.0 kernel sources — surfacing at JIT as a cascade of `'args'/'dfb'/'tensor' has not been declared` errors in *every* kernel. The misleading exit code sent the first diagnosis toward the kernels; the real fault was a silent host build failure. Recommendation for the recipe's "Running builds" section: always grep the build log for `error:` rather than trusting the exit code.

### Confusion
- Ported compute kernels **can** use `dfb::` / `tensor::` (the accumulation reference and quasar DFB compute kernels do). The initial `'dfb' has not been declared` errors *looked* like a compute-TRISC codegen limitation, but were purely the stale-`.so` artifact above — once the factory compiled, the generated bindings were injected for all kernels (DM and compute alike).

## Open items for downstream
- Dead / unused kernel args preserved verbatim (audit "Misc anomalies"): reader `elem_size`/`N_slices`/`fill_bits`; compute `elem_size` (and unused `W_tiles`/`H_tiles`); sharded reader `elem_size`. Left named-but-unused; removal is an ops-team cleanup, out of port scope.
- **Derived placement narrows the sharded op's footprint (behavior-equivalent):**
  - Mask DFBs (`right_mask`/`bot_mask`) are placed only on the right-edge / bottom-edge nodes that bind them, vs. the legacy allocation over `all_active_set`. Unused CBs on other cores were never touched, so output is identical.
  - **Interior block-sharded cores** (`has_right_pad==0 && has_bottom_pad==0`, i.e. no border tiles) get *no* kernels placed, vs. the legacy no-op kernels. This was also required to keep the sharded writer compilable — a writer KernelSpec with neither `HAS_RIGHT_PAD` nor `HAS_BOTTOM_PAD` would have an all-`#ifdef`'d-out body (unused `dfb::data_out`/base/tile_bytes → `-Werror`). Skipping zero-work cores is the natural Metal 2.0 shape (placement derived from bindings) and is output-identical. `fill_pad_program_factory.cpp` (sharded pass-1 and RTA loops).
- No cross-op kernel touches (all 5 sources op-owned; `fill_pad_compute.cpp` + `fill_pad_dataflow_common.hpp` are in-op shared, ported once).
