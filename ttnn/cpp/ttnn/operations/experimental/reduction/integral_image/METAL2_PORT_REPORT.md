# Metal 2.0 Port Report — integral_image (`experimental::reduction::integral_image`)

## Outcome

**PORTED** — the single `IntImgDeviceOperation` factory (interleaved, fixed 2×4 grid) is converted from the
direct-`create_descriptor` (`ProgramDescriptorFactoryConcept`) form to `MetalV2FactoryConcept`
(`create_program_artifacts` returning a `ProgramSpec` + `ProgramRunArgs`). All three kernels (reader / compute /
writer) and the two in-directory shared headers converted together. No other factory remains — the op is fully
ported. **Build/test verification is the orchestrator's** (I did not build or run tests).

## Provenance

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept`, as the audit chose. The op was `HasDirectDescriptor` (a bare `create_descriptor` on the
device-op struct, no `program_factory_t`). The Metal 2.0 adapter's `resolve_program_factory` fallback for a
`program_factory_t`-less op is `DirectDescriptorFactory`, which wraps `create_descriptor` — **not**
`create_program_artifacts` — so a bare `create_program_artifacts` on the struct would not have been dispatched.
The port therefore introduces `struct ProgramFactory { static ProgramArtifacts create_program_artifacts(...); }`
and `using program_factory_t = std::variant<ProgramFactory>;` (single alternative → the adapter auto-selects it,
no `select_program_factory` needed) and deletes `create_descriptor`.

### Device-op-class edits
- Custom `compute_program_hash` deleted: none (op never had one).
- Pybind entry points removed: none. `intimg_nanobind.cpp` binds only the free function `ttnn.experimental.intimg`;
  it never bound `create_descriptor`, so no pybind surface changed and no downstream Python caller is affected.

### Open items
- Tensor-parameter relaxation: none applied; the op validates a fixed 4D/tile/bf16-or-fp32 input, so strict
  `TensorSpec` matching is correct. No relaxation candidate noticed.

## Handoff points

None. The port stayed entirely within the op directory; no `sem::`/`tensor::` boundary crossing, no kernel-lib or
LLK gap, no cross-op/shared kernel (the op owns all three kernels), no framework gap.

## Successes

- **DFB endpoint census (recipe §CB endpoints / plan re-derivation).** Re-deriving the 9 CB dispositions from the
  kernel-touch census reproduced the brief exactly: 4× legal 1P+1C (START, INPUT, OUTPUT, AXIS_3_BUFFER) and 5×
  self-loop (ACC, CUMSUM_STAGE_0/1/2, AXIS_2_BUFFER). The self-loop pattern (bind compute PRODUCER **and** CONSUMER)
  maps cleanly; the writer's readback of output-tensor memory into AXIS_3_BUFFER is a `TensorAccessor` read (writer =
  PRODUCER of AXIS_3_BUFFER), not a hidden co-filler, so no multi-binding flag was needed — the "Watch for" note held.
- **`dfb::name → uint32_t` decoupling shim (whitelist rule 2).** The kernels pass CB ids through `uint32_t cb`
  helper parameters (`send_block`, `write_to_dram`, `cumsum_cube_axis_2`, the RAII guards). Passing `dfb::<name>`
  directly at those call sites (implicit `DFBAccessor → uint32_t`) let the helper signatures stay untouched — a
  genuinely minimal diff, exactly as rule 2 promises.
- **hw_config Style B (recipe §Compute kernels).** The legacy op sets a Metal `ComputeConfigDescriptor` with literal
  values (no TTNN `ComputeKernelConfig`), so building `ComputeGen1Config` directly and copying the resolved values
  (HiFi4, `math_approx_mode=false → Precision::Precise`, `fp32_dest_acc_en → enable_32_bit_dest`) was the right path;
  routing through the TTNN helper would have flipped the unset fields to the high-performance defaults.
- **`unpack_modes` required-entry rule (migration guide DataflowBufferSpec validator note).** The guide's warning
  fired correctly in reasoning: for fp32 input, `enable_32_bit_dest=true` and every consumed Float32 DFB needs an
  explicit entry. Legacy left the mode default (`UnpackToSrc`), so the 8 consumed DFBs get `UnpackMode::UnpackToSrc`;
  OUTPUT (producer-only) is excluded. bf16 input needs no entries.

## Friction

### Gaps
- **Reader `get_dataformat` metadata move — rule 7 has no working spelling here (docs gap).** The brief and
  kernel-side whitelist rule 7 direct `get_dataformat(cb_id) → dfb::input.get_dataformat()` (move onto the DFB
  object). That literal spelling does **not** compile: `get_dataformat()` is a member of `DataflowBuffer`, not of
  the `dfb::input` token (which is a `DFBAccessor` exposing only `operator uint32_t`). The object-getter route also
  fails structurally in *this* site: the reader uses the format in a `constexpr` template argument
  (`std_type_t<get_dataformat(...)>`, `intimg_reader.cpp:51`) to pick a C++ POD element type, but
  `DataflowBuffer`'s constructor is **not** `constexpr`, so `DataflowBuffer(dfb::input).get_dataformat()` cannot be
  used in a constant expression. Resolved with the sanctioned rule-2 shim: pass `dfb::input` (implicit→`uint32_t`)
  to the still-`constexpr` free function `get_dataformat(...)` (`dataflow_api.h:300`), which reads the same
  `unpack_src_format[]` slot the member getter would. **Suggested doc fix:** rule 7 (and the whitelist §A metadata
  table) should note that the DFB member getters require a `DataflowBuffer` object and so are unavailable in a
  `constexpr`/type-selection context; there, the `constexpr` cb-id free function via the `dfb::name → uint32_t`
  shim is the correct spelling, not a rule-7 violation. (The audit's own "Recipe notes" already flagged
  `get_dataformat` as forcing a judgment call; this is the concrete failure of the object-getter route.)

### Confusion
- **`KernelRunArgs` for RTA-less kernels.** After the port every kernel has zero runtime args (both address RTAs
  became `TensorBinding`s). The recipe says the run-args entry "may be omitted," but `program_run_args.hpp:90`
  says "A `KernelRunArgs` must be specified for ALL kernels." To satisfy the stronger header invariant
  unambiguously, the port emits an empty `KernelRunArgs{.kernel = ...}` for each of the three kernels rather than
  omitting them. Worth reconciling the two statements in the docs. **(Verification note for the orchestrator: if the
  spec validator rejects the empty entries, drop `run_params.kernel_run_args` entirely — both should be equivalent.)**

## Open items for downstream

- **Cross-op kernel touches:** none. All three kernels and both shared headers (`common.hpp`, `common_dataflow.hpp`)
  live in and are owned by this op directory; no fork, no in-place edit of an out-of-directory kernel.
- **Pre-existing anomalies observed but intentionally left untouched (route to the ops team, not this port):**
  - Reader uses `ctas.tile_width` where writer/compute use `ctas.tile_height` for the same `num_blocks_in_column`
    quantity (`intimg_reader.cpp:55` vs `intimg_writer.cpp:67`, `intimg_compute.cpp:258`). Harmless only because
    tiles are square (32×32); would break for a non-square tile. Preserved verbatim (behavior-neutral today).
  - The `num_batches` loops in all three kernels are dead-bounded to 1 (`validate_on_program_cache_miss` hard-fails
    `input_shape[0] != 1`). The `num_batches` CTA is always 1. Left as-is.
  - The dead `// create_cb(... AXIS_3_BUFFER_1 ...)` comment in the legacy factory was not carried over (the whole
    factory body was rewritten); the corresponding enum `IntImgCB` (and its dead `AXIS_3_BUFFER_1` mention) lived in
    the old factory's anonymous namespace and is gone with the rewrite. Zero functional change.
- **Test coverage note:** the only located test is `tests/ttnn/unit_tests/operations/reduce/test_intimg.py`
  (bf16 + fp32, DRAM, four shapes). No C++ gtest exercises this op. The fp32 parametrization is what exercises the
  `unpack_modes`/`enable_32_bit_dest` path, so it must stay in the no-regression run.

## Test command(s)  *(orchestrator runs these; build/test verification is the orchestrator's)*

Build (Metal + all TTNN test binaries):
```
./build_metal.sh --build-tests
```

No C++ gtest for this op. Correctness (the no-regression baseline; covers both bf16 and fp32 = the fp32-dest path):
```
pytest tests/ttnn/unit_tests/operations/reduce/test_intimg.py -x -v
```
