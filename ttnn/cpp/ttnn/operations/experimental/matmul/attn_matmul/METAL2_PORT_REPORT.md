# Port Report — experimental/matmul/attn_matmul

Port of `ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul` from the `ProgramDescriptor` API to Metal 2.0,
per `ai/port/metal2_port.md`. Line references into the legacy code are against `origin/main` @ `fd25f0f4aac`;
references into the ported code are against this branch.

## Outcome

**PORTED** — the op's single factory (`AttnMatmulProgramFactory`) and its two owned kernels converted; the
borrowed writer binds the existing `_metal2` fork. Verification on a Blackhole p150b with Watcher on and the
Metal 2.0 legality checks forced on:

| Suite | Result |
|---|---|
| `tests/ttnn/nightly/unit_tests/operations/matmul/test_attn_matmul.py -k "attn_matmul and not group_attn_matmul"` (300 cases: `test_attn_matmul`, `test_attn_matmul_fp32`, `test_attn_matmul_with_program_cache`, `test_attn_matmul_with_program_cache_exhaustive`, `test_attn_matmul_compute_grid_shorter_than_device`) | 282 passed, 18 skipped (the test's own shard-shape guards at `:849`, `:854`, `:857`), 0 failed |
| `models/demos/falcon7b_common/tests/unit_tests/test_falcon_attn_matmul.py` (attn_matmul + both `attn_matmul_from_cache` modes, 4 geometries) | 12 passed |
| Bit-for-bit comparison against the legacy factory (36 outputs: 6 geometry/dtype configs × {attn_matmul, from_cache pre, from_cache post} × {cache miss, cache hit}; bf16 / bf8 / fp32 / mixed inputs, fp32 accumulation on and off, full and half grid) | **36/36 bit-exact** (`torch.equal` on the raw outputs); both builds created 18 program-cache entries |

`tests/ttnn/distributed/test_multidevice_TG.py::test_galaxy_attn_matmul` needs a Galaxy and was not run.
The `METAL2_CHECKS_FORCED` marker was present in every test log (16 / 668 / 24 occurrences).

## Provenance

- **Recipe docs (this port):** `akertesz/op-porting-recipe` @ `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(read from a verbatim local import of the recipe branch's `docs/…/metal_2.0/` tree, which this branch does not carry)*
- **Audit docs (inherited):** the same recipe revision, `akertesz/op-porting-recipe` @ `4bd4bf42bfe`
- **Code ported from:** `origin/main` @ `fd25f0f4aac` (2026-10-26)

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`, as the audit chose: `AttnMatmulProgramFactory::create_program_artifacts`
(`device/attn_matmul_program_factory.hpp`, `.cpp`) returns one `ProgramSpec` + `ProgramRunArgs`; no
`override_runtime_arguments`, no op-owned tensors. The framework refreshes the three tensor bindings
(`input_a`, `input_b`, `output`) on a cache hit, which is exactly what the pointer-patching descriptor path
refreshed (the three `Buffer*` run-arg slots).

### Device-op-class edits
- Pybind entry points removed: none (`attn_matmul_nanobind.cpp` binds only the two public wrappers).
- Custom `compute_program_hash`: none (default reflection hash; untouched).
- Nothing outside the factory body and the two owned kernels changed. `attn_matmul_device_operation.{hpp,cpp}`,
  `attn_matmul_device_operation_types.hpp`, `attn_matmul.{hpp,cpp}` and the nanobind file are byte-identical.

### Open items
- Relaxation candidate (not applied): the kernels take `Mt` / `Kt` / `Nt` as run-args and only the `in0`
  DFB capacity (`Kt` entries) is baked into the program, so every new `kv_len` / `num_tokens` costs a program
  build under the strict `TensorSpec` match (PR #41267 made the same observation about the hash). A shape
  relaxation on the non-`K` dims would need the custom concept and an override that recomputes the geometry
  run-args on cache hits; ops-team call.

## Handoff points

none — no capitulation, no boundary-rule violation, no kernel-lib gap, no removed pybind surface.

## Successes

- **Audit brief → port, 1:1.** Every item in `METAL2_PORT_BRIEF.md` mapped directly: the self-loop on
  `intermed0` (two `DFBBinding`s with one accessor name on the compute `KernelSpec`), the rung-1 reuse of
  `writer_unary_interleaved_start_id_metal2.cpp` with its vocabulary (`dfb::out`, `tensor::dst`, `num_pages`,
  `start_id`, no defines), and the idle-core zero-arg shape (the work unit spans the whole device grid and every
  node gets a value for every named run-arg; the `0` that stood in the `Buffer*` slot simply vanished).
- **Compute `hw_config` — "check for a dropped field" (recipe, Hardware configuration / Style A).** The legacy
  descriptor applied only `math_fidelity` and `fp32_dest_acc_en`; the helper would have handed the caller's
  `math_approx_mode` / `dst_full_sync_en` back. The recipe's instruction to force the legacy-default result on
  the returned config is what keeps the port behaviour-identical for a caller that sets either knob
  (`attn_matmul_program_factory.cpp`, the `compute_hw` block).
- **`unpack_modes` required-entry rule.** `test_attn_matmul_fp32[in_dtype=FLOAT32]` exercises Float32 `in0` /
  `in1` / `intermed0` / `intermed2` under `enable_32_bit_dest`; the conditional `UnpackToSrc` entries derived
  from the legacy default satisfied the validator first time.
- **Pattern: Pass DFB handles directly to LLKs and kernel-lib helpers.** `dfb::name` tokens worked unchanged as
  the `uint32_t` non-type template parameters of the id-only `compute_kernel_lib::untilize<…>` / `tilize<…>`
  forms and as LLK call arguments (`transformer_attn_matmul.cpp`), with no `.id` extraction.
- **Whitelist rule 7 / §A.** `get_tile_size(cb_id)` on a runtime `const` became `dfb_in0.get_tile_size()`
  (DM kernels have `chlkc_descriptors.h` in the JIT build, so the getter is available there too).

## Friction

- **Gap (workspace):** a fresh `git worktree` has no submodules checked out; `./build_metal.sh` fails at configure
  ("Missing submodules") until `git submodule update --init --recursive` runs. `workspace_setup.md` covers clones
  only.
- **Gap (workspace):** `./build_metal.sh` does not enable ccache unless `-c` / `--enable-ccache` is passed; the
  recipe's quoted command builds cold.
- **Confusion (recipe, Hardware configuration):** the text names `ComputeGen1Config` / `DataMovementGen1Config`
  and a `std::get<ComputeGen1Config>(compute_hw)` form, and says `to_compute_hardware_config(device->arch(),
  config)`. On this tree `ComputeHardwareConfig` is a flat struct (`fpu_math_fidelity`, `sfpu_precision_mode`,
  `enable_32_bit_dest`, `double_buffer_dest`, `unpack_modes`, optional `config_1xx` / `config_2xx`), the TTNN
  helper takes only the config (`to_compute_hardware_config(const ComputeKernelConfig&)`), and the DM helpers
  take no arch (`ttnn::create_reader_datamovement_config()`). The field table is right; the type and signature
  names are stale.
- **Confusion (recipe, Ensure the legality checks are enabled):** `grep -n 'bool skip_validation'
  tt_metal/impl/metal2_host_api/*.cpp` misses `program_spec/program_spec.cpp` (a subdirectory). Grepping the
  directory recursively finds `BuildProgramFromSpec`. Worth noting the TTNN side: `ttnn::CONFIG` reports
  `validate_program_args=false` by default, so without the force the spec-path validation is in fact off.
- **Gap (verification):** the recipe's test step has no bit-exactness check against the legacy build; "tests
  still pass" is a PCC bar. A capture-and-compare (same seeded inputs through the legacy and ported `.so`,
  `torch.equal` on the outputs, cache-miss and cache-hit) is cheap with ccache and is the only direct evidence
  of the zero-functional-change invariant; see the Outcome table.
- **Confusion (kernel-lib):** the id-only `compute_kernel_lib::untilize` / `tilize` forms the legacy kernel
  uses carry `KERNEL_LIB_ID_ONLY_HELPER`, which expands to `[[deprecated]]` under `ARCH_QUASAR` only
  (`ttnn/cpp/ttnn/kernel_lib/id_only_helpers.hpp`). The port keeps the legacy form (minimal diff; Gen1 target);
  a Quasar uplift will have to move to the by-reference form. Neither the recipe nor the catalog mentions the
  macro.

## Open items for downstream

- **Shared kernel touch (rung 1 — reused an existing fork; no new file created):**
  `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` →
  bound `…/writer_unary_interleaved_start_id_metal2.cpp` (issue #52228). Remaining legacy binders on `main` (by
  bound path, quasar copies and same-named private copies excluded):
  `data_movement/reshape_on_device/device/reshape_tile_program_factory.cpp`,
  `eltwise/unary_backward/device/unary_backward_program_factory.cpp`,
  `examples/example/device/multi_core_program_factory.cpp`,
  `examples/example/device/single_core_program_factory.cpp`. The sunset of the legacy copy waits on those.
- **Quasar-uplift debt recorded by this port** (not Gen1 work): the compute kernel keeps the id-only kernel-lib
  helper forms (deprecated on Quasar, see Friction); the reader keeps its hand-built loopback endpoint
  (`UnicastEndpoint` from `my_x[noc.get_noc_id()]` / `my_y[…]` with a raw `CoreLocalMem` destination,
  `reader_transformer_attn_matmul.cpp`) — the Device 2.0 guide's "Avoid" form — and its two cursor peeks taken
  before any FIFO op, which rely on `intermed1` / `intermed2` being single-entry DFBs. All three are faithful to
  the legacy kernel and belong to a post-port style / semantic pass.
- **Latent issues in the device-op class, not touched by the port (ops team):**
  - `attn_matmul_device_operation.cpp:72-75` — the only `compute_with_storage_grid_size` ≤ device-grid check is a
    `TT_ASSERT`; in release an oversized grid reaches `split_work_to_cores` unchecked.
  - `attn_matmul_program_factory.cpp` (legacy and port) — `math_approx_mode`, `packer_l1_acc` and
    `dst_full_sync_en` from the caller's `compute_kernel_config` are resolved and ignored; the port reproduces
    that (precise SFPU, double-buffered Dest).
  - The legacy header / factory comments claiming `padded_shape` is folded into `compute_program_hash()` referred
    to a custom hash removed by PR #47232; the port restates the substance in its own words and drops the
    dangling reference.
- **Test coverage note:** `test_attn_matmul_fp32` passes `math_approx_mode=True` (`test_attn_matmul.py:106-110`),
  so a port that let the helper's `Approximate` through would have diverged from legacy on that test's config —
  but the kernel runs no SFPU op, so no test can observe the knob either way. Nothing passes
  `dst_full_sync_en=True`.
- **Sibling carry-over:** `experimental/matmul/group_attn_matmul` has the same shape (descriptor, pointer-patching,
  sheet row 193, same owners) and the same provenance chain; this port is a direct template for it.
