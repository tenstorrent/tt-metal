# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/randn`

> Audit cleared all gates. One gate was cleared by user waiver; see below. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ *(user waiver: readiness-sheet row is stale, see below)* · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section)*

**TTNN gate waiver (record in the port report).** The live readiness sheet (fetched 2026-10-07) still describes this op as it was before PR #57409 (`f5093e705ae`, 2026-09-25). Its `randn` / `RandnDeviceOperation` / `ProgramFactory` row shows:
- `Concept` = `legacy device-op`
- `Override runtime args method?` = `n/a`

Both are primary-column conflicts, which the audit recipe classes as "spreadsheet-broken". In the code, the factory is `descriptor` (`create_descriptor`, `device/randn_device_operation.hpp:31`) and carries a PD-style `override_runtime_arguments` (`:38`, defined `device/randn_program_factory.cpp:146`). Every other cross-checked column matches the code, and the sheet's own verdict is `yes (with PD step)`, with the PD step already landed. On 2026-10-07 the user waived this gate ("Waive"). The sheet refresh is owed to the readiness-sheet owner and does not block the port.

**The target concept rests on the audit, not the sheet.** The sheet's `Porting Target` reads `TBD (study op structure)` and the row is not owner-validated (`Diego validation` = `no`). The audit derived `CustomProgramSpecFactoryConcept` from the code (descriptor + `override_runtime_arguments`). If anything during the port contradicts that, stop and raise it rather than switching concepts.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `descriptor`
- **Op-owned tensors:** none. The op has **no input tensors** (`tensor_args_t {}`, `randn_device_operation.hpp:25`); its only tensor is the output.
- **Target concept:** **`CustomProgramSpecFactoryConcept`**. Translate `ProgramFactory::override_runtime_arguments` (`randn_program_factory.cpp:146-172`) into a method returning a `ProgramRunArgs` (port recipe: *Translating override_runtime_arguments*). In-repo precedents on this concept: `kv_cache/device/update_cache_multi_core_program_factory.hpp:25`, `fill_cache_multi_core_program_factory.hpp:25`, and `experimental/transformer/nlp_create_qkv_heads`.
- **Custom hash:** `RandnDeviceOperation::compute_program_hash` (`randn_device_operation.cpp:47-52`) hashes every attribute except `seed`. **Leave it exactly as is** (`ttnn_factory.md`, *the cache key*).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a `TensorParameter relaxation` that is neither `none` nor an analysis pointer · `get_dynamic_runtime_args` (deprecated hook). A custom hash, an `override_runtime_arguments`, and a pybound `create_descriptor` are **not** in this list: none of them gate. On this op the first two are present (above); there is no pybound `create_descriptor` (`randn_nanobind.cpp:53` binds only the function).

## Construct — to do

One factory, two kernels (writer DM + compute) over `all_cores` from `split_work_to_cores` (`randn_program_factory.cpp:50-51`).

**Tensor bindings** (per binding):

- `output` — **Case 1** (via `TensorAccessor`) → `TensorParameter` / `TensorBinding`. Today it is the `Buffer*` writer RTA 0 (`randn_program_factory.cpp:135`) plus `TensorAccessorArgs` at writer CTA 1 (`:80`). The writer builds `TensorAccessor(dst_args, dst_addr)` (`writer_standard_normal.cpp:25`), which becomes `TensorAccessor(tensor::output)`.

**The custom override, translated.** On a cache hit the legacy override refreshes two things; the `ProgramRunArgs` must carry both:
1. **Per-core compute seed** (compute RTA 0), redrawn every dispatch from `seeded_rng` (seeded from `operation_attributes.seed`) or the persistent `default_rng` (`:158-165`). This is why the op needs the custom concept: the seed is excluded from the hash, so it must be re-supplied each time. **Keep the draw order identical to `create_descriptor`**: cores `i = 0..num_cores-1` via `randn_core(i, grid)` (column-major, `:36`), one draw per core. A seeded call must give the same numbers on cache miss and cache hit.
2. **Output address** (writer RTA 0, `:167-170`). This becomes the `output` `TensorBinding` in the returned `ProgramRunArgs`; the raw `runtime_args[0] = output_addr` patch goes away.

Tile offsets and per-core counts derive from the hashed shape and stay as built.

**TensorParameter relaxation:** `none`

**TensorAccessor 3rd arg:** none — no accessor passes a 3rd arg

**CB endpoints:** `c_0` (`randn_program_factory.cpp:67-76`) — compute produces (`compute_standard_normal.cpp:25,83`), writer consumes (`writer_standard_normal.cpp:33,52,57,68`). **Legal 1:1.** The writer's `get_read_ptr` calls are peeks on its own consumer binding. No dead CBs.

## Watch for

- **CB endpoints (multi-binding):** none
- **Cross-op / shared kernels:** none. Both kernels are owned and used only by this op, and every `#include` is `tt_metal/*`. Convert them in place.
- **RTA varargs:** none. Name every arg:
  - compute: `seed`, `num_tiles` (`compute_standard_normal.cpp:104-105`). Keep `seed` a **named per-node RTA**: it is the value the override rewrites on every hit.
  - writer: `dst_addr` → binding, `start_id`, `num_tiles` (`writer_standard_normal.cpp:18-20`)
  - CTA 0 on both kernels is the CB id (`dst_cb_id`), which becomes a `dfb::` token.
- **Anything else the porter needs:**
  - **Two core groups, one kernel each:** `split_work_to_cores` returns two groups that differ only in `units_per_core`, which already travels as an RTA (`:121-133`). One `KernelSpec` per role over `all_cores` is enough; no per-group CTA split.
  - **Core-walk invariant:** `randn_num_cores` / `randn_core` (`:33-36`) mirror `split_work_to_cores`' column-major order, guarded by a `TT_FATAL` (`:52-57`). The translated override depends on them. Keep both helpers and the guard.
  - **Global RNG state:** `default_rng` (`:24`) and `distribution` (`:20`) are file-scope globals shared by `create_descriptor` and the override. Keep them unchanged; unseeded calls rely on `default_rng` advancing across dispatches. (They are not thread-safe; that's recorded for the ops team, not yours to fix.)
  - **Compute config:** `fp32_dest_acc_en` is **forced `true`** (`:108-109`, with a precision-range comment), and `OUTPUT_DTYPE_BFLOAT16` is a define set only for BF16 (`:92-96`). Carry both over verbatim. `math_fidelity`, `math_approx_mode`, and `dst_full_sync_en` come from the kernel config (`:98-111`). Legacy left `opt_level` unset; recent ports restate O3 explicitly because a Metal 2.0 `KernelSpec` defaults to O2.
  - The compute kernel passes `cb_dst.get_cb_id()` / `dst_cb_id` to compute LLK calls (`pack_tile`, `pack_reconfig_data_format`, `compute_kernel_hw_startup`, `copy_init`). Those take the `dfb::` token directly after the port.
  - The writer reads its page size with `cb_dst.get_tile_size()` (`writer_standard_normal.cpp:29`), a wrapper method, so it carries over onto the DFB object.
  - The kernels are already Device 2.0 (`Noc`, `CircularBuffer`, `TensorAccessor`, `CoreLocalMem`), so the kernel change is a binding-layer swap.
  - **Verification:** `tests/ttnn/nightly/unit_tests/operations/rand/test_randn.py`. Include a seeded call repeated twice (cache miss, then hit) and check that the outputs match, plus an unseeded repeat to check they differ. That exercises the translated override. Check the test's grid needs against the 8×8 card before porting.
