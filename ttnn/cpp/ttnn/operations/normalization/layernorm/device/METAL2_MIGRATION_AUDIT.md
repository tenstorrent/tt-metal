# layer_norm Metal 2.0 Migration Audit

Date: 2026-06-01
Scope: `ttnn/cpp/ttnn/operations/normalization/layernorm/device`
Guides audited against (from `origin/akertesz/metal2-documentation`):

- `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_migration_guide.md`
- `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/port_op_to_metal2_audit.md`
- `docs/source/tt-metalium/tt_metal/apis/kernel_apis/data_movement/device_api_migration_guide.md`

## Overall Result

**YELLOW (host GREEN, device-side completion gaps remain).**

- Host factory migration to Metal 2.0 is substantially complete (`ProgramSpec`/`ProgramRunParams` shape present).
- Device-side migration is not fully closed in Welford compute variants and one row-major helper path.

## Findings

### 1) Host Metal 2.0 factory migration

**Status: GREEN**

Evidence:

- `layernorm_op_multi_core.cpp` uses `m2::ProgramSpec`, `m2::KernelSpec`, `m2::DataflowBufferSpec`, `m2::WorkUnitSpec`, `m2::ProgramRunParams`.
- `layernorm_op_multi_core_sharded.cpp` uses the same Metal 2.0 spec/run-params model and semaphore specs.
- No `CreateProgram` / `CreateKernel` / `CreateCircularBuffer` / `SetRuntimeArgs` usage found in op code.

Assessment:

- Matches host guide migration intent (immutable `ProgramSpec` + mutable `ProgramRunParams`, named resource bindings).

### 2) Device-side Welford kernels still instantiate `CircularBuffer`

**Status: RED**

Evidence:

- `kernels/compute/layernorm_welford.cpp`:
  - `CircularBuffer cb_x_welford_obj(cb_x_welford);`
- `kernels/compute/layernorm_sharded_welford.cpp`:
  - `CircularBuffer cb_x_welford_obj(cb_x_welford);`

Assessment:

- These paths still rely on legacy CB wrapper semantics in otherwise DFB-migrated kernels.
- This is the most likely source for legacy `cb_reserve_back` watcher waypoints (`CRBW`) in Welford-enabled paths.

Recommended action:

- Replace alias-path `CircularBuffer` usage with `DataflowBuffer`-native alias handling and DFB bindings.

### 3) Legacy compile-time arg retrieval style in Welford kernels

**Status: YELLOW**

Evidence:

- `layernorm_welford.cpp` and `layernorm_sharded_welford.cpp` use:
  - `get_named_compile_time_arg_val("cb_x_welford")`
  - `get_named_compile_time_arg_val("welford_fp32_alias")`

Assessment:

- Host guide recommends the unified named retrieval API: `get_arg(args::name)`.
- Current style is functionally workable but not endpoint-consistent with migration guidance.

Recommended action:

- Move to `get_arg(args::cb_x_welford)` and `get_arg(args::welford_fp32_alias)` with matching schema entries.

### 4) Row-major helper is still `CircularBuffer::AddrSelector`-coupled

**Status: YELLOW**

Evidence:

- `kernels/dataflow/layernorm_dataflow_utils.h`:
  - `use<CircularBuffer::AddrSelector::READ_PTR>(cb_out_rm)` inside `write_row_major_block_from_cb(...)`.

Assessment:

- Helper is templated on `CB` but still references `CircularBuffer` selector type explicitly.
- This weakens pure-DFB portability for the row-major writer path.

Recommended action:

- Refactor helper to DFB-native read-pointer usage (or add DFB-compatible selector abstraction).

## Audit Conclusion

- `layer_norm` host-side Metal 2.0 migration is in strong shape.
- Device-side migration is **not fully complete** due to remaining `CircularBuffer` and legacy arg-retrieval patterns in Welford paths, plus one helper coupling.
- A follow-up migration patch should target those specific files before this op is treated as fully closed against Metal 2.0 migration guidance.
