# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/normalization/layernorm`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ · Features ✓

## TTNN factory analysis

The factory concept is selected downstream from these facts (→ `port_op_to_metal2_ttnn_factory.md`); the port does not pick it here. Carry these forward:

- **Op-owned tensors:** none
- **MeshWorkload:** not needed
- **Pybind `create_descriptor`:** delete at `layernorm_nanobind.cpp:319-358` (LayerNormMultiCoreProgramFactory) and `layernorm_nanobind.cpp:361-393` (LayerNormShardedProgramFactory)
- **Other risky pybind:** `LayerNormDeviceOperation` class exposed with `compute_program_hash`, `create_output_tensors`, `compute_output_specs`, `select_program_factory` at `layernorm_nanobind.cpp:250-316`; `LayerNormParams` and `LayerNormInputs` structs with all fields exposed at `layernorm_nanobind.cpp:223-248`
- **Custom `override_runtime_arguments`:** none

## Construct — to do

**Tensor bindings** (per binding, per factory):

*LayerNormMultiCoreProgramFactory:*

- `input (a)` — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. Kernel builds `TensorAccessor(ta::input)`. Remove `a_addr` RTA (`layernorm_op_multi_core.cpp:131, 587`) and the `TensorAccessorArgs` CTA injection (`layernorm_op_multi_core.cpp:358`).
- `residual (b)` — **Case 1** → re-express. Remove `b_dram_addr` RTA (`layernorm_op_multi_core.cpp:188, 595`) and CTA injection (`layernorm_op_multi_core.cpp:359`).
- `gamma` — **Case 1** → re-express. Remove `gamma_dram_addr` RTA (`layernorm_op_multi_core.cpp:189, 593`) and CTA injection (`layernorm_op_multi_core.cpp:360`).
- `beta` — **Case 1** → re-express. Remove `beta_dram_addr` RTA (`layernorm_op_multi_core.cpp:190, 594`) and CTA injection (`layernorm_op_multi_core.cpp:361`).
- `output` — **Case 1** → re-express. Remove `dst_addr` RTA (`layernorm_op_multi_core.cpp:131, 604`) and CTA injection (`layernorm_op_multi_core.cpp:378`).

*LayerNormShardedProgramFactory:*

- `input (a)` — **clean** (borrowed-memory DFB via `CBDescriptor.buffer`). Port uses `DataflowBufferSpec::borrowed_from = "input"`.
- `residual (b)` — **clean** (borrowed-memory DFB). Port uses `borrowed_from = "residual"`.
- `stats` — **clean** (borrowed-memory DFB). Port uses `borrowed_from = "stats"`.
- `output` (CB16/CB17) — **clean** (borrowed-memory DFB). Port uses `borrowed_from = "output"`.
- `gamma` (DRAM) — **Case 1** → re-express. Remove `gamma_dram_addr` RTA (`sharded_layernorm_factory_helpers.cpp:1501`) and CTA injection (`sharded_layernorm_factory_helpers.cpp:683-684, 693-694`).
- `beta` (DRAM) — **Case 1** → re-express. Remove `beta_dram_addr` RTA (`sharded_layernorm_factory_helpers.cpp:1502`) and CTA injection (`sharded_layernorm_factory_helpers.cpp:683-684, 693-694`).

**Custom hash:** none to delete.

## Watch for

- **Aliased CBs** @ multiple sites → `DataflowBufferSpec::advanced_options.alias_with`. The Welford-fp32 alias pattern creates CBs with 2 `CBFormatDescriptor` elements sharing SRAM between two buffer indices. Sites (all conditional on `welford_fp32_alias` / `welford_state_fp32_alias` flags):
  - MultiCore: CB0+c_29 (`layernorm_op_multi_core.cpp:686-694`); CB18+c_30 (`:707-715`); CB19+c_31 (`:731-740`); CB23+c_29 (`:808-816`).
  - Sharded: CB0+c_29 (`sharded_layernorm_factory_helpers.cpp:1007-1012`); CB24+c_29 (`:1066-1071`).
  - Alias is an advanced/ninja feature — see `DFBAdvancedOptions` for legality constraints (mutually `alias_with`, same `num_entries * entry_size`, bound to same kernels).

- **Borrowed-memory DFBs** @ many sites → `DataflowBufferSpec::borrowed_from`. Sharded input (`a_buffer`), residual (`b_buffer`), stats (`stats_buffer`), output (`output_buffer`/`output_reshard_buffer`), and the Welford reciprocal LUT (`recip_buffer`) all use `CBDescriptor.buffer`. Each must declare `borrowed_from = <tensor_parameter_name>` on the `DataflowBufferSpec`.

- **Fake CBs (address-only):** `cb_reciprocals` (CB `c_25`) in both factories — backed by `recip_tensor.buffer()` but accessed in the Welford compute kernels via raw `get_tile_address` (no FIFO). Metal 2.0 `DataflowBufferSpec` requires ≥1 producer and ≥1 consumer. Port applies the sanctioned fake-CB workaround (see porting recipe). Sites: `layernorm_op_multi_core.cpp:825-833`; `sharded_layernorm_factory_helpers.cpp:1177-1186`.

- **Cross-op / shared kernels:** none (all kernel files owned by this op's own directory).

- **RTA varargs:** none.
