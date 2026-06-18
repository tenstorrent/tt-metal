# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/pool/generic`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ · Features ✓

Op: `Pool2D` device-operation, single factory `Pool2D::MultiCore` (`device/pool_multi_core_program_factory.cpp`, builds via `pool2d_multi_core_sharded_with_halo_v2_impl_new`). Two code paths share the factory: a non-indices path (`reader_pool_2d.cpp` + `compute_pool_2d.cpp`) and a `return_indices` MPWI path (`reader_mpwi.cpp` + `compute_mpwi.cpp`); both port together. A `config_tensor_in_dram` flag selects DRAM-vs-sharded delivery of the reader-indices / scalar-config tensors and changes how those bindings classify (see below).

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `port_op_to_metal2_ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Op-owned tensors:** Yes — `create_workload_descriptor` builds & parks a reader-indices tensor (`pool_multi_core_program_factory.cpp:1133-1152`) and, for avg-pool with non-trivial scalar layout, a scalar-config tensor (`:1192-1212`). Wire these as `op_owned_tensors`.
- **MeshWorkload:** op-owned tensors only (NOT a genuine multi-program need). The per-coord program is built once and copied into each coord-range entry (`:1217-1280`); Pool2D is morally single-program.
- **Pybind `create_descriptor`:** none.
- **Other risky pybind:** none.
- **Custom `override_runtime_arguments`:** none. (Per-core RTAs are emitted inline at `:937-974`; the framework patches them by kernel index — keep the deterministic kernel-push order at `:976-983`.)

## Construct — to do

**Tensor bindings** (per binding):

- **input** (`raw_in_cb` / `in_shard_cb_id`) — **Case 2 (raw pointer)** → bind as `TensorParameter`, pull the shard base via `get_bank_base_address`, keep the existing raw address arithmetic in `read_kernel_with_top_left_index` unchanged (`reader_pool_2d.cpp:94-101`, `reader_mpwi.cpp:289-297`). The CB is address-only (no producer/consumer FIFO) → resolve with the fake-CB workaround. Dataflow kernels — bridge available; NOT the compute-kernel-blocked case.
- **reader_indices** (`in_reader_indices_cb_id`) — **per-path:**
  - DRAM path (`config_tensor_in_dram`): **Case 1 (via `TensorAccessor`)** → express as `TensorParameter`; kernel builds `TensorAccessor(ta::name)` (the CTA-baked DRAM address #35 at `:793` and `TensorAccessorArgs(reader_indices_buffer)` at `:816` both disappear). Reads via `load_config_tensor_if_in_dram` (`pool_kernels_common.hpp:82`).
  - Sharded path: address-only fake CB (read via `get_read_ptr` at `reader_pool_2d.cpp:280`) → fake-CB workaround.
- **scalar config** (`config_cb_id`, avg-pool only) — same per-path split: Case 1 in DRAM path (CTA-baked addr #33 at `:791`, `TensorAccessorArgs(config_buffer)` at `:818`); address-only fake CB in sharded path (`reader_pool_2d.cpp:294`).
- **output(s)** (`out_cb_id`, `out_idx_cb_id` when `return_indices`) — **clean** borrowed-memory DFBs (real producer+consumer) → `borrowed_from`.

**Custom hash:** delete custom `compute_program_hash` → default (sanctioned exception). `device/pool_op.cpp:168-185`, declared `device/pool_op.hpp:68`. (It omits `TensorSpec`; the default keys on it correctly.)

## Watch for

- **Notable constructs:**
  - Aliased CB @ `pool_multi_core_program_factory.cpp:657-675` — `pre_tilize_cb_id` / `fast_tilize_cb_id`, two `CBFormatDescriptor` views over one L1 allocation → one `DataflowBufferSpec` per index with `advanced_options.alias_with` mutually naming the other; same `num_entries*entry_size`, bound to the same kernels; do NOT split. TILED-output path only (`has_pre_tilize`).
  - Fake CBs (address-only edges): `raw_in_cb` (both readers), and the sharded-path edges of `in_reader_indices_cb` / `config_cb`. Same CBs are genuine DFBs on their DRAM-path edge.
- **Cross-op / shared kernels:** all four kernels `#include` `ttnn/cpp/ttnn/operations/pool/device/kernels/pool_kernels_common.hpp` (in-family, `pool/`), which includes `experimental_device_api.hpp`. Both are Device 2.0. These headers are shared by sibling `pool/` ops, so their Metal 2.0 rewrite (named-token bindings, `dfb::name` casts) is a port-the-family-together change — coordinate. The donor's `load_config_tensor_if_in_dram` takes `TensorAccessorArgs<N>` (pass `ta::name.args`) plus a CTA-baked DRAM address NTTP (eliminated by the Case-1 binding).
- **RTA varargs:** none — kernels read fixed positional RTAs; prefer named RTAs.
- **Split reader on the borrowed input shard:** both readers (BRISC + NCRISC) read the same address-only input-shard CB and split the work; keep the fake-CB workaround consistent across both reader bindings.
