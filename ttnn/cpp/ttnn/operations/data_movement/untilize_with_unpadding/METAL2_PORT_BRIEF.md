# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ · Features ✓

**Scope:** the **five not-yet-ported factories** — `SingleCore`, `MultiCoreBlockInterleaved`, `MultiCoreColInterleaved`, `MultiCoreSharded`, `MultiCoreNDSharded`. `MultiCoreInterleaved` is already on Metal 2.0; use it (`untilize_with_unpadding_multi_core_interleaved_program_factory.cpp`) as the in-repo reference — it establishes the named DFB/TensorParameter binding style, the vararg-RTA approach, and the `_metal2` kernel-fork convention this port reuses. The atomic porting unit is one ProgramFactory; if context runs short, port a subset and launch the rest from a fresh instance.

## TTNN factory analysis

These feed the port's TTNN ProgramFactory wiring (→ `port_op_to_metal2_ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`.

- **Op-owned tensors:** none.
- **MeshWorkload:** not needed (single-program per factory).
- **Pybind `create_descriptor`:** none.
- **Other risky pybind:** none.
- **Custom `override_runtime_arguments`:** none.

Nothing to delete on the device-op / nanobind side.

## Construct — to do

**Tensor bindings** (per binding) — all **Case 1** or **clean**; no Case 2, no compute-kernel blocker:

- **SingleCore** — `input`, `output`: **Case 1**. Express as `TensorParameter`/`TensorBinding`; kernels build `TensorAccessor(ta::input)` / `TensorAccessor(ta::output)`. Legacy `Buffer*` RTAs (`emplace_runtime_args`) disappear.
- **BlockInterleaved** — `input`, `output`: **Case 1** (legacy pushes `buffer->address()` *directly* into the RTA list — the silent-wrong-on-cache-hit hazard; the typed binding fixes it). Bind both; kernels use `TensorAccessor(ta::name)`.
- **ColInterleaved** — `input`, `output`: **Case 1** (same `->address()`-in-RTA hazard as block). Bind both.
- **Sharded** — `input`: **clean borrowed-memory DFB** → `DataflowBufferSpec{ .borrowed_from = <input> }` on CB `c_0` (reader does the fake-push; compute consumes). `output` (out-sharded path): **borrowed-memory DFB** on CB `c_17` → `borrowed_from = <output>`; this edge is **producer-only** (no in-kernel consumer — it is the resident output), so if the DFB spec validator rejects it, apply the **fake-CB workaround** (see porting recipe). `output` (interleaved-out path): **Case 1** via the shared `writer_unary_stick_layout_interleaved_blocks.cpp`.
- **NDSharded** — `input`: **Case 1**, bound on **both** the reader and the writer (writer needs `accessor_src.shard_pages`). `output`: **Case 1**.

**Custom hash:** none.

**Kernel forks (the bulk of the mechanical work).** The five factories share their compute/reader kernels with un-ported ops, so rewrite-in-place would break co-borrowers. Follow the interleaved port's `_metal2` fork convention — fork the source, swap CB indices for `dfb::in`/`dfb::out` tokens and the address-via-RTA for `ta::name`, leave the legacy file for the co-borrowers:

- Compute: `untilize.cpp`, `untilize_wh.cpp`, `untilize_w.cpp`, `untilize_variable_num_blocks.cpp` (a `untilize_variable_num_blocks_metal2.cpp` fork already exists — reuse it for NDSharded).
- Readers: the eltwise/unary `reader_unary_interleaved_*` / `reader_unary_sharded.cpp` and the sharded-pool `reader_unary_nd_sharded_blocks.cpp`.
- The op's **own** writers (`writer_unary_unpad_dims_split_rows.cpp`, `writer_unary_stick_layout_wh_multicore.cpp`, `writer_unary_stick_layout_col_multicore.cpp`, `writer_unary_unpad_*_sharded.cpp`, `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp`) are already Device 2.0; they still need the `ta::name` / `dfb::name` binding-token swap (fork if any are shared).

## Watch for

- **Borrowed-memory DFBs:** sharded factory `c_0` (`...sharded...:97`) and `c_17` (`...sharded...:126`) → `borrowed_from`. The `c_17` producer-only edge may need the fake-CB workaround.
- **Cross-op / shared kernels:** every reader and compute kernel is borrowed from a shared pool/family — fork with `_metal2`, do not rewrite in place (full port-together set in `METAL2_PREPORT_AUDIT.md` → Team-only).
- **RTA varargs:** none in these five factories (fixed RTAs / CRTAs only).
- **Stale `sharding_addrgen.hpp` include** in the two ND-sharded kernels is unused — leave it or drop it, but don't build a donor bridge for it.
