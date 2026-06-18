# Metal 2.0 Port Plan — Reshard legacy factories

Scope: port the three legacy `ProgramDescriptor` reshard factories to the Metal 2.0
host API (`ProgramArtifacts` = `ProgramSpec` + `ProgramRunArgs`), matching the shape of
the already-ported `NdReshardCopyLocalShardFactory` / `NdReshardCopyPagesFactory`.

| Factory | Selected for | Source kernels |
|---|---|---|
| `ReshardSameHeightFactory<local_is_output>` | WIDTH_SHARDED ↔ WIDTH_SHARDED | `reshard_same_height_reader/writer.cpp` |
| `ReshardSameWidthFactory<local_is_output>` | HEIGHT_SHARDED ↔ HEIGHT_SHARDED | `reshard_same_width_reader/writer.cpp` |
| `ReshardGenericFactory` | general reshard | `reshard_reader.cpp` / `reshard_reader_diff_width.cpp` |

## Kernel forking

The legacy kernels live in `sharded/device/kernels/dataflow/`. They are forked into this op's
own `reshard/device/kernels/dataflow/` directory so the legacy copies remain untouched for any
other consumer, and all migrated code stays inside the reshard op directory (scope discipline).

## Mechanical kernel changes (per recipe whitelist)

1. **Local sharded CB → borrowed DFB.** The local shard CB is backed by the local tensor's buffer
   (legacy `cb.buffer = local_buffer`). Modeled as a `DataflowBufferSpec` with
   `borrowed_from = <local TensorParameter>`. The kernel uses `DataflowBuffer(dfb::shard_cb)` and
   reads `get_write_ptr()` / `get_read_ptr()` for the local base L1 address (no FIFO sync — the CB
   is used purely as an address source, exactly as in legacy).
2. **`Buffer*` / raw base-address RTA → `TensorAccessor`.** The remote buffer base address was
   passed as a runtime arg (`Buffer*` or `address()`). Replaced by `TensorAccessor(ta::remote)`
   and `get_bank_base_address()` on device. (Case 2 raw-pointer binding.)
3. **Positional RTAs → named RTAs + positional varargs.** Scalar per-core args become named RTAs
   (`runtime_arg_schema.runtime_arg_names`). The variable-length per-segment tail becomes
   positional varargs accessed with `get_vararg(i)`.

## DFB endpoint binding (fake / self-loop CBs)

Both data-movement kernels (k0/k1) run the *same* source and both only need the local base address.
To satisfy the DFB invariant (≥1 producer, ≥1 consumer per node) we bind k0 as PRODUCER and k1 as
CONSUMER of each local DFB. Both `get_write_ptr()` and `get_read_ptr()` are always valid regardless
of endpoint role, so address retrieval is unaffected.

The borrowed local `TensorParameter` must still carry a `TensorBinding` on ≥1 kernel even though the
kernel never dereferences its `TensorAccessor`; we bind it on both kernels.

## Varargs (variable per-core tail length)

`num_runtime_varargs_per_node` is deprecated. The scalar `num_runtime_varargs` is expanded uniformly
to every node. So we set `num_runtime_varargs = max-over-cores(tail length)` and **pad** each core's
vararg vector to that max with zeros. The kernel only reads the real prefix (it loops over a named
count: `num_segments` / `num_reads` / `num_ranges`), ignoring the padding.

## Generic factory specifics

The generic kernels index *backwards* into a leading block of `physical_core_coords`
(`get_arg_val(start_x_index)` with `start_x_index < num_x_cores`). The whole legacy positional RTA
list is emitted as one contiguous vararg block — **minus** the dropped `input_addr` slot (now from
`ta::input`). Random indices into the core-coords prefix and the sequential reads after it both use
`get_vararg(i)`; removing the `input_addr` element keeps every downstream sequential read aligned.

## Out of scope

No algorithmic changes, no host-side arg-packing optimizations (e.g. the generic factory recomputes
page ranges per core — preserved as-is). CRTA promotion of uniform scalars is left as a future
optimization; uniform scalars are emitted as per-node named RTAs for simplicity.
