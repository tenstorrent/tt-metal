# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/full`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4d6e9518bf5 2026-07-30 docs(metal_2.0): require an explicit opt_level when porting compute kernels` *(carry this line into the port report's Provenance section)*

## What you are porting

One device operation, `FullDeviceOperation` ([device/full_device_operation.hpp:20](device/full_device_operation.hpp#L20)), with three factories, each owning one kernel:

| Factory | Kernel | Selected when |
|---|---|---|
| `FullInterleavedProgramFactory` ([device/full_program_factory_interleaved.cpp:17](device/full_program_factory_interleaved.cpp#L17)) | [device/kernels/writer_full.cpp](device/kernels/writer_full.cpp), bound **twice** | output is interleaved |
| `FullShardedProgramFactory` ([device/full_program_factory_sharded.cpp:20](device/full_program_factory_sharded.cpp#L20)) | [device/kernels/writer_full_sharded.cpp](device/kernels/writer_full_sharded.cpp) | output is sharded **with** an explicit `shard_spec` |
| `FullNDShardedProgramFactory` ([device/full_program_factory_nd_sharded.cpp:20](device/full_program_factory_nd_sharded.cpp#L20)) | [device/kernels/writer_full_nd_sharded.cpp](device/kernels/writer_full_nd_sharded.cpp) | output is sharded **without** a `shard_spec` |

Selection logic: [device/full_device_operation.cpp:13-22](device/full_device_operation.cpp#L13-L22).

Two facts that shape the whole port:

- **The op takes no input tensors.** `tensor_args_t` is an empty struct ([device/full_device_operation_types.hpp:22](device/full_device_operation_types.hpp#L22)). The output tensor it creates is the only tensor in play, so there is exactly **one** tensor binding per factory. The fill value is a scalar that rides an RTA.
- **All three kernels are pure data movement.** There is no compute kernel, no semaphore, and no multi-core coordination anywhere in the op. Each kernel fills one CB page with the fill value, then writes that page to every output page it owns.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (all three factories declare `static ProgramDescriptor create_descriptor(...)`)
- **Op-owned tensors:** none
- **Target concept:** `ProgramSpecFactoryConcept` (the readiness sheet's `Porting Target` column states this directly for all three factory rows)
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · `get_dynamic_runtime_args` (deprecated hook) · `override_runtime_arguments` (not-yet-supported replacement) · pybind `create_descriptor` · other migration-risky pybind. All `no` on the sheet and zero grep hits in the op. [full_nanobind.cpp](full_nanobind.cpp) binds only the `moreh_full` free function.

## Construct — to do

**Tensor bindings** (per binding):

- `output` — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(tensor::name)`. Identical in all three factories.

  Today the base is delivered by the `Buffer*`-binding form: the factory pushes `output.buffer()` (the pointer object, not `->address()`) into the runtime-arg list, and the kernel unpacks a raw `uint32_t` at RTA index 0 and feeds it straight to a `TensorAccessor` constructor. Three pieces disappear together when you bind the tensor:

  | Piece | Sites |
  |---|---|
  | `output.buffer()` in the RTA list | [interleaved :104](device/full_program_factory_interleaved.cpp#L104), [:108](device/full_program_factory_interleaved.cpp#L108), [:110](device/full_program_factory_interleaved.cpp#L110) · [sharded :89-90](device/full_program_factory_sharded.cpp#L89-L90) · [nd_sharded :69](device/full_program_factory_nd_sharded.cpp#L69) |
  | Host-side `TensorAccessorArgs(output.buffer()).append_to(...)` CTA plumbing | [interleaved :53](device/full_program_factory_interleaved.cpp#L53), [:80](device/full_program_factory_interleaved.cpp#L80) · [sharded :57](device/full_program_factory_sharded.cpp#L57) · [nd_sharded :57](device/full_program_factory_nd_sharded.cpp#L57) |
  | Kernel-side `output_addr` unpack + `TensorAccessorArgs<N>` | [writer_full.cpp:13](device/kernels/writer_full.cpp#L13) and [:21](device/kernels/writer_full.cpp#L21) · [writer_full_sharded.cpp:13](device/kernels/writer_full_sharded.cpp#L13) and [:23](device/kernels/writer_full_sharded.cpp#L23) · [writer_full_nd_sharded.cpp:13](device/kernels/writer_full_nd_sharded.cpp#L13) and [:22](device/kernels/writer_full_nd_sharded.cpp#L22) |

  Note the accessor-args base index differs per kernel (`<3>`, `<5>`, `<6>`) because each factory pushes a different number of leading CTAs. All three collapse to `TensorAccessor(tensor::name)`.

**TensorParameter relaxation:** none. The op has no custom hash and the sheet lists `none` on all three rows.

**TensorAccessor 3rd arg:** none. All three accessor constructions are two-argument (`TensorAccessor(dst_args, output_addr)` at [writer_full.cpp:58](device/kernels/writer_full.cpp#L58), [writer_full_sharded.cpp:60](device/kernels/writer_full_sharded.cpp#L60), [writer_full_nd_sharded.cpp:59](device/kernels/writer_full_nd_sharded.cpp#L59)). There is no page-size override to drop. The kernels already *query* the size off the accessor via `get_aligned_page_size()`, which stays as-is.

**CB endpoints:** self-loop on every CB, in every config. Each CB has exactly one toucher, and that one kernel is both the FIFO producer (`reserve_back` / `push_back`) and the FIFO consumer (`wait_front` / `pop_front`), so bind it PRODUCER **and** CONSUMER:

| CB | Config | Bind |
|---|---|---|
| `c_0` (fill-value page) | interleaved, reader present (`num_pages > num_cores`) | self-loop on the Writer-config instance of `writer_full.cpp` |
| `c_0` (fill-value page) | interleaved, no reader | self-loop on the same instance |
| `c_1` (fill-value page, reader copy) | interleaved, reader present only | self-loop on the Reader-config instance of `writer_full.cpp` |
| `c_24` (fill-value page) | sharded | self-loop on `writer_full_sharded.cpp` |
| `c_24` (fill-value page) | ND-sharded | self-loop on `writer_full_nd_sharded.cpp` |

No 1P+1C assignment, no multi-binding advanced option, no dead-CB drop anywhere in this op.

**In-directory shared header:** [device/kernels/full_kernel_common.hpp](device/kernels/full_kernel_common.hpp) is included by all three kernels and holds `zero_buffer(uint32_t cb_id, uint32_t bytes)` ([:15](device/kernels/full_kernel_common.hpp#L15)), the `value` union, and the `onepage` constant. `zero_buffer` takes a CB index and constructs a `CircularBuffer` from it internally; it is called at [writer_full.cpp:34](device/kernels/writer_full.cpp#L34), [writer_full_sharded.cpp:36](device/kernels/writer_full_sharded.cpp#L36), and [writer_full_nd_sharded.cpp:35](device/kernels/writer_full_nd_sharded.cpp#L35). Its `uint32_t cb_id` parameter is the "✓ OK" handle form, so a `dfb::` handle passes through unchanged via the constexpr cast and no signature change is forced. If you do change it, all three consumers are in scope for this port and convert together. This is an op-owned header, not a shared-kernel-library file, and it is **not** a Device 2.0 holdover (see the audit's Device 2.0 section for why).

## Watch for

- **CB endpoints (multi-binding):** none, but here is the thing that looks like one so you do not have to re-derive it. The interleaved factory **is** the dual-instance work-split pattern: it pushes the same `kernel_source` into two `KernelDescriptor`s that differ only by `WriterConfigDescriptor` / `ReaderConfigDescriptor` and their per-instance page-split args, both over the same `all_cores` range ([device/full_program_factory_interleaved.cpp:55-88](device/full_program_factory_interleaved.cpp#L55-L88), split at [:101-:111](device/full_program_factory_interleaved.cpp#L101-L111)). Normally that means every co-touched CB needs a 1P+1C assignment. **Here it does not**, because each instance gets its *own* CB index through CTA 0 (Writer → `c_0` at [:52](device/full_program_factory_interleaved.cpp#L52), Reader → `c_1` at [:79](device/full_program_factory_interleaved.cpp#L79)), with a separate `CBDescriptor` for each ([:38](device/full_program_factory_interleaved.cpp#L38), [:69](device/full_program_factory_interleaved.cpp#L69)). No CB in this op is touched by two kernels. Also note the `c_1` CB and the whole reader `KernelDescriptor` are **conditional** on `has_reader` ([:66](device/full_program_factory_interleaved.cpp#L66)); the ported spec has to keep that conditional, so an interleaved output with `num_pages <= num_cores` builds one kernel and one DFB.
- **No hidden second writer to hunt.** Confirmed absent: no `get_local_cb_interface`, no `fifo_wr_ptr` / `fifo_rd_ptr`, no `evil_set_write_ptr` / `evil_set_read_ptr`, and no semaphore anywhere in the op. The only raw CB pointer access is `cb.get_write_ptr()` by that CB's own FIFO producer ([writer_full.cpp:31](device/kernels/writer_full.cpp#L31) and siblings), which is a peek on a binding the kernel already holds.
- **Cross-op / shared kernels:** none in either direction. The op owns all three kernel sources, no other op instantiates them, and no `_metal2` fork exists beside any of them. No fork rung applies: each factory has its own kernel file, and the interleaved factory's two instances of `writer_full.cpp` both convert in the same change. Nothing to sunset, nothing to coordinate.
- **RTA varargs:** none, prefer named RTAs throughout. Every runtime arg is read at a distinct literal index and every one is nameable. The names to use are already in the kernels:
  - `writer_full.cpp` ([:13-16](device/kernels/writer_full.cpp#L13-L16)): `output_addr` (becomes the tensor binding), `fill_value`, `num_pages_per_core`, `start_id`
  - `writer_full_sharded.cpp` ([:13-17](device/kernels/writer_full_sharded.cpp#L13-L17)): `output_addr` (becomes the tensor binding), `fill_value`, `start_page_id`, `num_pages_per_shard_row`, `num_pages_per_shard_col`
  - `writer_full_nd_sharded.cpp` ([:13-15](device/kernels/writer_full_nd_sharded.cpp#L13-L15)): `output_addr` (becomes the tensor binding), `fill_value`, `start_shard_id`
- **The `defines` must survive.** All three kernels select their fill loop with `#ifdef OUTPUT_DTYPE_BFLOAT16` / `OUTPUT_DTYPE_INT32` / `OUTPUT_DTYPE_FLOAT32`, supplied by `get_writer_defines(dtype)` ([device/full_program_factory_common.hpp:38-47](device/full_program_factory_common.hpp#L38-L47)) and converted by `defines_from_map` ([:49](device/full_program_factory_common.hpp#L49)). Exactly one is set per build. Carry them onto `KernelSpec::compiler_options.defines`; if none reaches the kernel, the fill loop compiles out entirely and the CB silently holds garbage. `defines_from_map` exists only to bridge a `std::map` into the legacy `KernelDescriptor::Defines` vector, so check whether it is still needed once the spec is in place.
- **Compile-time arg numbering shifts as you go.** The leading CTA in every kernel is the CB index (`cb_value` at CTA 0), which becomes a `dfb::` binding and stops being an arg. The sharded and ND-sharded factories also push an `aligned_page_size` CTA at index 3 that **no kernel reads** ([sharded :55-56](device/full_program_factory_sharded.cpp#L55-L56) versus [writer_full_sharded.cpp:19-23](device/kernels/writer_full_sharded.cpp#L19-L23); [nd_sharded :55-56](device/full_program_factory_nd_sharded.cpp#L55-L56) versus [writer_full_nd_sharded.cpp:17-22](device/kernels/writer_full_nd_sharded.cpp#L17-L22)). It is recorded in the audit's Misc anomalies as an ops-team item, so do not treat cleaning it up as port work, but do not let it silently misalign your named `compile_time_args` schema either.
