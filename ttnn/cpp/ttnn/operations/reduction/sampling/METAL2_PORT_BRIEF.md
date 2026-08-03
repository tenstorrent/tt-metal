# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/reduction/sampling`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `ccf3df7c4ab 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels` *(carry this line into the port report's Provenance section)*

**What you are porting:** one device operation, `SamplingDeviceOperation`, with one factory, `SamplingProgramFactory` (`device/sampling_program_factory.cpp`), and three kernels, all owned by this op:

- `device/kernels/dataflow/reader_values_indices_tensor.cpp` (created once over the whole core grid)
- `device/kernels/dataflow/writer_interleaved.cpp` (one instance per core)
- `device/kernels/compute/sampling.cpp` (one instance per core)

No semaphores are declared, so there is no `SemaphoreSpec` work.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (see `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor`, a `create_descriptor` returning a plain `tt::tt_metal::ProgramDescriptor` (`device/sampling_program_factory.hpp:14-15`).
- **Op-owned tensors:** none. The factory never populates a `buffers` vector; the output is a normal framework-allocated tensor (`device/sampling_device_operation.cpp:183-190`).
- **Target concept:** `ProgramSpecFactoryConcept`, with no op-owned tensors.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · `get_dynamic_runtime_args` (deprecated hook) · `override_runtime_arguments` (not-yet-supported replacement) · pybind `create_descriptor` · other migration-risky pybind (which would have surfaced as a `safe` warning). All `no` on the readiness sheet and all confirmed absent in the code.

## Construct — to do

### Tensor bindings (six, all Case 1)

Every one is the mechanical case: express the binding as a `TensorParameter` / `TensorBinding`, have the kernel build `TensorAccessor(tensor::name)`, and delete the address argument together with its `TensorAccessorArgs` plumbing.

| Binding | Legacy delivery | Kernel arg it replaces | Accessor site |
|---|---|---|---|
| `input_values` | `emplace_runtime_args` @ `device/sampling_program_factory.cpp:377` | reader RTA 0, `values_addr` (`reader_values_indices_tensor.cpp:52`) | `:74` |
| `input_indices` | same call, `:378` | reader RTA 1, `indices_addr` (`:53`) | `:76` |
| `output` | `emplace_runtime_args` @ `:422` | writer RTA 0, `dst_addr` (`writer_interleaved.cpp:39`) | `:239` |
| `temp` | same call, `:422` | writer RTA 1, `temp_addr` (`:40`) | `:115` |
| `k` | same call, `:422` | writer RTA 2, `k_addr` (`:41`) | `:93` |
| `p` | same call, `:422` | writer RTA 3, `p_addr` (`:42`) | `:104` |

No Case 2 bindings, so no `get_bank_base_address` bridge is needed anywhere. No borrowed-memory CBs either, since every `CBDescriptor` leaves `buffer` and `tensor` at `nullptr`, so no `DataflowBufferSpec::borrowed_from` work.

Host-side `TensorAccessorArgs` plumbing that disappears: `device/sampling_program_factory.cpp:362-363` (reader) and `:388-391` (writer).

**Note the legacy delivery shape**, since it is not the usual one: the factory already hands whole `MeshTensor`s to `emplace_runtime_args` rather than `buffer()->address()` values. So there is no `->address()` expression to hunt for in this op. The kernels still receive a plain `uint32_t` base through `get_arg_val`, and that is what the typed binding replaces.

### TensorParameter relaxation

**none.** The readiness sheet reads `none`, consistent with the op having no custom hash.

### TensorAccessor 3rd arg

**none.** All six accessors already use the 2-argument form, so there is nothing to drop.

### CB endpoints

18 CBs, all allocated over the same `core_grid`. There is a **single config path**: the factory has no sharding or layout branches, and `sub_core_grids`, `num_users` and `Wt` change which and how many cores run, never which kernel touches which CB. So each CB below carries one disposition, and you do not need to re-census per config.

Per node, exactly one reader instance, one writer instance and one compute instance are co-resident. The writer and compute are instantiated once per core over a disjoint single-core `core_ranges` (`CoreRangeSet single_core{CoreRange(core, core)}`, `device/sampling_program_factory.cpp:385`), so this is **not** the dual-instance work-split shape: each node sees one instance of each kernel.

**Self-loop, 8 CBs.** Bind the single touching kernel PRODUCER **and** CONSUMER. Kernel code is untouched.

| CB | Name | Sole toucher | Why single-toucher |
|---|---|---|---|
| `c_5` | `input_transposed` | compute | full FIFO cycle inside `top_k` (`compute/sampling.cpp:238`, `:281`, `:290`, `:329`) |
| `c_6` | `index_transposed` | compute | same cycle (`:239`, `:282`, `:291`, `:330`) |
| `c_7` | `values` | compute | produced in `top_k` (`:348`, `:354`), then consumed in place by the softmax chain (`:474-482`) |
| `c_9` | `cb_cur_max` | compute | produced by `reduce_c` (`:477`), consumed by `sub_exp_block_bcast_cols_inplace` (`:65`, via `:479`) |
| `c_10` | `cb_cur_sum` | compute | produced by `reduce_c` (`:480`), cycled by `recip_block_inplace` (`:161-175`), consumed by `mul_block_bcast_cols` (`:150`) |
| `c_13` | `output` | writer | role-free raw peek only: `cb_out.get_write_ptr()` (`writer_interleaved.cpp:147`) plus `use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_out)` as the NoC write source (`:242`). No FIFO ops. |
| `c_14` | `k` | writer | locked producer with no consumer: NoC read staging, read back via `k_ptr[core_id]` (`:94-102`) |
| `c_15` | `p` | writer | same shape (`:105-113`) |

**Already legal 1:1, 10 CBs, no action:** `c_0` `input_values`, `c_1` `cb_local_vals`, `c_2` `index`, `c_3` `scaler_max`, `c_17` `scaler_sum`, `c_4` `topk_mask`, `c_8` `output_ind`, `c_11` `rand_tile`, `c_12` `final_indices_rm`, `c_16` `temp`. The per-CB producer and consumer sites are tabulated in `METAL2_PREPORT_AUDIT.md`.

**Multi-binding advanced option: not needed on any CB.** **Dead CBs: none** (all 18 `buffer_index` values are referenced).

Two points worth keeping in view while you rewrite the writer, because they look like extra endpoints and are not:

- `c_16` `temp` is touched twice by the writer: a raw NoC read into `cb_temp.get_write_ptr()` (`writer_interleaved.cpp:117-119`) and then `generate_bcast_unary_scalar`'s own `reserve_back` / `push_back` (`:127`). Both belong to the same kernel, so `c_16` stays one producer plus one consumer (compute consumes it at `compute/sampling.cpp:397`, via `:475`). The `cb_temp.reserve_back(1)` / `push_back(1)` lines at `:116` and `:121` are commented out in the legacy source; leave them as they are, since removing them is a behavior-neutral cleanup that belongs to the ops team, not to this port.
- `c_12` `final_indices_rm` is read by the writer at an offset, `cb_final_indices.get_read_ptr() + core_id * final_indices_stick_size` (`writer_interleaved.cpp:145`). That is a device-side CB-pointer offset, not a second endpoint and not an offset base pointer.

## Watch for

- **CB endpoints (multi-binding):** none. No CB in this op needs the flag, and no hidden second writer exists (there are no semaphores to coordinate one).
- **Cross-op / shared kernels:** the op instantiates **no** borrowed kernel files. All three `kernel_source` paths are its own (`device/sampling_program_factory.cpp:366-367`, `:416-417`, `:448`), no other op or test references them, and no `_metal2` fork exists beside any of them. So there is no fork to reuse, none to create, and no sunset list.

  The coupling that does exist is through **headers**, and one entry needs care:

  - ⭐ **`generate_bcast_unary_scalar(CircularBuffer cb, uint32_t scalar)`** in `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp:44`, called at `writer_interleaved.cpp:127`. The donor takes a **`CircularBuffer` by value**, which is the donor-shape table's flagged entry: op-by-op porting plus DFB-replacing-CB on the consumer side leaves no clean per-op story today, so it is **for cross-team discussion, not for you to resolve in the port diff**. The header is broadly shared (nine other kernels across `normalization/softmax`, `transformer/sdpa` and `data_movement/bcast`), so do not change the donor signature. Mechanically the call site already constructs the `CircularBuffer` from a CB id, and a `dfb::` token satisfies that constructor through its constexpr `uint32_t` cast, so a compiling call site is reachable. If it is not, that is an assumption-violation stop, not a fix to invent.
  - ✓ **`generate_mask<cb_mask_in, PNHt>(...)`** in `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp:215`, called at `writer_interleaved.cpp:130`. Cross-family donor, but the CB travels as a template `uint32_t` non-type parameter, which `dfb::name`'s constexpr cast handles. No donor work.
  - ✓ **`calculate_and_prepare_reduce_scaler<dfb_id, PoolType, ReduceDim>()`** (`writer_interleaved.cpp:77-80`) and **`compute_kernel_lib::reduce<...>`** (`compute/sampling.cpp:190-198`), both from `ttnn/cpp/ttnn/kernel_lib/`. Every CB rides as a `uint32_t` NTTP. No donor work; the lib team owns these files.

- **RTA varargs:** none. Both dataflow kernels read a fixed run of runtime args at constant indices (`reader_values_indices_tensor.cpp:52-53`; `writer_interleaved.cpp:39-42`), and all six are the tensor bases above, so they all become `TensorBinding`s rather than named args. **Prefer named args** for anything else you touch; nothing in this op needs the vararg mechanism.

- **The writer kernel is the only one still on `CircularBuffer`.** `writer_interleaved.cpp:10` includes `api/dataflow/circular_buffer.h` and `:84-91` construct eight `CircularBuffer` objects, while the reader (`reader_values_indices_tensor.cpp:8`) and compute (`compute/sampling.cpp:21`) are already on `api/dataflow/dataflow_buffer.h` and `DataflowBuffer`. Expect the writer to carry the bulk of the CB-to-DFB rewrite, including the handle naming, and note that the interaction with the ⭐ donor above lands in exactly this file.

- **The compute and writer kernels are instantiated per core with a baked `core_id` CTA.** The factory emits one writer and one compute `KernelDescriptor` per running core, each over a single-core range, differing only in the `i` / `core_id` compile-time arg (`device/sampling_program_factory.cpp:383-458`, with `core_id` read at `writer_interleaved.cpp:64`). Up to 32 of each. These cover **disjoint** node sets, so this is the ordinary 1:1 shape and **not** the demoting-per-group-CTA anti-pattern: keep `core_id` a compile-time arg on a per-node `KernelSpec`; do not consolidate the instances or demote it to a runtime arg.

- **Architecture-gated compile-time configuration to preserve verbatim.** The factory chooses `use_32bit_index` and `stable_sort` from `device->arch()` (`device/sampling_program_factory.cpp:44`, `:51`), which drive the index CB data format (`:57`), the compute config's `fp32_dest_acc_en` (`:454-456`), and CTAs read by all three kernels. Carry these through unchanged; they are not port decisions.

- **Two latent issues are recorded in `METAL2_PREPORT_AUDIT.md` under *Misc anomalies* and route to the ops team, not into your diff.** Most relevant to you: a `W == 32` (`Wt == 1`) input hangs the compute kernel's local-sort loop, so do not treat that configuration as a working baseline if you reach for it while testing.
