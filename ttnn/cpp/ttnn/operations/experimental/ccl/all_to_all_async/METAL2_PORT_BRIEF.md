# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorParameter Case 2 ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `6d8d49be280 2026-11-26 docs(metal_2.0): state that a port is performance- and functionality-neutral` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `descriptor (per-coord)`.
- **Op-owned tensors:** none.
- **Target concept:** `MeshWorkloadSpecFactoryConcept`.
  - Ported-from per-coordinate entry point: `AllToAllAsyncProgram::create_descriptor`, `device/all_to_all_async_program_factory.cpp:149`. Emit one `programs` entry per coordinate, `range = MeshCoordinateRange(coord)`.
  - Varies per coordinate: `ring_index` (CTA 0 of every kernel), the forward/backward fabric peers, the receiver core the semaphore increment targets, and the per-device strides/offsets.
  - Override to translate: `device/all_to_all_async_program_factory.cpp:607`. It refreshes `input`, `intermediate` and `output` on every kernel that takes them, plus the caller `GlobalSemaphore` on the sender writer and receiver reader. It ignores `coord`.
- **Custom `compute_program_hash`:** `device/all_to_all_async_device_operation.cpp:246`. Leave it intact.
- **Gate-cleared, confirmed absent:** a non-clearing `TensorParameter relaxation` · `get_dynamic_runtime_args`.

## Construct — to do

**Tensor bindings** (per binding; all arrive today as `Buffer*` RTAs):

- `input`: **Case 1** → `TensorParameter`; the sender reader (RTA 0, `TensorAccessorArgs<7>`) and receiver reader (RTA 1, `TensorAccessorArgs<10>`) build `TensorAccessor(tensor::input)`.
- `intermediate`: **Case 1** → the sender writer (RTA 0, `TensorAccessorArgs<13>`, passed to the fabric write helpers) and receiver reader (RTA 0, chained accessor).
- `output`: **Case 1** → the receiver writer (RTA 0, `TensorAccessorArgs<8>`). The sender writer also receives it (RTA 1) and builds an accessor it never uses.

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** none.

**CB endpoints:**

- Sender cores: `c_in0` legal 1P+1C (reader → writer). `c_in1` (packet headers) **self-loop** on the writer.
- Receiver cores: `c_in0` legal 1P+1C (receiver reader → receiver writer).
- The two `c_in0`s are on disjoint core ranges: two DFBs.

**CCL constructs** (`ccl_port.md`):

- **One caller `GlobalSemaphore`** (`operation_attributes.semaphore`).
  - Bound by the sender writer (RTA 2) and the receiver reader (RTA 2).
  - Rebind it in the translated override on every hit (*Caller-supplied* case).
  - Kernel side: `GlobalSemaphore(global_sem::…).address()` where the RTA was read. Every use stays as is: `safe_get_noc_addr` + packet header in the writer; `noc_semaphore_wait_min` / `noc_semaphore_set` in the receiver reader.
- **`FabricConnectionManager`, forward + backward, one link, one sender core** → two one-target `FabricConnectionsSpec`s (`forward`, `backward`), each bound by the sender writer.
  - Kernel side: `build_from_args` → `build_from_optional_bindings` (`interleaved_all_to_all_writer.cpp:65-66`).
  - Peers come from `get_ring_fabric_node_ids()` (`device/all_to_all_async_program_factory.cpp:171-193`); resolve them with `ccl_port.md`'s `fabric_target`.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** none. All four kernels are this op's own and bound only by this factory: convert in place.
- **Indexed-collection args:** none.
- **Verification needs ≥3 devices in a ring.** The op aborts on 2 devices before any kernel runs: the entry point turns `Ring` into `Linear` (`device/all_to_all_async_device_operation.cpp:295-298`) and the factory requires `Ring` (`device/all_to_all_async_program_factory.cpp:164-167`). Its functional test is `tests/nightly/t3000/ccl/test_all_to_all.py` (1×8). On an N300, the most you can check is that the same 2-device failure happens before and after.
- **Dead kernel args** (see the audit's Misc anomalies): several RTAs/CTAs are read and never used, and the sender writer builds an unused `output` accessor. A port reproduces the legacy op, so check the recipe for how dead args port rather than dropping them silently.
- **Shadowed name:** both receiver kernels read an RTA `num_pages_per_packet` that shadows a CTA of the same name (`interleaved_all_to_all_receiver_reader.cpp:18, 45`; `interleaved_all_to_all_receiver_writer.cpp:18, 39`). Named args may make that a collision.
