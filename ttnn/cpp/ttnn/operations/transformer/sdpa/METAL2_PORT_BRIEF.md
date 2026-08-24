# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/transformer/sdpa`

> **Scoped port.** The `sdpa` directory is a whole-op **RED** (five of seven DeviceOperations are blocked — see
> `METAL2_PREPORT_AUDIT.md`). This brief covers **only the clean subset that cleared every gate:**
> **`SDPADeviceOperation`** (`SDPAProgramFactory`) and **`JointSDPADeviceOperation`** (`JointSDPAProgramFactory`).
> Do **not** touch `SparseSDPA*`, `RingDistributedSDPA`, `RingJointSDPA`, or `ExpRingJointSDPA` — they are blocked and
> re-audited when their gates clear. The full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared (for this subset):** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `d6087d9353f 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources` *(carry into the port report's Provenance section)*

## TTNN factory analysis

Both cleared DeviceOperations port to `ProgramSpecFactoryConcept`. These facts feed the TTNN ProgramFactory wiring
(→ `ttnn_factory.md`):

- **Current concept:** `descriptor` (both — `create_descriptor` returning a `ProgramDescriptor`:
  `sdpa_device_operation.hpp:25`, `joint_sdpa_device_operation.hpp:26`).
- **Op-owned tensors:** none (both).
- **Target concept:** `ProgramSpecFactoryConcept` (both — `Override runtime args method? = no`, so the framework
  refreshes tensor bindings on a cache hit and the factory writes one method).
- **Gate-cleared, confirmed absent** (each would have blocked this brief): a non-`none` `TensorParameter relaxation`
  (`none` for both) · `get_dynamic_runtime_args` (absent). Neither DeviceOperation carries a custom hash, an
  `override_runtime_arguments`, or a pybound `create_descriptor` — none of which would gate anyway.

## Construct — to do

**Tensor bindings** (all **Case 1** — via `TensorAccessor`). In the legacy code the base address arrives as a
`uint32_t` runtime arg (delivered by the framework's `Buffer*`-binding — the factory pushes `Buffer*` objects into the
runtime args, e.g. `sdpa_program_factory.cpp:1407-1413`, `joint_sdpa_program_factory.cpp:592-599`) and is immediately
wrapped in `TensorAccessor(args, addr)`. Express each as a `TensorParameter` / `TensorBinding`; the kernel builds
`TensorAccessor(tensor::name)` and the address-via-RTA plus its `TensorAccessorArgs` compile-time plumbing both
disappear. No Case-2 (raw-pointer) bindings; no borrowed-memory (`borrowed_from`) reads.

- **SDPA** (`reader_interleaved.cpp:211-216`, `writer_interleaved.cpp:84,114`):
  `q_in`, `k_in`, `v_in`, `mask` (optional), `page_table` (chunked), `attention_sink` (optional),
  `chunk_start_idx` (flexible-chunked) → reader; `out`, `cu_window_seqlens` (windowed) → writer.
- **JointSDPA** (`joint_reader.cpp:56-61`, `joint_writer.cpp:52-53`):
  `q`, `k`, `v`, `joint_q`, `joint_k`, `joint_v` → reader; `out`, `joint_out` → writer.

**TensorParameter relaxation:** none (both).

**TensorAccessor 3rd arg:**
- **SDPA** — drop the redundant page-size arg @ `device/kernels/dataflow/dataflow_common.hpp:83`
  (`read_page_table_for_batch`). It is **Class 2** (interleaved page table; the value is
  `page_table_tensor.buffer()->aligned_page_size()`, `sdpa_program_factory.cpp:165`, so it equals the aligned page and
  the interleaved accessor realigns it anyway → inert). Pure no-op drop — **do not** set `dynamic_tensor_shape`.
- **JointSDPA** — none (JointSDPA does not use the page-table path).

**CB endpoints** (per `(CB, config)`; no dead CB, no multi-binding for either DeviceOperation):
- **SDPA:**
  - *self-loop* (one toucher — bind the single kernel PRODUCER **and** CONSUMER): the compute intermediates `qk_im`,
    `out_im_A`, `out_im_B`, `max_A`, `max_B`, `sum_A`, `sum_B`, `exp_max_diff` (always); plus `page_table` (chunked),
    `cu_window_seqlens` (windowed), `recip_scratch` (streaming).
  - *plain 1P+1C* (already legal — one producer, one consumer): `q_in`, `k_in`, `v_in`, `mask_in`, `identity_scale_in`,
    `col_identity`, `chunk_start_idx_compute`, `chunk_start_idx_writer`, `attention_sink`, `out`.
  - *conditional DFB* (make the spec conditional on the config that keeps it live; the factory allocates it only there
    already): `mask_in`, `page_table`, `attention_sink`, `chunk_start_idx_compute`, `chunk_start_idx_writer`,
    `recip_scratch`, `cu_window_seqlens`.
- **JointSDPA:**
  - *self-loop*: `cb_qk_im` (c_24), `cb_out_im_A` (c_25), `cb_out_im_B` (c_26), `cb_max_A` (c_27), `cb_max_B` (c_28),
    `cb_sum_A` (c_29), `cb_sum_B` (c_30), `cb_exp_max_diff` (c_31).
  - *plain 1P+1C*: `cb_q_in` (c_0), `cb_k_in` (c_1), `cb_v_in` (c_2), `cb_mask_in` (c_3), `cb_identity_scale_in` (c_5),
    `cb_col_identity` (c_7), `cb_out` (c_16 — compute produces, writer consumes).
  - *conditional DFB*: `cb_mask_in` (only when `use_joint_mask`).

## Watch for

- **CB endpoints (no multi-binding to set), but two SDPA shapes not to misread:**
  - **KV-chain cross-core semaphore forwarding** (SDPA, **non-causal only**) — `k_in`/`v_in` are FIFO-produced locally
    by the reader, and a *peer core's* reader instance writes tiles into this core's CB via `noc.async_write` +
    `sender`/`receiver`/`valid` semaphores (`reader_interleaved.cpp:412-420,474-517,604-612,665-709`; semaphores
    declared `sdpa_program_factory.cpp:837-856`). This is the **same reader source on another node**, not a second
    on-node kernel — per node it is still 1 producer (reader) + 1 consumer (compute), so bind `k_in`/`v_in` as **plain
    1P+1C, not multi-binding**. Port the three semaphores as ordinary `SemaphoreSpec`s; the cross-core writes stay
    faithful. (JointSDPA has **no** semaphores at all.)
  - **`mask_in` producer flips by config** (SDPA) — the **reader** produces it when `use_provided_mask`, the **writer**
    otherwise (generated / lightweight / windowed). Mutually exclusive, so 1P+1C per config, but the producing *kernel*
    differs; bind accordingly per config.
  - **`cu_window_seqlens` aliases `q_in`'s CTA index** when not windowed (`writer_interleaved.cpp:778`) — a benign alias
    to keep `get_tile_size` well-formed; carry the conditional DFB, don't treat the alias as a second binding.
- **Cross-op / shared kernels** (function-call escapes only — SDPA/Joint own all their kernel `.cpp` files, no file-path
  borrows):
  - `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` → `calculate_and_prepare_reduce_scaler<uint32_t dfb_id, …>()`
    is already Device-2.0-native (the `uint32_t` cb/dfb-id template param is handled by `dfb::name`'s constexpr cast).
    Pass `dfb::name` in template position — **no donor change, no fork.** (`writer_interleaved.cpp:90`, `joint_writer.cpp:63`.)
  - `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` → `generate_bcast_col_scalar(CircularBuffer cb, …)` takes
    the **legacy `CircularBuffer`**; a `_metal2` fork **`generate_bcast_scalar_metal2.hpp` already exists beside it**
    (in `ttnn/cpp/ttnn/kernel/dataflow/`, **not** quasar), taking a `DataflowBuffer`. **Bind the existing fork** — swap
    the include and pass a named `DataflowBuffer` built from the token; **do not create a new fork.** (`writer_interleaved.cpp:95`,
    `joint_writer.cpp:68`.) Other consumers of the legacy header are a **sunset list**, not authorization to convert it in place.
- **RTA varargs:** **none** — every kernel reads a fixed, enumerable arg set. The `if (num_phases == 2)` and
  `if constexpr (!is_causal)` chain-metadata blocks are fixed-width optional-field reads (name each field), **not**
  variable-count loops. `TensorAccessorArgs<N>() + next_compile_time_args_offset()` chaining is compile-time-constant.
- **Compute kernels** (`compute/sdpa.cpp`, `compute/joint_sdpa.cpp`) consume/produce only CBs — no tensor-binding work.
