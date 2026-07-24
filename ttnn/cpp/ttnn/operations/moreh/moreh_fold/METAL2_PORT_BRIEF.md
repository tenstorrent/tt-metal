# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_fold`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — single factory `MorehFoldOperation (single-descriptor)` (`device/fold_program_factory_rm.cpp`), `create_descriptor()` returns `ProgramDescriptor`.
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · migration-risky pybind. All `no` on the readiness sheet and confirmed in code.

## Construct — to do

**Tensor bindings** (per binding):

- `input` — **Case 1** (via `TensorAccessor`) → express as a `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(tensor::name)`. The legacy base rides reader RTA[0] as `input.buffer()` (`Buffer*`-binding form, `fold_program_factory_rm.cpp:174`), consumed as `input_addr` (`reader_fold_rm.cpp:16,49`) — the RTA base and the `TensorAccessorArgs<3>` compile-time plumbing both disappear.
- `output` — **Case 1** (via `TensorAccessor`) → same shape. Legacy base is `output.buffer()` in writer RTA[0] (`fold_program_factory_rm.cpp:197`), consumed as `output_addr` (`writer_fold_rm.cpp:13,24`).

**TensorParameter relaxation:** **none.** Do **not** add `dynamic_tensor_shape`. (The dated 3rd-arg triage doc suggests Class 1 / `dynamic_tensor_shape`, but the op has no custom hash and the readiness sheet lists relaxation = none; adding it would be a behavior change. See `METAL2_PREPORT_AUDIT.md` → Questions. If the sheet/triage owner later confirms a relaxation is wanted, that becomes a separate custom-hash change, not this port.)

**TensorAccessor 3rd arg:** drop the redundant page-size 3rd arg (Class 2, pure no-op — Metal 2.0 supplies `aligned_page_size` implicitly):
- `reader_fold_rm.cpp:49` — `TensorAccessor(input_args, input_addr, input_cb_page_size)` → 2-arg.
- `writer_fold_rm.cpp:24` — `TensorAccessor(output_args, output_addr, output_cb_page_size)` → 2-arg.
- Once the bases move to bindings, the `input_cb_page_size` / `output_cb_page_size` RTAs that fed the dropped 3rd arg are also dead — remove them.

**CB endpoints:**
- `c_0` (input) — **self-loop** (one toucher: reader only), all configs. Bind the reader PRODUCER **and** CONSUMER.
- `c_1` (scratch) — **self-loop** (one toucher: reader raw-peek only), in the configs where it is allocated (DRAM source with unaligned page, or Blackhole — `fold_program_factory_rm.cpp:101`). Bind the reader PRODUCER **and** CONSUMER.
- `c_16` (output) — **legal 1:1**, no action: reader is the producer (`reserve_back`/`push_back`), writer is the consumer (`wait_front`/`pop_front`).

## Watch for

- **CB endpoints (multi-binding):** none — no hidden second writer, no multi-reader. Straightforward census (2 self-loops + 1 legal 1:1).
- **Cross-op / shared kernels:** none — both kernels are op-owned and include only `tt_metal` HAL/firmware headers; no port-together coupling.
- **RTA varargs:** none — all RTAs are distinct fixed-index fields (reader 0-20, writer 0-3); name each. Note the reader's `aligned` flag (RTA 20) and the two-step-read path gated on it.
