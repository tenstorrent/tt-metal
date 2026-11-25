# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/isin`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `` c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form `` *(carry this line into the port report's Provenance section)*

Audit run 2026-11-03 at `977421c3a1c`. Readiness-sheet row 191 was reconciled and re-fetched live on 2026-11-25, and the code was re-checked at `233a155fba1`. Since the audit, the op has changed only in its `tt-metalium` include paths (#67012), so every `file:line` below still holds.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `descriptor`. The single factory `IsInProgramFactory` has `create_descriptor(const IsinParams&, const IsinInputs&, Tensor&)` (`device/isin_program_factory.hpp:13`, body `isin_program_factory.cpp:42-155`) and no coordinate parameter.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept` (the sheet's `Porting Target`). There is no `override_runtime_arguments` to translate. The execution model is `SPMD`, and the port is 1:1: one `ProgramDescriptor` becomes one `ProgramSpec`.
- **Gate-cleared, confirmed absent:** `TensorParameter relaxation` is `none`, and there is no `get_dynamic_runtime_args` hook.
- **Non-gating facts:** there is no custom hash (the op uses the default reflection hash over `IsinParams` + `IsinInputs`), no pybound `create_descriptor` (`isin_nanobind.cpp` binds only the free function), and no device-op-class edit forced.

## Construct — to do

**Tensor bindings** (per binding). All three are `Buffer*` entries in `emplace_runtime_args` today (`isin_program_factory.cpp:145-147`):

- `elements` — **Case 1** (via `TensorAccessor`; reader RTA0, `isin_reader.cpp:86`, `:97-98`) → express as `TensorParameter` / `TensorBinding`; the kernel uses `TensorAccessor(tensor::name)`.
- `test_elements` — **Case 1** (reader RTA1, `isin_reader.cpp:87`, `:99-102`) → same.
- `output` — **Case 1** (writer RTA0, `isin_writer.cpp:12`, `:19`) → same. This binds the device-op's return tensor, which is `*optional_out` when the caller passes `out` (`isin_device_operation.cpp:47-51`).
- These disappear with the bindings: the three `get_arg_val` address reads, and the `TensorAccessorArgs(*buffer).append_to(...)` plumbing (`isin_program_factory.cpp:90-92`; kernel side `isin_common.hpp:77-79`).

**TensorParameter relaxation:** `none`. Keep strict `TensorSpec` matching on all three bindings.

**TensorAccessor 3rd arg:** drop the redundant page-size arg at `isin_reader.cpp:97-98`, `isin_reader.cpp:99-102` and `isin_writer.cpp:19`. All three are Class 2, so do **not** set `dynamic_tensor_shape`. (The writer site passes an element count, not bytes. It's inert because every access is `page_id = 0`.)

**CB endpoints:**

- Self-loop `c_0` `ELEMENTS` and `c_1` `TEST_ELEMENTS`: the reader is their only toucher in every config, so bind it as both PRODUCER and CONSUMER. These are DM self-loops: legal on Gen1, Quasar-uplift debt.
- `c_2` `OUTPUT` is a legal 1:1: reader PRODUCER (`reserve_back` / `push_back`), writer CONSUMER (`wait_front` / `pop_front`).
- There is no multi-binding, dead CB or conditional DFB. All three CBs are 1-page and unconditional over `all_cores` (`isin_program_factory.cpp:108-111`).

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** none. `isin_reader.cpp` and `isin_writer.cpp` are op-owned and bound only by `IsInProgramFactory` (`isin_program_factory.cpp:114`, `:116`). No `_metal2` fork exists and none is needed.
- **RTA varargs:** none. RTAs are fixed-index (reader `0-3`, writer `0-2`). After the address RTAs become bindings, each kernel keeps two named per-node RTAs, `subchunks_per_core` and `subchunks_offset`, whose values vary by core group and running offset (`isin_program_factory.cpp:135-150`).
- **`get_dataformat` in a template argument** (`isin_reader.cpp:91`, `using elements_number_type = std_type_t<get_dataformat(ctas.elements_cb)>;`). This is a constant-expression context, so keep the free function and pass it the binding token: `get_dataformat(dfb::<elements>)`. The member getter can't produce a constant expression. The host must set `data_format_metadata` on that DFB for the token's getter to work. Record the site in the port report as Gen1-only token-form debt.
- **Shared kernel-side CTA decoder:** both kernels call `IsInCTAs` / `get_ctas()` (`isin_common.hpp:56-93`; called at `isin_reader.cpp:83` and `isin_writer.cpp:10`). It decodes every CB index (CTAs 3-5) and every `TensorAccessorArgs` (from CTA 12). After the port, each kernel binds only its own resources, so the helper must not reference a `dfb::` / `tensor::` token the calling kernel doesn't bind. The file is op-private.
- **Dead CTA placeholders:** CTA slots 0-2 are `0u` and never read (`isin_program_factory.cpp:75-80`). Named CTAs dissolve them. The "index 11" comments (`isin_program_factory.cpp:75`, `isin_common.hpp:54-55`) are stale; the accessor args start at 12.
- **Helpers re-wrap the CB on every call:** `load_to_cb` / `write_to_dram` (`isin_common.hpp:96-143`) take a `uint32_t` CB index and build their own `CircularBuffer`, while `kernel_main` holds a second wrapper over the same CB (`isin_reader.cpp:105-107`). On Gen1 both forms behave identically; this is Quasar-uplift debt only.
- **Raw L1 casts stay as they are** (`isin_reader.cpp:28-34`, `:63-64`, on `get_read_ptr()` / `get_write_ptr()` results). This shape is correct on a DFB.
- **Stale include and handle names:** the kernels include `api/dataflow/circular_buffer.h` (`isin_common.hpp:7`, `isin_reader.cpp:8`) and use `*_cb` / `cb_obj` names. Don't carry either into the ported files.
- **Kernel configs:** `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` (`isin_program_factory.cpp:123`, `:130`). Neither sets `opt_level`, so both resolve to O2. Carry the same reader/writer RISC and NOC assignment into the port.
