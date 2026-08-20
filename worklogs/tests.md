# Tests — LLK operand metadata on Metal 2 memory objects

**Branch:** `riverwu/m2-neat`. Spec: [`SPEC.md`](./SPEC.md). Background: [`BACKGROUND.md`](./BACKGROUND.md).

How to test the host configuration (Part I) and `to_llk_mem_descriptor(token)` (Part II). This is a test plan, not an implementation.

The function under test on the device is the additive conversion already sketched next to `to_llk_mem_descriptor(Cb<CbId>)`:

```cpp
constexpr LLKMemDescriptor to_llk_mem_descriptor(DFBBindingToken);
constexpr LLKMemDescriptor to_llk_mem_descriptor(ScratchpadBindingToken);
constexpr LLKMemDescriptor to_llk_mem_descriptor(TensorBindingToken<…>);
```

It returns the compile-time half (`format` + `TensorShape`). Address is not in the conversion.

---

## 0. How we test, and where

All new cases go in the existing Metal 2.0 mega file `tests/tt_metal/tt_metal/api/metal2_host_api/test_program_spec.cpp` (`unit_tests_api`). Same helpers (`test_helpers.hpp`), same fixtures, same mock devices.

Do **not** add a sibling test file, `sources.cmake` entries, `test_kernels/` sources, or anything under `tests/tt_metal/tt_metal/llk/` / `unit_tests_llk`.

| Kind | Fixture / site | Why |
|---|---|---|
| `ScratchpadSpec` still aggregate / hashable | file-scope `static_assert` next to the existing `hashable_v` / `is_aggregate_v` lists | `ScratchpadSpec` is missing from both lists today |
| Host `MakeProgramFromSpec` **rejects** | `ProgramSpecTestQuasar` | Same fixture as `DFBWithComputeEndpointRequiresDataFormat`. These never reach JIT. |
| Host accepts that carry LLK metadata | `ProgramSpecTestGen1` + `detail::CompileProgram` | No enqueue, but compilation still runs. Kernel `static_assert`s the folded descriptor so the values are proven on the token, not only that the spec validated. Mock Quasar cannot JIT TRISC. |
| Other folded descriptors + DRAM `is_dram` | `ProgramSpecTestGen1` + `detail::CompileProgram` | Same compile path; DFB / tensor / DRAM cases that are not part of the §2.2 Scratchpad matrix |

Two mechanisms. Rejects use only the first. Accepts that pass format/tile/face down use both (validate, then compile — still no dispatch):

| Mechanism | Proves | Does not prove |
|---|---|---|
| `MakeProgramFromSpec` succeeds or `ThrowsMessage` | SPEC §14 illegal config; §3.2 “binding ≠ LLK operand” | That the facts reached the token |
| Inline `KernelSpec::SourceCode` + kernel `static_assert` + `CompileProgram` | `to_llk_mem_descriptor(token)` is a constant and the mapping is right | Enqueue, silicon, experimental `copy_tile` / `pack_tile`, address arithmetic |

That is the same “does this kernel compile?” pattern as `ScratchpadAccessorBindingJITSmokeComputeKernel` and `TtKernelComputeShimCompiles`.

---

## 1. Existing coverage to keep (do not rewrite)

These already pin parts of the contract. New fields default to `nullopt`, so the designated-init sites **are** the “old Scratchpad inits still compile” regression — do not churn them.

| Already in this file / suite | What it pins |
|---|---|
| Scratchpad binding / size / hash tests (`ValidScratchpadSucceeds` … `DifferentScratchpadSizeProducesDifferentKernelHash`) | Scratchpad is still just working memory |
| `ScratchpadAccessorBindingJITSmokeComputeKernel` | Format-less `Scratchpad<T>(token)` still compiles on compute |
| `DFBWithComputeEndpointRequiresDataFormat` | Compute DFB still requires `data_format_metadata` |
| `DataFormatNotSupportedOnTargetArchitectureFails` | DFB format must be arch-supported (`Bfp8` on Quasar mock) |
| `AggregateSpecTypes.DataflowBufferSpecDesignatedInitializers` | DFB field names / designated init unchanged |
| `TensorBindingOnComputeKernelIsAccepted` | TensorParameter grows no LLK fields; compute bind stays legal |
| HW: `ScratchpadWriteReadback`, `LocalTensorAccessorBindingCompileComputeKernel` | Address seams (`get_base_address` / `get_bank_base_address`) — do not re-test here |

---

## 2. Host configuration (SPEC Part I / §14)

Rejection is `TT_FATAL` / `std::runtime_error` at `MakeProgramFromSpec`, not a converter `static_assert`. Rejects stay on the Quasar mock (including the `Bfp8` arch check). Accepts that carry LLK metadata go on the Gen1 mock so `CompileProgram` can run — nothing is enqueued, but the kernel is still compiled and `static_assert`s the values that landed on the token.

### 2.1 `ScratchpadSpec` type contract

Add `ScratchpadSpec` to the existing lists (it is the one program-scope spec that is missing):

```cpp
static_assert(hashable_v<ScratchpadSpec>, "ScratchpadSpec must be hashable via ttsl reflection");
static_assert(std::is_aggregate_v<ScratchpadSpec>,
              "ScratchpadSpec must remain an aggregate to support designated initializers");
```

Living-doc case next to `DataflowBufferSpecDesignatedInitializers`:

- Construct with only `.unique_id` / `.size_per_node` → the three new fields are `nullopt`.
- Construct with `.data_format_metadata = Float16_b` → that field is set; tile / face stay `nullopt`.

Do not nest the three fields. SPEC §2: designated init must stay `.data_format_metadata = …`, same names as DFB.

### 2.2 Validation — what to add

SPEC §1: **“has LLK metadata” means `data_format_metadata` is set.** Geometry without a format is illegal. SPEC §14 is the host reject table. Compute bind of a format-less scratchpad is **legal** (binding ≠ LLK operand). Contrast: `DFBWithComputeEndpointRequiresDataFormat`.

Every Scratchpad case below binds the scratchpad to a compute kernel. That is the point.

**Rejects** — Quasar, `MakeProgramFromSpec` only. These never JIT.

| Test | Spec fields | Expect |
|---|---|---|
| `ScratchpadFormatUnsupportedOnArchFails` | `Bfp8` on Quasar mock | throws; message names the ScratchpadSpec (mirror the DFB `Bfp8` test) |
| `ScratchpadTileWithoutFormatFails` | `tile_format_metadata` set, format omitted | throws |
| `ScratchpadFaceGeometryWithoutFormatFails` | `unpack_face_geometry_metadata` set, format omitted | throws |
| `ScratchpadInvalidFaceGeometryFails` | format + `FaceGeometry{face_r_dim=0}` (or `num_faces=0`, or `face_r_dim > FACE_HEIGHT`) | throws — same predicates as `CircularBufferConfig::set_unpack_face_geometry` |
| `ScratchpadFaceGridDoesNotFitTileFails` | format + default/32×32 tile + `FaceGeometry{face_r_dim=9, num_faces=8}` | throws — same overflow as `compute_num_faces_rc_dims` (9×4 = 36 rows > 32) |

DFB format-required / arch-supported are already tested. Add the two §14 DFB rejects that do **not** exist today (they are not covered by the compute-endpoint rule):

| Test | Spec fields | Expect |
|---|---|---|
| `DFBInvalidFaceGeometryFails` | compute endpoint, format set, `FaceGeometry{face_r_dim=0}` | throws |
| `DFBFaceGridDoesNotFitTileFails` | compute endpoint, format set, tile + overflowing face (same 9×8 example) | throws |

`Tile` construction already throws on an unsupported `{height, width}`. Do not add a second tile-shape table as a ProgramSpec test.

**Accepts** — Gen1. `MakeProgramFromSpec` then `CompileProgram`. Nothing is dispatched, but compilation still runs. The compute kernel constructs `Scratchpad<T>(scratch::pad)` and, when format is set, `static_assert`s `to_llk_mem_descriptor(scratch::pad)` against the §3.2 cookbook. Same kernel-as-assertion pattern as §3.1.

| Test | Spec fields | Host | Device `static_assert` |
|---|---|---|---|
| `ScratchpadComputeBindWithoutFormatSucceeds` | none | succeeds | Compile `Scratchpad<T>(token)` only. Do **not** call `to_llk_mem_descriptor` (SPEC §12 UB). Same idea as `ScratchpadAccessorBindingJITSmokeComputeKernel` — this case is the validation-matrix counterpart, not a second smoke. |
| `ScratchpadFormatAloneSucceeds` | format only (`Float16_b`) | succeeds | `Float16_b`, `{16,16,2,2}` |
| `ScratchpadFormatAndTileSucceeds` | format + `Tile{{16,32}}` | succeeds | `{16,16,1,2}` |
| `ScratchpadFormatAndFaceGeometrySucceeds` | format + `FaceGeometry{face_r_dim=1, num_faces=4}` (no tile) | succeeds | `{1,16,2,2}` |

These three format-bearing accepts **are** the Scratchpad fold checks. Do not add a second Scratchpad compile table in §3.3.

### 2.3 Host must still accept

Do **not** add a reject for:

- Scratchpad bound to compute with no LLK fields (covered by `ScratchpadComputeBindWithoutFormatSucceeds`).
- DFB with no compute endpoint and no format (already legal; existing DM-only fixtures).
- Format alone, no tile / no face.
- TensorParameter — no extra LLK fields, no host DRAM-as-LLK-source check (that is the device `is_dram` assert in §3.3).

---

## 3. Device — `to_llk_mem_descriptor` (SPEC Part II)

`ProgramSpecTestGen1`. Inline kernel source. Call the conversion only — not Blackhole-gated experimental ops.

### 3.1 The kernel is the assertion

```cpp
void kernel_main() {
    DataflowBuffer in(dfb::in);   // or Scratchpad<T> / LocalTensorAccessor<T> — additional, not a replacement
    constexpr auto desc = experimental::to_llk_mem_descriptor(/* token */);
    static_assert(desc.format == static_cast<uint8_t>(DataFormat::Float16_b));
    static_assert(desc.shape.face_r_dim == 16);
    static_assert(desc.shape.face_c_dim == 16);
    static_assert(desc.shape.num_faces_r_dim == 2);
    static_assert(desc.shape.num_faces_c_dim == 2);
}
```

`CompileProgram` succeeding means the conversion folded and the mapping is right. `CompileProgram` throwing means a kernel `static_assert` (or the existing DRAM `is_dram` check) fired.

Also construct the usual Metal 2 object from the **same** token (`DataflowBuffer` / `Scratchpad<T>` / `LocalTensorAccessor<T>`). SPEC §8: `LLKOperand` is additional construction, not a replacement. Do **not** write the `LLKOperand<static_cast<DataFormat>(desc.format), desc.shape>` weld — that cast is LLK’s.

Prefer formats that do not hit the host→HW remap (`Float16_b`, `Float32`, tensor `BFLOAT16` → `Float16_b`) so the `static_assert` can compare against `DataFormat` enumerators.

### 3.2 Expected `TensorShape` (cookbook)

Same arithmetic SPEC §3.3 / JIT `compute_num_faces_rc_dims` already use. `face_c_dim` is always 16.

| Host input | Folded `TensorShape` `{face_r, face_c, nfaces_r, nfaces_c}` |
|---|---|
| Format only (no tile, no face) | `{16, 16, 2, 2}` — default 32×32 |
| Format + `Tile{{16, 32}}` | `{16, 16, 1, 2}` — wide |
| Format + `Tile{{32, 16}}` | `{16, 16, 2, 1}` — narrow |
| Format + `FaceGeometry{face_r_dim=1, num_faces=4}`, default tile | `{1, 16, 2, 2}` |

The wide vs narrow pair is why MeshTensor must go through `Tile`, not `FaceGeometry` / `tensor_shape_from_num_faces`. Flat `num_faces==2` cannot tell them apart.

### 3.3 Conversion succeeds and folds

Build host specs with the existing helpers (`MakeMinimalGen1ValidProgramSpec`, `MakeMinimalDFB`, `MakeShardedTensorParameter`, `BindTensorParameterToKernel`). For a non-default tensor tile, construct `PageConfig(Layout::TILE, Tile{{r,c}})` rather than using the helper’s default 32×32.

Scratchpad format-bearing mapping is already the §2.2 accept table. This section is DFB + MeshTensor (and DRAM in §3.4).

| Test | Token | Host | Kernel `static_assert` |
|---|---|---|---|
| `ToLlkMemDescriptorDFBDefaultTileCompiles` | `dfb::in` | `Float16_b`, no tile / face | `Float16_b`, `{16,16,2,2}` |
| `ToLlkMemDescriptorDFBFaceGeometryCompiles` | `dfb::in` | format + `{face_r_dim=1, num_faces=4}` | `{1,16,2,2}` |
| `ToLlkMemDescriptorL1TensorDefaultCompiles` | `tensor::a` | L1-sharded `BFLOAT16` 32×32 (`MakeShardedTensorParameter`) | `Float16_b`, `{16,16,2,2}` |
| `ToLlkMemDescriptorL1Tensor16x32Compiles` | `tensor::a` | L1 tensor, `Tile{{16,32}}` | `{16,16,1,2}` |
| `ToLlkMemDescriptorL1Tensor32x16Compiles` | `tensor::a` | L1 tensor, `Tile{{32,16}}` | `{16,16,2,1}` — **not** `{16,16,1,2}` |

Format-less `Scratchpad<T>(token)` is `ScratchpadComputeBindWithoutFormatSucceeds` (§2.2) / `ScratchpadAccessorBindingJITSmokeComputeKernel`. Do not add a conversion case for it.

### 3.4 Conversion refused

Only DRAM. Format-less scratchpad conversion is **UB** (SPEC §12) — do not add a compile-fail or a defined-success case for it. Host already covers illegal configs (§2.2).

| Test | Kernel | Host | Expect |
|---|---|---|---|
| `ToLlkMemDescriptorDramTensorFailsCompile` | `to_llk_mem_descriptor(tensor::a)` | interleaved DRAM `TensorParameter` (`MakeMinimalTensorParameter`, default `BufferType::DRAM`). Do **not** construct `LocalTensorAccessor` | existing `static_assert(!is_dram)` / `CompileProgram` throws |

Keep `HasSubstr` loose (`DRAM`). RISC-V diagnostics are not a stable API.

### 3.5 Kernel cache

If format / shape are baked onto the generated token (they must be, or the conversion cannot fold), they are part of `kernel_bindings_generated.h` and must change `compute_hash`. The file already does this for scratchpad **size**.

Add two cheap hash cases next to `DifferentScratchpadSizeProducesDifferentKernelHash`:

| Test | Differ by | Expect |
|---|---|---|
| `ScratchpadLlkFormatAffectsKernelHash` | same size; `Float16_b` vs `Float32` (or format vs none) | hashes differ |
| `DFBTileMetadataAffectsKernelHash` | same compute DFB; default tile vs `Tile{{16,32}}` | hashes differ |

“No metadata” must not hash equal to “format 0 + default 32×32”.

---

## 4. Out of this plan

- New test files, `unit_tests_llk`, enqueue, silicon, or experimental `copy_tile` / `pack_tile` / `make_llk_operand`.
- Address seams (`get_read_ptr` / `get_base_address` / `get_bank_base_address`) — already covered; SPEC does not specify pointer units.
- Format-less `to_llk_mem_descriptor` (UB). Do not expect a converter diagnostic.
- Reading `format` / `shape` / slot off the token (SPEC §8 — not a public API; a review check, not a gtest).
- Growing `unpack_modes` to Scratchpad / Tensor names.
- Deleting `DFBBindingToken::operator uint32_t()`.
- Host JIT remaps (Int16, Mx\*) and Bfp\* stride.
- Cross-node DFB; repo-wide DFB designated-init rename; nested `ComputeOperandMetadata`.
- A host-only `TensorSpec` → `{format, TensorShape}` table. Useful if a mapper helper lands; not required — §3.3 `static_assert`s already pin those numbers on the token.
