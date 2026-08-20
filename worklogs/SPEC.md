# Spec — LLK operand metadata on Metal 2 memory objects

**Branch:** `riverwu/m2-neat`. Background: [`BACKGROUND.md`](./BACKGROUND.md).

This file is the spec of work on this branch. It is written section by section.

**Status:** Part I (host configuration) and Part II (device: BindingToken → descriptor) specify the behavior. Implementation plumbing (how constexpr format/`TensorShape` land on the token) and LLK’s `LLKOperand` welding are out of this spec, not unspecified behavior.

---

# Part I — Host configuration

## 0. Scope of this part

Host configuration only: how `ProgramSpec` declares the **compile-time** half of an LLK operand (`DataFormat` + tile / face geometry) on each of the three memory objects a compute kernel can be given.

Not in this section: BindingToken NTTPs, headergen, `chlkc_descriptors.h`, address seams, experimental compute APIs, `unpack_modes`.

The device contract this configuration must eventually feed (from #53193):

```cpp
template <DataFormat Format, TensorShape Shape>
struct LLKOperand {
    std::uint32_t l1_address;  // runtime only
};
```

Host must be able to produce `Format` and `Shape` from each object **without** going through a CB/DFB slot id.

---

## 1. The three facts (unchanged meaning)

Copied from `DataflowBufferSpec` (`dataflow_buffer_spec.hpp` lines 92–111). These are LLK facts, not FIFO facts.

| Field | Type | When required | Default if omitted |
|---|---|---|---|
| `data_format_metadata` | `optional<tt::DataFormat>` | Required if this object is an LLK operand | none — LLK cannot default a format |
| `tile_format_metadata` | `optional<Tile>` | Optional | default 32×32 tile |
| `unpack_face_geometry_metadata` | `optional<FaceGeometry>` | Optional override | derive faces from `Tile` (`face_r_dim` + `num_faces`) |

`FaceGeometry` is **not** a second tile type. `Tile` is the storage tile. `FaceGeometry` is set only when the operand occupies **less than a full tile** (fewer faces and/or shorter faces), so unpack/pack read exactly that much. Same override `CircularBufferConfig::set_unpack_face_geometry` already has.

**“Has LLK metadata” means `data_format_metadata` is set.** Tile and face are optional refinements of that format, not a second presence bit.

**Host knowledge differs by object.** These fields describe an LLK operand. Whether `MakeProgramFromSpec` can *require* a format depends on whether the host can tell that the object will be one:

- **DFB + compute endpoint.** A compute DFB *is* an LLK operand (today the token is the CB slot LLK indexes). Format is always required. Tile/face without format is not a separate DFB reject: no format already fails. A DM-only DFB is not an LLK source; format (and geometry) may be omitted.
- **Scratchpad + compute bind.** Binding a scratchpad does not mean LLK will ingest it — it is often just working memory — so the host cannot require a format. Geometry without a format is still illegal: that combination claims LLK geometry but omits the one field LLK cannot default. Calling `to_llk_mem_descriptor` with no format is UB (§12), not a host reject.

| `data_format_metadata` | `tile_format_metadata` | `unpack_face_geometry_metadata` | Host | Device `to_llk_mem_descriptor` |
|---|---|---|---|---|
| unset | unset | unset | Legal for Scratchpad and DM-only DFB. **Reject** for DFB with a compute endpoint | **UB** if `to_llk_mem_descriptor` is called (§12). `Scratchpad<T>` / DM DFB still construct. |
| unset | set | any | **Reject** on Scratchpad (geometry without format). DFB: no extra rule — compute already requires format; DM-only is not an LLK source | — |
| unset | any | set | **Reject** on Scratchpad (geometry without format). DFB: same as the row above | — |
| set | unset | unset | Legal. Tile defaults to 32×32, faces from that tile | Folded descriptor |
| set | set | unset | Legal. Faces from that `Tile` | Folded descriptor |
| set | unset | set | Legal. Tile defaults to 32×32; face is the unpack override (tilize tests do this) | Folded descriptor |
| set | set | set | Legal iff the face grid fits the tile (§14) | Folded descriptor |

FIFO sizing (`entry_size`, `num_entries`), credits, borrowed L1, and Scratchpad `size_per_node` stay where they are. They are not LLK metadata.

---

## 2. Decision: no dedicated public struct on the specs (yet)

A shared `struct ComputeOperandMetadata { DataFormat; Tile; FaceGeometry; }` nested into `DataflowBufferSpec` would be the tidy API.

**We will not do that in this work.** Reasons:

- `DataflowBufferSpec` is initialized almost everywhere with designated initializers: `.data_format_metadata = …`. There are hundreds of those sites (TTNN factories, Metal 2 tests).
- C++20 designated initialization does not flatten inherited or nested members. Nesting or inheriting the three fields forces `.llk = {.data_format_metadata = …}` (or `.ComputeOperandMetadata{…}`) at every DFB site.
- The fields’ meaning is already documented on `DataflowBufferSpec`. Copying the same three names onto the one spec that is missing them is cheaper than a repo-wide DFB refactor.

A **normalized view type for runtime plumbing** is allowed later (a function that returns `{format, tile, face}` from a DFB / Scratchpad / TensorParameter). That is an impl detail, not a host-API change, and is not required to land the fields.

If a later cleanup wants the public struct, it is a mechanical rename of DFB sites and is out of this spec.

---

## 3. Per-object host configuration

### 3.1 DFB — leave the three fields alone

`DataflowBufferSpec` already carries them. No layout change.

Existing validation stays:

- If any endpoint is a compute kernel, `data_format_metadata` is required (`program_spec.cpp`).
- If set, the format must be arch-supported.
- `tile_format_metadata` / `unpack_face_geometry_metadata` remain optional.

DFB credits and FIFO sizing are still DFB’s job. This spec does not move them.

### 3.2 Scratchpad — add the same three fields

`ScratchpadSpec` today is `{unique_id, size_per_node}`. It has no format or geometry. A compute kernel can already bind a scratchpad as raw L1 (`Scratchpad<T>`), and that use must keep working **without** filling LLK metadata.

Add the three fields with the **same names, types, defaults, and comments** as DFB:

```cpp
struct ScratchpadSpec {
    ScratchpadSpecName unique_id;
    uint32_t size_per_node = 0;

    // LLK operand metadata. Only needed when this scratchpad is used as an LLK operand.
    std::optional<tt::DataFormat> data_format_metadata = std::nullopt;
    std::optional<tt::tt_metal::Tile> tile_format_metadata = std::nullopt;
    std::optional<FaceGeometry> unpack_face_geometry_metadata = std::nullopt;
};
```

**Validation (stricter than “copy DFB’s compute-endpoint rule”):**

| Rule | Why it differs from DFB |
|---|---|
| Fields optional by default | Scratchpad is often just working memory. Existing compute kernels that bind a scratchpad and never call LLK on it must not start failing. |
| If `data_format_metadata` is set, it must be arch-supported | Same check as DFB. |
| If either tile or face-geometry is set, `data_format_metadata` is required | Geometry without a format cannot build `LLKOperand`. |
| Do **not** require format merely because a compute kernel binds the scratchpad | Binding ≠ LLK operand. DFB requires format on any compute endpoint because today’s DFB token *is* the CB slot LLK indexes. Scratchpad is not. |

Existing Scratchpad designated-init sites (~20) keep compiling: new members default to `nullopt`.

`Scratchpad<T>`’s template argument stays a C++ view type (`uint32_t`, `int32_t`, …). It is not `tt::DataFormat` and is not a substitute for `data_format_metadata`.

### 3.3 MeshTensor / TensorParameter — no new fields; derive everything from `TensorSpec`

`TensorParameter.spec` already has format **and** face geometry. Do not add `data_format_metadata`, `tile_format_metadata`, or `unpack_face_geometry_metadata`.

| LLK fact | Already on `TensorSpec` / `Tile` | Mapping |
|---|---|---|
| L1 `DataFormat` | `spec.data_type()` | `datatype_to_dataformat_converter` |
| tile rows/cols | `spec.tile().get_height()` / `get_width()` | `Tile` ctor from `{tile_r, tile_c}` |
| `face_r_dim` | `spec.tile().get_face_shape()[0]` | `TILE_FACE_HW_CHOICES` in `tile.cpp` |
| `face_c_dim` | `spec.tile().get_face_shape()[1]` | always 16 on supported tiles |
| `num_faces` | `spec.tile().get_num_faces()` | `tile_hw / face_hw` |
| narrow / partial | `get_narrow_tile()` / `get_partial_face()` | width < 32 / height < 32 |

`TensorShape` for the operand is the same decomposition JIT already does in `compute_num_faces_rc_dims` from a `Tile` (no FaceGeometry override):

```text
format          = datatype_to_dataformat_converter(spec.data_type())
face_r_dim      = tile.get_face_shape()[0]
face_c_dim      = 16
num_faces_c_dim = min(tile.get_width() / 16, tile.get_num_faces())
num_faces_r_dim = tile.get_num_faces() / num_faces_c_dim
```

That is a **better** geometry source than DFB’s `FaceGeometry`. `FaceGeometry` is a flat `{face_r_dim, num_faces}` pair and cannot tell 16×32 (1×2 faces) from 32×16 (2×1). `Tile` stores both dimensions, so `narrow_tile` is not lost.

**Why DFB still has a FaceGeometry override, and TensorParameter does not:** a DFB entry is untyped FIFO storage. Callers use `unpack_face_geometry_metadata` when the page is **not** a `Tile` — pool windows packed onto compact pages, conv scalar `{face_r_dim=1, num_faces=4}`, tilize tests that leave `tile_format_metadata` unset and only tag shortened faces. A MeshTensor page **is** `spec.tile()`. Unpacking a tensor as an LLK operand means unpacking that tile. There is no second layout to override.

If some future kernel wanted to unpack a *subset* of a tensor tile, that would be a DFB-shaped operand (borrowed DFB over the tensor, with its own face override), not a property of `TensorParameter`.

**Constraints:**

- This path is only meaningful for tensors that have a node-local L1 region (`LocalTensorAccessor`: L1-sharded or interleaved L1). DRAM tensors remain a compile-time error as LLK sources.
- `DataType` is a **subset** of `tt::DataFormat`. Formats with no `DataType` (non-`_b` Float16, Mx\*, …) cannot be declared as MeshTensor LLK operands. That is a pre-existing TensorSpec limit; this spec does not add a `DataFormat` override to punch through it.
- `datatype_to_dataformat_converter` is the encoding LLK will consume from a tensor. Host JIT remaps (Int16, Mx\*) still apply later, in the token/JIT section — not here.

---

## 4. How the three objects then look (host)

| Object | Format | Tile / TensorShape | Face-geometry override | Required when |
|---|---|---|---|---|
| **DFB** | `data_format_metadata` | `tile_format_metadata` (opt, default 32×32) | `unpack_face_geometry_metadata` (FIFO page ≠ tile) | Compute endpoint → format required (existing) |
| **Scratchpad** | `data_format_metadata` | `tile_format_metadata` (opt, default 32×32) | `unpack_face_geometry_metadata` | Only if used as LLK operand; not required on mere compute bind |
| **MeshTensor** | derived from `spec.data_type()` | derived from `spec.tile()` (incl. faces) | **none** — tile *is* the geometry | Always present for a valid TensorSpec; usable as LLK source only if L1-local |

No object needs a CB/DFB slot to *declare* these facts. (Getting them onto the device token is the next section.)

---

## 5. Out of this part

- How Format/Shape are represented on the BindingToken (NTTP vs sidecar). That is codegen, not the kernel-facing contract — see Part II for the contract.
- Whether Scratchpad/Tensor skip `chlkc_descriptors.h` entirely.
- Growing `ComputeHardwareConfig::unpack_modes` beyond DFB names.
- Refactoring DFB’s three fields into a nested struct.

---

## 6. Open (host-config only)

None. Scratchpad bound to compute with no format is **legal config**. Calling `to_llk_mem_descriptor` on that token is **UB** (§12). Host rejection of illegal configs is §14.

---

# Part II — Device: BindingToken → LLKOperand

## 7. Scope of this part

The kernel-facing contract: how a compute kernel turns a Metal 2 BindingToken into the compile-time half of an `LLKOperand`.

Not in this part: how headergen / JIT puts constexpr format and `TensorShape` onto the token (NTTP vs generated sidecar, `chlkc_descriptors.h`, CTA vs CRTA). Those are plumbing. This part only says what the kernel is allowed to *do* with a token.

## 8. BindingToken stays an opaque handle

A BindingToken is not a thing the kernel inspects. It is a handle the host names (`dfb::in`, `scratch::pad`, `tensor::a`) and the kernel **constructs an object from**.

That is already the Metal 2 rule:

```cpp
DataflowBuffer in(dfb::in);
Scratchpad<uint32_t> pad(scratch::pad);
LocalTensorAccessor<uint32_t> a(tensor::a);
```

The token types document this: “the user will never directly interact with this type.” `ScratchpadBindingToken` keeps members private and friends `Scratchpad`. That is the model.

**Do not template DFB or Scratchpad tokens** on Format, `HasLlk`, or anything else to carry LLK facts or to refuse conversion. They stay the value types they are today. `TensorBindingToken<CTA, CRTA>` keeps those two NTTPs — they are the address/layout seam, not an LLK tax. How format/`TensorShape` are stored (members vs sidecar) is plumbing; the kernel-facing type of `dfb::in` / `scratch::pad` must not grow a template head.

What a kernel must **not** do with a token:

- Read `format` / `shape` / slot id off it as a public API.
- Pass it into LLK as a CB id (`DFBBindingToken::operator uint32_t()`). That conversion exists for legacy Gen1 `copy_tile(cb_id, …)` and is not the id-free path.

Credits stay on `DataflowBuffer`. Indexed L1 access stays on `Scratchpad` / `LocalTensorAccessor`. Those objects are still constructed from the same token. `LLKOperand` is an *additional* construction, not a replacement.

## 9. What #53193 already did (CB)

The experimental compute API consumes `LLKOperand<Format, Shape>` — Format/Shape as NTTPs, `l1_address` as the only runtime member.

CB cannot go through a runtime `CircularBuffer` id (that cannot fold). The PR adds a type-level CB accessor plus a conversion:

```cpp
template <uint32_t CbId>
struct Cb { … };   // compile-time identity in the type

constexpr LLKMemDescriptor to_llk_mem_descriptor(Cb<CbId>);
```

Kernel pattern on the PR (`eltwise_copy_fp8_2_0.cpp`) converts the descriptor into `LLKOperand<DataFormat, Shape>` with a `static_cast<DataFormat>(in_desc.format)`. That cast exists because `LLKMemDescriptor::format` is `uint8_t` (copied from the `chlkc` arrays) while `LLKOperand` is parameterized by `enum class DataFormat`. **That mismatch is LLK’s.** This branch does not design around it and does not ask Metal 2 kernels to write the cast.

`to_llk_mem_descriptor` is the conversion. It returns the compile-time half (`LLKMemDescriptor` = L1 format + `TensorShape`). Address is a separate runtime seam.

The header already sketches additive overloads for `DFBBindingToken` and `ScratchpadBindingToken`. That is the hook this branch fills. `TensorBindingToken` is the third.

`Cb<CbId>` is a CB-only stepping stone (Phase 2). Metal 2 kernels do not grow a parallel `Cb`-like wrapper; they already have BindingTokens.

## 10. Construct the descriptor from the token

For each Metal 2 token that carries constexpr format + `TensorShape`, add an overload:

```cpp
constexpr LLKMemDescriptor to_llk_mem_descriptor(DFBBindingToken);
constexpr LLKMemDescriptor to_llk_mem_descriptor(ScratchpadBindingToken);
constexpr LLKMemDescriptor to_llk_mem_descriptor(TensorBindingToken<…>);
```

This is the same function the PR already has for `Cb<CbId>`. Same meaning: token in, folded descriptor out. The kernel does not care how the facts got onto the token.

The conversion takes the **token**, not `DataflowBuffer` / `Scratchpad` / `LocalTensorAccessor`. Those objects hold a runtime address (and DFB a runtime slot). A runtime value cannot be an NTTP; folding needs compile-time identity, which the token has.

Kernel-facing construction on this branch is: token → descriptor, address from the memory object. How LLK turns that descriptor into `LLKOperand<DataFormat, Shape>` (including any `uint8_t` ↔ `DataFormat` cast) is their API.

```cpp
DataflowBuffer in(dfb::in);   // credits, FIFO pointers — as today
constexpr auto in_desc = to_llk_mem_descriptor(dfb::in);
// in_desc is the compile-time half. Address from `in`. LLK welds them into LLKOperand.
```

A helper that returns an `LLKOperand` (`make_llk_operand(dfb::in, addr)` or equivalent) is LLK’s to provide if they want to hide the `using` / cast. Ops still take `LLKOperand`.

## 11. Address is not in the conversion

`to_llk_mem_descriptor` does not return an address. `LLKOperand::l1_address` is runtime and per-tile.

Address comes from the object already constructed from the same token:

| Source | Object constructed from token | L1 tile address |
|---|---|---|
| DFB | `DataflowBuffer` | FIFO read/write pointer (plus tile index, same idea as `cb_read_address` / `cb_write_address`) |
| Scratchpad | `Scratchpad<T>` | `get_base_address()` (plus byte/tile offset the kernel chooses) |
| MeshTensor (L1) | `LocalTensorAccessor<T>` | `get_bank_base_address()` (plus tile offset the kernel chooses) |

Exact pointer arithmetic (16-byte words, page stride, Bfp\*) is not specified here. The rule is: **descriptor from the token, address from the memory object.**

DRAM tensors: `LocalTensorAccessor` is already a compile-time error. No `to_llk_mem_descriptor` for a DRAM `TensorBindingToken`.

## 12. Conversion without a format — undefined behavior

**Config vs use.** Binding a format-less scratchpad to a compute kernel is legal (working memory). Calling `to_llk_mem_descriptor` on that token is **undefined behavior**. DFB bound to compute without format never reaches this: host already `TT_FATAL`s.

Do **not** add a device-side check (`HasFormat`, missing overload, converter `ASSERT`). Tokens stay untemplated. LLK does not reject an `Invalid` descriptor (it can program as `Bfp2_b` via the 4-bit format mask). This work does not define a fail for that.

Unused `chlkc` slots for CB/DFB that compute never converts stay `Invalid` as today. We are not changing that path.

DRAM tensors: `LocalTensorAccessor` is already a compile-time error via `args_t::is_dram`. Keep that; it is type-level and already exists. `to_llk_mem_descriptor` on a DRAM token should fail the same way.

| Call | Format omitted | What happens |
|---|---|---|
| Compute DFB, no format | yes | **Host** `TT_FATAL` — never JIT |
| Compute scratchpad, no format, `Scratchpad<T>(token)` | yes | Succeeds |
| `to_llk_mem_descriptor(scratch::pad)`, no format | yes | **UB** |
| `to_llk_mem_descriptor`, format set | n/a | Folded descriptor (defaults in §1) |
| `to_llk_mem_descriptor(tensor::a)` DRAM | n/a | Fail via existing `is_dram` |

## 13. Out of this part

- Representation of Format/Shape on the token (members vs sidecar vs leftover `chlkc` index). Any of those is fine as long as `to_llk_mem_descriptor(token)` folds **and the token type is not templated on those facts**.
- `template <bool HasLlk>` / device presence check for forgotten format. Rejected — format-less conversion is UB (§12).
- Whether `DFBBindingToken::operator uint32_t()` is deleted; it stays until legacy CB-id compute APIs are gone.
- Per-tile stride, Bfp\* page size, `unpack_modes`.
- `LLKMemDescriptor::format` being `uint8_t` vs `LLKOperand` taking `DataFormat`. LLK owns that round-trip.
- Explicit `LLKMemDescriptor(token)` converting constructor (optional sugar next to `to_llk_mem_descriptor`). Parked; not part of this spec.

## 14. When the host rejects an LLK config

Reject at `MakeProgramFromSpec` (same place as today’s DFB format checks). This is **illegal configuration**. Device-side conversion of a format-less token is UB (§12), not a second host check.

**Reject:**

| Rule | Applies to |
|---|---|
| Compute endpoint and no `data_format_metadata` | DFB (existing) |
| `data_format_metadata` set but not `is_data_format_supported` for the arch | DFB (existing), Scratchpad |
| `tile_format_metadata` or `unpack_face_geometry_metadata` set, but no format | Scratchpad only (§1). DFB does not get this check: a compute DFB already requires format; a DM-only DFB is not an LLK operand. |
| `unpack_face_geometry_metadata` present and invalid (`face_r_dim == 0` or `> FACE_HEIGHT`, `num_faces == 0`) | DFB, Scratchpad — same checks as `CircularBufferConfig::set_unpack_face_geometry` |
| Face grid does not fit the tile (same overflow check as `compute_num_faces_rc_dims`) | DFB, Scratchpad, when both tile and face are set |

`Tile` construction already throws on an unsupported `{height, width}`. Do not add a second Tile-shape table.

**Do not reject:**

- Scratchpad bound to a compute kernel with no LLK fields (working memory).
- DFB with no compute endpoint and no format.
- Format alone (no tile, no face) — default 32×32. This **is** giving LLK parameters.
- TensorParameter — no extra LLK fields; DRAM-as-LLK-source stays the existing `is_dram` device check (§12).
- Growing `unpack_modes` to scratchpad/tensor names (out of this spec).

Host rejection is `TT_FATAL` / exception at program construction, not a device compile `static_assert`.
