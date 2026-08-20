# Implementation — LLK operand metadata on Metal 2 BindingTokens

**Branch:** `riverwu/m2-neat`. Behavior: [`SPEC.md`](./SPEC.md). Context: [`BACKGROUND.md`](./BACKGROUND.md). Tests: [`tests.md`](./tests.md).

This file is the implementation plan for Parts I and II. It chooses the plumbing SPEC left open: how constexpr format and `TensorShape` land on each BindingToken, and which Metal 2 / filegen / LLK files change.

**Status:** plan. Not behavior spec. If a later choice disagrees with a SPEC rule, SPEC wins.

SPEC §12 / §14 (resolved): illegal LLK config is a **host** `MakeProgramFromSpec` reject. Format-less scratchpad conversion is **UB** — no device check (`HasLlk`, missing overload, converter `ASSERT`). Tokens stay untemplated value types. DRAM still fails via existing `args_t::is_dram`.

---

## 0. What we are building

A compute kernel must be able to write:

```cpp
DataflowBuffer in(dfb::in);
Scratchpad<uint32_t> pad(scratch::pad);
LocalTensorAccessor<uint32_t> a(tensor::a);

constexpr auto in_desc  = ckernel::experimental::to_llk_mem_descriptor(dfb::in);
constexpr auto pad_desc = ckernel::experimental::to_llk_mem_descriptor(scratch::pad);  // only if host filled LLK fields
constexpr auto a_desc   = ckernel::experimental::to_llk_mem_descriptor(tensor::a);     // L1 tensors only
```

`in_desc` / `pad_desc` / `a_desc` are folded `LLKMemDescriptor`s (L1 format + `TensorShape`). Address still comes from the memory object (`get_read_ptr` / `get_base_address` / `get_bank_base_address`). Welding descriptor + address into `LLKOperand<Format, Shape>` is LLK’s API (SPEC §10).

Today only DFB reaches LLK, and only because `DFBBindingToken::operator uint32_t()` is a CB slot that indexes `chlkc_descriptors.h`. Scratchpad and MeshTensor have address CRTAs and no compile-time format/shape on the token.

```
ProgramSpec (DFB / ScratchpadSpec / TensorParameter)
        │  host: validate + normalize to {hw_format, TensorShape parts}
        ▼
Kernel binding handles  (JitBuildSettings callbacks)
        │
        ▼
filegen: kernel_bindings_generated.h   ← Format/Shape baked onto the token
         chlkc_descriptors.h           ← unchanged; still needed for Gen1 CB-id APIs
        ▼
device:  to_llk_mem_descriptor(token)  ← folds; no slot lookup on the new path
```

Do **not** delete `set_dfb_data_fmt_and_tile`, `chlkc_descriptors.h`, or `operator uint32_t()`. Legacy compute still keys those arrays by slot.

---

## 1. Representation decision (SPEC §13)

SPEC allows NTTPs, generated constexpr members, or leftover `chlkc` indexing, as long as `to_llk_mem_descriptor(token)` folds.

**Choice: constexpr member variables on the existing value tokens. Filegen fills them in the constructor. `to_llk_mem_descriptor` reads the members, not `chlkc`.**

This is how `DFBBindingToken::id_` and `ScratchpadBindingToken::{crta_offset_, size_in_bytes_}` already work. Format and shape join that list. They are **not** NTTPs. SPEC §8: do not template DFB or Scratchpad tokens on Format, `HasLlk`, or anything else.

Device kernels are **C++17** (no `consteval`). That does not matter for omission: conversion without a format is UB (§1.6), not a `static_assert`.

| Token | Today | After |
|---|---|---|
| `DFBBindingToken` | Value `{id}` + `operator uint32_t()` | Same type. Extra ctor when host has format: `{id, {format, faces…}}`. Id-only ctor stays for DM-only / format-less DFBs. |
| `ScratchpadBindingToken` | Value `{crta_word, size}` | Same type. Extra ctor when host filled LLK fields. Two-arg ctor unchanged. |
| `TensorBindingToken<CTA, CRTA>` | Type NTTPs for layout + address only; `constexpr name{}` | **Keep those NTTPs** (address/layout seam). Add format + four face-grid **members**. DRAM uses existing `args_t::is_dram`. |

`LocalTensorAccessor` / `TensorAccessor` / `DataflowBuffer` / `Scratchpad<T>` ctors do not change shape. They already take the token by value / const-ref and ignore any extra members.

### 1.1 Why members fold

`dfb::in` is a `constexpr` object. A `constexpr` function that reads its members folds the same way `operator uint32_t()` already folds `id_`. `-ftt-nttp` is required on `LLKOperand<Format, Shape>`, not on the token that *produces* the descriptor.

```cpp
constexpr LLKMemDescriptor to_llk_mem_descriptor(DFBBindingToken token) {
    return {token.format_, TensorShape{token.face_r_dim_, token.face_c_dim_,
                                       token.num_faces_r_dim_, token.num_faces_c_dim_}};
}
// to_llk_mem_descriptor(dfb::in)  →  constant
```

### 1.2 Why not `ckernel::TensorShape` as the member type

Token headers are compiled for **data-movement** kernels too. They must not include `tt-llk/common/tensor_shape.h`.

Store the same four bytes as `uint8_t`s. The LLK-side overload builds `TensorShape`. `face_c_dim` is always 16 on supported tiles; store it anyway so the token is a complete face grid.

```text
format            : uint8_t   // HW encoding (see §3.3), not host DataFormat
face_r_dim        : uint8_t
face_c_dim        : uint8_t   // 16
num_faces_r_dim   : uint8_t
num_faces_c_dim   : uint8_t
```

### 1.3 Why not NTTPs

NTTPs would put format/shape in the **type**, which forces:

- templated `DFBBindingToken` / `ScratchpadBindingToken`
- extra NTTPs on `TensorBindingToken` and a ctor/CTAD churn on every accessor
- a packed `uint32_t` just to keep the template-head readable

Members do the same job with the types we already have. `operator uint32_t()` already proves a constexpr member on a constexpr token is enough.

### 1.4 Why not “DFB still indexes `chlkc`” as the end state

A Phase-2 `to_llk_mem_descriptor(DFBBindingToken)` that does `unpack_src_format[id]` is a few lines and already sketched next to `Cb<CbId>`. It does **not** help Scratchpad or MeshTensor (no slot), and it leaves DFB coupled to the CB-sized arrays.

This branch’s job is the runtime half of Phase 3: every Metal 2 compute operand that *can* be an LLK source carries its own constexpr facts. DFB gets the same pipe so the three objects look the same on the device. The `chlkc` path stays for Gen1 `copy_tile(cb_id, …)`.

### 1.5 Members stay private

SPEC §8: the kernel does not read `format` / `shape` / slot off the token. Members are private; `to_llk_mem_descriptor` is a friend. Reviews should reject `dfb::in.format_` at call sites (it should not compile).

A generated sidecar (`dfb::llk::in_format`) cannot be found from `to_llk_mem_descriptor(dfb::in)` without the token pointing at it.

### 1.6 Host rejects illegal config; format-less conversion is UB

SPEC §12 / §14. Do not implement a device-side presence check.

| Situation | Where it is caught |
|---|---|
| Compute DFB, no format | Host `TT_FATAL` (existing). Never JIT. |
| Geometry without format; unsupported format; bad `FaceGeometry`; face grid vs tile overflow | Host `MakeProgramFromSpec` (SPEC §14) |
| Scratchpad bound to compute, no LLK fields, `Scratchpad<T>(token)` | Legal. Working memory. |
| `to_llk_mem_descriptor(scratch::pad)` with no format | **UB.** Converter still matches and reads members (default `Invalid` / 0xFF + 32×32). No `HasLlk`, no `ASSERT`, no missing overload. Do not write a test that expects this to fail compile. |
| DRAM `to_llk_mem_descriptor` | Existing `static_assert(!args_t::is_dram)` — type-level, already there |

C++17 cannot `static_assert` on an instance member of a parameter. That is why SPEC chose UB instead of a device diagnostic for the scratchpad hole.

---

## 2. Device token shapes

Add private instance members + an extra constexpr constructor. Same untemplated types as today. Tensor keeps `CTA`/`CRTA` only.

### 2.1 Extract slim token headers (so LLK can include them)

`DFBBindingToken` currently lives in `dataflow_buffer.h` (pulls CB/DFB interfaces, Quasar overlay, NoC). `ScratchpadBindingToken` lives in `scratchpad.h` (pulls `CoreLocalMem`).

The converter needs to be a friend of the tokens. Including `dataflow_buffer.h` from an LLK header is too much. Split:

| New header | Holds | Included by |
|---|---|---|
| `tt_metal/hw/inc/api/dataflow/dfb_binding_token.h` | `DFBBindingToken` | `dataflow_buffer.h`, filegen, LLK conversion header |
| `tt_metal/hw/inc/api/scratchpad_binding_token.h` | `ScratchpadBindingToken` | `scratchpad.h`, filegen, LLK conversion header |
| `tt_metal/hw/inc/api/tensor/tensor_binding_token.h` | already slim | unchanged role |

`dataflow_buffer.h` / `scratchpad.h` include the new headers and stay the kernel-facing includes.

### 2.2 Shared member layout

A tiny POD in a header both tokens can include (or duplicate the five `uint8_t`s — do not invent a public kernel API):

```cpp
inline constexpr uint8_t kNoLlkFormat = 0xFF;

struct LlkOperandMembers {
    uint8_t format = kNoLlkFormat;
    uint8_t face_r_dim = 16;
    uint8_t face_c_dim = 16;
    uint8_t num_faces_r_dim = 2;
    uint8_t num_faces_c_dim = 2;
};
```

This is a device-side layout helper, not the host `LlkOperandFacts` (§3.3). It is a public **aggregate** so filegen can designated-initialize it. Filegen prints the field names; it does not include this header from the host.

### 2.3 `DFBBindingToken`

```cpp
struct DFBBindingToken {
    explicit constexpr DFBBindingToken(uint16_t id) noexcept : id_(id) {}
    constexpr DFBBindingToken(uint16_t id, LlkOperandMembers llk) noexcept : id_(id), llk_(llk) {}

    constexpr operator uint32_t() const noexcept { return id_; }

private:
    friend constexpr LLKMemDescriptor ckernel::experimental::to_llk_mem_descriptor(DFBBindingToken);

    uint16_t id_;
    LlkOperandMembers llk_{};
};
```

`DataflowBuffer(DFBBindingToken)` is unchanged.

Filegen today:

```cpp
constexpr DFBBindingToken in{3};
```

Filegen after, compute DFB with format (designated initializers, declaration order):

```cpp
constexpr DFBBindingToken in{3, {.format = 1u,
                                 .face_r_dim = 16u,
                                 .face_c_dim = 16u,
                                 .num_faces_r_dim = 2u,
                                 .num_faces_c_dim = 2u}};
```

DM-only / format-less DFB stays `constexpr DFBBindingToken in{3};` (members default to `kNoLlkFormat` + 32×32). Converting that token is UB.

### 2.4 `ScratchpadBindingToken`

```cpp
class ScratchpadBindingToken {
public:
    explicit constexpr ScratchpadBindingToken(uint32_t crta_offset, uint32_t size_in_bytes) noexcept;
    constexpr ScratchpadBindingToken(uint32_t crta_offset, uint32_t size_in_bytes, LlkOperandMembers llk) noexcept;

private:
    template <typename T> friend class Scratchpad;
    friend constexpr LLKMemDescriptor ckernel::experimental::to_llk_mem_descriptor(ScratchpadBindingToken);

    uint32_t crta_offset_;
    uint32_t size_in_bytes_;
    LlkOperandMembers llk_{};
};
```

`Scratchpad<T>(const ScratchpadBindingToken&)` does not change. Existing emission `constexpr ScratchpadBindingToken name{crta, size}` keeps compiling.

LLK scratchpad:

```cpp
constexpr ScratchpadBindingToken pad{4u, 2048u, {.format = 1u,
                                                .face_r_dim = 16u,
                                                .face_c_dim = 16u,
                                                .num_faces_r_dim = 2u,
                                                .num_faces_c_dim = 2u}};
```

### 2.5 `TensorBindingToken`

Layout/address NTTPs stay. LLK facts are members. Accessor ctors keep `TensorBindingToken<CTA, CRTA>` — no extra template parameters, no CTAD guide edits.

```cpp
template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
struct TensorBindingToken {
    using args_t = TensorAccessorArgs<CTA_OFFSET>;
    static constexpr args_t args{};
    static constexpr uint32_t addr_crta_offset = ADDR_CRTA_OFFSET;

    constexpr TensorBindingToken(LlkOperandMembers llk) noexcept : llk_(llk) {}

private:
    template <uint32_t C, uint32_t A>
    friend constexpr LLKMemDescriptor ckernel::experimental::to_llk_mem_descriptor(TensorBindingToken<C, A>);

    LlkOperandMembers llk_;
};
```

Every tensor token is constructed with facts (`TensorSpec` always has dtype + tile). DRAM tokens still carry them; conversion `static_assert`s `!args_t::is_dram`.

Filegen today:

```cpp
using a_t = ::tensor_accessor::TensorBindingToken<12u, 16u>;
constexpr a_t a{};
```

Filegen after:

```cpp
using a_t = ::tensor_accessor::TensorBindingToken<12u, 16u>;
constexpr a_t a{{.format = 1u,
                 .face_r_dim = 16u,
                 .face_c_dim = 16u,
                 .num_faces_r_dim = 2u,
                 .num_faces_c_dim = 2u}};
```

Default-construct (`a{}`) goes away: there is no empty-facts tensor token. That is the only emission-shape change that is not “add optional trailing args.”

Do not have `LocalTensorAccessor` / `TensorAccessor` read `llk_`.

---

## 3. Host: facts and validation (Part I)

### 3.1 `ScratchpadSpec` fields

Add the three DFB fields to `scratchpad_spec.hpp` with the same names, types, defaults, and comments (SPEC §3.2). Existing designated-init sites (~20) keep compiling.

Includes: `<optional>`, `tt_backend_api_types.hpp`, `tile.hpp`, `face_geometry.hpp`.

### 3.2 Validation in `program_spec.cpp` (SPEC §14)

Next to the existing DFB format checks (~L1654–1684). This is **illegal configuration** — `TT_FATAL` at `MakeProgramFromSpec`, not a device `static_assert`.

| Rule | Applies to |
|---|---|
| Compute endpoint and no format | DFB (existing). Do **not** copy this onto Scratchpad. |
| Format set but not `is_data_format_supported` for the arch | DFB (existing), Scratchpad |
| Tile or face set, no format | Scratchpad. DFB is already covered when a compute endpoint exists. |
| `FaceGeometry` invalid (`face_r_dim == 0` or `> FACE_HEIGHT`, `num_faces == 0`) | DFB, Scratchpad — same checks as `CircularBufferConfig::set_unpack_face_geometry` |
| Face grid does not fit the tile (same overflow as `compute_num_faces_rc_dims`) | DFB, Scratchpad, when both tile and face are set |

`Tile` construction already throws on an unsupported `{height, width}`. Do not add a second tile-shape table.

**Do not reject:** compute-bound scratchpad with no LLK fields; DM-only DFB with no format; format alone (default 32×32); TensorParameter (DRAM stays `is_dram` on device).

No TensorParameter field changes. No FaceGeometry on tensors.

### 3.3 Normalized compile-time facts (impl-only)

SPEC §2 allows a plumbing view type. Add a host-only helper, **not** on the public specs:

```cpp
// tt_metal/impl/metal2_host_api/llk_operand_facts.hpp  (or jit_build/ — host only)
struct LlkOperandFacts {
    uint8_t hw_format = 0;       // encoding LLK consumes
    uint8_t face_r_dim = 16;
    uint8_t face_c_dim = 16;
    uint8_t num_faces_r_dim = 2;
    uint8_t num_faces_c_dim = 2;
    bool present = false;
};

uint8_t host_data_format_to_hw(tt::DataFormat);  // extract from genfiles.cpp emit_formats_array

LlkOperandFacts facts_from_tile(tt::DataFormat host_fmt, const Tile& tile);
LlkOperandFacts facts_from_dfb(const DataflowBufferSpec&);
LlkOperandFacts facts_from_scratchpad(const ScratchpadSpec&);
LlkOperandFacts facts_from_tensor_spec(const TensorSpec&);
```

**Format remap.** `genfiles.cpp` already maps host `DataFormat` → HW codes (`Int16→9`, `MxFp4_2x_B→24`, `MxInt8→2`, `MxInt4→3`, `MxInt2→11`). Token emission must use the same function. Lift `host_data_format_to_hw` out of the anonymous namespace in `genfiles.cpp` so `chlkc` arrays and tokens cannot drift.

**`TensorShape` from a `Tile`** (SPEC §3.3) — same arithmetic as `compute_num_faces_rc_dims` in `genfiles.cpp` for one operand:

```text
face_r_dim      = tile.get_face_shape()[0]
face_c_dim      = 16
num_faces_c_dim = min(tile.get_width() / 16, tile.get_num_faces())
num_faces_r_dim = tile.get_num_faces() / num_faces_c_dim
```

Extract a single-element helper and call it from both `compute_num_faces_rc_dims` and `facts_from_tile` so JIT arrays and tokens agree.

**FaceGeometry override (DFB + Scratchpad only).** Reuse the effective-tile logic already in `JitBuildOptions::set_cb_data_fmt_tile_and_face_geometry` / `tile_from_unpack_face_geometry`:

- If override set: `face_r_dim` / `num_faces` from `FaceGeometry`; tile rows/cols from the derived or requested `Tile`.
- Then the same `num_faces_{r,c}_dim` split.
- Flat `num_faces==2` cannot distinguish 16×32 vs 32×16. That is accepted for DFB/scratch (SPEC §3.3). Tensors never take this path.

**Tensor path.**

```text
host_fmt = datatype_to_dataformat_converter(spec.data_type())
tile     = spec.tile()
facts    = facts_from_tile(host_fmt, tile)   // then remap to hw_format
```

No FaceGeometry. DRAM vs L1 is not a fact on `LlkOperandFacts`; the device constraint is `TensorAccessorArgs::is_dram`.

**Defaults when tile/face omitted** (DFB compute, or scratch with format only): `Tile{}` → 32×32 → `{16,16,2,2}`.

### 3.4 Where facts are attached to the kernel

Filegen only sees `JitBuildSettings` (the `Kernel`). It cannot look up `ProgramSpec`. Facts must already be on the binding handles.

| Handle | File | Change |
|---|---|---|
| DFB map `name → slot` | `kernel.hpp` `DataflowBufferBindingHandleMap` | Promote to a struct vector (like tensor/scratch): `{accessor_name, slot, LlkOperandFacts}`. `MakeDataflowBufferBindingHandles` fills facts from `DataflowBufferSpec`. |
| `ScratchpadBindingHandle` | `kernel.hpp` | Add `LlkOperandFacts` (or the packed fields). `ResolveScratchpadBindingsForKernel` copies from `ScratchpadSpec`. |
| `TensorBindingHandle` | `kernel.hpp` | Add `LlkOperandFacts`. `ResolveTensorBindingsForKernel` derives from `resolved_tensor_parameters[name].spec` (the `TensorParameter.spec` already used for CTA payload). |

`LlkOperandFacts` on the handle is optional-in-spirit: `present==false` for format-less DFB/scratch. Tensor always `present==true`.

### 3.5 `JitBuildSettings` callbacks

Today (`jit_build_settings.hpp`):

```cpp
process_dataflow_buffer_binding_handles(name, logical_dfb_id)
process_scratchpad_binding_handles(name, size_bytes, addr_crta_word)
process_tensor_binding_handles(name, cta, addr_crta, num_rt_words)
```

Extend each callback with the five integers + `present` (`hw_format`, `face_r_dim`, `face_c_dim`, `num_faces_r_dim`, `num_faces_c_dim`). Default empty implementations stay; `Kernel` overrides pass the new fields.

Callers to update together (signature must match or they will not compile):

- `tt_metal/jit_build/genfiles.cpp` — `write_kernel_bindings_generated_header`
- `tt_metal/impl/kernels/kernel.cpp` — `process_*` + `compute_hash`
- `tt_metal/impl/emulation/emulated_program_runner.cpp` — `build_metal2_snapshot` + `emit_metal2_namespaces` (this file **reimplements** the generated header; it must stay text-equivalent)

### 3.6 Kernel cache hash

`Kernel::compute_hash` must include anything that appears in `kernel_bindings_generated.h`.

- DFB: today hashes `name + slot`. Add `present` + the five integers. (Slot-indexed `hlk_desc` is already in `stable_hash_hlk_desc`; the token bake is a second copy of the same facts and still has to be in this hash.)
- Scratchpad: today hashes `name + size + crta_word`. Add facts.
- Tensor: today hashes `name + cta + crta + num_rt_words`. Add facts. `tensor_parameter_name` stays omitted (not in the header).

Hash `present` as well as the integers, so “no metadata” ≠ “format 0 + default 32×32”.

---

## 4. Filegen

### 4.1 `write_kernel_bindings_generated_header`

`tt_metal/jit_build/genfiles.cpp` (~L105–241).

Collect the extra fields in the existing `dfb_entries` / `ScratchEntry` / `TaEntry` structs. Emission is extra constructor arguments, not a new type.

Emit `LlkOperandMembers` with **designated initializers**, fields in declaration order (C++20 designated-init rule; gcc accepts this as a C++17 extension, which is what the device toolchain uses):

```cpp
std::string emit_llk_members(const LlkOperandFacts& f) {
    return fmt::format(
        "{{.format = {}u, .face_r_dim = {}u, .face_c_dim = {}u, "
        ".num_faces_r_dim = {}u, .num_faces_c_dim = {}u}}",
        f.hw_format, f.face_r_dim, f.face_c_dim, f.num_faces_r_dim, f.num_faces_c_dim);
}
```

Do **not** emit a positional `{1, 16, 16, 2, 2}` — a field reorder would silently swap geometry.

**DFB**

```cpp
if (facts.present) {
  content << "constexpr DFBBindingToken " << name << "{" << id << ", " << emit_llk_members(facts) << "};\n";
} else {
  content << "constexpr DFBBindingToken " << name << "{" << id << "};\n";
}
```

**Scratchpad** — two-arg ctor if `!present`; `{crta, size, emit_llk_members(facts)}` if present. Same type either way.

**Tensor** — alias stays `TensorBindingToken<CTA, CRTA>`. Always emit `constexpr name_t name{emit_llk_members(facts)};` (never `name{}`).

Determinism: keep today’s sort (DFB/sem by name; tensor/scratch in declaration order). New integers are per-entry; no extra sort key.

### 4.2 Do not emit Format/Shape into `chlkc_descriptors.h`

`generate_all_descriptors` / `set_dfb_data_fmt_and_tile` stay as they are. Scratchpad and MeshTensor still do **not** get a CB slot and still do not appear in `unpack_src_format[]`.

### 4.3 Emulation twin

`emulated_program_runner.cpp` `emit_metal2_namespaces` (~L888–917) must emit the same tokens. Extend `Metal2BindingsSnapshot` the same way as the callbacks. If emulation lags, Metal 2 kernels that call `to_llk_mem_descriptor` will not compile under emule.

---

## 5. LLK-side additions (alongside, not inside, tt-llk)

Ops already take `LLKOperand`. We only add converters.

### 5.1 New header

```
tt_metal/hw/inc/api/compute/experimental/2_0/binding_token_llk.h
```

Responsibilities:

- `#include` the three slim token headers + `llk_mem_descriptor.h`
- friend access to each token’s `LlkOperandMembers`
- assemble `TensorShape` from the four `uint8_t`s
- DRAM: keep the existing `static_assert(!args_t::is_dram)`
- **No** format-present check. Reading a format-less scratchpad/DFB token is UB (SPEC §12).

```cpp
namespace ckernel::experimental {

constexpr LLKMemDescriptor llk_desc_from_members(LlkOperandMembers m) {
    return LLKMemDescriptor{
        m.format,
        TensorShape{m.face_r_dim, m.face_c_dim, m.num_faces_r_dim, m.num_faces_c_dim}};
}

constexpr LLKMemDescriptor to_llk_mem_descriptor(DFBBindingToken token) {
    return llk_desc_from_members(token.llk_);
}

constexpr LLKMemDescriptor to_llk_mem_descriptor(ScratchpadBindingToken token) {
    return llk_desc_from_members(token.llk_);
}

template <uint32_t CTA, uint32_t CRTA>
constexpr LLKMemDescriptor to_llk_mem_descriptor(
    tensor_accessor::TensorBindingToken<CTA, CRTA> token) {
    static_assert(
        !tensor_accessor::TensorBindingToken<CTA, CRTA>::args_t::is_dram,
        "to_llk_mem_descriptor: DRAM TensorBindingToken has no node-local L1 region");
    return llk_desc_from_members(token.llk_);
}

}  // namespace ckernel::experimental
```

Replace the “Future source accessors” comment in `llk_mem_descriptor.h` with `#include "binding_token_llk.h"`. Prefer that include so the PR kernel pattern (`#include "…/llk_mem_descriptor.h"`) picks up the overloads. Slim token headers keep this from pulling `dataflow_buffer.h` into every TRISC compile.

Do **not** add `make_llk_operand` here (SPEC §10: LLK’s sugar). Do **not** add `LLKMemDescriptor(token)` converting constructors (SPEC §13).

### 5.2 `uint8_t` vs `DataFormat`

`LLKMemDescriptor::format` is `uint8_t`. `LLKOperand` is parameterized by `enum class DataFormat`. The `static_cast<DataFormat>(desc.format)` in `eltwise_copy_fp8_2_0.cpp` stays LLK’s problem. Metal 2 kernels on this branch stop at the descriptor.

### 5.3 Address seam (not in the converter)

| Object | Existing getter | Notes |
|---|---|---|
| `DataflowBuffer` | `get_read_ptr()` / `get_write_ptr()` | Byte / arch units. Experimental `cb_read_address` uses **16-byte words** (`fifo_rd_ptr - 1 + page * i`). **Verify units before the first id-free DFB kernel.** If they differ, add a small helper next to the converter (e.g. tile address from the DFB object) — do not change FIFO credit APIs. Quasar tile-counter / strided L1 is a different story; first tests are Blackhole, same as #53193. |
| `Scratchpad<T>` | `get_base_address()` | Byte address. Kernel adds tile offset. |
| `LocalTensorAccessor<T>` | `get_bank_base_address()` | Byte address. Same. |

SPEC does not define Bfp\* extra exponent bytes or multi-tile pages. Experimental ops still assume page == one linear tile. Compute DFB `entry_size` is typically one tile; do not “fix” stride in this work.

---

## 6. What we are not doing

From SPEC §5 / §13, plus plumbing we considered and rejected:

- Nested public `ComputeOperandMetadata` on the specs
- `unpack_face_geometry_metadata` on `TensorParameter`
- Requiring format because a compute kernel binds a scratchpad
- Deleting `DFBBindingToken::operator uint32_t()`
- Growing `ComputeHardwareConfig::unpack_modes` beyond DFB names (still slot-indexed via `BuildUnpackToDestModeVector`)
- Dropping `set_dfb_data_fmt_and_tile` / `chlkc_descriptors.h`
- Host JIT register-format derivation for the new operands (LLK `data_format_derive.h` on the experimental path)
- Wormhole / Quasar copies of the experimental ops (stay Blackhole-first, same as #53193)
- TTNN factory migration
- Changing `Scratchpad<T>`’s `T` to mean `tt::DataFormat`
- Format / shape / `HasLlk` as NTTPs on DFB or Scratchpad tokens (SPEC §8 / §13)
- Device-side check for format-less conversion (`ASSERT`, missing overload, `static_assert` on members) — that call is UB
- `consteval` (device is C++17; unused anyway)

---

## 7. Suggested slices

PRs from this branch target `riverwu/m2-neat-base`. Tests live in [`tests.md`](./tests.md) (`test_program_spec.cpp` only).

### Process — every slice

Before moving on, each slice **must**:

1. **Compile-check.** Host-side: build `unit_tests_api` (or the `test_program_spec` objects) so `MakeProgramFromSpec` / new gtests actually compile. After filegen / device headers land (B, C), also confirm a Metal 2 compute kernel still JIT-compiles (existing `ScratchpadAccessorBindingJITSmokeComputeKernel` or the new Gen1 cases).
2. **Commit.** One commit for the slice. Message says why (host fields, host tests, token bake, converter).
3. **Push.** `git push` the branch (`-u` the first time). Do not force-push `main` / `master`.

Do not start the next slice until those three are done.

### Slice A — Host config only (no new tests yet)

- `scratchpad_spec.hpp` fields
- `program_spec.cpp` validation (SPEC §3.2 / §14)
- Existing designated-init sites keep compiling — no edits expected

No token / filegen / `to_llk_mem_descriptor`. Then compile-check, commit, push.

### Slice A.tests — Host-side tests ([`tests.md`](./tests.md) §2.1 + §2.2 rejects + format-less accept)

Implement **only** what compiles against Slice A (no converter, no baked token facts):

- `ScratchpadSpec` `hashable_v` / `is_aggregate_v` + designated-init living-doc case
- Quasar `MakeProgramFromSpec` **rejects**: Scratchpad unsupported format / tile-or-face without format / invalid `FaceGeometry` / face-grid overflow; DFB invalid `FaceGeometry` / face-grid overflow
- `ScratchpadComputeBindWithoutFormatSucceeds` — `MakeProgramFromSpec` (+ existing compute `Scratchpad<T>(token)` smoke; do **not** call `to_llk_mem_descriptor`)

**Hold until C** (need filegen + converter + `CompileProgram` fold):

- §2.2 format-bearing accepts (`ScratchpadFormatAloneSucceeds`, `…AndTile…`, `…AndFaceGeometry…`)
- §3.3 DFB / L1 tensor fold cases
- §3.4 DRAM `is_dram`
- §3.5 hash cases (`ScratchpadLlkFormatAffectsKernelHash`, `DFBTileMetadataAffectsKernelHash`)

Then compile-check `unit_tests_api` / `test_program_spec`, commit, push.

### Slice B — Facts helper + handle plumbing + filegen

- `LlkOperandFacts` + `host_data_format_to_hw` + single-operand face-grid helper (share with `compute_num_faces_rc_dims`)
- Handle structs, `Resolve*BindingsForKernel`, `MakeDataflowBufferBindingHandles`
- Callback + `compute_hash` + `genfiles.cpp` (designated `LlkOperandMembers`) + emulation snapshot
- Device token extra ctors + private `LlkOperandMembers` + slim header split. **No** `HasLlk`. Accessor / `DataflowBuffer` / `Scratchpad<T>` signatures unchanged
- Existing Metal 2 compute JIT smokes still compile. Do not call `to_llk_mem_descriptor` yet

Then compile-check, commit, push.

### Slice C — `to_llk_mem_descriptor` + remaining tests

- `binding_token_llk.h` + include from `llk_mem_descriptor.h`
- Remaining [`tests.md`](./tests.md) cases: §2.2 format-bearing Scratchpad accepts, §3.3 DFB/tensor folds, §3.4 DRAM, §3.5 hashes
- Do **not** test format-less conversion (UB). No `unit_tests_llk` / experimental ops / silicon

Then compile-check (host gtests + Gen1 `CompileProgram` cases), commit, push.

### Order inside Slice B if it needs to split

Each sub-step still gets compile-check + commit + push:

1. Slim token headers (NFC move; existing emission still works)
2. Facts helper + remap extraction
3. Handles + callbacks + hash + filegen + emulation
4. Extra token ctors + private members (codegen starts designated-init `LlkOperandMembers`)

---

## 8. File checklist

### Host API / validation

| File | Change |
|---|---|
| `tt_metal/api/tt-metalium/experimental/metal2_host_api/scratchpad_spec.hpp` | Three optional LLK fields |
| `tt_metal/impl/metal2_host_api/program_spec.cpp` | Scratchpad validation; attach facts in `MakeDataflowBufferBindingHandles` / `ResolveScratchpadBindingsForKernel` / `ResolveTensorBindingsForKernel` |
| `tt_metal/impl/metal2_host_api/llk_operand_facts.hpp` (+ `.cpp` if needed) | **New.** Normalized facts, face-grid, tensor/DFB/scratch entry points |

### Kernel / JIT / filegen

| File | Change |
|---|---|
| `tt_metal/impl/kernels/kernel.hpp` | Handle structs grow facts; DFB map → vector of structs |
| `tt_metal/impl/kernels/kernel.cpp` | `process_*` callbacks; `compute_hash` |
| `tt_metal/jit_build/jit_build_settings.hpp` | Callback signatures |
| `tt_metal/jit_build/genfiles.cpp` | Extra ctor args on tokens; extract `host_data_format_to_hw`; share face-grid helper |
| `tt_metal/impl/emulation/emulated_program_runner.cpp` | Snapshot + `emit_metal2_namespaces` |
| `tt_metal/jit_build/jit_build_options.cpp` | Optional: move `tile_from_unpack_face_geometry` next to facts helper so DFB/scratch/CB share one override |

### Device tokens / accessors

| File | Change |
|---|---|
| `tt_metal/hw/inc/api/dataflow/dfb_binding_token.h` | **New.** Untemplated `DFBBindingToken` + `LlkOperandMembers` |
| `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h` | Include slim header; ctor unchanged |
| `tt_metal/hw/inc/api/scratchpad_binding_token.h` | **New.** Extra LLK ctor |
| `tt_metal/hw/inc/api/scratchpad.h` | Include slim header; ctor unchanged |
| `tt_metal/hw/inc/api/tensor/tensor_binding_token.h` | Extra ctor + `LlkOperandMembers`; CTA/CRTA NTTPs unchanged |
| `tt_metal/hw/inc/api/tensor/local_tensor_accessor.h` | **No change** |
| `tt_metal/hw/inc/api/tensor/tensor_accessor.h` | **No change** |

### LLK-adjacent

| File | Change |
|---|---|
| `tt_metal/hw/inc/api/compute/experimental/2_0/llk_mem_descriptor.h` | Include conversion header; drop “future overload” comment |
| `tt_metal/hw/inc/api/compute/experimental/2_0/binding_token_llk.h` | **New.** Overloads |
| `tt_metal/tt-llk/` | **No change** unless LLK later adds `make_llk_operand` |

### Tests

Placement and cases are specified in [`tests.md`](./tests.md). Summary:

| File | Change |
|---|---|
| `tests/tt_metal/tt_metal/api/metal2_host_api/test_program_spec.cpp` | **A.tests:** §2.1 + §2.2 rejects + format-less accept. **C:** §2.2 format-bearing accepts, §3.3–§3.5 ([`tests.md`](./tests.md)) |

Do not add new test files, `sources.cmake` entries, `test_kernels/` sources, or cases under `tests/tt_metal/tt_metal/llk/`.

Grep after Slice B for leftover `process_dataflow_buffer_binding_handles(` — the callback signature change is the complete host call-site list. Device accessor signatures do not change.

---

## 9. Risks and open implementation notes

1. **DFB handle storage.** `dataflow_buffer_binding_handles_` is `unordered_map<string,uint16_t>`. `compute_hash` sorts that map by key. A vector-of-structs must hash in a **stable** order (sort by accessor name, like today) even if filegen also sorts. Do not hash in declaration order unless filegen stops sorting DFB names.

2. **UB vs host reject.** Do not add a converter check for missing format. Cover the hole with SPEC §14 host tests. Only DRAM keeps a device `static_assert`.

3. **Address units.** First DFB id-free kernel must compare `get_read_ptr()` to `cb_read_address(id)`. If they disagree, add a helper; do not paper over it in the test kernel.

4. **Host `DataFormat` vs HW.** Forgetting the remap on the token is a silent wrong-format bug for Int16 / Mx\*. Share one function with `chlkc` emission.

5. **Emulation drift.** Any filegen change that is not copied into `emit_metal2_namespaces` will only fail under emule.

6. **`unpack_modes`.** Still DFB-name → slot vector. Scratchpad / tensor LLK operands do not get an unpack-mode entry in this work. Experimental ops treat unpack-to-dest as the kernel-wide `UnpackToDestEn` constexpr.

7. **Quasar.** Token emission is arch-agnostic. Experimental ops and the first device test are Blackhole. Quasar DFB addressing is out of scope.

8. **Borrowed DFB.** Format/shape still come from `DataflowBufferSpec`, not from the borrowed `TensorParameter`. A future “unpack a subset of a tensor tile” is a DFB with its own face override (SPEC §3.3), which this pipe already supports.

---

## 10. Done when

- ScratchpadSpec can declare the three fields; validation matches SPEC §3.2 / §14; old sites compile.
- A Metal 2 compute kernel can `to_llk_mem_descriptor` a compute DFB token, an LLK-tagged scratchpad token, and an L1 tensor token, and the results fold (no `chlkc[id]` on those overloads).
- Format-less scratchpad conversion is undocumented UB (no device check). DRAM still `static_assert(!is_dram)`.
- `DataflowBuffer` / `Scratchpad` / `LocalTensorAccessor` construction is unchanged for existing kernels.
- `chlkc_descriptors.h` and `operator uint32_t()` still feed Gen1 CB-id compute.
- Kernel cache hashes the baked facts.
- Emulation emits the same tokens as JIT filegen.
