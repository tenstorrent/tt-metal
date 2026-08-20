# Background — Metal 2 runtime, LLK, and operand metadata

**What this file is:** context freeze for later work on `riverwu/m2-neat`. It is **not** a spec of that work. Tasks for this branch have not been specified yet.

**Branch:** `riverwu/m2-neat`, based on `origin/riverwu/m2-neat-base` @ `eaf65b9` (tip of [PR #53193](https://github.com/tenstorrent/tt-metal/pull/53193) / `rtawfik/l1spec-datacopy`). PRs from this branch should target `riverwu/m2-neat-base`, not `main`.

**LLK tracking:** [issue #53456](https://github.com/tenstorrent/tt-metal/issues/53456). Inspection worktree (same commit): `/workspace/.claude/worktrees/l1spec-datacopy`.

**Sources:** Metal 2 host API under `tt_metal/api/tt-metalium/experimental/metal2_host_api/`; experimental id-free compute API on #53193 (`tt_metal/hw/inc/api/compute/experimental/2_0/`).

---

## 1. The coupling

Compute kernels call Low-Level Kernel (LLK) primitives (`copy_tile`, `pack_tile`, tilize/untilize, …) that sit as close to Tensix assembly as the stack gets.

Those primitives need two kinds of facts about each operand:

| Kind | Examples | When it is known |
|---|---|---|
| Compile-time metadata | data format, tile geometry (faces, rows, tile size) | Program construction / JIT |
| Runtime location | L1 (SRAM) tile address | Program execution |

Today both facts are smuggled through the Circular Buffer (CB) interface. A single integer CB id indexes JIT-generated `unpack_*` / `pack_*` arrays **and** resolves the L1 address via the CB FIFO (`fifo_rd_ptr` / `fifo_wr_ptr` / `fifo_page_size`).

That couples compute to circular buffers. Compile-time facts cannot fold independently of which CB slot the operand landed in. Non-CB L1 objects cannot be LLK operands without first being stuffed into a CB-shaped slot.

LLK’s stated direction ([#53456](https://github.com/tenstorrent/tt-metal/issues/53456), [PR #53193](https://github.com/tenstorrent/tt-metal/pull/53193)): an **id-free operand** — format + geometry as compile-time NTTPs so they fold/DCE, L1 address as the only runtime member — and no requirement that the source be a CB.

**Constexpr metadata is a perf requirement.** A runtime `unpack_src_format[id]` lookup is what they are retiring.

---

## 2. Metal 2 host model

### 2.1 Vocabulary

A **node** is a NOC endpoint in the accelerator grid (historically called a “core” in the overloaded sense). RISC-V processors *inside* a node are cores. `NodeCoord` is an x,y address (`node_coord.hpp`).

A **kernel** is `kernel_main()` running on a node’s baby RISC-V cores. It is either **compute** or **data-movement**. LLK is the compute-side layer. A `KernelSpec` is one compiled specialization of a kernel source (compile-time args, bindings, compiler options, hardware config). One source may appear as several KernelSpecs. At enqueue, one **kernel instance** runs on each node the kernel is placed on (SPMD: same binary, per-node runtime args).

A **WorkUnitSpec** is a set of kernels that run together on a set of nodes. Placement of kernels (and of node-local resources bound to them) is derived from WorkUnit membership.

A **Program** is the host object that is compiled and enqueued to the device — a piece of executable work. It is built from a `ProgramSpec`.

| | `ProgramSpec` | `ProgramRunArgs` |
|---|---|---|
| Analogy | Function signature + body | Call arguments |
| When | Declared once | Specified per enqueue |
| Contents | Kernels, program-scope resources, tensor *declarations*, work units | Kernel RTAs, MeshTensor *values*, optional DFB size overrides |

`MakeProgramFromSpec` / `SetProgramRunArgs` are the construction and parameterization APIs (`program.hpp`). Runtime args for Metal 2 Programs go through those, not the legacy host APIs.

**Program-scope resources** (allocated for this Program’s execution, from node-local L1): DFBs, semaphores, scratchpads.

**User-managed resources** (lifetime owned by the caller): MeshTensors, declared as `TensorParameter`s and bound at enqueue.

Compute kernels cannot have semaphore bindings. Semaphores are not LLK data operands.

### 2.2 Three memory objects passed to the device

These are the objects a kernel can be given. Each is named in the ProgramSpec, bound on a `KernelSpec`, and appears on device as a BindingToken in generated `kernel_bindings_generated.h`.

| Object | Host spec | Bound how | Device token / accessor | Lifetime | What it is |
|---|---|---|---|---|---|
| **DFB** | `DataflowBufferSpec` | `DFBBinding` producer/consumer | `dfb::<name>` → `DataflowBuffer` | Program-scope | Software FIFO between a producer kernel instance and a consumer kernel instance on the same node |
| **MeshTensor** | `TensorParameter` (layout) + `MeshTensor` at enqueue | `TensorBinding` | `tensor::<name>` → `TensorAccessor` / `LocalTensorAccessor` | User-managed (RAII) | Owning handle to allocated device memory laid out by `TensorSpec` |
| **Scratchpad** | `ScratchpadSpec` | `ScratchpadBinding` | `scratch::<name>` → `Scratchpad<T>` | Program-scope | Private uninitialized node-local L1 working memory; no sync |

**DFB.** Entry size × num entries in that node’s L1. Invariant: exactly one producer instance and one consumer instance per node (multiple KernelSpecs on one endpoint are legal if node sets don’t overlap and binding-site parameters match). Optional `data_format_metadata` / `tile_format_metadata` / `unpack_face_geometry_metadata` exist *only* to feed LLK when a compute kernel is an endpoint — a software FIFO type owning facts that belong to the compute operand. Credits (`wait_front` / `reserve_back` / `pop_front` / `push_back`) stay on DFB; they are not LLK metadata. May borrow L1 from a `TensorParameter` (`borrowed_from`). Cross-node DFB is sketched in the API and **not supported**. Hardware: Gen1 has a fixed DFB-per-node cap; Gen2’s cap depends on endpoint configuration (tile counters).

**MeshTensor.** Sole owner of its device allocation (`mesh_tensor.hpp`). Non-copyable, movable. `TensorParameter.spec` is the required `TensorSpec` (logical shape, `DataType`, layout, `Tile`, memory config). The actual `MeshTensor` is supplied in `ProgramRunArgs.tensor_args` and must match (relaxations exist). Compute on a node uses `LocalTensorAccessor` — the node-local L1 shard, or the local L1 region of an interleaved L1 tensor. DRAM tensors have no node-local L1 region; constructing `LocalTensorAccessor` from them is a compile-time error. `TensorSpec` already carries dtype + tile. Token address is a CRTA (per-enqueue base).

**Scratchpad.** `size_per_node` bytes of raw L1, private to the bound kernel instance (multiple KernelSpecs may share a ScratchpadSpec only on disjoint node sets). No format, no tile, no producer/consumer, no synchronization. Unpack/Math/Pack on a compute kernel are different RISC-V cores; multi-threaded kernels likewise — races are the author’s problem. Token carries CRTA word index + compile-time size.

All three can be L1 sources for compute. Only DFB currently reaches LLK, and only because its token **is** a CB slot. MeshTensor and Scratchpad already have address seams (CRTA).

### 2.3 BindingTokens (how the kernel names the object)

Headergen (`jit_build/genfiles.cpp` → `kernel_bindings_generated.h`):

| Token | Generated as | Carries today |
|---|---|---|
| `DFBBindingToken` | `constexpr DFBBindingToken name{id};` in `namespace dfb` | Compile-time slot id (implicit CTA). `operator uint32_t()` for Gen1 LLK APIs that still take a CB id |
| `TensorBindingToken<CTA, CRTA>` | `using name_t = TensorBindingToken<…>; constexpr name_t name{};` in `namespace tensor` | Type-level NTTPs: static layout CTA offset + base-address CRTA byte offset. Pattern for putting more constexpr facts on the type without touching kernel source |
| `ScratchpadBindingToken` | `constexpr ScratchpadBindingToken name{crta_word, size_bytes};` in `namespace scratch` | CRTA word index of base address + compile-time size |

DFB/semaphore ids were chosen as implicit CTAs (cheaper, can miss kernel cache if ids churn). Tensor base address is the first binding category to use implicit CRTAs (per-enqueue). Folding of Format/Shape needs those facts in the **type** or as constexpr members; a runtime id cannot be an NTTP.

### 2.4 Layers

```
Host runtime
  ProgramSpec / ProgramRunArgs → Program
    DataflowBufferSpec, TensorParameter (+ MeshTensor at enqueue), ScratchpadSpec
        │  compile-time: format, tile, face geometry  (keyed by CB/DFB slot today)
        │  runtime:     L1 base / FIFO pointers       (keyed by the same slot today)
        ▼
JIT (`tt_metal/jit_build/genfiles.cpp`)
  emits chlkc_descriptors.h  —  unpack_src_format[id], unpack_tile_r_dim[id], …
  emits kernel_bindings_generated.h — dfb:: / tensor:: / scratch:: tokens
        ▼
Device kernel
  BindingTokens  ──today DFB only──►  uint32_t id   (MeshTensor / Scratchpad: address only)
        ▼
Compute API (`tt_metal/hw/inc/api/compute/`)
  copy_tile(cb_id, …), pack_tile(…, cb_id)
  and, on #53193, experimental::copy_tile(LLKOperand, …)
        ▼
CKernels wrappers (`tt_metal/hw/ckernels/{arch}/metal/llk_api/`)
  get_operand_src_format(id) → unpack_src_format[id]
  get_local_cb_interface(id).fifo_wr_ptr  → L1 address
        ▼
LLK library (`tt_metal/tt-llk/`)
  unpack / math / pack, plus per-format Tensix register writes
```

Metal integration stack (LLK docs): Layer 1 tt-llk → Layer 2 ckernels `llk_api` → Layer 3 compute API → Layer 4 TTNN kernels.

---

## 3. Current metadata bus (the hack)

### 3.1 One id, two jobs

`copy_tile(in_cb_id, in_tile_index, dst_tile_index)` uses `in_cb_id` as:

1. **Metadata key.** `llk_unpack_A_init` / `llk_pack_hw_configure` index:

   - `unpack_src_format[id]`, `unpack_dst_format[id]`
   - `pack_src_format[id]`, `pack_dst_format[id]`
   - `unpack_tile_{r,c}_dim[id]`, `unpack_tile_num_faces[id]`, `unpack_tile_face_r_dim[id]`, `unpack_tile_size[id]`, …
   - derived `TensorShape` via `get_operand_tensor_shape(id)`

   Those arrays are constexpr, emitted per kernel into `chlkc_descriptors.h`. They are sized to `max_cbs` (architecture circular-buffer limit), not to how many operands this kernel actually uses.

2. **Address key.** Pack/unpack resolve the L1 tile from the CB/DFB hardware interface:

   - unpack: read pointer + page size × tile index
   - pack: `get_local_cb_interface(id).fifo_wr_ptr` + `fifo_wr_tile_ptr` / `fifo_page_size`

   Experimental LLK APIs in #53193 drop FIFO-page-size / write-pointer assumptions and treat page size as one tile. Host still *has* those FIFO fields because CB/DFB is still the address oracle.

### 3.2 Host sources of the metadata

**Legacy CB** — `CircularBufferConfig`

- per-index `data_formats_`
- per-index `tiles_` (`set_tile_dims`)
- per-index `unpack_face_geometry_` (`set_unpack_face_geometry`) — `FaceGeometry` is rows-per-face + number of faces, used when an entry is not a full tile

**Metal 2 DFB** — `DataflowBufferSpec`

- `data_format_metadata` — required if any compute kernel is an endpoint
- `tile_format_metadata` — optional; default 32×32 tile
- `unpack_face_geometry_metadata` — optional; shorter/fewer faces than a full tile
- plus FIFO sizing (`entry_size`, `num_entries`) — DFB’s real job

**MeshTensor** — `TensorParameter.spec` already has `DataType` + `Tile`. Not copied into `hlk_desc` or onto the token.

**Scratchpad** — size only.

### 3.3 Runtime plumbing today

At Program compile (`ProgramImpl::compile`):

1. `set_cb_data_fmt_and_tile` walks CBs on the kernel’s cores and writes `tt_hlk_desc` arrays keyed by `CBIndex`.
2. `set_dfb_data_fmt_and_tile` (`dataflow_buffer.cpp`) does the same for DFBs: **`dfb->device_slot` is cast to `CBIndex`**. Format-less DFBs (no compute consumer) are skipped so a slot is not filled from `DataFormat::Invalid`.
3. `BuildUnpackToDestModeVector` (`program_spec.cpp`) translates Metal 2 `unpack_modes[DFBSpecName]` into a `max_cbs`-long vector indexed by the same device slot. The comment there calls this DFB/CB translation layer “confusing” and notes it applies on Wormhole, Blackhole, **and Quasar**.
4. JIT `compute_data_formats` / `generate_all_descriptors` (`jit_build/genfiles.cpp`) derives **register** formats from the CB-indexed L1 formats (`get_unpack_src_formats`, `get_pack_src_formats`, …) and emits `chlkc_descriptors.h`. Host currently owns that derivation (exponent family, fp32 dest-acc, unpack-to-dest, Mx formats, Fp8 special cases). `emit_formats_array` remaps some host `DataFormat` enums to HW values (Int16, Mx*).

Device kernels then receive `constexpr DFBBindingToken name{slot};`. Compute still passes that token into APIs that expect a CB id, via `DFBBindingToken::operator uint32_t()`.

`unpack_modes` (UnpackToSrc vs UnpackToDest) is a compute-kernel property of an operand. Unpack-to-dest into a 16-bit dest is illegal for 32-bit formats; Gen1 also forbids it for narrower formats on perf grounds. Kernel-wide `UnpackToDestEn` / `DST_ACCUM_MODE` are JIT constexprs, not per-operand on `LLKOperand`.

Kernel cache identity today hashes every CB slot via `stable_hash_hlk_desc` (format + tile r/c dim per slot, plus math fidelity / approx).

### 3.4 Who can be an LLK operand today

| Object | Device token | Can LLK use it as an operand today? |
|---|---|---|
| Circular Buffer (legacy) | raw `uint32_t` / `CBIndex` | Yes — this *is* the key |
| **DFB** | `DFBBindingToken` | Yes, only because the token **is** a CB slot |
| **MeshTensor** | `TensorBindingToken` → `LocalTensorAccessor` (L1) | Address only (CRTA). `TensorSpec` has dtype + tile but they are not on the token / not in `hlk_desc` |
| **Scratchpad** | `ScratchpadBindingToken` | Address only (CRTA + size). No format/geometry anywhere |

---

## 4. What `LLKOperand` is (PR #53193 @ `eaf65b9`)

File: `tt_metal/hw/inc/api/compute/experimental/2_0/llk_mem_descriptor.h`. Namespace `ckernel::experimental`. **Blackhole-only** so far (`#ifdef ARCH_BLACKHOLE` on the ops). Wormhole/Quasar copies of `data_format_derive.h` are called out as future. Shared `llk_*_impl` cores with the legacy CB-id wrappers.

### 4.1 The two types

```cpp
struct LLKMemDescriptor {
    std::uint8_t format;  // L1 buffer format (unpacker reads / packer writes)
    TensorShape shape;    // 4 bytes: face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim
};

template <DataFormat Format, TensorShape Shape>
struct LLKOperand {
    std::uint32_t l1_address;  // ONLY runtime member — absolute L1 tile base
    static constexpr LLKMemDescriptor descriptor =
        LLKMemDescriptor{static_cast<std::uint8_t>(Format), Shape};
};
```

| Half | Where it lives | Why |
|---|---|---|
| **What** — L1 `DataFormat` + `TensorShape` | NTTPs on `LLKOperand<Format, Shape>` → `::descriptor` | Fold/DCE (`-ftt-nttp`). A runtime value cannot be an NTTP |
| **Where** — absolute L1 address | `l1_address` member | Known only at enqueue / FIFO walk |

Register formats are **not** on the operand. Arch LLK derives them from `DESC.format` + `DST_ACCUM_MODE` in `data_format_derive.h` (`infer_unpack_dst_format` / `infer_pack_src_format`). The compute API never sees a register format. That file mirrors the *scalar cores* of host `jit_build/data_format.cpp`, scoped to datacopy (single operand); Mx formats are absent from the BH enum; the host’s different-exponent-width pack branch is unreachable for a single-operand op.

`TensorShape` (`tt_metal/tt-llk/common/tensor_shape.h`) is a 4-byte packed NTTP:

```text
face_r_dim        // 1/2/4/8/16; full face is 16
face_c_dim        // always 16 for HW (MAX_FACE_C_DIM)
num_faces_r_dim   // face grid rows
num_faces_c_dim   // face grid cols
```

Default 32×32 tile = `{16, 16, 2, 2}`. Tile rows = `face_r_dim * num_faces_r_dim`; tile cols = `face_c_dim * num_faces_c_dim`. Host `Tile` + `FaceGeometry` map into this; JIT already computes `num_faces_{r,c}_dim` in `compute_num_faces_rc_dims`. Helper `tensor_shape_from_num_faces` maps flat `num_faces==2` to 1×2 (wide), never 2×1 (narrow) — Metal paths that know real row/col dims should not use it.

Not on `LLKOperand`:

- CB / DFB id
- Register formats (`unpack_dst_format`, `pack_src_format`)
- FIFO `page_size` / `wr_ptr` / `rd_ptr` as op state — ops use **absolute** addressing. Per-tile stride folds from `SCALE_DATUM_SIZE(Format, Shape.total_tensor_size()) >> 4` (assumes page size == one linear tile; **wrong for Bfp\*** which have extra exponent bytes, and for padded / multi-tile pages). `block==1` tests are unaffected
- `unpack_to_dest` — still a kernel-wide JIT constexpr (`UnpackToDestEn`), not per-operand

### 4.2 How a kernel builds one today (CB source)

From `eltwise_copy_fp8_2_0.cpp`:

```cpp
constexpr auto in_cb = experimental::Cb<tt::CBIndex::c_0>{};
constexpr auto in_desc = experimental::to_llk_mem_descriptor(in_cb);  // folds: indexes chlkc arrays
using InOp = experimental::LLKOperand<static_cast<DataFormat>(in_desc.format), in_desc.shape>;

experimental::copy_tile(InOp(in_cb.read_address()), /*dst=*/0);
experimental::pack_tile(OutOp(out_cb.write_address()), /*from_dst=*/0);
```

`Cb<CbId>` is a **type-level** accessor (`id` is an NTTP). Folding needs compile-time source identity in the type; a runtime `CircularBuffer` id cannot fold.

`to_llk_mem_descriptor(Cb<CbId>)` still reads `unpack_src_format[CbId]` / `pack_dst_format[CbId]` and the matching `*_tile_face_r_dim` / `*_num_faces_{r,c}_dim` arrays. Host equalizes unpack-src and pack-dst to the same L1 format, so either thread yields the same descriptor. Thread-partitioned arrays: unpack_* on UNPACK/MATH, pack_* on PACK.

Address seam (CB specialization):

```cpp
cb_read_address(cb_id, tile_index)  // fifo_rd_ptr - 1 + fifo_page_size * tile_index
cb_write_address(cb_id, tile_index) // fifo_wr_ptr - 1 + fifo_page_size * tile_index
```

16-byte-word convention (`- 1` on the FIFO pointer). Absolute (out-of-order) addressing: the op packs/unpacks at exactly this address. Legacy `pack_tile` auto-advances an internal FIFO tile pointer; experimental `pack_tile` does **not** — the caller supplies the per-tile address every call.

Experimental kernels still call `compute_kernel_hw_startup(cb_id, …)` and still use `CircularBuffer` for credits (`wait_front` / `reserve_back`). Credits and LLK metadata are already split at the call site; only the op itself is id-free.

### 4.3 Ops on the PR (Blackhole)

| Family | Headers | Notes |
|---|---|---|
| Datacopy | `tile_move_copy.h`, `pack.h` | `copy_tile` / `pack_tile` + inits |
| Tilize | `tilize.h` | `tilize_init` / `tilize_block` / `tilize_uninit`; block loop owns Dest sync; output stride from `SCALE_DATUM_SIZE` |
| Untilize | `pack_untilize.h` | Needs id-free `llk_pack_reconfig_data_format` because `_llk_pack_init_` does not program packer format registers (datacopy/tilize skipped this because `hw_startup` pre-set the CB format) |
| Eltwise binary | `eltwise_binary.h` | add/sub/mul. **Format-free at the op**: formats programmed at `compute_kernel_hw_startup`; op forwards geometry from operand A + two L1 addresses. No `data_format_derive`. Pack via experimental `pack_tile` |

Tests: differential vs legacy (id-free output bit-identical). Binary reuses shipping `eltwise_binary.cpp` as the baseline. Tilize/untilize keep dedicated `*_legacy.cpp` because the shipping tilize/pack_untilize kernels are Metal 2 DFB kernels (heavier harness). Verified on Blackhole silicon (`LLKBlackholeSingleCardFixture`).

Math still takes `LLKMemDescriptor` (format + geometry) even though MATH never touches L1; register format is derived from `DESC.format`. DEST-only math should not get a fake SRAM address.

### 4.4 What each BindingToken would have to expose for `to_llk_mem_descriptor`

The LLK header sketches additive overloads, gated on each source exposing **constexpr format + TensorShape**:

```cpp
constexpr LLKMemDescriptor to_llk_mem_descriptor(DFBBindingToken token);
constexpr LLKMemDescriptor to_llk_mem_descriptor(ScratchpadBindingToken token);
// TensorBindingToken / LocalTensorAccessor likewise
```

Facts the descriptor + operand need, vs what each object has on the host today:

| Fact | When | DFB | MeshTensor (L1 / `LocalTensorAccessor`) | Scratchpad |
|---|---|---|---|---|
| L1 `DataFormat` | constexpr | `data_format_metadata` | `TensorParameter.spec.data_type()` (map to `DataFormat`) | **missing** on `ScratchpadSpec` |
| `TensorShape` | constexpr | `tile_format_metadata` + `unpack_face_geometry_metadata` | `TensorParameter.spec.tile()` (+ face geometry if needed) | **missing** |
| L1 tile address | runtime | DFB FIFO ptr | tensor CRTA (node-local shard / L1 region) | scratch CRTA |

Today’s `DFBBindingToken` is only `{id}` + `operator uint32_t()`. That is enough to index `chlkc` (same idea as `Cb<CbId>`). It is not enough to drop those arrays. MeshTensor/Scratchpad have no `chlkc` slot at all.

`TensorBindingToken<CTA, CRTA>` is the type-NTTP pattern if Format/Shape become NTTPs on the token. MeshTensor is the object whose host spec already has dtype + tile; the token does not carry them yet.

---

## 5. LLK’s published phases ([#53456](https://github.com/tenstorrent/tt-metal/issues/53456))

This is the LLK team’s plan, not a task list for `riverwu/m2-neat`.

- **Phase 0 / 1 (in flight on #53193).** `LLKOperand` / `LLKMemDescriptor`. Ops take them. Shared `llk_*_impl`. `to_llk_mem_descriptor(Cb<CbId>)` + `cb_read/write_address` as the CB-only translator. Register formats derived in-LLK. FIFO page size / wr_ptr purged from experimental ops. Phase 0 notes: tilize/untilize/binary/datacopy added; experimental APIs assume page == one tile; JIT-inferred register formats stopped for those APIs (no obvious perf hit, incomplete format-combination coverage). Experimental APIs still share impl with legacy, so flag cleanups cannot ride along.

- **Phase 2.** One `to_llk_mem_descriptor(constexpr CB id)`: constexpr CB id → folded descriptor (thread-aware chlkc lookup). CB-only stepping stone. Already sketched as `Cb<CbId>`. A `DFBBindingToken` overload that uses the token’s constexpr id to index the same arrays is the same idea. Scratchpad / LocalTensorAccessor still cannot be LLK sources.

- **Phase 3.** Additive `to_llk_mem_descriptor` overloads on BindingTokens — no raw ids. Header comments name `DFBBindingToken` and `ScratchpadBindingToken`; `TensorBindingToken` / `LocalTensorAccessor` are the third memory object. Ops/call sites that already take `LLKOperand` would not change. Requires each token to expose constexpr Format + `TensorShape`.

---

## 6. Unresolved in the current code / LLK PR

Not a work plan. Things the existing design leaves open:

1. **Token shape.** Format + Shape as NTTPs on the token type (like `TensorBindingToken<…>`), or a value token plus generated sidecar (`dfb::in` has `::format` / `::shape`)? Folding needs them constexpr and visible to `to_llk_mem_descriptor`.

2. **Scratchpad Format/Shape.** MeshTensor already has dtype + tile on `TensorParameter.spec`. Scratchpad has only `size_per_node`. Grow `ScratchpadSpec`, or kernels write `LLKOperand<Format, Shape>(addr)` at the call site?

3. **Page size = tile size.** Experimental ops derive stride from Format+Shape via `SCALE_DATUM_SIZE`. Wrong for Bfp* and padded / multi-tile pages. Compute DFB `entry_size` is typically one tile.

4. **Host `DataFormat` vs HW encoding.** JIT remaps some host enums (Int16, Mx*) when emitting `chlkc` arrays. Whatever is baked into a token must be the encoding LLK actually consumes.

5. **Quasar / Wormhole.** #53193 is Blackhole-only. Slot-indexed JIT arrays exist on Quasar too. Quasar DFB hardware (tile counters, strided L1) is a different address story than `get_local_cb_interface`.

6. **`DFBBindingToken::operator uint32_t()`.** Still the bridge to every Gen1 compute API that takes a CB id.

7. **Math vs SRAM.** Datacopy math takes `LLKMemDescriptor` without an L1 address. Binary is format-free at the op. DEST-only math should not get a fake SRAM descriptor.

---

## 7. Key files

| Role | Path |
|---|---|
| **`LLKOperand` / `LLKMemDescriptor` / `to_llk_mem_descriptor` / `Cb`** | `tt_metal/hw/inc/api/compute/experimental/2_0/llk_mem_descriptor.h` (on this branch via `m2-neat-base`; not on `main`) |
| Id-free compute ops | `tt_metal/hw/inc/api/compute/experimental/2_0/{tile_move_copy,pack,tilize,pack_untilize,eltwise_binary}.h` |
| BH register-format derive | `tt_metal/hw/ckernels/blackhole/metal/llk_api/data_format_derive.h` |
| Id-free LLK wrappers (BH) | `tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/2_0/` |
| `TensorShape` | `tt_metal/tt-llk/common/tensor_shape.h` |
| Metal 2 ProgramSpec / KernelSpec / WorkUnitSpec | `tt_metal/api/tt-metalium/experimental/metal2_host_api/program_spec.hpp`, `kernel_spec.hpp` |
| Metal 2 Program + RunArgs | `…/program.hpp`, `…/program_run_args.hpp` |
| DFB / Scratchpad / TensorParameter host specs | `…/dataflow_buffer_spec.hpp`, `…/scratchpad_spec.hpp`, `…/tensor_parameter.hpp` |
| MeshTensor | `tt_metal/api/tt-metalium/tensor/mesh_tensor.hpp` |
| `TensorSpec` (dtype, tile, layout) | `tt_metal/api/tt-metalium/tensor/spec/tensor_spec.hpp` |
| Node vs core | `…/node_coord.hpp` |
| Legacy CB host spec | `tt_metal/api/tt-metalium/circular_buffer_config.hpp` |
| ProgramSpec validation + unpack-mode translation | `tt_metal/impl/metal2_host_api/program_spec.cpp` |
| DFB slot → `hlk_desc` | `tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp` (`set_dfb_data_fmt_and_tile`) |
| CB → `hlk_desc` | `tt_metal/impl/program/program.cpp` (`set_cb_data_fmt_and_tile`) |
| `tt_hlk_desc` | `tt_metal/jit_build/hlk_desc.hpp` |
| JIT array emission + BindingToken generation | `tt_metal/jit_build/genfiles.cpp` |
| Host register-format derivation (legacy) | `tt_metal/jit_build/data_format.cpp` (referenced by `data_format_derive.h`) |
| `DFBBindingToken` (id only, today) | `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h` |
| Scratchpad token | `tt_metal/hw/inc/api/scratchpad.h` |
| `TensorBindingToken` | `tt_metal/hw/inc/api/tensor/tensor_binding_token.h` |
| Local tensor accessor | `tt_metal/hw/inc/api/tensor/local_tensor_accessor.h` |
| Example id-free kernels | `tests/tt_metal/tt_metal/test_kernels/compute/{eltwise_copy_fp8_2_0,tilize_2_0,pack_untilize_2_0,eltwise_binary_add_idfree}.cpp` |
