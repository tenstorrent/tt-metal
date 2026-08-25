# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `2cd0286fa17 2026-08-25 docs(metal_2.0): a run in flight freezes the kernel sources` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to **`CustomProgramSpecFactoryConcept`**. Carry them forward:

- **Current concept:** `descriptor` — one factory, `RotaryEmbeddingProgramFactory`, with **two internal descriptor variants** selected by shape at `create_descriptor` (`device/rotary_embedding_program_factory.cpp:893-901`): single-tile (`Wt == 1`) and multi-tile. Both port inside the one factory.
- **Op-owned tensors:** none.
- **Target concept:** **`CustomProgramSpecFactoryConcept`** — the op declares `override_runtime_arguments` (`device/rotary_embedding_program_factory.cpp:903-992`); translate it into one returning `ProgramRunArgs` (recipe: *Translating override_runtime_arguments*). What survives the translation: only the **token-idx-derived decode scalars** — `cos_sin_start_id` (reader) and `cos_sin_offset` (writer), both core-invariant, recomputed from `operation_attributes.token_idx` (`:949-956`). The address re-writes (`:958-969`) become typed tensor bindings (refresh natively) and the `UpdateDynamicCircularBufferAddress` block (`:980-991`) dissolves into the `borrowed_from` DFBs. Prefill needs no re-application at all.
- **Custom hash:** present @ `device/rotary_embedding_device_operation.cpp:146-162` — leave it exactly as is (it deliberately keys `token_idx.has_value()` but not the value, so decode positions cache-hit one program; that is precisely why the translated override must re-emit the two decode scalars).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a non-`none` `TensorParameter relaxation` · `get_dynamic_runtime_args` (deprecated hook). A pybound `create_descriptor` is also absent.

## Construct — to do

**Tensor bindings** (per binding; today all four ride the `Buffer*` `emplace_runtime_args` form — `:448-463` single-tile, `:861-877` multi-tile):

- `src` (input) — **Case 1** in interleaved-input configs (both interleaved readers build `TensorAccessor(src_args, src_addr)`) → `TensorParameter`/`TensorBinding`; kernel uses `TensorAccessor(tensor::name)`. In **sharded-input** configs: **clean** — no src arg; CB `c_0` is borrowed (`CBDescriptor::buffer = input.buffer()` @ `:154`/`:539`) → `DataflowBufferSpec::borrowed_from`.
- `cos` — **Case 1**, all configs, all four readers.
- `sin` — **Case 1**, all configs, all four readers.
- `dst` (output) — **Case 1** in interleaved-output configs (writer `TensorAccessor(dst_args, dst_addr)` @ `writer_rotary_embedding_interleaved_start_id.cpp:28`). Under **`OUT_SHARDED`**: **clean** — CB `c_16` borrowed (`:238`/`:633`) → `borrowed_from`; the writer's accessor and dst arg are compiled out (`#ifndef OUT_SHARDED`), so make the dst binding conditional in step with the define.
- The legacy `TensorAccessorArgs` CTA plumbing (`:324-340`, `:723-743`) disappears with the Case-1 conversions.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** drop the redundant page-size arg (all Class 2 — each passes `get_tile_size(<cb>)`, which equals the true page size) at:

- `reader_rotary_embedding_single_tile_interleaved_start_id.cpp:103, 106, 109`
- `reader_rotary_embedding_single_tile_interleaved_start_id_sharded.cpp:95, 98`

(The multi-tile readers and the writer already use 2-arg accessors.)

**CB endpoints** (R = reader, W = writer, C = compute; full census in the audit):

- **Legal 1:1 — bind and go:** `c_0` input (R→C; + `borrowed_from` when in-sharded), `c_1` (multi-tile rotated_input R→C; single-tile trans_mat R→C), `c_2` cos / `c_3` sin (R→C), `c_4` scalar (multi-tile, R→C), `c_16` out (C→W; + `borrowed_from` under `OUT_SHARDED`), and the decode sync CBs `c_5`/`c_6` (W→C).
- **Self-loop** (one toucher — compute — bind PRODUCER **and** CONSUMER): `c_24`, `c_25`, `c_26` (both variants); decode-only `c_29`/`c_30` (retilized cos/sin).
- **Multi-binding advanced option** (`allow_instance_multi_binding`) on `c_27` and `c_28`, decode configs, both variants: compute is locked producer **and** consumer (untilize pushes, tilize waits/pops) and the writer is a second **locked consumer** (`wait_front(Wt)` @ `writer_...start_id.cpp:47,62`, never pops) plus a role-free raw in-place writer. Two locked consumers — no relabelling fits 1P+1C.
- **Aliased DFB pairs:** `c_27`+`c_5` and `c_28`+`c_6` each share one allocation (one `CBDescriptor`, two `CBFormatDescriptor`s @ `:271-303` single-tile / `:664-696` multi-tile) → express as two DFBs with `DFBAdvancedOptions::alias_with` (`advanced_options.hpp:113-131`), same total size, same node set.
- **Conditional DFBs:** the decode-only CBs (`c_27`/`c_5`, `c_28`/`c_6`, `c_29`, `c_30`) are already conditionally allocated host-side (`if (token_idx.has_value())` @ `:250`/`:643`) — carry the conditional into the spec; do not allocate them in prefill.
- **No dead CBs** — nothing to drop.

## Watch for

- **CB endpoints (multi-binding):** `c_27`/`c_28` decode — the second consumer is the writer's in-place row-shuffle (local NoC copy from `get_read_ptr()+cos_sin_offset` back to `get_read_ptr()`), sequenced by the aliased sync indices, not by CB FIFO sync alone. The shuffle is a raw peek+write on memory the compute side later tilizes from — keep the aliasing and the sync push order byte-for-byte.
- **Cross-op / shared kernels:** `device/kernels/compute/rotary_embedding_single_tile.cpp` is **lent** — `rotary_embedding_hf`'s `RotaryEmbeddingHfMultiCore` binds it (`rotary_embedding_hf/device/rotary_embedding_hf_multi_core_program_factory.cpp:257-260, 273-276`). **No `_metal2` fork exists beside it** (checked locationally) → rung 2: this port creates `rotary_embedding_single_tile_metal2.cpp` beside the original (your own directory — lent kernel) and leaves the pointer comment in the original. Name the bindings for the kernel's roles, not this op's locals — `rotary_embedding_hf` will bind the same fork (it binds without `DECODE_MODE` and without the decode CTAs, so keep the decode surface behind the define exactly as the legacy file does). Other binding ops: `rotary_embedding_hf` — **sunset list, not authorization to convert the kernel in place.** Heads-up: `rotary_embedding_hf` is being audited/ported in parallel; if its port lands the fork first, reuse it (rung 1) — an add/add conflict is the convention working. The other six kernel sources have no external binders.
- **RTA varargs:** none — every arg is a fixed distinct index; name them all (per kernel-variable names: `src_addr`→`tensor::src` etc.). No CTA varargs.
- **Defines drive structure:** supply `DECODE_MODE` (reader/writer/compute) and `OUT_SHARDED` (writer) per config, and keep the decode-only DFBs, bindings, and args conditional in step — the interleaved reader's arg *list* also differs from the sharded reader's (7 vs 5 args; `cos_sin_start_id` sits at legacy idx 6 vs 4). Named args make the index shift moot, but the per-config schemas differ.
- **Config-dead legacy args** — don't carry them into named schemas where a config never reads them: `start_row_id` is unread under `DECODE_MODE` (both interleaved readers); the writer never reads `dst_addr`/`start_id` under `OUT_SHARDED`, and reads `cos_sin_offset`/`Wt`/`Wbytes` only under `DECODE_MODE`.
- **Constexpr metadata form:** `constexpr uint32_t out_tile_size = get_tile_size(cb_id_out);` @ `writer_...start_id.cpp:81` — a `DataflowBuffer` object is never constexpr; use the token form for constexpr metadata (whitelist rule 7 / port_patterns).
- **Preserve the g1/g2 compute-config asymmetry (multi-tile):** group-1 compute uses a default `ComputeConfigDescriptor{}` while group-2 sets `math_fidelity`/`fp32_dest_acc_en` (`:812-814` vs `:828-831`) — deliberate legacy parity per the in-code comment; replicate it, don't "fix" it. (The single-tile variant sets both on both groups.)
- **Borrowed DFBs never co-occur with two compute groups:** the sharded work split forces `core_group_2` empty (`compute_rotary_work_split` @ `:57-70`), so every `borrowed_from` config has a single compute KernelSpec — relevant if you are minding multi-work-unit + borrowed-DFB interactions.
- **Kernel-lib helpers are already Metal 2.0-friendly:** `compute_kernel_lib::tilize/untilize` take `uint32_t` DFB ids as NTTPs and run on `DataflowBuffer` internally — `dfb::name`'s constexpr cast passes straight in; no donor work.
