# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding`

- **`RotaryEmbeddingDeviceOperation`** (`device/rotary_embedding_device_operation.hpp`)
  - `RotaryEmbeddingProgramFactory` (`device/rotary_embedding_program_factory.cpp`) — the op's **single** ProgramFactory. Internally it builds one of **two descriptor variants**, selected by shape at `create_descriptor` (`rotary_embedding_program_factory.cpp:893-901`):
    - **single-tile** (`Wt == 1`, i.e. `padded_shape[-1] == TILE_WIDTH`) — `create_single_tile_descriptor` (`:91`)
    - **multi-tile** (`Wt >= 2`) — `create_multi_tile_descriptor` (`:477`)
  - Orthogonal config axes: **decode** (`token_idx.has_value()`, `DECODE_MODE` define) vs **prefill**; **in-sharded** vs interleaved input; **out-sharded** (`OUT_SHARDED` define) vs interleaved output.
  - Kernels (all op-owned, all referenced):
    - `device/kernels/dataflow/reader_rotary_embedding_interleaved_start_id.cpp` (multi-tile, interleaved in)
    - `device/kernels/dataflow/reader_rotary_embedding_interleaved_start_id_sharded.cpp` (multi-tile, sharded in)
    - `device/kernels/dataflow/reader_rotary_embedding_single_tile_interleaved_start_id.cpp` (single-tile, interleaved in)
    - `device/kernels/dataflow/reader_rotary_embedding_single_tile_interleaved_start_id_sharded.cpp` (single-tile, sharded in)
    - `device/kernels/dataflow/writer_rotary_embedding_interleaved_start_id.cpp` (all configs)
    - `device/kernels/compute/rotary_embedding.cpp` (multi-tile)
    - `device/kernels/compute/rotary_embedding_single_tile.cpp` (single-tile; **also bound by `rotary_embedding_hf`** — see shared-kernel finding)

This audit covers `rotary_embedding` only — **not** the sibling ops `rotary_embedding_hf`, `rotary_embedding_llama`, `rotary_embedding_llama_fused_qk` (separate directories, separate audits).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `2cd0286fa17 2026-08-25 docs(metal_2.0): a run in flight freezes the kernel sources`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `RotaryEmbeddingDeviceOperation` → `RotaryEmbeddingProgramFactory` (two internal descriptor variants: single-tile / multi-tile) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 7 kernels on `Noc` / `CircularBuffer` / `TensorAccessor`; kernel_lib donors already on `DataflowBuffer` |
| *Prereqs* — Cross-op escapes | Ok — only `tt_metal` API headers + `ttnn/kernel_lib` (tilize/untilize helpers, already DFB-based, `uint32_t` NTTP shape ✓) |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok (no varying-index CTA reads; not an Appendix A entry — see Recipe notes) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (concept is `descriptor`; sheet `Execution Model` = `SPMD`) |
| *TTNN Readiness* — Custom hash | **Yes** (not a gate; port leaves it intact): `compute_program_hash` @ `device/rotary_embedding_device_operation.cpp:146-162` |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (grep of op directory: zero hits; sheet agrees) |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** (not a gate; selects **CustomProgramSpecFactoryConcept**): `RotaryEmbeddingProgramFactory::override_runtime_arguments` @ `device/rotary_embedding_program_factory.cpp:903-992` |
| *TTNN Readiness* — Pybind `create_descriptor` | No (`rotary_embedding_nanobind.cpp` binds only the public op function) |
| *TTNN Readiness* — Op-owned tensors | No (sheet cell empty; `descriptor` concept cannot carry them) |
| *TTNN Readiness* — Target concept | **`CustomProgramSpecFactoryConcept`** |
| *Port work* — Offset base pointer | none (clean scan; op absent from the offset-base-pointer triage tables — reconciled "no fold, not in tables") |
| *Port work* — Tensor bindings (per binding) | `src` Case 1 (interleaved) / clean borrowed-DFB (sharded-in) · `cos` Case 1 · `sin` Case 1 · `dst` Case 1 (interleaved out) / clean borrowed-DFB (OUT_SHARDED) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | drop (**Class 2 — redundant**) @ 5 sites in the two single-tile readers (matches the 3rd-arg triage row for `rotary_embedding`) |
| *Port work* — CB endpoints | mostly legal 1:1 · self-loop (compute intermediates) · **multi-binding flag** on `c_27`/`c_28` in decode configs · **aliased DFB pairs** `c_27↔c_5`, `c_28↔c_6` · no dead CBs |

**CB endpoints** are dispositions, not gates: every out-of-window CB here has a port-time resolution (self-loop, or the multi-binding advanced option). See the census below.

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md` beside this file). All five gates cleared: Device 2.0 ✓, Feature compatibility ✓ (all N/A), TTNN factory concept ✓ (`Is able to port?` = `yes`, cross-check clean), Offset base pointers ✓, TensorAccessor 3rd arg ✓ (all sites Class 2). TensorParameter relaxation = `none`. Target: **`CustomProgramSpecFactoryConcept`** (the op declares `override_runtime_arguments`, which the port translates rather than deletes).

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN — the readiness sheet (fetched fresh this session, 2026-08-25) has exactly one row for this op: `Op = experimental/transformer/rotary_embedding`, `Device operation = RotaryEmbeddingDeviceOperation`, `Factory (variant) = RotaryEmbeddingProgramFactory`, **`Is able to port?` = `yes`**, `TensorParameter relaxation` = `none`, `Known op issues` = (empty), `Smuggled pointer` = `no`, `Op Classification` = `PD Op (custom)`, `Porting Target` = `CustomProgramSpecFactoryConcept`. Cross-check clean on every primary column:
  - `Concept` = `descriptor` ✓ — `create_descriptor()` returning `ProgramDescriptor` @ `rotary_embedding_program_factory.cpp:893`.
  - `Custom hash` = `yes` ✓ — `compute_program_hash` @ `rotary_embedding_device_operation.cpp:146`. No backdoor `attribute_values`/`to_hash` (sheet: `no`) ✓.
  - `Runtime-args update (get_dynamic_runtime_args)` = `no` ✓ — zero grep hits in the op directory.
  - `Override runtime args method?` = `yes` ✓ — `override_runtime_arguments` @ `rotary_embedding_program_factory.cpp:903` on a `descriptor` op → target-concept signal, not the legacy-concept signature.
  - `Pybind descriptor` = `no` ✓ — `rotary_embedding_nanobind.cpp` binds only `ttnn::experimental::rotary_embedding` via `bind_function`; no `create_descriptor` binding, no `nb::class_` of the device op.
  - Factory-set match ✓ — one factory in code (`program_factory_t = std::variant<RotaryEmbeddingProgramFactory>` @ `rotary_embedding_device_operation.hpp:18`), one sheet row. The two descriptor variants are internal branches of the single factory, not separate factories — no phantom/missing rows.
  - Cross-column invariants hold (`get_dynamic_runtime_args` = `no`; `descriptor` concept with no op-owned tensors).
- **Device 2.0 (every kernel used):** GREEN. All five dataflow kernels use the Device 2.0 surface throughout: `Noc` object + `noc.async_read`/`async_write`/barriers, `CircularBuffer` wrapper objects with methods (`reserve_back`/`push_back`/`wait_front`/`pop_front`/`get_write_ptr`/`get_read_ptr`), `TensorAccessor`/`TensorAccessorArgs`, `CoreLocalMem`, `UnicastEndpoint`. Both compute kernels use `CircularBuffer` wrappers for FIFO ops plus the standard compute LLK API. Specifics checked:
  - `get_tile_size(cb_id)` free-function calls (all seven kernels) — **sanctioned**, not holdovers.
  - `my_x[noc.get_noc_id()]` / `my_y[...]` in the multi-tile sharded reader (`:92-101`) and writer (`:55-56, :70-71`) — accepted Device 2.0 style (broad precedent in migrated kernels: clone, concat, conv2d readers/writers).
  - No `InterleavedAddrGen*`/`ShardedAddrGen`, no raw sem addresses, no `noc_async_*` free functions, no `get_read_ptr(cb_id)`-style CB-index holdovers.
  - Donor code (`ttnn/kernel_lib/tilize_helpers` / `untilize_helpers`) is already on `DataflowBuffer` internally (`untilize_helpers.inl:199-268`, `tilize_helpers.inl:166-214`) — fully Device 2.0.
- **Feature compatibility:** every Appendix A entry N/A.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no type reference, no `.global_circular_buffer` field on any `CBDescriptor`, no remote-CB idioms |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `address_offset` anywhere; the `UpdateDynamicCircularBufferAddress` calls @ `rotary_embedding_program_factory.cpp:986,988` are the **three-arg** form (no offset) — false-positive-guarded, not this rule |
  | GlobalSemaphore | N/A | no type reference, no `global_semaphore.hpp` include; op uses no semaphores at all |

- **CB endpoints (GATE-free):** full census below (Port-work summary). Headline: all producer/consumer pairs are legal 1:1 or compute-internal self-loops, **except** the decode-mode untilized-cos/sin data CBs `c_27`/`c_28`, which have two locked consumers on every node (compute + writer) → **multi-binding advanced option**; and each of those shares its allocation with a sync index (`c_27↔c_5`, `c_28↔c_6`, one `CBDescriptor` with two `CBFormatDescriptor`s) → **aliased DFBs** (`DFBAdvancedOptions::alias_with`). No dead CBs in any config.
- **Offset base pointers:** GREEN — no address RTA folds a host-side offset into its base. All tensor addresses reach kernels as **`Buffer*` runtime args** (the descriptor `emplace_runtime_args` Buffer-pointer form) with no arithmetic: `rotary_embedding_program_factory.cpp:448-463` (single-tile) and `:861-877` (multi-tile). The decode scalar `cos_sin_offset` (`:431`, `:838`) is a byte offset *within an L1 CB*, passed as its own separate scalar arg and applied kernel-side to a CB `get_read_ptr()` (`writer_...start_id.cpp:49,64`) — not a folded device pointer. Op is absent from the offset-base-pointer triage tables (`2026-07-19_offset_base_pointers.md`); reconciliation outcome: *no fold, not in tables → clean*.
- **TensorAccessor 3rd argument:** GREEN — sites found and classified **Class 2 (redundant → drop)**. Five sites, all in the single-tile readers, all passing `get_tile_size(<cb>)` where the CB format is `datatype_to_dataformat_converter(dtype)` — i.e. exactly the tensor's true tile/page size (correct magnitude; also verbatim-correct if a tensor were sharded):
  - `reader_rotary_embedding_single_tile_interleaved_start_id.cpp:103` (`s0`, `input_tile_bytes`), `:106` (`s1`, `cos_tile_bytes`), `:109` (`s2`, `sin_tile_bytes`)
  - `reader_rotary_embedding_single_tile_interleaved_start_id_sharded.cpp:95` (`s1`), `:98` (`s2`)

  The multi-tile readers and the writer construct 2-arg accessors (no 3rd arg). The 3rd-arg triage doc (`2026-07-06_tensor_accessor_3rd_arg_triage.md:71`) lists `rotary_embedding` as `2 — Redundant` — my classification agrees.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per config):
  - `src` (input) — **Case 1** in interleaved-input configs (`Buffer*` RTA @ `:453`/`:867` → `TensorAccessor(src_args, src_addr)` in both interleaved readers); **clean** (borrowed-memory DFB) in sharded-input configs (`CBDescriptor::buffer = input.buffer()` @ `:154`/`:539`; no src RTA — the sharded readers publish/consume CB `c_0` directly → `DataflowBufferSpec::borrowed_from`).
  - `cos` — **Case 1** in all configs (`Buffer*` RTA @ `:449`/`:453`/`:863`/`:867` → `TensorAccessor(cos_args, cos_addr)` in all four readers).
  - `sin` — **Case 1** in all configs (same shape).
  - `dst` (output) — **Case 1** in interleaved-output configs (`Buffer*` RTA @ `:462-463`/`:876-877` → `TensorAccessor(dst_args, dst_addr)` in the writer, `writer_...start_id.cpp:28`); **clean** (borrowed-memory DFB) under `OUT_SHARDED` (`CBDescriptor::buffer = output.buffer()` @ `:238`/`:633`; the writer's accessor is compiled out, `writer_...start_id.cpp:27-29`).
  - All four are the **`Buffer*`-binding form** (pointer object, not `->address()`), which the framework already auto-registers as `BufferBinding`s — routine port work, not a stale-pointer hazard. Additionally `override_runtime_arguments` re-writes the address slots per hit (`:958-974`), since declaring the hook supersedes `resolve_bindings` (per the header comment @ `rotary_embedding_program_factory.hpp:25-31`).
  - CTA-baked `TensorAccessorArgs` plumbing (`:324-340`, `:723-743`) disappears with the Case-1 conversions.
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** drop the redundant page-size arg at the 5 Class-2 sites listed in Gate detail.
- **CB endpoints** — census per `(CB, variant, config)`; kernels: R = reader, W = writer, C = compute:

  | CB | Variant / config | Touchers | Disposition |
  |---|---|---|---|
  | `c_0` input | both; interleaved-in | R FIFO-P, C FIFO-C | legal 1:1 |
  | `c_0` input | both; sharded-in | R FIFO-P (publish push; multi-tile R also raw-peeks its own buffer — same toucher), C FIFO-C | legal 1:1 + `borrowed_from` input |
  | `c_1` rotated_input | multi-tile | R FIFO-P, C FIFO-C | legal 1:1 |
  | `c_1` trans_mat | single-tile | R FIFO-P (fills matrix once), C FIFO-C (`wait_front`, never pops — persistent) | legal 1:1 |
  | `c_2` cos / `c_3` sin | both; all configs | R FIFO-P, C FIFO-C (prefill: wait+pop; decode: untilize wait+pop) | legal 1:1 |
  | `c_4` scalar | multi-tile | R FIFO-P, C FIFO-C (`wait_front` only — persistent) | legal 1:1 |
  | `c_24` / `c_25` / `c_26` interm | both | C only (FIFO-P + FIFO-C in the same kernel) | **self-loop** |
  | `c_16` out | both; interleaved-out | C FIFO-P, W FIFO-C (wait+pop+peek) | legal 1:1 |
  | `c_16` out | both; OUT_SHARDED | C FIFO-P, W FIFO-C (`wait_front` only) | legal 1:1 + `borrowed_from` output |
  | `c_29` / `c_30` retilized cos/sin | both; **decode only** | C only (tilize push; `wait_front` in MUL, no pop) | **self-loop** |
  | `c_27` untilized-cos data (aliased w/ `c_5`) | both; **decode only** | C locked-P (untilize push) **and** locked-C (tilize wait+pop); W locked-C (`wait_front(Wt)` @ `writer_...start_id.cpp:62`, no pop) + role-free raw in-place write (`:64-73`) | **multi-binding flag** (2 locked consumers) |
  | `c_5` untilized-cos sync (aliased w/ `c_27`) | both; **decode only** | W FIFO-P (`reserve/push` @ `:63,:75`), C FIFO-C (`TILIZE` sync wait+pop) | legal 1:1 |
  | `c_28` / `c_6` untilized-sin data/sync | both; **decode only** | mirror of `c_27`/`c_5` (`writer_...start_id.cpp:47-60`) | `c_28`: **multi-binding flag** · `c_6`: legal 1:1 |

  - **Aliasing:** `c_27`+`c_5` share one `CBDescriptor` (one allocation, two `CBFormatDescriptor`s) @ `rotary_embedding_program_factory.cpp:271-286` (single-tile) / `:664-679` (multi-tile); `c_28`+`c_6` likewise @ `:288-303` / `:681-696`. Metal 2.0 expresses this as two DFBs with `DFBAdvancedOptions::alias_with` (`advanced_options.hpp:113-131`). The scheme is a deliberate in-place ping-pong: compute untilizes into `c_27`, the writer waits on `c_27`, row-shuffles the data **in place** via a local NoC copy (`get_read_ptr()+cos_sin_offset` → `get_read_ptr()`), then pushes the **sync** index `c_5` so compute's tilize (which waits on `c_5`, reads/pops `c_27`) sees the shuffled rows.
  - **Dead CBs:** none — every allocated CB is touched in every config it is allocated for. The decode-only CBs are already **conditionally allocated** host-side (`if (token_idx.has_value())` @ `:250`/`:643`), so the conditional DFB spec is a direct translation, not new structure.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** `c_27` and `c_28` in decode configs, both variants — the writer is a genuine second locked consumer (`wait_front`, no pop) plus a raw in-place writer coordinated by the aliased sync CBs (`c_5`/`c_6`), not CB FIFO sync alone. Bind compute P+C and set `allow_instance_multi_binding` for the writer's extra consumer binding; do **not** try to relabel — both consumers are FIFO-locked.
- **Cross-op / shared kernels:** `device/kernels/compute/rotary_embedding_single_tile.cpp` is **lent**: `rotary_embedding_hf`'s `RotaryEmbeddingHfMultiCore` factory binds it (`rotary_embedding_hf/device/rotary_embedding_hf_multi_core_program_factory.cpp:257-260, 273-276`). No `_metal2` fork exists beside it (checked locationally). Filename census over `ttnn/cpp/ttnn/operations/` found no other binders for any of the op's seven kernels (the `sources.cmake` hit for `rotary_embedding.cpp` is a build file / host-file name match — discarded). Consumer set {`rotary_embedding`, `rotary_embedding_hf`} is a **sunset list, not authorization to convert in place**. Note: `rotary_embedding_hf` is being audited (and may be ported) in parallel — a concurrent port may create the fork first; an add/add conflict there is the convention working (`port_patterns.md`, Caution: Porting a shared kernel).
- **RTA varargs:** none — every `get_arg_val` in all five dataflow kernels reads a fixed, distinct index; no counted-loop or data-selected reads. No CTA varargs either (`TensorAccessorArgs<N>` blocks sit at fixed constexpr offsets). All args get names.
- **`override_runtime_arguments` translation** (`rotary_embedding_program_factory.cpp:903-992`): under `CustomProgramSpecFactoryConcept` the typed tensor bindings refresh addresses natively and the borrowed DFBs replace the `UpdateDynamicCircularBufferAddress` block (`:980-991`), so the translated hook only needs to re-emit the **token-idx-derived decode scalars**: `cos_sin_start_id` (reader; legacy arg idx 4 sharded / 6 interleaved) and `cos_sin_offset` (writer; legacy arg idx 3). Both are core-invariant. Prefill touches neither.
- **Defines gate program structure:** `DECODE_MODE` (all three kernel classes) and `OUT_SHARDED` (writer) — the KernelSpecs must supply the same defines per config, and the decode-only DFBs/args must be conditional in step with them.
- **Constexpr metadata form:** the writer's `constexpr uint32_t out_tile_size = get_tile_size(cb_id_out);` (`writer_...start_id.cpp:81`) is a constexpr CB-metadata lookup — mind the token-form vs member-getter decision when converting (a `DataflowBuffer` object is never constexpr).
- **Borrowed DFBs co-occur only with a single core group:** the sharded work split forces `core_group_2` empty (`compute_rotary_work_split` @ `:57-70`), so every config with a `borrowed_from` DFB has exactly one compute KernelSpec; the two-compute-group shape occurs only in fully interleaved configs, which borrow nothing.
- **Preserve the g1/g2 compute-config asymmetry (multi-tile variant):** group-1 compute deliberately uses a default `ComputeConfigDescriptor{}` while group-2 sets `math_fidelity`/`fp32_dest_acc_en` (`:812-814` vs `:828-831`, comment says legacy parity). Carry it over as-is; see Misc anomalies.

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean**.
  - Function-call escapes (kernel `#include`s outside the op directory):

    | Op kernel | Donor | Class | Shape | Status |
    |---|---|---|---|---|
    | all 7 kernels | `tt_metal/hw/inc/api/...` (dataflow_api, noc, circular_buffer, compute/*, core_local_mem, tensor/noc_traits, endpoints) | 1 — framework | — | ✓ no concern |
    | both compute kernels | `ttnn/kernel_lib/tilize_helpers.hpp` — `compute_kernel_lib::tilize<block_width_tiles, input_dfb, output_dfb, ...>(num_blocks)` | 2 — official kernel lib | `uint32_t` DFB ids as NTTPs; internals already on `DataflowBuffer` (`tilize_helpers.inl:166-214`) | ✓ excellent |
    | both compute kernels | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` — `compute_kernel_lib::untilize<...>(num_blocks)` | 2 — official kernel lib | same (`untilize_helpers.inl:199-268`) | ✓ excellent |

    Per-call detail omitted — all rolls ✓. (Note the two compute kernels spell the untilize include with the long `ttnn/cpp/ttnn/kernel_lib/` prefix and the tilize one short — cosmetic.)
  - Borrowed kernel files (file-path instantiation): this op borrows **none** (all seven kernel sources are op-owned). One of its kernels is **lent** — `compute/rotary_embedding_single_tile.cpp` → `rotary_embedding_hf` (`RotaryEmbeddingHfMultiCore`), detailed in Heads-ups. No `_metal2` fork exists yet.
- **Relaxation candidates:** none observed. The custom hash (`rotary_embedding_device_operation.cpp:146-162`) keys `seq_len`, `token_idx.has_value()` (deliberately not the value — decode positions share one cached program, re-offset per hit), `output_mem_config`, `compute_kernel_config`, and the three input tensors; it is a *tightening* for correctness, not a relaxation of the strict TensorSpec match.
- **TTNN factory analysis:** current concept `descriptor`; no op-owned tensors; custom hash `yes` (@ `:146`, port leaves intact); `get_dynamic_runtime_args` absent; `override_runtime_arguments` present (@ `rotary_embedding_program_factory.cpp:903`) → target **`CustomProgramSpecFactoryConcept`**; pybind of internals absent; smuggled pointer absent (all address RTAs are the annotated `Buffer*` form). Sheet row cross-checked clean — no sheet/code disagreements.

## Misc anomalies  *(team-only, non-gating)*

- **Promised dtype constraint is absent:** the validation comment @ `rotary_embedding_device_operation.cpp:44-49` says the single-tile path should "constrain input/cos/sin to bfloat16" (WH LLK corrupts bfp8-input @ bf16-trans_mat matmul packs), but no `TT_FATAL` enforcing a dtype follows — and the factory explicitly handles a Bfp8_b trans_mat (`rotary_embedding_program_factory.cpp:113-115`). Either the comment is stale or the check was dropped; if the LLK issue is still live, a bfp8 input on the `Wt==1` path would hit it unguarded.
- **g1/g2 compute-config asymmetry (multi-tile variant):** group-1 compute runs with a default-constructed `ComputeConfigDescriptor{}` (no `math_fidelity`/`fp32_dest_acc_en`) while group-2 sets both from `compute_kernel_config` (`rotary_embedding_program_factory.cpp:812-814` vs `:828-831`). The comment says this preserves legacy `create()` behavior, but it means the two core groups of one dispatch run at different fidelity. Latent behavioral inconsistency for the ops team; the port must *preserve* it (zero-functional-change).
- **Config-dead args:** `start_row_id` reader RTA is unread under `DECODE_MODE` (`reader_...interleaved_start_id.cpp:20,65` — `ht` only used in the `#ifndef DECODE_MODE` path; same in `reader_...single_tile_interleaved_start_id.cpp:88,123`). The writer's `dst_addr` RTA and dst `TensorAccessorArgs` CTAs are unread under `OUT_SHARDED` (`writer_...start_id.cpp:15,20,27-29`), as is `start_id` (`:17`). Harmless legacy plumbing; the porter will naturally resolve these when naming args per config.

## Recipe notes

- The `METAL2_PREPORT_AUDIT.md` template's status-summary table carries a row `*Feature Support* — Variadic-CTA | Ok / Unsupported`, but Appendix A contains no Variadic-CTA entry — the RTA-varargs subject explicitly says CTA varargs are *supported* (`KernelAdvancedOptions::compile_time_varargs`). I filled the row as `Ok`; suggest either dropping the row or adding the entry it refers to.
- The audit recipe never mentions the **multi-`CBFormatDescriptor` `CBDescriptor`** (two CB indices aliasing one allocation), which this op uses for its decode sync scheme (`c_27`+`c_5`, `c_28`+`c_6`). Per the Appendix-A-is-authoritative rule I did not gate on it; `DFBAdvancedOptions::alias_with` (`advanced_options.hpp:113-131`) appears to be the intended Metal 2.0 expression, and I recorded it as PORT WORK. Worth a sentence in the CB-endpoints subject (the aliased pair also complicates the census: the per-*index* census differs from the per-*allocation* census).
