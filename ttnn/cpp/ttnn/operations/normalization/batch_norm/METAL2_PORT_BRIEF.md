# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/normalization/batch_norm`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `40b61b016a1 2026-07-29 docs(metal_2.0): fix stale API symbol names across the porting docs` *(carry this line into the port report's Provenance section)*

## What you are porting

**Two DeviceOperations, one directory, one port unit.** They share no factory and no kernel, but share a host
util, all three donor headers, an identical structural shape, and a single user-facing entry point
(`ttnn::batch_norm`). Everything below is stated per DeviceOperation where it differs.

| DeviceOperation | Factory | Kernels |
|---|---|---|
| `BatchNormOperation` | `BatchNormFactory` (`device/batch_norm_program_factory.cpp`) | `dataflow/reader_batch_norm.cpp` · `dataflow/writer_batch_norm.cpp` · `compute/batch_norm_kernel.cpp` · `compute/batch_norm_sfpu_kernel.cpp` |
| `RunningStatistics` | `RunningStatisticsProgramFactory` (`device/running_statistics_program_factory.cpp`) | `dataflow/reader_running_statistics.cpp` · `dataflow/writer_running_statistics.cpp` · `compute/running_statistics_kernel.cpp` · `compute/running_statistics_sfpu_kernel.cpp` |

**Read this before you plan the kernel work: each factory ships *two* compute-kernel source files, not one.**
The path is built at descriptor time —
`fmt::format(".../compute/batch_norm_{}.cpp", (fp32_dest_acc_en || any_float32) ? "sfpu_kernel" : "kernel")`
(`batch_norm_program_factory.cpp:388-390`, `running_statistics_program_factory.cpp:438-440`). They are two
separate `.cpp` files sharing one CTA list, so **both are in the port and must land in the same change**. The
**SFPU variant is the default path** — `resolve_compute_kernel_config` sets `default_fp32_acc = true`
(`device/batch_norm_utils.cpp:31`), so the non-SFPU file only runs when a caller explicitly passes
`fp32_dest_acc_en = false` *and* every tensor is bf16. Test both.

**The kernels are already on `DataflowBuffer` — your delta is smaller than usual.**
`bed70038e18 (#49173)` migrated all 8 kernels from `CircularBuffer` to `DataflowBuffer`: `dfb_*` naming,
`api/dataflow/dataflow_buffer.h`, and `get_tile_size(cb_id)` already replaced by the member
`dfb.get_entry_size()`. So the wrapper-object rewrite and the whitelist rule-7 metadata move are **done**. What
remains on the kernel side is swapping the DFB *ids* for `dfb::name` binding tokens and the address RTAs for
`tensor::name`. Keep `get_entry_size()` as-is — it is the sanctioned member for entry size, and on Gen1
tile-formatted DFBs it is the same value the kernels previously read as a tile-byte count.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to
`ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — both DOps. Plain single-program; neither returns a `WorkloadDescriptor`.
- **Op-owned tensors:** none. `BatchNormOperation` accepts a caller-supplied preallocated `output`
  (`batch_norm_device_operation.cpp:113-118`), but that is an ordinary optional output tensor, not an op-owned
  tensor.
- **Target concept:** **`ProgramSpecFactoryConcept`** (base form — cache hit is `UpdateTensorArgs` only),
  `ttnn/api/ttnn/operation_concepts.hpp:119`. The recipe docs name the same concept — no divergence to work
  around.
- **⚠ The concept flip is atomic — you must *remove* `create_descriptor`.** `ProgramSpecFactoryConcept` requires
  `!ProgramDescriptorFactoryConcept` (`:116-119`), and `all_factories_valid` (`:176-182`) permits exactly one of
  the five concepts per factory. Each DOp's `program_factory_t` holds a single factory struct, so
  `create_descriptor` must go in the same change that adds `create_program_artifacts`.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash ·
  `get_dynamic_runtime_args` · `override_runtime_arguments` · pybind `create_descriptor` · smuggled pointer.
  All `no` on both readiness-sheet rows and independently confirmed in code.
- **Note for the record, not an action:** `BatchNormOperation::operation_attributes_t::to_hash()` exists
  (`batch_norm_device_operation.cpp:121-123`). The sheet does **not** score it as a custom hash and the gate is
  unaffected — **do not treat it as a relaxation, and do not change it.**

## Construct — to do

### Tensor bindings — 11 bindings, all **Case 1**, all mechanical

Every one follows the same shape: the factory pushes a `Buffer*` into the per-core RTA list, the kernel reads that
slot as a `uint32_t`, and feeds it into a `TensorAccessor` built from a `TensorAccessorArgs<N>` CTA block. Express
each as a `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)`, and **both the
RTA slot and its `TensorAccessorArgs` CTA block disappear**. No Case 2 anywhere — no kernel does hand-rolled NoC
arithmetic on a base pointer, so `get_bank_base_address` is not needed. No borrowed-memory DFBs — the op declares
no buffer-backed CBs.

`BatchNormOperation` — 6 bindings:

| Binding | Host site (RTA slot) | CTA block | Kernel accessor |
|---|---|---|---|
| `input` | `batch_norm_program_factory.cpp:90` (reader RTA 1) | `:307` | `reader_batch_norm.cpp:38` |
| `batch_mean` | `:111` (writer RTA 0) | `:319` | `writer_batch_norm.cpp:53` |
| `batch_var` | `:112` (writer RTA 1) | `:321` | `writer_batch_norm.cpp:61` |
| `weight` *(optional)* | `:113` (writer RTA 2) | `:322-323` | `writer_batch_norm.cpp:65` |
| `bias` *(optional)* | `:114` (writer RTA 3) | `:324` | `writer_batch_norm.cpp:69` |
| `output` | `:115` (writer RTA 4) | `:320` | `writer_batch_norm.cpp:57` |

`RunningStatistics` — 5 bindings:

| Binding | Host site (RTA slot) | CTA block | Kernel accessor |
|---|---|---|---|
| `batch_mean` | `running_statistics_program_factory.cpp:88` (reader RTA 1) | `:351` | `reader_running_statistics.cpp:39` |
| `batch_var` | `:109` (writer RTA 0) | `:364` | `writer_running_statistics.cpp:52` |
| `running_mean` *(optional, read-modify-write)* | `:110` (writer RTA 1) | `:366-367` | `writer_running_statistics.cpp:58` |
| `running_var` *(optional, read-modify-write)* | `:111` (writer RTA 2) | `:368-369` | `writer_running_statistics.cpp:61` |
| `output` | `:112` (writer RTA 3) | `:365` | `writer_running_statistics.cpp:55` |

**Two shapes to get right up front:**

1. **An absent optional tensor is delivered as a literal `0u`, not a missing arg.** When `weight` / `bias` /
   `running_mean` / `running_var` is absent, the factory pushes `std::variant<uint32_t, Buffer*> arg = 0u`
   (`batch_norm_program_factory.cpp:101-108`, `running_statistics_program_factory.cpp:99-106`) paired with
   `TensorAccessorArgs(nullptr)` (`batch_norm_program_factory.cpp:322-324`,
   `running_statistics_program_factory.cpp:366-369`). The kernel still *constructs* the accessor unconditionally
   and never uses it, guarded by a `..._has_value` CTA. Presence is a **compile-time branch**, so **simply do not
   declare the `TensorParameter` in the absent configuration** — do not bind a null tensor.
2. **`running_mean` / `running_var` are read *and* written through one binding.**
   `writer_running_statistics.cpp:86-99` reads and `:102-110` writes back through the *same* `TensorAccessor`
   (in-place update; `batch_norm.cpp:124-128` documents the ordering constraint this creates for the caller).
   **One `TensorParameter` covers both directions — do not split it into an in-binding plus an out-binding.**

### TensorParameter relaxation

**none.** `TensorParameter relaxation = none` on both readiness-sheet rows, and there is no custom
`compute_program_hash` to reconcile against.

### TensorAccessor 3rd arg

**none.** All eleven `TensorAccessor` constructions are the two-argument form; no page-size override exists.
Nothing to drop.

### CB endpoints

Every CB is allocated over `all_device_cores`, and all three kernels of each factory run over that same range —
so each node hosts reader, writer and compute, and the census is uniform across nodes. **No CB anywhere reaches
≥3 touchers or doubles a FIFO role: the multi-binding advanced option is never needed. No dead CB to drop.**
Compute sites below cite the SFPU variant (the default); the non-SFPU variant has the same touchers.

**`BatchNormFactory`:**

- **legal 1:1 — bind normally, no special action:**
  `input_tensor_cb` (`c_0`, R→C) · `batch_mean_tensor_cb` (`c_1`, W→C) · `batch_var_tensor_cb` (`c_3`, W→C) ·
  `eps_cb` (`c_4`, R→C) · `weight_tensor_cb` (`c_5`, W→C, *weight present*) · `bias_tensor_cb` (`c_6`, W→C,
  *bias present*) · `output_tensor_cb` (`c_2`, C→W, *no-typecast config*) · `writer_cb` (`c_9`, C→W,
  *typecast config only*).
- **self-loop (one toucher — bind compute PRODUCER *and* CONSUMER):**
  `den_cb` (`c_7`) · `temp_1_cb` (`c_8`) · **`output_tensor_cb` (`c_2`) under the typecast config** — when
  `needs_output_typecast` the writer is redirected to `c_9`, leaving compute as the only toucher (it produces at
  `batch_norm_sfpu_kernel.cpp:142,159` and consumes at `:164,183`).
- **assign cosmetic 1P+1C (writer PRODUCER, compute CONSUMER) — `weight_tensor_cb` (`c_5`) and
  `bias_tensor_cb` (`c_6`) under the *absent* config.** These are allocated unconditionally (`:260-269`,
  `:270-279`) and both kernels still *name* them (`writer_batch_norm.cpp:49` wrapper + `:64` `get_entry_size()`;
  `batch_norm_sfpu_kernel.cpp:49` wrapper), but no FIFO or pointer access executes. **Do not treat this as a dead
  CB and do not drop it** — the kernels reference the DFB, so a binding is required; the roles are free on Gen1.
  **This disposition is confirmed by the op owner** — a decision, not a suggestion.

**`RunningStatisticsProgramFactory`:**

- **legal 1:1:** `batch_mean_tensor_cb` (`c_0`, R→C) · `batch_var_tensor_cb` (`c_1`, W→C) ·
  `output_tensor_cb` (`c_2`, C→W) · `momentum_cb` (`c_5`, R→C) · `one_cb` (`c_6`, R→C) ·
  `old_running_mean_tensor_cb` (`c_3`, W→C, *present*) · `old_running_var_tensor_cb` (`c_4`, W→C, *present*) ·
  `updated_m_cb` (`c_7`, C→W, *no mean typecast*) · `updated_v_cb` (`c_8`, C→W, *no var typecast*) ·
  `wm_cb` (`c_12`) and `wv_cb` (`c_13`) (C→W, *typecast configs only*).
- **self-loop:** `tmp1_cb` (`c_9`) · `tmp2_cb` (`c_10`) · `tmp3_cb` (`c_11`) · **`updated_m_cb` (`c_7`) and
  `updated_v_cb` (`c_8`) under their typecast configs** — the writer moves to `c_12`/`c_13`, leaving compute as
  the only toucher (produces at `running_statistics_sfpu_kernel.cpp:162,183` / `:264,281`, consumes inside
  `maybe_typecast_stat` at `:20,39`).
- **assign cosmetic 1P+1C:** `old_running_mean_tensor_cb` (`c_3`) and `old_running_var_tensor_cb` (`c_4`) under
  the *absent* config — same reasoning as `c_5`/`c_6` above.

**Roll-up:** legal 1:1 ×13–15 · self-loop ×5–8 · cosmetic 1P+1C ×0–4 · **multi-binding ×0** · **dead-CB drop ×0**
(counts vary with config).

### Hardware-config translation — five points, four of which fail *silently* if copied straight across

0. **The data-movement configs are empty today and become explicit Gen1 configs.** Both factories set
   `reader_desc.config = ReaderConfigDescriptor{}` and `writer_desc.config = WriterConfigDescriptor{}` —
   default-constructed, no fields (`batch_norm_program_factory.cpp:337,346`,
   `running_statistics_program_factory.cpp:379,388`). In Metal 2.0 these become a
   `DataMovementHardwareConfig` built by the Gen1 helpers **`CreateReaderGen1DataMovementConfig()`** and
   **`CreateWriterGen1DataMovementConfig()`** (`tt_metal/api/tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp:58,88`).
   Mechanical — four sites, no fields to carry across. *Use those exact names: an earlier revision of the docs
   called them `CreateReader1xxDataMovementConfig()` / `CreateWriter1xxDataMovementConfig()`, which never existed
   in the headers; the recipe fixed this in `40b61b016a1`, so current docs and headers now agree.*

1. **`unpack_to_dest_mode` re-keys and re-means.** Both factories build a `std::vector<UnpackToDestMode>` of
   length `NUM_CIRCULAR_BUFFERS` and set selected slots to `UnpackToDestFp32`
   (`batch_norm_program_factory.cpp:352-368`, `running_statistics_program_factory.cpp:394-411`). Metal 2.0's
   `ComputeHardwareConfig::unpack_modes` is keyed by **DFB name**, not CB id, and the value's sense flips.
   **Translate entry by entry; do not memcpy the vector.**
   **Mind the conditional entry:** `batch_norm_program_factory.cpp:365-367` adds `output_tensor_cb` to the set
   **only when `needs_output_typecast`** (landed in #51313 — `c_2` holds `Float32` there and is unpacked at
   `batch_norm_sfpu_kernel.cpp:171`). The ported config must stay conditional on the same predicate.
2. **`dst_full_sync_en` inverts.** Both factories set `ComputeConfigDescriptor::dst_full_sync_en` from
   `get_compute_kernel_config_args` (`batch_norm_program_factory.cpp:397`,
   `running_statistics_program_factory.cpp:447`). The Metal 2.0 field is **`double_buffer_dest` with the opposite
   polarity** — a straight copy is silently wrong.
3. **Two compute-kernel DFB handles are chosen by a *runtime* ternary.**
   `auto dfb_affine_or_out = (weight_has_value || bias_has_value) ? dfb_tmp_1 : dfb_output_0;` and
   `auto dfb_scaled_output = (bias_has_value) ? dfb_tmp_1 : dfb_output_0;`
   (`batch_norm_kernel.cpp:31-32`, `batch_norm_sfpu_kernel.cpp:42-43`), inside `batchnorm_bcast_tiles` whose
   `weight_has` / `bias_has` parameters are plain runtime `uint32_t` even though every caller passes a
   `constexpr`. This ports fine — `dfb::name`'s `constexpr operator uint32_t()` makes both ternary arms
   `uint32_t` — but it means **the compute kernel must bind both `temp_1_cb` and `output_tensor_cb`
   unconditionally.** Do not narrow the bindings to whichever arm a given config selects.
4. **A local compute helper takes `DataflowBuffer&`:** `maybe_typecast_stat(DataflowBuffer& src_obj, ...)`
   (`running_statistics_sfpu_kernel.cpp:15-18`). In-file `ALWI` helper, **not a donor**, so it does not hit the
   donor table's `CircularBuffer&` flag — update the signature alongside the kernel.

### RTAs and CTAs

- **Name every runtime arg.** All eight kernels read args at literal constant indices (reader `0..8`, batch-norm
  writer `0..11`, running-stats writer `0..10`, compute `0..2` / `0`). No counted loop, no `arg_index++`, no
  data-selected index. **There are zero vararg cases in this op** — do not reach for the vararg mechanism.
- **Eight of the pushed RTAs are already dead — do not carry them into the schema.** `cHt` and `cWt` are pushed
  to every dataflow kernel and never read: reader slots 9-10 (`batch_norm_program_factory.cpp:98-99`,
  `running_statistics_program_factory.cpp:96-97`), batch-norm writer slots 12-13 (`:123-124`), running-stats
  writer slots 11-12 (`:120-121`). The `num_reader_args = 11` / `num_writer_args = 14` / `13` idle-core zero-fill
  constants (`batch_norm_program_factory.cpp:61-62`, `running_statistics_program_factory.cpp:60-61`) encode the
  same inflated counts and shrink with them. *(Dropping unread args is behaviour-preserving; if you prefer strict
  zero-diff, carry them and note it — but do not name them as though they were live.)*
- **Size the compile-time-arg schema per compute variant.** Each factory pushes one CTA list consumed by both
  compute files: `batch_norm_kernel.cpp` reads indices `0..10` of 15 pushed
  (`batch_norm_program_factory.cpp:370-385`); `running_statistics_kernel.cpp` reads `0..13` of 19
  (`running_statistics_program_factory.cpp:416-435`). The SFPU siblings read the full lists.

## Watch for

- **CB endpoints (multi-binding):** **none.** I ran the three-face hunt explicitly: no hidden second writer (the
  op has **no semaphores at all**, so the semaphore-gated raw co-fill face cannot occur), no multi-reader CB, and
  no dual-instance work-split (each factory instantiates each kernel source exactly once; reader/writer/compute
  are three *distinct* sources over one core range). The raw-pointer writes that do exist —
  `fill_tile_with_first_element*(dfb_*.get_write_ptr())` at
  `writer_batch_norm.cpp:89,91,101,103,112,114,124,126` and `writer_running_statistics.cpp:95,97,124,126` — are
  performed by the **same kernel that FIFO-produces that CB**, so they are same-binding peeks and add no toucher.
  If your own census disagrees, re-check against the audit's table before setting the flag.
- **Cross-op / shared kernels: none — no fork to reuse, none to create, no sunset list.** The op **owns all 8
  kernel sources and no other op binds any of them** (census run; the only outside hits were
  `ttnn/ttnn.egg-info/SOURCES.txt`, a build artifact the disambiguation rule discards). So none of the
  shared-kernel rungs apply — not borrowed, not lent, not intra-op. The locational `_metal2` sibling check was
  run on all four relevant directories and found nothing, so **you are not reusing a fork and you should not
  create one.**

  Three donor **headers** are consumed by `#include` — a function-call escape, a different mechanism from
  file-path sharing — and **none needs an edit for this port**:
  - `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp` — cross-family and
    *broadly* shared (~35 kernel files across `eltwise/binary_ng`, `eltwise/ternary`,
    `experimental/quasar/binary_ng`). Every function this op calls takes a bare `uint32_t l1_write_ptr` — no
    buffer handle, no tensor handle — so there is nothing to translate. Pass `dfb::name.get_write_ptr()` and move
    on. **Do not "modernize" this header** — you would break ~35 co-borrowers.
  - `ttnn/cpp/ttnn/kernel/dataflow/cb_fill_helpers.hpp` — `fill_cb_with_value(uint32_t cb_id, ...)`.
    `dfb::name`'s constexpr cast covers the call site unchanged.
  - `ttnn/cpp/ttnn/kernel/compute/dest_format_helpers.hpp` — `pack_tile_with_dt`, `*_tiles_init_with_dt`,
    `ckernel::{add,sub,mul}_tiles_to_cb`, all `uint32_t icb`-shaped.

  Both `ttnn/cpp/ttnn/kernel/` donors are still `CircularBuffer`-native *internally* even though your kernels are
  now DFB-native. That is invisible at the boundary (they take plain `uint32_t` ids) and is **not** yours to fix —
  it is a tidy-up for the kernel-pool owners.
- **`experimental/quasar/` is out of bounds.** There is no quasar copy of `batch_norm`, so you are unlikely to
  trip over one — but if a grep surfaces a `*_metal2.cpp` under that tree, it is not a fork to bind, not a
  naming source, and not evidence that a construct is portable.
- **RTA varargs:** none — name every arg (see above).
- **Three latent bugs in the non-default compute path — do NOT fix them in the port diff.** The audit found real
  defects in `running_statistics_kernel.cpp`: a `push_back` on `dfb_out0` with **no matching `reserve_back`**
  (`:57-59`); a nested `tile_regs_acquire()` bracket whose `pack_tile(0, dfb_out0)` packs whatever DST reg 0 holds
  after the inner helpers' releases (`:40-58`); and, when both running-stat tensors are absent, an output tile
  packed from undefined DST. They are masked today because the SFPU sibling is the default path. **These route to
  the ops team and are out of port scope** — the port makes no functional changes. Flagged here only so you do not
  mistake the odd FIFO shape for something you introduced, and do not "tidy" it: port the missing reserve as-is.
  Full write-up in the audit's *Misc anomalies* (items 1-3).
