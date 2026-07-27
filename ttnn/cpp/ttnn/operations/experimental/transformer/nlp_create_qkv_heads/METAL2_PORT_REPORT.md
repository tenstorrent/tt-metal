# Port Report — `experimental/transformer/nlp_create_qkv_heads`

*Captured during the port; committed alongside the port code and the audit/brief/plan.*

## Outcome

**PORTED** — both factories (`Interleaved` and `Sharded`) converted to `MetalV2FactoryConcept`
(`create_program_artifacts`). The confirmed no-regression test set passes identically pre- and post-port:
**74 passed, 37 deselected** (`test_nlp_create_qkv_heads.py`, both counts unchanged; the deselected 37 are the
sibling `nlp_create_qkv_heads_falcon7b` tests). Coverage includes interleaved (with/without transpose_k_heads,
with/without a KV tensor, GQA, and fp32/bf16/bf8) and sharded (incl. GQA 16q/1kv and 32q/2kv, fp32/bf16/bf8,
and both program-cache tests).

## Provenance

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` for **both** factories. `NlpCreateHeadsDeviceOperation::{Interleaved,Sharded}::create_descriptor`
→ `create_program_artifacts` returning `ttnn::device_operation::ProgramArtifacts`. The `program_factory_t`
variant now has both alternatives on Metal 2.0; the device-op-class methods (`validate_*`, `compute_output_specs`,
`create_output_tensors`, `select_program_factory`) were left untouched.

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** (the op never defined one).
- Pybind entry points removed: **none** (`nlp_create_qkv_heads_nanobind.cpp` binds only the top-level op; no
  factory entry point was pybound, so nothing had to be removed).

### Open items
- **Tensor-arg relaxation candidates:** none applied (kept strict). Not obviously needed here.
- Both factories stayed on the default concept with no device-op edits — the success case.

## Handoff points

- **Cross-op shared compute kernel — `transpose_wh.cpp` (forked).** The Interleaved factory's `transpose_k_heads`
  path binds `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp`, a shared compute-pool donor **outside** the op
  directory whose other consumers (`nlp_create_qkv_heads_boltz`, `nlp_create_qkv_heads_vit`,
  `split_query_key_value_and_split_heads`) are **not** yet on Metal 2.0. Per the shared-dataflow-kernel Caution,
  consumers cannot co-migrate → **fork**. The fork lives in-scope at
  `device/kernels/compute/transpose_wh_metal2.cpp` (all writes stay inside the op directory; the shared original
  is untouched). **Sunset:** delete the fork and re-point this factory at the shared kernel once the last legacy
  consumer ports (and the shared kernel is itself converted). Owner: the porters of the three remaining consumers.

## Successes

- **Borrowed-DFB placement avoided the known id-hole framework bug ([[metal2-borrowed-dfb-id-hole-bug]]).** The
  sibling `nlp_create_qkv_heads_decode` port hit a garbage borrowed-DFB write pointer when a node's present
  DFB-id set had an *interior hole* (a middle DFB absent from that node). This op's Sharded structure — one source
  instantiated reader-config + writer-config, **both on `q_cores`** — makes all three output DFBs (q_out=0,
  k_out=1, v_out=2) derive a `q_cores` node set, so every `q_core` hosts the contiguous id set `{0,1,2}` with no
  interior hole. The GQA sharded tests (`16q/1kv`, `32q/2kv`, where `q_cores ⊃ k_cores`) pass with PCC ≥ 0.9999.
  The port-plan flagged the borrowed-backing-on-`q_cores\k_cores` risk as a possible capitulation; it did **not**
  materialize — the framework resolves the borrowed base correctly and the kernel simply never touches k_out/v_out
  on the extra cores (runtime `read_kv_heads` guard). *Recipe/patterns success:* the two-toucher (c_16 → 1P+1C)
  and self-loop (c_17/c_18) dispositions from the brief matched the kernel-touch census exactly.
- **Case 2 `get_bank_base_address` bridge** (whitelist rule 5) fit the Sharded reader/writer cleanly — the raw NoC
  walk on the input shards was preserved byte-for-byte; only the base RTAs (slots 6/15) moved to `tensor::in_q` /
  `tensor::in_kv` bindings.
- **Conditional bindings** (whitelist rule 6) worked as documented for the Interleaved transpose/KV paths — the
  legacy `TRANSPOSE_K_HEADS` / `READ_FROM_INPUT_TENSOR_KV` defines drive both host-side conditional DFB/tensor
  bindings and kernel-side `#ifdef`-gated `dfb::k_in`/`tensor::in1` aliases; no unconditional-binding workaround
  needed.

## Friction

### Gaps
- **The reference port (accumulation, `akertesz/porting-experiment-accumulation-jun10`) is substantially stale
  against the current headers** — enough to mislead on the load-bearing details: it uses `create_program_spec`
  (not `create_program_artifacts`), `DataMovementHardwareConfig{.role = RoleHint::READER}` (the current header has
  no `.role`; the arch-agnostic `ttnn::create_reader_datamovement_config(arch)` is the current path), the old
  `ComputeHardwareConfig{.math_fidelity/.fp32_dest_acc_en/.dst_full_sync_en/.math_approx_mode/.unpack_to_dest_mode}`
  field names (now `ComputeGen1Config{fpu_math_fidelity/enable_32_bit_dest/double_buffer_dest/sfpu_precision_mode/
  unpack_modes}` with a bool→enum for unpack and a *node-first* vs *name-first* RTA-table shape difference), and
  `std::cref(tensor)` in `tensor_args` (current form passes `tensor.mesh_tensor()` directly). The recipe *does*
  warn that pre-completion ports aren't current best practice, but the accumulation port is singled out as the
  "first worked end-to-end reference," so a porter who leans on it will write code that doesn't compile. The
  reliable references were the **current headers** under `tt_metal/api/.../metal2_host_api/` and a **currently-on-
  branch** ported op (`experimental/quasar/interleaved_to_sharded`). Suggest the recipe point at a current op for
  shape, or refresh the accumulation reference.
- **Kernel header count.** The whitelist says a port "adds exactly two headers" (`experimental/kernel_args.h`,
  `api/dataflow/dataflow_buffer.h`). The Sharded kernel additionally needed `api/tensor/tensor_accessor.h`, because
  it had **no** `TensorAccessor` before the port (its inputs were raw-NoC Case 2) and now constructs
  `TensorAccessor(tensor::in_q)` for the `get_bank_base_address` bridge. Minor, but the "exactly two" phrasing
  doesn't cover a kernel that newly adopts `TensorAccessor`.

### Confusion
- **"KernelRunArgs must be specified for ALL kernels" vs. "may be omitted if no RTAs."** `program_run_args.hpp`
  says a `KernelRunArgs` is required for every kernel; the recipe says it may be omitted when the kernel has no
  RTAs. The Interleaved compute kernels carry only a CTA (`NHtWt`). I included empty `KernelRunArgs{.kernel = …}`
  entries for them to satisfy the header's stronger statement; tests pass. Worth reconciling the two docs.

## Open items for downstream

- **Cross-op kernel fork** (coordination signal for the next sibling-op porter): forked
  `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp` → `.../nlp_create_qkv_heads/device/kernels/compute/transpose_wh_metal2.cpp`
  (CB→DFB + `dfb::k_in`/`dfb::k_out` + named `NHtWt` CTA). Remaining unmigrated consumers of the shared original:
  `nlp_create_qkv_heads_boltz`, `nlp_create_qkv_heads_vit`, `split_query_key_value_and_split_heads`. Delete the
  fork when the last of these ports.
- **RTA→CRTA cleanup (separate pass, not port work).** In the Sharded factory, several per-node RTAs carry the
  *same* value on every node — `head_size`, `num_x`, `num_q_heads_per_core`, and especially the NoC-coordinate
  **varargs** (identical across all nodes). These are really common runtime args (CRTA / `common_runtime_varargs`).
  Left as per-node RTAs to preserve the legacy dispatch semantics faithfully; flag for a later name-first / RTA→CRTA
  cleanup. (Retained varargs are legitimate here — the NoC-coord arrays are genuine runtime-indexed collections,
  count = `num_cores_x`/`num_cores_y`, indexed by `q_x`/`kv_x`.)
- **Borrowed-DFB node-set observation (framework note).** The Sharded k_out/v_out DFBs are placed on all of
  `q_cores` (the reader/writer grid) even though their backing K/V output tensors are sharded only on `k_cores`
  (⊆ `q_cores`). It works — but it means a borrowed DFB is instantiated on nodes where its backing tensor has no
  shard (the kernel never touches it there). If the framework ever tightens borrowed-DFB validation to require the
  backing tensor to cover the DFB's whole node set, this op (and the `nlp_create_qkv_heads_decode` family) would
  regress. Same area as [[metal2-borrowed-dfb-id-hole-bug]]; worth the runtime team's awareness.
- **Triage doc stale (already noted by audit):** `analyses/2026-07-19_offset_base_pointers.md` over-lists this op
  (and `_boltz`) as Type-1; the fold was removed in `86872e0a06a`. For the triage-doc owner.
