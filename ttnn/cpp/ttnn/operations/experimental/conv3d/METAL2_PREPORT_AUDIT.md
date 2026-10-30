# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/conv3d`

- **`Conv3dDeviceOperation`** (`device/conv3d_device_operation.hpp:21-34`; `program_factory_t = std::variant<Conv3dProgramFactory>` at :26)
  - `Conv3dProgramFactory` (`device/conv3d_program_factory.cpp:255`, `create_descriptor`). One program over the `compute_with_storage_grid_size` rectangle with three kernels, all owned by this op:
    - reader `device/kernels/reader_vol2col.cpp` (`ReaderConfigDescriptor`, factory :927-932)
    - compute `device/kernels/compute.cpp` (factory :1017-1028)
    - writer `device/kernels/writer.cpp` (`WriterConfigDescriptor`, factory :1065-1070), carrying the kernel_lib weight-multicast channel attached at :1452-1457
- Host-only code outside the device op: `conv3d.cpp` (the user-facing wrapper, which prepares weights) and `prepare_conv3d_weights.cpp`. Neither builds a program.
- No unreferenced kernel files. The only other file in `device/kernels/` is `conv3d_gather_tuning.hpp`, a constants header that the reader and the factory share.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** ``c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form``

**Readiness sheet:** the *Operations analysis* sheet (file ID `1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`). The launching session fetched it at **2026-10-30 12:21 UTC** (`download_file_content` as `text/csv`, base64-decoded) into `analyses/ttnn_op_porting_readiness.csv`. This audit read that copy and did not fetch again. The code was read on branch `iwrosz/conv3d_2.0` at HEAD `bb9c7d974bc`, which is recipe branch `c41f7f4ef0a` with `origin/main` `012c19beaae` merged in.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/conv3d` |
| **Overall** | **RED**: blocked on Device 2.0 (isolated holdovers). No portable subset. |
| **DOps / Factories** | `Conv3dDeviceOperation` → `Conv3dProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **No.** RED, routed to the Device 2.0 track. These are **isolated holdovers**, not broad Device 1.0: 3 constructs, 2 in conv3d's `reader_vol2col.cpp` and 1 in pool's shared `experimental_device_api.hpp`, which the reader calls. `writer.cpp`, `compute.cpp` and the kernel_lib helpers are clean. |
| *Prereqs* — Cross-op escapes | Inventory skipped: whole-op RED with no portable subset; re-audit on unblock. The one donor that gates (pool `experimental_device_api.hpp`) is named under Device 2.0. |
| *Feature Support* — overall | GREEN: no Appendix A entry fires (all three are `N/A`). |
| *Feature Support* — Variadic-CTA | Ok. This is not an Appendix A entry (see Recipe notes). |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (`yes`). The primary cross-check is clean. |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Execution Model | `SPMD`. The concept is not mesh-shaped, so `Secretly SPMD Workload?` does not apply. |
| *TTNN Readiness* — Custom hash | No `compute_program_hash`. There is a backdoor `Conv3dConfig::attribute_values()` (`device/conv3d_device_operation_types.hpp:71-84`; the sheet records it as `(complete)`). It does not gate, and the port leaves it intact. |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | The sheet's `Porting Target` reads `ProgramSpecFactoryConcept`. Recorded verbatim; the TTNN porting shape subject was skipped. |
| *Port work* — Offset base pointer | None. GREEN: all 7 address RTAs carry clean bases. |
| *Port work* — Tensor bindings (per binding) | Skipped (subject not run): whole-op RED with no portable subset. The offset scan found all 7 address RTAs consumed as `TensorAccessor` bases. |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | None. No accessor passes a 3rd argument. |
| *Port work* — CB endpoints | Skipped (not censused): whole-op RED with no portable subset; re-audit on unblock. |

No CB census was taken. CB endpoints is the porter-only subject that the Red scoping rule defers here.

## Result

**RED → blocked on Device 2.0.** This is an op issue ("fix the op"), routed to the Device 2.0 migration track. Data-movement code that the op uses still has three Device 1.0 holdovers:

1. A raw `noc_async_read` in conv3d's own reader (`reader_vol2col.cpp:105-106`).
2. A raw `volatile` L1 read in the same reader (`reader_vol2col.cpp:902-904`).
3. A raw `noc_async_read_set_trid` inside `experimental::set_read_trid`, in pool's shared `experimental_device_api.hpp:81`. The reader's trid ring calls it.

**RED at op level; no portable subset.** The op has one factory and one kernel set. All three holdovers sit in CTA-gated branches of the single reader kernel (DRAM read staging, logical-pad mask mode, and the trid ring). None of them is in a separable factory or host-side code path. A port converts the reader file as a whole, so it cannot carry only the clean configurations.

Every other gate-bearing subject clears today:

- Feature compatibility: every entry is `N/A`.
- TTNN factory concept: `yes`, and the cross-check is clean.
- Offset base pointers: no fold.
- TensorAccessor 3rd argument: no site.

**Path forward:** land the Device 2.0 fix for the three constructs as its own PR, test it, then re-audit. The re-audit should be cheap because nothing else gates today.

## Gate detail

- **TTNN factory concept (`Is able to port?`): GREEN.** The sheet has one row for this op (`experimental/conv3d` / `Conv3dDeviceOperation` / `Conv3dProgramFactory`), and its `Is able to port?` cell reads `yes`. None of the blocking columns fires:
  - `Runtime-args update (get_dynamic_runtime_args)` = `no`
  - `Smuggled pointer` = `no`
  - `Known op issues` is blank
  - `TensorParameter relaxation` = `none`
  - `Concept` = `descriptor`

  The primary-column cross-check matches the code (see Team-only → TTNN factory analysis), and the cross-column invariants hold.

- **Device 2.0 (every kernel used): RED. These are isolated holdovers, not broad Device 1.0.** The kernels the op uses are:
  - its own `reader_vol2col.cpp`, `writer.cpp` and `compute.cpp`
  - pool's `ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp`, a cross-family donor header that the reader and writer include
  - kernel_lib `mcast/kernel/mcast_args.hpp` and its pipes (writer)
  - kernel_lib `tilize_helpers.hpp`, `untilize_helpers.hpp` and `dest_helpers.hpp` (compute)

  All three kernels do the bulk of their work through `Noc`, `CircularBuffer`, `Semaphore<>` and `TensorAccessor` endpoints. (`experimental::CB` is a plain alias for `CircularBuffer`, `experimental_device_api.hpp:24`.) `writer.cpp`, `compute.cpp` and the kernel_lib helpers are clean. The violations are:

  | # | File | Line | Call | Wrapper in scope | Compiled in when |
  |---|---|---|---|---|---|
  | 1 | `ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/reader_vol2col.cpp` (own) | 105-106 | `noc_async_read<NOC_MAX_BURST_SIZE + 1, true>(aligned_src_noc_addr, scratch_l1_addr, aligned_read_size, noc.get_noc_id())`. This is a raw read. The source is a 64-bit NoC address taken from `reader.get_noc_addr(page_idx, c_in_offset_bytes, …)` and aligned down by hand (:96-99). The destination is a raw L1 address rounded up from `scratch_cb.get_write_ptr()` (:101-102). | `Noc noc`, `scratch_cb` (`CircularBuffer`), `reader` (`TensorAccessor`) | DRAM read staging: CTA 40 `enable_dram_read_staging` (factory :99-100, :912), with an interleaved-DRAM input accessor (`reader_vol2col.cpp:134`) |
  | 2 | same file | 902-904 | `reinterpret_cast<volatile tt_l1_ptr uint32_t*>(off_l1)`, followed by `off_p[0]` and `off_p[1]`. This reads the `[h_start, w_start]` offset page through a raw pointer. | `pad_off_cb` (`CircularBuffer`) | Mask mode: CTA 43 `mask_mode` (factory :109-110, :915) |
  | 3 | `ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp` (donor, owned by the **pool** family) | 81 | `experimental::set_read_trid` → `noc_async_read_set_trid(trid, noc.get_noc_id())`. This sets a sticky read tag that later untagged `noc.async_read` calls inherit. It is called from `reader_vol2col.cpp:413` and `:562`. | `Noc noc` | Trid ring: CTA 36 `gather_trids != 0` (factory :855-862, :908) |

  The migration guide has a Device 2.0 form for each one:
  - #1: `noc.async_read` with an endpoint (section *Async Read*).
  - #2: `CoreLocalMem<uint32_t>` (section *Safe Local Memory Access*).
  - #3: a per-call `NocOptions::TXN_ID` with `NocOptVals{.trid}` (section *Transaction ID Support*). The ported padded_slice reader already uses this form in tree (`experimental/padded_slice/device/kernels/dataflow/padded_slice_reader_rm_interleaved_start_id.cpp:166-172`).

  **Sizing notes for the Device 2.0 team:**
  - Each holdover compiles only under its own CTA, so the fix has to exercise DRAM read staging, mask mode and the trid ring separately. Tests live in `tests/ttnn/unit_tests/operations/conv/test_conv3d.py` and `tests/ttnn/nightly/unit_tests/operations/conv/test_conv3d.py`.
  - #1 and #2 are local, mechanical replacements.
  - #3 is not a pure call-site swap:
    - Device 2.0 has no sticky-trid primitive; the trid rides on each call.
    - The ring in `gather_rows_to_shard` (reader :350-572) relies on the sticky tag reaching reads issued inside helpers (`read_input_row*`, :41-140).
    - It also reaches the zero-pad reads of `noc.async_write_zeros` (`zeroPad`, :36-39). That function takes no `NocOptions` (`tt_metal/hw/inc/api/dataflow/noc.h:826`), and on Gen1 it issues plain loopback reads (`tt_metal/hw/inc/internal/tt-1xx/noc_zero_l1.inl:13-43`).
    - The helper is shared. The `transformer/sdpa` sparse gathers call it too (`sparse_sdpa_gather.hpp:90`, `:100`; `sparse_sdpa_msa_gather.hpp:34`, `:48`), so a donor-side change reaches them as well.

  **Observed but not counted** toward the gate. These use the Device 2.0 API surface; they are listed so the Device 2.0 team can confirm:
  - **Hand-built loopback `UnicastEndpoint` from `my_x`/`my_y`.** This is the guide's "Avoid (Device 2.0 API, but hand-built loopback endpoint)" form.
    - Pool helper `experimental::local_addr` (`experimental_device_api.hpp:37-39`), plus the `set_read_state`/`read_with_state` wrappers built on it (:43-75).
    - Reader call sites :113-119, :256, :265-266, :271, :290-300, :646, :659 and :661-667.
    - The kernel_lib mcast pipes do the same for their local copy (`kernel_lib/mcast/kernel/pipes/chain_pipe.inl:120-125`, `mcast_pipe.inl:355-363`).
    - Why it is not counted: it is Device 2.0 API, and a ported Metal 2.0 kernel already in tree uses the same helper (the padded_slice reader above, :170). The framework's own Gen1 `Noc::async_write_zeros` also builds the same loopback (`noc_zero_l1.inl:23-25`). See Questions.
  - **Raw pointer over the runtime-arg region:** `writer.cpp:117` and `:119` (`(tt_l1_ptr uint32_t*)(get_arg_addr(argidx))` for the worker-coordinate lists). kernel_lib `mcast/kernel/mcast_args.hpp:282-284` uses the same pattern. Argument access is outside the guide's surface, and a Metal 2.0 port reads these through the vararg API anyway.
  - **Sanctioned, not flagged:** `get_tile_size(cb_weight_tiled)` and `get_tile_size(cb_matmul_interm_tiled)` at `writer.cpp:122-123`.

- **Feature compatibility: GREEN (no gate fired).**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | None of the signals appears in the op, its pool donor header, or the kernel_lib mcast library: no `GlobalCircularBuffer` type, no `global_circular_buffer` field, no `CreateCircularBuffer(…, global_cb)`, no `.remote_index(`, no `remote_circular_buffer.h`. The 14 possible `CBDescriptor`s (factory :397-635) set only `total_size`, `core_ranges` and `format_descriptors`. The `remote_cb` / `remote_cb_obj` identifiers in `compute.cpp:257-291` and `:383-424` are only a parameter name. They refer to the plain reduction CB (`cb_reduction_tiled`, factory :461-470), which holds the partial sums read from worker cores. The name cue matched, but the disambiguator guard clears it. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `CBDescriptor` sets `.address_offset`. There is no `set_address_offset`, no 4-argument `UpdateDynamicCircularBufferAddress`, and no `cb_descriptor_from_sharded_tensor`. No CB borrows a buffer. |
  | GlobalSemaphore | N/A | The op has one plain `SemaphoreDescriptor` (factory :810-815). The mcast library adds plain descriptor semaphores (`kernel_lib/mcast/host/program_adapters/mcast_descriptor_adapter.cpp:66-74`). There is no `GlobalSemaphore` type or include. |

- **CB endpoints (GATE-free):** skipped because of the whole-op RED with no portable subset; re-audit on unblock. Its precondition would hold, because the holdovers are isolated and the CB idioms are intact.

- **Offset base pointers: GREEN.** No address RTA folds a host-side offset into its base. conv3d is not in the triage tables (`2026-07-19_offset_base_pointers.md`), which agrees with this scan.

  | Kernel + role | Host source (factory) | Kernel use | Fold? |
  |---|---|---|---|
  | reader RTA `in_addr` (position 0) | `Buffer*` `input_tensor.buffer()`, :1072 → :1360 | `TensorAccessor(in_args, in_addr)` (reader :867, :884) | none |
  | reader RTA `halo_addr` (halo mode, trailing) | `Buffer*` halo buffer, :1073 → :1372 | `TensorAccessor(halo_args, halo_addr)` (:879, :887) | none |
  | reader RTA `pad_offset_addr` (mask mode, trailing) | `Buffer*` pad-offset tensor, :1074 → :1375 | `TensorAccessor(offset_args, pad_offset_addr)` (:880, :894) | none |
  | writer RTA `out_addr` (position 0) | `Buffer*` output, :1075 → :1399 | `TensorAccessor(out_args, out_addr)` (writer :68, :124) | none |
  | writer RTA `weight_addr` (position 1) | `Buffer*` weight, :1076 → :1400 | `TensorAccessor(weight_args, weight_addr)` (:69, :125) | none |
  | writer RTA `bias_addr` (position 2) | `Buffer*` bias, or literal `0` when absent, :1077 → :1401-1405 | `TensorAccessor(bias_args, bias_addr)` (:70, :126) | none |
  | writer RTA `weight_lo_addr` (position 3) | `Buffer*` weight_lo, or literal `0` when absent, :1078 → :1406-1410 | `TensorAccessor(weight_lo_args, weight_lo_addr)` (:71, :127) | none |

  No address is baked into a compile-time argument; the CTAs carry only `TensorAccessorArgs` (:919-925, :1058-1063). The mcast runtime payload holds topology words and no buffer address (`kernel_lib/mcast/host/mcast_impl.cpp:585-596`). The kernels do apply offsets, but on-device to a clean base: `c_in_offset_bytes`, `c_out_block * C_out_block_bytes`, and the staged reader's align-down at reader :96-99.

- **TensorAccessor 3rd argument: N/A.** No accessor the op uses passes a 3rd argument, so the subject never fires. There are 7 constructions, all two-argument: reader :884, :887, :894 and writer :124-127. conv3d is not in the triage table (`2026-07-06_tensor_accessor_3rd_arg_triage.md`).

## Port-work summary *(mirrors the brief)*

Skipped: whole-op RED with no portable subset, so no brief is issued. The re-audit after the Device 2.0 fix produces this section.

## Heads-ups *(mirrors the brief)*

Skipped, for the same reason.

## Team-only

- **Out-of-directory coupling & donor shape:** skipped (whole-op RED, no portable subset; re-audit on unblock). The one donor that gates is named under Device 2.0. File-path coupling: the factory instantiates only the op's own three kernel files and borrows none.
- **Relaxation candidates:** none to mine. There is no custom `compute_program_hash`.
- **TTNN factory analysis.** Sheet values compared with the code:

  | Sheet column | Sheet value | Code evidence | Match |
  |---|---|---|---|
  | `Concept` | `descriptor` | `Conv3dProgramFactory::create_descriptor(const Conv3dParams&, const Conv3dInputs&, Tensor&)` returns a `ProgramDescriptor` (`device/conv3d_program_factory.hpp:77-80`, `.cpp:255-256`). It takes no `MeshCoordinate`, and there is no `create_workload_descriptor`. | ✓ |
  | `Custom hash (compute_program_hash)` | `no` | The device op has no `compute_program_hash` (`device/conv3d_device_operation.hpp:21-34`). | ✓ |
  | `Backdoor custom hash (attribute_values / to_hash)` | `(complete)` | `Conv3dConfig::attribute_values()` lists all 11 members (`device/conv3d_device_operation_types.hpp:71-84`). Not on the cross-check list and not a gate; the port leaves it alone. | — |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | No such hook. | ✓ |
  | `Override runtime args method?` | `no` | No `override_runtime_arguments`. | ✓ |
  | `Pybind descriptor` | `no` | `conv3d_nanobind.cpp` binds only the `ttnn.experimental.conv3d` function (:26-73), `prepare_conv3d_weights` (:76-94) and the `Conv3dConfig` attribute struct (:96-149). It binds no `create_descriptor` and no device-op class. | ✓ |
  | `Secretly SPMD Workload?` / `Why secretly SPMD?` | blank | Does not apply: the concept is not mesh-shaped. | ✓ |
  | Factory-set match | one row, `Conv3dProgramFactory` | `program_factory_t = std::variant<Conv3dProgramFactory>` (`device/conv3d_device_operation.hpp:26`) | ✓ |
  | `Smuggled pointer` (a blocking signal) | `no` | The op has no `->address()` anywhere. Every tensor address rides a `Buffer*` `RTArgList` entry, which the framework patches on a cache hit (factory :1072-1078, :1360-1410). This is consistent with `Op Classification` = `PD Op (pointer-patching)`. | ✓ |
  | Invariants | `get_dynamic_runtime_args` `no`; `Op-owned tensors?` blank | Consistent with a `descriptor` row. | ✓ |

  For the record, the other cells read: `Diego validation` `yes`, `Pointer patching perf issue?` `OK (fixed latent bug)`, `Formerly custom hashed?` `yes`.

## Misc anomalies

Skipped (Incidental anomalies subject): whole-op RED with no portable subset; re-audit on unblock.

## Subjects skipped under the Red scoping rule

The blocker clears **on the op-code side**: the Device 2.0 migration rewrites conv3d's reader kernel and a helper it calls. The recipe lists "a Device 2.0 migration" as op-code side, so the seven purely-informational subjects are skipped. Recipe notes discuss whether that fits isolated holdovers.

- **TTNN porting shape:** skipped (whole-op RED, no portable subset; re-audit on unblock). The sheet's `Porting Target` reads `ProgramSpecFactoryConcept`, recorded verbatim and not analyzed.
- **TensorParameter relaxations:** the cell (`none`) was read for the TTNN gate and clears. No analysis doc applies, and there is no custom hash to mine.
- **TensorParameter analysis:** skipped (whole-op RED, no portable subset; re-audit on unblock).
- **CB endpoints:** skipped (whole-op RED, no portable subset; re-audit on unblock).
- **Out-of-directory coupling:** skipped (whole-op RED, no portable subset; re-audit on unblock).
- **RTA varargs:** skipped (whole-op RED, no portable subset; re-audit on unblock).
- **Incidental anomalies:** skipped (whole-op RED, no portable subset; re-audit on unblock).

## Questions for the user

1. **Do the guide's "Avoid" forms count toward the Device 2.0 gate?** The Device 2.0 guide marks a hand-built loopback `UnicastEndpoint` as "Avoid (Device 2.0 API, but …)". This audit did not count it; the sites are listed under Device 2.0 → "Observed but not counted". If the Device 2.0 team counts it, about ten more reader call sites and the pool helpers `local_addr` / `set_read_state` / `read_with_state` join the fix. The verdict is RED either way; only the size of the fix changes. The pool header is included by about 45 kernel files across about ten op families, including an already-ported Metal 2.0 kernel (padded_slice), so a "yes" reaches well beyond conv3d.
2. **Run the seven informational subjects now?** The Device 2.0 fix touches about four lines (`reader_vol2col.cpp:105-106`, `:902-904`, and the trid helper). None of the seven subjects keys on those lines, so a census taken now would likely survive the fix. That includes the CB endpoints census, tensor bindings, RTA varargs and the mcast vararg layout. They were skipped per the rule; ask for them if you want port planning to run in parallel with the Device 2.0 PR.

## Recipe notes

1. **Device 2.0 sizing taxonomy.** *Isolated holdovers* is defined only as "CB-index-keyed free-function holdovers". This op's holdovers have other shapes, though they are just as local: a raw `noc_async_read`, a raw `volatile` L1 read, and a raw `noc_async_read_set_trid` inside a donor helper. I classified them as isolated, with idioms intact. The bullet could cover "a small number of raw Device 1.0 calls in an otherwise Device 2.0 kernel".
2. **The guide's "Avoid (Device 2.0 API, but …)" forms** (a hand-built loopback endpoint, a raw-address `AllocatorBank`). The recipe does not say whether these count as violations. I did not count them: ported Metal 2.0 kernels in tree use the loopback helper, and the framework's Gen1 `async_write_zeros` builds the same loopback. One sentence in the gate's Green bullet would settle it.
3. **The skip rule and isolated Device 2.0 REDs.** The rule files "a Device 2.0 migration" under op-code side, so the subjects are skipped. Here the fix touches about four lines, and none of the seven subjects keys on them: no CB, binding, RTA or coupling changes. So the staleness argument barely applies. The CB-endpoints precondition already separates isolated from broad Device 2.0 REDs, and the skip rule could use the same split. Two of the seven subjects are also free sheet lookups (`Porting Target`, the relaxation cell) that the TTNN gate reads anyway, so skipping them saves nothing. I followed the rule as written.
4. **Code-path scope with CTA-gated holdovers.** All three holdovers compile only under specific CTAs, which reads like a config-scoped gate. But one factory and one kernel file cannot port a config slice without splitting the factory and stripping a kernel copy. The Code-path scope section could say that a subset must be separable at factory or kernel-file granularity.
5. **GlobalCircularBuffer "remote CB" name cue.** The `remote_cb_*` identifier cue fired on `compute.cpp:257-291` and `:383-424`, where `remote_cb` is just a parameter name for a plain CB. The disambiguator guard resolved it. Keying the cue on `remote_cb_config`, `.remote_index(` or `remote_circular_buffer.h` would avoid the false match.
6. **Template row *Feature Support — Variadic-CTA*.** It has no Appendix A entry, and the RTA-varargs subject says CTA varargs are non-gating. The row may be stale; I filled it as "Ok".
7. **"Do not fetch or cross-check in a subagent."** This whole audit ran as one subagent, working from a CSV that the main session fetched. The cross-check has to stay with whoever holds the code, which is this auditor. "Do not delegate the fetch or the cross-check to a helper subagent" would say what the rule means.
