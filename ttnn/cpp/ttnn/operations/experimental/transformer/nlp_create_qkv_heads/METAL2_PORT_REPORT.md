# Port Report — nlp_create_qkv_heads

Metal 2.0 port of `nlp_create_qkv_heads`. The **Sharded** factory is ported to
`ProgramSpecFactoryConcept`; the **Interleaved** factory is a grounded stop, left on legacy
`create_descriptor` (mixed-concept variant). **Tests not yet run — pending orchestrator build/test.**

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept` for `NlpCreateHeadsDeviceOperation::Sharded` (single-program, no
op-owned device resources). `NlpCreateHeadsDeviceOperation::Interleaved` stays on
`ProgramDescriptorFactoryConcept`. `select_program_factory`, `validate_*`, `compute_output_specs`,
`create_output_tensors` unchanged. The framework dispatches each factory per `input.is_sharded()`.

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** (op used the default reflection hash).
- Pybind entry points removed: **none** — the nanobind file exposes only the user-facing op, no
  `create_descriptor`/factory-innards binding.
- Header: `device/nlp_create_qkv_heads_device_operation.hpp` now also includes
  `ttnn/device_operation.hpp` + `ttnn/metal2_artifacts.hpp` and the `Sharded` struct's method became
  `create_program_spec` returning `ttnn::device_operation::ProgramArtifacts`. `Interleaved` keeps
  `create_descriptor`/`ProgramDescriptor` (so `<tt-metalium/program_descriptors.hpp>` stays included).

### Open items
- Strict tensor matching kept (default). No relaxation candidates noted.

## Handoff points

- **Interleaved factory blocked on the cross-op compute kernel `transpose_wh.cpp` (API: requires
  Metal-2.0 prep by the kernel owner).** File:
  `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp` — outside the op directory, shared by 4 ops
  (`nlp_create_qkv_heads`, `nlp_create_qkv_heads_boltz`, `nlp_create_qkv_heads_vit`,
  `split_query_key_value_and_split_heads`). It is bound by the Interleaved factory's
  `transpose_k_heads` path. Two incompatibilities prevent porting that factory without editing the
  kernel (which the porter must not do): (1) it reads a **positional** CTA
  `get_compile_time_arg_val(0)` (line 10) — a Metal-2.0 host emits only named args, so this
  `static_assert`s at JIT; it needs `get_arg(args::NHtWt)`. (2) it **hardcodes** physical CB indices
  `tt::CBIndex::c_0` and `tt::CBIndex::c_16` (lines 11, 18, 19, 25, 28, 29) rather than taking
  `uint32_t` CB ids as parameters, so a ported factory cannot thread `dfb::name` into it via the
  implicit `operator uint32_t`. To unblock: the kernel's owner should Metal-2.0-prep it — convert the
  positional CTA to a named arg and parameterize the two CB ids (so `dfb::name` can flow in). Until
  then the Interleaved factory stays on legacy `create_descriptor`. (The Interleaved reader/writer
  also encode the K route through fixed physical indices CB0→compute→CB16, so they are coupled to the
  compute kernel and cannot be ported independently.)

- **Case-2 `input_q` / `input_kv` bindings (TensorAccessor enhancement candidate).** The input shards
  are read by an exotic per-core NoC walk (`{.noc_x,.noc_y,.addr}` unicast, `head_size`-granularity,
  per-head striding) that `TensorAccessor` page iteration cannot express. The port re-expresses the
  bindings as `TensorParameter`s and recovers each base kernel-side via
  `TensorAccessor::get_bank_base_address()` (sanctioned Case-2 bridge), leaving the NoC arithmetic
  intact. A future per-core sub-shard gather mode for `TensorAccessor` would let this become Case 1.

## Successes

- **Borrowed-memory + fake-CB outputs (patterns catalog: Fake CB → self-loop, Borrowed-memory DFB).**
  Legacy CBs `c_16`/`c_17`/`c_18` (`.buffer = output.buffer()`, write-only via `get_write_ptr()`, no
  FIFO) ported cleanly as three `DataflowBufferSpec{ borrowed_from = … }`. q_out is touched by both
  reader and writer on the same `q_cores`, so it is bound reader=PRODUCER / writer=CONSUMER (the
  reference `nlp_concat_heads_decode` shape for a shared write-only borrowed CB). The reader-private
  k_out and writer-private v_out are same-kernel self-loops (PRODUCER+CONSUMER). The reference port
  next door made this decision easy.
- **Common runtime varargs.** The NoC coordinate arrays (`in0_mcast_noc_x/y`, read in legacy via
  `get_arg_addr(19)` + `get_arg_addr(19+num_x)` pointer arithmetic) map cleanly to
  `num_common_runtime_varargs` + `get_common_vararg(i)`; coords are identical across all output cores.
  Layout `[x.., y..]`; kernel reads `get_common_vararg(idx)` / `get_common_vararg(num_x + idx)`.
- **Same source bound twice with different second DFB.** Reader and writer share one kernel source;
  legacy distinguished them by the second CB-index CTA (17 vs 18). Modeled 1:1 as two KernelSpecs
  binding the same accessor name `kv_out` to different DFBs (K_OUT vs V_OUT) — exactly the legacy
  shape, no demotion.

## Friction

- **Confusion — Case-2 offsets are per-dispatch RTAs, not CTAs.** The recipe's *Host-computed
  base-pointer offset → CTA offset* pattern is CTA-only and says to capitulate if the offset is
  per-dispatch-varying (RTA). Here the q/k/v start offsets are genuinely per-core/per-iteration RTAs.
  The reference port (`nlp_concat_heads_decode`) resolved the identical situation by passing the
  per-head offset as an RTA and adding it to the accessor base kernel-side — and it is tested. I
  followed the reference rather than capitulating, since the kernel behavior is byte-identical to
  legacy (same absolute addresses), only the host/kernel boundary moves. Worth a catalog sub-note:
  the offset-RTA + accessor-base form is the established shape for these sharded transformer ops.
- **Gap — multiple Case-2 source bases on one kernel.** The sharded kernel derives K/V bases either
  from a *second* input tensor (`input_kv`) or as a fixed offset into the Q input shard. This needed a
  conditional second tensor binding (`READ_FROM_INPUT_TENSOR_KV`) feeding the kernel-side base, plus a
  uniform `kv_base_offset`/`kv_start_offset` RTA pair the kernel adds to whichever base it recovered.
  The catalog's Case-2 / conditional-binding entries each cover half of this; their composition (a
  *conditionally-sourced* Case-2 base) isn't spelled out.

## Open items for downstream

- **Fake-CB self-loop bindings (interim hack — flag prominently).** All three output DFBs in the
  Sharded factory are **tensor-local-view** fake CBs (write-only address source; no real FIFO):
  - `q_out` (`borrowed_from = Q_OUT_TENSOR`) — bound reader=PRODUCER / writer=CONSUMER.
  - `k_out` (`borrowed_from = K_OUT_TENSOR`) — self-loop on the reader (PRODUCER+CONSUMER).
  - `v_out` (`borrowed_from = V_OUT_TENSOR`) — self-loop on the writer (PRODUCER+CONSUMER).
  Site: `device/nlp_create_qkv_heads_program_factory.cpp` (`*_dfb_spec` + reader/writer `DFBBinding`s).
  These are validator-satisfying devices, not real FIFOs; the eventual "local `TensorAccessor`"
  migration should replace them.
- **Cross-op kernel touches:** none performed. `transpose_wh.cpp` was **not** modified or forked — it
  is the grounded-stop blocker for the Interleaved factory (see Handoff points). Remaining unmigrated
  consumers if/when it is Metal-2.0-prepped: this op's Interleaved path, `nlp_create_qkv_heads_boltz`,
  `nlp_create_qkv_heads_vit`, `split_query_key_value_and_split_heads`.
- **Interleaved factory port (remaining work).** Blocked only on `transpose_wh.cpp`. Once that kernel
  is prepped, the Interleaved reader/writer (`reader_/writer_tm_tile_layout_nlp_create_qkv_heads.cpp`)
  are in-dir and straightforward: their `TensorAccessorArgs`+Buffer*-RTA inputs are Case-1 (clean
  `TensorAccessor(ta::name)`), CB-id CTAs become DFB bindings, and the conditional `cb_id_k`
  (0/1/16) + `TRANSPOSE_K_HEADS` define map to conditional DFB bindings.
- **Test coverage:** `tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads.py`
  exercises both interleaved (default `transpose_k_heads`) and sharded configurations. After the
  orchestrator build, the **sharded** sub-cases validate this port; the interleaved sub-cases continue
  to run on the unchanged legacy path. Not run by the porter.
