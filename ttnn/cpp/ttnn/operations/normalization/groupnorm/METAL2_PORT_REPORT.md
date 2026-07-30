# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/normalization/groupnorm`

## Outcome

**`CAPITULATED`** — all three factories (`GroupNormShardedProgramFactory`,
`GroupNormMcastProgramFactory`, `GroupNormNoMcastProgramFactory`).

Every writer kernel in the op calls a shared-pool helper that takes a `CircularBuffer` **by value**, a
parameter type a Metal 2.0 kernel has no sanctioned way to supply. A factory converts atomically with
the kernels it binds, and each of the three factories binds one of the four affected writers, so no
factory is portable until the helper grows a `DataflowBuffer` form. See
[Handoff points](#handoff-points) entry 1.

**No source file in the op was modified.** The op still builds and runs exactly as before. The
deliverable is this report plus `METAL2_PORT_PLAN.md`, which carries the full legacy inventory and the
sharded factory's planned spec shape so a resumed port starts from analysis rather than from scratch.

The stop was surfaced to the invoker before any code was written, with the alternative (bridging via an
explicit `CircularBuffer(dfb::eps)` wrapper at the four call sites and flagging it) laid out alongside
it. The invoker chose to treat it as the assumption violation the audit had pre-flagged.

## Provenance

- **Recipe docs (this port):** `4d6e9518bf5 2026-07-30 docs(metal_2.0): require an explicit opt_level when porting compute kernels`
- **Audit docs (inherited):** `4d6e9518bf5 2026-07-30 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## TTNN ProgramFactory

### Concept realized

None — no factory was converted. The audit's choice, `ProgramSpecFactoryConcept` for all three
factories, was confirmed as correct during planning and is carried in `METAL2_PORT_PLAN.md`. Nothing in
the inventory contradicted it: the op is single-program SPMD, has no op-owned tensors, no custom hash,
no `get_dynamic_runtime_args`, no `override_runtime_arguments`, and no pybound factory entry point.

### Device-op-class edits

- Custom `compute_program_hash` deleted: none — the op already uses the default reflection-based hash.
- Pybind entry points removed: none — `groupnorm_nanobind.cpp` binds only `ttnn::group_norm` and the two
  program-config structs.

### Open items

- **A borrowed-memory DFB's backing `TensorParameter` may have no `TensorBinding`.** In the sharded
  factory the input and output tensors reach the kernels *only* as borrowed-memory DFBs (`c_0` / `c_16`,
  `.buffer = a.buffer()` / `output.buffer()`); no kernel constructs a `TensorAccessor` over either. The
  port must still declare `TensorParameter`s for them, because `DataflowBufferSpec::borrowed_from` names
  a `TensorParameter`. But the spec validator rejects a `TensorParameter` with zero `TensorBinding`s
  across the program's kernels
  ([migration_guide.md — TensorParameter](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/migration_guide.md#tensorparameter)).
  Whether `borrowed_from` counts as a binding for that rule is not stated in the headers, the migration
  guide, or the recipe. It is the first thing a resumed port will hit. Worth settling in the docs
  either way, since borrowed-memory DFBs on sharded ops are common and this configuration — borrowed
  with no accessor — is the normal one for them, not an edge case.
- **Relaxation candidates:** none. The op has no custom hash to mine and no kernel reads
  `ArgConfig::Runtime*`, so strict tensor-arg matching is the right default throughout.

## Handoff points

### 1. `generate_bcast_*` takes a `CircularBuffer` by value — no Metal 2.0 form exists

**Owner:** the team that owns `ttnn/cpp/ttnn/kernel/dataflow/`.
**Tag:** API: requires a `DataflowBuffer` overload.
**This is the capitulation.**

**Callee:** `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp`

```cpp
FORCE_INLINE void generate_bcast_col_scalar  (CircularBuffer cb, uint32_t scalar);  // :13
FORCE_INLINE void generate_bcast_row_scalar  (CircularBuffer cb, uint32_t scalar);  // :29
FORCE_INLINE void generate_bcast_unary_scalar(CircularBuffer cb, uint32_t scalar);  // :44
```

**Call sites in this op** — all four writer kernels, all of the form
`generate_bcast_col_scalar(CircularBuffer(eps_dfb_id), eps)`:

| kernel | line | factory that binds it |
|---|---|---|
| `device/kernels/dataflow/writer_unary_sharded_gn_rm_gb_v2.cpp` | 148 | sharded |
| `device/kernels/dataflow/welford_writer_unary_sharded_gn_rm_gb_v2.cpp` | 68 | sharded |
| `device/kernels/dataflow/writer_unary_gn_rm_gb.cpp` | 156 | mcast, no-mcast |
| `device/kernels/dataflow/welford_writer_unary_gn_rm_gb.cpp` | 106 | mcast, no-mcast |

**Why mechanical conversion fails.** The `dfb::name → uint32_t` bridge that carries named handles into
LLK and kernel-lib call sites does not reach a parameter of type `CircularBuffer`.
`CircularBuffer`'s only id constructor is `explicit CircularBuffer(uint32_t cb_id)`
(`tt_metal/hw/inc/api/dataflow/circular_buffer.h:27`), and C++ will not chain two user-defined
conversions, so `generate_bcast_col_scalar(dfb::eps, eps)` does not compile. The call site must
materialise the wrapper itself — `generate_bcast_col_scalar(CircularBuffer(dfb::eps), eps)` — which
compiles, but leaves a live `CircularBuffer` construction plus its `api/dataflow/circular_buffer.h`
include inside a ported kernel. The kernel-side whitelist's rule-1 sweep states the CB→DFB transition is
total and that a grep for `CircularBuffer` across the op directory must return zero hits in code, and
the verification checklist repeats it. There is no `DataflowBuffer` overload of any of the three
helpers and no `DataflowBuffer → CircularBuffer` conversion.

The porter cannot fix this from inside the op directory: the header is outside it, and it is also
outside the `_metal2`-fork convention, which covers kernels owned by ops under
`ttnn/cpp/ttnn/operations/` — this is a shared kernel-code pool.

**Sketch of the change that would unblock it.** Add a `DataflowBuffer` overload beside each existing one
(the bodies use only `reserve_back` / `get_write_ptr` / `push_back`, all of which `DataflowBuffer`
provides under the same names, so each overload is the existing body with the parameter type swapped):

```cpp
FORCE_INLINE void generate_bcast_col_scalar(DataflowBuffer dfb, uint32_t scalar);
```

Keeping the `CircularBuffer` overloads alongside means no existing caller changes. Once the overload
exists, the four groupnorm call sites become `generate_bcast_col_scalar(DataflowBuffer(dfb::eps), eps)`
and the port is unblocked with no other change to the plan.

**Blast radius beyond groupnorm.** The same three helpers are called from **25 other kernel files across
11 other ops** — `normalization/layernorm`, `normalization/layernorm_distributed`,
`normalization/softmax`, `transformer/sdpa`, `transformer/sdpa_decode`, `reduction/sampling`,
`data_movement/bcast`, `experimental/ccl/rms_allgather`,
`experimental/ccl/dit_fused_distributed_rmsnorm`,
`experimental/transformer/fused_distributed_rmsnorm`, and
`experimental/transformer/dit_layernorm_post_all_gather`. Every one of them will hit this same stop when
it reaches the porting queue, so the fix is worth doing ahead of the wave rather than per-op. layernorm
in particular is a near neighbour of groupnorm and will hit it immediately.

### 2. `preferred_noc_for_dram_read` / `_write` are `detail::` functions used as production hardware config

**Owner:** the runtime / kernel-types owners, and the groupnorm op owners.
**Tag:** API: `detail::` reached from op code.

All three groupnorm factories pick their NOCs with
`tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch())` and its write sibling
([groupnorm_sharded_program_factory.cpp:524-525](device/groupnorm_sharded_program_factory.cpp#L524-L525),
and the corresponding lines in the other two factories). Those functions carry the comment *"These are
only used in op_profiler, are unstable and have not been designed for general use"*
([kernel_types.hpp:131-146](../../../../../../tt_metal/api/tt-metalium/kernel_types.hpp#L131-L146)) —
which is no longer true, and the disclaimer sits directly above two functions whose return values now
determine a production op's hardware config.

The port would carry the *resolved* values (`NOC_0` for read, `NOC_1` for write on every arch today), so
this is not a behaviour question. It is a question of whether a Metal 2.0 `KernelSpec::hw_config` should
be reaching into `detail::` for its NOC — and, if the intent is that ops keep doing this, whether the
comment and the namespace should change.

## Successes

- **The recipe's warning that a legacy DM config may match *neither* role default fired exactly as
  written, and caught a real one.** groupnorm's resolved triples are reader
  `(RISCV_0, NOC_0, DM_DEDICATED_NOC)` and writer `(RISCV_1, NOC_1, DM_DEDICATED_NOC)` — the RISC
  assignment is *swapped* relative to the Metal 2.0 helpers, whose reader default is `(RISCV_1, NOC_0)`
  and writer default `(RISCV_0, NOC_1)`. Reading only the descriptor's role name
  (`reader_mcast_sender_desc`, `writer_desc`) and reaching for
  `create_reader_datamovement_config` / `create_writer_datamovement_config` would have moved both
  kernels to the other RISC — a silent perf change with no test net, in an op where the reader is the
  large kernel and Wormhole's RISCV_1 has the 16 kB instruction-memory limit. The
  [Hardware configuration](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#hardware-configuration)
  section's "Match on the *values*, not the role name" instruction is the reason this was checked at
  all. Recorded in the plan at
  [groupnorm_sharded_program_factory.cpp:539-542](device/groupnorm_sharded_program_factory.cpp#L539-L542)
  and `:624-627`.

- **`grep -n opt_level` instead of reading `config` was the right instruction.** No `KernelDescriptor`
  anywhere in this op sets `opt_level`, so reading the configs would have concluded "nothing to carry."
  The rule that an absent field still resolves to `O3` on a `ComputeConfigDescriptor` is what makes the
  two compute `KernelSpec`s per factory need an explicit `O3` they would otherwise silently lose.
  ([Compiler options](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compiler-options).)

- **The brief's "watch the helper-mediated FIFO ops" note prevented a wrong census.** `c_8` `ex_partial`
  in the sharded compute kernel reads as consumer-only if you grep for `reserve_back` / `push_back`; its
  producer is the `compute_kernel_lib::reduce<…, dfb_ex_partial_id>` at
  [groupnorm_sharded_v2.cpp:283](device/kernels/compute/groupnorm_sharded_v2.cpp#L283), which does the
  FIFO work internally. Counting it as consumer-only would have made `c_8` look like a one-toucher
  needing a self-loop when it is a genuine 1P+1C while mcasting. The same trap sits on `c_1` (produced by
  `tilize<…, dfb_in_id>`) and `c_2` (produced by `calculate_and_prepare_reduce_scaler<c_2>` in the
  writer).

- **Re-deriving the endpoint census rather than transcribing the brief was cheap and confirmed it.** The
  independently-run census agreed with the brief on every DFB, including the four config-scoped dead-CB
  drops and the "no multi-binding anywhere" verdict. That agreement is itself worth recording: the
  brief's dispositions for this op are trustworthy, so a resumed port can lean on the plan's table.

## Friction

### Gaps

- **The recipe names a `TensorParameter` field that does not exist.**
  [Plan the spec](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#plan-the-spec)
  and the
  [ttnn_factory](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/ttnn_factory.md)
  doc both refer to `TensorParameter::advanced_options` (holding `dynamic_tensor_shape` /
  `match_padded_shape_only`). The header has no such field: `TensorParameter` carries
  `TensorSpecRelaxations relaxations`
  (`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp:44`). Harmless here since
  this op needs no relaxation, but a port that does need one follows the doc to a compile error. The
  "go to the headers first" instruction is what resolved it, so the recipe's own advice held — but the
  paraphrase should be corrected.

- **Nothing in the docs says whether `borrowed_from` satisfies the "every `TensorParameter` needs ≥1
  `TensorBinding`" rule.** Detailed under [Open items](#open-items) above. This is the single largest
  unknown left in the sharded factory's plan, and it is structural rather than incidental: for a sharded
  op, "input reaches the kernel only as a borrowed DFB" is the *typical* shape. Either the migration
  guide's `TensorParameter` section or the `dataflow_buffer_spec.hpp` `borrowed_from` comment should
  state the answer.

- **The recipe has no guidance for a donor whose parameter is a `CircularBuffer` object.** The
  [Crossing the boundary](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#read-this-first)
  section covers two cases: `dfb::name` crosses freely into `uint32_t` parameters, and `sem::` /
  `tensor::` do not cross at all (an assumption violation). A third shape exists and this op is full of
  it — a Device 2.0-era donor taking a `CircularBuffer` *object*. It sits between the two: mechanically
  bridgeable with an explicit wrapper, but only at the cost of the rule-1 "no `CircularBuffer` survives"
  sweep. The recipe should say which way that goes, because the answer decides whether roughly a dozen
  ops capitulate or ship with a documented exception. If the intended answer is "capitulate," saying so
  explicitly would also let the *audit* gate on it — right now the audit flags it as a ⭐ heads-up
  ("if you cannot bridge it") and still returns GREEN, which sends a porter into a full inventory before
  the stop surfaces.

### Confusion

- **"If you cannot bridge it" is ambiguous when the bridge exists but is dirty.** The audit's ⭐ entry
  reads *"The call site currently materialises a `CircularBuffer` from a raw id; in the port there is no
  id to materialise it from."* That premise is not quite right — `dfb::eps` **is** an id by way of
  `DFBBindingToken::operator uint32_t()`, so `CircularBuffer(dfb::eps)` compiles fine. The real obstacle
  is not availability of an id, it is the rule-1 sweep. Restating the ⭐ entry in those terms would have
  made the stop obvious in minutes instead of requiring a trip through `circular_buffer.h`,
  `dataflow_buffer.h`, and the whitelist to establish what "cannot bridge" meant.

- **A capitulation discovered at the *whitelist* stage still costs a full inventory.** The workflow puts
  the kernel-side whitelist inside `Scope discipline`, which sits between planning and construction — so
  a porter reaches it only after the legacy inventory and the spec plan are done. For a blocker that is
  visible from a single grep of the kernel sources (`grep -n CircularBuffer device/kernels/`), that is
  late. A short "before you plan, grep the kernels for these shapes" checklist at the top of the port
  recipe — donor signatures taking `CircularBuffer`, `get_cb_tiles_acked_ptr`, `GlobalCircularBuffer` —
  would surface this class of stop in the first ten minutes. (In this port the audit *had* already
  flagged it, which is the system working; the checklist would help where the audit's ⭐ list is thinner.)

## Open items for downstream

- **Shared kernel touches: none performed, one large one pending.** No kernel was reused, forked, or
  modified. But a resumed port must know: `GroupNormMcastProgramFactory` and
  `GroupNormNoMcastProgramFactory` are an **intra-op shared-kernel pair** — six sources are bound by
  both: `reader_mcast_sender_unary_gn.cpp`, `writer_unary_gn_rm_gb.cpp`, `compute/groupnorm.cpp`, and
  their three `welford_` siblings. (`reader_mcast_receiver_unary_gn.cpp` and
  `welford_reader_mcast_receiver_unary_gn.cpp` are mcast-only.) No `_metal2` fork exists beside any of
  them. Converting one factory in place breaks the other, so they must be
  co-ported as a single unit, or six forks created. The **sharded** factory has no such coupling — its
  eight `_v2` sources are bound by nothing else — which is why it is the natural first factory and why
  `METAL2_PORT_PLAN.md` plans it in full.

- **Ops-team question, unanswered and still blocking one configuration.** Audit *Questions for the user*
  #1: in the mcast and no-mcast factories, is the output CB `c_16` genuinely unused when
  `!UNTILIZE_OUT && !fuse_gamma && !fuse_beta`? Both compute and the writer resolve `dfb_out_id` to
  `c_22` in that branch ([groupnorm.cpp:171](device/kernels/compute/groupnorm.cpp#L171),
  [writer_unary_gn_rm_gb.cpp:81](device/kernels/dataflow/writer_unary_gn_rm_gb.cpp#L81)) and nothing
  names `c_16`. A dead *output* CB is unusual enough to want confirmation before the port drops the
  allocation, and if it is genuinely dead it is also wasted SRAM today. The sharded factory does not
  depend on the answer.

- **Audit anomalies, none of them port work, all still open.** Recorded here so they are not lost with
  the audit file:
  1. `bool block_wt_last = (per_core_Nt + num_groups_per_core - 1) / num_groups_per_core;` — a tile count
     assigned to a `bool`, collapsing every non-zero value to `1`, then handed to the kernels as the
     `block_w_last` compile-time arg and used in tile arithmetic
     ([groupnorm_sharded_program_factory.cpp:226](device/groupnorm_sharded_program_factory.cpp#L226),
     [groupnorm_mcast_program_factory.cpp:195](device/groupnorm_mcast_program_factory.cpp#L195),
     [groupnorm_no_mcast_program_factory.cpp:206](device/groupnorm_no_mcast_program_factory.cpp#L206)).
     Either the kernels rely on the collapsed value and the type and name are both misleading, or a real
     tile count is being lost. A port carries the value through unchanged either way, so this needs an
     owner, not a porter.
  2. `packer_l1_acc` is destructured from the compute-kernel config in all three factories and never
     used (e.g. [groupnorm_sharded_program_factory.cpp:713](device/groupnorm_sharded_program_factory.cpp#L713)).
     It still participates in the operation-attributes hash through `compute_kernel_config`, so setting
     it changes the cache key while having no effect — hashed but ignored.
  3. The sharded writer's compile-time arg 10 (a `page_size`-ish value) is computed on the host
     ([:583-591](device/groupnorm_sharded_program_factory.cpp#L583-L591)) and read by neither sharded
     writer kernel. The port would drop it; today it is dead host work.
  4. `my_x[0]` / `my_y[0]` hardcodes NoC index 0 for self-addressed local transfers at 14 sites, while
     the kernels run on `preferred_noc_for_dram_read` / `_write`, which is not necessarily NOC 0. One
     sibling kernel does it the other way —
     `my_x[noc.get_noc_id()]` at
     [welford_reader_mcast_sender_unary_sharded_gn_v2.cpp:142](device/kernels/dataflow/welford_reader_mcast_sender_unary_sharded_gn_v2.cpp#L142)
     and `:302`. My guess is the `my_x[0]` form only works because the coordinate spaces coincide for the
     self case on current silicon, but the inconsistency inside one op is worth resolving. A port copies
     these lines verbatim.
  5. Four `get_dataformat(...)` locals across the reader kernels are computed and never read
     ([reader_mcast_sender_unary_sharded_gn_v2.cpp:132](device/kernels/dataflow/reader_mcast_sender_unary_sharded_gn_v2.cpp#L132),
     [reader_mcast_receiver_unary_sharded_gn_v2.cpp:49](device/kernels/dataflow/reader_mcast_receiver_unary_sharded_gn_v2.cpp#L49),
     [reader_mcast_sender_unary_gn.cpp:222](device/kernels/dataflow/reader_mcast_sender_unary_gn.cpp#L222),
     [reader_mcast_receiver_unary_gn.cpp:140](device/kernels/dataflow/reader_mcast_receiver_unary_gn.cpp#L140)).

- **Per-op carry-over.** `normalization/layernorm` is the closest neighbour and will hit handoff point 1
  at `writer_unary_sharded_ln.cpp:69`, `writer_unary_sharded_ln_rm_gb.cpp:72`,
  `reader_unary_interleaved_ln_rm_gb.cpp:110`, `reader_unary_interleaved_ln.cpp:149`,
  `reader_unary_interleaved_ln_large_tensor.cpp:140`, and
  `reader_unary_interleaved_ln_large_tensor_welford.cpp:73`. `normalization/softmax` hits it through
  `generate_bcast_unary_scalar` at five reader kernels. Sequencing the `generate_bcast_*` overload ahead
  of the normalization family's ports would avoid five or six repeat capitulations.

- **Test coverage notes.** The no-regression baseline confirmed with the invoker was
  `tests/ttnn/unit_tests/operations/fused/test_group_norm.py` (18 tests) and `test_group_norm_DRAM.py`
  (4), `tests/ttnn/nightly/unit_tests/operations/fused/test_group_norm.py` (15) and
  `test_group_norm_DRAM.py` (7), and
  `tests/ttnn/nightly/unit_tests/operations_compute_only/fused/test_group_norm.py` (6). None was run:
  with no code change there is nothing to regress, and a green run would say nothing about the port. A
  resumed port should capture a pre-change baseline over that set before touching a file — the sharded
  factory alone spans welford × tilize-in × untilize-out × repack × four optional tensors × mcast /
  single-core-group, and the plan's dead-CB drops and conditional bindings are per-configuration, so
  partial coverage of that matrix is the likeliest source of a silent miss.
