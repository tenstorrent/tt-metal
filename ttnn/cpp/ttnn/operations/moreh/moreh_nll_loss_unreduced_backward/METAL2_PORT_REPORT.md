# Port Report — `moreh_nll_loss_unreduced_backward`

## Outcome

**`PORTED`** — the op's single factory (`MorehNllLossUnreducedBackwardDeviceOperation::Factory`) and all
three of its rank configurations (2d / 3d / 4d) converted together, along with all four kernel sources
it binds. No factory is left behind: this op has exactly one. Build green on the first attempt; the
confirmed test set is **43 passed / 28 skipped**, byte-identical to the pre-port baseline (the 28 skips
are the unconditionally-skipped `bfloat8_b` parametrizations, unchanged).

## Provenance

- **Recipe docs (this port):** `f6033c9ec2d 2026-08-19 docs(metal_2.0): a direct-descriptor op converts to a real program factory`
- **Audit docs (inherited):** `f6033c9ec2d 2026-08-19 docs(metal_2.0): a direct-descriptor op converts to a real program factory`

(Same line: the re-audit and the port ran against the same doc revision.)

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept` — the **base** concept, exactly as the audit chose. Nothing changed, so
nothing needed re-deciding with the invoker. `override_runtime_arguments` was absent pre-port and stays
absent, so the *Translating `override_runtime_arguments`* step did not apply and no cache-hit refresh
logic was written; the framework refreshes the four tensor bindings on a cache hit and nothing else,
which is what the op needs (its only per-dispatch mutable state is the tensor identities).

One thing worth flagging for the next porter of a `descriptor`-concept op, because it is a *silent*
failure rather than a loud one: `create_descriptor` has to actually **disappear**, not merely gain a
`create_program_artifacts` sibling. `ProgramSpecFactoryConcept` is spelled with
`!ProgramDescriptorFactoryConcept<T>`, and that concept is satisfied by the bare *presence* of
`&T::create_descriptor` ([`operation_concepts.hpp:72-74, 119-121`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/api/ttnn/operation_concepts.hpp)).
Leaving both methods in place would leave the op on the descriptor path with a dead-code Metal 2.0
factory beside it — it compiles, and the tests still pass, because the legacy path still works.

### Device-op-class edits

- **Pybind entry points removed:** **none.** `moreh_nll_loss_unreduced_backward_nanobind.cpp` never
  bound `create_descriptor`, so exceptions 1 and 2 did not fire. **This port carries no user-visible
  API change.**
- **Direct-descriptor conversion (exception 3):** **did not fire.** The op already had
  `struct Factory` and `using program_factory_t = std::variant<Factory>`
  (`..._device_operation.hpp:35-42`), so the port was a method swap inside the existing struct and the
  struct was *not* renamed to `MorehNllLossUnreducedBackwardProgramFactory` — renaming an existing
  factory struct is not port work. (Worth stating explicitly because a moreh op whose factory struct is
  named bare `Factory` looks exactly like the shape that exception targets; the brief pre-warned about
  this and the warning landed.)
- **Custom `compute_program_hash`:** **none** — default reflection-based hash, untouched. No backdoor
  `attribute_values` / `to_hash` either. The custom-hash `TensorSpec` failure mode could not arise and
  did not.
- **Everything else in the device-operation class is byte-identical**, verified rather than asserted:
  `git diff` reports `..._device_operation.cpp`, `..._nanobind.{cpp,hpp}` and the op's top-level
  `.{cpp,hpp}` unchanged. The only header edit is the `Factory` method declaration plus its two
  includes (`ttnn/metal_v2_artifacts.hpp` added, `<tt-metalium/program_descriptors.hpp>` dropped —
  it was there only for `ProgramDescriptor`).

### Open items

- **Relaxation candidates: none applied, and none obviously available.** The audit's cell reads `none`
  and the port kept strict matching. The readers are *not* shape-agnostic — each bakes in its rank's
  index arithmetic and reads `C` / `Ct` / `Nt` / `Wt` / `num_inner_tile` as runtime args — so
  `dynamic_tensor_shape` is not the free win it is for `eltwise`. Not investigated further; not a port
  decision.
- **Capability this op would benefit from: none.** Single-program, no op-owned tensors, no
  `GlobalSemaphore`, no `GlobalCircularBuffer`. It sits comfortably inside the base concept.

## Handoff points

**No capitulation, no boundary-rule violation, no kernel-lib gap, no framework gap, no removed pybind
surface.** Nothing in this port reached outside the op's own directory. Recorded here so the absence is
explicit rather than inferred:

- No `sem::name` / `tensor::name` was required at an out-of-op call site. The op has no semaphores at
  all, and both `TensorAccessor`s reach the donors as a **template parameter**
  (`read_tile<AddrGen>` / `read_line<AddrGen>`), which the boundary rule already permits.
- No shared-kernel fork was created or reused — see *Open items for downstream*.
- No `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr`, no `LocalCBInterface` field write, no
  cursor surgery: the CB→DFB swap needed only §A (canonical FIFO, unchanged names), §C
  (`get_read_ptr` / `get_write_ptr` peeks, unchanged names) and one §A metadata getter.

## Successes

- **[Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)
  is what made this op portable at all, and its *hard gate* is what kept me honest.** Five of this op's
  six DFBs have no distinct consumer — `target` / `output_grad` / `weight` are reader-local FIFOs, and
  `weight_scratch` / `output_grad_scratch` have no FIFO ops whatsoever. Without the pattern, five
  "DataflowBuffer with 0 consumers" validator failures would have looked like five invitations to add
  a `pop_front` to the kernel, which the pattern names as the wrong move. The gate ("count the distinct
  kernels; self-loop applies only at exactly one") is also what let me *verify* rather than transcribe
  the brief's dispositions: I ran the census on the kernel bodies
  (`reader_..._2d.cpp:25-41`, `writer_....cpp:20-34`) and it agreed row for row, including the two
  never-popped consumers the entry explicitly tells you not to read as missing endpoints.
- **The "re-derive, don't transcribe" instruction paid for itself in the opposite direction too.** The
  brief was right about the endpoints, so re-deriving changed nothing — but the same posture is what
  made me check the brief's `FP32_DEST_ACC_EN` claim instead of acting on it, which *did* change the
  port (Friction → Gaps below). The value of the instruction is not that briefs are often wrong; it is
  that verifying is cheap and the failure it prevents is silent.
- **[Whitelist rule 7's `constexpr`-vs-`const` test is a genuinely sharp tool, and it decided a real
  line.** `writer_....cpp:26` was `const auto input_grad_tile_bytes = get_tile_size(cb_input_grad);` —
  `const`, not `constexpr`, so it takes the **member getter**: `dfb_input_grad_obj.get_tile_size()`.
  Had it been `constexpr` the answer would have been the opposite (`get_tile_size(dfb::input_grad)`),
  and the [whitelist's §A note](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/cb_dfb_api_whitelist.md#tile--format-metadata-jit-descriptors)
  is explicit that the *declaration* is the entire test — no judgement about whether the `constexpr`
  is "really needed". Having a mechanical rule here meant zero deliberation on a line that is
  otherwise a coin flip, and (because this site is `const`) the port needed the token form nowhere,
  so it incurs none of the Gen1-only token-conversion debt the whitelist asks porters to record.
- **The scope-discipline section correctly stopped an "improvement" I was already reaching for.** Six
  dead `get_dataformat` locals and two dead `C` RTAs (2d, 3d) are sitting in this op, and the
  compute-kernel-config attribute multiplies program-cache entries for no effect. Only the
  invoker-confirmed deletion went in; the rest are written up below and left alone. The
  [`TT_FATAL` census](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#anti-pattern-self-audit)
  is the check that made this auditable rather than self-reported: 17 guards in the device-op class and
  3 `TT_THROW`s in the factory, before and after, no delta.
- **The denominator instruction caught nothing but was worth the two seconds.** Every "expect zero
  hits" sweep in this report carries its file count, and the CB sweep in particular
  (`0 hits / 11 files`) would have been indistinguishable from a mis-scoped path without it.

## Friction

### Gaps

- **The brief and audit told me to drop a define that is actually read, because their grep was scoped
  to the wrong directory.** Both say the `FP32_DEST_ACC_EN` define is *"read by nothing"* / *"no kernel
  reads (zero hits under `device/kernels/`)"*, and the brief instructs: *"don't carry that define into
  the port."* The grep is right and the conclusion is wrong — the three readers
  `#include "ttnn/kernel/dataflow/moreh_common.hpp"`, and that donor branches on the macro at
  [`moreh_common.hpp:22`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp),
  selecting between two definitions of `FP32_DEST_ACC_FTYPE` and two `fp32_dest_acc_cast` overloads.
  The pre-port test log confirms `-DFP32_DEST_ACC_EN=1` reaching the reader compile, and the post-port
  JIT cache holds **four** variants per reader — WEIGHT × FP32_DEST_ACC_EN — so the define still
  partitions the kernel build.

  **The port keeps the define.** This op's kernels happen not to *use* those two symbols, so the
  emitted binary is very likely identical either way — but "very likely" is not the porting invariant,
  and carrying it forward costs one line.

  *Doc-actionable, and the general lesson is bigger than this op:* a "which kernel reads this define?"
  question is not answerable by grepping the op's own kernel directory, because a define reaches every
  transitively-included header — and moreh ops all pull in a large shared donor header. Suggest the
  audit's define-liveness check be specified as *the kernel's include closure*, not `device/kernels/`;
  minimally, `grep -rn <DEFINE> $(the op's kernels' direct includes)`. As written, the check is
  systematically wrong for exactly the op family where a fat donor header is the norm.

- **The recipe's `hw_config` section tells you to compare *resolved values*, but the legacy resolution
  for the default case is two indirections deep and one of them is arch-dependent.** The section's
  table gives the reader/writer default triples as literals, and the natural reading is that
  `ReaderConfigDescriptor{}` resolves to them directly. It doesn't: `ReaderConfigDescriptor` lowers to
  `ReaderDataMovementConfig` ([`program.cpp:418`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/impl/program/program.cpp)),
  whose constructor sets `.noc = detail::preferred_noc_for_dram_read(arch)`
  ([`kernel_types.cpp:13-27`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/impl/kernels/kernel_types.cpp))
  — an **arch-switched function**, not a literal. It happens to return `NOC_0` on every arch today
  (and `preferred_noc_for_dram_write` `NOC_1`), so the table is correct and the TTNN helper is a
  faithful swap. But a porter who trusts the table without following the chain has not actually done
  the comparison the section asks for, and would not notice if a future arch case diverged.

  *Doc-actionable:* the table would be more useful with the two function names beside it and a note
  that the legacy default NOC is arch-resolved — one line, and it turns "trust the table" into
  "confirm the function still agrees with the table."

### Confusion

- **"Self-loop" reads as one thing but covers two, and the entry's own vocabulary is what resolves
  it.** Five DFBs here land on the self-loop, but for two different reasons: `weight_scratch` /
  `output_grad_scratch` are *sync-free* (no FIFO ops at all — the donor NoC-writes them and reads back
  via `get_write_ptr()`), while `target` / `output_grad` / `weight` are *single-ended* (real
  `reserve_back` / `push_back` / `wait_front`, just no distinct consumer). Both resolve to "bind the
  one toucher PRODUCER + CONSUMER", so the distinction is invisible in the resulting code and easy to
  stop thinking about — which is a problem, because the **role-free vs locked** tag it corresponds to
  is precisely what the ≥2-toucher branch turns on. I nearly recorded all five as "sync-free" in the
  plan. The [two-toucher entry](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)'s
  step-1 tagging (locked producer / locked consumer / role-free) is the vocabulary that keeps them
  apart; the self-loop entry describes the two shapes but does not name them with those tags.
  *Suggestion:* tag the two shapes in the self-loop entry's recognition signal with the same
  locked/role-free words, so a porter carries one vocabulary across both entries.

- **The recipe's log-reading procedure prescribes a subagent, which this session was instructed not to
  use.** The [Running builds and tests](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#running-builds-and-tests-without-flooding-your-context)
  step says to hand each log to a Sonnet subagent. This session's operating instructions forbade
  spawning agents, so I met the section's actual *goal* — keep the noise out of context — with
  targeted greps against the backgrounded log (`grep -cE 'FAILED|ERROR'`, the pytest summary line, a
  small extractor script for kernel compile flags). Total context cost was a few lines per check.
  *Suggestion:* state the section's requirement as the invariant ("never load a raw build/test log
  into context") and offer the subagent as the recommended means rather than the mechanism, so a
  porter without agent access doesn't read it as a blocker.

## Open items for downstream

- **Shared kernel touches: none.** All four kernel sources live in this op's own `device/kernels/` and
  each has exactly one binder repo-wide (census: `grep -rl <filename> ttnn/cpp/ttnn/operations/`, hits
  disambiguated — the only non-factory hits were this op's own two `METAL2_*.md` artifacts). No
  `_metal2` fork was reused (rung 1: none exists beside any of them), none was created (rung 2 not
  needed), and nothing was converted in place for other consumers (rung 3 not applicable). The writer
  is shared by all three *rank configurations*, but those are configurations of one factory and
  converted together, so it is not an intra-op shared-kernel case either. **No sunset list, no
  coordination signal, nothing for a next porter to pick up here.**

- **Two dead `C` runtime args, preserved.** `reader_..._2d.cpp:17` and `reader_..._3d.cpp:16` read
  `C` (the host's `channel_size`) into a local that is never used again — verified by
  `grep -nw C` on each reader: one hit, the declaration. The 4d reader does use it
  (`_4d.cpp:65-66`). The port keeps the arg and the read on all three: dropping it would change the
  host's arg set, which is behaviour, not syntax. Cheap to remove on an ops-team pass (delete the
  schema entry, the `AddRuntimeArgsForNode` pair, and the kernel line, in the 2d and 3d configs only).

- **Six runtime args that are really common runtime args.** On every config, only `num_tiles_per_core`
  and `start_id` vary per node. `ignore_index`, `C`, `Ct`, and the rank-specific
  `Nt` / `Wt` / `num_inner_tile` carry the **same value on every node** — they are CRTAs
  (`common_runtime_arg_values`). Left as RTAs deliberately: the recipe is explicit that RTA→CRTA
  changes dispatch semantics and belongs to a later pass, not to the port. This is a real dispatch-cost
  win (six per-node arg writes × up to 64 nodes, per enqueue, for values that never differ) and the
  conversion is mechanical now that the args are named.

- **The compute-kernel-config path is entirely vestigial and it costs program-cache entries.** Carried
  forward from the audit's *Misc anomalies* because the port had to preserve it and a reader of the
  ported factory will wonder why it is there. `operation_attributes_t` carries a
  `DeviceComputeKernelConfig` (`..._device_operation.hpp:22`); each of the three configs destructures
  all five values from `get_compute_kernel_config_args`; **there is no compute kernel** and no
  `ComputeHardwareConfig` is built. Four of the five values (`math_fidelity`, `math_approx_mode`,
  `packer_l1_acc`, `dst_full_sync_en`) are dead. The fifth, `fp32_dest_acc_en`, drives the
  `FP32_DEST_ACC_EN` define — which, per Friction → Gaps, *is* read by the donor header but selects
  symbols this op's kernels never use. With no custom hash, the attribute participates in the default
  program hash, so **calls differing only in `compute_kernel_config` occupy separate cache entries that
  compile to functionally identical programs** — and the op's own test matrix parametrizes over
  `compute_kernel_options`, so it is exercising exactly that duplication. Removing the attribute is a
  public-API change and squarely the ops team's call (the invoker scoped it out of this port
  explicitly). The port's own `-D` accounting is now unambiguous, which should make the decision
  easier: drop the attribute and the define drops with it.

- **Five DM self-loop DFBs and two sync-free DFBs for the Quasar-uplift pass.** `target`,
  `output_grad`, `weight`, `weight_scratch`, `output_grad_scratch` are all bound PRODUCER + CONSUMER by
  the one reader — legal on Gen1, **rejected on Gen2**. Two of them (`weight_scratch`,
  `output_grad_scratch`) additionally have no FIFO ops at all: they exist only because
  `read_line`'s DRAM-alignment staging needs somewhere to land when DRAM's minimum read size exceeds
  L1's, and a CircularBuffer was the only primitive available for "a chunk of L1" when the donor was
  written. Both facts are now declarative on the host and greppable
  (`bind_self_loop` call sites in the factory), so the uplift can find them without re-reading the
  kernels — no tracking needed here. Worth noting that the fix for the scratch pair is not local to
  this op: it lives in the donor's `read_line`
  ([`moreh_common.hpp:739`](https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp)),
  which every moreh op that reads a tilized `(1, W)` row shares.

- **Test coverage note: no C++ gtest and no sweep for this op.** The confirmed baseline is one nightly
  pytest file (`tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_nll_loss_unreduced.py`),
  which is *good* coverage for a port — it parametrizes rank (2d/3d/4d/5d-collapsed), `none_weight`,
  `compute_kernel_options` and dtype, and its two `*_callback` tests assert program-cache entry counts
  across weight-present/absent transitions, which is exactly the cache-hit behaviour the port must not
  disturb. But it is **nightly-only**, so a regression here would not be caught by a PR-gating run.
  Nothing acted on; flagged because the op's port safety rests entirely on a suite that CI does not
  run per-PR.

- **Per-op carry-over: the sibling moreh nll_loss ops are the same shape and should port the same way.**
  `moreh_nll_loss/moreh_nll_loss_step1` / `step2` share this op's idioms — the `push_cb` helper (byte
  identical in `moreh_nll_loss_step2`, and its anonymous-namespace name collides with the one this
  port just deleted), the `moreh_common.hpp` `read_tile` / `read_line` donors, the `Buffer*`-in-RTA
  delivery, and the vestigial compute-kernel-config path. A porter picking one of those up can reuse
  this port's `push_dfb` / `bind_self_loop` shape directly. One caveat that does *not* carry over:
  `step2` has a compute kernel, so it needs the whole compute `hw_config` translation table this port
  got to skip.
