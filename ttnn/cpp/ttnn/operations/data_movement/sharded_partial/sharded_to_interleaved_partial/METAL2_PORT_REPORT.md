# Metal 2.0 Port Report — `data_movement/sharded_partial/sharded_to_interleaved_partial`

## Outcome

**`CAPITULATED`** — grounded scope-limit stop at the planning step. The single `descriptor` factory
(`ShardedToInterleavedPartialProgramFactory`) owns **zero** in-directory kernels; all four selectable
kernel entry points it binds live **outside** the op directory, and the orchestration constraint for
this port forbids editing or forking any kernel outside the op directory. The recipe's atomic-unit rule
(factory + every bound kernel entry point flip together; no half-Metal-2.0 factory) cannot be satisfied
within the permitted writeable surface. **No code was modified.** This is a success-tier outcome per the
recipe's [When the discipline doesn't fit](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md) off-ramp.

## Provenance

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

- **Concept targeted:** `MetalV2FactoryConcept` (inherited from audit).
- **Concept realized:** none — port stopped before construction.
- **Custom `compute_program_hash` deletion:** N/A — op has no custom hash (audit cross-checked).
- **Pybind entry points removed:** none — nanobind binds only the free function
  `&ttnn::sharded_to_interleaved_partial`; there is no `create_descriptor` pybind to remove.
- **Device-op-class edits forced:** none (none reached, since the port did not proceed).

## Handoff points

### 1. Port capitulation — factory binds only out-of-directory kernels (scope limit)

- **Op / factory:** `ttnn/cpp/ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial`,
  `ShardedToInterleavedPartialProgramFactory::create_descriptor`
  (`device/sharded_to_interleaved_partial_program_factory.cpp:47`).
- **The constructs that cannot convert within scope:** all four kernel sources the factory selects are
  outside the op directory and none is on Metal 2.0 (each reads positional CTAs/RTAs; the writers also use
  `TensorAccessorArgs<N>()` + a buffer-address RTA):
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp`
    (positional CTA `get_compile_time_arg_val(0)` at `:13`, positional RTA at `:12`; ~18 co-borrowers).
  - `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp`
    (positional CTA `:22`, `TensorAccessorArgs<1>()` `:23`, buffer-address RTA `dst_addr` `:11`, positional
    RTAs `:11-19`, `get_tile_size(cb_id_out)` `:26`; ~3 co-borrowers). *Live path.*
  - `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp`
    (RM path; validate-blocked/dead at runtime but still a selectable source that flips with the factory).
  - `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` (positional CTA; `cb_*` FIFO-sync free functions;
    only when `convert_df`; ~4 co-borrowers).
- **Why mechanical conversion failed:** the recipe's atomic-unit rule requires the factory and every kernel
  entry point it binds to flip to Metal 2.0 together — a Metal 2.0 factory emits only named args via the
  framework-generated headers, and a legacy kernel reading positional args `static_assert`s at JIT. Since the
  op owns no in-directory kernel, every conversion target is out-of-directory. The recipe would normally
  allow porting cross-op kernels together (adopt the rewrite in all co-borrowers) or forking them, but the
  **orchestration constraint for this port overrides the recipe** and forbids editing or forking any kernel
  outside the op directory. Both routes are therefore closed, and no in-scope route remains.
- **Sketch of the needed change (for maintainer evaluation):** port the four shared kernels to Metal 2.0
  (CB→DFB is already done for Device 2.0; remaining work is named-token bindings — `dfb::`/`tensor::`/`args::`
  — replacing positional CTAs/RTAs, `TensorAccessorArgs`, and the buffer-address RTA), landing each rewrite
  across all co-borrowers of that source (or forking a `_metal2` copy per the shared-dataflow-kernel Caution).
  Because `reader_unary_sharded.cpp` alone has ~18 co-borrowers, this is a coordinated multi-op port-together
  unit — the audit's own "Team-only" note calls it out and says `reader_unary_sharded.cpp` "should be
  sequenced as its own port-together unit." Sequencing those shared-kernel units, then porting this factory
  once its kernels are Metal-2.0-ready, is the path forward. This op cannot be the vehicle that first ports
  those shared kernels under a constraint that forbids touching them.

**Note on the GREEN audit.** The capitulation is *not* a contradiction of the GREEN audit. The audit's
GREEN is about Metal 2.0 *capability*: the op's constructs (borrowed-memory input DFB, one Case-1 accessor
binding, all CB endpoints legal 1:1, no unsupported feature) are all expressible in Metal 2.0. The audit
correctly flagged the out-of-directory kernel coupling in "Watch for" / "Team-only" but did not gate on it,
because under the *unconstrained* recipe cross-op kernels are porter-touchable (port-together or fork). The
blocker here is introduced by the **orchestration constraint** layered on top (no out-of-dir edits, no
forks), not by a Metal 2.0 capability gap. Given that constraint, the op should not have been dispatched for
an isolated port ahead of its shared-kernel port-together units.

## Successes

- **Recipe atomic-unit note + "reaching past the op's own directory" stop signal
  ([Read this first](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md) /
  [When the discipline doesn't fit]).** These sections named this exact situation precisely and made the
  stop unambiguous rather than a judgment call: "there is no half-Metal-2.0 factory" + "if you find yourself
  reaching past the op's own directory to make kernel changes, that's the signal." The warning fired
  correctly and prevented a broken half-port.
- **Audit brief "Cross-op / shared kernels" heads-up.** The brief listing every kernel as a file-path borrow
  with co-borrower counts made the ownership situation immediately legible, so the blocker was identifiable
  during inventory without spelunking.

## Friction

### Gaps
- **Recipe vs. orchestration-constraint interaction is implicit.** The recipe treats cross-op kernels as
  porter-touchable (port-together / fork); a per-port orchestration constraint that forbids all out-of-dir
  edits/forks turns an op that owns *no* kernels into an unportable target. The recipe does not explicitly
  discuss what happens when an op owns zero in-directory kernels under a no-out-of-dir-edits regime — it is
  derivable from the atomic-unit rule, but a one-line note ("an op owning no in-directory kernels cannot be
  ported in isolation; it must follow its shared-kernel port-together units") would make the outcome
  immediate. Right answer taken here: capitulate on scope, hand off the shared-kernel sequencing.

### Confusion
- None material — once the four kernel paths were confirmed all out-of-directory, the atomic-unit rule made
  the conclusion mechanical.

## Open items for downstream

- **Cross-op kernel port-together units (coordination signal for the next port).** None modified or forked
  by this port (capitulated). The remaining work, for whoever sequences it:
  - `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` — widest coupling (~18 co-borrowers);
    audit recommends its own port-together unit.
  - `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp`
    — in-family, ~3 co-borrowers (incl. `sharded/interleaved_to_sharded`, this op).
  - `data_movement/sharded/device/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp`
    — in-family, RM path (dead here).
  - `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` — shared compute pool, ~4 co-borrowers.
  Once these are Metal-2.0-ready, this factory's port is straightforward (plan already recorded in
  `METAL2_PORT_PLAN.md`).
- **Pre-existing findings (NOT touched — device-op / factory-body code outside the Metal 2.0 transformation;
  routed here per scope discipline):**
  - **Dead RM writer path.** `validate_on_program_cache_miss` hard-asserts `input.layout() == Layout::TILE`
    (`device/sharded_to_interleaved_partial_device_operation.cpp:24`), yet the factory still selects the RM
    writer kernel and RM RTA branch on non-TILE input (`_program_factory.cpp:182-186`, `:259-308`). The RM
    writer and its RTA branch are unreachable. Additionally, the RM branch pushes `num_units_per_row` at
    `writer_rt[1]` (`:295`) which the RM kernel never reads (a dead RTA in a dead path). The ops team may wish
    to remove the dead RM branch (separate PR).
  - **`is_l1_aligned` hardcoded `true`** (`_program_factory.cpp:55`) makes the surrounding conditional
    (`:287-291`, RM path) always take the L1-alignment branch, leaving `is_blackhole` / `dst_is_dram`
    effectively dead there. In the dead RM path regardless; flagged for ops-team awareness.
- **Doc-evolution suggestion.** Consider the one-line recipe note described under Friction → Gaps.
