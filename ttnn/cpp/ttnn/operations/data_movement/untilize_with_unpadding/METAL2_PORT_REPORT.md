# Metal 2.0 Port Report — `data_movement/untilize_with_unpadding`

## Outcome

**`PORTED` (3 factories) · `CAPITULATED` (1 factory).** Both halves are success-tier; the
capitulation is the more load-bearing deliverable of the two, because it caught a **silent
wrong-numerics framework defect** before it shipped.

| factory | in the brief's scope? | result |
|---|---|---|
| `UntilizeWithUnpaddingSingleCoreProgramFactory` | yes | **PORTED** — `MetalV2FactoryConcept` |
| `UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory` | yes | **PORTED** — `MetalV2FactoryConcept` |
| `UntilizeWithUnpaddingMultiCoreNDShardedProgramFactory` | yes | **PORTED** — `MetalV2FactoryConcept` |
| `UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory` | yes | **CAPITULATED** — stays on `create_descriptor`; blocked by two framework defects (Handoff points 1 and 2) |
| `UntilizeWithUnpaddingMultiCoreShardedProgramFactory` | no — Device 2.0 gate | untouched, stays on `create_descriptor` |

The three ported factories pass the invoker-confirmed test set with **no regression**: the set is
green pre-port and post-port alike (see *Verification*, including a caveat about a misleading first
baseline run). `program_factory_t` is
mixed-concept post-port (three Metal 2.0, two `ProgramDescriptor`); the framework dispatches per
factory, and the op builds and runs.

## Provenance

- **Recipe docs (this port):** `1e4a7a62362 2026-07-27 docs(metal_2.0): split the runtime-args porting gate into its two sheet columns`
- **Audit docs (inherited):** `66ac84052d4 2026-07-27 docs(metal_2.0): split the runtime-args porting gate into its two sheet columns`

*(The brief's provenance line names `66ac84052d4`; the same doc revision appears on this branch as
`1e4a7a62362` — same subject, different hash, i.e. the audit ran against a rebase of the doc
branch.)*

## TTNN ProgramFactory

### Concept realized

`MetalV2FactoryConcept`, exactly as the audit chose, on the three factories that landed. Each
declares

```cpp
static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output);
```

with `create_descriptor` removed. No `op_owned_tensors`, no `override_runtime_arguments` — the base
`ProgramSpecFactoryConcept` path (tensor-binding refresh on cache hit) is what the op needs.

The audit's concept choice was not revised. The capitulated factory is a *framework* block, not a
concept mismatch: its Metal 2.0 spec builds and validates cleanly and only misbehaves at dispatch.

### Device-op-class edits

- **Custom `compute_program_hash` deleted:** none — the op never had one.
- **Pybind entry points removed:** none — `untilize_with_unpadding_nanobind.cpp` binds only the
  public `ttnn::untilize_with_unpadding` free function.
- `device/untilize_with_unpadding_device_operation.{hpp,cpp}` is **byte-identical to pre-port**:
  `select_program_factory`, `validate_on_program_cache_miss`, `compute_output_specs` and every
  `TT_FATAL` untouched.

The port forced **zero** device-op-class edits — the success case the integration doc describes.

### Open items

- **Relaxation candidates:** none applied, none warranted. Every `TensorParameter` keeps the
  default strict `TensorSpec` match; the op has no custom hash, so there is no legacy relaxation to
  mirror, and the confirmed readiness sheet names none.
- **Capabilities this op would benefit from:** (a) a per-node runtime-vararg count that is not
  deprecated (Handoff point 3); (b) DFBs that can be partitioned across disjoint node sets without
  corrupting L1 (Handoff point 1) — without it, any op whose legacy CBs are sized per core group is
  unportable.

## Handoff points

### 1. Metal 2.0 framework — per-node DFB config region is **sized by DFB count** but **addressed by DFB id** (WH/BH). Silent L1 corruption.

**Owner:** Metal 2.0 runtime. **Severity:** silent wrong numerics, no validator error.
**This is what blocked `MultiCoreBlockInterleaved`.**

**Mechanism.** Two sites disagree about what indexes the per-node DFB config region:

- **Sizing** — `tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:799-838` (`finalize_dfbs`):
  for each `KernelGroup` it sums `serialized_size()` over the DFBs whose `core_ranges` intersect
  that group and takes the max across groups. On WH/BH `serialized_size()` is a fixed 4 words
  (`dataflow_buffer.cpp:345-348`), so the region is `16 × (max number of DFBs **resident on** any
  one kernel group)` — a **count**.
- **Addressing** — `tt_metal/impl/program/dispatch.cpp:1381-1390`: each DFB's 4-word config is
  written at `dfb->id × UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG × 4` bytes from
  `program_config.dfb_offset`, where `dfb->id` is the **program-global** index assigned in
  `ProgramSpec::dataflow_buffers` order (`dataflow_buffer.cpp:885`,
  `metal2_host_api/program_spec.cpp:2938-2950`). The device side agrees — the firmware's
  `get_local_cb_interface(id)` reads slot `id`.

So whenever a node hosts a DFB whose `id ≥ (number of DFBs resident on that node)` — i.e. whenever
DFBs are **partitioned across disjoint node sets** — the dispatch writes up to
`16 × (max_id + 1)` bytes into a region reserved for `16 × count`, running off the end and
clobbering whatever L1 section follows.

**The legacy CB path does this correctly**, which is the clearest statement of the fix:
`finalize_cbs` (`tt_metal/impl/program/dispatch.cpp:307-320`) sizes `local_cb_size` from
`max_local_end_index` — the **highest CB index in use**, recovered from the kernel group's
`local_cb_mask` — not from a count.

**Repro.** `tests/ttnn/unit_tests/base_functionality/test_to_layout.py::test_untilize_w4`
(bf16 `[1, 1, 32, 10912]` TILE input, `output_tensor_end = [0, 0, 0, 10911]`), against the
per-sub-region DFB shape described in `METAL2_PORT_PLAN.md` → *Planned Spec Shape →
MultiCoreBlockInterleaved*. That configuration yields 4 `DataflowBufferSpec`s (ids 0-3) over two
disjoint node sets of 56 and 1 cores, each node hosting 2. Observed:

```
Finalize dfb: dfb_offset == base_offset: 112, dfb size: 32, return value: 144
```

— a **32-byte (2-slot)** region, while ids 2 and 3 are written at byte offsets 32 and 48.
Result: 8093 of 10912 output elements wrong, then
`free(): invalid next size (fast)` / `Fatal Python error: Aborted` at device close, inside
`MeshDevice::close()` → `RealtimeProfilerManager::shutdown()` → `ReadFromDeviceL1`.

**Confirming experiment.** Collapsing the same program to a **single shared DFB pair** (2 DFBs,
ids 0-1, resident on every node — everything else, including the four per-region reader / writer /
compute `KernelSpec`s and four `WorkUnitSpec`s, unchanged) makes that shape and every other
2-region shape produce **bit-exact** output. That isolates the defect to DFB multiplicity, not to
the kernel translation or the argument plumbing.

**Suggested fix.** Size the region from the highest **resident id** (mirroring `finalize_cbs`), or
pack the WH/BH configs sequentially and give the device an id→slot map. A cheap interim guard: in
`finalize_dfbs`, `TT_FATAL` when `max_resident_id + 1 > resident_count` so the failure is loud
instead of silent.

**Why the port did not work around it.** The two available workarounds — a single max-sized DFB
pair, or binding every DFB on every node — both change the per-node L1 footprint that
`cb_block_size_limit`
(`untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp:83`) exists to bound.
Per [§When the discipline doesn't fit](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#when-the-discipline-doesnt-fit),
an API that behaves as though it has a bug gets reported, not papered over.

### 2. Metal 2.0 framework — a second, unresolved wrong-numerics failure that survives the DFB collapse

**Owner:** Metal 2.0 runtime (unconfirmed). **Status:** open hypothesis, not root-caused —
recorded per the recipe's stuck-point time budget rather than chased further, because Handoff
point 1 already blocks the factory independently.

With the single-shared-DFB-pair variant from the experiment above (so Handoff point 1 is not in
play), bf16 `[1, 1, 1280, 1280]` unpadded to `[0, 0, 1279, 1279]` still returns
**120627 / 1638400 elements wrong**. The legacy `create_descriptor` factory returns bit-exact
output for the same shape.

What is known:

- That shape produces **four** non-empty work-split sub-regions (36 `full` + 6 `cliff_row` +
  6 `cliff_col` + 1 `cliff_col_row` cores) → 4 `WorkUnitSpec`s, and one source instantiated as 4
  `KernelSpec`s for each of reader / writer / compute.
- Four regions is **not** by itself the trigger: `[1, 1, 2048, 2080]` also produces four regions
  with the same region kinds and **passes**.
- The distinguishing feature of the failing shape is that its `full` region is the first case where
  a compute kernel streams **multiple blocks** (`block_size_col = 6`, so
  `untilize<6>(6)` = 6 blocks of 6 tiles) through the DFB; every passing case had
  `block_size_col = 1`.
- Untested hypothesis worth checking first: `MakeDataflowBufferConfig` derives `num_producers` /
  `num_consumers` / `stride_in_entries` from the number of **bound `KernelSpec`s** rather than the
  per-node instance count. On Gen1 the 4-word serialized config does not carry those, so they
  *should* be inert — but the credit/capacity derivation is the only part of the pipeline that
  scales with binding multiplicity, and multi-block streaming is the only workload that exercises
  the credit path hard.

**Regression shapes for whoever picks this up** (all bf16, TILE input, interleaved output,
`use_multicore=True`; all six pass on `create_descriptor`):

| shape | `output_tensor_end` | sub-regions | per-region DFBs | shared DFB pair |
|---|---|---|---|---|
| `[1, 1, 32, 1536]` | `[0, 0, 31, 1535]` | 1 (`full`) | pass | pass |
| `[1, 1, 32, 2048]` | `[0, 0, 31, 2047]` | 1 (`cliff_col`) | pass | pass |
| `[1, 1, 32, 10912]` | `[0, 0, 0, 10911]` | 2 | **FAIL** | pass |
| `[1, 1, 32, 10912]` | `[0, 0, 31, 10911]` | 2 | **FAIL** | pass |
| `[1, 1, 1280, 1280]` | `[0, 0, 1279, 1279]` | 4 | **FAIL** | **FAIL** |
| `[1, 1, 2048, 2080]` | `[0, 0, 2047, 2079]` | 4 | **FAIL** | pass |

None of these shapes is covered by the op's existing test suite; `test_untilize_w4` is the only
in-tree test that reaches the multi-sub-region block path at all, and it exercises exactly one of
the six. **Test-coverage gap worth closing regardless of the port** (see *Open items*).

### 3. Metal 2.0 API — the only path for a ragged (per-node) runtime-vararg count is a `[[deprecated]]` field

**Owner:** Metal 2.0 API. **Severity:** design debt, not a build problem.

`MultiCoreInterleaved`'s writer (`writer_unary_stick_layout_split_rows_multicore.cpp:73-99`) reads
a 5-tuple-per-group vararg block whose **length varies per node** — the group count follows each
core's block assignment, produced at
`..._multi_core_interleaved_program_factory.cpp:196-222`. The scalar
`KernelAdvancedOptions::num_runtime_varargs` cannot express that, so the port uses
`KernelAdvancedOptions::num_runtime_varargs_per_node`
(`tt_metal/api/tt-metalium/experimental/metal2_host_api/advanced_options.hpp:80-87`), which carries
`[[deprecated("Per-node-vararg-count feature is deprecated and will be removed.")]]` and the
comment *"This feature is truly bizarre. It will be removed from the API once existing uses are
refactored to avoid it."*

There is no in-scope refactor: the ragged length is intrinsic to the op's work split, and padding
every node to the maximum would change the RTA dispatch-buffer size on most nodes. Either the
typed-`std::array` argument work that supersedes varargs needs to cover the ragged case, or this
field needs to outlive its deprecation. (`-Werror` is on for the host build and the use does not
trip `-Wdeprecated-declarations`, so nothing is broken today.)

### 4. Ops team (`data_movement`) — dead compile-time args

Host-emitted CTA values no kernel reads. All are carried forward verbatim as *named* CTAs so the
port stays a pure syntax swap; deleting them is a separate, behavior-neutral cleanup.

- `..._single_core_program_factory.cpp:136` → CTA `unpadded_stick_size`.
  `writer_unary_unpad_dims_split_rows.cpp` reads only CTA 0 and `TensorAccessorArgs<2>()`.
  **New finding** — not in the audit's Misc-anomaly list.
- `..._multi_core_nd_sharded_program_factory.cpp:154, :164` → CTAs `output_stick_size` (slot 1) and
  `input_single_tile_size` (slot 8). Already audit Misc anomaly 2; confirmed during the port.

### 5. Ops team (`data_movement`) — dead runtime args

RTA values read into a kernel local that is never used. Also carried forward verbatim.

- `writer_unary_unpad_dims_split_rows.cpp`: `num_unpadded_X`, `padded_X_size`,
  `num_blocks_w_input` (legacy RTA slots 8, 9, 10). **New finding.**
- `writer_unary_stick_layout_wh_multicore.cpp`: `single_block_size_row_arg` (legacy RTA slot 4) is
  re-read inside the `third_dim` loop and never used. **New finding** (this kernel reverted with the
  capitulation, so the finding is against the legacy file).

### 6. Ops team (`data_movement`) — degenerate zero-block configuration would now hard-error

With `num_blocks == 0` (an input whose last padded dim is 0) `MultiCoreInterleaved` computes
`ncores == 0`, so `all_cores` and every sub-range are empty. Legacy emitted a `ProgramDescriptor`
whose kernels covered no cores — a silent no-op. Metal 2.0 requires *"A valid ProgramSpec has at
least one WorkUnitSpec"*, so the ported factory would `TT_FATAL` instead. The guard
`input_shape[-1] == 0 ? 0 : …` at `..._multi_core_interleaved_program_factory.cpp:60` suggests
someone once hit this shape. Nothing in the confirmed test set exercises it, and inventing a new
empty-program behavior is out of the porter's scope — flagging so the owner can decide whether to
reject it earlier in `validate_on_program_cache_miss` or confirm it unreachable.

### 7. Not re-raised

The `unpad_tensor_w_16` + interleaved-output latent bug (audit Misc anomaly 1) sits in the
out-of-scope `MultiCoreSharded` factory and was not touched.

## Successes

- **[Caution: Avoid varargs — the "non-signal" callout](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary)
  fired exactly right.** The brief and the catalog both single out
  `writer_unary_stick_layout_wh_multicore.cpp:65-70` as a *non*-vararg: six args re-read inside the
  `third_dim` loop at **constant** indices. Read cold, a loop body full of `get_arg_val` looks like
  a vararg block; the reason it isn't is that the loop mutates its locals and needs a fresh copy
  each pass, not that it indexes a collection. The warning is what kept those six as named args.
  Keep this callout — it is the clearest worked example of the distinction in the catalog.

- **[Hardware configuration — "read the *resolved* settings, not the constructor spelling"](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#hardware-configuration).**
  Followed literally: `ReaderConfigDescriptor{}` lowers (`tt_metal/impl/program/program.cpp:414`) to
  `ReaderDataMovementConfig`, whose resolved triple is
  `RISCV_1 / preferred_noc_for_dram_read(arch) / DM_DEDICATED_NOC`
  (`tt_metal/impl/kernels/kernel_types.cpp:19-22`, `kernel_types.hpp:134-139`) — byte-for-byte the
  reader default, so `ttnn::create_reader_datamovement_config(arch)` is the correct target and not
  merely a "close" helper. Same for the writer. Chasing values rather than names turned this into a
  two-minute check.

- **[Anti-pattern: Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta).**
  `MultiCoreInterleaved`'s two compute descriptors and `MultiCoreBlockInterleaved`'s four carry
  different CTA values, and collapsing each into one `KernelSpec` with an extra RTA was the tidy-
  looking move. The entry's flat statement that multiple `KernelSpec`s per source is *supported*
  settled it before any code was written.

- **The [§When the discipline doesn't fit](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#when-the-discipline-doesnt-fit)
  off-ramp is why this PR does not ship silently-wrong numerics.** When `test_untilize_w4` started
  failing, the two obvious "fixes" were both one-line spec changes that make the symptom disappear
  (collapse to one DFB pair; oversize the DFBs). The section's instruction — *an API that behaves as
  though it has a bug: report it, don't paper over it* — is what turned a workaround into a root
  cause, a repro, and a framework ticket. Worth keeping the sentence about API bugs specifically;
  the rest of the section reads as being about *missing* capabilities, and a defect in a capability
  that exists is easy to file under "my mistake" instead.

## Friction

### Gaps

1. **The shared-dataflow-kernel recognition signal covers only kernels living outside the op
   directory, so an in-directory kernel with an outside legacy consumer is not detected by the
   legacy-inventory step.**
   `device/kernels/dataflow/writer_unary_stick_layout_wh_multicore.cpp` is owned by *this* op and
   sits squarely inside the porter's writeable surface — the
   [legacy-inventory](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#legacy-inventory)
   step's "flag any path outside the op's own directory" test does not fire on it, and neither does
   the plan template's *Cross-op kernels* section ("List any kernel `source` path outside the op's
   directory"). But it is instantiated **by file path** from
   `data_movement/untilize/device/factories/untilize_multi_core_block_program_factory.cpp:196`,
   which stays on `create_descriptor`. A Metal-2.0-ifying in-place edit would have broken `untilize`
   silently — surfacing as a JIT `static_assert` in *another op's* tests, exactly the failure the
   [shared-dataflow-kernel Caution](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-modifying-a-shared-dataflow-kernel)
   exists to prevent. **The right test is "who instantiates this file?", not "where does this file
   live?"** — the inventory step should run
   `grep -rl <kernel-filename> ttnn/cpp/ttnn/operations/` over **every** kernel the port touches,
   in-directory ones included, and route any hit outside the port's factory set to the fork
   decision. *(Caught here only because the invoker named it up front.)*

2. **No guidance for a legacy CB emitted several times, at one `buffer_index`, over disjoint core
   ranges, with different sizes — and following the obvious reading walks into Handoff point 1.**
   `MultiCoreBlockInterleaved` calls `push_cb_pair(...)` up to four times
   (`..._multi_core_block_interleaved_program_factory.cpp:127-166`), producing up to four `c_0`/`c_16`
   pairs whose `total_size` differs per sub-region. The recipe's default —
   *"DataflowBufferSpecs: one per legacy `CBDescriptor`"* — gives the right count but says nothing
   about the consequence: a DFB's placement is **derived** from the union of its bound kernels'
   `WorkUnitSpec`s, so preserving the legacy per-node L1 footprint forces the **reader and writer to
   be split per region too**. That is a *placement*-driven multiplicity, and
   [Preserved Multiplicity](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#preserved-multiplicity)
   frames multiplicity exclusively in terms of *per-group CTAs*, so it reads as not-applicable.
   Worth a catalog entry on its own — and the entry should carry the Handoff-point-1 warning, since
   the shape it prescribes is currently broken at dispatch. **Suggested pattern:** *"Per-region CB
   sizes → per-region DFBs force per-region reader/writer KernelSpecs (blocked on framework
   issue: DFB config-region sizing)"*.

3. **The vararg documentation stops at the scalar count.** Both the
   [migration guide](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/migration_guide.md#programrunargs)
   and the [catalog Caution](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary)
   present `num_runtime_varargs` / `num_common_runtime_varargs` as *the* schema-side mechanism, and
   the worked example (an N-D shape bounded by a `rank` CTA) has a count uniform across nodes. A
   **ragged** block — same kernel, different element count per node — is a distinct shape with a
   distinct API (`num_runtime_varargs_per_node`) that nothing in the recipe, guide, or catalog
   mentions. Found only by reading `advanced_options.hpp:80-87` after the validation math in
   `program_run_args.cpp:173-186` made clear the count is enforced *per node*. One sentence in the
   vararg caution — *"if the count differs per node, the schema field is
   `num_runtime_varargs_per_node`"* — plus the deprecation caveat from Handoff point 3 closes this.

4. **No disposition stated for a dead CTA / dead RTA.** The
   [Dropped Plumbing](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#dropped-plumbing)
   list enumerates six categories, all of which are *replaced* by a Metal 2.0 primitive. A host arg
   the kernel simply never reads (five sites in this op) fits none of them. Dropping it is a
   behavior-neutral cleanup that scope discipline says to route to the report; carrying it forward
   means minting a named CTA/RTA with no kernel-side reader, which looks like a mistake to a
   reviewer. The recipe has a precedent for the analogous CB case (*"For a **dead CB** … build **no**
   spec … recording each with `file:line` in the report"*) but not for args. This port chose
   **carry-forward + report**, reasoning that a dead CB is *forced* out by the bindingless-DFB
   validator error while a dead arg has no forcing function, so preserving it is the smaller diff —
   but the rule should be stated rather than inferred.

5. **The recipe's verification step assumes the op's own tests reach the code you changed.** All
   four in-scope factories are selected by `select_program_factory` from shape/config heuristics,
   and the confirmed test set turns out to reach `MultiCoreBlockInterleaved` through exactly **one**
   test (`test_to_layout.py::test_untilize_w4`, and only via `to_layout`, not via the op's own test
   file). Had that one test not existed, this port would have shipped green with corrupt numerics on
   a whole factory. A line in [Run tests](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#run-tests)
   — *"for a multi-factory op, confirm the confirmed test set actually selects each ported factory;
   if a factory has no coverage, write a shape-targeted probe before trusting a green run"* — would
   make that explicit. Writing such a probe (six shapes, ~40 lines) is what localized both defects
   here.

### Confusion

1. **`get_local_cb_interface(cb).fifo_page_size` → `get_entry_size()` is not a raw-field rename.**
   The [CB→DFB whitelist §B](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/cb_dfb_api_whitelist.md)
   maps the two directly, which reads as "same value, nicer spelling". It isn't:
   `DataflowBuffer::get_entry_size()` returns `address_units_to_bytes(fifo_page_size)`
   (`tt_metal/hw/inc/internal/tt-1xx/dataflow_buffer.inl:35-41`), i.e. `fifo_page_size << cb_addr_shift`.
   The swap is exact **only because `cb_addr_shift == 0` off the TRISC path**
   (`tt_metal/hw/inc/internal/circular_buffer_interface.h:143-148`); in a compute kernel the same
   substitution would change the value. `reader_unary_interleaved_start_id.cpp:20` is a DM kernel so
   the port is safe, but establishing that took a three-file trace. The whitelist already carries a
   nearby note (*"TRISC size getters return sizes in bytes, not 16B units"*) — spelling out that this
   is precisely why the DM raw-field read and the getter agree turns the trace into a read.

2. **The `unpack_modes` legality table reads more restrictive than it is.** The recipe's
   [Hardware configuration](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compute-kernels)
   warns that the validator *"rejects a `UnpackToDest` that can't fit its DFB's format (a 32-bit
   format into a 16-bit Dest …; a ≤16-bit format with `UnpackToDest` — rejected on Gen1 as a pure
   perf loss)"*. Every factory here sets `unpack_to_dest_mode[c_0] = UnpackToDestFp32` whenever
   `fp32_dest_acc_en`, **regardless of input dtype** — so on that reading, bf16 input with
   `fp32_dest_acc_en = true` would be a faithful translation the validator rejects, forcing either a
   divergence from legacy or a capitulation. Reading the validator
   (`tt_metal/impl/metal2_host_api/program_spec.cpp:1011-1013`) resolves it: `enable_32_bit_dest ==
   true` **short-circuits both format checks**, so the 1:1 translation is legal in every reachable
   config. The bullet list does say "enable=false" on those two lines, but the prose does not lead
   with the short-circuit — and the accumulation reference port *did* add a format gate, which reads
   as evidence the plain translation is unsafe. Leading with *"with `enable_32_bit_dest = true`,
   `UnpackToDest` is always accepted — copy the legacy entry as-is"* would remove the doubt.

3. **The reference port is stale enough to mislead on the API, not just on conventions.** The recipe
   already warns against leaning on ported ops and offers
   `akertesz/porting-experiment-accumulation-jun10` as a *shape* reference. Its
   `accumulation_program_factory.cpp` predates several renames: the entry point is
   `create_program_spec` (now `create_program_artifacts`); `hw_config` takes a flat
   `DataMovementHardwareConfig{.role = …}` and `ComputeHardwareConfig{.math_fidelity,
   .fp32_dest_acc_en, .dst_full_sync_en, .math_approx_mode, .unpack_to_dest_mode}` (now
   generation-split variants with renamed and one **inverted** field); and `runtime_arg_values` is
   populated node-first against what is now a name-first
   `Table<std::string, Table<NodeCoord, uint32_t>>`. Copying its shape produces code that does not
   compile — the benign failure — but the `ComputeHardwareConfig` field set is the dangerous one,
   because `math_approx_mode` / `dst_full_sync_en` → `sfpu_precision_mode` / `double_buffer_dest` is
   exactly the silent perf/precision flip the recipe warns about. Either refresh the branch or label
   it *"pre-`ProgramArtifacts`; do not copy `hw_config`"* where it is recommended.

## Open items for downstream

### Cross-op kernel touches — all **forks**, none in-place

Four kernels were forked with the `_metal2` suffix and shipped. No consumer of any of them could
co-migrate in this PR, so every legacy copy stays. Each fork carries a header comment naming its
unmigrated consumers and the sunset condition.

| kernel path (legacy) | path taken | fork path | remaining unmigrated consumers |
|---|---|---|---|
| `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | **fork** | `…/reader_unary_interleaved_start_id_metal2.cpp` | `copy/typecast`, `data_movement/copy`, `data_movement/pad`, `data_movement/untilize` (×2), `examples/example` (×2), `examples/example_multiple_return`, `experimental/transformer/nlp_create_qkv_heads_falcon7b`, `reduction/prod`, plus `tests/ttnn/unit_tests/gtests/test_generic_op.cpp` and `tests/ttnn/unit_tests/operations/debug/test_generic_op.py` |
| `data_movement/untilize/device/kernels/compute/untilize.cpp` | **fork** | `…/untilize_metal2.cpp` | `data_movement/fold`, `data_movement/untilize` (×4), `pool/upsample`, `sliding_window/halo`, `experimental/padded_slice`, `experimental/deepseek_prefill/combine`, plus `tests/tt_metal/tt_metal/llk/test_untilize_tilize.cpp` |
| `data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp` | **fork** | `…/untilize_variable_num_blocks_metal2.cpp` | `data_movement/untilize` (×3) |
| `data_movement/sharded/device/kernels/dataflow/reader_unary_nd_sharded_blocks.cpp` | **fork** | `…/reader_unary_nd_sharded_blocks_metal2.cpp` | `data_movement/untilize` (ND-shard-input factory) |

Three further forks were written and then **deleted** with the capitulation, because only
`MultiCoreBlockInterleaved` used them: `reader_unary_interleaved_wh_multicore_metal2.cpp`,
`untilize_wh_metal2.cpp`, `writer_unary_stick_layout_wh_multicore_metal2.cpp`. Their conversions
were mechanical and are not what blocked the factory; recreate them from the legacy originals when
it is re-attempted.

Kernels modified **in place** (this op is the sole non-Quasar consumer; the
`experimental/quasar/untilize_with_unpadding` clone carries its own copies under its own
`kernels/dataflow/` directory and does **not** borrow these by path — verified by grep):

- `device/kernels/dataflow/writer_unary_unpad_dims_split_rows.cpp`
- `device/kernels/dataflow/writer_unary_stick_layout_split_rows_multicore.cpp`
- `device/kernels/dataflow/writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp`

**Sunset checklist.** `data_movement/untilize` is the single largest unblocker: it is the sole
remaining consumer of two shipped forks and would be the sole remaining consumer of two of the three
deleted ones. The audit's planning note — *"`untilize_with_unpadding` and `data_movement/untilize`
share five kernels and should be sequenced as one unit"* — is confirmed by the port: **sequence
`untilize` next.** The two broad forks (`reader_unary_interleaved_start_id`, `untilize.cpp`) pull
~25 factories between them and need a coordinated wave, not a single follow-up PR.

### Per-op carry-over

- **`data_movement/untilize`** is a near-mirror of this op (five shared kernels; its block / ND /
  single-core / multi-core factories are structurally the same). Everything in the plan transfers —
  **including the block-factory blocker**: `untilize_multi_core_block_program_factory.cpp` has the
  same `push_cb_pair`-per-sub-region structure and will hit Handoff point 1 the same way. Port its
  other factories first.
- The `experimental/quasar/untilize_with_unpadding` clone is an independent copy with its own
  kernels; untouched and unaffected, but it is a second place these factories will need the same
  treatment if that tree is kept in sync.

### Doc-evolution suggestions

- Add the **per-region-CB → per-region reader/writer** pattern to the catalog, carrying the
  Handoff-point-1 warning (Friction → Gaps 2).
- Change the legacy-inventory cross-op-kernel test from *path-based* to *consumer-based*
  (Friction → Gaps 1).
- Add the "confirm the test set actually selects each ported factory" line to the verification step
  (Friction → Gaps 5).

### Test coverage notes

- **`MultiCoreBlockInterleaved` is effectively untested.** Of the confirmed test set, exactly one
  test reaches it (`test_to_layout.py::test_untilize_w4`), and the op's own test file
  (`tests/ttnn/unit_tests/operations/data_movement/test_untilize_with_unpadding.py`, 348 tests)
  reaches it **zero** times. The six-shape table in Handoff point 2 is a ready-made regression set
  and should be added to that file whether or not the port is re-attempted.
- **Cold-cache flakiness in the confirmed set.** The first run of the set produced 44 failures
  across `test_to_layout.py`, the `tt_eager` sweep wrapper and `test_sharded.py` that did not
  reproduce on a re-measured pre-port tree (see the *Verification* caveat). Whatever the trigger —
  cold JIT cache, device state carried from a preceding aborted run — these three files are not
  reliably green on a first execution, which is a hazard for anyone using them as a port baseline.
- No test in the confirmed set exercises the `num_blocks == 0` shape (Handoff point 6).

---

## Verification

- **Build:** `./build_metal.sh --build-tests` — SUCCESS, zero compiler errors or warnings
  (`Warnings as errors: ON`).
- **C++ gtest:** `./build/test/ttnn/unit_tests_ttnn --gtest_filter='*UntilizeWithUnpadding*'` —
  PASS, before and after.
- **Pytests:** the invoker-confirmed set (all but the sweeps), run identically before and after.
  **Every file is green in both runs — zero failures, before and after.**

| test file | pre-port and post-port (identical) |
|---|---|
| `unit_tests/operations/data_movement/test_untilize_with_unpadding.py` | 348 passed, 6 skipped, 6 xfailed |
| `nightly/.../test_universal_input_tm_untilize_with_unpadding.py` | 54 passed |
| `nightly/.../test_untilize.py` | 1 passed |
| `base_functionality/test_tilize_untilize_2D.py` | 240 passed |
| `base_functionality/test_untilize_bfloat8_b.py` | 176 passed |
| `base_functionality/test_to_layout.py` | 653 passed, 45 skipped, 8 xfailed |
| `tt_eager/.../pytests/tt_dnn/test_untilize_with_unpadding.py` | 5 passed |
| `tt_eager/.../unit_testing/misc/test_sharded.py` | 114 passed, 103 skipped, 2 xpassed |

> **Baseline caveat, recorded because it nearly became a false finding.** The *first* pre-port run
> of this set reported 44 failures spread across the last three files
> (`test_to_layout.py` 39, the `tt_eager` sweep-wrapper 3, `test_sharded.py` 2). Those failures did
> not reproduce: re-running the identical set on a freshly rebuilt **pre-port** tree at the end of
> the port gave 653 / 5 / 114 passed — exactly the post-port numbers. The first run was the session's
> first execution against a cold JIT kernel cache and followed an aborted earlier invocation, so the
> failures were an artifact of that run, not a property of the branch. **The real pre-port baseline
> is all-green**, which is what the table above compares against. Two lessons for the recipe's
> [Run tests](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#run-tests)
> step: take the baseline on a warm cache, and re-measure any "pre-existing" failure before writing
> it down — a fabricated pre-existing-failure list is exactly the cover a real regression would hide
> behind.

- **Factory-selection coverage.** Confirmed that the three shipped factories are each exercised:
  `SingleCore` via the `use_multicore=False` parameterizations in `test_tilize_untilize_2D.py` and
  `test_untilize_bfloat8_b.py`; `MultiCoreInterleaved` throughout the op's own test file;
  `MultiCoreNDSharded` via its ~10 dedicated tests there. The six-shape probe in Handoff point 2 was
  written specifically because `MultiCoreBlockInterleaved` had almost no coverage.

### Anti-pattern self-audit

Run against the three shipped factories and their four shipped/forked kernels plus the three
in-place kernels.

| check | result |
|---|---|
| No `tensor.buffer()->address()` survived | ✅ zero hits |
| No magic-number CB indices in CTAs | ✅ zero `CBIndex` / `CBDescriptor` / `CircularBuffer` / `.cbs` hits in the ported factories |
| No `TensorAccessorArgs<N>()` in any ported kernel | ✅ zero hits |
| Conditional DFB bindings follow the pattern | n/a — no conditional resource anywhere in the subset |
| No `.id` extraction at LLK call sites | ✅ `compute_kernel_hw_startup(dfb::in, dfb::out)` and `compute_kernel_lib::untilize<…, dfb::in, dfb::out, …>` pass the handle directly (NTTP position included) |
| No CTA→RTA demotion in compute kernels | ✅ `MultiCoreInterleaved`'s per-group block count stays a CTA on two separate `KernelSpec`s |
| No unnecessary multi-binding flag, never stacked with a self-loop | ✅ `allow_instance_multi_binding` appears nowhere; every DFB is a re-derived 1P+1C |
| All CTAs are named | ✅ every `compile_time_args` is `{{name, value}, …}` |
| No nameable argument smuggled into varargs | ✅ two vararg blocks retained, both genuine indexed collections (RTA per-group 5-tuples; CRTA shape dims); the leading writer args are named |
| Every `hw_config` reproduces the legacy resolved values | ✅ DM: `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` → `create_reader/writer_datamovement_config(arch)`, verified against `kernel_types.cpp:19-22,34-38`. Compute: `enable_32_bit_dest = fp32_dest_acc_en`; `unpack_modes = {IN → UnpackToDest}` iff `fp32_dest_acc_en`; `fpu_math_fidelity` HiFi4, `sfpu_precision_mode` Precise, `bfp_pack_precision_mode` Approximate, `double_buffer_dest` true — all four Metal 2.0 defaults coincide with the legacy `ComputeConfigDescriptor` defaults the op left unset |
| CB sweep — no `CircularBuffer` / `CBDescriptor` survives | ✅ zero hits in the three ported factories and all ported kernels (the two factories left on `create_descriptor` keep their legacy CBs by design) |
