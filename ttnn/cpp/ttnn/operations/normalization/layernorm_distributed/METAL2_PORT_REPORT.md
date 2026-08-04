# Metal 2.0 Port Report — `normalization/layernorm_distributed`

## Outcome

**`PORTED`** — all **five** factories converted to `ProgramSpecFactoryConcept`:
`LayerNormPreAllGatherProgramFactory`, `LayerNormPreAllGather2DProgramFactory`,
`LayerNormPreAllGatherWelfordProgramFactory`, `LayerNormPostAllGatherProgramFactory`,
`LayerNormPostAllGatherWelfordProgramFactory`. Nothing is left on the legacy concept; no
`create_descriptor` remains in the directory.

Verification status, including the paths this host cannot exercise, is under *Verification*.

## Provenance

- **Recipe docs (this port):** `4d6e9518bf5 2026-07-30 docs(metal_2.0): require an explicit opt_level when porting compute kernels`
- **Audit docs (inherited):** `4d6e9518bf5 2026-07-30 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept`, as the audit chose. All five factories return
`ttnn::device_operation::ProgramArtifacts` from `create_program_artifacts`; both
`program_factory_t` variants flip wholesale, so no per-factory concept mixing was needed.

### Device-op-class edits

- **Custom `compute_program_hash` deleted:** none — neither DeviceOperation declared one.
- **Pybind entry points removed:** none — `layernorm_distributed_nanobind.cpp` binds only the two
  user-facing functions, never a factory entry point, so no pybind surface changed.
- The only device-op-class change is the factory method signature in the two `*_device_operation.hpp`
  files (`create_descriptor` → `create_program_artifacts`) and the include swap that goes with it
  (`<tt-metalium/program_descriptors.hpp>` → `"ttnn/metal_v2_artifacts.hpp"`). `validate_on_program_cache_miss`,
  `compute_output_specs`, `create_output_tensors` and `select_program_factory` are untouched.

### Open items

- **Relaxation candidates:** none applied and none obviously available. Neither DeviceOperation has a
  custom hash, so no relaxation was ever in force to mirror.
- **Concept fit:** clean. No op-owned tensors, no op-owned `GlobalSemaphore`s, no per-coordinate
  program variation.

---

## Handoff points

### 1. `LayerNormPostAllGatherWelfordProgramFactory` is broken on `main`, and the port cannot reproduce the breakage — OPS TEAM

**Tagged: pre-existing defect surfaced by the port; a behavior change on one path is unavoidable.**

The shared post-allgather reader
(`device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_post_allgather.cpp`) reads **9** compile-time
args (`reduce_factor` at index 8, then `TensorAccessorArgs<9>()`) and **10** runtime args (`eps` at 5,
`gamma_addr` at 6, `beta_addr` at 7, `stats_addr` at 8, `y_offset` at 9). Pre-port, the Welford post
factory emitted **8** compile-time args (no `reduce_factor`) and **11** runtime args with
`packed_winv_value` inserted at slot 5, so:

- every runtime arg from `eps` onward landed one slot late (`eps` read `packed_winv_value`,
  `gamma_addr` read `eps`, `beta_addr` read the gamma address, `stats_addr` read the beta address);
- `TensorAccessorArgs<9>()` started one word inside the input accessor's argument block.

The default post factory has always matched the reader exactly, so the mismatch is Welford-only.

**Confirmed empirically on the pre-port tree.** A single-device
`layer_norm_pre_all_gather` → `layer_norm_post_all_gather` run with
`LayerNormDefaultProgramConfig(use_welford=True)` returned `max_abs_err = 10.48` against the torch
reference, with a NaN cosine similarity. The path survives on `main` because it is exercised only by
`mesh_device=(1, 8)` tests, which skip on a single-device host.

**Why the port cannot preserve it:** Metal 2.0 arguments are addressed by name, not by position, so
there is no positional shift left to reproduce — the ported factory binds `eps` to `eps` and the stats
address to the stats tensor. The ported Welford post path is therefore expected to be *correct* where
the legacy one produced garbage. This is the one place in the port where "no behavior change" does not
hold, and it is not a change the port could have declined to make.

**Ask:** confirm the ported behavior is the intended one, and decide whether the legacy path also needs
a fix on `main` ahead of this port landing, for anyone still on the old code.

### 2. RMSNorm + gamma + beta drove an unallocated buffer index — FUNCTIONAL FIX BUNDLED AT INVOKER'S DIRECTION

**Tagged: deliberate scope exception, authorized by the invoker mid-port.**

`rmsnorm_post_allgather.cpp` sets `cb_times_gamma_out_idx = tt::CBIndex::c_13` whenever both gamma and
beta are present and then drives it, but `layernorm_post_all_gather_program_factory.cpp` allocated
`c_13` only under `if (!is_rmsnorm)`. That config therefore drove an unconfigured buffer index (audit
*Misc anomalies* #1 / *Questions* #2). Metal 2.0 cannot bind a buffer no spec declares, so the port had
to resolve it one way or another.

Three options were put to the invoker: (a) faithful swap, letting that config fail at kernel-compile
time; (b) allocate the buffer for RMSNorm too; (c) reject the config in validation. **The invoker chose
(b).** The ported factory declares `TIMES_GAMMA_OUT` whenever gamma **and** beta are present, RMSNorm
included.

This is a **functional change inside a port diff**, which the recipe's scope discipline otherwise
forbids. It is called out here so a reviewer bisecting this commit knows the numerics of
RMSNorm + gamma + beta changed (from undefined to defined) for a reason unrelated to Metal 2.0.

### 3. Shared kernel touches in the peer `rmsnorm_distributed` directory — RMSNORM_DISTRIBUTED OWNERS

Two compute kernels this op file-path-instantiates live in a peer op's directory. **They took different
rungs, because their consumer censuses differ:**

| original | what the port did | remaining legacy consumers |
|---|---|---|
| `rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp` | **converted in place** (rung 3, invoker-authorized) | **none** |
| `rmsnorm_distributed/device/kernels/compute/rmsnorm_post_allgather.cpp` | **fork created** — `…/rmsnorm_post_allgather_metal2.cpp` + pointer comment in the original (rung 2) | **one, see below** |

`rmsnorm_pre_allgather.cpp` has no consumer other than this op: a repo-wide search for the basename
and the stem returns only this op's 1D factory, and the apparent hits under
`experimental/transformer/fused_distributed_rmsnorm` are that op's own separate copy with the same
basename and its own `wan_fused_rmsnorm_pre_allgather` API. With nothing left behind, a fork would
protect nothing while leaving two copies to keep in sync forever, since no consumer migration would
ever trigger its sunset. It was therefore converted in place at the invoker's explicit direction.
**This makes the port modify a peer op's source, which no CODEOWNERS entry gates** — flagged here so it
is a deliberate ownership crossing rather than something a reviewer discovers in the diff.

`rmsnorm_post_allgather.cpp` **must stay on legacy idioms.**
`tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py`
(`TestCrossOpCompilation`) lists that path in its `KERNEL_PATHS`, opens the file *as text*, splices its
file-scope section into a synthetic fused kernel, and compiles it through a legacy
`ttnn.KernelDescriptor` with `SourceType.SOURCE_CODE`. That path has no generated binding headers, so a
Metal 2.0 version of the file would not compile there. Its fork is load-bearing, not cautious.

**A note on how to count who else uses a kernel file, because this port got it wrong first.** Before
converting a kernel that other code might share, the porter has to establish who else uses it. An earlier
draft of this report claimed both originals had no user left, based on searching for program factories
that name them as a kernel source. That search missed `TestCrossOpCompilation`, which reads one of the
files with `open()` and pastes its text into a generated kernel instead of naming it as a source. The
lesson: search for the filename, then read each hit and ask what it does with the file. "Which factory
names this as its kernel source?" is too narrow a question to establish that a file is unused.

No build file was touched: the per-family `file(GLOB_RECURSE kernels …)` already covers that directory.

### 4. A shared kernel-pool helper still takes a `CircularBuffer` — KERNEL-LIB OWNERS

`generate_bcast_col_scalar(CircularBuffer cb, uint32_t scalar)` in
`ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` takes the legacy wrapper **by value**. The
post-allgather reader calls it, so one `CircularBuffer` construction survives in this op's kernel code:

```cpp
generate_bcast_col_scalar(CircularBuffer(dfb::eps), eps);
```

The audit cleared this shape as ✓ (by value, not by reference), and it works because `dfb::eps`
converts to `uint32_t` at constexpr time. But it means the recipe's "no `CircularBuffer` reference
survives in the op directory" sweep cannot come back completely clean for this op. The helper is
outside the porter's scope, so it was not modified. A `DataflowBuffer` overload (or a templated one)
would let the last reference go.

---

## Successes

- **[Conditional / optional DFB bindings]** fired exactly as documented, and the warning that
  `if constexpr` does not gate name lookup was the single most load-bearing thing in the catalog for
  this op. Both post compute kernels gated their gamma/beta chains on `do_gamma` / `do_beta` **CTAs**
  via `if constexpr`; the pattern's *"promote a CTA gate to a define"* paragraph is exactly what those
  needed. Without it, the natural port would have kept the CTAs and hit a name-lookup failure on every
  gamma-absent build.

- **The same pattern's note about always-emitted defines** caught a subtler variant. The legacy
  `FUSE_PRE_ADD` define was always emitted (as `"0"` or `"1"`) and tested with `#if`, which *looks*
  like it already gates correctly. It does not: `#if FUSE_PRE_ADD` still leaves `dfb::res` in the token
  stream on the unfused path. The port emits the define only when true and tests with `#ifdef`.

- **The catalog's instruction to work out each buffer's producer and consumer independently, rather than
  copying the audit's answer, is what caught the one case where the audit's answer could not be built.**
  The audit brief lists, per buffer, which kernel should be bound to which end, and it would have been
  natural to transcribe that list. The catalog says to re-derive it from the kernels themselves and to
  follow your own reading where the two disagree. For the Pre-Welford scratch buffer they do disagree, and
  the brief's answer is one the framework rejects outright; *Friction* #1 has the detail.

- **[Compiler options]** is a section that would have been skipped without its own warning. Four of the
  five factories set no `opt_level` at all, and the recipe's insistence that an absent
  `KernelDescriptor::opt_level` still resolves to `O3` on a `ComputeConfigDescriptor` is what put an
  explicit `KernelBuildOptLevel::O3` on all five compute `KernelSpec`s. Nothing would have failed
  without it — it would just have been quietly slower.

- **[Hardware configuration]** steered the port to `to_compute_hardware_config` and the reader/writer
  helpers rather than hand-built Gen1 configs, removing the whole class of "did I invert
  `dst_full_sync_en`?" mistakes. The `double_buffer_dest = !dst_full_sync_en` inversion in particular is
  a trap the helper closes.

- **Metal 2.0's requirement that every 32-bit-float buffer state its unpack destination works as a
  completeness check on the porter's own reasoning.** It found no defect in this op — every version of
  the code, legacy included, unpacks the buffer in question the same correct way. What it caught was that
  the set of conditions I had written for these entries did not actually cover every buffer, which is a
  mistake with no other symptom: the values were right, so no test could have flagged it. The check only
  works if the entries are written out individually; see *Friction* #4 for the case and for how a blanket
  rule suppresses exactly this signal.

---

## Friction

### Gaps

*Background for the first three entries, which all turn on the same mechanism.* A **dataflow buffer** is
Metal 2.0's replacement for a circular buffer: a small FIFO in a node's SRAM that one kernel fills and
another drains. Every kernel that touches one declares a **binding** saying which end it is on, the
**producer** (fills it) or the **consumer** (drains it). **Program-spec validation** is a host-side check
that runs before the program is built; when a set of bindings breaks one of its rules it aborts with a
`TT_FATAL` rather than letting the program run. Two of its rules matter below:

- **One of each, per node.** On every node where a buffer exists, exactly one producer kernel instance
  and one consumer kernel instance must run. A buffer with a producer but no consumer on some node is
  rejected, as is one with two producers there. Several kernels may share an end as long as their node
  sets do not overlap, so each node still sees one.
- **A buffer only one kernel touches is expressed as a self-loop:** that kernel is bound as both producer
  and consumer. When a buffer is self-looped, validation additionally requires that the set of kernels on
  the producer end and the set on the consumer end be *the same set* — a self-looping kernel may not share
  the buffer with an unrelated one.

**1. When one kernel drives both ends of a buffer and a second kernel also fills it, the audit's
prescribed resolution cannot be expressed.**

The Pre-Welford factory has such a buffer, `c_1`. Two kernels touch it: the reader pushes a
reduce-scalar tile into it (correct in the 1D factory, which shares that reader and really does use
`c_1` as its scalar buffer), while the Welford compute kernel independently uses it as scratch for its
post-Welford transpose, both filling and draining it.

The audit brief prescribes the `allow_instance_multi_binding` advanced option, which relaxes the
one-of-each-per-node rule for buffers that genuinely have more touchers than roles. That resolution is
not available here: it would mean binding the compute kernel to both ends and the reader to the producer
end, and the self-loop rule above then requires the two ends to name the same kernels, which they do not.
The flag does not relax that second rule, so validation rejects the spec either way.

What works is to give the two kernels opposite ends: reader as producer, compute as consumer, no flag.
Every node then has exactly one of each. The compute kernel still fills the buffer at runtime, and that
is fine on Wormhole and Blackhole, where a dataflow buffer is implemented as a plain circular buffer
whose FIFO counters live in SRAM and are updated by whichever processor executes the call. The binding is
bookkeeping the host uses to check the topology and to configure the hardware; it does not gate access.
Nor does the choice of end affect the buffer's data format or its `unpack_modes` slot, both of which are
properties of the buffer itself.

*Suggested doc change:* the endpoint-assignment procedure's third bullet ("≥2 kernels locked to the
same FIFO role → multi-binding") needs a carve-out: **if one of those kernels drives both ends,
multi-binding is not available — drop the self-loop and give the two kernels opposite ends.** Worth
stating both in the patterns catalog and in the audit document's own table of buffer-to-binding
recommendations, since the auditor working from that table arrives at the same unusable answer first.

**2. A producer and consumer covering different node sets forces one kernel source to be instantiated
twice, and that shape is not described anywhere.**

In the 2D pre factory the final output buffer is filled by the compute kernel, which runs on the whole
grid, and drained by the writer, which runs only on the merge row. Under legacy circular buffers that was
fine. Under the one-of-each-per-node rule it is not: on every worker node the buffer would have a
producer and no consumer, and validation aborts.

The audit brief does flag this buffer, under *"DFB core range narrower than its binding kernel's core
range"*, but asks the porter to "confirm the spec validator accepts it." It does not accept it, so the
wording sends the porter looking for reassurance instead of a redesign.

The way out is to describe the one compute source as **two** kernels rather than one, covering the merge
row and the worker rows separately. Only the merge one binds the output buffer, so on worker nodes that
buffer no longer exists and the rule is satisfied. Because the buffer is bound on one of the two and not
the other, the flag selecting between them has to be a preprocessor define rather than a runtime
argument: the kernel-side name for a buffer exists only where the host binds it, so a runtime `if` around
the merge block would still leave an undeclared name for the compiler to resolve on worker builds.

Nothing in the recipe or the patterns catalog describes this. Instantiating one source several times *is*
covered, but only as a way to preserve per-core compile-time constants from a work split, and this op has
none of those, so the port plan's corresponding section legitimately reads "no work-split multiplicity in
legacy". Both statements were true, and duplication was still required, for an unrelated reason.

*Suggested doc change:* add a pattern entry — *"asymmetric producer / consumer placement → split the
wider kernel into per-region KernelSpecs"* — and reword the audit's watch-for from "confirm the
validator accepts it" to "this will be rejected; plan for a KernelSpec split." A porter who takes the
watch-for at face value discovers this only at first run, after the whole factory is written.

**3. Two rules disagree about whether the last `CircularBuffer` reference may remain.**

Metal 2.0 renames the kernel-side circular-buffer type to `DataflowBuffer`, and the recipe's kernel-side
rules call that replacement total: after the port, searching the op's directory for `CircularBuffer`
should find nothing. Separately, the audit checks every function this op's kernels call across a
directory boundary and passes each as portable. One of those, the shared helper
`generate_bcast_col_scalar`, takes a `CircularBuffer` **by value**, and the audit passes it precisely
because by-value is the easy case: the caller can build the old wrapper on the spot from the new binding
handle.

Both rules are sensible and they cannot both be followed. Building that wrapper is the only way to call
the helper, and the helper is outside the porter's writeable surface, so one `CircularBuffer` reference
necessarily survives (details in *Handoff points* #4).

*Suggested doc change:* qualify the "total" claim with "except a wrapper built at a call site whose
out-of-directory callee still takes `CircularBuffer` by value," and say to report each such site rather
than leaving the porter to decide which rule to break.

**4. Filling the `unpack_modes` table from a blanket rule silently cancels the safeguard it is meant to
satisfy.**

*Background, for a reader meeting this field for the first time.* Before a compute kernel can do math on
a tile, the tile has to be moved out of SRAM into one of the compute engine's register files. There are
two possible destinations, and they are not equivalent:

- **SrcA / SrcB** — the operand registers the FPU (the matrix-and-vector math unit) reads from. Values
  are narrowed on the way in, so a 32-bit float loses precision here.
- **Dest** — the accumulator register. Unpacking straight into it keeps the full 32 bits and is the only
  path the SFPU (the transcendental unit) can consume, but it is slower on Wormhole and Blackhole.

`KernelSpec`'s compute hardware config carries an `unpack_modes` table naming which destination each
buffer uses. For a buffer holding 32-bit floats the choice changes the op's numerics, so **Metal 2.0
refuses to pick for you**: if a compute kernel enables the 32-bit Dest register (`enable_32_bit_dest`)
and consumes a buffer whose format is `Float32`, the program spec must contain an `unpack_modes` entry
for that buffer. Program-spec validation, which runs on the host before the program is built, rejects
the spec otherwise, with a message of the form *"consumes FP32 DFB 'x' with enable_32_bit_dest=true, but
provides no unpack_modes entry"*. The legacy API had no such requirement: its equivalent was a vector
indexed by buffer id whose unset slots meant "SrcA/B", so legacy always had an answer, just never an
explicit one.

*Why this op feels the rule heavily.* When `fp32_dest_acc_en` is set, this op's intermediate buffers are
created as `Float32`, so most of the buffers each compute kernel consumes need an entry. Which buffers
exist at all varies with the config: layernorm versus RMSNorm, gamma present, beta present, residual
present. Producing that list correctly is most of the work of setting this field.

*The hazard.* It is tempting to satisfy the requirement mechanically: walk the compute kernel's consumed
buffers, and for every one whose format is `Float32`, write in "SrcA/B" (the value legacy's unset slots
meant). This produces correct values and always satisfies validation. **That is exactly why it is
wrong.** The condition such a rule tests — *32-bit Dest enabled, buffer consumed, format is Float32* —
is the same condition validation tests. A rule keyed on it therefore does not answer the question the
requirement asks; it guarantees the question is never asked, for every buffer in the op, including
buffers added years later. The requirement exists to make a human choose per buffer, and a blanket rule
turns it back into a default with no one behind it.

The entries in this op are consequently written one per buffer, next to the binding they describe, using
`unpack_via_src` / `unpack_via_dest` so each reads as a decision rather than a table update.

*What this cost in practice.* One buffer in this op needs an entry that is easy to miss when the entries
are written by hand. In the Pre-Welford factory the fused a+b buffer is created as `Float32` whenever the
32-bit Dest register is enabled — its format follows that setting, not the input's dtype. The factory's
one reason to route it into Dest, however, is to keep 32-bit precision through the pre-add, and that
reason only applies when the **input tensor** is also `Float32`. So the natural set of conditions to write
(and the set the legacy code used) mentions this buffer only on the fp32-input path, leaving
`fp32_dest_acc_en` + bfloat16 input + residual with a consumed `Float32` buffer and no entry.

**No behavior was ever wrong, in any version.** SrcA/B is the correct destination for that buffer in that
config, and it is what every version does:

| version | what happens | numerics |
|---|---|---|
| legacy, pre-port | the entry is absent and absence means SrcA/B | correct |
| port with the blanket rule | the rule writes SrcA/B | identical to legacy |
| port with the rule removed, entry not yet written | program-spec validation rejects the spec: **the op cannot run at all in that config** | n/a, hard error |
| port as it stands | the entry says SrcA/B, next to the binding | identical to legacy |

So this was **not a bug, latent or otherwise**, and the blanket rule did not introduce one — it produced
the right value. What it did was make an incomplete hand-written list impossible to notice: with the rule
in place the spec validated, so there was no signal that the conditions I had written did not in fact
cover every buffer. Remove the rule and the omission is not subtle at all — it is a hard failure at
program build.

**The tests caught it immediately.** `test_layernorm_pre_all_gather_welford_residual[bf16_inp_bf16_stats-…]`
covers exactly that combination, and three of its cases failed with
`consumes FP32 DFB 'prewf_fused' with enable_32_bit_dest=true` the moment the rule came out. There was
nothing for a test to catch before that: the earlier states were all numerically correct. That is the real
argument for writing these entries out by hand — not that a blanket rule gets the values wrong, but that
it removes the only feedback telling you whether your own reasoning about the buffer set was complete.

SrcA/B is right here for a reason worth recording: on that path the pre-add is `add_tiles`, an FPU
operation, and the FPU reads its operands from SrcA/B. Nothing is lost by narrowing on the way in either,
since with a bfloat16 input and residual the sum carries no more precision than bfloat16 to begin with —
the buffer is `Float32` only because the Dest register is.

*Suggested doc changes:* (a) state plainly that deriving these entries from a rule keyed on "Float32 and
consumed" defeats the requirement, and that entries belong beside the bindings they describe. The
existing "do not guess" instruction implies this, but a porter facing a dozen config-dependent buffers
will reach for the mechanical option unless told not to. (b) Add a row to the anti-pattern self-audit:
*every `unpack_modes` entry is an individual decision, not one filled in by a rule that tests the same
condition validation tests.*

### Confusion

**5. "Preserved Multiplicity" reads as being about work splits, so the *absence* of per-group CTAs felt
like a green light.** All five factories put their per-core row count in a runtime arg, so that plan
section legitimately reads "none." Correct for the anti-pattern it guards, but it primed the
expectation that one `KernelSpec` per legacy `KernelDescriptor` would always be right — which Friction
#2 then contradicted. Splitting the section into "multiplicity from work splits" and "multiplicity from
placement" would remove the false reassurance.

**6. The `Table` vs `Group` distinction bites hardest at `compiler_options.defines`.** `defines` is
`Table<std::string, std::string>`, and the natural conditional construction (declare empty, add entries
under an `if`) needs `emplace`, not `push_back`. The recipe does call this out and the call-out was
useful; noting it only because `defines` is the one field where conditional construction is the
*common* case, so it would be worth naming explicitly in that paragraph.

---

## Open items for downstream

### Shared kernel touches

Full detail in *Handoff points* #3: in `rmsnorm_distributed/device/kernels/compute/`,
`rmsnorm_pre_allgather.cpp` was converted in place (no consumer left behind) and
`rmsnorm_post_allgather.cpp` got a `_metal2` fork plus a pointer comment, because
`TestCrossOpCompilation` still reads its legacy source as text. Only the post kernel carries sunset
debt, and it cannot retire until that test stops reading the legacy file.

Four kernel sources **inside** this op's directory are shared between its own factories:
`writer_unary_interleaved_start_id_blocked.cpp` (all five factories),
`reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp` (two),
`reader_unary_interleaved_ln_rm_gb_post_allgather.cpp` (two), and the in-directory
`compute/chain_llk.hpp` header (two). Because all five factories converted in this one change, these
were converted **in place** and needed no intra-op fork. Porting the factories one at a time would have
forced three forks plus a duplicated header inside this directory — the concrete argument for keeping
this op's factories together, and worth knowing for any op with the same shape.

### Dead buffer allocations dropped

Every drop below removes an allocation with no toucher, so none changes behavior. The audit found the
first four; the last two are additional, config-conditional cases its per-`(CB, config)` table did not
separate out.

| buffer | site (pre-port) | scope |
|---|---|---|
| `c_9` (var + epsilon) | `layernorm_post_all_gather_program_factory.cpp:490-496` | dead in all configs |
| `c_9` | `layernorm_post_all_gather_welford_program_factory.cpp:554-560` | dead in all configs |
| `c_7` (mean²) | `layernorm_post_all_gather_welford_program_factory.cpp:583-589` | dead — Welford factory only |
| `c_8` (var) | `layernorm_post_all_gather_welford_program_factory.cpp:545-551` | dead — Welford factory only |
| `c_6` (reduced stats) | `layernorm_post_all_gather_program_factory.cpp:472-478` | dead **when `is_rmsnorm`** — that kernel reduces the stats straight into `c_8` and never names `c_6` |
| `c_13` (×gamma intermediate) | `layernorm_post_all_gather_program_factory.cpp:536-545` | dead **when beta is present but gamma is not** — the ×gamma stage is the buffer's only toucher, and it does not run |

Two kernel-side declarations that were dead alongside `c_9` went with it:
`layernorm_post_allgather.cpp:115` and `rmsnorm_post_allgather.cpp:52` (the latter only in the new
`_metal2` fork; the legacy original keeps it, since it must stay on legacy idioms).

### Findings left for the ops team

1. **The Pre-2D factory ignores `is_rmsnorm`** (audit *Misc anomalies* #6). It hardcodes
   `layernorm_pre_allgather_2d.cpp` with no RMSNorm branch and forces `out0_tiles = 1`, while
   `compute_output_specs` sizes a LAYERNORM output at two tile columns — so a LAYERNORM +
   `use_2d_core_grid` request appears to produce only `E(x²)` into a two-column output tensor. The port
   reproduces this exactly and does not act on it.

2. **The Post-Welford factory's `is_rmsnorm` compute-source branch is unreachable** — validation
   rejects RMSNorm + Welford before the factory runs. The branch is kept as written, and the buffer set
   and argument schema are built for the Welford kernel, the only reachable source. Worth deleting on
   the ops track, together with a decision on whether RMSNorm + Welford should ever be supported.

3. **`log_debug(tt::LogOp, "device_id: {}", gamma.value().device()…)`** in
   `layernorm_post_all_gather_welford_program_factory.cpp` dereferences `gamma` unconditionally, so a
   Welford post-allgather call without a weight faults there before any validation message. Left
   exactly as-is.

4. **`packer_l1_acc` is destructured and then ignored in all five factories** (audit *Misc anomalies*
   #7). Callers that set `packer_l1_acc=true` silently get no effect, while the value still feeds the
   default program hash and so causes cache misses that change nothing. The port preserves this exactly:
   `to_compute_hardware_config` also does not translate `packer_l1_acc`. The destructure is retained
   (marked `[[maybe_unused]]` in the two factories where nothing else reads it) so the validation it
   performs on the config still runs.

5. **The pre-allgather reader pushes a reduce-scaler tile into the Welford factory's scratch buffer**
   (audit *Misc anomalies* #2). Still true, still harmless, and now visible in the spec as the reason
   that buffer needs a reader-producer / compute-consumer assignment instead of a plain self-loop.
   Gating the reader's scaler generation behind a define, or giving the Welford factory a scratch index
   the reader does not touch, would let it become an ordinary compute self-loop.

6. **`layernorm_pre_allgather_welford.cpp` pushes into the output buffer with no preceding
   `reserve_back`** (audit *Misc anomalies* #8). Unchanged by the port.

### Test coverage notes

**The Post-Welford path has no single-device coverage.** Every test that reaches it is a
`mesh_device=(1, 8)` parametrization, which skips on a one-device host — which is exactly why the
defect in *Handoff points* #1 went unnoticed on `main`. The 2D core-grid paths are *not* in this gap:
`test_distributed_rmsnorm_allgather.py::test_rmsnorm_2d_core_grid_single_device[use_2d_core_grid=True-…]`
exercises both the Pre-2D factory and the post factory's 2D work split on a single device.

Adding a single-device Post-Welford case would be cheap and worth doing. The probe written to confirm
*Handoff points* #1 is the whole recipe: `layer_norm_pre_all_gather` → `layer_norm_post_all_gather`
with `LayerNormDefaultProgramConfig(use_welford=True)` and a `create_layer_norm_reciprocals` tensor
reaches the Welford post factory with no mesh and no all-gather at all, and compares directly against
`torch.nn.functional.layer_norm`.

---

## Verification

### Build

`./build_metal.sh -e --enable-fake-kernels-target` — **SUCCESS**, no warnings introduced (the tree
builds with `-Werror`).

### Tests

Baseline (pre-port) and post-port runs over the invoker-confirmed set:

- `tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_exhaustive.py`
- `tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_sharded.py`
- `tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_pre_allgather.py`
- `tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_post_allgather.py`
- `tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_rmsnorm_allgather.py`

`test_distributed_layernorm.py` and `tests/ttnn/distributed/test_distributed_layernorm_TG.py` skip
unconditionally today (`LEGACY_CCL_SKIP`, tt-metal#26649) and were excluded by the invoker.

| | passed | failed | skipped | xfailed |
|---|---|---|---|---|
| **Baseline (pre-port)** | 421 | 0 | 557 | 10 |
| **Post-port** | 421 | 0 | 557 | 10 |

Identical. The 557 skips are the `mesh_device=(1, 8)` parametrizations; this host has one device.

The full set was re-run after the `unpack_modes` change of *Friction* #4 (blanket rule removed, entries
written out one per buffer) and returned the same 421 / 0 / 557 / 10. Numerics are unchanged by that
change, and were expected to be: it alters how the spec states each buffer's unpack destination, not which
destination any buffer gets.

Between the two states the spec was briefly incomplete — the blanket rule gone but the Pre-Welford fused
buffer's entry not yet written — and three
`test_layernorm_pre_all_gather_welford_residual[bf16_inp_bf16_stats-…]` cases failed with
`consumes FP32 DFB 'prewf_fused' with enable_32_bit_dest=true`. That is a program-build rejection, not a
wrong answer, and it is recorded here only because it is what identified the missing entry.

A confirmation run of the three nightly files after the earlier cleanup pass (dead destructures removed,
ternaries parenthesized) gave 231 passed, 0 failed, 102 skipped.

**Paths this run does cover:** Pre 1D (LN and RMSNorm, with and without a residual, dtype mismatches),
Pre Welford (including the fp32-precision and residual cases), Post default (1D and 2D work splits,
gamma-only, beta-only, non-tile-aligned width), and the Pre-2D factory via the single-device
2D-core-grid RMSNorm test.

**Path this run does not cover:** Post Welford — see *Test coverage notes*. It was verified instead by
the direct probe below.

### Direct probe of the Post-Welford path

Single-device `layer_norm_pre_all_gather` → `layer_norm_post_all_gather` with
`LayerNormDefaultProgramConfig(use_welford=True)`, gamma and beta present, compared against
`torch.nn.functional.layer_norm`:

| tree | max abs error | cosine similarity |
|---|---|---|
| pre-port (`main`) | 10.475094 | **NaN** |
| post-port | 0.244583 | 0.999938 |

This is the behavior change described in *Handoff points* #1: the legacy path returned garbage, the
ported path is correct. The residual max-abs error is ordinary bfloat16 rounding on the i/o tensors.

### Anti-pattern self-audit

| check | result |
|---|---|
| No `tensor.buffer()->address()` survived | ✅ zero hits in code |
| No magic CB indices / `CBIndex` / `CBDescriptor` in the op directory | ✅ zero hits |
| No `TensorAccessorArgs<N>()` in any ported kernel | ✅ zero hits |
| Every `unpack_modes` entry is an individual decision, not one filled in by a blanket rule | ✅ each entry is a separate named call sitting beside the binding it describes, conditioned the same way. Deriving them from a rule that tests "buffer is Float32 and consumed" would satisfy program-spec validation while cancelling the choice it asks for; see *Friction* #4 |
| Conditional DFB bindings follow the pattern | ✅ `FUSE_PRE_ADD`, `FUSE_GAMMA`, `FUSE_BETA`, `IS_MERGE_CORE` each bound conditionally on the host, emitted as a `compiler_options.defines` entry, and `#ifdef`-gated on both the alias and every use. No binding made unconditional as a workaround. |
| No `.id` extraction or temp DFB wrappers at LLK call sites | ✅ zero hits |
| No CTA→RTA demotion | ✅ no per-group CTA existed to demote |
| No unnecessary multi-binding flag, never stacked with a self-loop | ✅ `allow_instance_multi_binding` appears nowhere in the port |
| All CTAs named | ✅ every `compile_time_args` is a `{name, value}` table |
| No nameable argument smuggled into varargs | ✅ no vararg mechanism used anywhere |
| No ephemeral doc cited from code | ✅ zero `.md` references in any changed `.cpp` / `.hpp` |
| Every `hw_config` reproduces the legacy resolved values | ✅ readers and writers use `create_reader_datamovement_config` / `create_writer_datamovement_config`, which reproduce the legacy `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` defaults exactly; compute uses `to_compute_hardware_config`, which reads the same four fields the legacy `ComputeConfigDescriptor` was given, with the `dst_full_sync_en` → `double_buffer_dest` inversion handled inside the helper. `bfp_pack_precision_mode` stays default, matching the legacy unset `bfp8_pack_precise`. `unpack_modes` reproduces each legacy `unpack_to_dest_mode` vector entry for entry. |
| Every `KernelSpec`'s `opt_level` matches its legacy kernel's | ✅ explicit `KernelBuildOptLevel::O3` on all five compute specs (legacy `ComputeConfigDescriptor` default); readers and writers left at the `O2` default (legacy DM default) |

The one deliberate exception is the surviving `CircularBuffer(dfb::eps)` shim at
`device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_post_allgather.cpp`, forced by an
out-of-scope donor signature — see *Handoff points* #4.
