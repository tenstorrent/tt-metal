# Metal 2.0 Port Report — nlp_create_qkv_heads

Port of `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads` (both factories) from the
`ProgramDescriptor` host API to Metal 2.0, 2026-09-14, on the `main` worktree at `f8c78f9bdf6`, Wormhole
n150. Three commits on `vsuresh/metal2-port-nlp-create-qkv-heads`:

1. `83142201337` — **pre-port, ops-side**: pass bare shard bases to the Sharded kernel, add offsets on
   device (the offset split the audit's `Sharded` GREEN was contingent on).
2. `a655b543214` — **pre-port, ops-side, found during this port's verification**: stop the sharded kernel
   indexing one row past its NoC coordinate table (two silent out-of-bounds reads; see *Handoff points*).
3. the port itself, with the four `METAL2_*.md` artifacts.

Commits 1 and 2 are each verified against the **legacy** host code before the port was applied on top, so
the port diff stays a pure Metal 2.0 translation and either fix can be evaluated (or dropped) on its own.

## Outcome

**`PORTED`** — both factories (`Interleaved`, `Sharded`) converted; the confirmed test set passes with the
Metal 2.0 legality checks forced on (`METAL2_CHECKS_FORCED` from both `program_spec.cpp` and
`program_run_args.cpp` in every log) and Watcher enabled (`TT_METAL_WATCHER=10`, no tripped asserts):

| test file | result |
|---|---|
| `tests/ttnn/unit_tests/operations/experimental/transformer/test_nlp_create_qkv_heads_program_cache.py` | 4 / 4 passed (interleaved address-change-on-hit for both `transpose_k_heads` values, shape change, sharded address-change-on-hit) |
| `tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads.py` | 147 / 147 passed (falcon7b / llama / generic interleaved incl. FLOAT32 × `transpose_k_heads`, `kv_tied` interleaved and sharded, program-cache loops, sharded sweeps over 32/16/2/1 KV heads, bf8 / bf16 / fp32) |

Pre-port baseline (same tree, same day, recorded in the brief): 4 / 4 and 147 / 147. The
`tests/sweep_framework/sweeps/model_traced/nlp_create_qkv_heads_model_traced.py` sweep was not run (needs
the sweep infrastructure); no C++ gtest references this op.

**Verification history, for the record.** The first post-port run tripped Watcher on
`test_sharded_nlp_create_qkv_heads_test[32-1-64-32-32-False-BFLOAT8_B]` (104 tests had passed before it):
`BRISC accessed unique runtime arg index out of bounds` on the last core. Root cause: the legacy kernel's
past-the-end coordinate reads (commit 2). The port's kernel was re-derived from the fixed legacy kernel and
the full set re-run: 151 / 151.

## Provenance

- **Recipe docs (this port):** `git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` prints **nothing** — the recipe set is an untracked overlay on the `main` worktree, restored from `origin/akertesz/op-porting-recipe` tip `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`.
- **Audit docs (inherited):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` (carried from `METAL2_PORT_BRIEF.md`).

## TTNN ProgramFactory

### Concept realized
`CustomProgramSpecFactoryConcept` on both factories, as the audit chose. Each `override_runtime_arguments`
now returns a `ProgramRunArgs` (no `Program&` parameter) whose `tensor_args` names **every** io-tensor
`TensorParameter` the factory declares — `input_q`, `input_kv` (when the separate KV tensor is passed),
`q`, `k`, `v` — and nothing else, mirroring the legacy overrides exactly:
- `Interleaved` legacy re-applied reader slots 0/1 and writer slots 0/1/2 → the five tensor bindings.
- `Sharded` legacy re-applied slots 6/14 on both instances and re-pointed the three borrowed CBs via
  `UpdateDynamicCircularBufferAddress` → the same five tensor arguments (the borrowed `q_out`/`k_out`/`v_out`
  DFBs re-resolve their L1 base from `q`/`k`/`v`).
No `kernel_run_args` on either override: no legacy override refreshed a non-address runtime arg.

### Device-op-class edits
- Pybind entry points removed: **none** (`nlp_create_qkv_heads_nanobind.cpp` binds only the public op).
- Custom `compute_program_hash`: **none** (nothing to leave intact).
- The only header change is the two factory structs' method declarations in
  `device/nlp_create_qkv_heads_device_operation.hpp` (plus the include swap the new signatures need).
  `device/nlp_create_qkv_heads_device_operation.cpp`, `nlp_create_qkv_heads.cpp` and the nanobind file are
  byte-identical to `main`.

### Open items
- **Relaxation candidates:** none observed; both factories key every DFB size and every RTA off the
  `TensorSpec`s, so strict matching is the faithful choice.
- **Concept fit:** clean. Both overrides are pure `tensor_args` echoes (the "address-only override" shape
  the audit's recipe note describes); no friction with the entry-point wiring.

## Handoff points

No capitulation, no out-of-op call site needed a `sem::` / `tensor::` handle, no kernel-lib or LLK change
was needed, no pybind surface was removed. The one write outside the op directory is the sanctioned
shared-kernel fork (*Open items → Shared kernel touches*). One entry, because it is a decision the invoker
should be able to reverse:

- **Pre-port kernel fix made by the porter, not the op owner (commit `a655b543214`).** *Op / factory:*
  `nlp_create_qkv_heads` / `Sharded`; file
  `device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp`. *What:* two reads one
  row past the end of the NoC y-coordinate table held in the runtime-arg area — (a) the source-core advance
  after the **last** head on the last core of the last source row wraps `q_y`/`kv_y` to `num_cores_y` and
  refreshes the coordinates from `in0_mcast_noc_y[q_y]` (legacy `:60-61`, `:113`); (b) the writer-config
  instance on a core whose Q output holds one head reads **zero** Q heads, but the host's per-core builder
  hands it the coordinates the next instance would start from (`q_y = 32 / 8 = 4` on the last core), and
  the pre-loop lookup read them anyway (legacy `:37-38`). Both values are dead (no next iteration / no
  iteration), so both reads were silent: the legacy kernel walked the table through a raw
  `tt_l1_ptr uint32_t*` derived from `get_arg_addr(18 + num_x)`, and only the *base* index is sanitized.
  *Why the port hit it:* the tables became runtime varargs, and `get_vararg(num_x + y)` goes through
  `get_arg_addr`, whose Watcher `ASSERT(arg_idx < rta_count, DebugAssertRtaOutOfBounds)` fires on the
  out-of-range index — a faithful syntax swap turned a silent UB read into a device stop, on every sharded
  config with `num_kv_heads == num_q_heads` (`[32-1-64-32-32-*]`). *Resolution:* guards
  `q + 1 < num_q_heads` / `kv + 1 < num_kv_heads` on the advance and `if (num_q_heads > 0)` on the pre-loop
  lookup (now `:64`, `:122`, `:42` of the ported kernel) — semantic no-ops — committed **separately**, verified
  against the legacy host code (28 sharded misc tests + 1 sharded program-cache test, Watcher on) before
  the port, mirroring how the audit session handled the offset fold at the user's request. *If the op
  owners prefer to own this fix:* drop commit 2; `Sharded` then reverts to legacy and the port ships
  `Interleaved` only (the two factories share no kernel, so nothing else moves). The recipe's rule is
  "don't fix the legacy kernel in the port"; this is deliberately *beside* the port, but it is still a
  porter-made change to legacy behaviour and is flagged as such. **The same kernel copy in
  `nlp_create_qkv_heads_boltz` has both reads and the offset fold.**

## Successes

- **[Two-toucher DFB → assign 1P+1C](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)**
  read the Sharded `q_out` case exactly: both instances raw-write `get_write_ptr() + q_offset`
  (`reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp:49`), so the reader-config instance is
  PRODUCER, the writer-config instance CONSUMER (`nlp_create_qkv_heads_program_factory.cpp`, `make_instance`),
  no multi-binding flag. The validator accepted it first time.
- **[Same-FIFO aliasing, path-dependent variant](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names)**
  — the "one object per FIFO, never two objects on one accessor" rule shaped the K handle in the
  Interleaved reader/writer: a second `DataflowBuffer` under `!TRANSPOSE_K_HEADS` would have aliased the
  `qv` object, so the K handle is a reference to it there
  (`reader_tm_tile_layout_nlp_create_qkv_heads.cpp:33-37`, writer `:35-39`). Legacy built two
  `CircularBuffer` objects on index 1; the port keeps one object and the same FIFO.
- **[Anti-pattern: Demoting per-group CTA to RTA](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta)**
  fired where it should: the landed `data_movement/transpose` fork takes `NHtWt` as an RTA, and binding it
  would have been the one-line "reuse" — the catalog entry (and the brief) kept `NHtWt` a CTA in the new
  fork with two compute `KernelSpec`s (`compute_g1` / `compute_g2`) over disjoint core groups.
- **[Compiler options](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compiler-options)**
  — the legacy compute descriptor set no `opt_level`, so it resolved to O3; the section's rule 2 put the
  explicit `KernelBuildOptLevel::O3` on the compute spec (`nlp_create_qkv_heads_program_factory.cpp:298`,
  inside the `make_compute` helper both specs route through). Nothing else would have flagged its absence.
- **[Hardware configuration → Compute kernels, Style B](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compute-kernels)**
  — the legacy `ComputeConfigDescriptor{.fp32_dest_acc_en = …}` maps to `ComputeGen1Config{.enable_32_bit_dest
  = …}` with every other field at its (matching) default, and the "newly-required explicit `unpack_modes`
  entry" item predicted the Float32 case exactly: the legacy vector was empty (Default → UnpackToSrc), so
  `unpack_modes = {{k_in, UnpackToSrc}}` is emitted only when `enable_32_bit_dest` is on. The FLOAT32 ×
  `transpose_k_heads` tests pass with the validator on.
- **[Ensure the legality checks are enabled](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#ensure-the-metal-20-host-side-legality-checks-are-enabled)**
  — the grep found nine `skip_validation` sites (one more than the recipe's own list: `MergeKernelRunArgsInto`
  in `program_run_args.cpp`), which is exactly why the step is a grep. Both markers appear in every log.
- **Run every test with Watcher on** — this is the instruction that found the two out-of-bounds reads. Without
  it the port would have passed 151 / 151 reading a stale L1 word, and shipped the first Metal 2.0 kernel in
  this op with a latent `DebugAssertRtaOutOfBounds` waiting for the next person to debug a model.

## Friction

### Gaps

- **`get_arg_addr`-based array walks become bounds-checked when they turn into varargs — the recipe and
  audit should say so.** The audit judged the sharded kernel's `tt_l1_ptr uint32_t* p =
  (tt_l1_ptr uint32_t*)get_arg_addr(N)` idiom "argument plumbing, dissolves into varargs" (its Recipe-notes
  entry asks for a sentence settling it). It does dissolve — but the legacy idiom sanitizes only the base
  index, while `get_vararg(i)` sanitizes every element read. A kernel that indexes such a table past its
  end (as this one did, harmlessly) passes every legacy test and stops the device after the port. Suggested
  addition to whitelist rule 4 / the varargs caution: *"a `get_arg_addr`-derived array walk becomes a
  per-element bounds-checked read; before porting, check that every index the kernel can form is within the
  table — an out-of-range legacy read is a pre-port op-owner fix, not a port change."* The audit's Device 2.0
  section could add the same check to its raw-arg-area-pointer verdict.
- **Per-node-conditional borrowed DFB → RTA-gate promotion has no catalog entry.** The Sharded K/V output
  CBs were declared over `k_cores` while both kernel instances run over `q_cores`, gated in the kernel by a
  per-core **RTA** (`read_kv_heads`). The catalog's
  [Conditional / optional resource bindings](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-resource-bindings)
  covers promoting a **CTA** gate to a define; whitelist rule 6 says "the condition … moves from a CTA to
  a kernel-side `#define`". Here the gate is an RTA whose value is a host constant per core
  (`i < k_cores.num_cores()`), so the resolution is: split the node set into two `WorkUnitSpec`s
  (`kv_cores` = `k_cores`, `q_only_cores` = `q_cores.subtract(k_cores)`, factory `:701`), two same-source
  `KernelSpec`s per instance, and a define (`READ_KV_HEADS`, factory `:742`) emitted only on the `kv_cores`
  specs; the RTA is dropped and the kernel block becomes `#ifdef READ_KV_HEADS`
  (`reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp:78`). Judged inside rule 6's intent, but the
  brief was right to flag it as a whitelist question — a catalog entry ("per-node-conditional resource →
  split WorkUnits, promote the node-selecting RTA to a define, legitimate only when the RTA is
  host-deterministic per node") would make the next port not have to re-derive it.
- **The Case 2 bridge can introduce `TensorAccessor` to a kernel that never had one.** The sharded kernel
  constructed no `TensorAccessor`; `TensorAccessor(tensor::input_q).get_bank_base_address()` (`:23`) adds
  one. The whitelist says "types the kernel keeps using unchanged … you neither add nor touch their
  includes" — true here only because `api/dataflow/dataflow_api.h` already includes
  `api/tensor/tensor_accessor.h`. A sentence covering "the Case 2 bridge may introduce `TensorAccessor`;
  no new include is needed on the DM path" would remove a moment of doubt.

### Confusion

- **Brief vocabulary vs self-audit on the fork's accessor names.** The brief asked for `dfb::cb_in` /
  `dfb::cb_out` in the new `transpose_wh_metal2.cpp` fork, to match the landed `data_movement/transpose`
  fork of the same body. The recipe's self-audit forbids any `cb` in a DFB name the port introduces and
  names `dfb::cb_in0` as the costly miss. The recipe outranks the brief, so the fork uses `dfb::in` /
  `dfb::out` (the kernel's own role words minus the prefix). The two forks of one compute body now carry
  different vocabularies; whoever consolidates them (see *Open items*) should pick the `cb`-free one. The
  audit doc's "adopt the landed fork's vocabulary" heads-up could add "unless it contains `cb`".
- **`ttnn::Tensor` copies and `TensorArgument` identity.** The legacy Interleaved factory copied the
  optional KV input into a `std::optional<const Tensor>` local (`:91`). The TTNN doc says a `TensorArgument`
  must reference the tensor the framework enumerates, so the port reads the optional through the
  `tensor_args` reference instead. A one-liner in the recipe's *Extracting the tensor* — "don't take
  `.mesh_tensor()` off a local `Tensor` copy" — would name the trap; the legacy idiom is common.
- **`Table` construction for the compute `unpack_modes`.** `ComputeUnpackModes` is a `Table`, so it is
  built with `emplace` inside the `if (fp32_dest_acc_en)`; the recipe's example shows the brace-init form
  only. Not a blocker; the recipe's "Tables are maps" paragraph covers it.

### Environment (not doc friction, recorded for the next porter on this box)

- The recipe's `./build_metal.sh --build-tests` was not used: on this machine a backgrounded ninja is
  killed for memory and the default ccache fills the home quota, so each build was
  `ninja -C build_Release ttnn` in a foreground chunk (one chunk sufficed each time), then
  `cmake --install build_Release` plus a copy of `_ttnn.so` into `ttnn/ttnn/`. Symbols
  (`create_program_artifacts` present, `create_descriptor` absent) and the forced-check marker were
  verified in the installed libraries before every test run; the op's JIT kernel cache was purged between
  runs.
- The session's tool policy blocked a scripted edit of `tt_metal/impl/metal2_host_api/*.cpp`; the nine
  `skip_validation = false;` inserts and two `METAL2_CHECKS_FORCED` markers were applied one line at a
  time with the file-edit tool instead. None of it is committed.

## Open items for downstream

### Shared kernel touches
- **`ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp`** (borrowed, shared pool) — **rung 2: created the fork**
  `ttnn/cpp/ttnn/kernel/compute/transpose_wh_metal2.cpp` (bindings `dfb::in` CONSUMER / `dfb::out` PRODUCER,
  named CTA `NHtWt`); the pointer comment landed at the top of the legacy original. **Remaining unmigrated
  consumers** of the legacy file (the sunset checklist): `nlp_create_qkv_heads_boltz`
  (`nlp_create_qkv_heads_boltz_program_factory.cpp:169,179`), `nlp_create_qkv_heads_vit`
  (`nlp_create_qkv_heads_vit_program_factory.cpp:103,111`), `split_query_key_value_and_split_heads`
  (`split_query_key_value_and_split_heads_program_factory.cpp:127`).
- **Two Metal 2.0 forks of one compute body now exist**: this one (CTA `NHtWt`, `dfb::in`/`dfb::out`) and
  `ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_metal2.cpp` (RTA
  `NHtWt`, `dfb::cb_in`/`dfb::cb_out`). Consolidating them is a planners' call, not this port's.
- The op's own three dataflow kernels were converted in place (no other op binds them).

### Per-op carry-over
- **`nlp_create_qkv_heads_boltz`** carries its own copy of the sharded kernel and per-core builder with
  the **pre-fix** Type 1 offset fold **and both past-the-end coordinate reads**; commits 1 and 2 of this
  branch are the template, then the same two-WorkUnit port applies. Its audit should start from that shape.
- **`nlp_create_qkv_heads_vit`, `split_query_key_value_and_split_heads`** bind `transpose_wh.cpp` and can
  now reuse the fork (rung 1) — adopt `dfb::in` / `dfb::out` and the CTA `NHtWt`.

### Findings routed to the op owners (behaviour preserved, not changed)
- **Sharded kernel, Q loop, source-core advance without a row wrap does not refresh the source
  coordinates.** In the K/V loop the coordinate refresh sits *outside* the `if (kv_x == num_x)` (every
  core advance refreshes); in the Q loop it sits *inside* the row-wrap branch, so a Q walk that moves to
  the next core in the **same row** keeps reading the previous core's coordinates at
  `q_src_addr += head_size` (legacy `:55-65`; ported `:64-74`). No test configuration has an output core
  whose Q heads span two input cores, so it is unexercised; if that shape is ever reachable it reads the
  wrong shard. Preserved as-is; flagging for the owners to confirm intent.
- `device/nlp_create_qkv_heads_program_factory.cpp` — the Sharded factory delivers several values that are
  identical on every core (`head_size`, `num_q_heads_per_core`, `num_kv_heads_per_core`, `num_kv_tiles`,
  `num_x`, and the whole NoC coordinate table) as **per-node** runtime args / varargs. They are CRTA
  candidates; not converted (RTA→CRTA changes dispatch semantics, out of port scope).
- Same file — the Sharded per-core builder computes start coordinates for the writer-config instance even
  when that instance reads zero Q heads (the second past-the-end read's source). Harmless after commit 2;
  the builder could skip them, an owner cleanup.
- Same file — the Interleaved reader receives `in1_tensor_tile_id` on every core even without a KV tensor
  (always 0 then; the kernel reads it and never uses it). Dead per-node arg in that config, preserved.
- Same file — the Sharded factory bakes **physical** worker NoC coordinates
  (`worker_core_from_logical_core`) into the vararg block. Correct today; worth knowing if
  virtual-coordinate work touches this op.
- `device/nlp_create_qkv_heads_device_operation.cpp:245,260` — sharded output specs use
  `PageConfig(input_tensor.layout())`, interleaved hardcode `PageConfig(Layout::TILE)`; equivalent under
  the `Layout::TILE` validation, just inconsistent. Off-limits file, untouched.
- The comment banner "Grayskull Device Setup" in the Interleaved factory is stale (kept verbatim).

### Quasar-uplift debt introduced by this port (Gen1-legal shapes)
- Two **DM self-loops**: Sharded `k_out` (reader-config instance) and `v_out` (writer-config instance),
  bound PRODUCER + CONSUMER on one kernel. Rejected on Gen2; the uplift refactors them into a real
  producer→consumer topology.
- The Sharded `q_out` 1P+1C labels are cosmetic (both instances write); on Gen2 a CONSUMER cannot write.
- The retained **runtime varargs** (NoC coordinate tables) in the sharded kernel — the one vararg use in
  this port; genuinely data-indexed (`noc_x[q_x]` with `q_x` advanced by the walk).
- `TensorAccessor(tensor::input_q).get_bank_base_address()` (Case 2 bridge) in the sharded kernel.
- No `constexpr` tile-size token-form sites: both Interleaved tile-size reads were legacy `const`, so they
  use the member getter.

### Doc-evolution suggestions
- Whitelist rule 4 / varargs caution / audit Device 2.0 section: the bounds-check consequence of turning a
  `get_arg_addr`-derived array walk into varargs (see *Friction → Gaps*, first entry).
- Catalog entry for the per-node-conditional resource shape (see *Friction → Gaps*).
- Recipe self-audit item 12 (`opt_level`): when compute specs are built through one helper, the grep shows
  one line for N specs; the item could say "one line per construction site, and confirm every spec routes
  through it" — which is what was done here (`make_compute` builds both).

### Test coverage notes
- The sharded program-cache test uses 16 Q / 8 KV heads only; the `num_kv_heads == num_q_heads`
  (single work unit) shape gets no cache-hit coverage. Both shapes pass their cache-miss tests
  (`[32-1-64-32-32-*]`, fused and separate KV, all three dtypes).
- No test configuration has an output core whose Q heads span two input cores in one row (the
  unrefreshed-coordinate finding above); worth a case if that shape is meant to be supported.
