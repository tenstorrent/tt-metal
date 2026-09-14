# Metal 2.0 Port Report — `data_movement/sharded/interleaved_to_sharded`

## Outcome

**`PORTED`** — `InterleavedToShardedProgramFactory`, the op's only factory, is on
`ProgramSpecFactoryConcept`. All six reachable configs (`TILE·{plain,convert_df}·{dst-L1,dst-DRAM}`,
`RM·{dst-L1,dst-DRAM}`) are converted together; there is no remaining factory.

**Verification**, on a Wormhole card with `TT_METAL_WATCHER=10`:

| run | result |
|---|---|
| `./build_metal.sh --build-tests` | SUCCESS |
| `tests/ttnn/unit_tests/operations/data_movement/test_interleaved_to_sharded.py` | **91 passed, 16 skipped** (the skips are the file's own deterministic `bfloat8_b` × `ROW_MAJOR` guards, not a regression) |
| `tests/tt_eager/python_api_testing/unit_testing/misc/test_sharded.py` | **107 passed, 107 skipped, 8 xfailed** |

**The Metal 2.0 legality checks were forced on and proved live for both runs.** `skip_validation` was
pinned `false` at every site `grep -n 'bool skip_validation' tt_metal/impl/metal2_host_api/*.cpp`
named (9 sites across the two files), and both `METAL2_CHECKS_FORCED` markers appear in each test log
(`program_spec.cpp` and `program_run_args.cpp`) — so both translation units were genuinely rebuilt and
the spec validator and the cache-hit `UpdateTensorArgs` path both ran. That scaffolding is working-tree
only and has been reverted; `git diff` against the merge-base contains no `tt_metal/` file and no
`METAL2_CHECKS_FORCED` / `DO NOT COMMIT` string.

No C++ gtests exist for this op (see *Test coverage notes*), so the pytest layer carries the whole
local signal; the CI Sanity run on this branch is the broader check.

## Provenance

```
git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/
```

- **Recipe docs (this port):** *(prints nothing)* — the recipe docs are not on this branch. The port
  branch is based on plain `main`, which carries no `docs/…/metal_2.0/` directory, so the version
  cannot be pinned from the tree. Read out of the doc branch instead:
  `origin/akertesz/op-porting-recipe` @ `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base
  wall as a category, not as slice's current state`.
- **Audit docs (inherited):** same revision — the audit was run against the same recipe commit, as
  recorded in `METAL2_PORT_BRIEF.md`.

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept` — the concept the audit chose, realized as written. The ported-from
factory had no `override_runtime_arguments`, so there was nothing to translate and no reason to
revisit the choice. The framework refreshes tensor bindings on cache hits; the port adds no override.

### Device-op-class edits

- **Pybind entry points removed:** none. `interleaved_to_sharded_nanobind.cpp` binds only the two
  public `ttnn::interleaved_to_sharded` overloads — no `nb::class_` of the device op, no pybound
  `create_descriptor` — so the vanishing factory entry point had no exposed surface. The file is
  byte-identical.
- **Custom `compute_program_hash`:** **left intact**, untouched, at
  `device/interleaved_to_sharded_op.hpp:35` / `device/interleaved_to_sharded_op.cpp:144-162`. It keys
  on the whole input `TensorSpec` (plus the output's when pre-allocated), so it is at least as strict
  as the strict `TensorParameter` match the port introduces — the containment test passes under the
  declared `none` relaxation, and no `TensorSpec` legality failure appeared on cache hits.
- No other device-op-class file changed. `interleaved_to_sharded_op.cpp`, `…_op.hpp`,
  `…_op_types.hpp`, `interleaved_to_sharded.cpp/.hpp` and both nanobind files are byte-identical to
  the pre-port revision.

### Open items

- **Relaxation candidate (not applied):** the sheet's informational
  `Provisional relaxation finding (Edwin)` cell reads `fix merged, then match_padded_shape`, i.e. a
  `match_padded_shape` relaxation may be proposed for this op later. The gating
  `TensorParameter relaxation` column reads `none`, which is what the port declared. Not a port-time
  call; recorded for the relaxation roadmap.
- **First `TensorParameter` exposure for this op.** The op declared no `TensorParameter` before the
  port, so `ValidateTensorArgs` never ran on it; it now runs, comparing `tensor_layout()` (alignment
  included) exactly. #55495 pre-hardened the program hash for exactly this transition, and its
  rationale is recorded inline in the hash comment — the two changes are designed to meet here, and
  they did.

## Handoff points

- **`Diego validation` cell flipped to `no` on the readiness sheet** (it read `yes` on 2026-09-11;
  it reads `no` as of the 2026-09-14 fetch). It is **not** a column the audit reads or gates on — it
  appears nowhere in the recipe's *TTNN factory concept prerequisite* — so the port did not act on
  it. Flagging it for the readiness-sheet owner in case the flip was meant to signal something about
  this row, since it moved in the same window as the `Custom hash` reconciliation this port was
  waiting on.
- No boundary-rule assumption violations: no out-of-op call site required a `sem::` or `tensor::`
  handle. No kernel-lib gaps. No framework gaps. No removed pybind surface.

## Successes

- **[Caution: Porting a shared kernel] → *Check fit before committing to reuse* fired correctly, and
  it was the one place this port could have shipped a silent wrong answer.** Two `_metal2` forks of
  `eltwise_copy.cpp` exist in the tree —
  `data_movement/sharded/device/kernels/compute/eltwise_copy_metal2.cpp` and
  `ttnn/cpp/ttnn/kernel/compute/eltwise_copy_metal2.cpp` — and they are not interchangeable: the
  first reads `per_core_tile_cnt` as a **runtime** named arg (`:22`), the second as a **`constexpr`**
  compile-time one (`:23`). i2s emits that count **per core** (it differs on the end core), so only
  the sibling fits, and the locational rule ("the fork that counts is the sibling of the copy your
  factory actually binds") picks it. A stem-wide grep would have found `sharded_to_interleaved`
  already binding the *other* one and made it look like the family's established choice.
- **[CB→DFB API whitelist §A] — `constexpr` metadata keeps the free-function form.** The tile reader's
  `constexpr uint32_t tile_bytes = get_tile_size(cb_id_in0)` feeds a non-type template argument
  (`get_barrier_read_threshold<tile_bytes, num_readers>()`), so demoting it to the member getter would
  not have compiled — a loud failure, but the whitelist's rule ("the legacy declaration is the entire
  test") got it right the first time. The two writer sites, declared `const`, took the member getter.
  Three sites, one file apart, splitting two ways on the declaration alone.
- **[CB endpoints] — the recipe's insistence on re-deriving the census rather than transcribing it
  paid off on `c_1`.** The audit's `(0 touchers under TILE, 1 under RM)` reproduced exactly, which is
  what licensed dropping the spec on the tile path and self-looping it on the row-major one.
- **[Compiler options] — the `opt_level` check is genuinely an *absent line*.** `grep -n opt_level`
  over the ported-from factory printed nothing, which under the recipe's rule means the compute
  kernel resolved to `O3` and needs it stated explicitly. Nothing in the new code would have looked
  wrong without it.

## Friction

### Gaps

- **The conditional-binding pattern has a cheaper form the catalog doesn't name.**
  [Pattern: Conditional / optional resource bindings] is written entirely around one kernel source
  whose *uses* of a sometimes-absent token must be `#ifdef`-gated, and it says the gate "has to happen
  at the preprocessor level." That is true when one source serves both paths. Here it does not: the
  layout axis already selects between two *different* reader sources, and only the row-major one names
  `dfb::scratch`, so the conditional host binding needs no define and no `#ifdef` at all — the token is
  absent exactly where nothing references it. A porter reading the pattern top-to-bottom would add a
  `defines` entry and a `#ifdef` block that do nothing. Worth one sentence: *"if the condition also
  selects the kernel source, the gate is already structural — bind conditionally and stop."*
- **`get_entry_size()` needed a unit check the whitelist waves at.** §B maps
  `get_local_cb_interface(...).fifo_page_size` to `get_entry_size()`, with a note that "TRISC size
  getters return sizes in **bytes**, not 16B units." The kernel in question is a *data-movement*
  kernel, and the legacy value feeds raw L1 address arithmetic
  (`scratch_l1_base + slot * page_size`), so a unit change there would be silently wrong addresses,
  not a compile error. Resolving it meant reading `dataflow_buffer.inl` and
  `circular_buffer_interface.h` to establish that `cb_addr_shift == 0` on BRISC/NCRISC, making
  `address_units_to_bytes()` the identity and the swap value-identical. The note would be more useful
  stated as the general fact ("`get_entry_size()` is bytes on every RISC; on DM the raw field already
  was, so the swap is value-identical there") than as a TRISC-only aside.

### Confusion

- **"Dead CB → build no spec" and "conditional DFB" read as two dispositions for one buffer.** The
  audit's census gives `c_1` as *dead under `TILE`, live under `RM`*, and the recipe's construct step
  lists a dead CB (drop it, record `file:line`) and a conditional binding (gate it) as separate
  bullets. They are the same action here — build the spec on one path and not the other — but it took
  a pass over both to be sure that "drop" did not mean "drop unconditionally." The audit's own
  wording (**"Do not drop it: it is live under `RM`"**) is what disambiguated it, which suggests the
  construct step could borrow that phrasing.

## Open items for downstream

### Shared kernel touches

The op owns no kernels; all six come from the in-family pool
`ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/`.

| kernel | rung taken | fork path | pointer comment in original | remaining unmigrated consumers |
|---|---|---|---|---|
| `dataflow/writer_unary_sharded.cpp` | **1 — reused existing fork** | `dataflow/writer_unary_sharded_metal2.cpp` (no new file) | n/a — rung 1 does not touch the original | `interleaved_to_sharded_partial`, `tilize_multi_core_sharded*`, `untilize_*_nd_shard_type_*`, `experimental/padded_slice`, `experimental/transformer/nlp_kv_cache_load_slice` — tracked in #52228 |
| `compute/eltwise_copy.cpp` | **1 — reused existing fork** | `compute/eltwise_copy_metal2.cpp` (no new file) | n/a | `interleaved_to_sharded_partial` |
| `dataflow/reader_unary_sharded_blocks_interleaved_start_id.cpp` | **2 — created the fork** | `…_metal2.cpp` (new) | ✅ added | `interleaved_to_sharded_partial`; tt-metal DM microbenchmark `tests/tt_metal/tt_metal/data_movement/interleaved_to_sharded_hardcoded/test_interleaved_to_sharded_hardcoded.cpp` |
| `dataflow/reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | **2 — created the fork** | `…_metal2.cpp` (new) | ✅ added | same two |
| `dataflow/writer_unary_sharded_blocks_start_id.cpp` | **2 — created the fork** | `…_metal2.cpp` (new) | ✅ added | same two |
| `dataflow/writer_unary_sharded_stick_layout_start_id.cpp` | **2 — created the fork** | `…_metal2.cpp` (new) | ✅ added | same two |

**The four new forks' binding vocabulary, which every later consumer inherits:**

| fork | DFBs | tensors | named args |
|---|---|---|---|
| `reader_unary_sharded_blocks_interleaved_start_id_metal2` | `dfb::in` (PRODUCER) | `tensor::src` | CTA `num_readers`; RTAs `block_height_tiles`, `block_width_tiles`, `padded_offset_bytes`, `input_width_offset_tiles`, `block_num_tiles`, `start_id_offset`, `start_id_base` |
| `reader_unary_stick_layout_sharded_blocks_interleaved_start_id_metal2` | `dfb::in` (PRODUCER), `dfb::scratch` (**PRODUCER + CONSUMER — self-loop**) | `tensor::src` | CTA `num_trids`; RTAs `block_height`, `block_width_bytes`, `padded_block_width_bytes`, `aligned`, `aligned_input_width_offset_bytes`, `aligned_block_width_bytes`, `aligned_offset`, `start_id` |
| `writer_unary_sharded_blocks_start_id_metal2` | `dfb::out` (CONSUMER) | `tensor::dst` | RTAs `block_height_tiles`, `block_width_tiles`, `padded_offset`, `block_width_padded_num_tiles`, `output_width_tiles`, `start_id_offset`, `start_id_base` |
| `writer_unary_sharded_stick_layout_start_id_metal2` | `dfb::out` (CONSUMER) | `tensor::dst` | RTAs `block_height`, `block_width_bytes`, `padded_block_width_bytes`, `start_id`, `output_width_in_pages` |

A factory binding the row-major reader **must declare both endpoints for `dfb::scratch`** — the kernel
fills and drains it itself, so a single-role binding fails the validator's ≥1-producer-and-≥1-consumer
rule.

**Sunset blocker:** `interleaved_to_sharded_partial` binds all six of these kernels and is blocked on
its own gate (`Is able to port? = no`, `TensorParameter relaxation = (legality - pending analysis)`,
`Custom hash = yes`, `Override runtime args method? = yes`). Until it ports, none of the four legacy
originals can be retired — and even then the tt-metal DM microbenchmark binds two of them directly, so
it keeps them alive past the op ports. The next porter to reach any of these four should land on
rung 1.

### Findings — behavior preserved, not fixed

Each of these is shipped forward unchanged, per the porting invariant. They are the op owner's to act
on.

1. **The public `keep_l1_aligned` argument is inert.** It is a documented Python kwarg on both
   overloads (`interleaved_to_sharded_nanobind.cpp:79`, `:94`, default `False`) and is plumbed into
   `InterleavedToShardedParams` (`device/interleaved_to_sharded_op_types.hpp:15`), but the factory
   hardcodes `bool keep_l1_aligned = true;` with the attribute read commented out
   (`device/interleaved_to_sharded_program_factory.cpp:37-38` post-port) and never consults the
   attribute. **A caller passing `keep_l1_aligned=False` silently gets the aligned behaviour.** It is
   also deliberately excluded from the program hash
   (`device/interleaved_to_sharded_op.cpp:147-148`), which is the right call *given* that the factory
   ignores it — the two would have to be fixed together.
2. **Dead runtime arg in the row-major reader.** The ported-from factory pushed ten reader args
   (`…_program_factory.cpp:387-398` pre-port) but the kernel read indices 0 and 2-9 only: arg **1**
   (`num_units_per_row`) was never read. Named args have no slot for a value nothing reads, so the
   port stops emitting it. Zero functional change; noted because it is a *removal* a reviewer will see
   in the diff. `sharded_to_interleaved` made the same call for its own dead arg 1.
3. **The alignment scratch buffer was allocated in tile configs that never touch it.** Pre-port,
   `…_program_factory.cpp:179`'s condition is layout-independent and its last disjunct is the
   hardcoded `keep_l1_aligned`, so **every** tile-layout program allocated `c_1` at
   `num_trids * align(input_unit_size + dram_alignment, dram_alignment)` — for a bf16 tile on
   Blackhole, 4 × 2112 ≈ **8.4 KB of L1 per core, burned for nothing**. The port builds no
   `DataflowBufferSpec` for it on the tile path (a dead CB has no behavior, and a bindingless DFB is
   rejected by the validator), so that L1 comes back. This is the one place the port's L1 footprint
   differs from the ported-from op's, and it differs by *not wasting* memory no kernel could reach.
4. **`starting_idx_h` is structurally always zero.** `num_slices = 1` / `slice_index = 0` are
   hardcoded for backward compatibility (`…_program_factory.cpp:28-29` post-port, issue #32752), and
   `calculate_starting_idx_h` returns `0` whenever `num_slices <= 1` (`sharded_common.cpp:17-19`). So
   the tile reader's `start_id_base` and the tile DRAM writer's `start_id_base` are constant zero, and
   the kernel-side `start_id_base + start_id_offset` additions are dead arithmetic. Preserved exactly.
5. **A standing TODO in the `aligned` computation.** For an L1 source on non-Blackhole/Quasar the
   factory sets `aligned = true` unconditionally while the Blackhole/Quasar path checks `curr_idx_w`
   and `padded_offset_bytes` against the L1 alignment; the code flags the asymmetry as unverified
   (`// TODO: is this right, leaving non BH case the same for now, should investigate`). Carried
   verbatim.
6. **Misnamed variable, self-flagged.** In the row-major branch `num_units_per_shard_width_last` holds
   a byte size, not a page count, with its own TODO. Carried verbatim.
7. **A comment in the device-op class is now stale — deliberately left stale.**
   `device/interleaved_to_sharded_op.cpp:156` reads *"The factory reads its shard spec, core ranges
   and CB sizes off the output"*. Post-port the factory reads **DFB** sizes; there are no CBs left in
   this op. It is a one-word fix, and it sits in the device-operation class, which the port's
   host-side scope discipline puts off-limits — so it is reported rather than changed. (It is also
   the single remaining hit of the port's `cb`-leftover sweep over the op directory, 2 hits / 9 files
   scanned before this port's own comment was reworded, 1 / 9 after. Naming it here is what keeps
   that residue from reading as an oversight.) Worth folding into whatever PR next touches that file.

### Incidental cleanups the port made (not scope creep — forced by the API change)

- `#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"` dropped. It was already unused before
  the port (no `shard_builder` / addrgen-helper symbol is referenced;
  `get_optimal_worker_cores_for_sharded_tensor` comes from `ttnn/tensor/tensor_utils.hpp`), but it sat
  beside `<tt-metalium/program_descriptors.hpp>` and `<tt-metalium/tensor_accessor_args.hpp>`, both of
  which the port genuinely removes. Left in, it would have been the only legacy-API-era include still
  standing. Flagging it because it is the one dropped line in the diff that is *not* strictly forced.
- The anonymous-namespace helper `push_i2s_cb_pair` is gone — it built `CBDescriptor`s, and the CB
  transition is total.

### Test coverage notes

- **The op's discovered test set**, presented here rather than confirmed with the invoker ahead of the
  run (the port ran end-to-end under a single instruction; the CI Sanity run is the broader check and
  its result is reported with this branch):
  - `tests/ttnn/unit_tests/operations/data_movement/test_interleaved_to_sharded.py` — the primary
    pytest, and the no-regression baseline.
  - `tests/tt_eager/python_api_testing/unit_testing/misc/test_sharded.py` — broad sharded-family
    coverage that exercises i2s heavily via `ttnn.interleaved_to_sharded` and the sharded matmul /
    binary paths.
  - `tests/sweep_framework/sweeps/data_movement/interleaved_to_sharded/interleaved_to_sharded_e2e.py`
    and `tests/sweep_framework/sweeps/model_traced/interleaved_to_sharded_model_traced.py` — sweeps,
    not run locally.
  - **No C++ gtests exist for this op** — `grep -l 'interleaved_to_sharded\|InterleavedToSharded'` over
    `tests/ttnn/**/*.cpp` returns nothing, so the gtest layer of the recipe's two-layer verification
    has no content here and the pytests carry the whole local signal.
- **All six configs have unit coverage** — checked rather than assumed, because the spec shape differs
  per config and a config with no test is a silent false-GREEN:

  | config | covering test |
  |---|---|
  | `TILE·plain·dst-L1` | `test_interleaved_to_sharded_hash` (the `first_dtype` call); `test_interleaved_to_sharded_nd_with_equivalent_2d` |
  | `TILE·convert_df·dst-L1` | `test_interleaved_to_sharded_hash` (the `second_dtype` call — six dtype pairs, all conversions) |
  | `TILE·plain·dst-DRAM` | `test_interleaved_to_dram_height_sharded`, `test_interleaved_to_dram_width_sharded`, `…_via_to_memory_layout` |
  | `TILE·convert_df·dst-DRAM` | `test_interleaved_to_dram_sharded_convert_dtype` |
  | `RM·dst-L1` | `test_interleaved_to_sharded_nd_with_equivalent_2d` (ROW_MAJOR param); `test_sharded.py` |
  | `RM·dst-DRAM` | the height/width-sharded and `via_to_memory_layout` tests at `ROW_MAJOR_LAYOUT` |

- **`test_interleaved_to_sharded_hash` is the single most valuable test for this port**, and worth
  naming so the next porter of a sibling op knows to look for its equivalent. It calls the op **five
  times in a loop** specifically to run it hot off the program cache, alternating two dtypes over a
  block-sharded L1 output. That is exactly the shape that surfaces a custom-hash-vs-`TensorSpec`
  legality failure — which, per the recipe, appears only on the *second and later* dispatch and is
  the one failure mode this op's custom hash could plausibly have produced. It also parametrizes
  `keep_l1_aligned` both ways, so the inert-argument behavior (finding 1) is pinned in both
  directions.
