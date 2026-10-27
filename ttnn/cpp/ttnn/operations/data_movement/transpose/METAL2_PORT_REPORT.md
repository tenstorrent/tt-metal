# Port Report — `data_movement/transpose` (HC sharded + WH sharded RM factories)

Port of `TransposeHCShardedProgramFactory` and `TransposeWHShardedRMProgramFactory`
(`ttnn/cpp/ttnn/operations/data_movement/transpose`) from the `ProgramDescriptor` API to Metal 2.0. The other six
factories of the op were already on Metal 2.0 (PR #54333) and are untouched. Plan: `METAL2_PORT_PLAN.md`; audit
inputs: `METAL2_PORT_BRIEF.md`, `METAL2_PREPORT_AUDIT.md`. `file:line` references to legacy code are against
`origin/main @ 0df0dde3316` (as in the brief); references to ported code are against this branch.

## Outcome

**`PORTED`** — both in-scope factories converted (`CustomProgramSpecFactoryConcept`), with the five kernels they
bind (four DM kernels converted in place; the shared compute fork extended with the sharded path). The confirmed
test set passes with the same counts as the untouched baseline, with the Metal 2.0 legality checks forced on, and
every captured output is bit-exact against the pre-port baseline (details under *Verification*). No factory of
this op remains on the descriptor API.

## Provenance

- **Recipe docs (this port):** `git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`
  prints nothing in this checkout — the recipe branch is an orphan history that `main` does not carry. The docs
  were read from a verbatim local export of **`akertesz/op-porting-recipe @ 4bd4bf42bfe`** (the same revision the
  audit used).
- **Audit docs (inherited):** `akertesz/op-porting-recipe @ 4bd4bf42bfe` (copied from `METAL2_PORT_BRIEF.md`).

## Verification

Bench: one Blackhole p150b card per run (this machine has two BH cards and no Wormhole — **Wormhole is not
verified by this port**). Every run used `TT_METAL_WATCHER=10`, a fresh per-process `TT_METAL_CACHE`, and the
worktree's own `_ttnn.so` (overlay venv). Test file:
`tests/tt_eager/python_api_testing/unit_testing/misc/test_transpose.py`, selection
`test_transpose_hw_sharded_rm*`, `test_transpose_wh_sharded_rm_override_addr_change`,
`test_transpose_hc_sharded_with_program_cache`, `test_transpose_hc_sharded_cache_hit_new_input`,
`test_transpose_hc_sharded_override_addr_change`, `test_transpose_sharded*` (incl. `_rm_width`), and the fork's
other consumer `test_transpose_hw_rm_with_padding` / `_no_padding` / `_with_program_cache`.

| run | pytest (selection) | capture script (22 shapes × 3 dispatches) |
|---|---|---|
| baseline (untouched tree, `2c413adb27e`) | **145 passed, 17 skipped, 0 failed** (skips: 16 parametrized `bfloat8_b`/HC-unsupported combos, 1 `skip_for_blackhole` #12349) | 19 shapes ok and bit-exact vs torch on all 3 dispatches; 3 shapes fail on the legacy tree (see Open items) |
| post-port (checks forced) | **145 passed, 17 skipped, 0 failed** (same 162 tests, per-test status identical to the baseline per the junit reports; markers: 180 + 180 lines) | 19 shapes ok and bit-exact vs torch on all 3 dispatches; the same 3 shapes fail |
| bit-exact comparison | — | **57 / 57** tensor comparisons (19 shapes × 3 dispatches), all identical (`torch.equal`, same shape and dtype); the 3 legacy-failing shapes fail identically (same exception class and, for the compile failure, the same TRISC2 `llk_pack_untilize_api.h` error; the `W % 32 != 0` direct-pack case's PCC varies from run to run even on the untouched tree — 0.27 and −0.002 across the two baseline runs — i.e. its output is uninitialized memory) |

Legality checks: all nine `skip_validation` sites under `tt_metal/impl/metal2_host_api/` were forced to `false`
with `METAL2_CHECKS_FORCED` markers in `BuildProgramFromSpec` and `SetProgramRunArgs`; both markers fired in every post-port run (capture run: 65 `program_spec.cpp:26` + 64 `program_run_args.cpp:602` lines; pytest run: 180 + 180), with zero spec / run-args validation failures. The
scaffolding is reverted; `git diff --stat` against `origin/main` touches only the op directory.

Factory routing (so the capture really exercised the two ported factories): confirmed per shape from the kernel
stems the runtime JIT-compiled into that case's `TT_METAL_CACHE` — `reader/writer_unary_transpose_hc_sharded_rm`
for 8 HC shapes (2 generic-path, 6 special-case incl. single- and multi-H-block), `reader/writer_unary_transpose_wh_sharded_rm`
+ `transpose_wh_rm` (`_metal2` post-port) for 11 WH RM shapes (`ht ∈ {2, 4}` and `ht ∈ {10, 16}` — the
`out_staging` / writer-transfer path that no existing test reaches — with `W % 32 != 0`, narrow-row, bf16, fp32
and int32 variants), and the interleaved `TransposeHCRM` factory for the 3 small HC shapes whose shards are not
native (as `select_program_factory` predicts). Cache hits at new addresses: every shape dispatches three times
with the previous dispatch's tensors kept alive, asserting the cache-entry count is constant and the input and
output addresses moved — the override's borrowed-buffer re-point is what makes dispatches 2-3 correct.

Anti-pattern self-audit (hits / denominator): CB-name sweep over the 9 ported files: **0 / 9**; over the whole op
directory: 11 / 47 files, all outside the port (the unported legacy kernels left behind by PR #54333, the six
sibling factories that bind shared forks whose vocabulary is `cb_in` / `cb_out`, and `transpose.cpp`) — none
introduced here. `CircularBuffer` / `CBDescriptor` / `CBIndex` / positional `get_compile_time_arg_val` /
`get_arg_val` / `get_arg_addr` in the ported files: 0 / 9. `.id` extraction, `TensorAccessorArgs`,
`allow_instance_multi_binding`: 0 / 9. Buffer addresses in run-args (`buffer()->address()`, `Buffer*`,
`emplace_runtime_args`): 0 / 2 factories. Positional `compile_time_args`: 0. `.md` citations in changed code files:
0 / 15. `opt_level`: the one compute `KernelSpec` (`transpose_wh_sharded_rm_program_factory.cpp`) carries an
explicit `O3`; the four DM specs are default `O2` = legacy. `hw_config`: DM specs via
`create_reader/writer_datamovement_config()` (legacy `Reader/WriterConfigDescriptor{}` defaults); compute
`ComputeHardwareConfig{.enable_32_bit_dest = (src0 == Float32)}` + `unpack_modes = {{IN, UnpackToDest}, {TILIZE,
UnpackToDest}}` under the same Float32 condition — the legacy `unpack_to_dest_mode[c_24] / [c_25] =
UnpackToDestFp32` entries; every other field at its (matching) default. TT_FATAL census: unchanged in every file
except `transpose_wh_sharded_rm_program_factory.cpp` (+1: the restated `total_size % page_size` invariant, see
Open items); the two legacy `TT_ASSERT`s per factory moved verbatim into `create_program_artifacts`. Conditional
binding: `OUT_STAGING` bound iff `ht > 8`, define emitted on the compute **and** writer specs, kernel-side alias
and uses `#ifdef`-gated. Retained varargs: 8 `get_vararg` sites, all indexed-collection reads (listed under Open
items). The fork's non-sharded path is textually untouched (the diff of `transpose_wh_rm_metal2.cpp` removes only
header-comment lines; `dfb::in0` / `dfb::tilize` / `dfb::out`, `args::Ht` / `Wt` / `HtWt`, the RTA
`num_hw_blocks_per_core` and `unary_op_init_common` are all still there), and `test_transpose_hw_rm_*` (its
existing consumer) passes.

## TTNN ProgramFactory

### Concept realized
`CustomProgramSpecFactoryConcept` on both factories, as the audit chose. Each `override_runtime_arguments` now
returns a `ProgramRunArgs` whose `tensor_args` carries a `TensorArgument` for **every** io-tensor
`TensorParameter` — `INPUT` and `OUTPUT`, the only two — and no `kernel_run_args`, mirroring the legacy
cb-addr-only overrides (`hc_sharded:468-473`, `wh_sharded_rm:259-264`) exactly: nothing skipped, nothing added.
No op-owned tensors. The signature follows the adapter's call
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:1034-1038`).

### Device-op-class edits
- Pybind entry points removed: none (`transpose_nanobind.cpp` binds only the public op).
- Custom `compute_program_hash`: none (default reflection hash) — nothing to leave intact.
- `transpose_device_operation.{hpp,cpp}` untouched (both factories already lived in `program_factory_t`).

### Open items
- None on the concept fit. No relaxation candidate: both factories bake `H`, `C`, `N`, `W`, the shard spec and the
  page geometry into CTAs and buffer sizes, so strict `TensorSpec` matching is the right declaration.

## Handoff points

- **Recipe deviation, sanctioned by the invoker — extending a fork that has a consumer.** The compute kernel
  `device/kernels/compute/transpose_wh_rm.cpp` is `#ifdef SHARDED`-split; its fork `transpose_wh_rm_metal2.cpp`
  (consumer: `TransposeWHProgramFactory`) carried only the non-sharded path. *Caution: Porting a shared kernel*
  makes a consumed fork read-only, forbids a fork of the fork, and closes the `#ifdef` escape — leaving no rung for
  the second variant of an `#ifdef`-split kernel. On the invoker's explicit decision the fork was extended behind the
  **same `SHARDED` define the legacy file already uses as its dialect boundary**, which cannot affect the existing
  consumer (it never defines `SHARDED`; its names and args are untouched and its tests pass). For the recipe
  owners: a sentence sanctioning this shape — *when the define is the legacy file's own dialect boundary, extending
  the fork behind that same define is permitted* — would close the gap the audit's Recipe note 2 raised.
- **Legacy-path defects surfaced by the capture (owner: op owners; not port issues — reproduced identically before
  and after the port on this BH card):**
  1. `(2, 16, 48, 64)` height-sharded on 8×4, bf16 (WH RM, `ht = 2`, `H % 32 = 16` → narrow-row pack path): output
     uncorrelated with the reference (PCC ≈ 0). Same class as `test_transpose_hw_sharded_rm`'s
     `skip_for_blackhole("Mismatching on BH, see #12349")`.
  2. `(2, 16, 64, 48)` on 8×4, bf16 (WH RM, `ht = 2`, `W % 32 = 16`, `H % 32 = 0`): PCC ≈ 0. On this path the
     compute kernel packs full tiles straight into the output shard (`transpose_with_pack_untilize`, `reserve_back(Ht)`),
     but the output buffer is paged by `output_page_size = (W % 32) * 32 * elem` (half a tile) — the reserved
     bytes are half what the packer writes. The `ht > 8` staging path (`(2, 16, 320, 48)`) is correct, so the
     defect is confined to `ht <= 8` with `W % 32 != 0`.
  3. `(2, 16, 304, 64)` on 8×4, bf16 (WH RM, `ht = 10`, narrow-row): the compute kernel fails to compile
     (TRISC2, inside `llk_pack_untilize_api.h`) — `transpose_with_pack_untilize_narrow_row` instantiates
     `pack_untilize_dest_init<Ht, Ht, …>` with `Ht = 10` dest tiles, more than the Dest register holds.
  None of these is reachable from the checked-in tests. They are recorded here so the op owners can decide; the
  port preserves them.
- No capitulation, no boundary-rule violation, no kernel-lib gap, no framework gap.

## Successes

- *Pattern: Conditional / optional resource bindings* fired exactly where the brief said it would: the compute
  kernel's file-scope ternary `(Ht > 8) ? c_27 : c_16` (`transpose_wh_rm.cpp:133-134`) and the writer's
  unconditional `DataflowBuffer dfb_src(c_27)` (`writer_unary_transpose_wh_sharded_rm.cpp:21`, `:27`) both
  needed the preprocessor gate; the `#ifdef OUT_STAGING` alias / construction compiled on both branches on the
  first attempt, and the validator (forced on) accepted the conditional binding set.
- *Pattern: Two-toucher DFB → assign 1P+1C* — the HC special case's reader + writer co-gather into the same two
  borrowed buffers reads like "two writers" at a glance; the census (both role-free peeks) gave 1P+1C and the
  multi-binding flag was never reached.
- *Caution: Avoid varargs* — the five HC scalars sit before the arrays and are read once each; they are named even
  though the legacy generic reader walked them with `arg_idx++` (`reader_unary_transpose_hc_sharded_rm.cpp:103-108`).
  Only the `[core]` / `[shard id]`-indexed arrays stayed varargs.
- *Compiler options* rule 2 — the compute spec would otherwise have silently dropped from O3 to O2; the one-command
  `grep -n opt_level` check made the absent line visible.
- *Hardware configuration* — Style B copy of `fp32_dest_acc_en` and the two `UnpackToDestFp32` entries; with the
  checks forced, the validator's required-entry rule for Float32 consumers was exercised by the fp32 capture shapes.
- *Go to the headers first* — `compute_hardware_config.hpp` settled the flat `ComputeHardwareConfig` shape (the
  recipe's `ComputeGen1Config` / `to_compute_hardware_config` names are stale), and `advanced_options.hpp:70-79`
  settled that the vararg count is one per `KernelSpec`.

## Friction

**Gaps**
- The recipe's `grep -n 'bool skip_validation' tt_metal/impl/metal2_host_api/*.cpp` misses
  `program_spec/program_spec.cpp` (one directory down, four of the nine sites — including the spec-side choke point
  `BuildProgramFromSpec`). `grep -rn … tt_metal/impl/metal2_host_api/` finds all nine.
- `ttnn.CONFIG` defaults `validate_program_args=false`, so TTNN passes `skip_validation = true` to every Metal 2.0
  entry point in production — the recipe's warning is exactly right and worth stating as a fact rather than a
  suspicion.
- A legacy CB sized by `total_size` + a non-tile `page_size` (WH RM's `out`, `wh_sharded_rm:103-113`) maps to
  `num_entries = total_size / page_size`, which is exact only because `CircularBufferConfig::set_page_size`
  rejected non-divisible totals (`circular_buffer_config.cpp:164`) — a check the DFB lowering does not run. The
  *DataflowBufferSpec* bullet could tell porters to restate that invariant when the quotient is not obviously exact.
- The vararg count is one per `KernelSpec` (`advanced_options.hpp:70-79`); for a split-reader op whose per-core
  count is computed per core, the recipe could say up front "take the maximum and zero-pad" (the audit's Recipe
  note 4 already asks for this).
- Factory routing cannot be confirmed from the logger in a Release build (`log_debug(tt::LogOp, …)` compiles out);
  a tip that the per-process `TT_METAL_CACHE` kernel directory names identify the kernels each case compiled would
  save the next porter the detour.
- `fuser` is absent on this bench; `/proc/*/fd` scanning stood in for it.

**Confusion**
- Fork vocabulary vs host spec names: the fork's role name `in0` is "the tilize input", which under `SHARDED` is
  `c_24` — so the compute kernel binds the host's `IN` spec under accessor `in0`, while the reader binds the host's
  `IN0` spec (the resident input shard, `c_0`) under its own `in0`. Both are correct and each is local to its
  kernel, but it takes a second read; the host comments at both binding sites spell it out.
- *Translating `override_runtime_arguments`* reads as if there is always something to inventory; for a cb-addr-only
  override the whole translation is the two `tensor_args` entries. The audit's Recipe note 3 asks for a sentence
  acknowledging that case; seconded.
- First capture attempt exhausted L1 on `(16, 128, 128, 16)` because the script kept three dispatches' sharded
  tensors alive at once — a script bug, fixed with a one-dispatch keep-alive window; noted so the next porter does
  not mistake it for a legacy OOM.

## Open items for downstream

- **Shared kernel touches.** `device/kernels/compute/transpose_wh_rm.cpp` (intra-op): rung taken — **extended the
  existing fork** `device/kernels/compute/transpose_wh_rm_metal2.cpp` with the `#ifdef SHARDED` path (no new file;
  the legacy original's pointer comment updated to say the fork carries both paths). Remaining unmigrated
  consumers of the legacy copy on `main`: **none** (`experimental/quasar/transpose/` binds its own duplicate, not
  this file). The legacy copy is at its sunset point: a follow-up can delete it and rename the fork, which was not
  done here (sunset is not the porter's job). The four DM kernels are bound only by this op and were converted in
  place.
- **Dropped plumbing.** `transpose_wh_sharded_rm_program_factory.cpp:143-153` — dead `c_26` ("untilize" CB, `ht >
  8` only, referenced by no kernel) not re-created; `:178-179` — reader CTAs 7-8 (duplicates of CTAs 2-3, never
  read) dropped.
- **Named-but-dead arguments, kept because the kernel reads them:** HC generic reader CTA `N` and RTA `curr_n`
  (`reader_unary_transpose_hc_sharded_rm.cpp`, incremented and never used); compute CTAs `pack_num_pages` and
  `pack_num_pages_last_row` (`transpose_wh_rm_metal2.cpp`, read and never used).
- **Retained varargs (8 sites, all indexed-collection reads):** HC special-case reader and writer — three
  `num_cores_read`-long arrays (`read_stick_offset`, `noc_coord_x`, `noc_coord_y`), `num_runtime_varargs = 3 *
  num_non_repeat_cores` (maximum over cores, zero-padded; uniform on every shape captured); HC generic reader —
  `shard_grid_x_map` (`num_cores_x`) + `shard_grid_y_map` (`num_cores_y`), indexed by a data-derived shard id.
- **DM self-loops (Gen1-legal; Quasar-uplift debt):** WH RM `in0` on the reader, `out` on the writer when `ht > 8`;
  HC generic `in0` and `out` on the reader. Compute self-loop: `tilize`. The WH RM reader / writer keep the
  hand-built loopback `UnicastEndpoint` from `my_x` / `my_y` (the migration guide's "Avoid" form; a transfer
  rewrite is out of port scope).
- **Guard added (the only `TT_FATAL` delta):** `transpose_wh_sharded_rm_program_factory.cpp` checks
  `stick_size_bytes * shard_height % output_page_size == 0` before deriving the `out` buffer's `num_entries`. It
  restates the legacy `CircularBufferConfig` divisibility check the DFB path no longer runs; without it a
  non-divisible shape would silently get a truncated buffer instead of the legacy throw.
- **Remote-shard reads assume one L1 address for the input buffer on every core** (`reader_unary_transpose_hc_sharded_rm.cpp`,
  `writer_unary_transpose_hc_sharded_rm.cpp`): holds because a `borrowed_from` buffer inherits the single sharded
  allocation's address; stated in the host comment.
- **Legacy factory anomalies carried unchanged** (for the op owners): `hc_sharded:105-122` `height` / `cores`
  computed and unused; `:238-243` a loop whose body only `break`s; `:236-237` TODO for multi-batch shards;
  `:336-337` `logical_shape()` vs the helpers' `padded_shape()`; `wh_sharded_rm:84` `num_cores` only logged; the
  writer's `dfb_dst(dfb::out)` construction + peek on the `ht <= 8` path where its transfer loop is compiled out.
- **Legacy-path defects** (1)-(3) under Handoff points — `ht <= 8` narrow-row and `W % 32 != 0` direct-pack give
  wrong output on BH; `ht > 8` narrow-row does not compile. Candidates for either a fix or a tighter
  `select_program_factory` guard, by the op owners.
- **Test coverage.** No checked-in test reaches `ht > 8` in WH sharded RM (the `out_staging` / writer-transfer
  path) or the fp32 WH RM path; the scratch capture script covered `(2, 8, 512, 16)` on 2×8, `(2, 16, 512, 16)` and
  `(2, 16, 320, 48)` on 8×4 (bf16) and `(2, 16, 512, 16)` fp32. Worth adding as parametrizations of
  `test_transpose_wh_sharded_rm_override_addr_change` in a follow-up (test edits were kept out of this diff).
  `test_transpose_hw_sharded_rm` is skipped on Blackhole (#12349), so the narrow-row path's only checked-in
  coverage is Wormhole-only.
- **Wormhole not verified** — this bench has only Blackhole cards. The port is architecture-agnostic host code
  plus kernel argument plumbing, but a WH run of the same selection is the obvious CI check for the PR.
- **Readiness sheet rows 94/95/97/98/99/100** still read `Concept = descriptor` for the six factories ported by
  PR #54333 (audit Gate detail); after this port rows 96 and 101 need the same update.
