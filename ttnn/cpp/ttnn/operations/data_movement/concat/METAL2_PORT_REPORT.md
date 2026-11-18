# Port Report — concat (`ConcatS2SMultiProgramFactory`)

## Outcome

**`PORTED`** — `ConcatS2SMultiProgramFactory` only, to `ProgramSpecFactoryConcept`. Before and after the port, the confirmed test set gave identical per-test results (953 passed, 86 skipped, 0 failed), with Watcher on and the Metal 2.0 legality checks forced on and proven live. 89 of those tests reach this factory. The other concat factories are unchanged; see the brief for their status.

## Provenance

- **Recipe docs (this port):** the recommended `git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` prints nothing: that path doesn't exist in this checkout. The recipe was read from an untracked copy at `/localdev/edwinlee/metal2_port.md`, so its version can't be pinned.
- **Audit docs (inherited):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` (from `METAL2_PREPORT_AUDIT.md`; the brief carries no provenance line of its own).
- **Port base:** `0de8c7f3aad` (origin/main, 2026-11-08), branch `edwinlee/port_concat`.

## TTNN ProgramFactory

- **Concept realized:** `ProgramSpecFactoryConcept`. `create_program_artifacts` returns `ProgramArtifacts{spec, run_params}`, matching the audit's choice. No `override_runtime_arguments`; the cache-hit refresh is the adapter's `UpdateTensorArgs`.
- **Custom hash:** `concat_device_operation.cpp:102-136`, left intact. On a cache hit, the strict `TensorParameter` match gave no `TensorSpec` legality failure: the 18 `test_sharded_concat` cases with `cache_mode=True` rerun with fresh buffers, and an ad-hoc aliasing check (below) hit the cache with aliased and distinct input sets.
- **Device-op class edits:** none. **Pybind entry points removed:** none (S2SMulti had no pybound `create_descriptor`).
- **Open items:** see below (common-array candidate; stale arg-model comment).

## Handoff points

- **Metal runtime / JIT build (not a port issue; reproduced on the legacy tree):** `ttnn.concat` of 64 or more interleaved DRAM inputs hangs indefinitely in the host JIT build. Repro: `tests/ttnn/unit_tests/operations/data_movement/test_concat_iterative.py::test_concat_small_tensors_2[tid=1-input_shapes=((64, 64), (64, 128))-num_inputs=64-tensor_layout=Layout.TILE-dim=-2]`. It hung for 21 minutes in the full run and for 15 minutes in isolation, both on the pre-port tree, Release build, WH n150, Watcher on. With that case deselected, `num_inputs=128, TILE, dim=-1` hung the same way. The main-thread stack is `ProgramImpl::compile → sync_build_steps → shared_future::get` under `MeshWorkloadFromSpecs` for `ConcatProgramFactory`. All 47 taskflow workers sit idle in `Executor::_wait_for_task`, no compiler child processes exist, and the device is idle (every core `GW`). So a build task's future is never completed. The repo's `pytest-timeout` (300 s, `signal` method) never fires, because the main thread is blocked in a C++ futex and never returns to the interpreter. Owner: Metal runtime (JIT build / taskflow). For this port, the iterative file ran with `num_inputs ∈ {2, 32}` only, before and after. Every case in that file is interleaved DRAM, so none of it reaches S2SMulti.

No port capitulation. All three of the brief's unproven seams held (see Successes).

## Successes

- **The brief's three unproven seams all worked on first contact.**
  - An LTA passed as the NoC source to `set_async_read_state` / `async_read_with_state` / `async_read` (`reader_s2s_tensor_concat.cpp:54-55, 66-70`) compiled. It was byte-correct across the 89 reaching tests, including `test_width_sharded_height_concat_page_over_noc_burst`, which takes the `page_size > NOC_MAX_BURST_SIZE` `async_read` branch.
  - `make_local_tensor_accessors<uint32_t>(tensor::inputs)` worked from a ttnn kernel with a runtime index.
  - Runtime array args reached the kernel through TTNN's `SetProgramRunArgs` path; the generated `kernel_args_generated.h` holds `RtaArrayArg<uint32_t, N>` for all five arrays.
- **Recipe, "Ensure the Metal 2.0 host-side legality checks are enabled":** the force-and-prove markers removed any doubt. Both `METAL2_CHECKS_FORCED` markers appeared about 1,530 times in the baseline and 1,578 times after the port, in every test file.
- **Recipe, "While a test run is in flight, kernel sources are frozen":** the port was kept as a patch and applied only between runs (baseline → reach check → port), so no run mixed kernel versions.
- **Recipe, "Anti-pattern self-audit" denominator note:** each sweep below reports hits over the files scanned.
  - CB-name sweep: 0 over the 3 port files. Op-wide it hits 153 times, all in unported factories and the device op.
  - Buffer address or `Buffer*` in the factory: 0. `TensorAccessorArgs`, `get_compile_time_arg_val`, `get_arg_val`, `get_vararg`, `my_x`/`my_y`/`UnicastEndpoint` in the kernel: 0. `allow_instance_multi_binding`: 0.
  - `.md` citations: 0 over 5 diffed source files.
  - `TT_FATAL`/`TT_ASSERT`/`TT_THROW` census across all 45 op files: unchanged. The factory's `<= 16` check keeps its limit, with the message reworded.
  - `opt_level`: no compute kernels; DM defaults to O2 on both sides.
  - `hw_config`: reader is RISCV_1 / NOC_0 / dedicated and writer is RISCV_0 / NOC_1 / dedicated, identical to `ReaderDataMovementConfig` / `WriterDataMovementConfig`.

## Friction

- **Gap: the recipe's force step misses the spec-side site.** The grep `tt_metal/impl/metal2_host_api/*.cpp` finds only `program_run_args.cpp`. `BuildProgramFromSpec` now lives in `tt_metal/impl/metal2_host_api/program_spec/program_spec.cpp`, which the glob doesn't reach, so it was forced by hand. The grep should recurse (`-r`).
- **Gap: the port skill's preconditions don't check that the base has what the brief needs.** The workspace started at `3d23c7655d7` (Oct 27), which predates runtime array args (#63805). The audit's 11-06 update said "port from current main", but nothing checks the base has the features the brief relies on. Rebased to `0de8c7f3aad` on a new branch.
- **Gap: the recipe's workspace setup doesn't mention the container split.** The host has no clang-20, so `./build_metal.sh` fails at configure outside the reservation container. Builds and tests ran inside it as the user's UID. The second session was already inside the container, so the handoff's `docker exec` wrapper didn't apply.
- **Gap: the "run tests in background, read the log" pattern has no answer for a hang.** One `pytest` session with a single `--junitxml` lost about 580 completed results when the iterative file hung, because JUnit XML is written only at session end. The runner was rebuilt as one session per test file, each with its own XML and `timeout -s KILL`. The recipe should suggest per-file runs and warn that the repo's `pytest-timeout` can't break a C++-blocked hang.
- **Gap: no gtests.** The recipe orders gtests first; concat has none, so the port was verified on pytests only and the build skipped `--build-tests`.
- **Confusion: the kernel-side whitelist conflicts with the brief.** The whitelist says transfers stay as they are; the brief overrides it for this factory (endpoints become LocalTensorAccessors). The recipe has no notion of a brief-authorised deviation from the whitelist. Recorded in the plan under Deferred / Flagged.
- **Confusion: no audit route for a waived factory.** The audit closed RED with no brief. The port ran on a user waiver, with the Nov 8 handoff serving as the brief. The skill's precondition accepts "GREEN (user waiver)", but the audit recipe has no way to issue a brief for a waived factory.
- **Confusion: the brief's reach list was partly wrong.** The reach check (a temporary `TT_THROW` in the legacy factory) found 89 reaching tests, but `test_concat_cache_aliasing.py`, `test_concat_program_cache.py` and nightly `test_concat.py::test_sharded_tiled_concat_int` reach none. What does reach: unit `test_concat.py` (`test_sharded_concat` ×18, `…_rank_relative_dim` ×28, `…_height_leading_dims_width_sharded` ×23, `…_rank3_width_multicore` ×4, `…_height_interleaves_leading_dims` ×2, `test_width_sharded_height_concat_page_over_noc_burst` ×2), and nightly `test_concat_memory_configs_and_layouts.py` (14 across `TestHeightShardedConcat`, `TestWidthShardedConcat`, `TestShardedToInterleavedConcat`). Run the throw rather than trusting a list.

## Open items for downstream

- **Stale arg-model comment:** `concat_device_operation.cpp:385-404` (`calculate_max_tensors_per_concat`) still describes S2SMulti's legacy layout: positional CTAs including `cb_dst_id` and `num_input_tensors`, `5N` RTAs, and the "CB 0..N-1, output at 16" ceiling. After this port, S2SMulti has 5 named CTAs, five runtime arrays of length N, and no CBs. The device op is off-limits for this port, so the comment goes stale.
- **The 16-input ceiling is now arbitrary for S2SMulti.** The factory's `TT_FATAL(num_input_tensors <= 16)` was kept with the same limit (message reworded so it no longer blames CB ids). No CB ids remain, so the real limit is the RTA budget (5N per kernel) and tensor-binding count. Raising it is a separate change, together with the batching caveat in the comment above.
- **Per-node runtime arrays hold identical values on every node.** They are candidates for common runtime arrays (`common_runtime_array_arg_values`): the reader and writer are separate KernelSpecs, so their different values can't collide. Kept per-node by Edwin's decision; the legacy comment blaming common args for "precision failures" (now above the `KernelRunArgs` construction in the factory) was never root-caused. The existing tests would catch a regression, as they did last time.
- **Test coverage, aliasing:** no suite test sends aliased inputs (`concat([x, y, x])`) through S2SMulti. An ad-hoc script outside the repo covered it, on 3-input height-sharded (width concat) and width-sharded (height concat) TILE bf16 over 2 cores. It ran distinct inputs as a miss then a hit with fresh buffers, `[0,1,0]` as a miss then a hit with fresh buffers, `[0,0,0]`, then distinct again. All 12 cases matched `torch.cat` exactly, and cache-entry counts showed the aliasing signature keying separate programs. Candidate: a sharded case in `test_concat_cache_aliasing.py`.
- **Test coverage, dead branches:** `test_concat_iterative.py::test_concat_mid_tensor_1` has `sharded_width` / `sharded_height` / `sharded_block` branches, but its only `input_mem_config` value is `interleaved_dram`, so they never run.
- **Shared kernel touches:** none. `reader_s2s_tensor_concat.cpp` is bound only by this factory and was modified in place.

---

# Port Report — concat (`ConcatBlockShardedProgramFactory`)

## Outcome

**`PORTED`** — `ConcatBlockShardedProgramFactory` only, to `ProgramSpecFactoryConcept`. Before and after the port, the same 9 concat test files gave identical per-test results (1,080 passed, 84 skipped, 4 xfailed, 1 failed), with Watcher on and the Metal 2.0 legality checks forced on and proven live. 78 of those tests reach this factory, all passing before and after. The one failure is unrelated and identical on both sides (below). Device kernel time is unchanged within noise.

## Provenance

- **Recipe docs:** the same untracked recipe copy as the S2SMulti port; not pinnable.
- **Brief:** `BlockShardedHandoff.md`, written against `0de8c7f` and PR #62362's Oct 30 revision; appended to `METAL2_PORT_BRIEF.md`.
- **Port base:** `e38dfa66343` (origin/main, 2026-11-18), which contains #62362 (`9490a5f3a2d`). Branch `edwinlee/port_concat`, rebased onto it from `0de8c7f3aad`.

## TTNN ProgramFactory

- **Concept realized:** `ProgramSpecFactoryConcept`; `create_program_artifacts` returns `ProgramArtifacts{spec, run_params}`. No `override_runtime_arguments`.
- **Custom hash:** `concat_device_operation.cpp:102-136`, left intact. No `TensorSpec` legality failure on cache hits across the run.
- **Device-op class edits:** none. **Pybind entry points removed:** none.

## Handoff points

- **Not a port issue, reproduced before the port:** `test_concat_codegen_routing.py`, the 49-input bfloat8_b case, fails with Watcher on because `reader_concat_interleaved_start_id` (the interleaved codegen route, untouched here) overflows NCRISC code space (0x4230 > 0x4000). It passes with Watcher off. Owner: concat codegen / interleaved reader.

## Successes

- **Every unproven seam in the brief held.** `with_input` combined with `ShardView::ResolvedShard` compiles and is byte-correct; runtime array args with per-RISC sizes and zero padding work, including a writer with zero copies on every core (array size 1).
- **Brief, "temporary development assert":** comparing each shard id's core (`BufferDistributionSpec::cores()[id]`) against the legacy coordinate table caught the brief's `shards_per_row` error on the first TILE test, before any data was wrong. It was then silent across all 78 reaching tests and removed before commit.
- **Reach check:** a temporary log marker (not a throw) in the factory, read back per test from the logs, gave the baseline and the reach set in one run. Reach: `test_concat_block_sharded.py` (63), `test_concat_memory_configs_and_layouts.py` (10), unit `test_concat.py` (5: `…_leading_dims_block_sharded` ×4, `…_height_interleaves_leading_dims` ×1). Covered: ROW_MAJOR and COL_MAJOR, width and height concat, RM and TILE, ragged widths and heights, leading dims. No case had to be added.
- **Binary size (brief asked):** NCRISC `.text` with Watcher on is 3,024 B legacy, 3,876 B ported at ≤ 4 inputs, and 10,724 B at the 16-input cap (checked with an ad-hoc 16-input RM/TILE × width/height run, all exact). Limit 16,384 B.
- **Performance (brief asked):** device kernel duration, median of 20 runs after 3 warm-ups, Watcher and DPRINT off, WH n150, 4 × 4 grid:

  | case | before (ns) | after (ns) | change |
  |---|---|---|---|
  | RM width, narrow 16 B rows | 918 | 882 | −3.9% |
  | RM height, narrow | 951 | 908 | −4.5% |
  | RM width, 4 inputs | 2,930 | 2,921 | −0.3% |
  | RM height, 4 inputs | 2,988 | 2,995 | +0.2% |
  | RM height, 8 leading indices | 2,502 | 2,310 | −7.6% |
  | RM width, COL_MAJOR | 948 | 914 | −3.5% |
  | RM height, COL_MAJOR | 921 | 894 | −3.0% |
  | TILE width | 729 | 738 | +1.2% |
  | TILE height | 720 | 740 | +2.9% |
  | TILE width, COL_MAJOR | 1,034 | 1,055 | +2.0% |
  | TILE height, 3 inputs, COL_MAJOR | 1,310 | 1,304 | −0.4% |

  Small RM copies (many rows per copy) are slightly faster; TILE cases (few rows per copy, so per-copy cost shows) move by ±3%, within run-to-run spread (minimums move in both directions).
- **Anti-pattern self-audit** (3 port files, 45 op source files): buffer address / `Buffer*` in factory 0; legacy kernel idioms (`get_arg_val`, `get_compile_time_arg_val`, `TensorAccessorArgs`, `DataflowBuffer`, `UnicastEndpoint`, `bank_base`) 0; CB names 0; positional CTAs 0; multi-binding 0; `.md` citations 0; `tt_metal/` files in diff 0. `TT_FATAL` census: factory 12 → 14, all 12 legacy checks kept; the two new ones are the brief's output core-order check and a "has a distribution spec" guard. `hw_config` reader/writer defaults as legacy; no `opt_level` either side (DM O2).

## Friction

- **Brief error: `shards_per_row` from the distribution spec.** `div_up(tensor_shape_in_pages()[-1], shard_shape_in_pages()[-1])` is right for RM (`[64, 2]` / `[32, 1]`) but wrong for TILE: the spec can merge dimensions, so a 2 × 2 grid of one-tile shards is `[4]` / `[1]` and shard (1, 0) became id 4 instead of 2. Shard ids still run along shard rows; only the row width has to come from the tensor's 2D physical width over the shard width. Anyone porting from coordinates to shard ids elsewhere should not read the grid shape off `tensor_shape_in_pages()`.
- **Recipe gap: the profiler refuses to start while DPRINT is enabled** (`Both DPRINT and Profiler cannot be enabled`). Perf runs need `TT_METAL_DPRINT_CORES` unset.
- **Recipe gap: `python -m tracy` with a test file outside the repo** segfaults in the `device` fixture (the repo `conftest.py` isn't found); the perf script had to sit under `tests/` temporarily.
- **Rebase cost:** #63424 moved the Metal 2.0 headers out of `experimental/`, which broke the already-ported S2SMulti factory's includes after the rebase. Fixed in a separate commit, mirroring what #63424 did to its siblings.

## Open items for downstream

- **16-input cap is now arbitrary here too.** No CB ids remain; the real limits are the RTA budget (31 copies per RISC) and kernel size (10.7 KB of 16 KB at 16 inputs, Watcher on). `calculate_max_tensors_per_concat` (`concat_device_operation.cpp:378-382`) still cites CB ids.
- **Per-node arrays are padded to the busiest core.** For uneven copy counts every core carries the longest list's RTA words. Harmless at today's sizes.
