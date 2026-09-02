# QUASAR_UPLIFT_REPORT — ttnn.rms_norm (shared layernorm device op)

Audit driven by `docs/source/ttnn/ttnn/ai/quasar_porting.md` + `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/audit/quasar_audit.md`.
Target: the op exercised by `models/experimental/llama32_1b_quasar/tests/graph_ops/test_rms_norm.py`.
Leave this file uncommitted; delete before merge.

## Status: GREEN — no changes needed

The op is already Metal 2.0 on Gen1, and the Quasar-uplift audit found **no statically-determinable,
clearly-required fix** on the code paths the test exercises. Per the recipe (§2: §7–§8 fixes are
reactive; "do not manufacture changes") the uplift is a no-op at this stage. **Zero source files
were changed** — only this report was added. One §6-class hazard sits directly on the tested path
(NoC-loopback self-copy in the two `rm_gb` kernels, below); it is a "can spin / can drop"
emulator hazard with a prescribed reactive fix, not a statically-rejected construct, so it is
recorded as the first item to apply if the Quasar run fires its symptom.

## Which op / which code path

`ttnn.rms_norm` is a thin composite in
`ttnn/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm.cpp` that routes every non-degenerate call
into the **shared layernorm device op** `ttnn::prim::layer_norm`
(`ttnn/cpp/ttnn/operations/normalization/layernorm/`) with
`LayerNormType::RMSNORM`, `DistributedLayerNormStage::NOT_DISTRIBUTED`. That shared op is what this
audit covers.

The test's 4 captured cases reach two program factories
(`select_program_factory`, `device/layernorm_device_operation.cpp:18` — sharded input → sharded
factory, else multi-core):

| Cases | Config | Factory | Kernels (all in `device/kernels/`) |
|---|---|---|---|
| 00, 01 | `LayerNormShardedMultiCoreProgramConfig`, width-sharded L1, `use_welford=0`, RM gamma | `LayerNormShardedProgramFactory` (`layernorm_op_multi_core_sharded.cpp` + `sharded_layernorm_factory_helpers.cpp`) | `dataflow/reader_mcast_sender_unary_sharded_ln.cpp`, `dataflow/reader_mcast_receiver_unary_sharded_ln.cpp`, `dataflow/writer_unary_sharded_ln_rm_gb.cpp` (RM gamma ⇒ `use_row_major_kernel`), `compute/layernorm_sharded.cpp` |
| 02, 03 | default program config, interleaved DRAM, RM gamma | `LayerNormMultiCoreProgramFactory` (`layernorm_op_multi_core.cpp`) | `dataflow/reader_unary_interleaved_ln_rm_gb.cpp`, `dataflow/writer_unary_interleaved_start_id_blocked.cpp`, `compute/layernorm.cpp` (RMSNORM + FUSE_GAMMA; RM gamma skips the large-tensor variant) |

All 4 cases pass gamma as ROW_MAJOR `[1,1,64,32]`, so both paths use the `rm_gb` gamma loaders.

## §1 gate: Metal 2.0 on Gen1 — PASS

- Both factories are `create_program_artifacts` → `ttnn::device_operation::ProgramArtifacts`
  (`layernorm_op_multi_core.cpp:167`, `layernorm_op_multi_core_sharded.cpp:32`) with
  `DFBSpecName`/`TensorParamName`/`SemaphoreSpecName` bindings.
- Every kernel (compute and DM) uses the device-2.0 APIs: `api/dataflow/dataflow_api.h`,
  `api/compute/*`, `experimental/kernel_args.h`, `Noc`, `DataflowBuffer`, `TensorAccessor`,
  `Semaphore<>`, `get_arg(args::…)`. A full-token sweep of `device/kernels/` found **zero** legacy
  tokens: no `cb_*` free functions, no positional `get_arg_val<>`, no `noc_async_*` free functions,
  no `get_local_cb_interface`/`fifo_page_size`, no `circular_buffer.h`.

## quasar_audit.md checks

**Check 1 — device-side CB/DFB redesign (self-loops):**
- All `bind_self_loop` sites in `layernorm_op_multi_core.cpp` (lines 821–877) and
  `sharded_layernorm_factory_helpers.cpp::bind_compute_resources` (lines 1086–1174) are on the
  **compute** kernel — a compute self-loop is legal on Gen2. No action.
- One **DM self-loop** exists: `sharded_layernorm_factory_helpers.cpp:991` self-loops `SCALER` on
  the **writer** — but only in the `is_post_all_gather` branch
  (`DistributedLayerNormStage::POST_ALL_GATHER`), which `ttnn.rms_norm` never sets
  (`NOT_DISTRIBUTED`). Off the tested path → **deferred item 2** below.
- Borrowed DFBs (input/output shard overlays) use `borrowed_from` with runtime-resolved capacity —
  the sanctioned form.

**Check 2 — non-zero-init semaphores: CLEAN.** The only `SemaphoreSpec` creation
(`sharded_layernorm_factory_helpers.cpp:1448`) sets only `unique_id` + `target_nodes`;
`SemaphoreAdvancedOptions::initial_value` stays at its default 0. Kernels use the public
`Semaphore<>::wait/wait_min/set/set_multicast` API — no raw L1 semaphore reads (`get_l1_addr`
absent).

## §7–§8 gotchas — applied vs. considered

**Applied: none.** No device run is possible in this session, so §7–§8 fixes were applied only
where statically determinable and clearly required — and none were.

Considered, with findings:

| Gotcha | Finding on this op |
|---|---|
| `disable_dfb_implicit_sync_*` (§7) | Not used anywhere. Clean. |
| `evil_set_read/write_ptr` ring rewind (§7) | Not used. Clean. |
| `fifo_page_size` / `get_local_cb_interface` (§5, §8.3) | Not used; kernels size via `get_entry_size()`/`get_tile_size()`. Clean. |
| `compute_kernel_hw_startup` once (§7) | Tested-path kernels are clean: `layernorm_sharded.cpp` has a single call (line 169); `layernorm.cpp`'s four calls at 133–139 are `#ifdef` **alternatives** (exactly one compiles). Mid-kernel re-inits exist only under `TILIZE_IN` (`layernorm.cpp:165–173`, RM **input** only — not this test) and in `layernorm_large_tensor*.cpp` / `layernorm_sharded_pre_allgather.cpp` (off-path), each already marked `TODO(#52395)` → deferred item 3. |
| uint16/uint32 device-format branches (§7) | None in the op. Formats are Float16_b/Float32/output dtype (`get_dfb_data_formats`). Clean. |
| Non-zero-init semaphores (§7) | Clean (above). |
| `unpack_modes` for FP32 DFB consumers (§4) | Handled in both factories (`layernorm_op_multi_core.cpp:896–922`, `sharded_layernorm_factory_helpers.cpp:1209–1224` via `set_compute_unpack_modes`). The test is bf16 anyway. |
| `data_format_metadata` valid (§4) | Every DFB spec gets a real format from `get_dfb_data_formats`; no `Invalid`. |
| NoC/multicast direction (§11) | Mcast rectangles are built top-left→bottom-right and swapped **only** when `reader_noc == NOC_1` (`sharded_layernorm_factory_helpers.cpp:1573`); `reader_noc` comes from `preferred_noc_for_dram_read(arch)` which returns `NOC_0` for **every** arch (`tt_metal/api/tt-metalium/kernel_types.hpp:140`) — the post-allgather override also forces `NOC_0`. So the mcast rectangle is always forward; no reverse-mcast to normalize. Degenerate-grid corner clamps don't arise in this factory (mcast spans the shard grid rows/columns as given). |
| `MEM_ZEROS_BASE`, `flush/invalidate_l2_cache_range` (§8.1/§8.3) | Not used. Clean. |
| `TTI_STALLWAIT`, hand-rolled `g_dfb_interface` pokes | Not used. Clean. |
| wait→pop / reserve→push TDMA hazard (§8.2/§8.5) | Reactive-only (hang symptom); not pre-applied. |
| Local NoC self-copy (§6) | **Present on the tested path** — see deferred item 1. Not applied pre-emptively: §6 describes it as an emulator hazard ("can spin on `can_post` or silently drop"), not a statically-rejected construct, and the replacement touches DFB implicit-sync-sensitive code that cannot be validated without a device run. |

## Deferred / follow-up items

1. **NoC-loopback self-copy in the RM gamma/beta loaders — first suspect if the Quasar run hangs or
   gamma is wrong.** Both tested paths load RM gamma by reading a row from DRAM and then issuing a
   NoC `async_read` **from the core's own coordinates** (`my_x[noc.get_noc_id()]`,
   `my_y[…]`) to move the second half-row into the second tile face:
   - `device/kernels/dataflow/writer_unary_sharded_ln_rm_gb.cpp:108–116` (gamma) and `:141–148` (beta)
   - `device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb.cpp:162–169` (gamma) and `:190–197` (beta)
   Per recipe §6 a same-core L1→L1 NoC loopback can spin on `can_post` or silently drop on the
   emulator; the prescribed fix is an `#ifdef ARCH_QUASAR`-guarded direct L1→L1 RISC copy (the
   source and destination addresses are both derived from `get_write_ptr()`, which returns
   uncached addresses on Quasar DM per `a00dd45`/#52769, so a plain RISC copy is safe). Apply
   reactively when the symptom fires — symptom: hang in the gamma/beta block before the first
   compute, or wrong gamma-scaled output on Quasar only.
2. **DM self-loop `SCALER` on the POST_ALL_GATHER writer**
   (`device/sharded_layernorm_factory_helpers.cpp:991`): a DM self-loop is rejected on Gen2 and
   needs an owner decision (drop the never-drained scaler generation on that path, or convert to a
   `Scratchpad`). Not reachable from `ttnn.rms_norm` (NOT_DISTRIBUTED); blocks only the distributed
   pre/post-allgather stages' own Quasar uplift.
3. **Mid-kernel `compute_kernel_hw_startup` re-inits (TODO #52395)** in
   `compute/layernorm.cpp` (TILIZE_IN branch), `compute/layernorm_large_tensor.cpp`,
   `compute/layernorm_large_tensor_welford.cpp`, `compute/layernorm_sharded_pre_allgather.cpp`:
   §7 says a mid-kernel re-init is worse on Quasar than WH/BH. These are shared WH/BH behavior
   (not guardable without functional redesign), already tracked upstream as #52395 — an op-owner /
   runtime follow-up, not an uplift edit. None are on this test's paths.
4. **Intra-tensix DFB tile-counter aliasing (§8.5)** — LayerNorm is an explicitly named candidate:
   its compute self-loop DFBs are intra-tensix DFBs (counter indices 16–31, aliasing via
   `index % 16` on non-remapped HW). The fix (tile-counter remapper) is **runtime-owned**; nothing
   to change at the op level. Recorded so a Quasar hang/`0x10000`-family symptom on this op is
   routed to the runtime team, not patched here.
5. **`writer_noc = NOC_1` + `DM_DEDICATED_NOC`** (`layernorm_op_multi_core_sharded.cpp:232`,
   helpers `:1244`): Quasar has no independent NOC1 (§11); how `NOC::NOC_1` maps on Gen2 is a
   runtime concern. No reverse-direction trick depends on it in this op (the mcast sender is the
   reader on NOC_0), so no op-level change; noted in case the runtime's NOC_1 mapping surfaces.

## Files changed

**None.** The only file added is this report (uncommitted, to be deleted before merge). The op's
directory and namespace are untouched.

## WH/BH parity claim (argued structurally — no device run in this session)

The working-tree diff for this op is **empty** (report aside). With zero source changes there is,
by construction, no WH/BH behavior, PCC, or perf change: both Gen1 archs compile and run exactly
the code that is on `main`. There are also no pre-existing `ARCH_QUASAR` guards in the op, so
Quasar and Gen1 currently share every line audited above.

## Test commands (user-run; recipe §9 — BH → WH → Quasar)

BH/WH parity (should be a no-op vs `main` since nothing changed):

```bash
pytest tests/ttnn/nightly/unit_tests/operations/fused/test_rmsnorm.py
pytest tests/ttnn/unit_tests/operations/fused/test_rms_norm_sharded.py
pytest tests/ttnn/nightly/unit_tests/operations/fused/test_layernorm.py
pytest tests/ttnn/nightly/unit_tests/operations/fused/test_layernorm_sharded.py
```

Quasar (emulator env; run with `TT_METAL_LLK_ASSERTS` both on and off, per §9):

```bash
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_rms_norm.py
```

If kernels are later edited, force JIT: `TT_METAL_FORCE_JIT_COMPILE=1` (and purge
`~/.cache/tt-metal-cache` between baseline and post-change runs).
