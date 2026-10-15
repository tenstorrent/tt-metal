# Metal 2.0 Port Report — `transformer/gated_delta_attn`

## Outcome

**PORTED.** The op's single program (`GatedDeltaAttnSeqDeviceOperation`) now runs on `ProgramSpecFactoryConcept`. It is bit-exact against the legacy op on 10/10 harness runs, and the op's device test passes 8/8 with the legality checks forced on.

History: the first attempt (2026-10-08, commit `[gated_delta_attn] Metal 2.0 port capitulated`) stopped on the state buffer `S`. The reader seeded it and compute recycled into it, and `ValidateProgramSpec` rejected every faithful binding. The op-owner fix, PR #60011 (`7c7fafc9758`, merged 2026-10-13), moved the seed into a reader-only `S0` (`c_7`). On 2026-10-15 the branch was rebased onto `origin/main` to pick that PR up, and the earlier conversion was replayed on top with `S0` / `S` re-bound. No DFB needs `allow_instance_multi_binding` any more.

## Provenance

- **Recipe docs (this port):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`. Hash from the `Port_Recipe` checkout (`/localdev/edwinlee/metal2_port.md` symlinks there; the Metal_Ports checkout has no copy of the recipe docs).
- **Audit docs (inherited):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Audit gate waiver:** the TTNN-factory-concept gate was RED only because the readiness-sheet row is stale after PD batch #57409. The row shows `legacy device-op`, `Custom hash = yes`, and the deleted `GatedDeltaAttnSeqProgramFactory`. The user waived it on 2026-10-07 ("Waive"). The sheet refresh is still owed to the readiness-sheet owner.
- **No re-audit after #60011.** The audit (2026-10-07) predates #60011, which changed `device/` and the kernels. The user explicitly asked for the port to proceed on the post-#60011 code. The PR's only effects are the `S0`/`S` split and +64 KiB L1 per core. The port re-derived the census for both buffers (plan, *Applied Patterns*), and the brief carries a note marking its `c_8` multi-binding item superseded.

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`, as the audit chose. There is no override; the framework's tensor-binding refresh is the whole cache-hit path. In the harness, every second run of a config is a cache hit with fresh tensors at shifted addresses, and all were bit-exact.

### Device-op-class edits
- **Exception 3 (direct-descriptor op):** `create_descriptor` moved into a nested `GatedDeltaAttnSeqProgramFactory::create_program_artifacts`, with `using program_factory_t = std::variant<GatedDeltaAttnSeqProgramFactory>` (`device/gated_delta_attn_device_operation.hpp`). The header comment about the cache-hit refresh moved with it.
- Pybind entry points removed: none (the only binding is `bind_function` on the user function).
- Custom `compute_program_hash`: none (default hash; #57409 removed the custom one).

### Open items
- No relaxation candidates.

## Handoff points

1. **Framework gap, still open for other ops: the validator rejects "seed then recycle" DFBs.** *(Owner: Metal 2.0 host API team.)* This op no longer needs the change, because #60011 split the buffer at the op side, at a cost of +64 KiB L1 per core. Any other op where one kernel self-loops a buffer that a second kernel also FIFO-produces into will hit the same wall. `ValidateProgramSpec` requires a self-looped DFB's producer set to equal its consumer set even under `allow_instance_multi_binding` (now `tt_metal/impl/metal2_host_api/program_spec/`, moved there upstream since the first attempt). The proposed one-guard relaxation, with evidence, stays in `METAL2_HOST_FIX_SELF_LOOP_MULTIBINDING.md`. `METAL2_OP_FIX_STATE_SEED.md` was removed in this commit because #60011 implemented it; it remains in the history of the capitulation commit.

2. **Pre-port: the legacy op cannot run under full Watcher on WH 8×8.** *(Owner: op owner / Watcher.)* With `TT_METAL_WATCHER=10`, the **legacy** op fails with `TT_FATAL: Program size (73424) too large for kernel config buffer (70656) on TENSIX, 2768 B over` (`tt_metal/impl/program/program.cpp:3760`). The kernels plus Watcher instrumentation overflow the kernel config buffer. Verification therefore used a reduced Watcher: `TT_METAL_WATCHER_DISABLE_{PAUSE,RING_BUFFER,WAYPOINT,STACK_USAGE}=1`, keeping NoC sanitize, asserts and CB sanitize on. Legacy and port both pass 8/8 under it.

## Successes

- **[Two-toucher → 1P+1C](port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split) and [self-loop](port_patterns.md#pattern-self-loop-dfb-binding) after #60011.** Re-deriving the census gives `S0` reader PRODUCER → compute CONSUMER, and `S` as a compute-only PRODUCER + CONSUMER. Both are ordinary bindings, and the validator accepted them on the first build (`device/gated_delta_attn_program_factory.cpp:104-105,135-187`).
- **[Conditional / optional resource bindings](port_patterns.md#pattern-conditional--optional-resource-bindings)** fit the optional `initial_state` exactly. The tensor is bound and `HAS_INITIAL_STATE` is emitted only when present. The reader `#ifdef`s the accessor and the read branch, and the zero-fill stays as the `#else`. Both presence states ran bit-exact, including at NC>1.
- **[Hardware configuration](metal2_port.md#hardware-configuration) "decide on the resolved values"** caught a silent trap the audit missed. The legacy DM configs are `{RISCV_1, NOC::RISCV_1_default}` (reader) and `{RISCV_0, NOC::RISCV_0_default}` (writer): reader on **NOC_1**, writer on **NOC_0**, the reverse of the defaults. Reaching for `create_reader_datamovement_config()` would have swapped both NOCs and passed every test. They were ported verbatim as custom Gen1 configs.
- **The `unpack_modes` required-entry rule** (FP32 DFB consumed with `enable_32_bit_dest`): every consumed DFB, now including `S0`, gets an explicit `UnpackToSrc` (legacy `Default`). It is derived from the bindings, so the new buffer needed no extra code.
- **Whitelist §A `constexpr` rule:** `constexpr uint32_t f32_tile = get_tile_size(dfb_S0)` / `get_tile_size(dfb_out)` kept the token form and compiles.
- **Runtime DFB selection through the `uint32_t` conversion:** `const uint32_t dfb_state = (c == 0) ? dfb_S0 : dfb_S;` (compute `:236`) and `dst_dfb = is_last_chunk ? dfb_final_state : dfb_S` (`:426`) are the legacy runtime CB-id selects, carried over unchanged.

## Friction

### Gaps
- **Recipe hw_config type names are stale.** The recipe says `DataMovementGen1Config{…}` / `ComputeGen1Config` / `std::get<ComputeGen1Config>(…)`. The headers have `DataMovementHardwareConfig{.config_1xx = …DataMovement1XXConfig{…}}` and a flat `ComputeHardwareConfig{…}`, and `create_reader_datamovement_config()` no longer takes an arch. The header was the authority, as the recipe advises.
- **No pattern covers "seed then recycle" state loops.** The patterns catalog's multi-binding entry should say Metal 2.0 rejects them today, and that the op-side fix is a separate seed buffer (#60011 is a worked example). **Suggested audit gate:** a CB that one kernel FIFO-produces *and* consumes while another kernel also produces into it is not portable until the validator relaxes or the op splits the buffer.
- **Mandatory Watcher vs kernel-config-buffer overflow.** The recipe has no guidance for an op whose *legacy* build already overflows under Watcher. The reduced feature set above worked; the recipe could name it as the fallback.
- **Pre-port gate vs a user-sanctioned upstream change.** The port skill stops when `device/` changed after the audit. Here the change *was* the requested unblock, and the user asked to proceed. A "re-derive only the affected endpoints, mark the brief item superseded" path would make this explicit.

### Confusion
- **Kernel-side temporaries.** The compute kernel builds `CircularBuffer(x)` temporaries throughout (~60 sites). The port kept them as `DataflowBuffer(dfb_x)` temporaries, the mechanical swap. Upstream #60306 (merged during this port) notes that `~DataflowBuffer()` drains on Quasar, so a temporary destroyed while tiles are held can spin forever, and it moved `eltwise_chain` callers to kernel-scope objects. This op is WH/BH-only today, but the pattern is Quasar debt; see Open items.

## Open items for downstream

- **Quasar debt: per-call `DataflowBuffer` temporaries in the compute kernel** (`device/kernels/compute/gated_delta_attn.cpp`, every `DataflowBuffer(dfb_x).wait_front/pop_front/reserve_back/push_back`, plus the helpers `matmul_reconfig_and_init`, `fwd_sub_row`, `fwd_sub_4rows`). The kernel is legal and identical on WH/BH, and the port preserved the legacy shape. A Quasar uplift should hoist them to kernel-scope objects, per #60306's rationale.
- **Test coverage.** `tests/ttnn/unit_tests/operations/transformers/test_chunk_gated_delta_rule.py::test_gated_delta_attn_seq_vs_golden` (added by #60011) is now the op's real device test: NC 1–4, with and without `initial_state`, a cache-hit second run, and golden PCC ≥ 0.999. It compares against the golden, not bit-exact against legacy, and runs only 3 heads. The scratch harness (bit-exact, up to 64 heads) was job scratch and is not committed.
- **Carried-over anomalies** (from the audit; unchanged, not port work): the factory ignores `attrs.compute_kernel_config` but hashes it, `output_mem_config` is unvalidated, and `NC == 0` hangs.
- **L1:** #60011 raised L1 use by 64 KiB per core; `test_gated_deltanet_chunked_ttnn` already overflows L1 on main (#60025, now Blackhole-only per #60328). The port allocates exactly the legacy buffers, so it adds nothing.
- **Shared kernel touches:** none. All three kernels are op-owned.

## Verification

Run on the branch rebased onto `origin/main` (`a0b789d0469`, 2026-10-15), Release build, single WH b0 8×8.

- **Build:** `./build_metal.sh --build-tests`, success (full legacy build, then incremental port build).
- **Legality checks:** all 9 `skip_validation` sites forced (uncommitted scaffolding). Markers present: `METAL2_CHECKS_FORCED (program_spec.cpp:27)` ×5 and `(program_run_args.cpp:675)` ×5 per harness run; 16 in each pytest log.
- **Legacy baseline (rebased tree, before the port):**
  - Harness saved 10 results. They are **bit-identical to the 2026-10-08 pre-#60011 baseline**, confirming #60011's no-change claim on this tree.
  - `test_gated_delta_attn_seq_vs_golden`: 8/8 pass; 8/8 under reduced Watcher.
- **Port:**
  - Harness: **10/10 bit-exact** vs the legacy baseline. Configs (BH, NC, initial_state) = (12,1,no), (12,3,yes), (5,2,no), (64,2,yes), (7,4,yes), each run twice, the second a program-cache hit at shifted addresses. Golden PCC 0.99990–0.99999.
  - `test_gated_delta_attn_seq_vs_golden`: 8/8 pass; 8/8 under reduced Watcher (no asserts, no NoC sanitizer trips).
- **Test set not confirmed with the invoker.** The broad sweep found the device test above, the golden-only tests in the same file, `models/experimental/gated_attention_gated_deltanet/tests/test_ttnn_validation.py` (Blackhole-only since #60328), and Qwen3.6 BH demos. The harness plus the device test were the baseline.
- **Self-audit** (11 `.cpp`/`.hpp` files scanned):
  - zero `cb` names, `CircularBuffer` / `CBDescriptor`, address RTAs, `Buffer*`, `TensorAccessorArgs`, varargs, `.id`, `allow_instance_multi_binding`, or `.md` citations in code;
  - TT_FATAL counts unchanged (`device_operation.cpp` 10, factory 1);
  - O3 set explicitly on the single compute spec (factory `:314`);
  - DM configs verbatim custom (reader RISCV_1/NOC_1, writer RISCV_0/NOC_0); compute HiFi2 / Precise / 32-bit Dest / `double_buffer_dest` default true;
  - no `tt_metal/` files in the diff.
