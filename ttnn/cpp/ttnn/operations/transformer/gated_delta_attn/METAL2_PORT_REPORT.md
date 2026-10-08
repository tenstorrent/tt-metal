# Metal 2.0 Port Report — `transformer/gated_delta_attn`

## Outcome

**CAPITULATED.** The single program (`GatedDeltaAttnSeqDeviceOperation`, direct `create_descriptor`) was fully converted. The Metal 2.0 validator then rejected the state buffer `S`'s binding shape: a compute self-loop plus a reader seed, under `allow_instance_multi_binding`. No legal binding describes the kernels honestly, so the port is held.

- **The conversion is not lost.** It lives on branch `worktree-metal2-gated-delta-attn`, WIP commit `e9c863c3085` (factory, device-op header, three kernels). It is not on `edwinlee/PD_Metal_Ports`; the op there stays on the legacy `ProgramDescriptor` path.
- **Two independent unblock paths are written up. Either one suffices:**
  - Host: `METAL2_HOST_FIX_SELF_LOOP_MULTIBINDING.md`, a one-guard validator relaxation owned by the Metal 2.0 team. The user decided the port does not touch `program_spec.cpp`.
  - Op: `METAL2_OP_FIX_STATE_SEED.md`, a pre-port op-owner PR that seeds the state through a new `c_7` buffer.
- **Everything except `S` is verified.** With `S` force-legalized by a local experiment (never committed), the converted op was **bit-exact vs the legacy op on 10/10 runs**, with the legality checks live. Once either fix lands, re-porting is just reapplying `e9c863c3085` and adjusting `S`'s bindings.

## Provenance

- **Recipe docs (this port):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`. The hash is from the `Port_Recipe` checkout: `/localdev/edwinlee/metal2_port.md` symlinks there, and the Metal_Ports checkout has no copy of the recipe docs.
- **Audit docs (inherited):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Audit gate waiver:** the TTNN-factory-concept gate was RED only because the readiness-sheet row is stale after PD batch #57409. The row shows `legacy device-op`, `Custom hash = yes`, and the deleted `GatedDeltaAttnSeqProgramFactory`. The user waived it on 2026-10-07 ("Waive"). The sheet refresh is still owed to the readiness-sheet owner.

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`, as the audit chose. It was realized on the WIP branch but not landed, because of the capitulation. There is no override, and the framework's tensor-binding refresh is the whole cache-hit path: in the harness, every second run at shifted addresses was a cache hit and bit-exact.

### Device-op-class edits
- **Exception 3 (direct-descriptor op):** on the WIP branch, `create_descriptor` moved into a nested `GatedDeltaAttnSeqProgramFactory::create_program_artifacts` with `using program_factory_t = std::variant<GatedDeltaAttnSeqProgramFactory>` (`device/gated_delta_attn_device_operation.hpp`). The header comment about the cache-hit refresh moved with it. Not landed.
- Pybind entry points removed: none (only `bind_function` on the user function).
- Custom `compute_program_hash`: none (default hash; #57409 removed the custom one).

### Open items
- No relaxation candidates.

## Handoff points

1. **Port capitulation: validator rejects the seed-then-recycle state loop.** *(Framework gap; owner: Metal 2.0 host API team, or the op owner via the op fix.)*
   - **Op / factory:** `ttnn/cpp/ttnn/operations/transformer/gated_delta_attn`, `GatedDeltaAttnSeqDeviceOperation` (single program).
   - **Construct:** CB `c_8` / `S`.
     - The reader `reserve_back`/`push_back`s it once (`device/kernels/dataflow/reader_gated_delta_attn.cpp:103-115`).
     - Compute `wait_front`/`pop_front`s it (`device/kernels/compute/gated_delta_attn.cpp:231,415`).
     - Compute also `push_back`s the recycled state into it on every non-last chunk (`:421-436`).
     - The census is two locked PRODUCERs (reader, compute) plus one CONSUMER (compute). The faithful binding is reader P, compute P+C, `allow_instance_multi_binding = true`, exactly as the brief prescribed.
   - **Why mechanical conversion fails:** `ValidateProgramSpec` (`tt_metal/impl/metal2_host_api/program_spec.cpp:2026-2045` @ `9e90e641f58`) requires that a self-looped DFB's producer set equal its consumer set, even under `allow_instance_multi_binding`:
     ```
     TT_FATAL: DFB 'S' is self-looped (some kernel appears as both producer and consumer), but the set of producer KernelSpecs differs from the set of consumer KernelSpecs. ...
     ```
     The only accepted alternatives misdescribe the kernels:
     - Unbinding compute's PRODUCER mislabels a real `push_back`er.
     - Adding a reader CONSUMER binding is the self-loop + multi-binding stacking the recipe calls a mis-slot.
   - **The rule looks inconsistent with the flag's own semantics.** The same function waives every other per-role check under the flag ("no processor mask, DFB scheduler, or credit machinery", `:1851-1857`), and the lowering calls the mask inert on Gen1 (`:3704-3713`).
   - **Sketch of the change (proposal; not made by the port):** guard the set-equality `TT_FATAL` with `if (!allow_multi)`. Details, tests, and evidence are in `METAL2_HOST_FIX_SELF_LOOP_MULTIBINDING.md`.
   - **Op-side alternative:** a new `c_7` `S0` seed buffer, so `S` is compute-only and the flag disappears. +64 KiB L1 per core; numerically bit-identical. Instructions are in `METAL2_OP_FIX_STATE_SEED.md`.
   - **Evidence:** with only that one cosmetic reader-CONSUMER binding added (local experiment, reverted, never committed), the port was bit-exact vs legacy on 10/10 harness runs with the legality checks forced and both markers logged (5× each). See *Verification* below.

2. **Pre-port: the legacy op cannot run under Watcher on WH 8×8.** *(Owner: op owner / Watcher.)* With `TT_METAL_WATCHER=10`, the **pre-port** op fails: `TT_FATAL: Program size (72320) too large for kernel config buffer (70656) on TENSIX` (`tt_metal/impl/program/program.cpp:3637`). The recipe's mandatory-Watcher step cannot be met for this op on this card, so all verification ran with Watcher off.

## Successes

- **[Conditional / optional resource bindings](port_patterns.md#pattern-conditional--optional-resource-bindings)** fit the optional `initial_state` exactly. The tensor is bound and `HAS_INITIAL_STATE` is emitted only when present. The reader `#ifdef`s the accessor and the read branch, and the zero-fill stays as the `#else`. Both presence states ran bit-exact, including the absent-state case at NC>1.
- **[Hardware configuration](metal2_port.md#hardware-configuration) "decide on the resolved values"** caught a silent trap the audit missed. The legacy DM configs are `{RISCV_1, NOC::RISCV_1_default}` (reader) and `{RISCV_0, NOC::RISCV_0_default}` (writer), i.e. reader on **NOC_1** and writer on **NOC_0**. That is the reverse of the reader/writer defaults (NOC_0 / NOC_1). Reaching for `create_reader_datamovement_config()` would have swapped both NOCs and passed every test. They were ported verbatim as custom Gen1 configs.
- **The `unpack_modes` required-entry rule** (FP32 DFB consumed with `enable_32_bit_dest`) was anticipated from the recipe. All 22 consumed DFBs got an explicit `UnpackToSrc` (legacy `Default`), and the validator accepted them on the first build.
- **Whitelist §A `constexpr` rule:** `constexpr uint32_t f32_tile = get_tile_size(dfb_S)` / `get_tile_size(dfb_out)` kept the token form and compiled cleanly in both DM kernels.
- **Endpoint census "re-derive, don't transcribe":** re-deriving `S` confirmed the brief's multi-binding call. The census agreed; the framework did not.

## Friction

### Gaps
- **Recipe hw_config type names are stale.** The recipe says `DataMovementGen1Config{…}` / `ComputeGen1Config` / `std::get<ComputeGen1Config>(…)`. The headers now have `DataMovementHardwareConfig{.config_1xx = DataMovementHardwareConfig::DataMovement1XXConfig{…}}` and a flat `ComputeHardwareConfig{fpu_math_fidelity, sfpu_precision_mode, enable_32_bit_dest, double_buffer_dest, unpack_modes, config_1xx}`. `create_reader_datamovement_config()` also no longer takes an arch. The header was the authority, as the recipe advises.
- **No pattern covers "seed-then-recycle" state loops.** The audit's own Recipe notes asked for an example under Multi-binding. The answer turns out to be "Metal 2.0 rejects it today", which the patterns catalog's multi-binding entry should say so the audit can gate on it. **Suggested audit gate:** a CB that one kernel FIFO-produces *and* consumes (self-loop) while another kernel also binds it → not portable until the validator relaxes, or the op splits the buffer.
- **Mandatory Watcher vs kernel-config-buffer overflow.** The recipe has no guidance for an op whose *legacy* build already overflows under Watcher. A reduced Watcher configuration (e.g. `TT_METAL_WATCHER_DISABLE_*`) might fit, but it was not tried, because the port capitulated first.

### Confusion
- **Brief vs validator on `S`.** The brief explicitly said "This is not a stacking error: `c_8` gets the flag only and is not self-looped", yet it prescribed compute PRODUCER + CONSUMER, which *is* a self-loop in the validator's sense (a kernel on both sides). The brief, the recipe's stacking guard, and the validator use "self-loop" with slightly different scopes.
- **Kernel-side temporaries.** The compute kernel builds `CircularBuffer(x)` temporaries throughout (~60 sites). The port kept them as `DataflowBuffer(dfb_x)` temporaries, the mechanical swap. The Same-FIFO catalog entry warns against two objects per accessor for debug-tool identity, but restructuring to one object per DFB would be a refactor. A note in the whitelist on legacy temporary-object kernels would help.

## Open items for downstream

- **Re-port recipe once unblocked:** cherry-pick `e9c863c3085` (branch `worktree-metal2-gated-delta-attn`) and fix `S`'s bindings:
  - After the host fix: no change needed (the WIP already has reader P, compute P+C, and the flag).
  - After the op fix: add an `S0` DFB (reader P → compute C); make `S` a compute self-loop with no flag; select `c == 0 ? dfb::S0 : dfb::S` in compute.

  Rerun the bit-exact harness and refresh the plan's census.
- **Verification harness:** `/home/edwinlee/.claude/jobs/daa75c58/tmp/gda_harness.py` + `baseline.pt` (legacy outputs, 10 runs). Job scratch, deleted with the job; copy it out to keep it.
- **Test coverage:** the only device test (`models/experimental/gated_attention_gated_deltanet/tests/test_ttnn_validation.py::test_gated_deltanet_chunked_ttnn`) runs NC = 1 with no `initial_state` and compares two runs of this op against each other. It cannot catch a port regression, never exercises the recycle path, and never sees `initial_state`. `tests/ttnn/unit_tests/operations/transformers/test_chunk_gated_delta_rule.py` covers only the golden. A direct op test like the harness (NC>1, both presence states, cache hits, golden + bit-exact) belongs in `tests/ttnn/unit_tests/operations/transformers/`.
- **Carried-over anomalies** (from the audit; unchanged, not port work): the factory ignores `attrs.compute_kernel_config` but hashes it, `output_mem_config` is unvalidated, and `NC == 0` hangs.
- **Shared kernel touches:** none. All three kernels are op-owned.

## Verification (of the converted code, `e9c863c3085`)

- **Build:** `./build_metal.sh --build-tests` (Release), success.
- **Legality checks:** the 9 `skip_validation` sites were forced (uncommitted scaffolding already in the main checkout). Both markers were present: `METAL2_CHECKS_FORCED (program_spec.cpp)` ×5 and `(program_run_args.cpp)` ×5.
- **As designed (reader P, compute P+C, flag on `S`):** `MakeProgramFromSpec` fails with the self-loop set-equality `TT_FATAL` above.
- **Experiment (+ reader CONSUMER on `S`, reverted):** **10/10 bit-exact** vs the legacy baseline.
  - (BH, NC, initial_state) = (12,1,no), (12,3,yes), (5,2,no), (64,2,yes), (7,4,yes).
  - Each config ran twice, the second as a program-cache hit with fresh tensors at shifted addresses.
  - Golden PCC was 0.99990–0.99999 for both outputs.
- **Watcher:** off for baseline and post-port (see Handoff 2).
- **Test set not confirmed with the invoker.** The broad sweep found the model test and the golden-only unit test above, plus Qwen3.6 BH demos that need Blackhole. The harness was the primary baseline because the existing tests can't detect a port regression.
- **Self-audit** (11 `.cpp`/`.hpp` files scanned):
  - zero `cb` names, `CircularBuffer`/`CBDescriptor` references, address RTAs, `TensorAccessorArgs`, or `.md` citations in code;
  - TT_FATAL counts unchanged;
  - O3 set explicitly on the single compute spec;
  - no `tt_metal/` files in the diff.
