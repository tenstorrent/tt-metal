# Port Report — experimental/matmul/group_attn_matmul

Port of `ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul` from the `ProgramDescriptor` API to
Metal 2.0, per `ai/port/metal2_port.md`. Line references into the legacy code are against `origin/main` @
`0df0dde3316` (`factory:` is `device/group_attn_matmul_program_factory.cpp`, `reader:` / `writer:` / `compute:`
the three kernels); references into the ported code are against this branch; `dfb.cpp` is
`tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp`.

## Outcome

**PORTED** — the op's single factory (`GroupAttnMatmulProgramFactory`) and its three owned kernels converted;
the full `group_attn_matmul` selection passes with the counts of the legacy build. One structural decision was
re-made mid-port: the audit's Question-1 shape for `in1` on the mcast-only cores (a second same-source reader
`KernelSpec` bound PRODUCER + CONSUMER in its own `WorkUnitSpec`) was built as instructed, rejected by the spec
validator on every `q_heads < 32` configuration, recorded as handoff point 1 and handed back; the invoker chose
the legacy placement (option (a) below), which is what the final factory builds. The rejected shape lives in the
branch history (commit `99af85c19e0`).

Verification on a Blackhole p150b (device 0), Watcher on (`TT_METAL_WATCHER=10`), Metal 2.0 legality checks
forced on (every `skip_validation` site set `false` in the working tree, `METAL2_CHECKS_FORCED` markers present
in every log — scaffolding only, not in the diff):

| Suite (`tests/ttnn/nightly/unit_tests/operations/matmul/test_attn_matmul.py -k group_attn_matmul -q --timeout 600 --tb=short`) | Legacy (`0df0dde3316`, same build flags, same env) | Ported, final shape (option (a)) | Ported, rejected shape (commit `99af85c19e0`) |
|---|---|---|---|
| all four `test_group_attn_matmul*` functions, 454 selected cases (300 `test_attn_matmul*` deselected) | **322 passed, 132 skipped, 0 failed**, 4:30 | **322 passed, 132 skipped, 0 failed**, 1:34 | 204 passed, 118 failed, 132 skipped |
| of which `q_heads >= 32` (`32` × 120, `48` × 42, `64` × 42: every sharding / orientation / buffer-type / optional-output combination, program-cache sweeps, bf16 and fp32 with 32-bit accumulation) | 204 passed | 204 passed | 204 passed |
| of which `q_heads < 32` (`test_group_attn_matmul` `q=16` × 16, `test_group_attn_matmul_fp32` `q=2` × 2, `test_group_attn_matmul_with_program_cache` `q=10` × 16, exhaustive `q=16` × 42 and `q=24` × 42 — the mcast-only-core protocol) | 118 passed | **118 passed** | 118 failed, all with the handoff-1 `TT_FATAL` at the first (cache-miss) dispatch, before any kernel ran |

The 132 skips are the exhaustive test's own guards (`test_attn_matmul.py:656`, `:659`) in every run. No Watcher
error, no hang, no timeout in any run; the `METAL2_CHECKS_FORCED` marker appears in every log (1000 occurrences
in the final run).

Not run: Wormhole; the Falcon demos (`models/demos/falcon7b_common`, `ttnn_falcon7b`, `t3000/falcon40b`). No
bit-exact capture against the legacy build was done (the attn_matmul port did one; here the device time went to
the full sweep, twice).

## Provenance

- **Recipe docs (this port):** `akertesz/op-porting-recipe` @ `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(read from a verbatim extraction of the recipe branch's `docs/…/metal_2.0/` tree into a scratch directory; this branch does not carry it, so `git log -1 -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` prints nothing here)*
- **Audit docs (inherited):** `akertesz/op-porting-recipe` @ `4bd4bf42bfe` (the brief's provenance line)
- **Code ported from:** `origin/main` @ `0df0dde3316` (2026-10-27)

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`, as the audit chose: `GroupAttnMatmulProgramFactory::create_program_artifacts`
(`device/group_attn_matmul_program_factory.{hpp,cpp}`) returns one `ProgramSpec` + `ProgramRunArgs`; no
`override_runtime_arguments`, no op-owned tensors. On a cache hit the framework refreshes the three tensor
bindings (`input_a`, `input_b`, `output`), which is exactly what the pointer-patching descriptor path refreshed
(the three `Buffer*` run-arg slots and the three borrowed CB addresses — the borrowed DFBs re-attach from the
same tensor arguments).

### Device-op-class edits
- Pybind entry points removed: none (`group_attn_matmul_nanobind.cpp` binds only the public wrapper).
- Custom `compute_program_hash`: none (default reflection hash; untouched).
- Nothing outside the factory body and the three owned kernels changed: `group_attn_matmul_device_operation.{hpp,cpp}`,
  `group_attn_matmul_device_operation_types.hpp`, `group_attn_matmul.{hpp,cpp}` and the nanobind files are
  byte-identical.

### Open items
- Relaxation candidate (not applied; the audit's team note): the kernels take `Mt` / `Kt` / `Nt` / the
  `KV_HEADS`-derived counts as run-args, but five DFB capacities and the mcast geometry run-args are keyed on
  `(q_heads, kv_heads, head_dim, kv_len)`, so every new combination costs a program build under the strict
  `TensorSpec` match. A relaxation would need the custom concept (geometry run-args go stale on a hit) plus DFB
  re-sizing; ops-team call.

## Handoff points

### 1. `in1` on the mcast-only cores: the validator rejects the two-`KernelSpec` split — **Decision: option (a), taken by the invoker**

**Decision and rationale (invoker, after reading this entry):** take option (a), legacy placement. The validator
rejects the node-set split on the two rules cited below (per-role kind uniformity, self-loop set equality); the
multi-binding flag would not admit it either; (b) introduces an undeclarable L1-ordering dependency; (d) is out
of port scope. (a) is validator-legal, needs no flag, is byte-identical to the legacy behaviour on Gen1, and the
118 then-failing `q_heads < 32` tests verify it empirically. The debt — a consumer FIFO op (`reader:327-328`)
issued by a PRODUCER-bound DM kernel on the mcast-only cores, sanctioned by no document and not viable on
Gen2 — is accepted for the Gen1 port and handed to the Quasar-uplift audit, which must redesign this op's mcast
protocol anyway (DM self-loop on `in2`, hand-built multicast). **Final shape:** one `WorkUnitSpec` over the whole
device grid with all three kernels, `in1` bound reader PRODUCER / compute CONSUMER on every node; the
`READER_MCAST_ONLY` spec, its work unit and its run-args are gone; no kernel edit. The analysis that led there is
kept below as the record, and feeds two open items (Metal 2.0 API owners; Quasar-uplift audit).


**Where:** `device/group_attn_matmul_program_factory.cpp`, `make_reader(READER_MCAST_ONLY, /*mcast_only=*/true)`
and the `mcast_only` `WorkUnitSpec` (emitted only when `q_heads < 32`).

**What was built (the invoker's instruction on audit Question 1):** `READER` (PRODUCER of `in1`) with `WRITER` and
`COMPUTE` on every device core except the mcast-only set; `READER_MCAST_ONLY` — the same source, CTAs, defines,
run-arg schema, varargs and semaphore bindings — PRODUCER **and** CONSUMER of `in1` (plus PRODUCER + CONSUMER of
`in2` under `IN1_SHARDED` and the `input_b` binding when interleaved) alone on exactly the mcast-only cores
(`q_heads <= i < 32`). Per node that is one producer and one consumer instance everywhere.

**Exact error** (every `q_heads < 32` case, e.g. `test_group_attn_matmul_fp32[... q_heads=2 ...]`, raised from
`MakeMeshWorkloadFromSpecs` on the cache miss, before any device work):

```
TT_FATAL: DFB 'in1' has multiple CONSUMER KernelSpecs mixing compute and data-movement kinds ('compute' is a
compute kernel; 'reader_mcast_only' is a data-movement kernel). All KernelSpecs bound to the same DFB role must
be of the same kind — the DFB's hardware config carries a single processor mask per role. (assert.hpp:104)
```

**Why:** two independent rules in `ValidateDFBEndpointPlacement` (`dfb.cpp`) reject the shape, so it cannot be
expressed by re-arranging the same bindings:
1. *Per-role kind uniformity* (`dfb.cpp:440-470`, the error above): every `KernelSpec` bound to one role of a DFB
   must be the same kind, because the DFB's hardware config carries a single processor mask per role. `in1`'s
   CONSUMER role would be held by a compute kernel on the main nodes and by a DM kernel on the mcast-only nodes.
   This is exactly what the per-node census in `dataflow_buffer_spec.hpp:41-45` lists as its third condition
   ("the same kernel kind"); the audit and brief read the invariant as per-node only and missed the per-role
   kind condition — the recipe has no wording for roles that differ per node class.
2. *Self-loop set equality* (`dfb.cpp:575-625`): when any `KernelSpec` self-loops a DFB, the set of PRODUCER
   `KernelSpec`s must equal the set of CONSUMER `KernelSpec`s. Here producers are `{reader, reader_mcast_only}`
   and consumers `{compute, reader_mcast_only}`. This rule is **not** relaxed by `allow_instance_multi_binding`
   (only the per-node count and the uniformity checks are, `dfb.cpp:420-430`), so the flag would not have
   admitted the split either — nor the single-reader variant (one full-grid reader bound PRODUCER + CONSUMER
   with compute CONSUMER: `{reader}` vs `{reader, compute}`).

**Options, with what each costs (for the invoker / API owners to decide; none taken on my own):**
- **(a) Legacy placement, no split.** One work unit over the whole device grid with all three kernels, exactly
  as the descriptor factory placed them; `in1` bound reader PRODUCER / compute CONSUMER on every node. Compute
  and writer return at their first statement on the mcast-only cores as today (`compute:21`, `writer:16-18`);
  the reader's `push_back` + `pop_front` there (`reader:327-328`) stays a kernel-internal FIFO op that Gen1's
  role-free circular-buffer lowering executes as it always has (the kernel-side token carries no role,
  `api/dataflow/dfb_binding_token.h`). Validator-legal, no flag, byte-identical behaviour, and the smallest diff
  (drops `READER_MCAST_ONLY`, the second work unit and run-args object, ~40 factory lines; no kernel change).
  Debt: a consumer FIFO op from a PRODUCER-bound DM kernel is sanctioned by no document and would break on Gen2
  — but so would the DM self-loop on `in2` and the hand-built multicast, and this op's mcast protocol needs a
  Quasar redesign regardless. The recipe's stop-signal text ("the validator's check is satisfied by adjusting
  the *binding* declaration on the host ... a tile produced and never consumed is harmless") is the closest
  sanction. This was the recommendation, and the shape the invoker chose.
- **(b) Relay DFB, as `matmul_multicore_reuse_mcast_2d_program_factory.cpp` does for `in0` (`in0_relay`).** A
  second `DataflowBufferSpec` with `in1`'s geometry, self-looped by `READER_MCAST_ONLY` under the same accessor
  name `in1` (kernel source unchanged); `in1` keeps reader → compute. Validator-legal. But the multicast
  destination is the sender's *local* `in1` write pointer (`reader:243-282`, `.offset_bytes = 0`), so the relay
  must sit at `in1`'s L1 offset on every node, and `allocate_dfb` (`tt_metal/impl/program/program.cpp:2377`)
  places each DFB at the running top of its own node set: a relay whose nodes host nothing before it lands at
  the allocator base, while `in1` follows `in0` on the other nodes. The precedent lives with the same constraint
  by declaring its pair first ("KEEP THIS PAIR FIRST ... Metal 2.0 has no way to state the requirement yet").
  Here that means declaring `in1` and the relay ahead of `in0` — a different L1 layout from legacy (functionally
  neutral) plus an undeclared ordering dependency.
- **(c) `allow_instance_multi_binding` on `in1`** — needs a validator change as well (rule 2 above), so it is not
  a drop-in either; the invoker ruled it out in any case.
- **(d) Framework:** teach the validator the "same-source DM `KernelSpec` holding the consumer role on a disjoint
  node set" case (relax the per-role kind check when the extra consumer is a self-loop participant on nodes the
  compute kernel does not cover, and the self-loop equality to "per node"), or give `DataflowBufferSpec` a way to
  pin two DFBs to one offset so (b) becomes declarable. Either is an API-owner change.

**State of the branch:** commit `99af85c19e0` carries the instructed (rejected) shape, 204 passed / 118 failed on
it; the following commit switches the factory to (a) — a factory-only change — and the full selection passes.

No other handoff: no boundary-rule violation (no `sem::` / `tensor::` handle leaves the op's kernels), no
kernel-lib gap, no removed pybind surface, no out-of-directory kernel edit.

## Successes

- **Audit brief → port, 1:1 apart from handoff point 1.** Every other item mapped directly: the three conditional
  tensor bindings paired with the `IN0_SHARDED` / `IN1_SHARDED` / `OUT_SHARDED` defines and the matching
  `borrowed_from` DFBs (all eight sharding combinations of `test_group_attn_matmul` pass), the conditional `in2`
  DFB with its two self-loop bindings, the two `SemaphoreSpec`s on the full grid read as `sem::` tokens (the
  remote `sender_sem.up(noc, x, y, 1)` / `receiver_sem.set_multicast(noc, …)` worked unchanged), and the
  `DataflowBuffer` as the multicast destination (`noc.async_write_multicast<MCAST_INCL_SRC>(mcast_src, dfb_in1,
  …)`), verified by the passing `q_heads = 32` sweep where cores 0..31 are senders and receivers at once.
- **Caution: avoid varargs — the one genuine block.** The sender coordinate tables became
  `num_runtime_varargs = num_x + num_y` with `get_vararg(x)` / `get_vararg(num_x + y)` replacing the
  `get_arg_addr` pointer walk; the two counts stayed named. The generated `get_vararg` bakes in the named-arg
  offset, so the kernel indexes from 0 as the recipe says.
- **Hardware configuration — "check for a dropped field" (Style A).** The legacy descriptor applied only
  `math_fidelity` and `fp32_dest_acc_en`; forcing `sfpu_precision_mode = Precise` and `double_buffer_dest = true`
  on the helper's result is what keeps a caller that sets `math_approx_mode` / `dst_full_sync_en` on legacy
  behaviour (`test_group_attn_matmul_fp32` passes `math_approx_mode=True`).
- **`unpack_modes` required-entry rule.** `test_group_attn_matmul_fp32[in_dtype=FLOAT32]` consumes Float32
  `in0` / `in1` / `intermed1` under 32-bit Dest; the three conditional `UnpackToSrc` entries satisfied the
  validator first time.
- **Whitelist rule 7 and the conditional-binding pattern on the kernel side.** `get_tile_size(cb_id)` on runtime
  `const`s became `dfb.get_tile_size()` in both DM kernels; gating `in2`'s declaration / object / peek under
  `#ifdef IN1_SHARDED` and giving the peeked address an `#else` fallback kept the dead direct-mcast branch
  compiling in the interleaved build without touching the logic.
- **`AddRuntimeArgsForNode` kept the legacy node-first loop.** Routing each core's reader args to one of two
  `KernelRunArgs` objects by node class was a one-line reference choice inside the untouched loop, so the
  per-core values could not drift from the flags that define the classes.
- **The self-audit greps** (`cb` names, legacy idioms, `opt_level`, `TT_FATAL` census, `.md` citations,
  `tt_metal/` files in the diff) all came back clean over 12 files on the first pass.

## Friction

- **Gap (recipe + catalog): endpoint roles that differ per node class inside one program.** The only shape the
  docs sanction for "same source, different bindings, disjoint nodes" is the CTA work-split in *Anti-pattern:
  Demoting per-group CTA to RTA*; nothing says the per-role *kind* must agree across those `KernelSpec`s
  (`dataflow_buffer_spec.hpp:43` does, in the header the recipe tells you to trust), nor that a self-loop
  participant forbids any non-self-loop binder of the same DFB (`dfb.cpp:605-625`, header-undocumented). The
  audit's recipe note 1 asked for a "per-node-set bindings" entry; this port shows the entry also has to say
  which role combinations the validator admits. Mcast senders without compute work are a recurring matmul shape
  (the in-tree 2D matmul port needed a relay DFB for the same reason).
- **Confusion (recipe, Hardware configuration):** the text names `ComputeGen1Config` / `DataMovementGen1Config`,
  `to_compute_hardware_config(device->arch(), config)` and `create_reader_datamovement_config(device->arch())`;
  on this tree `ComputeHardwareConfig` is a flat struct with optional `config_1xx` / `config_2xx`, and both TTNN
  helpers take no arch (the attn_matmul port reported the same).
- **Confusion (recipe, Ensure the legality checks are enabled):** `grep -n 'bool skip_validation'
  tt_metal/impl/metal2_host_api/*.cpp` misses `program_spec/program_spec.cpp` (a subdirectory); a recursive grep
  finds nine sites. Note that TTNN's spec adapter always validates on a cache miss (`mesh_device_operation_adapter.hpp`,
  "Cache miss is the cold path: always validate here"); `validate_program_args` only gates the hit-path re-checks.
- **Gap (migration guide, borrowed memory):** the legacy factory borrows `in0` / `in2` / `out` from sharded
  tensors on *every* device core, including cores outside the shard grid; the validator and `AttachBorrowedDFBBuffers`
  check only L1 residency and the per-bank size, so the full-grid placement is accepted as it was. Worth a
  sentence, since a porter has to decide whether that is still legal.
- **Gap (workspace):** a fresh `git worktree` needs `git submodule update --init --recursive` before
  `./build_metal.sh`; ccache is off unless `-c` is passed (both also reported by the attn_matmul port).
- **Friction (tests):** `-k group_attn_matmul` selects 454 cases of which 132 are skipped by the exhaustive
  test's own guards; the baseline sweep takes about 4.5 minutes on p150b with Watcher on.

## Open items for downstream

- **Shared kernel touches:** none — the three kernels are owned by this op and were converted in place; no fork
  created or reused.
- **Quasar-uplift debt carried faithfully from the legacy kernels (not Gen1 work):** the DM self-loop on `in2`
  (both reader specs); the hand-built loopback endpoints (`UnicastEndpoint` from `my_x[noc.get_noc_id()]` /
  `my_y[…]` with `CoreLocalMem` operands) in the reader's sharded `in1` staging copy and the writer's row gather;
  the multicasts sourced from peeked addresses (`in2`'s read pointer, `in1`'s write pointer) with the
  `DataflowBuffer` as destination; the reader's write-pointer peek after each `reserve_back`. All belong to a
  post-port style / semantic pass. **Added by the handoff-1 decision (Quasar-uplift audit):** on the mcast-only
  cores (`q_heads..31` when `q_heads < 32`) the reader — bound PRODUCER of `in1` — also issues `pop_front`
  (`reader:327-328`), a consumer FIFO op from a producer binding; Gen1's role-free circular-buffer lowering
  executes it as legacy did, Gen2's credit machinery would not. The mcast protocol of this op (same-CB send and
  receive, `CoreLocalMem` sources from peeked addresses) needs a Quasar redesign in any case.
- **For the Metal 2.0 API owners (from handoff point 1):** the audit, brief and recipe all read the DFB endpoint
  invariant as per-node only and missed the per-role *kind* condition (`dataflow_buffer_spec.hpp:43`,
  `dfb.cpp:440-470`) and the undocumented self-loop set-equality rule (`dfb.cpp:575-625`, not relaxed by
  `allow_instance_multi_binding`). The shape they forbid — a same-source DM `KernelSpec` holding the consumer
  role on a disjoint node set where no compute kernel runs — is the natural expression of mcast senders without
  compute work (this op; the 2D matmul port's `in0_relay`). Either admit it (relax the kind check when the extra
  consumer is a self-loop participant on nodes the compute consumer does not cover, and the equality rule to
  per-node), or document the rule and give `DataflowBufferSpec` a way to pin two DFBs to one offset so the relay
  shape becomes declarable.
- **Hardware config note:** both DM specs use the TTNN reader / writer helpers because the legacy resolved
  triples (`RISCV_1` + `preferred_noc_for_dram_read(arch)` = `NOC_0`; `RISCV_0` + the other NOC) equal the
  defaults by value as of `0df0dde3316`. The mcast rectangle's start / end orientation (`factory:382-385`) is now
  keyed off the NOC read back from the reader's config, so the two cannot drift; the legacy's arch-dependent
  intent lives only in the helper now.
- **Dropped with the port (zero behaviour):** `writer:59` `const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);`
  (unused; the writer binds no `in1`, decided by the invoker on audit Question 2) together with the `cb_id_in1`
  constant and its role comment (the same comment lives on the reader's `in1` binding); the stale
  `compute_program_hash()` / "CB-size re-application on cache hit" comments (`group_attn_matmul_program_factory.hpp:15-20`,
  `factory:58-59`, `:155-158`; the custom hash was removed by PR #47232 — the substance is restated at the DFB
  declarations); the `(these vary across cache hits)` qualifier on the shape-derived block.
- **Latent issues in the legacy code, not touched by the port (ops team):**
  - `factory:34-35` vs `:317-320` — `math_approx_mode`, `packer_l1_acc`, `dst_full_sync_en` resolved and ignored;
    reproduced (precise SFPU, double-buffered Dest).
  - The KV-cache attributes (`num_tokens`, `transpose_hw`) are unreachable from the public API
    (`group_attn_matmul.cpp:79-80`); the `transpose_hw` branch of the factory and kernels is ported but untestable.
  - `factory:351` `KV_HEADS == 0` guard is unreachable (validation already divided by it); `:92-94`
    `ONE_ROW_BFLOAT16_BYTES` is 128 under fp32 accumulation; `in1_start_id` is always 0, `in1_num_subblocks` and
    `out_subblock_h` are `constexpr 1` passed as run-args, `blocks` / `batch` are forced to 1 on active cores —
    all kept as named run-args with the same values.
  - `compute:159` says `intermed1` "comes from reader"; it is filled by the writer kernel. Kept as is.
  - The writer's `out_subblock_w` CTA and the `onetile` constants in writer and compute are unused; kept.
- **Candidate later cleanup (not port work):** the vararg coordinate vectors are identical on every node and
  could be common runtime varargs; the `has_work_*` flags are the only per-core values on the idle cores.
- **Test coverage notes:** the `q_heads < 32` parametrizations (`16` in `test_group_attn_matmul` and the
  exhaustive test, `24` in the exhaustive test, `10` in `test_group_attn_matmul_with_program_cache`, `2` in
  `test_group_attn_matmul_fp32` — 118 cases) are the only coverage of the mcast-only protocol; they pass on the
  final shape. Nothing passes `dst_full_sync_en=True`; `test_group_attn_matmul_fp32` is the only caller of
  `math_approx_mode=True` and the kernel runs no SFPU op, so neither knob is observable.
