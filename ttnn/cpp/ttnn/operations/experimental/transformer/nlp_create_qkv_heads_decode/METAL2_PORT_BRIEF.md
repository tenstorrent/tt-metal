# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `50e992a8ec2 2026-08-21 docs(metal_2.0): a run in flight freezes the kernel sources` *(carry this line into the port report's Provenance section; hash is the HEAD of `origin/akertesz/op-porting-recipe` — the recipe docs are not on the working branch)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — all three factories (`Interleaved`, `Sharded`, `ShardedSubcoregrid`) on one `NLPCreateQKVHeadsDecodeDeviceOperation`; they port together as one unit.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept` (all three factories; `Override runtime args method? == no`).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a non-`none` `TensorParameter relaxation` · `get_dynamic_runtime_args` (deprecated hook). Also absent on this op (none of these gate, but none is present either): custom hash, `override_runtime_arguments`, pybound `create_descriptor`.
- The op is **pure data movement** — no compute kernels. Every factory instantiates one kernel source 2× (or 4× in the sharded/subcoregrid non-overlap configs) with Reader/Writer configs — the dual-instance work-split shape.

## Construct — to do

**Tensor bindings** (per binding):

- `input_tensor` — **per-factory split**:
  - *Interleaved factory*: **Case 1** (via `TensorAccessor`) → express as `TensorParameter`/`TensorBinding`; the kernel builds `TensorAccessor(tensor::name)` (today: `TensorAccessor(qkv_args, q_start_addr)` @ `device/kernels/reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp:59`, base delivered as a `Buffer*` RTA, `TensorAccessorArgs` CTAs from `..._interleaved_program_factory.cpp:152`). The `Buffer*` RTA and the CTA-appended accessor args both disappear. Keep the **separate** per-core scalar RTA `in_tile_offset_by_batch` as a named arg — it is applied via `.offset_bytes` at the read sites, not folded into the base.
  - *Sharded + Subcoregrid factories*: **Case 2** (raw pointer) → bind the tensor as `TensorParameter`, pull the base via `TensorAccessor::get_bank_base_address`, and keep the existing raw shard-walk unchanged (`qkv_read_addr = q_start_addr + in_tile_offset_by_batch`, remote reads via `UnicastEndpoint` with explicit `noc_x/noc_y/addr` — `reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp:79-116`, `..._on_subcoregrids.cpp:79-113`). Do **not** rewrite the walk into accessor iteration.
- `batch_offset` (optional tensor; Sharded + Subcoregrid only) — **Case 1** → conditional `TensorParameter` binding; kernel side `TensorAccessor(tensor::name)` (today `TensorAccessor(index_args, batch_offset_tensor_addr)` @ kernel `:48`). Today's absent-path plumbing (literal-`0` RTA via the factories' `push_batch_offset` lambda + `TensorAccessorArgs(nullptr)` CTAs + `use_batch_offset` CTA) collapses into a define-gated conditional binding (see Watch-for).
- `q/k/v outputs` — **clean** (borrowed-memory): all three factories back CBs `c_16`/`c_17`/`c_18` with `output[0..2].buffer()` → `DataflowBufferSpec::borrowed_from` the three output `TensorParameter`s.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no accessor in the op passes one.

**CB endpoints** (full per-`(CB, config)` census in the audit doc):

- **1P+1C assign** — every q/k/v output CB in every factory: two co-resident same-source instances raw-write disjoint regions (`get_write_ptr() + offset`, no FIFO ops) — the dual-instance work-split. Bind one instance PRODUCER, the other CONSUMER (cosmetic on Gen1). Configs: interleaved (`c_16/c_17/c_18` on the q grid); sharded/subcoregrid overlap (q pair touches all three); sharded/subcoregrid non-overlap (q pair touches `c_16`/`c_18` on q-cores, k pair touches `c_17` on k-cores).
- **Self-loop** (one toucher):
  - Interleaved scratch `c_0` (Reader instance) and `c_1` (Writer instance) — each written by accessor reads and read back via `tt_memmove`, sync-free, single owner. **Conditional DFB**: they exist only when `use_aligned_path` (DRAM input and `sub_tile_line_bytes < dram_alignment`, factory `:98-133`) — the host conditional already exists; mirror it in the spec and gate the kernel-side binding (Watch-for).
  - Subcoregrid batch-offset CBs `c_15` (Reader instances) and `c_14` (Writer instances — CTA[15] overridden @ factory `:198,229`) — each a single locked producer (`reserve_back`/`push_back` + raw read-back). **Conditional DFB** on `batch_offset.has_value()`.
- **Multi-binding advanced option** — sharded factory `c_15` only, when `batch_offset` is present: **both** co-resident instances are locked producers (each does `reserve_back(1)`/`push_back(1)` on the same instance — CTA[16] is `c_15` for reader *and* writer, factory `:200-201`). Census can't fit 1P+1C → set the flag. Also a **conditional DFB** (`batch_offset.has_value()`).
- **Dead-CB drop** — sharded factory `c_14` @ `nlp_create_qkv_heads_decode_sharded_program_factory.cpp:79-88`: allocated whenever `batch_offset` is present, referenced by **no kernel in any config** of this factory (the kernel takes its CB index solely from CTA[16], which is always `c_15` here). Positively confirmed dead → drop the allocation (a dead CB has no behavior, so removing it changes none). There is no dead CTA to drop — CTA[16] is live (it carries `c_15`).

## Watch for

- **CB endpoints (multi-binding):** sharded `c_15` — don't hunt for a hidden second writer or a downstream consumer; there is neither. The shape is two instances each staging the same one-page scalar for their own read-back (an artifact of the factory not wiring the writer's own CB — see the audit's Misc anomalies; the subcoregrid factory wires it). Express it faithfully with the flag; do **not** "fix" it by giving the writer its own DFB — that's the ops team's change, not the port's.
- **Conditional bindings need define-gating, not `if constexpr`:** `if constexpr` does not gate `dfb::`/`tensor::`/`args::` name lookup. Config- and instance-conditional bindings in this op: the interleaved scratch DFBs (`USE_ALIGNED_PATH`), the batch-offset tensor + DFB (`use_batch_offset`), and — in sharded/subcoregrid **non-overlap** mode — the per-instance output DFBs (`PROCESS_QV` instances must not bind `k_out`, `PROCESS_K` instances must not bind `q_out`/`v_out`: the unused CB **does not exist on that node**, its core range is the other grid). Today's kernels construct all three `CircularBuffer` wrappers unconditionally (e.g. `reader_tm_...cpp:68-70`) and rely on `if constexpr` to skip use — the port must move each of these behind `KernelSpec` `defines` + `#ifdef`.
- **Cross-op / shared kernels:** none — all three kernel sources are op-owned and bound only by this op's factories; no `_metal2` fork exists anywhere for them and this port creates none (no sharing). Each source is bound by exactly one factory, and all three factories convert in this one port, so no fork rungs apply. Negative pointer: sibling ops `nlp_create_qkv_heads` / `nlp_create_qkv_heads_boltz` have similarly-named kernels — unrelated files, not consumers; and anything under `experimental/quasar/**` is out of bounds as a reference.
- **RTA varargs:** sharded + subcoregrid kernels — the NoC-coordinate tables are genuine varargs: CTA-bounded variable-count blocks read through `get_arg_addr` pointers with runtime indexing (`in0_mcast_noc_x/_y` @ `reader_tm_...cpp:42-43`, length `num_x`+`num_y`; `..._on_subcoregrids.cpp:42-43`, length `2 * in_num_cores`). Use the vararg mechanism for the two arrays; **name** the three leading scalars (`q_start_addr` → dissolves into the tensor binding, `batch_offset_tensor_addr` → dissolves into the conditional binding, `index_in_cores` → a named arg) — don't let them ride the varargs. Note `get_vararg` is read-only — the kernels' runtime-indexed reads (`in0_mcast_noc_x[qkv_x]`) are reads, so no mutation issue. Interleaved kernel: no varargs (two fixed RTAs: `in_tile_offset_by_batch` named; the base dissolves into the binding).
- **Donor call, no bridging needed:** the interleaved kernel's one out-of-directory call, `tt::data_movement::common::tt_memmove(noc, dst, src, bytes)` (`data_movement/common/kernels/common.hpp:143`), takes plain `uint32_t` L1 addresses + a leading `Noc` — Device 2.0 native, no resource handles, no fork, no donor change. Keep calling the leading-`Noc` overload (the Noc-less one is `[[deprecated]]`).
- **Same-source instances differ only by CTAs:** each factory's Reader/Writer (and q/k) instances share one kernel source, differentiated by CTA overrides (`PHASES_TO_READ`, `PROCESS_QV`/`PROCESS_K`, scratch/batch-offset CB index). Both instances cover **every** node of their grid (work-split, not disjoint node sets) — keep them as two `KernelSpec`s over the same core ranges; per-instance constants stay CTAs/defines (mind the demoting-per-group-CTA anti-pattern in `port_patterns.md` — it does not apply here, but the two shapes look alike).
