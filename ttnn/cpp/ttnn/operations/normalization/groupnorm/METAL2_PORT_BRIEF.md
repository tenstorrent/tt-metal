# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/normalization/groupnorm`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4d6e9518bf5 2026-07-30 docs(metal_2.0): require an explicit opt_level when porting compute kernels` *(carry this line into the port report's Provenance section)*

## Scope of this op

One device operation, `ttnn::prim::GroupNormDeviceOperation`, with **three** program factories that share a common kernel set:

| Factory | File | Kernels (non-welford / welford) |
|---|---|---|
| `GroupNormShardedProgramFactory` | `device/groupnorm_sharded_program_factory.cpp` | reader `reader_mcast_{sender,receiver}_unary_sharded_gn_v2` / `welford_…`; writer `writer_unary_sharded_gn_rm_gb_v2` / `welford_…`; compute `groupnorm_sharded_v2` / `welford_groupnorm_sharded_v2` |
| `GroupNormMcastProgramFactory` | `device/groupnorm_mcast_program_factory.cpp` | reader `reader_mcast_{sender,receiver}_unary_gn` / `welford_…`; writer `writer_unary_gn_rm_gb` / `welford_…`; compute `groupnorm` / `welford_groupnorm` |
| `GroupNormNoMcastProgramFactory` | `device/groupnorm_no_mcast_program_factory.cpp` | same kernel set as the mcast factory, but only the *sender* reader, instantiated twice over disjoint `group_1` / `group_2` core sets |

Every kernel the op runs lives in `device/kernels/` — the op instantiates **no** borrowed kernel file, so there is no `_metal2` fork to create or reuse and no cross-op coordination.

**Do not look at `ttnn/cpp/ttnn/operations/experimental/quasar/`** for precedent while porting this op. There is groupnorm-shaped code in there; it is a deliberately hacky pre-port copy and carries idioms this recipe forbids.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — all three factories are a `static ProgramDescriptor create_descriptor(...)`.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept` (plain, no op-owned tensors), for all three factories.
- **Gate-cleared, confirmed absent:** custom hash · `get_dynamic_runtime_args` · `override_runtime_arguments` · pybind `create_descriptor` · other migration-risky pybind. All `no` on the readiness sheet and all confirmed absent in the code.

## Construct — to do

### Tensor bindings

Seven bindings. Classification differs between the sharded factory and the two non-sharded ones — bind per factory.

**Sharded factory:**

- `input` — **clean** (borrowed-memory DFB). `CBDescriptor{.buffer = a.buffer()}` on `c_0`, [groupnorm_sharded_program_factory.cpp:837](device/groupnorm_sharded_program_factory.cpp#L837) and `:856` → `DataflowBufferSpec::borrowed_from`.
- `output` — **clean** (borrowed-memory DFB) on `c_16`, `:877`. **When `inplace`, `c_16` is a second `CBFormatDescriptor` on the input's allocation** (`:823-846`) — input and output share one borrowed buffer, with two buffer indices over it. Preserve that aliasing.
- `gamma` — **Case 1** → `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(tensor::gamma)`. Host site `:1235`, kernel site [writer_unary_sharded_gn_rm_gb_v2.cpp:152](device/kernels/dataflow/writer_unary_sharded_gn_rm_gb_v2.cpp#L152).
- `beta` — **Case 1**. Host `:1240`, kernel `:195`.
- `input_mask` — **Case 1**. Host `:1245`, kernel `:75`.
- `negative_mask` — **Case 1**, sharded only. Host `:1250`, kernel `:84`.

**Mcast and no-mcast factories:**

- `input` — **Case 1**. `a.buffer()` into the reader's `RTArgList` ([groupnorm_mcast_program_factory.cpp:1083](device/groupnorm_mcast_program_factory.cpp#L1083), `:1141`; [groupnorm_no_mcast_program_factory.cpp:1340](device/groupnorm_no_mcast_program_factory.cpp#L1340)) → `TensorAccessor(src0_args, src_addr)`.
- `output` — **Case 1**, bound by **two** kernels: the reader (re-reads the output for the third pass) and the writer. Reader host `:1084` / `:1141` / `:1341`; writer host `:1181` / `:1436`.
- `gamma`, `beta`, `input_mask` — **Case 1**, same shape as the sharded factory. Host `:1183/:1188/:1193` (mcast), `:1438/:1443/:1448` (no-mcast).
- `reciprocals` — **clean** (borrowed-memory DFB) on `c_18`, [groupnorm_mcast_program_factory.cpp:1050](device/groupnorm_mcast_program_factory.cpp#L1050) / [groupnorm_no_mcast_program_factory.cpp:1299](device/groupnorm_no_mcast_program_factory.cpp#L1299). Welford + reciprocals only.

Every one of these is the **`Buffer*`-binding form**, not the silent-wrong `->address()` form: the factories push the `Buffer*` object into the RTA list and the framework injects the base. There is no `->address()` expression anywhere in this op. So this is routine mechanical work — no correctness hazard to unwind.

**Optional tensors need optional bindings.** When `gamma` / `beta` / `input_mask` / `negative_mask` is absent, the factory pushes a literal `0u` into the RTA slot and appends a placeholder `TensorAccessorArgs()` — [groupnorm_sharded_program_factory.cpp:1234-1253](device/groupnorm_sharded_program_factory.cpp#L1234-L1253), `:597-609`. The kernel never dereferences the null accessor: use is gated behind the `fuse_gamma` / `fuse_beta` compile-time flags and the `FUSE_NEGATIVE_MASK` define. Model this as a binding that is present or absent per program build, not a live binding carrying zero.

### TensorParameter relaxation

None.

### TensorAccessor 3rd arg

None — every `TensorAccessor` construction in the op is the two-argument form. (Don't be misled by the `get_tile_size(dfb_id)` values passed to `noc.async_read(...)`: those are transfer sizes, not accessor page-size overrides.)

### CB endpoints

Each node runs exactly **one reader, one writer and one compute kernel**. Where the same kernel source appears twice, the two `KernelDescriptor`s cover disjoint core ranges (mcast-sender vs. mcast-receiver cores; `group_1` vs. `group_2`), so no node ever has two instances of one source. **No CB in this op needs the multi-binding advanced option.**

Dispositions:

- **1P+1C (the common case)** — all the writer-fed input CBs (`c_2` scaler, `c_3` eps, `c_4` scaler-global, `c_5` gamma, `c_6` beta, input mask, `c_14` negative mask, `c_26` ones in the sharded factory), the repack pair, and the reduce CBs while mcasting. Bind the writer PRODUCER and compute CONSUMER, or reader ↔ compute as the census dictates.
- **Self-loop (one toucher)** — the compute-only intermediates (`c_13` `x` and `c_17` `ex2pe` in the sharded factory; `c_24`, `c_25`, `c_23`, `c_22`, `c_27` in the non-sharded factories), the tilized-input CB `c_1` / `c_29` and its welford fp32 aliases `c_29` / `c_31` / `c_19`, the borrowed `c_18` reciprocals, and `c_16` in the sharded factory (one toucher in every configuration — the reader's raw write under `READER_REPACK && UNTILIZE_OUT`, otherwise compute's untilize/pack target).
- **Dead-CB drop, config-scoped** — see the next section; this is the item to be careful with.

The full per-CB, per-config table is in `METAL2_PREPORT_AUDIT.md` under *CB endpoints*.

**Watch the helper-mediated FIFO ops.** The compute kernels do a lot of their CB work *only* through `compute_kernel_lib::reduce<…, in_dfb, scaler_dfb, out_dfb>`, `tilize<…, in_dfb, out_dfb>`, `untilize<…>` and `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<dfb_id, …>`, which take DFB ids as template parameters and perform the `reserve_back` / `push_back` / `wait_front` / `pop_front` internally. A grep for the direct method calls misses those touchers. For example `c_8` `ex_partial` in the sharded factory looks consumer-only until you spot that `reduce<…, dfb_ex_partial_id>` at [groupnorm_sharded_v2.cpp:283](device/kernels/compute/groupnorm_sharded_v2.cpp#L283) is its producer.

### Dead CBs to drop (per configuration)

A DFB with no producer and no consumer binding is rejected by the spec validator, so these must be dropped in the build where they are dead — and only there; the same indices are live in sibling builds. All four were established from the compile-time guards; confirm each against the instantiation you are building.

1. **Sharded factory, `use_mcast == false`** (`num_cores_per_batch == 1 && num_cores_per_group == 1`, [groupnorm_sharded_program_factory.cpp:359](device/groupnorm_sharded_program_factory.cpp#L359)): the `c_9` + `c_15` descriptor (`:1066-1082`) and `c_10` (`:1052-1063`) have zero touchers. The reader's mcast block is behind `if constexpr (num_mcast_cores > 1)` ([reader_mcast_sender_unary_sharded_gn_v2.cpp:157](device/kernels/dataflow/reader_mcast_sender_unary_sharded_gn_v2.cpp#L157)); compute's `c_9` push and `c_10` reduce are behind `is_mcast_sender and num_cores_per_mcast_group > 1` ([groupnorm_sharded_v2.cpp:286-298](device/kernels/compute/groupnorm_sharded_v2.cpp#L286-L298)); and `dfb_ex_global_id` aliases to `c_8` in that config ([groupnorm_sharded_v2.cpp:83](device/kernels/compute/groupnorm_sharded_v2.cpp#L83)), which turns `c_8` into a self-loop.
2. **Sharded factory, receiver cores:** `c_9` is unbound there (the receiver reader never names it; compute's `c_9` block is `is_mcast_sender`-guarded). Bindings are per `KernelSpec`, so this is expressible — but `c_9` shares a `CBDescriptor` with `c_15`, so be deliberate about which buffer indices each side binds.
3. **No-mcast factory, always:** `c_9` `ex` and `c_13` `ex2` have zero touchers — every core is its own mcast group of size 1, so `num_mcast_cores > 1` is false everywhere ([reader_mcast_sender_unary_gn.cpp:450](device/kernels/dataflow/reader_mcast_sender_unary_gn.cpp#L450), [groupnorm.cpp:374-377](device/kernels/compute/groupnorm.cpp#L374-L377)). They are the second buffer index of the `c_15` / `c_14` allocations, which stay live.
4. **Mcast and no-mcast factories, `!UNTILIZE_OUT && !gamma && !beta`:** the output CB `c_16` is untouched — both compute and the writer resolve `dfb_out_id` to `c_22` in that branch ([groupnorm.cpp:171](device/kernels/compute/groupnorm.cpp#L171), [writer_unary_gn_rm_gb.cpp:81](device/kernels/dataflow/writer_unary_gn_rm_gb.cpp#L81)) and the output tensor is written from `c_22`. **This one is flagged as a question to the ops team** in the audit — a dead *output* CB is unusual. Do not drop it until that is answered.

## Watch for

- **CB endpoints (multi-binding):** none. I hunted the hidden-second-writer face specifically. The op's one raw-write shape — the readers' `dfb_out0.get_write_ptr()` into the borrowed `c_16` under `READER_REPACK && UNTILIZE_OUT` — is not a co-fill: in exactly that configuration compute's untilize target is `c_12` / `c_31` (`repack_out`), not `c_16`. The two semaphores coordinate the mcast reduction, not a CB.

- **Cross-op / shared kernels:** no borrowed kernel files — nothing to fork, no sunset list. One donor-signature flag to expect:
  - ⭐ `generate_bcast_col_scalar(CircularBuffer cb, uint32_t scalar)` (`ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp:13`) takes a `CircularBuffer` **by value**. All four writer kernels call it as `generate_bcast_col_scalar(CircularBuffer(eps_dfb_id), eps)` — [writer_unary_sharded_gn_rm_gb_v2.cpp:148](device/kernels/dataflow/writer_unary_sharded_gn_rm_gb_v2.cpp#L148), [welford_writer_unary_sharded_gn_rm_gb_v2.cpp:68](device/kernels/dataflow/welford_writer_unary_sharded_gn_rm_gb_v2.cpp#L68), [writer_unary_gn_rm_gb.cpp:156](device/kernels/dataflow/writer_unary_gn_rm_gb.cpp#L156), [welford_writer_unary_gn_rm_gb.cpp:106](device/kernels/dataflow/welford_writer_unary_gn_rm_gb.cpp#L106). The call site materialises the `CircularBuffer` from a raw id, and in the port there is no id to materialise it from. This is the `CircularBuffer&`-shaped donor case with no clean per-op story today — if you cannot bridge it, that is an assumption-violation stop, not something to work around.
  - Everything else bridges cleanly: the `kernel_lib` helpers take `uint32_t` DFB ids as NTTPs (covered by `dfb::name`'s constexpr cast), and the in-family `get_pointer_to_cb_data<To>(uint32_t cb_id, uint32_t tile_index)` (`ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/memory.h:30`, called once at [welford_groupnorm.cpp:247](device/kernels/compute/welford_groupnorm.cpp#L247)) takes a plain `uint32_t` cb id.

- **RTA varargs — genuine, in the four mcast sender readers.** The multicast group's per-core NoC coordinates arrive as a variable-count runtime-arg block read by pointer:

  ```cpp
  noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(17));
  noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(17 + num_mcast_cores));
  ```

  then indexed in a loop bounded by `num_mcast_cores` (`noc_coord_x[i + 1]`). Sites: [reader_mcast_sender_unary_sharded_gn_v2.cpp:82-107](device/kernels/dataflow/reader_mcast_sender_unary_sharded_gn_v2.cpp#L82-L107) (consumed at `:199-209`), [welford_reader_mcast_sender_unary_sharded_gn_v2.cpp:76-101](device/kernels/dataflow/welford_reader_mcast_sender_unary_sharded_gn_v2.cpp#L76-L101), [reader_mcast_sender_unary_gn.cpp:165-190](device/kernels/dataflow/reader_mcast_sender_unary_gn.cpp#L165-L190), [welford_reader_mcast_sender_unary_gn.cpp:92-117](device/kernels/dataflow/welford_reader_mcast_sender_unary_gn.cpp#L92-L117). Reach for the vararg mechanism — the count varies across instantiations, so there are no per-coordinate names to infer.

  **But name the scalars that precede the block.** Their legacy positions shift (7…16, 12…16, or absent) with `has_mcast_first_group` / `has_mcast_last_group`, yet each is a distinct field with a stable identity — `mcast_first_group_dest_noc_start_x` and friends. Metal 2.0 addresses named args in a section separate from the varargs, so the shifting legacy offset is irrelevant. What genuinely varies is whether the first-group and last-group field *sets* are populated, which the host already signals through the two booleans at positions 0 and 1.

- **Named compile-time args already exist in the two non-sharded factories** (`to_named_args_mcast` / `to_named_args_no_mcast`, read with `get_named_compile_time_arg_val("…")`), while the sharded factory is still on positional `get_compile_time_arg_val(N)`. Expect the naming work to be lopsided: mostly a pass-through for mcast / no-mcast, mostly fresh for sharded.

- **Config-dependent CB index aliasing in the compute kernels.** Several `dfb_*_id` constants are `constexpr` *expressions* over the compile-time args rather than fixed indices — `dfb_ex_global_id = num_cores_per_mcast_group == 1 ? dfb_ex_partial_id : c_15`, and the `dfb_outgamma_id` / `dfb_inbeta_id` / `dfb_outbeta_id` / `dfb_untilize_in_id` / `dfb_out_id` family that resolves differently under `UNTILIZE_OUT`, `FUSE_NEGATIVE_MASK`, `do_gamma` and `do_beta` ([groupnorm_sharded_v2.cpp:83](device/kernels/compute/groupnorm_sharded_v2.cpp#L83), `:89-144`; [groupnorm.cpp:168-199](device/kernels/compute/groupnorm.cpp#L168-L199)). Two named DFB handles can resolve to the same underlying buffer index in one config and to different ones in another. Work out the resolved index per build before assigning binding roles.
