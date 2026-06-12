# Port Report — nlp_concat_heads_decode

Metal 2.0 port of `nlp_concat_heads_decode` (both factories). Tests pass on Blackhole p150b.

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept` for both `NLPConcatHeadsDecodeProgramFactory` and `NLPConcatHeadsDecodeSubcoregridsProgramFactory`. Single-program, no op-owned device resources. `select_program_factory` unchanged; the framework dispatches each factory per `on_subcoregrids`.

### Device-op-class edits
- Custom `compute_program_hash` deleted: none (op used the default reflection hash).
- Pybind entry points removed: none (no `create_descriptor` was pybound).

### Open items
- Strict tensor matching kept (default). No relaxation candidates noted.

## Handoff points

- **Case-2 `input` binding (TensorAccessor enhancement candidate).** The input shard is read by an exotic per-core NoC walk (`{.noc_x, .noc_y, .addr}` unicast, sub-tile `SUBTILE_LINE_BYTES`/face granularity) that `TensorAccessor` page iteration cannot express. The port re-expresses the binding as a `TensorParameter` and recovers the base kernel-side via `TensorAccessor::get_bank_base_address()` (sanctioned Case-2 bridge), leaving the NoC arithmetic intact. A future `TensorAccessor` iteration mode for per-core sub-tile-line gather would let this become a clean Case 1. (Audit surfaced the verbatim user-facing note; user directed the port.)

## Successes

- **Borrowed-memory + fake-CB output (`port_op_to_metal2_recipe.md` / patterns catalog: Fake CB → self-loop, Borrowed-memory DFB).** Legacy CB `c_16` (`.buffer = output.buffer()`, write-only via `get_write_ptr()`, no FIFO) ported cleanly as `DataflowBufferSpec{ borrowed_from = OUTPUT }`. Since the two kernels (reader = phase 1, writer = phase 2) share the output and run on the same nodes, the validator's producer-and-consumer rule is satisfied by binding `reader = PRODUCER` / `writer = CONSUMER` rather than a same-kernel self-loop (which would have created two producers per node). Tests pass — the split is correct.
- **Common runtime varargs (`metal2 advanced_options`).** The NoC coordinate arrays (`in0_mcast_noc_x/y`, read via `get_arg_addr` + runtime loop index in legacy) map cleanly to `num_common_runtime_varargs` + `get_common_vararg(i)`, since the coords are identical across all output cores. Layout `[x.., y..]`; kernel reads `get_common_vararg(idx)` and `get_common_vararg(num_x + idx)`.

## Friction

- **Gap — fake-CB across two co-located kernels.** The patterns catalog's fake-CB → self-loop entry describes binding PRODUCER+CONSUMER on *one* reading kernel. Here two kernels (reader/writer) both write the output on the same nodes; a same-kernel self-loop on each would violate the DFB "one producer per node" invariant. The PRODUCER-on-reader / CONSUMER-on-writer split is the right shape but isn't spelled out in the catalog — worth a sub-note for multi-kernel fake CBs.
- **Confusion — build granularity.** A partial `cmake --build --target ttnncpp` left the kernel JIT path inconsistent: kernels failed JIT compile with `'dfb' has not been declared` / `'get_common_vararg' was not declared` even though the host `.so` built fine. A full `./build_metal.sh` (install) build resolved it. The recipe's [workspace doc] mentions `--build-tests` but could call out that the *full install* build is required for the generated-header injection to be consistent.
- **Doc vs API: `get_bank_base_address` signature.** The recipe shows `input.get_bank_base_address(bank_id)` (with an argument); the actual kernel API is no-arg `get_bank_base_address()` returning the buffer base. For a sharded L1 buffer this is the per-shard local address (identical across shard cores), which is exactly the legacy `q_start_addr`. Tests confirm correctness.

## Open items for downstream

- **Fake-CB self-loop binding (interim hack — flag prominently).** Output DFB `q_out` (`borrowed_from = OUTPUT`) in both factories is a **tensor-local-view** fake CB (write-only address source; no real FIFO). It is bound `reader = PRODUCER` / `writer = CONSUMER` purely to satisfy the validator. Sites: `nlp_concat_heads_decode_program_factory.cpp` and `nlp_concat_heads_decode_subcoregrids_program_factory.cpp` (the `out_dfb_spec` + reader/writer `DFBBinding`s). The eventual "local `TensorAccessor`" migration should replace this.
- **Cross-op kernel touches:** none.
- **Test coverage:** `test_nlp_concat_heads_decode.py` covers both factories (`test_concat_head`, `test_concat_head_subcoregrids`) across head/batch/subcore-grid sweeps — all pass post-port on Blackhole p150b.
