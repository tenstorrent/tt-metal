# rms_norm_ttnn (ttnn.bringup.rms_norm)

1. ai-generated perf optimized version
2. adding a residual add where it was needed: `return_residual_sum=True` (with `residual_input_tensor`) also returns
   the residual sum `t = x + residual`, so the separate add can go (`residual_sum_memory_config` places it; default off)
3. bug fix, fp32 statistics at `fp32_dest_acc_en=True`. This changes the default output, so it is not behind an
   option: it is the one exception to "behind an option, default off". The `fp32_dest_acc_en=False` program is
   unchanged: same kernels, same CBs, same blocking.
   - What: (a) the cross-chunk sum of squares is carried in fp32. The compute kernel's `accumulate_reduce_block`
     passes `AccumulateReloadMode::CopySeedSfpuAdd` (a new kernel constant `CARRY_RELOAD`, `CopySeedPairs` at
     16-bit DEST). (b) `cb_x_squared` and `cb_normalized` are held in fp32 (`stat_dtype` / `_stat_dtype`), as
     native `ttnn.rms_norm` holds its intermediates, and the partial-width 0/1 mask in `cb_scaler` follows
     `cb_x_squared`'s format. `cb_x_sum`, the returned residual sum, keeps its format. The L1 solve prices the two
     CBs at their new tile size, so the blocking changes at fp32 DEST. MiMo's (5120, 4096) weight shape goes from
     3 chunks of 43 to 4 chunks of 32, with or without the residual. Every build was re-solved; none failed to
     fit.
   - Why: the default reload, `CopySeedPairs`, adds an odd chunk's leftover tile with a DEST_TO_SRCB reuse add.
     That moves the fp32 carry into srcB, which is programmed to the bf16 input format, so every carry was
     truncated to bf16. Separately, x^2 was rounded to bf16 on the pack, and y was rounded twice. Measured on
     MiMo's golden layer 1, chunk 1 input with its attn_norm weight, as bringup vs float64, row-norm ratio / rel L2:
     1.00090 / 0.00248 before, 0.99968 / 0.00197 after; native was 0.99945 / 0.00205. With a unit weight the
     scale bias was +9.3e-4 and is now +1.7e-4; with no weight it was -1.6e-4 and is now +0.7e-4. The residue is
     the carry's copy_tile reload through srcA, which keeps only tf32 precision. It grows by about 0.65e-4 per
     chunk. Removing it needs an UnpackToDestFp32 accumulator, and cb_row_stat cannot be one because it is also
     an FPU operand.
   - Model cases: the MiMo norm case goes from pcc 0.9999972 / max abs 0.063 to 0.9999985 / 0.041. The
     residual-sum case goes from max rel 0.0112 to 0.0091. Both pass at their recorded limits.
   - Perf, at MiMo's shape: norm 0.215 -> 0.219 ms, fused residual 0.414 -> 0.417 ms.
   - Needed by: mimo_v2_6_d_p. The scale bias alone pushed test_swap_sliding_moe_02_attention over its ratio limit.
   - Files: `kernels/rms_norm_ttnn_compute.cpp`, `device/rms_norm_ttnn_program_factory.cpp`,
     `rms_norm_ttnn_program_descriptor.py`, and the new test `tests/unit/test_rms_norm_ttnn_fp32_stats.py`. That
     test covers both builders' CB formats and the scale bias across 2-7 chunks on device.
4. fp32 `cb_x_squared` only where it is an accumulator; `cb_normalized` back to the intermediate format (refines
   3). Changes the `fp32_dest_acc_en=True` program only; the `fp32_dest_acc_en=False` program is unchanged (same
   CBs and blocking, checked on 36 shape/operand builds on both builders).
   - What: the rule is that fp32_dest_acc_en keeps DEST 32-bit while it accumulates, and a value packed to L1 stays
     in the intermediate format unless that L1 value is itself an accumulator. (a) `cb_x_squared` is FLOAT32 only
     when the DEST fold is on (`x_squared_wt < wt_chunk`: each tile is a sum of several squares). Without the fold
     (a partial last tile, or a chunk with no divisor 2..SQ_FOLD_GROUP, e.g. 43) it holds single x^2 tiles in
     `intermediate_dtype` (`sq_dtype_of` / `_sq_dtype`). (b) `cb_normalized` is `intermediate_dtype`. (c) The
     partial-width mask follows `cb_x_squared`; a partial width never folds, so it is `intermediate_dtype`. (d)
     `CB_SQ_EXACT = true`, applied at fp32 DEST in all three solves (RESIDENT, ROW_RESIDENT, STREAM): each
     candidate chunk is priced at `cb_x_squared`'s real width and at the format that chunk would get, and
     `cb_normalized` at the intermediate tile size. At 16-bit DEST the price stays conservative (the exact price
     measured 0.987x there, see test_rms_norm_ttnn_dataflow_knobs.py). The CopySeedSfpuAdd carry from 3 is kept.
     The program-cache hash needs nothing new: the format is a function of the hashed shapes, dtypes and config.
   - Why: owner review of 3. Numbers at MiMo's (5120, 4096) bf16 weight shape, fp32 DEST, HiFi4:
     - Blocking: norm 32x4 -> 64x2 (x^2 folded in groups of 16, fp32); fused residual 32x4 -> 43x3 (prime chunk, no
       fold, bf16 x^2). L1 per core: norm `cb_x_squared` 8 -> 16 kB, `cb_normalized` 128 kB fp32 -> 128 kB bf16
       (64 tiles), total CBs 1042 -> 1178 kB; residual `cb_x_squared` 8 -> 86 kB, `cb_normalized` 128 -> 86 kB,
       total 1042 -> 1214 kB. The solve and the allocation agree to the byte.
     - Perf (1x4 mesh, 20 calls, device max): norm 0.219 -> 0.211 ms, fused residual 0.417 -> 0.413 ms.
     - MiMo golden layer 1, chunk 1 input vs float64, row-norm ratio / rel L2: with the attn_norm weight 0.99968 /
       0.00197 -> 1.00006 / 0.00232 (native 0.99945 / 0.00205, before 3 1.00090 / 0.00248); unit weight 1.00017 /
       0.00168 -> 1.00007 / 0.00167; no weight 1.00007 / 0.00167 -> 0.99999 / 0.00167. The scale bias is smaller;
       the rel L2 with a weight is back up, because y is rounded to bf16 in `cb_normalized` and again on the output.
       Random inputs with a random weight (test_precision_against_float64) show the same: rel L2 0.0018 -> 0.0024
       without a residual, 0.0024 -> 0.0029 with one; no weight unchanged (0.0017).
     - Models: MiMo 1x4 and 2x2 ladder rung last unchanged (1x4 L00-L05 0.998577 / 0.998451 / 0.998422 / 0.998510
       / 0.998787 / 0.998421, state_min 0.999281); 50k->55k chunk device 204.9 -> 204.9 ms (1x4), 249.9 -> 250.0 ms
       (2x2). Model cases: norm pcc 0.9999985 -> 0.9999972, residual-sum max rel 0.0091 -> 0.0109; both pass.
       Norm component tests and swap tests: all pass on 2x2; on 1x4 all pass except sliding_moe swaps 03-07, which
       fail on the sliding attention's per-token norm ratio upper bound 1.05 before this change too (3: [0.9742,
       1.0657], now [0.9553, 1.0691]; swap 02, whose limit is 1.08, passes).
   - Needed by: owner review of change 3.
   - Files: `device/rms_norm_ttnn_program_factory.cpp`, `rms_norm_ttnn_program_descriptor.py`,
     `tests/unit/test_rms_norm_ttnn_fp32_stats.py` (the format test for the new rule, a float64 precision test,
     scale-bias cases re-picked where the chunking moved), `tests/unit/test_rms_norm_ttnn_dataflow_knobs.py`.
5. bug fix, the weight and bias are applied in DEST at `fp32_dest_acc_en=True`, so y is rounded once (refines 4).
   Changes the default `fp32_dest_acc_en=True` program for calls with a weight or bias, so it is not behind an
   option (like 3). The `fp32_dest_acc_en=False` program is unchanged (same kernels' code path, CBs and blocking;
   the 16-bit-DEST outputs of every case in the precision probe are identical).
   - What: pass B is one DEST window when `PC_IN_DEST = DST_ACCUM_MODE && (HAS_G || HAS_B)`: x * (1/rms) on the FPU
     (the same `BinaryFpu<Mul, Col>` as before) into D0; gamma and bias unpacked with their Row broadcast into D1 /
     D2 (`UnaryBcast`, so only the L1 per-channel values pass through a source register); then `PcRowBcast`, an
     SFPU body that computes D0 = D0 * gamma + bias in fp32 and packs y once. `PcRowBcast` is not the stock
     `MulBinary` / `AddBinary`: a Row-broadcast tile's vector k in a face equals vector k & 1 (column parity is the
     SFPU's inner walk axis), so it loads the two operand vectors once per face into LREGs, reads one DEST vector
     per output vector instead of two, and does gamma and bias in one SFPMAD. It covers every mode the fork has:
     gamma, bias, both, with or without a residual (and `return_residual_sum`), TILE and ROW_MAJOR, all four
     placements, the cross-core combine (which no longer takes the D44 gamma-first order at fp32 DEST), partial
     last tile, multi-row blocks, bf16 / fp32 / bfp8 inputs and mixed per-channel formats. No mode keeps
     `cb_normalized` at fp32 DEST. The DEST block is `PASS_B_PC_BLK`, the largest divisor of WT_CHUNK within
     `DEST_AUTO_LIMIT / (1 + gamma + bias)`. `cb_normalized` is not allocated and the L1 solve does not price it
     (`norm_cb_depth` / `_norm_cb_depth` take `pc_in_dest`, the host mirror of the kernel predicate, on both
     builders). The program-cache hash needs nothing new: the predicate is a function of the hashed config.
   - Candidates, measured vs float64: (a) the stock SFPU `MulBinary` / `AddBinary` after `UnaryBcast`: rel L2 0.00168
     (random inputs, MiMo's shape), norm 0.233 ms; (b) `PcRowBcast`: 0.00168, 0.230 ms, taken as the most accurate
     and the faster of the two; (c) an FPU multiply with DEST reuse: rejected. `DestReuseBinary` has no broadcast
     (gamma needs Row), so it needs raw LLK, and it moves the fp32 DEST value into srcA at the source register's
     format: emulated on the same inputs rel L2 0.0037 at bf16, 0.0017 with a -3.4e-4 scale bias at truncated
     tf32. Neither DEST-side candidate is as fast as CHANGELOG 4's program: the SFPU pass is MATH-thread work
     (about 90 cycles per tile, measured by ablation) where the old second stage was an unpack/pack traversal that
     overlapped. An fp16 `cb_normalized` would round as little (emulated 0.00169) at CHANGELOG 4's speed, but keeps
     the CB; not taken, since the point of the change is to drop it.
   - Why: CHANGELOG 4 put `cb_normalized` back in bf16, so y was rounded twice with a weight. MiMo golden layer 1,
     chunk 1 input with its attn_norm weight, vs float64, rel L2 / row-norm ratio: 0.00232 / 1.00006 before, now
     0.00188 / 1.00015 (the bf16 rounding floor of that output is 0.00186; native 0.00205 / 0.99945; CHANGELOG 3
     0.00197 / 0.99968). Unit weight and no weight unchanged (0.00167). Random inputs with a random weight
     (test_precision_against_float64 / the precision probe): 0.00236 -> 0.00168 without a residual, 0.00287 ->
     0.00235 with one (the rest is t's own bf16 rounding; native 0.00191 / 0.00200).
   - L1 and blocking at MiMo's (5120, 4096) bf16 weight shape, fp32 DEST: blocking unchanged (norm 64 x 2, fused
     residual 43 x 3); CBs per core norm 1206272 -> 1075200 B, residual 1243136 -> 1155072 B. Other shapes re-solve
     to wider chunks where the freed L1 allows (2048x5120 40 x 4 -> 54 x 3, 2048x4022 42 x 3 -> 63 x 2).
   - Perf (1x4 mesh, 20 calls, device max): norm 0.211 -> 0.230 ms, fused residual 0.413 -> 0.429 ms.
   - Models: MiMo 1x4 and 2x2 chunk device time unchanged, 204.9 -> 204.9 ms (1x4), 250.0 -> 249.9 ms (2x2); ladder
     rung last 1x4 L00-L05 0.998575 / 0.998449 / 0.998425 / 0.998511 / 0.998791 / 0.998424, state_min 0.999281
     (2x2 worst layer 0.998414, state 0.999280). Norm component tests: rel L2 0.0028-0.0034 -> 0.0010-0.0029. All
     swap tests pass on both meshes, including 1x4 sliding_moe swaps 03-07 that failed before on the sliding
     attention's per-token norm ratio upper bound 1.05: [0.9553, 1.0691] -> [0.9627, 1.0461]. Model cases: norm
     pcc 0.9999972 -> 0.9999986, residual-sum max rel 0.0109 -> 0.0086 (limit 0.012).
   - Needed by: owner review of change 4.
   - Files: `kernels/rms_norm_ttnn_compute.cpp`, `device/rms_norm_ttnn_program_factory.cpp`,
     `rms_norm_ttnn_program_descriptor.py`, the new `tests/unit/test_rms_norm_ttnn_pc_in_dest.py` (CB presence on
     both builders; y vs float64 for every mode above at both DEST widths, limit 0.0021 at fp32 DEST with bf16,
     which the double-rounded program fails; mixed per-channel formats; a check that the cases reach multi-row
     blocks and several chunks), `tests/unit/test_rms_norm_ttnn_fp32_stats.py` (no `cb_normalized` at fp32 DEST,
     precision limits tightened to 0.0021 / 0.0028).

### Tests: hy4_preview_d_p cases
- What: appended the three random-input cases for the calls hy4_preview_d_p makes (2x2 mesh, 2560 rows per chip,
  widths 2048 / 512 / 6144, fp32 TILE input, fp32 ROW_MAJOR weight [1, 1, W/32, 32], HiFi4 + fp32 dest). The test
  draws host values in the captured dtype (fp32 stays fp32; BFLOAT16 cases keep their bf16 inputs). No op change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: hy4_preview_d_p O.1
- Files: `tests/cases.py`, `tests/test_rms_norm_ttnn.py`

### Known gap: memory_config under inplace (test skipped, 2026-09-30)
- The C++ host path (`rms_norm_ttnn.cpp`) does not refuse a `memory_config` that differs from the input's placement
  when `program_config.inplace` is set; the Python path refuses it. `test_refuses_memory_config_disagreeing_under_inplace`
  failed on it and is skipped until `validate()` in `rms_norm_ttnn.cpp` gets the same check. No model passes
  `inplace` with a disagreeing `memory_config`, so no bring-up result is affected.

### Port to the new multicast helpers (#57547; kernel helper library @ `73027b6e6ff`, 2026-10-01)
- What: `kernel_lib/host/mcast_host.hpp` and `kernel_lib/mcast_pipe.hpp` were replaced by
  `kernel_lib/mcast/{host,kernel}/` with a new wire (CT block 11 words with a FAMILY / ABSENT tag, RT block with
  sender coords + rectangle records + a role word) and attach()-only host queries. Host and kernels move together:
  - host (C++ factory and Python builder): `device/mcast_wire.hpp` / `_McastWire` attach each family to a scratch
    data-movement kernel and read the blocks back, so the wire still sits at the same positional bases
    (writer `McastArgs<18, 12>`, reader `McastArgs<PC_CT + 1, 13>`); constructors use `Mcast1DFixedSenderConfig`
    (Uniform / Diagonal) / `Mcast2DFixedSenderConfig` / `Mcast2DRotatingSenderConfig`; the old Mcast2D `num_active`
    is `McastConfig::ack_count_override`; `is_sender` reads the role word. The old per-kernel `pre_handshake` override
    (single-round combine) is kept by setting / clearing the wire's PRE_HANDSHAKE flag: semaphore ids unchanged.
    A non-combine writer gets the one-word ABSENT block (was 6 zeros).
  - kernels: pipes are `optional_sender` / `optional_receiver` (empty off-role or absent); the root's coordinates come
    through small templates so the combine-only code compiles with an absent block. The pipe no longer counts
    Counter signals itself (`receive(round)` waits for round + 1), so the per-channel receive passes the absolute
    signal count; a rotating Counter sender now counts its own send, so the split path waits for the total.
  - tests: the writer's pre-handshake flag word moved from CT 22 to 23 (combine_knobs / perf tests).
- Default behaviour: unchanged (same transfers, same semaphores). Tests (2026-10-01): unit 1327 passed / 47 skipped
  (incl. C++ vs Python program parity and bit-identical outputs); model cases 12 passed (4x2 box, 1x4 on cards
  0-3, 2x2 on cards 0,1,4,5). Known: the unshipped Python `PC_MCAST_MODE="split"` knob uses the rotating Counter
  sender, whose self-count is a non-atomic local `+=` racing remote increments (see mhc_pre_ttnn change 5); not
  used by any plan.
- Needed by: rebase onto origin/malimpic/llk_helper_library_rebased_0110_2 (the fork no longer compiled)
- Files: `device/mcast_wire.hpp` (new), `device/rms_norm_ttnn_program_factory.cpp`, `rms_norm_ttnn_program_descriptor.py`,
  `kernels/rms_norm_ttnn_reader.cpp`, `kernels/rms_norm_ttnn_writer.cpp`, `tests/unit/test_rms_norm_ttnn_combine_knobs.py`,
  `tests/unit/test_rms_norm_ttnn_perf.py`
