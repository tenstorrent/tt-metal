# rms_norm_ttnn (ttnn.bringup.rms_norm)

Source: ai-generated (codegen, `dnijemcevic/rms_norm_replacement_run0` @ `31b0b1cbc98`); drop-in for `ttnn.rms_norm`.

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

6. opt-in split program: `stats_out`, `partial_stats` / `stats_width`, `input_window`, `dtype` (default off: none
   of them set = the default program, unchanged)
   - What: a second device operation in the fork (`device/rms_norm_split_device_operation.*`, kernels
     `kernels/rms_norm_split_{reader,compute,writer}.cpp`), routed to by the binding only when one of the new options
     is set; the default program's factory, kernels, CBs and hash are untouched. Work unit = (row tile, column block
     of the window); reader / compute / writer on the whole grid.
     - `stats_out=True`: the chip's partial sums of x^2 over the input's (window's) columns, [.., R, 32 B] float32
       TILE (B <= 2 column blocks, chosen by the op): x^2 on the FPU (fp32 DEST), packer L1-accumulated in fp32;
       EVERY element of the output is a partial (no junk columns, unlike rms_norm_pre_all_gather on fp32, K-12), so
       an all_gather of it over the chips holding the row's other columns is the `partial_stats` input.
     - `partial_stats=s, stats_width=H`: rs = rsqrt(row sum of all of s's elements / H + eps) (FPU row reduce with a
       1.0 scaler, then scale, eps and rsqrt on the SFPU in fp32 DEST); y = x * rs (FPU, column broadcast) * weight
       (SFPU in DEST, the weight unpacked with its row broadcast; fp32 through unpack-to-dest, exact), packed once.
     - neither (window / dtype only): the statistic of the window itself (pass A + pass B on one core per row tile).
     - `input_window=(c0, c1)`: only the input's columns [c0, c1) (multiples of 32) are read; the output and the
       weight are c1 - c0 wide. `dtype`: the output's dtype (default the input's).
     Takes TILE interleaved float32 / bfloat16 inputs, a [1, 1, 1, W] TILE weight, fp32_dest_acc_en=True; refuses
     bias / residual / program_config with these options.
   - Why: deepseek_v41_flash_d_p P.18 (precision-debt J4): its norms were hand-built fp32 SFPU chains (8-15 calls a
     norm, ~330 a chunk). A column-split (TP) norm is now rms_norm(stats_out) + all_gather of the partials +
     rms_norm(partial_stats=): 3 calls; q_norm / kv_norm read their columns of the fused [wq_a | wkv] output in place
     (input_window), kv_norm / k_norm write bf16 (dtype): 1 call.
   - Measured: probe ds41_norm_split/probe_002 (2x2 BH, HiFi4 + fp32 DEST, bf16-valued weights, vs float64): split
     norm [1024, 2560] per chip fp32 rows -> fp32 rel 6.1e-4, median row-norm ratio -3.5e-4; -> bf16 rel 1.77e-3 (one
     bf16 rounding 1.66e-3); bf16-valued rows -1.6e-4 (the default program: -1.9e-4 on the same fp32 rows). The first
     version applied the weight through an fp32 L1 intermediate on the FPU: ratio -1.4e-3 with an fp32 weight
     (-7e-4 after moving it into DEST). Time (20 synced calls): stats_out [1024, 2560] fp32 0.032 ms, all_gather of
     the [1024, 64] partials 0.040 ms, apply fp32 -> bf16 0.060 ms, -> fp32 0.071 ms; input_window q [1024, 1280] of
     [1024, 1792] fp32 0.045 ms (the default program on a sliced copy 0.043 ms + the slice). Model (2x2, chunk
     53248-55296, plain profile, NORM_IMPL off -> on): attn_norm 1.83 -> 0.89 ms a chunk, ffn_norm 2.11 -> 1.19, q_a
     4.76 -> 3.79, kv 0.81 -> 0.22, compressor 1.65 -> 1.37; programs (4 chips) 6140 -> 4580; device 96.91 -> 93.18 ms.
   - Tests: new `tests/unit/test_rms_norm_ttnn_split.py` (48 cases: stats_out row sums vs float64; a row split in
     two halves through stats_out + concat + partial_stats vs the whole-row float64 norm, with a wrong stats_width
     shown to fail the same check; input_window at the model's columns and an off-centre one; dtype alone;
     refusals). Option-off regression: unit suite 1375 passed / 47 skipped (= 1327 + the 48 new; skips unchanged);
     fork_source 263 tests, 0 regressions; model cases 10 passed / 2 skipped (the 4x2-mesh xing40 cases on a
     4-device box).
   - Needed by: deepseek_v41_flash_d_p P.18
   - Files: `device/rms_norm_split_device_operation.hpp`, `device/rms_norm_split_device_operation.cpp`,
     `kernels/rms_norm_split_reader.cpp`, `kernels/rms_norm_split_compute.cpp`, `kernels/rms_norm_split_writer.cpp`,
     `rms_norm_ttnn.cpp` / `.hpp` (`rms_norm_split_full`), `rms_norm_ttnn_nanobind.cpp` (the five keyword
     arguments), `sources.cmake`, `tests/unit/test_rms_norm_ttnn_split.py`

### Tests: hy4_preview_d_p cases
- What: appended the three random-input cases for the calls hy4_preview_d_p makes (2x2 mesh, 2560 rows per chip,
  widths 2048 / 512 / 6144, fp32 TILE input, fp32 ROW_MAJOR weight [1, 1, W/32, 32], HiFi4 + fp32 dest). The test
  draws host values in the captured dtype (fp32 stays fp32; BFLOAT16 cases keep their bf16 inputs). No op change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: hy4_preview_d_p O.1
- Files: `tests/cases.py`, `tests/test_rms_norm_ttnn.py`

### Known gap: memory_config under inplace (test skipped, 2026-09-30; fixed 2026-10-27, see the last entry)
- The C++ host path (`rms_norm_ttnn.cpp`) did not refuse a `memory_config` that differs from the input's placement
  when `program_config.inplace` is set. `test_refuses_memory_config_disagreeing_under_inplace` failed on it and was
  skipped. No model passes `inplace` with a disagreeing `memory_config`, so no bring-up result is affected.

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

### Port to the unified multicast helper (`kernel_lib/mcast/host/mcast.hpp`)
- What: llk_helper_library replaced the Mcast1D / Mcast2D / McastFamily host helpers and their wire format with one
  `Mcast` (receivers + group size + sender config; attach(descriptor, prefix, kernels, first_semaphore_id)). Every
  multicast site moved to it, in the C++ factory and the Python builder alike: line groups (old Mcast1D PerRow /
  PerColumn) are `line_mcast` (one group per row or column), one box (old Mcast2D) is `box_mcast` (one group, the
  root's row-major index as the fixed sender); the old `ack_count_override` (idle cores in a group's bounding box) is
  `handshake_cores` = the cores doing work; Diagonal sender placement is `Staggered`. The combine's per-kernel
  pre-handshake override has no equivalent, so `CombineMcast` builds the channel twice, with and without the
  handshake, and the handshake build owns the semaphores either way (ids and the gather id unchanged).
  `device/mcast_wire.hpp` still reads the wire back from a scratch attach and decodes the role word with the new
  `RuntimeLayout`; the Python builder computes the per-channel senders from the topology. Kernels unchanged (the
  device `McastArgs` API is the same).
- Why: the old helper no longer exists, so the fork did not build.
- Default behaviour: same topology, roles and semaphore ids; the wire format is the new helper's.
- Needed by: rebase onto origin/llk_helper_library @ `bdc576fa652` (2026-10-12)
- Files: `device/mcast_wire.hpp`, `device/rms_norm_ttnn_program_factory.cpp`, `rms_norm_ttnn_program_descriptor.py`,
  `kernels/rms_norm_ttnn_writer.cpp` (comment only)

### Tests: glm53_flash_d_p kv_a_layernorm case re-captured (O.1, 2026-10-13)
- GLM's [1, 1, 5120, 512] fp32 case (no longer called) replaced by the striped-latent [1, 1, 1280, 512] fp32 call (`tests/cases.py`). No op change.

### Tests: deepseek_v41_flash_d_p cases (O.1, 2026-10-23)
- What: 8 random-input cases for the calls deepseek_v41_flash_d_p makes (tt/norm.py, 2x2, HiFi4 + fp32 DEST, fp32
  weights, eps 1e-20: stats_out, partial_stats / stats_width 5120, input_window (0, 1280) / (1280, 1792), dtype bf16,
  the default program), run by test_rms_norm_ttnn_ds41: rel L2 within noise_limit (bf16 outputs 1.68-1.77e-3 vs
  3.3e-3; stats_out row sums 2.8e-8 vs 1.1e-7), determinism. No op change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: deepseek_v41_flash_d_p O.1
- Files: tests/cases.py, tests/test_rms_norm_ttnn.py

### Refuse a disagreeing memory_config under inplace; tests: no seed-parity cases, no run-time skips (2026-10-27)
- What: (a) under `program_config.inplace=True` the output is the input, so a `memory_config` naming another placement
  (memory layout, buffer type, or a shard spec that differs) is refused with a ValueError instead of being dropped
  silently. One that names the input's placement, or a sharded one without a shard spec, is accepted as before; calls
  without `inplace` or without `memory_config` are unchanged (same program). Both builders: `check_inplace_placement`
  in `rms_norm_ttnn.cpp` validate(), `_inplace_placement_agrees` in the Python `resolve_program_config`. The old note
  that the Python path already refused was wrong: it accepted and ignored the placement too (its X-03 comment said
  so). Unskipped `test_refuses_memory_config_disagreeing_under_inplace`. Upstream refuses this case too ("Output memory
  config must be sharded for inplace operation"); no model, model case or source test passes `inplace`.
  (b) `tests/unit/test_rms_norm_ttnn_perf.py`: removed the seed-parity cases (`test_seed_parity`,
  `test_sharded_seed_parity`, `test_program_is_structurally_the_seeds`). They compare against the generated
  `ttnn.operations.rms_norm`, which exists only on the branch this op was generated on, so all 39 seed cases always
  skipped here. Their 11 fork-only halves are kept as coverage: the operand-free interleaved row is now
  `test_operand_cost[no_gamma-*]`; the gamma rows were identical calls to `test_operand_cost[gamma-*]` and
  `test_sharded_operand_cost[gamma-*]`, which stay. 72 collected (33 run, 39 skipped) -> 25 collected, 25 run.
  (c) Parameter combinations that do not apply are no longer generated (they skipped at run time):
  `test_mixed_per_channel_format` bfloat8_b x ROW_MAJOR (no row-major form), `test_prime_width_is_correct_on_device`
  97 rows at W 2848 except gamma_bias_residual. Test ids of the valid combinations unchanged.
- Why: owner, 2026-10-27: a test is run and passing, not generated, or fixed; never a skip that hides a dead test or
  a bug.
- Tests: unit matrix 227 collected (225 passed, 1 skipped bf8b, 1 skipped inplace) -> 226 (225 passed; the inplace
  refusal test fails on the library built before this change, as it should; after the build (2026-10-27, 2x2): it passes; option-off regression: tests/unit 1368 passed, fork_source 251 passed + 12 upstream skips, 0 failing, model cases 18 passed + 2 skipped (xing40 4x2 needs 8 chips));
  ragged width 192 (186 passed, 6 skipped) -> 186 passed; perf 25 passed. The Python path's refusal checked by probe (DRAM memory_config refused; the input's own or
  none accepted, identity kept); cpp_parity inplace cases pass.
- Needed by: fork test hygiene (owner, 2026-10-27)
- Files: `rms_norm_ttnn.cpp`, `rms_norm_ttnn_program_descriptor.py`, `tests/unit/test_rms_norm_ttnn_matrix.py`,
  `tests/unit/test_rms_norm_ttnn_perf.py`, `tests/unit/test_rms_norm_ttnn_ragged_width.py`

### Rebase onto llk_helper_library 6b80daf43e3: kernel API port (2026-10-27)
- What: kernels: `perf_instrumentation.hpp` includes `api/debug/kernel_profiler.hpp` (#60172); the root fold's
  `add_tiles_init(CB_IN, CB_IN, true)` -> `add_init` with the same arguments (#57797); the 14 eltwise chain calls of
  `rms_norm_ttnn_compute.cpp` (residual add, square, pass B's scale / gamma / bias stages, the fp32-DEST per-channel
  window) pass `DataflowBuffer` objects to every CB-bound element (#60306), declared at each chain site so only CBs
  that program configures get one; `rms_norm_split_reader.cpp` includes `api/tile_constants.h` instead of
  `<tt-metalium/constants.hpp>` (#61991). Host: `line_mcast` / `box_mcast` and the planners take the input's
  `MeshDevice*` (`kh::Mcast` takes a `MeshDevice` now, #61011). No behaviour change.
- Why: the default and split programs' kernels no longer compiled; the C++ factory no longer built.
- Needed by: rebase 2026-10-27
- Files: `kernels/perf_instrumentation.hpp`, `kernels/rms_norm_ttnn_compute.cpp`, `kernels/rms_norm_split_reader.cpp`,
  `device/rms_norm_ttnn_program_factory.cpp`

### output_tensor: write a caller's output buffer (2026-10-30)
- What: `output_tensor=` (default None = the op allocates its output, unchanged). Given, the op writes that tensor, which must have exactly the output's spec (shape, dtype, layout, memory config; checked on every call, miss and hit) and must not be the input; default and split programs; not with `inplace`. The result is a tensor of its buffer. Kernels and programs unchanged; outputs byte-identical to the allocating call.
- Why: each call allocated its output (~35 us of host). The model passes load-time buffers at the call sites whose
  output is read and dropped inside the layer. Measured: rms_norm 65.9 -> 43.6 us of host a call (8x4, 208 calls a chunk, census of the profile rung's last chunk).
- Tests: `tests/unit/test_rms_norm_ttnn_output_tensor.py` 4 passed on the 8x4 (bytes equal to the allocating call, refusals). Source regression: 26 passed; the unit suite was stopped by the owner (F.1 runs it).
- Needed by: deepseek_v41_flash_d_p P.16 (FORK_OUT_BUFFERS, 2026-10-30)
- Files: `rms_norm_ttnn.{hpp,cpp}`, `rms_norm_ttnn_nanobind.cpp`, `device/rms_norm_ttnn_device_operation{.hpp,.cpp,_types.hpp}`, `device/rms_norm_split_device_operation.{hpp,cpp}`, `tests/unit/test_rms_norm_ttnn_output_tensor.py`

### max_grid: the op's cores inside a rectangle (deepseek_v41_flash_d_p FIX.15)
- What: option `max_grid` (ttnn.CoreCoord, default None) on `ttnn.bringup.rms_norm` (its default and split programs): every program of the op places its cores (and splits its work) inside (0, 0) .. (x - 1, y - 1) of the compute grid. Default: the device's whole grid, the call as before (same program, same hash).
- Why: agent rule 18 (owner 2026-10-30): a Galaxy chassis' power supply trips when a chip's current rises too fast (di/dt); no op may run on more than 110 of a chip's 120 Tensix cores. DeepSeek-V4.1 passes tt/cores.max_grid() = 11 x 10 on a Galaxy, None elsewhere.
- Measured: 8x4 Galaxy, one 5120-token chunk (rms_norm): 120 -> 110 cores a chip (tests/perf/test_core_audit.py in deepseek_v41_flash_d_p); per-op device time in that model's BREADCRUMBS FIX.15.
- Needed by: deepseek_v41_flash_d_p FIX.15
- Files: device/rms_norm_split_device_operation.cpp, device/rms_norm_split_device_operation.hpp, device/rms_norm_ttnn_device_operation.cpp, device/rms_norm_ttnn_device_operation.hpp, device/rms_norm_ttnn_device_operation_types.hpp, device/rms_norm_ttnn_program_factory.cpp, rms_norm_ttnn.cpp, rms_norm_ttnn.hpp, rms_norm_ttnn_nanobind.cpp

### pack_precise and bfp8_twin: GEMM-input outputs of the split program (deepseek_v41_flash_d_p P.21)
- What: two opt-in options of the split program (any of them selects it). `pack_precise=True` (default False): the
  compute config's bfp8_pack_precise, so a bfloat8_b output is packed from the fp32 DEST in one rounding (as
  ttnn.typecast packs); a kernel define `RMS_SPLIT_PACK_PRECISE` gives the binary its own JIT cache key
  (ComputeKernel's config hash leaves the flag out, DS41-137). `bfp8_twin=t` (default None): a preallocated bfloat8_b
  TILE interleaved tensor of the output's shape the op also writes: pass B rounds y to bf16 in DEST on the SFPU (ties
  away from zero, the packer's bf16 rounding) into the DEST half the gamma tiles used (DEST block halved when a twin is
  given without gamma), packs it to a bfloat8_b CB (c_8, precise) beside the bf16 output; the writer writes both at
  the same pages. Needs a bfloat16 output, no stats_out. Defaults: program, hash and kernels as before (the new compile
  args are 0 / the twin's accessor args empty).
- Why: DeepSeek-V4.1's norms whose only reader is a bfloat8_b GEMM write it directly (attn_norm -> q_a, q_norm -> wq_b
  with the softmax scale in its gamma: one rounding fp32 -> bfp8 instead of fp32 -> bf16 -> bfp8), and ffn_norm /
  q_norm write the MoE head's / the indexer's bfp8 x beside their bf16 output (the typecast it replaces, bit for bit).
- Tests: `tests/unit/test_rms_norm_ttnn_twin.py` 7 passed on the 8x4 Galaxy (FABRIC_2D, 11 x 10 grid): pack_precise
  = typecast(fp32 output, bfloat8_b) bit for bit on 3 shapes, the default pack differs (negative control); the twin =
  typecast(bf16 output, bfloat8_b) bit for bit and the bf16 output unchanged, column-split (partial_stats) and window
  cases, a program-cache hit with other inputs; 4 refusals. Fork regression suites not run (owner, OV-25: F.1).
- Cores: unchanged (max_grid; the model passes 11 x 10 on a Galaxy: 110).
- Needed by: deepseek_v41_flash_d_p P.21 (NORM_INTO_MOE, NORM_GEMM_DTYPE)
- Files: `device/rms_norm_split_device_operation.{hpp,cpp}`, `kernels/rms_norm_split_compute.cpp`,
  `kernels/rms_norm_split_writer.cpp`, `rms_norm_ttnn.{hpp,cpp}`, `rms_norm_ttnn_nanobind.cpp`,
  `tests/unit/test_rms_norm_ttnn_twin.py`
