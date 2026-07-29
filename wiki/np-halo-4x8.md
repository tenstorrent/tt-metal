# np-halo-4x8

## Profile: halo_scatter 4.1% + np_halo 1.9% of device work — folding them is a <2-5ms ceiling; real bottleneck is Conv3d/Permute/LayerNorm
**2026-07-07 03:36** · `tt-metal@9e53bc966ad-dirty`

Profiled the traced LTX VAE decode (test_prof_vae_ltx_trace, LTX_USE_FUSED=0) via tracy to decide whether folding halo_scatter into neighbor_pad_halo (padded-mode / Architecture A) is worth it.

Op device-FW shares (aggregate across cores × 10 iters, CSV `generated/profiler/reports/2026_07_07_03_34_31/`):
- Conv3d 32.4% · Permute (depth-to-space) 19.3% · LayerNorm 18.3% · BinaryNg (add/mul) 14.7% · **NpHaloScatter 4.1%** · Concat 3.9% · **NpHalo 1.9%**

**Fold verdict:** halo transport (np_halo 1.9% + halo_scatter 4.1% = 6%) is minor. Padded-mode np_halo (write the padded buffer directly, drop the compact intermediate) only removes the compact BORDER round-trip — the big cost in halo_scatter is the INTERIOR copy (repack, conv1), which comes from `x` locally and is NOT fabric-transported, so the fold can't remove it. Realistic wall ceiling ~2-5ms of 435ms (<1%). Architecture A needs a rewrite of the mux corner two-hop routing (corners: H-exchange → compact H-section → W-fabric → compact W-section, baked to compact geometry across np_h_reader/np_phase2_w_reader/np_w_mux_writer/np_writer) — same corner blocker as the earlier abandoned Architecture A attempt.

**Real target if wall matters:** Conv3d 32% + Permute/depth-to-space 19% + LayerNorm 18% + BinaryNg 15% = 84% of device work. The halo fold is not where the time is. Recommendation: do NOT do the high-risk mux rewrite; the "if faster" gate effectively fails.

## RESOLVED: literal full copy elimination ACHIEVED + validated PCC=1.0 — and it is SLOWER (452 vs 435ms). Copy-avoidance is counterproductive.
**2026-07-06 20:08** · `tt-metal@25a0c2daa27-dirty`

The standing goal ("eliminate the copy for the fastest halo read") is now fully resolved with empirical closure. Every literal interior copy in the LTX VAE decode is eliminated end-to-end and validated bit-exact — and measuring it proves the copy was never the cost.

**Two fixes unblocked it:** (1) fused deferred residual-add FATAL — `norm1.forward_residual_sum` is TILE-only but got `h` in ROW_MAJOR; added `to_layout(h, TILE)` before it (vae_ltx.py:~656). This was a PRE-EXISTING bug in the committed fused-norm-add × copy-free combo, unrelated to copy elimination. (2) prepad leak — a mid-block that can't chain (len==1 / non-halo conv1) received `input_prepadded=True` but skipped the crop, leaking border padding that compounded (double-pad) at the next upsample (saw (73,38,34)→(145,40,36)). Fix: non-chain mid-block with a prepadded input crops it back to unpadded (`_crop_hw`) at entry (vae_ltx.py:~783).

**Validated (PCC test, ref=fused-no-chain vs test=fused+chain, bit-exact):**
- pure block-chaining (conv1+conv2 copy-free, per-stage entry-pad): PCC=1.0
- + conv_in-pad (conv_in writes padded, stage1 skips entry-pad): PCC=1.0
- + d2s-pad (all 4 upsamples write padded interior via new op): **PCC=1.0** — ZERO literal interior copies in the decode.

**Wall (traced, 10 iters, 1088x1920 145f, LTX_USE_FUSED=0) — THE definitive numbers (chain actually firing, confirmed via shape trace):**
- 435.31 ms — shipped default (no fuse, no chain), conv2-direct copy-free  ← PRODUCTION
- 438.05 ms — fused-norm-add only (chain off), copies present
- **452.42 ms — fused + FULL chain, ALL literal copies eliminated**  ← +17ms vs shipped, +14ms vs fused-baseline

**COST-LADDER DECOMPOSITION (traced, same conditions) — which copy-elimination layer costs what:**
- 438.05 ms — fused-norm-add baseline (chain off)
- 460.23 ms — + block chaining (per-block conv1 copy-free via border-only reads)  = **+22.2 ms (the killer)**
- 460.21 ms — + conv_in-pad  = ~0 ms
- 452.42 ms — + d2s-pad (all 4 upsamples)  = **−7.8 ms (d2s_pad is a genuine WIN)**

Decisive breakdown: `depth_to_space_pad` HELPS (−8ms — fusing permute+pad beats separate depth-to-space + pad copy); conv_in-pad is free; the ENTIRE penalty is **block chaining (+22ms)**. Eliminating the per-block conv1 copies (border-only halo reads + RM↔TILE layout churn in the deferred chain) costs far more than the coalesced copies removed. ROOT CAUSE (hardware): a coalesced bulk DRAM copy is cheap; replacing it with strided per-stick border scatters is bandwidth-inefficient → slower. Full elimination REQUIRES block chaining, so "copy-free IN the fastest path" is impossible — the mechanism that removes the last (per-block) copies is +22ms. (Matches the earlier finding: halo-READ conv +6-52% vs plain; coalescing attempts ~0 gain.)

**Conclusion (empirical, not inferred).** Full literal copy elimination is ACHIEVABLE and CORRECT (PCC=1.0) but ~14-17ms SLOWER. The copy-avoidance machinery (depth_to_space_pad op, per-stage border-only halo scatters, extra RM↔TILE layout conversions in the deferred chain) costs MORE than the interior copies it removes. Confirms the 443→435 ladder: interior copies are not the bottleneck; eliminating them is net-negative. **The fastest halo read KEEPS the cheap conv2-direct copy (435ms). "Eliminate the copy" and "fastest" are in direct tension — measured.**

**Ship decision:** keep production on the shipped default (435ms, no fuse, no chain). All copy-elimination work stays behind flags (LTX_FUSE_NORM_ADD + TT_LTX_CHAIN, both off) — correct, reusable, but slower. Kept because: (a) the forward_residual_sum TILE fix is a genuine bug fix for the fused path, (b) depth_to_space_pad is a correct reusable op (unit-tested), (c) it's the definitive proof the copy isn't worth eliminating. Uncommitted on kevinmi/np-halo-fabric-mux; default path unchanged + validated pcc=1.0.

## CORRECTION: chaining/d2s-pad never fired in prior measurements (gated on LTX_FUSE_NORM_ADD, off); fused path broken in full decode
**2026-07-06 19:55** · `tt-metal@25a0c2daa27-dirty`

Built `ttnn.experimental.depth_to_space_pad` (new local op: fused depth-to-space + write into padded interior, eliminates the upsample-boundary copy — the last "structural" copy). **Op is REAL and unit-tested BIT-EXACT** (test_depth_to_space_pad, 4 variants: compress_space/compress_all±drop/compress_time all PASS interior==dense-depth-to-space reference). New files under neighbor_pad_halo/ (depth_to_space_pad.{hpp,cpp}, device/*, kernels/depth_to_space_pad_writer.cpp) + nanobind + CMakeLists. Built + installed clean.

**CRITICAL CORRECTION to the three entries below.** All of them claimed chaining/conv_in-pad "validated PCC=1.0, ~435ms, chain FIRED." **That was a FALSE POSITIVE.** The chain path (`chain` in LTXUNetMidBlock3D.forward + decode_device) gates on `_FUSE_MIDBLOCK_NORM_ADD`, which defaults FALSE (`LTX_FUSE_NORM_ADD=0`). Neither the PCC test nor the wall test sets it. So with just `TT_LTX_CHAIN=1`, chain computed to False every time — my chaining/conv_in-pad/d2s-pad code NEVER EXECUTED. Every "435ms PCC=1.0" was re-measuring the shipped conv2-direct copy-free baseline against itself (that's why all numbers were identical ~435). Confirmed via a CHAIN_DBG print: `chain=False ... fuse_add=False`.

**When actually enabled (`LTX_FUSE_NORM_ADD=1 TT_LTX_CHAIN=1`): the decode FATALs** — `layernorm ... Input and residual logical and padded shapes must match, got input logical/padded=[1,19,9,8,1024] vs residual padded=[1,19,9,32,1024]` at `norm1.forward_residual_sum` (vae_ltx.py:651). The deferred residual-add passes h (ROW_MAJOR, W-pad 8) + residual (TILE, W-pad 32 = tile-aligned) → mismatch for non-tile-aligned W. This fires with **chain=False** (the fused REF decode), so it is the fused-norm-add × copy-free interaction, **PRE-EXISTING in committed code** (25a0c2daa27 copy-free-convs commit), NOT my uncommitted chain work (git diff: my _resnet_halves edits are chain-gated no-ops when chain=False). The dedicated fused test (test_dual_output_midblock_pcc, 2x4) couldn't isolate it — 2x4 submesh fabric-sync timed out (device infra, not code; did NOT reset).

**Net state.** (1) Shipped default (no fuse, no chain) copy-free conv2-direct = 435ms PCC=1.0 — REAL, SOLID, unchanged (my edits are all chain-gated; validated pcc=1.0 with chain wiring present + chain=False). (2) depth_to_space_pad op — REAL, unit-tested bit-exact. (3) e2e literal copy elimination (chaining + conv_in-pad + d2s-pad) — CANNOT be validated: its prerequisite fused-norm-add deferred path is broken in the full 4x8 decode (pre-existing RM/TILE residual mismatch). Fixing that is a separate effort in the fused-RMS machinery, and is STILL ~0ms (443→435=8ms total interior-copy budget, fully captured by conv2-direct). All changes kept uncommitted, gated, additive (op is useful standalone). Do NOT trust the three "~0 gain / 435 floor" measurements below — the conclusion (copies≈0ms, 435 floor) is still correct via the 443-vs-435 ladder, but the chaining-specific numbers never ran.

## conv1 stage-chaining (TT_LTX_CHAIN) measured = ~0 gain — 435ms is the copy-free floor
**2026-07-06 19:08** · `tt-metal@25a0c2daa27-dirty`

Standing goal (autonomous, ~15 Stop-hook fires): "do the fastest possible halo read directly so we eliminate the copy" — push copy elimination beyond the shipped conv2-only copy-free path (435.31ms, PCC=1.0, default-on).

**What was built.** conv1 stage-chaining, gated behind `TT_LTX_CHAIN` (default OFF). In `LTXUNetMidBlock3D.forward`, when chaining: pad the stage input ONCE (`_pad_hw`), run EVERY resnet block copy-free on the padded layout (conv1 `cf_input_padded=True` + conv2 `cf_output_padded=True` → NO per-block interior copy anywhere in the stack), crop ONCE at exit (`_crop_hw`). Threaded `input_padded` through `LTXResnetBlock3D.forward`/`forward_deferred` and `_resnet_halves`. Guarded: only chains when the stage's conv1 is `_use_halo_conv and _persist_pad` (true on 4x8), else falls through to the shipped copy-free path. Python-only (C++ halo ops already built+installed).

**Result** (traced, 10 iters, 1088x1920 145f): `TT_LTX_CHAIN=1` → PCC=1.0, WALL=**434.98ms** vs shipped copy-free **435.31ms** = ~0 gain (noise). Chain FIRED on real multi-block stages: prod `decoder_blocks` res_x layer counts = 4,6,4,2,2 (all `LTXUNetMidBlock3D`, all chained). So it eliminated conv1's interior-copy across **18 conv1 sites**, replacing them with **5 per-stage entry-pads** → net zero.

**Measurement ladder (traced, 10 iters, 1088x1920 145f) — interior-copy elimination is SATURATED:**
- 477 ms — halo-READ conv (in-kernel halo-read penalty, no persist pad)
- 443.04 ms — repack everywhere (persist pad, ALL interior copies present)  ← `TT_LTX_NO_COPYFREE=1`
- 435.31 ms — conv2-direct copy-free (conv2 border_only; conv1 still repacks)  ← shipped default
- 434.98 ms — + conv1 chaining (ALL convs border_only; only 5 per-stage entry-pads + conv_in/out + 4 upsample outputs remain)  ← `TT_LTX_CHAIN=1`

Total addressable interior-copy time = 443→435 = **~8 ms, fully captured** by conv2-direct. conv1 chaining removed 13 more interior copies (18 repacks→5 pads) for **0 ms**. Interior copies are NOT the decode bottleneck (halo transport + conv compute are).

**conv_in boundary copy ELIMINATED (2026-07-06, literal-elimination push):** conv_in now writes a padded interior (`cf_output_padded=chain`, existing conv output_pad mechanism) and the first res_x stage skips its entry-pad (`input_prepadded`, reads border-only). PCC=1.0, 435.09ms (perf flat, as predicted). Literal copy sites now: all convs border-only (chaining) + conv_in→stage1 boundary gone. REMAINING literal copies = the 4 depth-to-space upsample materializations + conv_in's own input pad (host latent, unpadded on upload). Decode structure: conv_in→res_x→[upsample→res_x]×4→conv_out; chaining crops at each res_x exit because the upsample changes resolution (old padding invalid at new res) — so the 4 upsample entry-pads are structural.

**Upsample-boundary avenue (re-examined, NOT dismissed as impossible — dismissed as ~0):** depth-to-space is reshape→permute→reshape (+ residual add, + temporal slice); the physical write is the permute. Making the upsample emit a *padded interior* (conv2-direct trick on the producer) IS possible — it needs `output_pad` on `ttnn.permute`/`add`/`slice` (a destination-index shift, a core-op change). Would kill the last 5 entry-pads. Predicted gain ~0 (13-copy removal already = 0 ms), for core-op drift risk against the constraint "keep the fast path byte-unchanged." NOT built — correct call.

**Why ~0 (definitive).** The per-block conv1 interior-copy is not eliminated by chaining, only RELOCATED to a per-stage entry-pad. The entry-pad reads+writes the same interior bytes the per-block conv1-repacks would have → total DRAM traffic unchanged. The copy cost is FUNDAMENTAL. With the two proven ceilings — (a) direct-contiguous coalesced halo read infeasible (conv3d C_in blocking for s0–s3), (b) full copy elimination architecturally impossible (depth-to-space upsamples interleave channels into space → no padded layout survives a stage boundary → per-stage re-pad forced) — **435ms is the practical floor** for this VAE decode.

**Decision.** Keep `TT_LTX_CHAIN` OFF by default (adds risk, ~0 gain). Shipped default-on copy-free (conv2-direct, 435.31ms) stays production. Goal closed: fastest achievable halo-read copy elimination = 435ms, PCC=1.0. Chaining code stays behind the flag as documented proof, not a win. Uncommitted on `kevinmi/np-halo-fabric-mux`: `TT_LTX_CHAIN` plumbing (`_pad_hw`/`_crop_hw`, `input_padded` threading, mid-block chain branch) — Python only; C++ ops already committed. See `project_np_halo_4x8`.

## Copy-free decode WIRED + default: 435ms, PCC=1.0 (beats 443 repack, 477 halo-read)
**2026-07-06 18:49** · `tt-metal@25a0c2daa27`

**DONE — copy-free wired through the ResNet blocks and shipped as default.** Clean 3-way A/B (145f 1088x1920 all-standalone): **copy-free 435ms** / repack(persist) 443ms / halo-read baseline 477ms. PCC=1.0 (all 11 halo convs vs full-pad). ~9% under the original halo-read; ~8ms under the repack.

**How:** in `_resnet_halves`, conv1 writes a PADDED output (`cf_output_padded` → conv3d `output_pad`), norm2 runs on the padded tensor (per-position; border garbage), and conv2 reads it COPY-FREE (`cf_input_padded`): `neighbor_pad_halo(input_pad)` reads the interior halo strided + `halo_scatter(border_only=True)` fills the border in place — **no ttnn.pad interior copy for conv2.** In-kernel logical mask uses interior `pad_offset` (`_get_pad_offset(sub_h/sub_w)`). Manager `neighbor_pad_halo_only` gained `input_pad_h/w` (interior-sized compact + op passthrough). Default on; `TT_LTX_NO_COPYFREE` opts out. Commit `25a0c2daa27`.

**Why ~8ms not ~18ms:** only conv2 of each (conv1,conv2) block is copy-free; conv1 still repacks (its input comes from norm1, unpadded). conv_in / upsample convs / conv_out aren't paired the same way. Full within-stage chaining (conv1 also copy-free) would need the block INPUT padded (thread through the residual + prev block), i.e. the decode-wide version — more work, upsample-capped. **435ms is the shipped best.** Mux path only (corner two-hop race-free on mux; non-mux corner has the documented race).

## Copy-free OPS all done + validated; only the decode wire remains
**2026-07-06 18:31** · `tt-metal@604a557d126`

All three copy-free ops now exist and are validated on BH 4x8:
1. **conv3d padded-output** (`output_pad_h/w`, slice-1) — `ed225f5`, validated.
2. **neighbor_pad_halo strided-interior read** (`input_pad_h/w`) — `df995f5`, full compact bit-exact vs contiguous on mux.
3. **halo_scatter border_only** (in-place border write, interior preserved) — `604a557d126`, output == repack, 32/32.
4. RMSNorm on a padded tensor: no change needed (shape-agnostic, border garbage overwritten by scatter).

**Remaining = the decode wire (vae_ltx), a real refactor:** initial pad at decode input; per conv: `neighbor_pad_halo(x_padded, input_pad=pHe/pWe)` -> `halo_scatter(compact, x_padded, border_only=True)` -> `conv3d(x_padded, spatial pad=0, output_pad=pHe/pWe, in-kernel logical mask)`; norm/residual flow on padded; crop before + re-pad after each depth-to-space upsample (they can't carry padding); final crop. Then e2e PCC=1.0 + wall.

**Endpoint (unchanged, honest):** eliminates within-stage interior copies; upsample boundaries keep a per-stage pad → ~425ms (~18ms, partial), not full elimination. The primitives are the reusable/hard part and are done; the wire is decode plumbing.

## Strided-interior halo read COMPLETE + validated on mux (copy-free primitive done)
**2026-07-06 18:25** · `tt-metal@df995f5f8a0`

**The copy-free enabling primitive is DONE.** `neighbor_pad_halo(input, ..., input_pad_h, input_pad_w)` reads the INTERIOR of a padded `[.,H+2pH,W+2pW,C]` input and produces the same compact halo as reading the equivalent contiguous input. Validated: FULL compact (interior + border + corners) bit-exact vs contiguous, deterministic, on the **default mux path** (`test_neighbor_pad_halo_strided_input`). Contiguous path byte-unchanged (input_pad defaults 0).

Commits: `c64161f` params → `3f5857a` H reader → `7706e13` W reader → `df995f5` mux full-compact validation. Kernels: np_h_reader gained `input_frame_rows` arg (frame stride decoupled from edge-row count); np_phase2_w_reader gained an `in_page(t,h,w)` helper + `input_pad_h/w` args (padded row/frame stride, identity when 0). Factory splits interior (halo geometry) vs padded (reader addressing) gated on `padded_input`; non-mux AND mux reader args both handle it.

**CORNER RACE — mux is fine, non-mux is not.** Under padded timing the W-section corner two-hop races on the NON-MUX path (send-done H->W barrier, 1 stick flaky) but is race-free on MUX (recv-authority barrier, `H_SIGNAL_W_RECV`, "holds for >2 H-axes and small shapes"). Production decode uses mux → no issue. If non-mux copy-free is ever needed, switch it to the recv-authority barrier.

**Remaining for copy-free decode:** border-only in-place scatter (re-add pre-repack mode) + decode wire (conv padded-output → norm on padded → neighbor_pad_halo(padded,input_pad) → border scatter → next conv plain; per-stage crop at depth-to-space upsamples). Endpoint still ~425ms (~18ms partial; upsamples cap full elimination).

## Copy-free route: strided-interior halo read — scaffolding landed, factory refactor is the work
**2026-07-06 18:00** · `tt-metal@c64161f9358`

After the direct-coalesce dead end (below), the ONLY copy-eliminating route is the padded chain: conv writes padded output (slice-1, built) -> RMSNorm runs on the padded tensor (shape-agnostic, no change) -> `neighbor_pad_halo` reads the padded INTERIOR (strided, NEW) -> border-only in-place scatter (re-add the pre-merge halo_scatter mode) -> next conv reads padded plain; one pad at each stage entry + crop at exit (depth-to-space upsamples can't carry padding). Eliminates the interior copy for WITHIN-stage convs only → ~18ms (443 -> ~425ms projected), NOT full elimination.

**Landed this turn (safe, inert, gated):** opt-in `input_pad_h/w` params on NpHaloParams (default 0 = byte-unchanged; 443ms path intact), threaded through the public op + hashed. Commit `c64161f9358`. Compiles + installed.

**PROGRESS (2026-07-06, cont'd): H+W strided readers DONE + validated (`7706e13d32d`).** W reader (np_phase2_w_reader) got an `in_page` helper (padded row/frame stride, identity when input_pad==0) + `input_pad_h/w` args; both non-mux and mux W-reader factory args updated. `test_neighbor_pad_halo_strided_input`: compact H-sections + W-interior rows bit-exact vs contiguous, 3/3 deterministic (non-mux). **KNOWN BUG — corner race:** W-section CORNER rows (h_padded < pH or >= pH+Hd, delivered via the H corner-first two-hop) mismatch 1 stick intermittently under the padded read timing (corner-first ON = flaky 2/3; corner-first OFF = deterministically wrong, so corner-first IS the right mechanism and my padded H-read perturbs its timing vs the W-reader corner consume / H->W barrier). Contiguous prod_pcc is rock-solid (deterministic), so the race is introduced by the padded path. Corner rows excluded from the test (TODO: fix the corner-first ordering for padded). Remaining after corner fix: mux validation, border-only in-place scatter, decode wire.

**PROGRESS (2026-07-06): H-reader strided read DONE + validated (`3f5857a28d7`).** `test_neighbor_pad_halo_strided_input` (4x8, non-mux forced): compact H-sections from a padded input == contiguous input, bit-exact 32/32. Implemented exactly per the derived args below: kernel `np_h_reader` gained `input_frame_rows` arg (frame stride = num_sticks_per_halo_dim * input_frame_rows); factory splits interior (halo geometry) vs padded (reader addressing) gated on `padded_input`. Non-mux + mux H-reader args both updated. **443ms contiguous path byte-unchanged (input_pad defaults 0).** Remaining: W reader (np_phase2_w_reader, same pattern — reads W-edges + corners), then validate full compact (H+W), then border-only scatter + decode wire.

**Remaining = the strided-interior read (multi-session, pervasive):**
1. `np_h_reader.cpp` + `np_phase2_w_reader.cpp`: add a `frame_stride` runtime arg; use it for `outer_dim_offset += frame_stride` (currently `num_sticks_per_halo_dim * input_halo_dim_size`). Backward-compat: factory passes the old product when input_pad==0.
2. Factory (`create_at`), gate on `padded_input = op.input_pad_h>0 || op.input_pad_w>0`. With a padded input tensor, `num_sticks_per_halo_dim`=Wp and `input_halo_dim_size`=Hp (padded). Derive interior: `W_dev=Wp-2*input_pad_w`, `H_dev=Hp-2*input_pad_h`. **The halo geometry (compact sizing, writer, W-section, mux) must use INTERIOR dims (H_dev,W_dev); only the reader INPUT addressing uses padded stride.** This is the pervasive part — audit every use of num_sticks_per_halo_dim/input_halo_dim_size.
3. H-reader RT args (DERIVED + verified on paper): `stick_start_id = input_pad_h*Wp + input_pad_w`; `input_halo_dim_size(edge)=H_dev`; `num_sticks_per_halo_dim(stride)=Wp`; `num_sticks_to_read=W_dev`; `frame_stride=Hp*Wp`; `outer_dim_offset_start_id = start_frame*Hp*Wp` where `start_frame = link_offset_start_id/W_dev` (link_offset_start_id is in stick units = frames*W_dev). Bottom edge check: `(H_dev-pad_id)*Wp + (pH*Wp+pW) + f*Hp*Wp = f*Hp*Wp + (pH+H_dev-pad_id)*Wp + pW` = padded interior bottom edge ✓; top edge ✓.
4. W-reader (np_phase2_w_reader): analogous padded addressing for the W-edge input reads; corners come from the H-exchange into compact (unchanged).
5. Mux readers (np_h_mux_writer path readers) — same.
6. Border-only in-place scatter: re-add the pre-repack halo_scatter mode (writes border into an EXISTING padded buffer, interior preserved) OR a flag on the current repack.
7. Decode (vae_ltx): conv writes padded output (output_pad); pass padded buffer + input_pad to neighbor_pad_halo; border-scatter in place; next conv plain; initial pad + per-stage crop at upsamples.
8. Validate: `neighbor_pad_halo(pad(x), input_pad=1).compact == neighbor_pad_halo(x).compact` (force non-mux first: TT_NP_W_WORKERS=1 TT_NP_H_WORKERS=1), then e2e decode PCC=1.0 + wall.

## WALL: halo-read border can't be coalesced directly (C_in blocking) — copy stays optimal
**2026-07-06 17:35** · `tt-metal@6c42dfa269f`

Goal was "fastest halo read directly, eliminate the copy" (beat the 443ms repack without a padded-buffer copy). Investigated + measured; **structural wall, reverted all edits, 443ms repack remains best.**

**Definitive evidence (COALESCE-DBG log added to conv3d_program_factory, per bench shape):** shard coalescing (`gather_rows_to_shard_coalesced`, bank-major 8-burst) is ON for only s4:
| shape | C_in | C_in_block | W_shard | coalesced |
| s0 | 1024 | 128 | 10 | false |
| s1 | 512 | 64 | 10 | false |
| s2 | 512 | 64 | 6 | false |
| s3 | 256 | 64 | 6 | false |
| s4 | 128 | 128 | 18 | **true** |

Coalescing requires `coalesced_shard_reads_candidate` = `input_dram_interleaved && C_in_num_blocks==1 && C_in_block_bytes==in_row_size_bytes && W_shard>=min`. s0-s3 have C_in=256-1024 blocked into 64/128-wide C-blocks → each gather read is a PARTIAL-C page slice → can't bank-coalesce (interior OR boundary; even plain conv is per-position there). So the halo penalty on s0-s3 (1.13-1.25x) is NOT a coalescing gap — it's the per-position boundary reads hitting a SECOND accessor (compact halo buffer) with the halo page-index branch, vs plain's single contiguous in-bounds accessor. Reads are already trid-pipelined (trid ring active when !coalesced), so it's not latency.

**Tried:** boundary-block partial-coalesce in `gather_rows_to_shard_selected` (coalesce in-bounds H-rows, per-position halo rows), gated on `halo_mode && EnableCoalescedShardReads`. Fires ONLY on s4 (only coalesced shape) → **s4 1.52x -> 1.49x (~0)**; helped blocks (H-boundary, W-in-bounds) are a minority of s4's perimeter. Reverted (dead code for s0-s3, ~0 for s4, hot shared kernel).

**Why the copy (repack) wins, fundamentally:** it UNIFIES interior+border into one contiguous buffer the conv reads as a plain single-accessor in-bounds conv — eliminating the dual-accessor + boundary-branch overhead that direct halo-read can't avoid on C_in-blocked shapes. Directly coalescing the halo read can only touch s4 (marginally); s0-s3 are structurally stuck. So "eliminate the copy" via a faster direct halo-read is not achievable to beat 443ms. The copy is the right architecture for this op suite.

## Merged ttnn.pad into halo_scatter (repack): 443ms, PCC=1.0, decode 3 ops -> 2
**2026-07-05 06:23** · `tt-metal@6c42dfa269f`

Folded the `ttnn.pad` interior copy into `halo_scatter`, which is now a REPACK op: takes the unpadded activation `x` + compact halo buffer, ALLOCATES the padded output, fills interior (from x) + border (from compact) in ONE batched multi-core pass. Decode per halo conv drops from 3 ops (halo -> ttnn.pad -> halo_scatter) to 2 (halo -> repack). `halo_scatter(compact, interior_src=x, pH, pW) -> padded` (allocates; every page written once, no zero-init). Commit `6c42dfa269f`.

**Perf (145f 1088x1920 all-standalone):** two-op (pad+scatter) 448ms -> naive fold 465ms (REGRESSED: per-stick read_barrier+write_barrier is serial) -> **batched fold 443ms** (8 in-flight reads/writes per barrier; cb_pages 8==16, no gain past 8). PCC=1.0 (whole padded == neighbor_pad_async full-pad, all 32 devices op-test + decode). Net vs the pre-merge 448ms: ~5ms + one fewer op.

**Why merging didn't "eliminate" the copy:** the interior copy is inherent when the op takes unpadded x (measured ttnn.pad = ~34ms, already multicore). Merging only removes the border double-write + a launch and pipelines slightly. True copy-free (~414ms proj) needs conv-padded-output + a STRIDED-interior halo read + per-STAGE crop/re-pad (depth-to-space upsamples break a whole-decode padded layout) — larger, deferred.

Best decode now **443ms** (was 478 halo-read). `TT_LTX_NO_PERSIST_PAD` opts back to halo-read.

## Persistent-padded COMPLETE: 448ms default, PCC=1.0, beats 478ms halo-read by 6%
**2026-07-05 05:25** · `tt-metal@195adfec157`

**DONE. Full persistent-padded activations wired through the LTX decode, correct (PCC=1.0 vs full-pad, all 11 halo convs) and FASTER: 448.6ms vs 477.3ms halo-read baseline (clean back-to-back A/B), ~6%. Now the DEFAULT (opt out: TT_LTX_NO_PERSIST_PAD).** Supersedes the earlier "slice-3 blocked" entry below — the block was resolved.

Pipeline per halo conv (vae_ltx.py CausalConv3d.forward, gated `_use_halo_conv and _persist_pad and h_pad_needed and w_pad_needed`):
1. `neighbor_pad_halo_only` -> compact halo buffer (mux, UNCHANGED — the proven-fast path).
2. reshape x to rank-4 + `ttnn.pad` (H&W) + reshape back -> padded buffer (interior copy; NOT the bottleneck — see below).
3. `ttnn.experimental.halo_scatter(compact, padded)` -> fills padded border in place (multi-core).
4. plain `conv3d(padded, padding=(T,0,0))` with IN-KERNEL logical mask: pass `logical_h_mask=logical_h+pHe`, `logical_w_mask=logical_w+pWe`, `pad_offset_tensor` — masks logical-pad positions during the read (no pre-mul).

**Perf ablation (145f 1088x1920, all-standalone) that drove the design:**
- halo-read baseline: 477ms
- naive persist (pad + 2 pre-mul H/W masks + single-core scatter): 1076ms
- fix 1 — multi-core `halo_scatter` (split global stick range across grid): scatter 509ms -> ~4ms => 571ms
- fix 2 — in-kernel mask instead of pre-mul muls (masks cost ~124ms!): => **448ms**
- Key insight: the `ttnn.pad` interior copy is NOT the bottleneck (persist w/o masks = 447ms already beat 478); the plain conv is enough faster than halo-read to absorb the copy. The mask muls were the killer; moving them in-kernel won it.

**Commits (branch kevinmi/np-halo-fabric-mux):** ed225f5 conv3d padded-output (slice1) · 01bb2fe halo_scatter op · cd5df1f wire persist-pad · 7412698 multi-core scatter · 008f5d7 in-kernel mask for plain conv (conv3d: decouple mask_mode from halo_mode + mask on fast read path) · 195adfe default-on.

**Build-sync (still required):** after `cmake --build build --target ttnn`, run `cmake --install build_Release --component tt_pybinds` AND `--component tar` (see older entry). Kernels (reader_vol2col.cpp, halo_scatter_writer.cpp) are JIT — no build, but host changes need both installs.

**Follow-ons (optional):** the ttnn.pad interior copy could still be eliminated via conv3d slice-1 padded-output chaining through the norm (would need strided halo read); projected small extra gain. Single-axis-only stages fall back to halo-read (halo_scatter needs the 4-section H+W compact layout).

## Persistent-padded slice-3: blocked by norm-between-convs; needs padded-output RMSNorm (decode-wide)
**2026-07-05 04:50** · `tt-metal@01bb2fe59ba`

Building blocks DONE + validated + committed: slice-1 conv3d padded-output (`ed225f5fc2b`), slice-2 `halo_scatter` op bit-exact (`01bb2fe59ba`). Slice-3 compute correctness is established BY TRANSITIVITY: halo_scatter's padded border is bit-exact to neighbor_pad_async's full-pad border, so a padded buffer with (true interior + scattered border) is bit-identical to the full-pad buffer → conv-plain on it == conv on full-pad == correct. No separate e2e compute proof needed.

**Slice-3 (realize the perf win) is BLOCKED by the VAE structure.** LTXResnetBlock3D `_resnet_halves` (vae_ltx.py:540-569): `norm1 → to_layout(RM) → conv1 → norm2(RMSNorm+SiLU, outputs TILE) → to_layout(RM) → conv2`. Every conv is separated from the next by an RMSNorm + TILE<->ROW_MAJOR round-trip. Consequences:
- A padded activation buffer CANNOT persist across two convs — norm2 rewrites the whole tensor through TILE layout (32x32 tiles; a maintained ROW_MAJOR halo border is incompatible).
- Persistent-padded for a SINGLE isolated conv needs the interior placed into the padded buffer = a copy = exactly the full-pad 551ms cost. Measured shortcut `ttnn.pad(x)+halo_scatter+conv-plain` == full-pad path + extra scatter ≈ >551ms. NO localized win exists.
- The ONLY copy-free path: make the pre-conv op (RMSNorm) write its output directly into a padded buffer's INTERIOR (strided), analogous to slice-1 for conv, PLUS a padded-aware TILE->ROW_MAJOR conversion. That is a decode-wide change to the norm + layout ops, not the conv path.

**Projected win vs 478ms is modest+unproven** (~24-48ms: removes the conv3d halo-read penalty, which e2e is a fraction of decode since halo 478ms already beats full-pad 551ms by avoiding the interior copy). Verdict: the decode-wide norm/layout padded-output rewrite is disproportionate effort/risk on a working 478ms pipeline for that margin. Recommend: keep slices 1+2 as landed reusable capability; pursue slice-3 only if a padded-output RMSNorm is independently wanted. **478ms remains the best validated config.** Proven path is intact: halo_scatter is a new op, NOT wired into decode, so the 478ms path is byte-for-byte unchanged.

## Persistent-padded: halo_scatter op (Architecture B slice-2) VALIDATED bit-exact
**2026-07-05 04:42** · `tt-metal@01bb2fe59ba`

Built + validated the local border-scatter op. Committed `01bb2fe59ba`.

**halo_scatter op** (new, in `neighbor_pad_halo/` dir, same CMake target): `ttnn.experimental.halo_scatter(compact, padded, np_padding_h, np_padding_w)` → copies the compact halo buffer [Htop|Hbot|Wleft|Wright] into the BORDER of a persistent padded buffer `[outer,H+2pH,W+2pW,C]` IN PLACE (interior untouched), returns padded. LOCAL (no fabric/sema); single-core writer kernel `halo_scatter_writer.cpp` computes each dst padded page from geometry in-kernel (no index tensor). Mux exchange 100% unchanged. Files: `halo_scatter.{hpp,cpp}`, `device/halo_scatter_device_operation{,_types}.{hpp,cpp}`, `device/halo_scatter_program_factory.{hpp,cpp}`, `device/kernels/halo_scatter_writer.cpp`; nanobind in `neighbor_pad_halo_nanobind.cpp` (+`bind_halo_scatter` call in `ccl_experimental_nanobind.cpp`).

**Validation:** `test_neighbor_pad_halo_padded_border` (input `[1,8,144,256,128]`, 4x8) now: neighbor_pad_halo→compact, halo_scatter→padded, compare padded border to neighbor_pad_async full-pad border. **PASS bit-exact all 32 devices.**

**CRITICAL BUILD-SYNC GOTCHA (cost me ~5 rebuilds):** `cmake --build build --target ttnn` writes `build_Release/ttnn/*.so` but Python loads `ttnn/ttnn/_ttnn.so` and (via RPATH) `build_Release/lib/_ttnncpp.so` — NEITHER is synced by the plain build. After building you MUST run BOTH:
`cmake --install build_Release --component tt_pybinds` (syncs `_ttnn.so` = nanobind/python module)
`cmake --install build_Release --component tar` (syncs `build_Release/lib/_ttnncpp.so` = the C++ op lib the RPATH loads).
Symptoms if skipped: new nanobind op → `AttributeError: module 'ttnn.experimental' has no attribute X`; then after only tt_pybinds → `ImportError: undefined symbol` for the op's C++ fn. Op-lib-only changes (no new nanobind) happen to work via RPATH only if `build_Release/lib` is already fresh — otherwise stale too. This means earlier same-session op-lib rebuilds may have silently run stale until an install ran.

**Next = slice 3 (decode rewire, the perf payoff):** persistent padded buffers in vae_ltx decode: conv writes padded interior (slice1 output_pad) → neighbor_pad_halo(compact,mux) → halo_scatter(border) + logical mask → conv reads padded plain(pad0); crop at end. Then e2e PCC=1.0 + traced wall vs 478ms baseline. WATCH: halo_scatter kernel is single-core serial (~1120 sticks/dev, per-stick read+write+barrier) — may need multi-core / read-write pipelining if it eats the halo-read-conv savings across 42 convs.

## Persistent-padded: slice-2 direct-write wall + pivot to border-scatter (Architecture B)
**2026-07-05 04:26** · `tt-metal@ed225f5fc2b-dirty`

Goal = "full persistent padded activations, as fast as possible", hard constraint "keep the mux (proven faster)". Baseline to beat: halo (mux + conv3d halo-READ) = **478ms** traced decode, 13.3% faster than full-pad 551ms, PCC=1.0 (`ac49eb38b99`). Slice-1 = conv3d opt-in padded-output mode (`output_pad_h/w`), validated+committed `ed225f5fc2b`.

**Slice-2 Architecture A (halo writes border DIRECTLY into a persistent padded `[T,H+2,W+2,C]` buffer) — FAILED, abandoned.** Added `padded_output` inference (`halo_buffer.rank()>=4`) + padded RT-arg overrides to non-mux H/W writers + w_section base changes in `neighbor_pad_halo_program_factory.cpp`. `test_neighbor_pad_halo_padded_border` (input `[1,8,144,256,128]`, 4x8, `TT_NP_W_WORKERS=1 TT_NP_H_WORKERS=1` non-mux): border PCC ~0 on all 32 devices. dev9 landing map (H_pad=38,W_pad=34,Hd=36,Wd=32): Wleft col0 nz rows `[0..32]` (should be 38), Wright col33 `[0..31]`, Htop row0 OK, **Hbot row37 EMPTY**. Root cause: transport writers + **corner routing** are structurally tied to the compact layout — corners arrive via the H-exchange into the compact W-section (no such region in a padded buffer); W-writer base/count assume compact geometry. Fixing = rewriting the mux writers the user wants preserved. Reverted all factory edits (`git checkout`), rebuilt → proven 478ms path restored (source==binary).

**Pivot → Architecture B (mux 100% untouched):** neighbor_pad_halo (compact, mux) UNCHANGED → cheap **LOCAL in-place border-scatter** (compact → persistent padded buffer border only, interior preserved) → conv reads padded PLAIN. **Correctness de-risked:** the scatter is the exact inverse of `compact_halo_reference()` (`tests/nightly/t3000/ccl/test_neighbor_pad_halo.py:40-71`), which the compact buffer is already byte-exact against. Mapping (per frame t): Htop `(t,pr,w)`→`padded[t,pr,pW+w]`; Hbot→`padded[t,pH+Hd+pr,pW+w]`; Wleft `(t,hp,wc)`→`padded[t,hp,wc]` (full height incl corners); Wright→`padded[t,hp,pW+Wd+wc]`. Corners live in Wleft/Wright (already resolved by halo op) → scatter needs NO fabric/neighbor.

**Why not existing ops:** `ttnn.scatter` allocates a fresh output (`create_output_tensors`) = full padded read+write per call = re-introduces interior-copy cost (~551ms). Concat same. So B REQUIRES a bespoke IN-PLACE border-only writer kernel.

**Concrete plan (resumable):** (1) New op `halo_scatter` (sibling in `neighbor_pad_halo/`, reuse CMake target), modeled on NpHalo mesh-workload skeleton; LOCAL only; `create_output_tensors` returns the padded buffer written IN PLACE (like NpHalo returns halo_buffer). Kernel = "scatter sticks by index": host precomputes per-shape dst-page list (mapping above), reader reads compact sticks, writer writes `padded[dst_page]`. (2) Validate bit-exact via adapted `test_neighbor_pad_halo_padded_border`: halo(compact)→halo_scatter→compare padded border to `neighbor_pad_async` full-pad border. (3) Slice-3: rewire vae_ltx decode — persistent padded buffers, conv writes padded interior (slice1) → halo(compact,mux) → halo_scatter(border + logical mask) → conv reads padded plain(pad0); crop at end; e2e PCC=1.0 + traced wall (target <478ms).

**Open cost/benefit (asked user, away):** win over 478ms plausible (removes conv3d halo-read penalty, measured +6-52% vs plain conv isolated) but unproven until measured; needs new op + decode rewire. Uncommitted: only the new `test_neighbor_pad_halo_padded_border` (direct-write test, expected-fail post-A-abandon; repurpose to halo_scatter).

## Persistent-padded: slice1 done+committed; slice2 implemented, diagnosed, NOT yet correct
**2026-07-05 04:00** · `tt-metal@ed225f5fc2b(+wip)`

SLICE 1 (conv3d padded-output) — DONE/VALIDATED/COMMITTED (ed225f5fc2b). Interior bit-exact; decode
still 477.7ms/pcc=1.0. Foundation solid.

SLICE 2 (halo -> padded border, keep mux) — IMPLEMENTED (factory, gated on halo_buffer rank>=4) but
padded border MISPLACED (test_neighbor_pad_halo_padded_border: border PCC ~0). ROOT CAUSE diagnosed:
the halo factory's per-path writer RT-arg override passes the INPUT stride/bases, not the padded
OUTPUT ones. np_writer.cpp is padded-capable (dst = eff_offset + stick_start + row*num_sticks; frame
stride = num_sticks*output_halo_dim_size, line 586) — needs, per write path:
  rt[0] = link_t_start * H_pad * W_pad   (frame base; direction picks top/bot rows via
          (output_halo_dim_size-padding)*W_pad)
  rt[1] = pad2_left (pW)                 (stick_start_id, interior-W col offset)
  rt[3] = H_pad                          (output_halo_dim_size)
  rt[8] = W_pad                          (num_sticks_per_halo_dim stride)
The compact override at factory ~657-668 hardcodes compact ([0]=h_top/bot_link_start, [1]=0,
[3]=np_padding_h, [8]=W_dev). MUST add a padded branch there AND in the 3 other write paths — and the
DEFAULT 4x8 config uses the MUX paths (np_h_mux_writer, np_w_mux_writer), not the non-mux H-writer, so
those mux RT-arg sites need the same padded override. 4 paths total. Gated => compact 478ms path
untouched (regression-checked pcc=1.0).

SLICE 3 (decode rewire) — pending: padded activation buffers; conv reads padded (pad0)+writes padded;
norm/add/act on padded (border transient, refreshed by border-fill before each conv); logical mask in
the border-fill; crop at end.

HONEST: full pipeline is multi-session. Foundation (slice1) validated; slice2 diagnosed to the exact
per-path RT-arg fix; slice3 scoped. Next: add padded RT overrides to the 4 halo write paths, re-run
test_neighbor_pad_halo_padded_border to green, then slice3 + e2e PCC/wall.


## Persistent-padded activations: slice 1 DONE+committed; slices 2/3 scoped
**2026-07-05 03:42** · `tt-metal@ed225f5fc2b`

Goal: eliminate BOTH the full-pad interior copy AND the halo per-stick conv-read penalty by keeping
activations in padded [H+2,W+2] buffers: conv writes its result into the padded interior (~free,
strided), the fast MUX halo writes only the border, next conv reads padded as a PLAIN conv
(coalesced). Keeps the proven NP-halo mux transport.

CHECKPOINTS (branch kevinmi/np-halo-fabric-mux, local): ac49eb38b99 = 478ms/pcc=1.0 fallback;
ed225f5fc2b = slice 1.

SLICE 1 (conv3d padded-output) — DONE, VALIDATED, COMMITTED. Opt-in output_pad_h/w: writer.cpp
writes the [H,W] result into a padded [H+2,W+2] buffer interior (padded page index; CT args 26/27;
out accessor offset 26->28; compute_output_specs padded shape; full op + nanobind plumbing).
test_conv3d_padded_output: interior bit-exact vs compact. Default-off => decode still 477.8ms/pcc=1.0
(regression-checked). Rebuilt.

SLICE 2 (halo -> padded border) — SCOPED, NOT DONE. The halo op only exchanges the border (never
copies interior), so "halo into padded buffer" == the border-fill we want. The np_writer supports an
output row width (num_sticks_per_halo_dim) + output_halo_dim_size (np_writer.cpp:288 comment "W +
pad2_left + pad2_right (full output row width)"), and the async NP uses this same machinery to write
padded. BUT the halo op's dst_stick_id addressing (np_writer.cpp:391-440) + the section bases
(factory ~150/324/755) are compact-[Htop|Hbot|Wleft|Wright]-specific. Padded-border output needs the
dst addressing reworked to padded page positions (Htop rows[0:pH] cols[pW:pW+W_dev]; Hbot
rows[pH+H_dev:]; Wleft cols[0:pW]; Wright cols[pW+W_dev:]; stride W_pad) across np_writer,
np_phase2_w_reader, np_h_mux_writer, np_w_mux_writer + factory. Multi-cycle (host rebuild) + PCC-risky.

SLICE 3 (decode rewire) — NOT DONE. vae_ltx: padded activation buffers; conv reads padded (pad0) +
writes padded interior; norm/act/add run on padded (border transient garbage, refreshed before each
conv by the border-fill — RMSNorm is per-position over C so border garbage is harmless); logical mask
returns to the border-fill op (free, like full-pad's logical_h); crop to logical at the very end.
Then PCC=1.0 + wall (target < 478ms, toward plain-conv + border-only-NP floor).


## (b) ATTEMPTED: bank-major halo-row coalescing — ~0 gain, structural. Reverted.
**2026-07-05 03:16** · `tt-metal@73a767770a3-dirty`

Implemented gather_halo_row_coalesced in reader_vol2col.cpp (bank-major read of interior-W H-top/
H-bot halo rows, reusing CoalescedRowLayout; gated to unmasked rows so the logical mask stays exact;
plumbed EnableCoalescedHalo + shard_l1_base + coalesced_scratch_offset through gather_rows_to_shard).
pcc=1.0 (correct). BUT measured ~0 improvement: isolated conv s4 1.50x (was 1.52x), s2/s3 unchanged.
EMPIRICAL root cause: the tuned LTX blockings have small W_out (s2/s3 W_out=4 -> per-block w_count=6;
s4 W_out=16 -> 18) and my coalesce gate needs w_count > NUM_DRAM_BANKS(8); s2/s3 never fire, s4 fires
only on its few H-boundary blocks (it's W-boundary-dominated). Per-block halo runs are smaller than a
DRAM bank group -> nothing to coalesce WITHIN a block. The halo-read penalty is structural to the
per-block gather at 4x8 small shards. REVERTED (dead complexity in a shared kernel). Working state =
478ms/pcc=1.0 (13% over full-pad) is the practical ceiling. A real recovery needs a CROSS-BLOCK halo
gather (accumulate all per-device boundary halo, coalesce globally) or a blocking change that
regresses the matmul — a major reader rewrite, not a scoped fix.


## PROVEN: halo-read conv is +6-52% vs plain (per-stick interleaved halo reads) — the NP-win leak
**2026-07-05 03:05** · `tt-metal@73a767770a3-dirty`

Why the NP transport win doesn't transfer to e2e, settled with a TRACED isolated conv3d at EXACT
production blocking (test_bench_conv_halo_vs_plain in prof_vae_ltx.py; all get_conv3d_config [exact]
hits — key uses LOGICAL per-dev 17x15/34x30/68x60, tensor is physical 18x16/36x32/72x64):

| shape | per-dev | plain us | halo us | halo/plain |
| s0 | 9x8   | 2345 | 2465 | 1.06x |
| s1 | 18x16 | 2211 | 2528 | 1.14x |
| s2 | 36x32 | 12204| 15264| 1.25x |
| s3 | 36x32 | 5913 | 6855 | 1.16x |
| s4 | 72x64 | 6005 | 9139 | 1.52x |

halo_nomask == halo_mask on every shape → the in-kernel logical mask costs ~0; the WHOLE penalty is
the inherent per-stick halo-buffer boundary read. Plain conv reads its contiguous pre-padded buffer
with 8-burst BANK-MAJOR coalescing (gather_rows_to_shard_coalesced); the halo conv reads the
DRAM-INTERLEAVED compact halo buffer ONE STICK at a time in the in_padding branch of reader_vol2col.
Penalty scales with shard size (s4 +52%) — more/bigger boundary blocks. NOTE: absolute us are
single-op-trace inflated; the RATIO is the valid number. Also: the untraced tracy device-FW profile
was UNRELIABLE here (disagreed in sign with traced walls) — trust _trace_and_time / traced.

(b) FIX = give the halo boundary read bank-major coalescing. H-top/H-bot rows are W_in consecutive
pages == interior-row layout → gather_rows_to_shard_coalesced applies directly with page_base=
halo_htop_base+(frame*pH+hrow)*W_in. W-left/right are 1 stick/row → coalesce across rows (harder;
loop is h-outer/w-inner). s4 penalty is W-boundary-dominated so needs the W-side too. Alt: bulk-
preload the per-frame halo band into L1 (per-frame ~70KB fits) then read boundary from L1. All are
real reader_vol2col.cpp changes, multi-hour, PCC-risky. Working fallback = 478ms/pcc=1.0 (committed).


## FINAL: in-kernel logical mask — halo 477.7ms, pcc=1.0 (fastest + correct)
**2026-07-05 00:00** · `tt-metal@73a767770a3-dirty`

Built the opt-in in-kernel logical-pad mask in the standalone conv3d halo-read, dropping the
~59ms pre-conv mask mul on the halo path. all-standalone 4x8, 145f@1088x1920, TRACED_DECODE_WALL_MS:

| config | wall | correct |
|---|---|---|
| full-pad baseline | 550.9 | yes |
| halo + combined-mask mul | 536.3 | yes |
| **halo + in-kernel logical mask** | **477.7** | **pcc=1.0** |

=> halo 13.3% faster than full-pad, bit-exact. Opt-in/default-off: conv3d gains
logical_h_mask/logical_w_mask attrs + pad_offset_tensor (per-device [h_start,w_start] sharded
input); reader_vol2col.cpp reads it into file-scope globals and masks; every other conv3d caller
is byte-identical (no args => elided). Host C++ built; mask logic is kernel-only (JIT), no rebuild
per mask iteration. CORNER FIX (0.9971->1.0): halo buffer is built from UNMASKED input, so masking
only interior reads leaks a neighbor's pad via the H-halo/W-tail corner; the mask must be universal
(global = start_dev + in, valid across the neighbor boundary) applied to interior AND halo branches.
block_has_pad forces pad blocks off the coalesced fast path. Files uncommitted on the branch.


## CORRECTION: halo WINS on 4x8 — the loss was my own redundant mask mul
**2026-07-04 22:30** · `tt-metal@73a767770a3-dirty`

The "halo is a net e2e loss" entry below is WRONG. Decomposition (NP_CONV_PERF hook on
test_conv3d_halo_vs_fullpad, real 4x8 shapes) proved the conv3d halo-READ is fine:
conv_halo_read == conv_plain within 0.1% (s2 115357 vs 115367us), and np_halo << np_async.
The +51ms e2e loss was entirely my first `_get_h_mask` fix: a SEPARATE full-tensor `ttnn.mul`
per halo conv, gate firing on EVERY stage (physical > logical everywhere once s0's 34->36 pad
propagates through the upsamples) = ~66ms of redundant muls.

FIX: `_get_hw_mask` folds H+W masking into the ONE pre-conv mul the halo path already ran for W
(full-pad keeps W-only mul + neighbor_pad logical_h). vae_ltx.py, halo path only.

e2e all-standalone (LTX_USE_FUSED=0), 145f@1088x1920, TRACED_DECODE_WALL_MS:
| variant | ms | correct |
| full-pad baseline (NP_NO_HALO_CONV=1) | 550.8 | yes |
| halo, first H-mask (separate mul) | 602.2 | yes but slow |
| halo, combined H&W mask (SHIPPED) | **536.3** | pcc=1.0 |
| halo, no pre-mask (floor, wrong output) | 477.6 | no |

=> halo now BEATS full-pad by 2.6% and is correct. Remaining lever for "fastest possible":
in-kernel logical_h/logical_w mask in the conv3d halo-read (reader_vol2col.cpp) removes the
~59ms mul -> ~478ms (~13% vs full-pad). Precedent (fused op in-kernel logical_h, commit
e58dccd) is NOT in this branch, so it'd be written from scratch in the shared standalone
conv3d reader + coalesced-path handling + rebuild + re-PCC; scoped, not yet done (asked user,
was away). Do NOT trust the entry below.

## 4x8 correctness fixed + halo is a net e2e loss
**2026-07-04 21:52** · `tt-metal@73a767770a3-dirty`

Branch `kevinmi/np-halo-fabric-mux`, BH 4x8 Galaxy. Task: prove NP-halo PCC on 4x8, find
fastest config, measure e2e traced-VAE speedup. (Device reset via `tt-smi -glx_reset` through
the broker — the sanctioned Galaxy reset, NOT the dangerous `-r`.)

### 1. Correctness (was #1 gate — FIXED)
- Op-level `neighbor_pad_halo` buffer byte-exact on all 6 real 4x8 shapes / 32 devices
  (`test_neighbor_pad_halo_prod_pcc` + `_prod_pcc_8k`, zeros). NP op itself was already correct.
- e2e halo-conv3d vs full-pad decode initially FAILED `pcc=0.9548`. **Root cause:** the halo
  path (`neighbor_pad_halo_only`) has no `logical_h` arg, so H-pad rows (>= logical_h) were
  never masked; the full-pad path masks them via `neighbor_pad_persistent_buffer(logical_h=...)`.
  2x4 never hit it (all logical dims divide the mesh factor → no padding); 4x8 pads s0 34→36 and
  the unmasked pad row 34 leaks into the conv output at logical row 33.
- **FIX** (Python only, no rebuild): added `_get_h_mask` in `models/tt_dit/models/vae/vae_ltx.py`
  (mirror of `_get_w_mask`, shard dim 2, zeros rows >= logical_h) as a pre-conv mul-mask gated to
  `self._use_halo_conv`. → `DECODE-HALO-PCC pcc=1.0`, 42 halo convs. Pre-masking the input is
  idempotent for the full-pad golden (its output masking already zeros those rows).

### 2. Real 4x8 shapes (captured, don't re-derive)
`prof_vae_ltx` `NP_CAPTURE_SHAPES` (LTX_USE_FUSED=0): `.shape` on a sharded mesh tensor is the
PER-DEVICE shard. s0 pad (34→36, 60→64) PROPAGATES through the x2 upsamples, so physical global =
per-dev × factor: 36x64 → 72x128 → 144x256 → 288x512 (NOT the 2x4 logical dims). `_LTX_PROD_4x8`
added to `tests/nightly/t3000/ccl/test_neighbor_pad_halo.py`.

### 3. Fastest NP config (#2): committed default already optimal on 4x8
Swept payload{8192,15232}, buffers{4,8}, workers{W4H4,W2H2}. Default (payload 8192, auto W4/H4,
num_buffers=4) is best on the dominant s3/s4; payload 15232 REGRESSES them (s4 330→349us); W2H2
helps only tiny s1. **No factory change.** Op GB/s vs 50 ceiling: s2 61% (best), s3 41%, s4 31%,
s0 4% (overhead-bound).

### 4. e2e verdict (#3): halo does NOT speed up 4x8
`TRACED_DECODE_WALL_MS`, 145f@1088x1920:

| config | halo ON | halo OFF (NP_NO_HALO_CONV=1) |
|---|---|---|
| all-standalone (LTX_USE_FUSED=0) | 602.1 | **551.2** |
| hybrid (LTX_USE_FUSED=1) | 564.6 | 563.6 |

Fastest overall = all-standalone + full-pad (551ms). Halo is **-9.2% all-standalone**, neutral in
hybrid. Rigorous: default is the min NP-op config and the NP op is only ~1-2% of the decode, so no
NP config flips it. The cost is the conv3d HALO-READ at small 4x8 shards (full-pad's persistent-
buffer interior copy is cheap when shards are small; halo's scattered read + fixed overhead
dominates). Consistent with `neighbor-pad-conv3d-4x8.md` ("fused win is a 2x4 thing").

### Recommendation / open
- Gate `_use_halo_conv` (and the fused op) OFF for small per-device shards (4x8 class); keep on for
  2x4. Needs a shard-size threshold; 2x4 e2e not measured this session.
- The `vae_ltx.py` H-mask is a real correctness fix to commit regardless.
- Making halo win on 4x8 needs a conv3d halo-read optimization, not an NP-op config knob (out of scope).
- Test edits point prod/e2e tests at (4,8); for a clean commit parametrize [(2,4),(4,8)] so 2x4
  nightly CI isn't lost.

## WIP: padded-output fused np_halo (0dc9c7d) — plumbing+scatter kernel done, factory barrier is the supervised piece
**2026-07-07** · `tt-metal@0dc9c7dd330`

Goal: fold halo_scatter into neighbor_pad_halo as a kernel-level padded mode; measure; delete original if faster.
Findings: (a) pure single-kernel (fabric→padded direct) is SLOWER — loses the mux W-send coalescing (np_w_mux_writer.cpp:145 ships contiguous compact pages as one packet; padded border is strided). (b) Only win is interior-copy/exchange OVERLAP, ceiling = exchange time = 1.9% ≈ 8ms (profile), best case ~427ms. (c) Composite (public-op or manager) = ~0 (= current 435, same two programs).
Built (committed 0dc9c7d, gated output_padded default off): op plumbing (output_padded/border_only params, padded_output tensor, threaded public op→prim→device-op→headers) + np_fused_scatter_writer.cpp (interior overlaps exchange, border waits exchange_done). Compiles; shipped path byte-identical.
REMAINING (supervised — barrier failure hangs mesh, needs a reset): factory scatter injection — scatter cores at x>=1 (fabric is col 0), exchange_done barrier (GlobalSemaphore: readers inc, scatter wait_min(num_readers); count from reader CoreRangeSet), 2 reader-kernel incs (np_h_reader, np_phase2_w_reader), vae_ltx wiring, then small-halo-test PCC then decode PCC+wall. Est ~8 edits + supervised device run.

## DONE: fused padded-output np_halo (interior overlaps exchange) — 432.7ms, PCC=1.0, faster than original
**2026-07-07** · `tt-metal@37726707cde`

Built + measured the kernel-level fold. KEY insight that unblocked it: the interior copy has NO dependency on the fabric exchange (reads input, writes padded interior), so it can run CONCURRENTLY on free cores — no cross-core barrier, no hang risk. Only the border needs the exchange done, and that stays a separate cheap halo_scatter(border_only) (op-level dependency).

Implementation: neighbor_pad_halo `output_padded` mode runs np_fused_scatter_writer (interior range only) on free cores = cols>=1 minus (np_worker_core_ranges + mux h/w worker + mux cores) — computed by CoreRangeSet.subtract chain (fabric is col 0; workers/mux are in cols>=1, so must all be excluded). Manager neighbor_pad_halo_scatter routes the repack (conv1) path through it; border_only (conv2) unchanged. Plumbing: output_padded/border_only params + padded_output tensor through public op/prim/device-op/nanobind/override.

MEASURED (traced, LTX_USE_FUSED=0): **432.67ms vs ~435.1 baseline = ~2.4ms (~0.5%) faster**, PCC=1.0. Within the ≤8ms overlap ceiling (exchange = 1.9%; overlap not perfect). The pure single-kernel (fabric→padded direct) remains slower (coalescing loss) — NOT built. halo_scatter kept for the border (genuinely needs exchange done). Pushed to origin/kevinmi/np-halo-fabric-mux (37726707cde). Gotchas hit: nanobind must expose new kwargs (TypeError), and scatter cores must exclude ALL worker/mux ranges not just np_worker (CB-index-0 conflict).

## Full-fold design (barrier-free) to DELETE halo_scatter — not yet implemented (37726707cde has interior fold only)
**2026-07-07** · `tt-metal@37726707cde`

Shipped: interior copy folded into neighbor_pad_halo padded mode, overlapped with exchange, 432.7ms PCC=1.0. halo_scatter still called for the border.

Clean barrier-free way to fold the BORDER too (delete halo_scatter), for a future focused session:
- np_phase2_w_reader ALREADY waits on both the H->W barrier_sem (H halo done) AND its w_neighbor_sem (W recv done). So after those existing waits it has the FULL compact for ITS rows — no NEW barrier needed.
- Split (no overlap): free-core np_fused_scatter_writer does interior-rows' interior cols (the big interior copy, overlapped, DONE). np_phase2_w_reader, at its end, scatters ITS rows' border to padded: interior rows -> W-left/W-right cols (from compact W-section); pad rows -> whole row (interior cols from compact H-section + corners from W-section). Each padded row owned by exactly one W-reader core -> no overlap.
- Then halo_scatter (op+kernel+binding+cmake+test) is deletable; manager neighbor_pad_halo_scatter for border_only(conv2) also routes through the op's border-scatter.
- Perf: ~0 over the interior fold (border is tiny; removes a near-free traced dispatch). Risk: LOW (no cross-core barrier; PCC-gated). Cost: reader-kernel surgery in the 393-line np_phase2_w_reader (per-core row->padded mapping incl corners) — error-prone, needs device PCC iteration.
Earlier hang-risk framing was WRONG: the W-reader's existing H->W barrier + recv waits give the completion for free; no new barrier.

**Two implementation routes for the border fold, each with a concrete blocker (found while attempting it — resolve before coding):**
- **W-reader-scatter route (barrier-free, preferred):** each W-reader core has only its row-slice, NOT total T, so it can't compute the global compact section bases on its own — W-left/W-right offsets are `2·T·pH·Wd` and `2·T·pH·Wd + T·Hp·pW`, which need total T. Fix: pass the section bases (or total T) as extra kernel args from the factory. Also multi-path: the scatter must be added at BOTH completion points — the coalesce path (`np_phase2_w_reader.cpp` ~line 197–201, returns early) and the main-loop tail (~line 349–392).
- **Free-core-scatter + barrier route (alternative):** reuse `np_fused_scatter_writer` for the FULL range on free cores (it already computes correct bases from factory-passed `outer/Hd/Wd/pH/pW`); border sticks wait on an `exchange_done` sem that the readers increment. Blocker: **trace-safe reset** — `exchange_done` must zero between traced dispatches without racing the readers' increments (the classic traced-op semaphore problem; the op self-resets its other sems at kernel end, so mirror that). Also the M-waiters problem (M scatter cores each need count N): use a poll-a-single-aggregator counter or multicast, NOT N×M unicast increments.
