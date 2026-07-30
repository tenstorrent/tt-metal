# d.2b — depth-4 blocker + N-EU line-rate kernel: research plan (multi-agent, 2026-07-29)

Distilled from an 8-agent deep-research workflow (6 research dimensions → synthesis → adversarial critic,
~950K tokens) over the DOCA 3.4 headers, the stock deep-RQ samples, the gateway sample/kernel, and the docs.
**Status: research done; plan is a de-risking experiment ladder, NOT yet an approved rewrite.**

## Root cause (converged, medium-high confidence)

The gateway hand-rolls an RC QP with **external umem+DBR+UAR** and shares **ONE `doca_dpa_completion` for both
send and recv**, hard-baked to `queue_size = 4` at `dpa_verbs_initiator_target_sample.c:1447` — **decoupled** from
the depth macro `VERBS_SAMPLE_QUEUE_SIZE` (`:88`, which drives RQ/SQ + the QP umem). "Only RQ=CQ=4 works" because
4 is the one value where the two independently-hard-coded numbers coincide. The stock deep-RQ reference
(`verbs_receive_packets_host.c`) proves the working contract: a **DOCA-managed `doca_verbs_eth_rq`** with a
**single knob** (`PACKETS_NUMBER=32`) used identically for CQ `queue_size` == RQ `wr_num` == posted-WR count, a
**dedicated recv-only completion**, and the **N-post / 1-commit / drain-N** device idiom. The gateway violates all
of that. **Caveat:** the aborting bodies (`doca_dpa_dev_verbs_qp_post_recv_wr`/`commit_recv`) ship as declarations
only — the fault instruction lives in the precompiled `libdoca_dpa_dev_verbs`, so the exact mechanism is not
source-visible. There is **no header-documented CQ:RQ:SQ ratio rule** (the critic confirmed `doca_verbs.h:1293`
"value should be same" is about `external_datapath_en`, not sizes).

## Bisect matrix (silicon, provenance verified via per-run grep of line 1447)

| RQ/SQ (:88) | CQ (:1447) | result |
|---|---|---|
| 4 | 4 | ✅ works (0.30–0.38 Mpps, byte-exact) |
| 8 | 8 | ❌ recv-post RPC abort |
| 4 | 128 | ❌ recv-post RPC abort |
| 64 | 128 | ❌ recv-post RPC abort |

**→ The critic's E1 ("bump both :88 and :1447 to 8, nothing else") == the RQ=CQ=8 row == already run == FAIL.**
So the trivial "tie :1447 to the macro" fix is **dead**; the blocker is genuinely in the shared/hand-rolled RC-QP
recv-post path. Remaining live fixes: **(a) separate recv-only completion sized ==rq_wr + batch N-post/1-commit**
(untested), or **(b) SRQ device path** (`doca_dpa_dev_verbs_srq_post_recv_wr`, distinct API — the escape hatch).

## Experiment ladder (REVISED per critic — run in THIS order, one variable each)

- **E3 FIRST — bandwidth-wall go/no-go KILL-SWITCH (needs no depth fix).** Re-run the A5/option-1 DPA-heap 2-SGE
  gather prefill-blast at **8 KB** frames at N4 and N6; measure aggregate Gbps. **If it stays ~134–146 G at 8 KB
  (same as 4 KB) → the DPA path is byte-bandwidth-bound → 200 G is UNREACHABLE at any EU count → abort d.2b, pivot
  to Arm-HW-TX (`doca_ttblast`, 198.3 G @4112B, proven, latency cost).** If it scales past ~180 G at 8 KB → the
  pps-wall model holds → the DPA rewrite is justified. **This one result decides whether d.2b is worth doing.**
- **E7 (trivial, read-only):** query the real RC caps on the SF — `doca_verbs_device_attr_get_max_qp_wr`
  (`doca_verbs.h:5490`) + `max_cqe` (`:5528`) — NOT `max_eth_rq_wr_num`. Establishes the true RC depth ceiling.
- **E2 (only if E3 passes) — single-variable SDK-trace decode:** re-run the RQ=CQ=8 build with
  `--sdk-log-level TRACE` + DPA fault decode to localize the abort: DBR-write vs RQ-WQE-store vs first
  `get_completion` consumer-index. Forks: dbr/umem → Finding 4; CQ consumer-index → Finding 1/3; intrinsic to
  `qp_post_recv` → pivot to SRQ.
- **E4:** `sq_wr=0` acceptance smoke (does an RC QP with zero SQ still take INIT→RTR→RTS and produce recv CQEs?).
- **E5:** DPA-private-DDR **remote-write** landing smoke — can a remote RoCE WRITE_IMM target `MMAP_TYPE_DPA`
  memory byte-exact at rate? (The 134 G number was DPA reading its OWN heap; remote-NIC writes to DPA DDR are
  unverified — this gates the Stage-2/3 landing-in-DDR assumption.)
- **E6:** SRQ escape-hatch + steering probe — does `srq_post_recv` survive depth>4 where `qp_post_recv` didn't,
  and do completions land per-QP-CQ (spread preserved) or one SRQ-CQ (spread collapses)? Decides Stage-1 pivot +
  Stage-3 architecture in one run.

## ★ E3 RESULT — GREEN, GO (silicon, 2026-07-29)

DPA-heap 2-SGE gather blast (ZC=1, NOCRC, prefill, dmac→p0→BH), 3M frames:

| frame | EUs | Mpps | Gbps |
|---|---|---|---|
| 4 KB (4080) | 6 | 4.36 | 144.0 (reproduces the A5 ~146 G plateau) |
| 8 KB (8192) | 2 | 2.83 | **186.7** |
| 8 KB (8192) | 4 | 3.00 | **197.9** |
| 8 KB (8192) | 6 | 3.00 | 197.8 |

**Conclusion: the 144 G@4 KB plateau is the ~4.4 Mpps PPS wall, NOT a byte-bandwidth wall.** At 8 KB the same
DPA gather reaches **~198 G (200 G line rate) with just 4 EUs** (2 EUs already 187 G) — 0 dropped, 8238 B jumbo
egresses fine on p0 (the "BH 4 KB payload cap" worry does not apply to the p0 uplink). **The DPA egress path CAN
drive line rate ⇒ the d.2b DPA rewrite is JUSTIFIED; do NOT pivot to Arm-HW-TX.** E7 (RC-cap read) + E2/E4/E5/E6
still pending but the go/no-go is decided.

**Caveat (honest):** this is the pure **egress** ceiling from a prefill blast (no concurrent RC-recv drain, no
per-frame re-head from a real RoCE arrival, no depth-4 recv-post on the same EUs). The FULL gateway
(recv-drain + seq-patch + gather-egress on the same EUs) will run below this; Stage 2/3 must measure the loaded
per-EU rate. But the fundamental question — "can the DPA reach 200 G at 8 KB at all" — is now **YES**.

## ★★ STAGE 1 SOLVED — the depth-4 lock was the DBR umem (silicon, 2026-07-30)

**Root cause: `VERBS_SAMPLE_DBR_SIZE = 64` (too small).** The external-datapath DPA QP needs a **full 4 KB page**
for its DBR umem; at `rq_wr > 4` a 64 B DBR faults the DPA recv-post with **Fatal error (0x2) on RPC polling**
(`unpacked_process_call:726`) — a DPA *memory* fault at the first `post_recv`, decoded via SDK TRACE (E2).

Single-variable bisect on silicon (all at RQ target > 4):
| RQ/SQ | CQ | DBR | result |
|---|---|---|---|
| 8 | 8 | 64 | ❌ Fatal 0x2 |
| 8 | 8 | 64, sq_wr=0 | ❌ Fatal 0x2 (rules out shared-completion + send-side theories, Findings 1/3) |
| 8 | 8 | **4096** | ✅ 1000/1000, then **50000/50000 byte-exact, exactly-once** |
| 64 | 64 | **4096** | ✅ 5000/5000, then **50000/50000 byte-exact, exactly-once** (sq_wr=macro — DBR is the SOLE fix) |

**Finding 4 (DBR/umem) was RIGHT; Findings 1/3 (which dismissed the DBR as "8 B, irrelevant") were WRONG.** This
one line also retroactively explains the **d.1 "async-ring 0x2 fault"** (the WIP bumped depth without the DBR) —
it was never a kernel-loop bug, just the same undersized DBR. **The fix is `dpa_verbs_initiator_target_sample.c`:
`VERBS_SAMPLE_DBR_SIZE 64 → 4096`** (+ `VERBS_SAMPLE_QUEUE_SIZE 4 → 64`, completion tied to the macro). No SRQ
pivot needed for the depth fix. Recv-post is now **depth-parametric (RQ=64 validated)**.

**Still 0.37 Mpps** — the kernel posts 1 recv/iteration (1-in-flight). Deep RQ is now *available*; converting it
to throughput is Stage 2 (post N recvs + pipeline). SRQ remains the Stage-3 multi-EU buffer model.

## ★ Stage 2 IMPLEMENTATION STATUS (2026-07-30) — code done, egress/completion blocker

**Implemented + deployed + builds + connects + runs multi-frame** (repo working tree; DPU built):
- `common_defs.h`: `#define TT_RING (256)`.
- `sample.c`: landing ring `calloc(TT_RING, plen)` + `reg_mr TT_RING*plen` (SF advertised + PF gather); DPA-heap
  header **ring** `mem_alloc(TT_RING*TT_FRAME_HDR)` + every slot pre-filled with the template + `mmap` full len.
- `kernels_dev.c`: async-ring `target_thread_kernel` (slot=seq%TT_RING, post eth SEND without per-frame wait,
  non-blocking eth-CQE drain, `ETH_MAX_INFLIGHT=128` backpressure) + batch pre-post `TT_PREPOST=48` recvs.
- Requester: `tt_p15_requester` rebuilt with `TT_RING=256` to match (slots align).

**Proven working via DPA `DOCA_DPA_DEV_LOG_INFO`:** the RC-recv → re-head → eth-post pipeline **runs multi-frame**
(recv#0..#5+, `eth_posted` climbs 0→5+), and the first frame egresses byte-exact. So the ring gather + recv path
are correct.

**★ BLOCKER: ETH sends post but do NOT egress/complete.** `eth_done` stuck at **1**; **p0 tx delta = +2** over a
run (only ~2 frames leave p0). Frames pile in the SF ETH SQ un-transmitted → once `eth_posted-eth_done` hits 128
the backpressure loop deadlocks → requester stalls. Two candidate causes, **not yet isolated**:
1. **The async eth-completion handling** — the sync kernel *blocked* on each eth CQE (which paced + drained the
   SQ); the async non-blocking drain may need a different reap/re-arm (critic's `request_notification` open-Q), or
   the per-frame `commit_send` + varying-header-slot gather interacts badly.
2. **The DPU rebooted mid-session** (SF/host RoCE IPs were wiped + re-added; steering flow was dropped + re-added
   **COLD**). A cold SF→p0 eswitch flow / post-reboot eswitch state could be backpressuring the SF ETH SQ so sends
   don't complete. **The clean isolation (re-run the Stage-1 SYNC kernel post-reboot to confirm SF→p0 still
   egresses 50000) was botched** — `git stash` didn't move the *untracked* kernels_dev.c, so that test actually
   ran Stage-2 + the wrong requester. REDO it first next session.

**★ REFINED ISOLATION (2026-07-30, decisive): NOT steering, NOT the reboot.** With IPs restored + steering flow
present, a run gives **ovs `n_packets` +1** (frames do NOT reach the eswitch) and **p0 tx +2** — so the frames are
stuck in the **SF ETH SQ**, not dropped by a cold eswitch flow. DPA log: **frame 0 (slot 0) transmits fine
(eth_done 0→1); frames 1+ (slots 1+) never transmit.** ⇒ the bug is the **per-slot ring gather for slot>0**, not
the eth-completion reap and not the bench. Prime suspects, in order:
1. **Header-ring gather addressing/coverage for slot>0** — `gsge[0].addr = hdr_base + slot*tt_frame_hdr` with
   `lkey = tt_hdr_mkey` (the DPA-heap ring mmap). If the mmap/mkey only validly covers slot 0, or the per-slot
   `h2d_memcpy` fill didn't take, slot>0 gathers bad memory → the ETH send errors/drops → no transmit.
2. **Landing-ring gather for slot>0** — `gsge[1].addr = land_base + slot*tt_plen`, `lkey = tt_pay_mkey`.

**★★ PINPOINTED (2026-07-30): it's the async eth-completion / no-wait, NOT the ring gather.** Ran the slot-0
isolation (gather slot 0 always + single-slot requester, async no-wait kept): **still stalls at ~2 frames
(p0 tx +2).** So the per-slot ring gather is **fine** — the bug is purely the **async ETH-send completion
handling**. The sync kernel worked because it *blocked* on each eth CQE (`while(!get_completion(eth)); ack(1)`),
which both paced the SQ and reaped every completion; the async non-blocking drain
(`while(get_completion(eth)){ack;eth_done++}`) reaps only ~1 then `eth_done` sticks → the ETH SQ stops
transmitting (only ~2 frames egress) → backpressure deadlock. This is exactly the critic's flagged open-Q.
**THE FIX is in the eth-completion path** (`kernels_dev.c` drain). REVISED RANKING (overhead-aware — see below):
1. **DIAGNOSTIC (free, do first): read the eth CQE status bits** on the completion that arrives + any that follow.
   - #1 SUCCESS and #2+ **never arrive** → delivery/drain issue → apply fix (2).
   - #2+ arrive as **`DOCA_DPA_DEV_COMP_SEND_ERR`** → NOT a drain issue; it's a bad WQE → apply fix (3).
2. **blocking-drain-one-per-iteration** (reap exactly one eth CQE each loop like the sync kernel, but keep the
   recv side pipelined). **~zero extra overhead** (one `get_completion`/frame, already done) + restores 1:1
   post:reap pacing so the SQ always drains. **Primary fix.**
3. **`send_wr`-reuse fix**: the loop reuses ONE `send_wr` struct + reconfigures/re-posts it each iteration without
   waiting — the sync kernel waited between posts (struct free), the async may re-post while the prior send still
   references it. Use a fresh/cleared `send_wr` per post, or confirm `post_send_wr` copies the WR into the WQE.
4. **`doca_dpa_dev_completion_request_notification(eth_comp)` re-arm — DEMOTED.** It's the EVENT/wakeup primitive
   for a *sleeping* thread; our kernel is a persistent SPINNING poller (the sync kernel polled `get_completion`
   with NO notification and reaped 50000), so it likely doesn't fix a polling drain AND adds a per-frame
   arm-call (~1M/s at target, ~1–3% tax). Only relevant if redesigning to an event-driven (sleep/wake) thread.
Load the `doca-verbs`/`doca-dpa` skill for ETH-SQ completion semantics.
Ring gather (header + landing) and recv/deep-RQ paths are CONFIRMED WORKING — do not touch them.

**(superseded) earlier gather-vs-else framing:** (b) Check the ETH send **CQE status** for slot>0 (log `ce` error bits — `DOCA_DPA_DEV_COMP_
SEND_ERR`) to see if slot>0 sends error vs silently don't transmit. (c) Verify the DPA-heap header-ring mmap
(`tt_hdr_dpa_mmap`, `memrange_len=TT_RING*TT_FRAME_HDR`) and the per-slot `h2d_memcpy` loop actually populated all
slots (d2h read-back a couple slots). Load the `doca-verbs`/`doca-dpa` skill for ETH-SQ gather + completion
semantics. Stage-2 source backed up in scratchpad `stage2_backup/`; gw/dpa_rehead_verbs/*.c were **untracked**
before — the Stage-2 commit added `kernels_dev.c`+`common_defs.h` to tracking.

## Stage 2 implementation plan (scoped 2026-07-30 — port the async-ring onto the DBR-fixed base)

The `async_ring_wip` **kernel** (`async_ring_wip_kernels_dev.c` `target_thread_kernel`) IS the pipelined design and
is sound: landing+header rings (slot = `seq % TT_RING`), post recv → re-head into `hdr[slot]` → 2-SGE gather
`[hdr[slot]]+[land[slot]]` → post eth SEND (SIGNALED) **without per-frame wait**, non-blocking eth-CQE drain, block
only when `eth_posted - eth_done >= ETH_MAX_INFLIGHT (128)`. Its sole blocker (Fatal 0x2) is now fixed (the DBR).

**DO NOT resurrect the WIP sample as-is — it has a latent bug + cruft:**
1. **Header-ring not allocated (bug):** `async_ring_wip_sample.c` allocates the LANDING ring
   (`calloc(TT_RING, plen)`, ~L1217; `reg_mr` size `TT_RING*plen`) but only a **single** header
   (`calloc(1, TT_FRAME_HDR)` ~L1230; `doca_dpa_mem_alloc(TT_FRAME_HDR)` ~L1072) — while the kernel indexes
   `hdr + slot*tt_frame_hdr`. → out-of-bounds header read/patch. **Fix: allocate `TT_RING*TT_FRAME_HDR` DPA-heap
   header ring and pre-fill EVERY slot with the 46B template** (loop `h2d_memcpy`, or build the full ring on host
   then one copy); `mmap` `memrange_len = TT_RING*TT_FRAME_HDR`.
2. **Two-thread cruft:** drop `tt_produced_addr` / `tt_egress_notif_handle` (leftover from the abandoned
   two-thread coupling).

**Recommended: port onto the current DBR-fixed known-good `dpa_verbs_initiator_target_*` (not the WIP).** Edits:
- `..._common_defs.h`: add `#define TT_RING` (start 256; must satisfy `TT_RING > ETH_MAX_INFLIGHT + RQ_depth`).
- `..._sample.c` (`create_tt_rehead_resources`): landing `calloc(TT_RING, plen)` + `reg_mr TT_RING*plen` on SF
  (advertised — this is why the single-slot MR overran the ring requester) AND PF (gather); header ring
  `doca_dpa_mem_alloc(TT_RING*TT_FRAME_HDR)` + fill all slots + `mmap` full length; keep the DBR=4096 fix + RQ=64.
- `..._kernels_dev.c`: replace the synchronous `target_thread_kernel` with the async-ring loop; pre-post RQ-depth
  recvs in `target_trigger_first_iteration_rpc` (loop, not 1) so the deep RQ is actually filled.
- Requester: use the **ring** `tt_p15_requester` (TT_RING matching the target; the single-slot variant was only for
  the synchronous kernel). Requester slot `(i%TT_RING)` must equal kernel slot `(seq%TT_RING)` — aligned since RC
  recvs complete in order.

**Watch (critic's open Qs):** (a) `get_wqe_counter` is 16-bit → handle wrap in the `eth_posted-eth_done` math at
line rate; (b) confirm the unattached ETH CQ doesn't need `doca_dpa_dev_completion_request_notification` re-arm
under batched draining (else a silent hot-loop stall); (c) the RC recv CQ (completion `queue_size`) must be
`>= RQ depth` for N-in-flight (currently tied to the macro = 64 — OK for RQ=64). Gate: single-EU sweep materially
above 0.37 Mpps toward ~0.95 Mpps (~7.8 G@1KB, ~31 G@4KB), byte-exact, exactly-once, no RNR.

## Staged rewrite (Stage 1 DONE — Stage 2 next)

- **Stage 1 — depth-parametric recv-post.** Separate recv-only completion sized `==rq_wr` (`:539/:545/:1447`);
  RC `sq_wr=0` (E4-gated); batch pre-post `rq_wr` recvs + single `commit_recv` in
  `target_trigger_first_iteration_rpc`/`tt_post_recv`. Gate: RQ=CQ=8 then 32 pass the P-GW1 correctness run. If it
  still aborts → SRQ path.
- **Stage 2 — pipelined single-EU** (`target_thread_kernel`): post N recvs, non-blocking recv-drain (cap C/pass),
  signal-batch the ETH sends (`SIG_BATCH≈32`), async-reap ETH CQEs with `M_ETH` backpressure, DPA-DDR landing.
  **The async-WIP 0x2 was the Stage-1 first-post host-sizing fault, not a loop bug** — but this is asserted, not
  proven; E2 must confirm. Gate: single-EU sweep materially above 0.30 Mpps toward ~0.95 Mpps (never yet measured).
- **Stage 3 — N-EU fan-out:** SRQ shared buffer pool + per-QP recv-CQ 1:1 to its own EU + per-EU ETH-SQ +
  DPA-DDR landing; per-flow seq keyed by QPN; 8 KB frames × ~4 EUs. Gate: E2E ≥150 G (stretch 198 G), byte-exact,
  exactly-once, latency not regressed.

## Load-bearing UNVERIFIED assumptions (from the critic — do not build on these blind)

1. **~0.95 Mpps/EU** — never measured (real 0.30). The whole N-EU 200 G math multiplies an unrealized number.
2. **pps-wall vs byte-bandwidth-wall** — 178 G single-thread blast > 146 G multi-EU plateau ⇒ likely
   bandwidth-bound ⇒ 8 KB may NOT reach 200 G on the DPA path. **This is the top go/no-go risk → E3 settles it.**
3. **Remote RoCE WRITE_IMM into DPA-private DDR at rate** — unverified (E5).
4. **SRQ escapes depth-4 AND preserves per-QP completion spread** — both open (E6); it gates two stages.
5. **Batched ETH-CQ draining needs no `request_notification` re-arm** — untested; a silent-hang risk.
6. **`get_wqe_counter+1` retires the whole SIG_BATCH** — 16-bit counter wrap not handled in the pseudocode.

## Bottom line

The DPA line-rate rewrite is a real multi-stage effort, but **its go/no-go hinges on one cheap experiment (E3)
that needs no code change.** Run E3 (+ E7) before sinking any time into the Stage-1 recv-post rework. If E3 says
bandwidth-bound, the answer is Arm-HW-TX, not a DPA rewrite. The trivial completion-sizing fix is already
disproven (RQ=CQ=8 = FAIL).
