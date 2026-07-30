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

## Staged rewrite (ONLY if E3 passes)

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
