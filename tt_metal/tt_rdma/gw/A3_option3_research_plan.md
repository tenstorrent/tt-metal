<!-- SPDX-License-Identifier: Apache-2.0 -->
# Option 3 — DPA-direct RoCE-CQ re-head: deep research plan / scope

**One line.** Make the **DPA poll the RoCE RC QP's completion queue directly**, so on every WRITE_IMM it
re-heads + egresses to the Blackhole with **zero Arm per-frame work and no doorbell/window toggle** — removing
the config-serialization wall that caps the doorbell drain at ~40G, and giving the lowest-latency gateway.

## ★ Execution-model correction (2026-07-29) — the doorbell/RPC path is NON-NATIVE
Everything built so far (A1..A3.3b, the doorbell drain, the option-1 DPA-heap doorbell) drives the DPA via
`flexio_process_call` (a host→DPA **RPC**) and then **busy-polls** — spinning on the SQ-CQ owner bit and on a
software **doorbell**. That is NOT how the DPA is meant to run. The DPA is an **event-driven RTOS with
hardware-triggered, run-to-completion threads**: a thread is meant to be woken by a **CQE hardware interrupt**,
process, then re-arm/reschedule — full CPU/Arm bypass, minimal activation latency. Our software doorbell +
`flexio_window` poll was a workaround for the device-split, and it is exactly what introduced the per-batch
window↔outbox config toggle that serialized the threads (~40G wall). **Option 3 done natively removes the
doorbell entirely: the RoCE RQ CQE IS the trigger (hardware-delivered) — no poll, no config toggle, no Arm per
frame.** So option 3 is not just lower-latency; it is the *correct* DPA model, and it dissolves the serializer
as a side effect. (What the scaffold DID get right: the memory model — gather from **DPA-heap / NIC-private
DDR** scales to 134G, gather from host memory contends at ~42G, matching "DMA-gather into DPA caches backed by
internal DDR". Keep the landing buffer in DPA-private DDR.) Not-yet-used DPA facilities to fold in: **async DMA
engines** (host↔DPA movement, vs the contended host-MR gather) and **EU affinity** (`FLEXIO_AFFINITY_STRICT`,
one armed handler per EU).

## Why this is the production / lowest-latency target
- **Lowest latency:** the Arm leaves the per-frame path entirely. The DPA reacts to the RoCE completion itself
  — no Arm→doorbell→DPA hop, no `flexio_window` read.
- **Scales:** the ~40G ceiling of the doorbell drain is the per-batch `flexio_window`↔outbox config toggle
  (proven: it serializes DPA threads, N=6 → 15.7G). CQ-polling has no such toggle → it scales like the A5
  blast (~146G).
- **Zero-copy is already there** (A4 gather; CPU never touches payload) and is preserved. Correctness (RoCE→
  DPA→BH byte-exact) is already proven; this only changes the trigger + who drains the CQ.

## Settled context (do not re-derive)
- Architecture B: ConnectX terminates RoCEv2; DPA re-heads to TT-RDMA; BH is a TT-RDMA drainer (198G proven).
  Native-RoCE-on-BH stays rejected. See [[tt-rdma-rocev2-gateway-arch-b]], [[tt-rdma-dpa-rehead-plan]].
- Proven: A1 DPA egress ~178G; A4 zero-copy gather byte-exact; A5 blast fan-out ~146G; A3.1/3.2/3.3b RoCE→DPA→
  BH byte-exact E2E (with an Arm doorbell); pool lands 198G (raw HW-TX sender); AICLK boosts to 1350 under load.
- **The doorbell drain does NOT scale** (window-toggle serialization) — this plan is the way past that.

## API surface found on the bench (DOCA 3.4.0112 / FlexIO 26.04)
- **`/opt/mellanox/doca/include/doca_dpa_dev_verbs.h`** — DPA-side verbs: `doca_dpa_dev_verbs_qp_t`, RC QP
  (comment "Reliable Connection (RC) QP"), `doca_dpa_dev_verbs_recv_wr` + `_qp_post_recv_wr` +
  `_qp_post_send_wr`, CQ-poll helpers. This is the clean DPA-side RC responder API.
- **`flexio_qp_create` + `struct flexio_qp_attr`** (flexio.h) — creates an RC QP with a FULL RoCE transition
  surface: `next_state` (reset/init/RTR/RTS), `remote_qp_num`, `rgid_or_rip`, `gid_table_index`, `path_mtu`,
  `next_rcv_psn`/`next_send_psn`, `dest_mac`, `min_rnr_nak_timer`, `retry_count`, `udp_sport`,
  `qp_access_mask=IBV_ACCESS_REMOTE_WRITE`, and **`rq_cqn`** = a `flexio_cq_create`'d CQ the DPA polls.
- **DPA CQ polling is already demonstrated** — the stock packet_processor's `process_packet` polls its eth RQ
  CQ on the DPA (`flexio_dev_cqe_get_owner` + `com_step_cq`). An RC RQ CQ is the same mechanism.
- **Topology:** `pci/0000:03:00.0` eswitch = **switchdev**; the PF (mlx5_0, the DPA's function) has **NO RoCE
  GID**; the RoCE GID (10.99.0.1) is on the SF (mlx5_2). SFs cannot host a DPA (A0). **This split is the crux.**

## Gates (ranked by risk)
- **G1 — device-split (MAKE-OR-BREAK, HIGH).** The DPA lives on the PF (switchdev, no GID); RoCE needs a GID.
  Can a RoCE-GID RC QP have its RQ-CQ polled by the DPA? Three sub-paths:
  - (a) **Co-locate** a RoCE GID + the DPA on ONE function (eswitch/mlxconfig; likely reboot; may disturb the
    OVS/SF bench — assess blast radius).
  - (b) **Cross-function CQ** — the PF DPA polls a CQ belonging to an SF QP (cross-GVMI; likely disallowed —
    verify).
  - (c) **DPA-Verbs RC QP created directly with a GID** on the DPA's function (does doca_dpa_dev_verbs let the
    QP carry a RoCEv2 GID on the switchdev PF? probably needs (a)).
- **G2 — DPA-side RC responder API (MED).** Confirm `doca_dpa_dev_verbs` (or raw flexio QP + DPA CQ poll)
  supports a **responder** RQ that receives external WRITE_IMM with the CQ DPA-drained + the landed payload in a
  DPA-gather-usable MR. doca_dpa host-side context bring-up = the `doca-dpa` skill.
- **G3 — connection method / interop (MED).** `flexio_qp_attr` transitions the QP MANUALLY (remote_qpn/gid/psn
  via an OOB exchange), NOT RDMA CM. So a DPA-owned QP is not CM-bound → generic RDMA-CM apps need the manual
  exchange (or a bridge). **Design fork:** (i) manual OOB (matched requester, lowest latency) vs (ii) HYBRID —
  the Arm does the RDMA-CM handshake + QP creation for generic-app interop, but places the QP's RQ-CQ in
  DPA-pollable memory and the DPA drains it (control on Arm, data on DPA). The hybrid is the best-of-both if a
  CM-created QP's CQ can be handed to the DPA — VERIFY.
- **G4 — landed-payload access (LOW).** WRITE_IMM lands in the QP's MR (HW). The DPA re-heads via gather from
  that MR (A3.1 pattern: same buffer, DPA-gather lkey). Confirm the MR is reachable by the DPA gather WQE.
- **G5 — scaling / latency (LOW, the payoff).** N DPA threads polling (one CQ striped, or N QPs) — no window
  toggle → expect A5-like ~146G. Measure latency vs the doorbell path (expect lower: no Arm hop).

## Phased de-risk
- **P0 — API + topology survey (0.5–1 d).** Read doca_dpa_dev_verbs.h + flexio_qp_create in full; pick the
  DPA-RC-responder API. Enumerate G1 sub-paths concretely: which mlxconfig/eswitch change gives GID+DPA on one
  function, and its bench blast radius (OVS/SF). Reference: `/opt/mellanox/doca/samples/doca_dpa/`, doca-dpa
  skill, doca-rdma skill (connection methods), doca-verbs skill.
- **P1 (RE-CAST) — DPA event-handler HARDWARE-TRIGGERED on an RC RQ CQ (1–2 d).** NOT a busy-poll: create a
  `flexio_event_handler`, attach the RC QP's RQ CQ to that handler's thread (`rqcq_attr.thread =
  flexio_event_handler_get_thread(eh)`), so a WRITE_IMM recv CQE **hardware-triggers** the EU. The handler runs
  the re-head to completion, then `flexio_dev_cq_arm(rq_cq)` + `flexio_dev_thread_reschedule()` and sleeps until
  the next CQE. NO `flexio_process_call`, NO doorbell, NO window poll. **The stock packet_processor RX-reflect
  path is exactly this template (`create_app_event_handler` + `flexio_event_handler_run` + the RQ-CQ-triggered
  `flexio_pp_dev`/`process_packet`) — which the A1 blast BYPASSED by switching to the RPC. Go back to it and
  swap the eth RQ for an RC RQ.** Loopback/local RC pair first (isolates G2 from G1: no external RoCE, no GID
  split). Success = the DPA egresses a re-headed frame per RC WRITE_IMM with the EU sleeping between events
  (verify via msg-stream/trace, not a spin), and N handlers on N EUs (`FLEXIO_AFFINITY_STRICT`) run truly
  concurrently. This is the native DPA path; the doorbell scaffold is retired here.
- **P2 — resolve the device-split G1 (2–4 d, THE gate).** Get a RoCE-GID RC QP whose RQ-CQ the DPA polls
  (co-locate via mlxconfig, or cross-fn). If neither works → option 3 infeasible → fall back to option 1.
- **P3 — external RoCEv2 WRITE_IMM → DPA CQ (2–3 d).** Real external requester (manual QP exchange first),
  WRITE_IMM, DPA polls the CQ, sees the completion + the landed payload.
- **P4 — full re-head loop (1–2 d).** DPA: poll RC CQ → gather-re-head → ETH SQ egress → BH land, byte-exact,
  Arm doing ZERO per-frame work.
- **P5 — scale + latency (1 d).** N DPA threads → ~146G landed at the BH pool; measure end-to-end latency vs
  the A3.3b doorbell path; confirm no serialization. Then optionally the HYBRID (Arm RDMA-CM setup + DPA drain)
  for generic-app interop.

Estimate ~1.5–3 wk, dominated by G1 (P2). HIGH risk concentrated in G1.

## Decision / fallbacks
- **If G1 co-location is viable** (a function with GID+DPA, acceptable bench impact) → option 3 is the production
  path: lowest latency, Arm-off-path, scales.
- **If G1 is infeasible** (no GID+DPA co-location, no cross-fn CQ) → **fall back to option 1** (DPA-heap doorbell
  the Arm direct-stores; no window toggle → scales to ~146G; keeps a 2-store/frame Arm touch, RDMA-CM interop
  via the existing A3.3b responder). Option 1 is the safe scaling win and should be prototyped IN PARALLEL as
  the risk hedge.
- **Interop note:** DPA-owned QP (option 3) is manual-connection → for generic RDMA-CM apps use the HYBRID
  (Arm CM setup + DPA CQ drain) IF a CM QP's CQ can be DPA-mapped (G3-ii); else keep the A3.3b CM responder for
  CM apps and option-3 for matched high-perf peers.
- **Pair with B2 (MR federation)** for true per-destination RDMA semantics regardless of trigger.

## First concrete step
**P1 (re-cast): the hardware-triggered event-handler on an RC RQ CQ** — go back to the stock packet_processor
RX-handler template (which the A1 blast bypassed with the RPC) and swap the eth RQ for an RC RQ. This is the
native DPA model and is the highest-info first move: it proves the CQE-triggered re-head works and retires the
doorbell scaffold. Option-1 (DPA-heap doorbell) already banked the 134G scaling number and the memory-model
finding (keep the landing in DPA-private DDR) — that stands, but it's a busy-poll scaffold, not the production
form. P2 (the risky co-location for a GID+DPA function) is only needed if P1's RC-RQ-CQ can't be armed on the
DPA without co-location — the same G1 gate.

## P1 build recipe (execution-ready, from the reference sample)
**Base sample: `/opt/mellanox/doca/samples/doca_dpa/dpa_verbs_initiator_target/`** — a DOCA DPA RC-verbs
initiator+target. Its `device/..._kernels_dev.c: target_thread_kernel` is a DPA-side RC responder:
`doca_dpa_dev_verbs_qp_post_recv_wr` + `_commit_recv`, and it waits on completions via
**`doca_dpa_dev_get_completion(dpa_comp_handle, &comp)`** (a DPA completion context — the event-driven
mechanism; interrupt-backed, closer to native than our `flexio_window` spin). Host side uses `doca_verbs` +
`doca_rdma_bridge.h`, OOB-TCP exchange of GID + `remote_qp_number` → manual `doca_verbs_ah_attr` transition
(NOT RDMA CM — matches the interop note; a matched/OOB requester, or a CM bridge for generic apps).
**Adapt steps:**
1. Build the stock sample first (DPACC via `build_dpacc_samples.sh`) and run its loopback/2-host RC ping-pong —
   confirms the DPA RC target + `doca_dpa_dev_get_completion` path works on this BF3.
2. Replace the target kernel body: instead of "increment + WRITE back", on each recv completion do the
   **RE-HEAD** — build the 46B TT-RDMA header, gather-egress `[hdr]+[landed payload]` on an **ETH SQ** to the
   BH (reuse the A4 2-seg gather + the ds=4 patch), re-post the recv, ack the completion. The landing MR is the
   RC QP's recv buffer in **DPA-private DDR** (per the memory-model finding -> scales).
3. Add the ETH SQ + the TX-to-vport steering rule to the sample (from the packet_processor / our patch).
4. **G1 test inside P1:** create the RC QP bound to the **SF's** GID/device while the DPA context is on the PF
   (doca_verbs QP on the SF ctx; cf. `flexio_qp_create(process, ibv_ctx=SF, ...)` where ibv_ctx "might be
   different than process'"). If that binds + the PF DPA can `get_completion` on its CQ -> G1 solved WITHOUT
   co-location. If not -> co-locate (mlxconfig) or fall back to option-1.
5. Drive: external RoCEv2 WRITE_IMM (matched OOB requester) -> DPA target re-heads -> BH pool byte-exact;
   then N target threads on N EUs (`FLEXIO_AFFINITY_STRICT` / doca_dpa thread affinity) -> ~146G, and measure
   latency vs the doorbell path (expect lower — no Arm hop, no window toggle).
This is a fresh multi-hour build with its own DPACC build/iterate cycle — schedule as its own unit.

## Status of the scaffold vs the native path
- **DONE (scaffold, RPC+busy-poll):** RoCE→DPA→BH byte-exact E2E; option-1 DPA-heap doorbell scales to 134G.
  Proves the datapath, byte-exactness, and the memory model. NOT the native execution model.
- **NEXT (native, event-driven):** P1 re-cast — CQE-hardware-triggered event handler, no doorbell, no poll,
  no Arm per frame. This is the production / lowest-latency form.

## P1 EXECUTION RESULTS (2026-07-29) — ★★★ G1 SOLVED by `doca_dpa_device_extend`
The stock reference sample **`/opt/mellanox/doca/samples/doca_dpa/dpa_verbs_initiator_target`** already
implements the PF-DPA / SF-RoCE device split — and it works on this BF3. **G1 (the make-or-break HIGH-risk
gate) is retired with NO co-location, NO mlxconfig, NO reboot.**

The mechanism (host sample, `create_local_resources`, DOCA_ARCH_DPU path):
```c
open_doca_device(pf="mlx5_0", &pf_dev);                 // PF (DPA function, uplink p0 -> Blackhole)
open_verbs_resources(sf="mlx5_2", &verbs_ctx,&pd,&dev); // SF (RoCE GID 10.99.0.1)  <-- QP lives here
doca_dpa_create(pf_dev, &pf_dpa_ctx);                   // DPA context on the PF
doca_dpa_device_extend(pf_dpa_ctx, dev/*SF*/, &dpa_ctx);// *** extends PF DPA onto the SF device ***
create_verbs_qp(...set_send_dpa_completion / set_receive_dpa_completion...); // RC QP on SF, CQ -> DPA
```
`doca_dpa_device_extend` is the official DOCA API that lets a **PF-hosted DPA drive an SF device's RC QP
completions** — exactly the hybrid the plan hoped for (cf. the `flexio_qp_create(process, ibv_ctx=SF)` idea,
but at the clean DOCA layer). This dissolves the entire P2 co-location risk.

### What is validated on silicon
- **Build:** stock sample DPACC-compiles + links on the **DPU Arm (aarch64)** and the **x86 host** (meson+ninja,
  DOCA/DPACC 3.4.0112). Recipe: `cp -r /opt/mellanox/doca/samples ~/doca_samples; cp /opt/mellanox/doca/VERSION
  ~/VERSION` (meson wants `../../../VERSION`), then `meson setup build . && ninja -C build`.
- **G1 chain:** DPU target reaches `connect_verbs_qp` — i.e. `doca_dpa_device_extend` + `create_verbs_qp`
  (with DPA send/recv completions) on the **SF** + OOB param exchange all SUCCEEDED. The extended-DPA/SF-QP
  binding is real.
- **Device map (this BF3):** mlx5_0 = PF0 (DPA, uplink p0→BH); mlx5_1 = PF1; **mlx5_2 = port-0 SF
  (10.99.0.1, RoCEv2 gid idx1)**; mlx5_3 = port-1 SF (10.99.0.2, gid idx1). Host: mlx5_0 = enp193s0f0np0 =
  10.99.0.10 (RoCEv2 gid idx3), **host DPA `supported`** but PRM-process create needs root.

### What is NOT yet shown (bench-harness limits, not DPA-path problems)
- The `doca_dpa_dev_get_completion` **drain** with real traffic. Two harness blocks:
  1. **RoCEv2 local loopback fails AH MAC resolution** (`ibv_create_ah ret=110 ETIMEDOUT`, "get remote MAC")
     — two SFs on one host in one subnet route locally, so no wire MAC path (neighbors 10.99.0.1↔10.99.0.2 =
     FAILED; but 10.99.0.10 host over the wire = resolved). Expected RoCE self/local-loopback limit.
  2. **Host-side DPA needs root** (`flexio_prm_create_process Status 0x5`); desktop-0 sudo for arbitrary
     binaries is password-gated (only ib_write_bw/ib_send_bw/ip/ethtool… are NOPASSWD).
- **Unblock = a matched plain-ibverbs requester** (no DPA, no root — verbs is world-rw on the host) speaking
  the sample's trivial OOB (send/recv: buff_addr u64, rkey u32, qpn u32, gid.raw 16B; RTR mtu=1K psn=0
  RoCEv2, ack_to=14 retry=7). This is SIMPLER for the re-head target (external side just posts WRITE_IMM/SEND,
  no ping-pong back) and is the requester P1.4 needs anyway. → task P1.5.

### Next (P1.4 + P1.5, one unit)
Adapt `target_thread_kernel`: on each recv completion build the 46B TT-RDMA header + A4 2-seg gather-egress
`[hdr]+[landed payload]` on an **ETH SQ** to the BH (landing MR in DPA-private DDR), re-post recv, ack. Add the
ETH SQ + TX→vport steering (from the packet_processor patch / `dpa_ttblast`). Drive with the P1.5 requester
(host WRITE_IMM → DPU SF → DPA re-head → p0 → BH pool byte-exact), then N target threads on N EUs
(`FLEXIO_AFFINITY_STRICT`) for ~146G, measuring latency vs the A3.3b doorbell path.

_Companion: [[tt-rdma-dpa-rehead-plan]] (chronological), `gw/A3_rehead_plan.md` (the doorbell path + E2E)._
