<!-- SPDX-License-Identifier: Apache-2.0 -->
# Option 3 — DPA-direct RoCE-CQ re-head: deep research plan / scope

**One line.** Make the **DPA poll the RoCE RC QP's completion queue directly**, so on every WRITE_IMM it
re-heads + egresses to the Blackhole with **zero Arm per-frame work and no doorbell/window toggle** — removing
the config-serialization wall that caps the doorbell drain at ~40G, and giving the lowest-latency gateway.

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
- **P1 — DPA polls a LOCAL RC QP's CQ (1–2 d).** Loopback/local RC pair; the DPA polls the RQ CQ and observes
  a WRITE_IMM recv completion. Isolates G2 from G1 (no external RoCE, no GID split yet).
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
P0 + P1 in parallel with an **option-1 prototype** (DPA-heap doorbell) as the hedge: P1 proves the DPA can drain
an RC CQ at all (low risk, high info); option-1 banks a scaling win while P2 (the risky co-location) is assessed.

_Companion: [[tt-rdma-dpa-rehead-plan]] (chronological), `gw/A3_rehead_plan.md` (the doorbell path + E2E)._
