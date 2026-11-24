# q_stem changelog

## Phase 0 (implementer)

Built per op_design.md R1/R2: one generic_op dispatch per call; x injector + Mcast to a column-split stage-1 rect;
q stays in L1 (cb_q_slice); port cores (row 0, 2 x num_links) pull remote head-rows and send them over FABRIC_2D
(2D node routes, ready fence, fused arrival increments) into the L1 landing scratch; own quarter delivered over the
NoC; stage 3 per head (matmul_block + eltwise copy), TILE output.

Deviations from the design (advisory items, recorded):
- Port placement: the centred row-0 layout of fabric_ring_exchange, not the Ethernet probe (the probe is an extra
  dispatch on first call). Lamp for later.
- Arrival counters count tiles; one fused increment per (destination core, row) group of segments (the design's
  `inc_every_packets` knob collapses to "per row" — the exchange measured hidden: ablating its payload saves 5 us).
- Raw fabric API on the port connection (FabricStream cannot express node-addressed 2D routes; one connection carries
  both 1-hop and 2-hop destinations).
- Injector split into a DRAM-read RISC (RISCV_0/NoC1) and a multicast RISC (RISCV_1/NoC0).
- Stage-1 rect sized to 2 columns per core (lamp L7) and load-weighted head placement (lamp L8); remote heads are
  consumed row-major across heads (all remote heads land with the same row profile), local heads head-major.
- s3_block_m_tiles is pinned to 1 (host assert): the landing row layout interleaves nope | rope per row.

Perf (QuietBox 2, 1x4 ring, Kimi K3.1 Phase-0 config: bf16, HiFi4, fp32 DEST; eval perf_measure.py, 5 experiments;
roofline 90.6 us, target 120.8 us):

| Version | fastest chip | slowest chip |
|---------|--------------|--------------|
| first passing build (99-core stage 1, head-major) | 223.7 us | 232.5 us |
| L7 + L8 + row-major remote heads + split injector + x depth 3 + flushed output pops | 190.9 us | 206.2 us |

Ablation breakdown of the first build (device kernel time, probe test, QSTEM_ABLATE defines): output writes -45 us,
x read+multicast -45 us, q_b read -38 us, wkv read -32 us, stage-1 matmul -18 us, fabric payload -5 us; every payload
stubbed at once: 131 us; plus q_b/wkv/fabric: 61 us; plus x multicast: 42 us (sync/launch floor).

Current breakdown (device zones S1/S3 on compute): q_b prologue ~31 us, stage 1 ends ~139 us (2-column cores are
compute-bound at HiFi4 + x supply), stage-3 tail to ~180 us = the stage-1 cores' heads after stage 1, bound by DRAM
output writes (ablating them: ~161 us). Next levers (not built): interleave stage-3 rows into the stage-1 compute loop
so the stage-1 cores' output writes overlap stage 1; K-split stage 1 (R6); a cheaper stage-3 row (per-row matmul
init + fp32 pack overheads).
