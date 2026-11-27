# What graduated from t2_tail_write_stream (Perf 1 of tournament 2)

The op graduated the **hop policy with default endpoints**: per core, the whole tail goes on the NoC with fewer torus hops to that NoC's own endpoint of the core's bank. The reader uses NoC0 and the bank's NoC0 endpoint; the writer uses NoC1 and the NoC1 endpoint. On ties, `TAIL_NC_PCT` applies. Measured te_gate 380.4 µs alone; together with the raw h×w K-walk, 360 µs.

`graduate_tail_policy.patch` in this dir is the subagent's version. It also lets the writer address the bank's **NoC0 endpoint from NoC1** (`ep_noc`, 376.4 µs). That part was NOT graduated:
- It is cross-NoC endpoint use.
- The mirror case (NoC0 into the NoC1 endpoints) corrupted DRAM.
- The NoC0 endpoint is CMFW's DRAM-telemetry port (`blackhole_140_arch.yaml`, SYS-1419).

See changelog.md, "Perf 1 (tournament 2)".
