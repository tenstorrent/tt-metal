# write_throttle: cap in-flight NoC writes / DRAM reads (tilize) -- NULL

WH B0 n150, 64 Tensix cores, AICLK 1000 MHz (cycles == ns). Metric: DEVICE KERNEL DURATION [ns],
whole op, one fresh-JIT run per variant dir. Every cell below passed a bit-exact check (`torch.equal`).

Hypothesis: the op is over-subscribing the NoC/DRAM, so capping the bytes in flight would raise
throughput. **Result: refuted.** Write caps are flat (the most extreme one, 1 un-ACKed tile write,
costs +1..5%). Read caps and read-staging depth 1 make it slower, by up to +60%. The zones show the
reader's DRAM reads take the same time whether writes are capped or not.

## Variants (gen.py tokens; `+` combines them)
- `wsN`: sliding ACK window in `store_rows`. A write is issued only while fewer than N tile writes
  are un-ACKed (`noc_nonposted_writes_acked - NIU_MST_WR_ACK_RECEIVED`).
- `wbN`: `noc_async_write_barrier` after every N tile writes (lock-step batches).
- `wpK`: `riscv_wait(K)` cycles after each tile write (time stagger).
- `rsM`: sliding read window in `read_bank_coalesced`. At most M bank reads are un-landed
  (`noc_reads_num_issued - NIU_MST_RD_RESP_RECEIVED`).
- `@BANK_COALESCE_STAGE_DEPTH=1`: host knob. The next unit's reads are issued only after the
  previous unit has landed.
- `base` = the current kernels. `ctl` = the same kernels in a different dir (an A/A noise control).

## Focus shape [1,1,16384,64] bf16, DRAM interleaved (ns; base measured in the same session)
| variant | ns (session base) |
|---|---|
| ws1 | 25428 (24304), 24413 (24127) |
| ws2 | 24191 (24304), 25189 (25417), 23641 (23539), 25111 (24843), 24602 (24127) |
| ws3 | 24535 (24956), 23909 (24287), 24802 (23618) |
| ws4 | 24976 (24304) |
| ws8 | 24565 (24304), 23190 (25417), 23589 (23539), 24482 (24843), 24344 (23859), 24973 (24203), 23937 (24624) |
| ws16 | 24668 (24956), 23836 (24287), 24512 (23618) |
| wb1 / wb2 / wb4 | 24267 / 23778 / 24539 (23697); wb2 again 24985 (25417), 25479 (23539), 25437 (24843) |
| wp32 / wp64 / wp128 | 25105 / 24152 / 24255 (23513) |
| rs1 / rs2 / rs4 / rs6 | 38014 / 28602 / 26424 / 26034 (23885) |
| STAGE_DEPTH=1 | 24715 (23697) |
| ws2+rs6 / wb2+rs6 | 25810/26176/26175 ; 25774/27131/26772 (bases 24956/24287/23618) |
| reverse lever: WRITE_WINDOW_MIN_TILES=8 / 16 | 24976 / 23317 (24127) |
| reverse lever: STAGE_DEPTH=3 / 4 | 25013/24368/24076 ; 24660/25906/23924 (bases 24084/24748/23157) |
| ctl (A/A) | 23876, 23489, 23867, 24297, 23717, 24872 |

Session-to-session base spread: 23157..25417. The A/A control moves up to about 5% within one
session, so the noise band is ±3..5%.

## Domain sweep (ns; base / ws2 / ws4 / ws8 / rs6)
`rs6` only changes the bank-coalesced reader, so it is a second A/A control on the other shapes.
| case | base | ws2 | ws4 | ws8 | rs6 |
|---|---|---|---|---|---|
| 32768x64 | 45290 | 46590 | 45710 | 46986 | 47321 |
| 16384x32 | 14311 | 14512 | 14487 | 14451 | 16548 |
| 8192x256 | 44697 | 45689 | 43893 | 43376 | (46889 ctl) |
| 2048x1024 | 46758 | 47367 | 48284 | 46323 | (47174 ctl) |
| 128x64 (co-read, 8 cores) | 2650 | 2696 | 2838 | 2614 | (2772 ctl) |
| 32x2048 (2-D split) | 3838 | 3893 | 3586 | 3692 | (3822 ctl) |
| 4096x64 L1->L1 | 6326 | 6318 | 6516 | 6174 | (6452 ctl) |
| 2048x512 height-sharded->DRAM | 17349 | 18101 | 18001 | 16398 | (17423 ctl) |

ws8 repeats (base / ws8 / ctl), 3 sessions:
- 16384x64: 23859/24344/23876, 24203/24973/23489, 24624/23937/23867
- 32768x64: 46903/47361/45809, 47462/46169/46614, 46612/46593/45346
- 8192x256: 43899/43752/43884, 44901/45561/43130, 43170/44277/43551
- height-sharded->DRAM: 17247/17312/17176, 16788/16696/17689, 16655/17954/16830
- 2048x1024: 45910/46587/45682, 45824/47456/46768, 47115/46807/46698

The medians agree within 1%. The 5% "win" at height-sharded->DRAM in the one-off sweep was noise.

## Mechanism (zones.py, focus shape, per-core sums in cycles, p50)
| | writer_issue | writer_flush | writer_wait | NCRISC-KERNEL | reader_barrier | BRISC-KERNEL max |
|---|---|---|---|---|---|---|
| base | 4310 | 3143 | 8849 | 15594 | 3716 | 23537 |
| ws1 | 13326 | 512 | 5949 | 15084 | 3457 | 23828 |
| ws2 | 7172 | 2190 | 7218 | 15104 | 3520 | 23973 |

Capping writes only moves the writer's stall from the flush into the issue loop. The reader's time
(NCRISC-KERNEL, reader_barrier) does not change, even at one write in flight. So writes are not
starving the reader's DRAM reads per core. The reader is the critical path: NCRISC max about 19 us,
then about 4.5 us of compute+write tail. Throttling that reader's reads makes it slower (rs*): the
reads are latency-bound, and they need every in-flight read they have. Aggregate DRAM traffic is
4 MiB in about 21 us of active time, about 200 GB/s. That is near the practical roofline in either
direction: deeper read or write windows are flat too.

Files: gen.py (variant generator), rows.py (ops-CSV -> test id mapper), ws8.diff (hook diff),
test: tests/ttnn/unit_tests/operations/tilize/test_tilize_perf1_write_throttle.py.
Caveat: more than about 5 variant dirs in one --profile session can trigger the profiler's
"Source location hashes are colliding" throw (16-bit zone hash over path:line). Batch variants.
