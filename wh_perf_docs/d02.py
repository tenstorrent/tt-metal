from common import *

D = dict(
    id="WH-02", short="L1 Accesses While Packing",
    summary="One L1 access by any agent other than the packers, at the wrong cycle while they run, moves the packers into the slow rhythm of WH-01. When the access comes at a random cycle (the host polling L1), the same build gives a different value on every run.",
    status="Fixed at #58068 head", status_cls="st-ok",
    depends=["WH-01"], used_by=["WH-04", "WH-05"],
    problem="re-measure (random cycle) and no-work change (fixed cycle)",
    what="""<ul>
<li>On main, a full Wormhole run gives the same value on rerun for only 68.7% of points.</li>
<li>With the host's L1 polling put back at #58068 head, one config gives five values in five runs: 88,263 / 78,073 / 84,713 / 71,873 / 83,553 cycles.</li>
<li>The values fall between the fast and the slow level of WH-01, because the access lands at a random tile and the loop is slow from there to the end.</li>
</ul>""",
    hw=f"""<ul>
<li>L1 has 16 banks behind 16 access ports, and clients share ports through round-robin muxes ({isa("TensixTile/L1.md", "ISA doc: L1")}). The packers' writes, the TRISCs' loads, stores and code fetches, and the NoC's reads of L1 all go through these ports.</li>
<li><b>Which L1 port each packer writes through</b> (WH RTL): packer 0 has its own port ({rtl("tensix/rtl/tt_tensix.sv", 1631)}); packer 1 shares port 1 with the scrubber and unpacker 1 ({rtl("tensix/rtl/tt_tensix.sv", 1211)}); packers 2 and 3 go through the two TDMA round-robin arbiters ({rtl("tdma/rtl/tt_tdma.sv", 3848)}, {rtl("tdma/rtl/tt_tdma.sv", 3855)}), whose outputs share <b>port 2</b> with NCRISC, TRISC0 (unpack) and BRISC ({rtl("tensix/rtl/tt_tensix.sv", 1304)}) and <b>port 3</b> with TRISC1 (math) and TRISC2 (pack) ({rtl("tensix/rtl/tt_tensix.sv", 1398)}). So every L1 access of a TRISC, including its code fetches, competes with packer 2 or packer 3 for a port. The public port diagram is in {isa("TensixTile/L1.md", "ISA doc: L1")}.</li>
<li>A TRISC load or store is a narrow access; a code fetch is a 128-bit read ({isa("TensixTile/L1.md", "L1, RISCV bandwidth")}).</li>
<li>One accepted access by another client can make a packer's L1 write wait one cycle. That is the push in WH-01 step 3.</li>
</ul>""",
    how="""<ol class="chain">
<li>The packers run in the fast rhythm.</li>
<li>Another client (the math core, the host over the NoC, BRISC) reads L1 in the cycle before packer 3 writes.</li>
<li>The L1 arbiter takes only one of the two packer write packets in the next cycle; packer 3's write waits one cycle.</li>
<li>From the next tile the packers are in the slow rhythm (WH-01), and they stay in it to the end of the loop.</li>
<li>If the access lands while packer 3 is between tiles (not writing), nothing changes. These "safe" windows repeat with the tile period.</li>
</ol>""",
    sketch="one tile of the four packers' L1 write lanes, an arrow from the math core's L1 read into the cycle before packer 3's write, and the shifted diagonal in the next tile. Then the same read moved into packer 3's gap.",
    versim=f"""<p>Repro with one ELF: the math core does one L1 read at spin iteration K; the host writes K before the run. Signals: the math core's <code>o_l1_rden</code>, each packer's <code>o_l1_wren</code> / <code>i_l1_req_ready</code>, the DEST crossbar.</p>
<div class="legend2"><span><span class="sw" style="background:var(--accent-soft)"></span>cycle the math read is accepted</span><span><span class="sw" style="background:var(--accent)"></span>math core L1 read</span><span><span class="sw" style="background:var(--good);opacity:.6"></span>packer L1 write accepted</span><span><span class="sw" style="background:var(--refc)"></span>L1 write waits / DEST read refused</span></div>
{fig(10, "K = 100 (slow, 9,323 cycles): packer 3's L1 write waits one cycle after the math read; from the next tile packers 1, 2, 3 wait 1, 1, 4 cycles per tile.")}
{fig(11, "K = 200 (fast, 9,070 cycles): the read lands while packer 3 is between tiles. Nothing changes.")}
<h3>Tile by tile</h3>
{fig(12, "Packer 3 refusals per tile (first 64 tiles). The read at tile 11 switches the loop to the slow rhythm for the rest of the run.")}
<h3>The random case cannot be simulated</h3>
<p>The host-poll config in Versim gives 71,873 cycles in both runs, and the waveform shows no NoC read of L1 during the kernel (<code>overlay_noc_nius_routers.o_mem_out_rden</code> = 0). The simulator host does not reach L1 through the NoC and the L1 arbiters. The hand-placed read above is the Versim model of the random case.</p>""",
    card="""<div class="tw"><table><tr><th>K (math core L1 read)</th><th class="n">0</th><th class="n">100–175</th><th class="n">200</th><th class="n">225–350</th><th class="n">375</th><th class="n">400</th></tr>
<tr><td>Card = Versim (cycles, 14 values of K)</td><td class="n">9,070</td><td class="n">9,323–9,314</td><td class="n">9,070</td><td class="n">9,309–9,294</td><td class="n">9,070</td><td class="n">9,289</td></tr></table></div>
<p>A nop, a branch or a wall-clock register read at the same place changes nothing; an L1 load or store does.</p>
<h3>Every source of L1 traffic we found during a measured loop</h3>
<div class="tw"><table><tr><th>Source</th><th>Card effect</th><th>#58068 change</th></tr>
<tr><td>Host polls TRISC mailboxes and the BRISC counter in L1</td><td>5 runs: 71,873 to 88,263; 63–71 of 200 values differ run to run (up to 27%)</td><td>b58fbfbd6d2</td></tr>
<tr><td>BRISC reads its command mailbox every 1 µs</td><td>54 of 736 values move once (−6.0% to +17.9%); in long kernels a nop moved 32 of 64 configs</td><td>e02b20f56ee</td></tr>
<tr><td>Zone start record stored in L1 inside the window</td><td>202 of 736 values change once, up to 112%</td><td>df14044db9e</td></tr>
<tr><td>Reader thread polls L1 for the measured zone (older harness)</td><td>one more L1 poll in the loop</td><td>not present at head</td></tr>
<tr><td>Idle threads read their exit code from L1</td><td>see WH-04</td><td>open</td></tr></table></div>""",
    fix=f"""<ul>
<li><b>b58fbfbd6d2</b>: the firmware also writes the completion flags and the BRISC counter to <code>STREAM_SCRATCH_0</code> of overlay streams 0..3 ({code("tt_metal/tt-llk/tests/helpers/include/boot.h", 70, "boot.h host_signal")}), and the host polls those registers ({code("tt_metal/tt-llk/tests/python_tests/helpers/device.py", 271, "device.py host_signal_address")}). Register reads do not use the L1 ports.</li>
<li><b>e02b20f56ee</b>: BRISC polls its command every 100 µs and spins on NOPs in between ({code("tt_metal/tt-llk/tests/helpers/src/brisc.cpp", 298, "brisc.cpp poll_period_us")}).</li>
<li><b>df14044db9e</b>: the zone reads the start time last in the constructor ({code("tt_metal/tt-llk/tests/helpers/include/profiler.h", 187, "profiler.h")}) and writes both records after the end read, in out-of-line helpers ({code("tt_metal/tt-llk/tests/helpers/include/profiler.h", 160, "zone_record")}).</li>
</ul>""",
    ba="""<div class="tw"><table><tr><th>Case (card)</th><th class="n">Before</th><th class="n">After (#58068 head)</th></tr>
<tr><td>config885 PACK_ISOLATE, 5 runs / 2 runs</td><td class="n">71,873 … 88,263</td><td class="n">71,873, 71,873</td></tr>
<tr><td>Rerun, 323 test cases (1,346 values)</td><td class="n">on main: 68.7% identical (full suite)</td><td class="n">0 values move</td></tr>
<tr><td>184 test cases, two identical runs</td><td class="n">49 of 736 move (host poll back)</td><td class="n">0</td></tr></table></div>""",
    open="""<ul><li>The port sharing is traced in the RTL (section 2). The host reads L1 over the NoC through other ports (4–7, 12–15), so they meet the packers at the banks, not at a port; that bank-level step is not traced.</li>
<li>Any new harness or firmware L1 access during a kernel brings this back. A CI check of rerun identity on a sample would catch it.</li></ul>""",
    repro="""<ul><li>Branch <code>nstojictt/p58-versim</code>: <code>LLK_HOST_POLL_L1=1</code> (host polls L1 again), <code>LLK_BRISC_OLD_POLL=1</code> (1 µs BRISC poll), <code>LLK_ZONE_OLD=1</code> (start record in the window).</li>
<li>Test: <code>perf_math_matmul.py::test_perf_math_matmul[MathFidelity.LoFi-matmul_config885-5-1]</code>, <code>LLK_PERF_RUN_TYPES=PACK_ISOLATE</code>, run 5 times.</li>
<li>K probe: branch <code>nstojictt/versim-m1-repro</code>, <code>repro_phase.sh hw|sim K=…</code>; waveforms <code>k0|k100|k200|k225.full.vcd.zst</code>.</li></ul>""",
    refs=f"""<ul class="refs"><li>Code: {code("tt_metal/tt-llk/tests/helpers/include/boot.h", 73)}, {code("tt_metal/tt-llk/tests/python_tests/helpers/device.py", 271)}, {code("tt_metal/tt-llk/tests/helpers/src/brisc.cpp", 296)}, {code("tt_metal/tt-llk/tests/helpers/include/profiler.h", 155)}.</li>
<li>ISA docs: {isa("TensixTile/L1.md", "L1")}, {isa("TensixTile/BabyRISCV/README.md", "Baby RISCV")}.</li>
<li>Earlier pages: K probe experiment, root-cause page.</li></ul>""",
)
