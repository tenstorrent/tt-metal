from common import *

FX = "{{FX}}"

D = dict(
    id="WH-04", short="Idle Threads",
    summary="In an isolate run type only one thread does work, but the other two still run their exit code while it is measured. Right after the barrier their instruction caches are empty, so they read every code line from L1, and those reads push the packers into the slow rhythm (WH-01). A 4-byte move of the unpack or math code changes PACK_ISOLATE by up to 28.6%.",
    status="Fixed in #58068 (52412ea8e67) · checked", status_cls="st-ok",
    depends=["WH-01", "WH-02", "WH-03"], used_by=[],
    problem="no-work change",
    what="""<ul>
<li>At #58068 head, 4 never-executed bytes in front of every function move 75 of 1,346 TILE_LOOP values by more than 2%, up to 28.6%: 48 PACK_ISOLATE, 2 MATH_ISOLATE, the rest L1_TO_L1 and L1_CONGESTION.</li>
<li>Moving only the pack code: 3 moves (L1_TO_L1). Moving only the unpack and math code: the same 75.</li>
<li>The measured pack loop and its callee are byte-identical, at the same addresses, in both builds.</li>
</ul>""",
    hw=f"""<ul>
<li>The barrier invalidates all TRISC instruction caches before it releases the threads ({code("tt_metal/tt-llk/tests/helpers/src/brisc.cpp", 165, "brisc.cpp")}), so every thread starts the zone with an empty cache ({isa("TensixTile/BabyRISCV/InstructionCache.md", "ISA doc: instruction cache, invalidation")}).</li>
<li><b>Which L1 port each packer writes through</b> (WH RTL): packer 0 has its own port ({rtl("tensix/rtl/tt_tensix.sv", 1631)}); packer 1 shares port 1 with the scrubber and unpacker 1 ({rtl("tensix/rtl/tt_tensix.sv", 1211)}); packers 2 and 3 go through the two TDMA round-robin arbiters ({rtl("tdma/rtl/tt_tdma.sv", 3848)}, {rtl("tdma/rtl/tt_tdma.sv", 3855)}), whose outputs share <b>port 2</b> with NCRISC, TRISC0 (unpack) and BRISC ({rtl("tensix/rtl/tt_tensix.sv", 1304)}) and <b>port 3</b> with TRISC1 (math) and TRISC2 (pack) ({rtl("tensix/rtl/tt_tensix.sv", 1398)}). So every L1 access of a TRISC, including its code fetches, competes with packer 2 or packer 3 for a port. The public port diagram is in {isa("TensixTile/L1.md", "ISA doc: L1")}.</li>
<li>Code fetches are 128-bit L1 reads on the same L1 ports as the packers' writes ({isa("TensixTile/L1.md", "ISA doc: L1")}).</li>
<li>Isolate run types have no exit barrier ({code("tt_metal/tt-llk/tests/helpers/include/counters.h", 549, "counters.h exit_barrier_for")}); the idle threads leave TILE_LOOP at once, for example unpack in PACK_ISOLATE: <code>return;</code> ({code("tt_metal/tt-llk/tests/sources/math_matmul_test.cpp", 85, "math_matmul_test.cpp")}). They park again only after <code>run_kernel</code> ({code("tt_metal/tt-llk/tests/helpers/src/trisc.cpp", 116, "trisc.cpp")}).</li>
</ul>""",
    how="""<ol class="chain">
<li>BRISC releases all three threads together, with empty instruction caches.</li>
<li>Pack starts its loop. Unpack and math have nothing to do: they close their TILE_LOOP zone, run <code>zone_record</code>, the end of <code>run_kernel</code> and part of <code>main</code>.</li>
<li>Each of those instructions comes from a code line that is read from L1: 28 lines (unpack) and 32 (math) in the first 200 cycles.</li>
<li>The packers start about 115 cycles after the release, so 7–8 unpack reads and 14 math reads land while the packers write their first tiles. One of them hits a packer write (WH-02), and the rhythm is set (WH-01).</li>
<li>Move the idle threads' code by 4 bytes: the reads land a few cycles earlier, unpack reads one more line, and a different rhythm is set.</li>
</ol>""",
    sketch="three time lines from the release: pack (loop starts at ~115), unpack and math (a burst of code-line reads 0–200). Mark where the bursts overlap the first packer writes.",
    versim=f"""<p>#58068 head with the barrier on in Versim (<code>LLK_SIM_BARRIER=1</code>), math_matmul config12601 PACK_ISOLATE (Float32 → Float16, HiFi2, 2 tiles, loop factor 1024). 71,873 / 92,263 cycles, the same as the card.</p>
<div class="legend2"><span><span class="sw" style="background:var(--accent)"></span>code line read from L1</span><span><span class="sw" style="background:var(--good)"></span>DEST read accepted</span><span><span class="sw" style="background:var(--refc)"></span>refused</span><span>dashed line = first DEST read</span></div>
{fig(15, "Fast (71,873): the idle threads' code-line reads at these cycles.")}
{fig(16, "Slow (unpack and math code +4 bytes per function, 92,263): the reads land a few cycles earlier, unpack reads one more line (0x5200), and packer 1 is refused in the second tile.")}
<div class="tw"><table><tr><th>Over the loop</th><th class="n">Fast</th><th class="n">Slow</th></tr>
<tr><td>Pack core mispredicts / code-line reads</td><td class="n">2,047 / 7</td><td class="n">2,047 / 7</td></tr>
<tr><td>Cycles per loop pass</td><td class="n">70</td><td class="n">90</td></tr>
<tr><td>Pack core waits on a full instruction queue</td><td class="n">54,976 cycles</td><td class="n">75,357 cycles</td></tr>
<tr><td>Packers 1 / 2 / 3 refused (60,000 cycles)</td><td class="n">1,714 / 1,714 / 1,714</td><td class="n">3,999 / 6,665 / 14,666</td></tr></table></div>
{fig(17, "Fast, mid-loop: 35 cycles per tile.")}
{fig(18, "Slow, mid-loop: 45 cycles per tile.")}
{FX}""",
    card="""<div class="tw"><table><tr><th>Code moved by 4 bytes per function</th><th class="n">184 cases: moves</th><th class="n">323 cases: moves</th><th class="n">Largest</th></tr>
<tr><td>all threads</td><td class="n">36</td><td class="n">75</td><td class="n">28.6%</td></tr>
<tr><td>pack only</td><td class="n">0</td><td class="n">3 (L1_TO_L1)</td><td class="n">5.4%</td></tr>
<tr><td>unpack and math only</td><td class="n">36</td><td class="n">75</td><td class="n">28.6%</td></tr></table></div>""",
    fix=f"""<p><b>Fixed in #58068 by 52412ea8e67 (9 October).</b> In PACK_ISOLATE, unpack and math now wait at the end of their TILE_LOOP zone until pack is back from <code>run_kernel</code> and flips the release level once more. Their exit code (zone record, return, park) runs only after the measured loop. Only PACK_ISOLATE builds do this. The flag they compare sits after every other variable, so no other address moves.</p>
<div class="tw"><table><tr><th>Check at the new head 24ffcccb24c (card, 276 PACK_ISOLATE values)</th><th class="n">Hold on</th><th class="n">Hold off (<code>LLK_NO_QUIET=1</code>)</th></tr>
<tr><td>Unpack and math code +4 B: values that move &gt; 0.5%</td><td class="n">0 (largest 1 cycle)</td><td class="n">54, ±22–28%</td></tr>
<tr><td>config12601, 0 / +4 B, card</td><td class="n">73,850 / 73,849</td><td class="n">92,260 / 92,260</td></tr>
<tr><td>config12601, 0 / +4 B, Versim</td><td class="n">73,850 / 73,849</td><td class="n">– / 92,260</td></tr></table></div>
<p>Details: WH-10. Our earlier fix is below for reference; it is not needed now.</p>
<p><b>Our earlier tested fix.</b> Tested fix, switch <code>LLK_ISO_SETTLE=N</code> (commits c403b67d07f, 58b9837efcc on <code>nstojictt/p58-versim</code>): in an isolate run type, after the TILE_LOOP release and before its zone opens, the measured thread runs N nops ({code("tt_metal/tt-llk/tests/helpers/include/counters.h", 628, "counters.h perf_counter_scoped")} is where it goes). The idle threads finish their exit code in that time. The wait is outside the measured window.</p>
<p>Better version, not built yet: the measured thread waits until the idle threads have parked, in place of a fixed count.</p>""",
    ba="""<div class="tw"><table><tr><th>Card, #58068 head</th><th class="n">Before</th><th class="n">After (settle 2,000)</th></tr>
<tr><td>All code +4 bytes per function: values that move (1,346)</td><td class="n">75, max 28.6%</td><td class="n">25, max 6.2%</td></tr>
<tr><td>… of them in isolate run types</td><td class="n">50</td><td class="n">0</td></tr>
<tr><td>config12601 PACK_ISOLATE, 0 / +4 bytes</td><td class="n">71,873 / 92,263</td><td class="n">71,791 / 71,791</td></tr>
<tr><td><b>Full suite</b> (CI, 842,256 points), +4 bytes per function: PACK_ISOLATE TILE_LOOP values that move &gt; 2%</td><td class="n">13,537 (up to 28.4%)</td><td class="n">0 (largest 0.8%)</td></tr></table></div>
<p>Full-suite arms: #58068 head 3ff45cbd1c7 against branch <code>nstojictt/p58-final</code> b0d4d554446 (head + this settle + the two removals of WH-09 and the fixed address), each with and without one nop before every function, two CI runs each. One time, head → final: 22,560 PACK_ISOLATE values get faster by more than 2% and 454 slower (median −1.5%), because the packers no longer start in the slow rhythm.</p>
<p>The 25 that still move are L1_TO_L1 and L1_CONGESTION, where unpack and math really run the moved code. The settle changes 133 values once (up to 22.4%): it moves the measured loop and its start.</p>""",
    open="""<ul><li>The hold covers PACK_ISOLATE only. In UNPACK_ISOLATE and MATH_ISOLATE at the new head, all code +4 B moves 1 value by more than 0.5% and 0 by more than 2%.</li>
<li>UNPACK_ISOLATE and MATH_ISOLATE: only 2 values moved, so pack as the idle thread is not a problem in this set; not proven for every module.</li></ul>""",
    repro="""<ul><li>Branch <code>nstojictt/p58-versim</code>: <code>LLK_FN_NOPS_THREADS=UNPACK,MATH LLK_THREAD_FN_NOPS=1</code>; fix: <code>LLK_ISO_SETTLE=2000</code>; Versim: <code>LLK_SIM_BARRIER=1 LLK_SIM_TIMEOUT=14400</code>.</li>
<li>Test: <code>perf_math_matmul.py::test_perf_math_matmul[MathFidelity.HiFi2-matmul_config12601-5-1]</code>, <code>LLK_PERF_RUN_TYPES=PACK_ISOLATE</code>.</li>
<li>Waveforms: <code>pf0|pf1.full.vcd.zst</code> (no fix), <code>fx0|fx1.full.vcd.zst</code> (with the settle).</li></ul>""",
    refs=f"""<ul class="refs"><li>Code: {code("tt_metal/tt-llk/tests/helpers/include/counters.h", 549)}, {code("tt_metal/tt-llk/tests/helpers/include/counters.h", 612)}, {code("tt_metal/tt-llk/tests/sources/math_matmul_test.cpp", 85)}, {code("tt_metal/tt-llk/tests/helpers/src/trisc.cpp", 116)}, {code("tt_metal/tt-llk/tests/helpers/src/brisc.cpp", 165)}.</li>
<li>ISA docs: {isa("TensixTile/BabyRISCV/InstructionCache.md", "Instruction cache")}, {isa("TensixTile/L1.md", "L1")}.</li></ul>""",
)
