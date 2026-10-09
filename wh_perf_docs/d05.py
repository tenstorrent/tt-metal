from common import *

D = dict(
    id="WH-05", short="Zone Helper Size",
    summary="At #58068 head the profiler zone helpers sit in fixed, aligned slots, so other code cannot move them. But their own size still matters: one more instruction in zone_reserve moves 35 of 1,346 TILE_LOOP values by up to 17.8%. A harness change to the zone code is not neutral for the numbers.",
    status="Open · fix tested", status_cls="st-open",
    depends=["WH-01", "WH-03", "WH-04", "WH-08"], used_by=[],
    problem="no-work change (in the harness)",
    what="""<ul>
<li>One nop added to the body of <code>zone_reserve</code>: 35 values move by more than 2% (L1_CONGESTION[PACK] 23, L1_CONGESTION[UNPACK] 7, PACK_ISOLATE of sfpu_binop_scalar 5), up to 17.8%.</li>
<li>With the WH-04 settle fix in place: still 37 (L1_CONGESTION 30, eltwise_binary MATH_ISOLATE 7), up to 14.1%. So it is a separate problem.</li>
</ul>""",
    hw=f"""<ul>
<li><code>zone_reserve</code> ({code("tt_metal/tt-llk/tests/helpers/include/profiler.h", 155, "profiler.h")}) is 4 instructions: 16 bytes, exactly one cache line. The linker puts it at a 256-byte boundary ({code("tt_metal/tt-llk/tests/helpers/ld/sections.ld", 61, "sections.ld")}).</li>
<li>The zone constructor calls it right after the barrier release, before the start time read ({code("tt_metal/tt-llk/tests/helpers/include/profiler.h", 187, "profiler.h")}). The cache was just invalidated (WH-03), so the line is read from L1.</li>
<li>Cache lines are 16 bytes ({isa("TensixTile/BabyRISCV/InstructionCache.md", "ISA doc: instruction cache")}; RTL line width 128 bits).</li>
</ul>""",
    how="""<ol class="chain">
<li>After the release every thread calls <code>zone_reserve</code> and reads its code line from L1.</li>
<li>With one more instruction, the helper spans two lines: one more L1 read per thread, a few cycles later.</li>
<li>That changes the start timing of each thread by a few cycles (and adds one L1 access near the packers' start).</li>
<li>In L1_CONGESTION and L1_TO_L1, where all threads start together, the new timing selects a different packer rhythm (WH-01) or a different unpacker/packer L1 order.</li>
</ol>""",
    sketch="the 16-byte line boundaries around 0xe100, the 4-instruction helper inside one line, and the 5-instruction helper crossing into the next line.",
    versim="""<p>sfpu_binop_scalar (Float16_b, dest_acc No, ScalarAdd), PACK_ISOLATE, #58068 head with the barrier on, <code>zone_reserve</code> with 0 or 1 extra nop:</p>
<div class="tw"><table><tr><th>Versim</th><th class="n">0 nops</th><th class="n">+1 nop</th></tr>
<tr><td>Cycles (card: 5,010 / 5,900)</td><td class="n">5,010</td><td class="n">5,900</td></tr>
<tr><td>First DEST read (simulator cycle)</td><td class="n">63,559</td><td class="n">63,563</td></tr>
<tr><td>Unpack / math L1 accesses in tile 1</td><td class="n">20 / 26</td><td class="n">14 / 28</td></tr>
<tr><td>Tile length from tile 3, refusals of packers 1/2/3</td><td class="n">38, 3/4/6</td><td class="n">45, 3/5/11</td></tr></table></div>
<p>Every thread calls <code>zone_reserve</code> right after the release, so the extra line delays every thread by a few cycles (the packers start 4 cycles later). The idle threads' exit activity (WH-04) then lands at different cycles of the first tiles, and the packers settle in a different rhythm. In run types where all threads work (L1_CONGESTION), the same start shift changes the order of the unpackers' and packers' L1 accesses.</p>""",
    card="""<div class="tw"><table><tr><th>Change (card, #58068 head, 1,346 values)</th><th class="n">Moves</th><th class="n">Largest</th></tr>
<tr><td><code>zone_reserve</code> + 1 nop</td><td class="n">35</td><td class="n">17.8%</td></tr>
<tr><td>same, with the WH-04 settle</td><td class="n">37</td><td class="n">14.1%</td></tr>
<tr><td>unpack and math code +4 B per function, helpers moving / helpers kept in place (184 cases)</td><td class="n">36 / 16</td><td class="n">28.5% / 25.7%</td></tr></table></div>""",
    fix=f"""<p>Not fixed in #58068. df14044db9e (records after the end read, helpers out of line) is right to keep: it removed an L1 store from every window (202 values, up to 112%). The fragility comes with it.</p>
<p><b>Tested fix: reserve before the barrier.</b> Switch <code>LLK_ZONE_EARLY_RESERVE=1</code> (commit ee01ba8b0a6 on <code>nstojictt/p58-versim</code>): the zone's reserve call runs before the TILE_LOOP rendezvous, so after the release the zone start reads the clock with no helper call. Card, 1,346 values:</p>
<div class="tw"><table><tr><th>Change</th><th class="n">Without</th><th class="n">With early reserve</th></tr>
<tr><td><code>zone_reserve</code> +1 nop</td><td class="n">35 (37 with settle)</td><td class="n">0</td></tr>
<tr><td><code>zone_reserve</code> +4 nops</td><td class="n">–</td><td class="n">0</td></tr>
<tr><td>all code +4 B per function, with the WH-04 settle</td><td class="n">25 (max 6.2%)</td><td class="n">10 (max 6.3%; 9 L1_TO_L1)</td></tr></table></div>
<p><b>Versim, same config with early reserve:</b> 5,888 cycles with 0 and with +1 nop in <code>zone_reserve</code> (KERNEL 33,419 both). The whole run is shifted by one simulator cycle, and every tile after that is identical: the packers start at cycle 63,660 / 63,661 with the same refusals in tile 1 (0/8/16) and the same 36-cycle tiles after it. The helper now runs before the barrier, so its size can no longer move anything inside the window.</p>
<p><b>But it breaks the layout pads (WH-08).</b> <code>perf/layout.py</code> finds the measured window by the clock read that follows a call to <code>zone_reserve</code> ({code("tt_metal/tt-llk/tests/python_tests/helpers/perf/layout.py", 94, "layout.py Kernel._sites")}). Without that call it finds no window and pads nothing, so some math_matmul MATH_ISOLATE configs ran up to 78% slower. A first change to the window search (8aee8479f3a) restores part of the pads, not all. The layout model needs a proper way to find the window before this fix can go in.</p>
<p>Other options:</p>
<ul><li>Give the helpers padded, fixed-size slots, and check the perf values in CI whenever the helpers change.</li>
<li>Run the helper code that comes before a zone once before the barrier invalidates the cache, or keep it inline in the barrier's restart block, so it is not a cold L1 read inside the start of the window.</li></ul>""",
    ba="""<p>No fix yet. Before / after of the change that created it (df14044db9e): with the old record, 202 of 736 values were up to 112% higher (the store was inside the window); with the new one, the helper size matters as above.</p>""",
    open="""<ul><li>The early-reserve fix needs the layout model to find the window without the reserve call.</li></ul>""",
    repro="""<ul><li>Branch <code>nstojictt/p58-versim</code>: <code>LLK_ZONE_RESERVE_NOPS=1</code> (with or without <code>LLK_ISO_SETTLE=2000</code>); stress set: <code>repro_phase_tools/card_stress.sh</code>, compare with <code>advan.py</code>.</li></ul>""",
    refs=f"""<ul class="refs"><li>Code: {code("tt_metal/tt-llk/tests/helpers/include/profiler.h", 155)}, {code("tt_metal/tt-llk/tests/helpers/include/profiler.h", 160)}, {code("tt_metal/tt-llk/tests/helpers/ld/sections.ld", 61)}.</li>
<li>ISA docs: {isa("TensixTile/BabyRISCV/InstructionCache.md", "Instruction cache")}.</li></ul>""",
)
