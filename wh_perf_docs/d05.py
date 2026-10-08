from common import *

D = dict(
    id="WH-05",
    short="Zone Helper Size",
    summary="At #58068 head the profiler zone helpers sit in fixed, aligned slots, so other code cannot move them. But their own size still matters: one more instruction in zone_reserve moves 35 of 1,346 TILE_LOOP values by up to 17.8%. A harness change to the zone code is not neutral for the numbers.",
    status="Open",
    status_cls="st-open",
    depends=["WH-01", "WH-02", "WH-03"],
    used_by=[],
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
    versim="""<p>Not simulated yet. The mechanism is inferred from WH-03 (a code-line read at the start of the loop selects the rhythm), which is proven in Versim. In an earlier run, letting the helpers move by 4 bytes (which also makes them cross into a second line) added 20 moves (36 against 16 with the helpers kept in place), most of them unpack_tilize PACK_ISOLATE.</p>""",
    card="""<div class="tw"><table><tr><th>Change (card, #58068 head, 1,346 values)</th><th class="n">Moves</th><th class="n">Largest</th></tr>
<tr><td><code>zone_reserve</code> + 1 nop</td><td class="n">35</td><td class="n">17.8%</td></tr>
<tr><td>same, with the WH-04 settle</td><td class="n">37</td><td class="n">14.1%</td></tr>
<tr><td>unpack and math code +4 B per function, helpers moving / helpers kept in place (184 cases)</td><td class="n">36 / 16</td><td class="n">28.5% / 25.7%</td></tr></table></div>""",
    fix=f"""<p>Not fixed. df14044db9e (records after the end read, helpers out of line) is right to keep: it removed an L1 store from every window (202 values, up to 112%). The fragility comes with it. Options:</p>
<ul><li>Give the helpers padded, fixed-size slots, and check the perf values in CI whenever the helpers change.</li>
<li>Run the helper code that comes before a zone once before the barrier invalidates the cache, or keep it inline in the barrier's restart block, so it is not a cold L1 read inside the start of the window.</li></ul>""",
    ba="""<p>No fix yet. Before / after of the change that created it (df14044db9e): with the old record, 202 of 736 values were up to 112% higher (the store was inside the window); with the new one, the helper size matters as above.</p>""",
    open="""<ul><li>No Versim waveform for this case.</li>
<li>We did not test a fix.</li></ul>""",
    repro="""<ul><li>Branch <code>nstojictt/p58-versim</code>: <code>LLK_ZONE_RESERVE_NOPS=1</code> (with or without <code>LLK_ISO_SETTLE=2000</code>); stress set: <code>repro_phase_tools/card_stress.sh</code>, compare with <code>advan.py</code>.</li></ul>""",
    refs=f"""<ul class="refs"><li>Code: {code("tt_metal/tt-llk/tests/helpers/include/profiler.h", 155)}, {code("tt_metal/tt-llk/tests/helpers/include/profiler.h", 160)}, {code("tt_metal/tt-llk/tests/helpers/ld/sections.ld", 61)}.</li>
<li>ISA docs: {isa("TensixTile/BabyRISCV/InstructionCache.md", "Instruction cache")}.</li></ul>""",
)
