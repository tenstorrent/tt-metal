from common import *

D = dict(
    id="WH-08",
    short="Instruction Cache Sets",
    summary="The math TRISC's instruction cache is 512 bytes: 2 ways of 16 sets of 16-byte lines. If three hot code lines of a loop sit 256 bytes apart, they share a set and evict each other on every pass, and the loop reads code from L1 on every pass. One matmul config runs 27% slower this way.",
    status="Avoided by layout pads",
    status_cls="st-ok",
    depends=[],
    used_by=[],
    problem="no-work change",
    what="""<ul>
<li>math_matmul config936, MATH_ISOLATE, at #58068 head: 88,211 cycles with the layout pads, 111,741 without, on the card and in Versim.</li>
<li>Removing the pads moves 56 of 100 math_matmul MATH_ISOLATE configs by more than 0.5%, up to 26.7%.</li>
<li>The #58068 text: moving INIT out of line made math_matmul MATH_ISOLATE up to 87% slower, with 21,522 modelled misses against 22.</li>
</ul>""",
    hw=f"""<ul>
<li>2 ways: {rtl("briscv/rtl/tt_risc_wrapper.sv", 542)} <code>WAY_COUNT = 2</code>. Sets: {rtl("briscv/rtl/tt_risc_wrapper.sv", 543)} <code>WAY_ADDR_WIDTH = (LARGE_TRISC_ICACHE == 2) ? 6 : … : 4</code>.</li>
<li>TRISC0 and TRISC2 get the large cache, TRISC1 (math) does not: {rtl("tensix/rtl/tt_tensix.sv", 3304)} <code>.LARGE_TRISC_ICACHE((trisc_id==0) ? 2 : (trisc_id == 2) ? 2 : 0)</code>. So math has 16 sets × 2 ways × 16 bytes = 512 bytes; unpack and pack have 2 KiB.</li>
<li>Replacement: least recently written way ({rtl("briscv/rtl/tt_icache_tags.sv", 115)}, {rtl("briscv/rtl/tt_icache_tags.sv", 144)}).</li>
<li>The public ISA doc gives the same sizes ({isa("TensixTile/BabyRISCV/InstructionCache.md", "Instruction cache: 2 KiB, 2 KiB, 512 bytes")}).</li>
</ul>""",
    how="""<ol class="chain">
<li>A line's set is address bits 4–7 on the math core, so lines 256 bytes apart share a set.</li>
<li>Without the pads, the loop has three hot lines in set 12 (<code>0xa2c0</code>, <code>0xa3c0</code>, <code>0xa4c0</code>) and three in set 13 (<code>0xa2d0</code>, <code>0xa3d0</code>, <code>0xa4d0</code>).</li>
<li>Each set keeps two lines. The third one evicts the oldest, which is needed next, and so on: all six miss on every pass.</li>
<li>Each miss is an L1 read the math core waits for; the pass takes 109 cycles instead of 86.</li>
</ol>""",
    sketch="a 16-row × 2-column grid (sets × ways) and the loop's lines dropped into rows by address bits 4–7; rows 12 and 13 get three lines each.",
    versim=f"""<div class="tw"><table><tr><th>Versim, config936</th><th class="n">Pads on</th><th class="n">Pads off</th></tr>
<tr><td>Cycles</td><td class="n">88,211</td><td class="n">111,741</td></tr>
<tr><td>Cycles per loop pass</td><td class="n">86.1</td><td class="n">109.1</td></tr>
<tr><td>Code lines read from L1 in 1,023 passes</td><td class="n">26 (first pass)</td><td class="n">6,159 (6 per pass)</td></tr>
<tr><td>Instruction cache misses (<code>perf_cnt_mshr</code>)</td><td class="n">60</td><td class="n">7,215</td></tr>
<tr><td>Mispredicts</td><td class="n">8,183</td><td class="n">8,185</td></tr></table></div>
<p>The predictor is the same in both builds, so the difference is the instruction cache alone.</p>
<div class="legend2"><span><span class="sw" style="background:var(--accent)"></span>math core reads a code line from L1</span><span><span class="sw" style="background:var(--refc)"></span>mispredict</span><span><span class="sw" style="background:var(--good)"></span>instruction to the math unit accepted</span><span>dashed = loop pass start</span></div>
{fig(3, "Pads on: two passes, 172 cycles, no code line read from L1.")}
{fig(4, "Pads off: two passes, 218 cycles, six code lines read from L1 in every pass.")}""",
    card="""<div class="tw"><table><tr><th>Card, #58068 head</th><th class="n">Pads on</th><th class="n">Pads off</th></tr>
<tr><td>config936 (cycles)</td><td class="n">88,211</td><td class="n">111,741</td></tr>
<tr><td>100 math_matmul configs: values that change &gt; 0.5%</td><td class="n">–</td><td class="n">56 (up to 26.7%)</td></tr></table></div>""",
    fix=f"""<p><b>3ff45cbd1c7</b>: <code>perf/layout.py</code> runs the measured window of each unpack and math thread in a small RV32IM interpreter, models the predictor and this cache ({code("tt_metal/tt-llk/tests/python_tests/helpers/perf/layout.py", 350, "layout.py, 2 ways")}, {code("tt_metal/tt-llk/tests/python_tests/helpers/perf/layout.py", 334, "cost")}), and picks NOP pads before the loop restart point and after the zone end ({code("tt_metal/tt-llk/tests/python_tests/helpers/perf/layout.py", 423, "choose")}). The pads never run inside the window. For config936 it picked math P = 224, Z = 20. Applied to unpack and math only ({code("tt_metal/tt-llk/tests/python_tests/helpers/test_config.py", 1862, "LAYOUT_THREADS")}).</p>
<p><b>054096d8efa</b> keeps the loop at its address mod 512 (WH-03), so a change outside the loop does not move its lines against each other.</p>""",
    ba="""<div class="tw"><table><tr><th>config936 MATH_ISOLATE</th><th class="n">Before (no pads)</th><th class="n">After (pads)</th></tr>
<tr><td>Card / Versim</td><td class="n">111,741 / 111,741</td><td class="n">88,211 / 88,211</td></tr>
<tr><td>Code lines from L1 per pass</td><td class="n">6</td><td class="n">0</td></tr></table></div>
<p>The pads do not make values more stable against code moves (36 moves with or without pads); they avoid a bad layout.</p>""",
    open="""<ul><li>The model's choice is checked here for one config. Whether it picks a conflict-free layout for every config is not measured.</li>
<li>A change inside the measured loop can still create a conflict; the pads are chosen again for the new code.</li></ul>""",
    repro="""<ul><li>Branch <code>nstojictt/p58-versim</code>: <code>LLK_NO_PADS=1</code> (pads off) or <code>LLK_FORCE_PADS="math:P,Z"</code>; Versim <code>LLK_SIM_BARRIER=1</code>.</li>
<li>Test <code>perf_math_matmul.py::test_perf_math_matmul[MathFidelity.LoFi-matmul_config936-0-1]</code>, <code>LLK_PERF_RUN_TYPES=MATH_ISOLATE</code>; analysis <code>ican.py &lt;run.cyc&gt; 1</code>.</li>
<li>Waveforms: <code>m1c0|m1c1.full.vcd.zst</code>.</li></ul>""",
    refs=f"""<ul class="refs"><li>RTL: {rtl("briscv/rtl/tt_risc_wrapper.sv", 543)}, {rtl("tensix/rtl/tt_tensix.sv", 3304)}, {rtl("briscv/rtl/tt_icache_tags.sv", 144)}.</li>
<li>Code: {code("tt_metal/tt-llk/tests/python_tests/helpers/perf/layout.py", 22)}, {code("tt_metal/tt-llk/tests/python_tests/helpers/test_config.py", 1862)}.</li>
<li>ISA docs: {isa("TensixTile/BabyRISCV/InstructionCache.md", "Instruction cache")}.</li></ul>""",
)
