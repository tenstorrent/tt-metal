from common import *

D = dict(
    id="WH-07",
    short="Branch-Type Cache",
    summary="Before the branch predictor is used, a small cache remembers which addresses hold a branch. It replaces entries at random, and the random number comes from a counter that steps on every clock. A loop with more branch addresses than the cache holds mispredicts a different number of times if it starts a few cycles earlier or later.",
    status="Partly fixed",
    status_cls="st-part",
    depends=[],
    used_by=[],
    problem="no-work change (through start timing)",
    what="""<ul>
<li>MATH_ISOLATE, matmul 2×1, Float16_b → Float32: 113,830 or 111,290 cycles, depending on one nop in the pack init. The pack thread does no work in MATH_ISOLATE.</li>
<li>Full suite (our stack, pack predictor off): about 170 math_matmul points in MATH_ISOLATE, UNPACK_ISOLATE and L1_TO_L1 moved by ±2–5% with +1 nop.</li>
</ul>""",
    hw=f"""<ul>
<li>16 entries, fully associative: {rtl("briscv/rtl/tt_predict_instrn_type.sv", 20)} <code>localparam DEPTH = 16</code>; tags {rtl("briscv/rtl/tt_predict_instrn_type.sv", 24)}.</li>
<li>On a miss, the entry at <code>rand_addr</code> is replaced: {rtl("briscv/rtl/tt_predict_instrn_type.sv", 47)} <code>tags[rand_addr[DEPTH_LOG2-1:0]] &lt;= i_update_pc</code>.</li>
<li><code>rand_addr</code> is a 5-bit LFSR ({rtl("briscv/rtl/tt_predict_instrn_type.sv", 55)}) that steps on every clock with no enable: {rtl("briscv/rtl/tt_lfsr.sv", 13)} <code>o_lfsr &lt;= {{o_lfsr[3:0], linear}}</code>, period 31.</li>
</ul>""",
    how="""<ol class="chain">
<li>The math matmul loop executes 68 different addresses; it keeps missing in the 16-entry branch-type cache.</li>
<li>Each miss replaces the entry that the LFSR points at in that clock cycle.</li>
<li>If the loop starts a few cycles later, every miss replaces a different entry, so different branches lose their "is a branch" mark and mispredict.</li>
<li>The pattern settles for the whole loop: some branches go from 3 mispredicts in 4 passes to every pass.</li>
</ol>""",
    sketch="a 16-slot cache, a ring of 31 LFSR states, and two time lines of the same misses starting 5 cycles apart, each picking a different slot.",
    versim=f"""<p>Versim, same config. The math code is identical in both builds. The math loop starts at cycle 46,897 (0 nops) or 46,902 (+1 pack nop), and the LFSR holds <code>11100</code> or <code>10001</code> at that moment. Cumulative math mispredicts over the loop:</p>
<div class="legend2"><span><span class="sw" style="background:var(--good)"></span>0 nops: 9,719 mispredicts, 111,064 cycles</span><span><span class="sw" style="background:var(--refc)"></span>+1 pack nop: 11,248 mispredicts, 113,594 cycles</span></div>
{fig(2)}
<p>Branch-type cache misses are the same in both runs (15,744 / 15,746); hits differ (21,649 / 24,603). Signals: <code>ifetch.predict_instrn_type.rand_addr</code>, <code>…hit</code>, <code>briscv.ex_bp_mispredict</code>.</p>""",
    card="""<div class="tw"><table><tr><th></th><th class="n">0 nops</th><th class="n">+1 pack nop</th></tr>
<tr><td>Card (cycles)</td><td class="n">113,830</td><td class="n">111,290</td></tr>
<tr><td>Versim (cycles)</td><td class="n">111,290</td><td class="n">113,822</td></tr></table></div>
<p>The same two values appear on both, swapped between the builds: the state is picked by a few cycles of start timing, and the card and Versim differ by a few cycles before the kernel starts.</p>
<div class="tw"><table><tr><th>Predictors off (card)</th><th class="n">MATH_ISOLATE configs moved by +1 pack nop (of 41)</th></tr>
<tr><td>pack only</td><td class="n">1</td></tr><tr><td>math only</td><td class="n">0</td></tr><tr><td>all three</td><td class="n">0</td></tr></table></div>""",
    fix=f"""<p><b>054096d8efa</b> releases all threads together from a flushed pipeline ({code("tt_metal/tt-llk/tests/helpers/src/brisc.cpp", 142, "brisc.cpp serve")}), so the start cycle repeats for the same build. A change that moves the start by a few cycles still selects another state. The LFSR runs on every clock, so no barrier can reset it.</p>
<p>Turning the math predictor off removes it (0 of 41), at the cost of a one-time shift of many values. A diagnostic flag is a possible option; the real fix is in hardware (replacement that does not depend on the clock).</p>""",
    ba="""<div class="tw"><table><tr><th>Full suite, +1 and +3 nops (our stack)</th><th class="n">Predictors on</th><th class="n">All predictors off</th></tr>
<tr><td>MATH / UNPACK / PACK_ISOLATE points moved &gt; 2%</td><td class="n">~300</td><td class="n">0 (largest 0.1% / 0.2% / 0.7%)</td></tr></table></div>""",
    open="""<ul><li>Why Versim and the card start the loop a few cycles apart is not measured (host and NoC timing before the kernel).</li>
<li>Not measured at #58068 head on the full suite.</li></ul>""",
    repro="""<ul><li>Sim branches <code>nstojictt/exp-b46-d0-sim</code> / <code>-d1-sim</code>, config2961 MATH_ISOLATE; predictor mask <code>REPRO_BP_MASK</code>.</li>
<li>Waveforms: <code>hs0|hs1.full.vcd.zst</code>.</li></ul>""",
    refs=f"""<ul class="refs"><li>RTL: {rtl("briscv/rtl/tt_predict_instrn_type.sv", 47)}, {rtl("briscv/rtl/tt_predict_instrn_type.sv", 55)}, {rtl("briscv/rtl/tt_lfsr.sv", 10)}, {rtl("briscv/rtl/tt_lfsr.sv", 13)}.</li>
<li>ISA docs: {isa("TensixTile/BabyRISCV/README.md", "Baby RISCV (branch prediction in the Frontend)")}.</li></ul>""",
)
