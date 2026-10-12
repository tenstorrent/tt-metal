import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

OUT = os.path.dirname(os.path.abspath(__file__)) + "/out"
TREE = "e28a67d1df7"  # mvlahovic exp-final code under test (bf6b1e1b6ca + hooks)


def gh(n):
    return f'<a href="https://github.com/tenstorrent/tt-metal/issues/{n}">#{n}</a>'


ROWS = [
    ("1", "BRISC firmware grows by 2 KB", "Not run", "st-part", "The moving values are about 1.1M cycles. Versim runs about 3k cycles per minute, so one run would take more than a day."),
    ("2", "BRISC releases the threads 1 cycle later (scheduled form)", "Explained", "st-ok", "The release times stay the same, but the branch-type cache LFSR phase of math and pack moves (WH-07). Math gets 2 fewer mispredict cycles and ends 8 cycles earlier."),
    ("3", "2 NOPs in the unpack setup / 1 NOP in the math setup, perf_pack_dest_bank", "Explained", "st-ok", "Unpack: branch predictor aliasing in the unpack INIT's own tile loop (WH-06). Math moves with unpack's change because the two INITs overlap. The math NOP is hidden in a stall."),
    ("4", "1 NOP in the math setup, perf_reduce Row Average Float32 → Bfp8_b", "Explained", "st-ok", "Cold instruction fetch: the same 42 code lines in the same order, but the NOP moves every instruction 4 bytes inside them, and the core stalls 6 more cycles."),
    ("5", "2 NOPs in the unpack setup, perf_unpack_transpose", "Does not reproduce here", "st-part", "On bgd-lab-t3005 all 24 launches move by +2, the correct answer. CI showed +3 in 4 launches."),
    ("6", "384 B of harness code that never runs (perf counters), eltwise_binary Elwmul Float32", "Does not reproduce here", "st-part", "0 of 576 INIT launches change on bgd-lab-t3005."),
    ("7", "2 NOPs per packed tile, or extra stand-in work (L1_CONGESTION mocks)", "Not run", "st-part", "Opt-in mocks; needs the exact config."),
]


def body():
    rows = "".join(f'<tr><td class="n">{a}</td><td>{b}</td><td><span class="status {cls}">{c}</span></td><td>{d}</td></tr>' for a, b, c, cls, d in ROWS)
    return f"""<title>WH-11 Open Cases In Versim</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>{A['css']}{EXTRA_CSS}</style>
<div class="wrap">
<header><div class="docid">WH-11 · Wormhole LLK perf · <a href="{URLS.get('WH-00')}">WH-00 Index</a></div>
<h1>The seven open cases, checked in Versim</h1>
<p class="lede">A list of the numbers that still move on the full Wormhole suite (seven cases). For each one we found the configuration on a lab card, then ran the same ELFs in Versim and compared the two builds cycle by cycle. Where Versim ran, it gave the same values as the card, to the cycle.</p></header>

<section><h2>Result</h2><div class="tw"><table><tr><th>#</th><th>What changes</th><th>Status</th><th>Why</th></tr>{rows}</table></div>
<p><b>Setup.</b> Code under test: {TREE} (the code of the final experiment branches, bf6b1e1b6ca plus the experiment hooks), with the hook flags of the experiment legs in <code>LLK_EXP_CFLAGS</code>. Card: bgd-lab-t3005; a rerun of the same build is exact. Versim: the same ELFs, branch <code>nstojictt/mv-final-versim</code> (section 6).</p></section>

<section><h2><span class="n">2</span>Case 2: BRISC releases the threads 1 cycle later</h2>
<p>Test: perf_transpose_dest, Float16_b → Float16_b, math transpose on, L1_TO_L1. Change: <code>-DLLK_RELEASE_GAP_SCHED=1</code>.</p>
<div class="tw"><table><tr><th>Cycles</th><th class="n">Card</th><th class="n">Versim</th></tr>
<tr><td>TILE_LOOP, gap 0 / gap 1</td><td class="n">2,351 / 2,343</td><td class="n">2,351 / 2,343</td></tr>
<tr><td>KERNEL, gap 0 / gap 1</td><td class="n">72,803 / 72,797</td><td class="n">72,803 / 72,797</td></tr></table></div>
<ol class="chain">
<li><b>The release times do not change.</b> In both builds unpack, math and pack start the TILE_LOOP zone at the same cycles (math 141 cycles and pack 1,081 cycles after unpack).</li>
<li><b>The LFSR phase changes.</b> Each TRISC has a free-running 5-bit LFSR that picks the entry to replace in its branch-type cache (WH-07). The BRISC flush resets it. At the release, unpack's LFSR value is the same in both builds; math's is one step further and pack's two steps further (for example math 6, 12, 24 … against 12, 24, 16 …).</li>
<li><b>Different replacement, different mispredicts.</b> In the TILE_LOOP zone math has 14 cycles of mispredicts with gap 0 and 12 with gap 1; pack 3 and 1. Code-line fetches stay the same (81 against 79 for math).</li>
<li><b>Math is the last thread.</b> Its last instruction comes 8 cycles earlier with gap 1, and that is the −8 (−0.34%).</li></ol>
<p>This is the mechanism {gh(59997)} describes, now seen cycle by cycle. A fix has to keep the cycles from each thread's flush to its release the same in every build, not only the release spacing.</p></section>

<section><h2><span class="n">3</span>Case 3: NOPs in the unpack or math setup, perf_pack_dest_bank</h2>
<p>Config: Float16_b, l1_acc No, 1 block of 8 tiles, loop factor 8, PACK_ISOLATE. In this test the unpack INIT is long (card 7,873 cycles): unpack's INIT body itself unpacks the test's tiles, in a loop. The card shows two groups of configs (−25 and −1); the table is the −25 group, the −1 group is below it.</p>
<div class="tw"><table><tr><th>INIT, delta against base</th><th class="n">Card</th><th class="n">Versim</th></tr>
<tr><td>+2 NOPs in the unpack setup: unpack</td><td class="n">−25</td><td class="n">−25</td></tr>
<tr><td>+2 NOPs in the unpack setup: math (math code unchanged)</td><td class="n">+1</td><td class="n">+1</td></tr>
<tr><td>+1 NOP in the math setup: math</td><td class="n">0</td><td class="n">0</td></tr></table></div>
<h3>Unpack −25: branch predictor aliasing in the INIT's tile loop (WH-06)</h3>
<p>The two NOPs move the unpack INIT body, and its tile loop, by 8 bytes. The loop has four branches. With the predictor's hash (address bits 2–8, 16 entries, no tag):</p>
<div class="tw"><table><tr><th>Build</th><th>Branch addresses → entry</th><th class="n">Mispredicts in the INIT</th></tr>
<tr><td>base</td><td class="m">0x56e0 → 2, 0x56f4 → 1, 0x56f8 → 2, 0x5738 → 6</td><td class="n">46</td></tr>
<tr><td>+2 NOPs</td><td class="m">0x56e8 → 0, 0x56fc → 3, 0x5700 → 2, 0x5740 → 10</td><td class="n">27</td></tr></table></div>
<p>In base, 0x56e0 and 0x56f8 share entry 2 and overwrite each other's target on every pass. After the move all four have their own entries. 19 fewer mispredicts give the −25. Code-line fetches are 79 and 80, so it is not the cache.</p>
<p><b>The −1 group</b> (1 block of 4 tiles): the same aliasing, 26 → 19 mispredicts, but the INIT only moves by −1 (card and Versim: unpack 7,704 → 7,703, math 284 → 283). Here the unpack thread spends 7,310 of its 10,398 cycles blocked on a full instruction queue: the unpacker hardware sets the pace, and the saved mispredict cycles turn into 8 more blocked cycles. The math INIT again overlaps the end of the unpack INIT (by about 190 cycles).</p>
<h3>Math +1 without a math change: the INITs overlap</h3>
<p>The unpack INIT window in Versim runs from cycle 261,406 to 271,972; math's from 271,615 to 272,035. Math starts while unpack still runs its last ~350 cycles, so "each INIT runs alone" is not true for this test. Unpack's shifted loop moves math's wait on its instruction queue by one cycle. To make each INIT alone in every test, BRISC should release the next INIT when the previous thread's INIT has ended, not after a fixed number of spins.</p>
<h3>Math NOP gives 0: the NOP is hidden</h3>
<p>In this INIT the math thread is blocked for 151 cycles on a full instruction queue. The NOP is one more committed instruction (150 against 149), and it runs in one of those blocked cycles (150 blocked afterwards). The INIT length does not change. This is correct; a NOP only costs a cycle when the thread is not waiting.</p></section>

<section><h2><span class="n">4</span>Case 4: 1 NOP in the math setup, perf_reduce</h2>
<p>Config: Row Average, Float32 → Bfp8_b, MATH_ISOLATE. Card and Versim: math INIT 225 → 230 (+5, expected +1).</p>
<div class="tw"><table><tr><th>Math INIT window (Versim)</th><th class="n">Base</th><th class="n">+1 NOP</th></tr>
<tr><td>Cycles</td><td class="n">216</td><td class="n">221</td></tr>
<tr><td>Code-line fetches from L1</td><td class="n">42</td><td class="n">42 (the same lines, in the same order)</td></tr>
<tr><td>Mispredicts</td><td class="n">4</td><td class="n">4</td></tr>
<tr><td>Cycles with no instruction committed</td><td class="n">71</td><td class="n">77</td></tr>
<tr><td>Cycle of the last fetch (the jump back to the end read)</td><td class="n">209</td><td class="n">214</td></tr></table></div>
<p>INIT runs from a cold instruction cache, so the core waits for almost every 16-byte code line. The NOP adds one instruction, and it moves every later instruction 4 bytes inside the same lines. The lines are the same, but the point where the core has to wait for the next line moves, so the overlap of fetch and execution changes. Here that costs 6 more stall cycles, spread over the body and largest at the jump back to the end read (+3). Net +5. This is {gh(59996)}: it needs INIT code placement that does not depend on unrelated code size (model-chosen INIT pads, or fetch-line alignment inside the INIT section).</p></section>

<section><h2><span class="n">5</span>Cases 5 and 6: not reproduced on this card</h2>
<ul><li>Case 5, perf_unpack_transpose, +2 NOPs in the unpack setup: +2 in all 24 launches (correct).</li>
<li>Case 6, eltwise_binary Elwmul Float32, with perf counters compiled in: 0 of 576 launches change.</li></ul>
<p>Both were seen on CI runners. The lab card gives the expected answer, so these may depend on the runner card or on a different config sample. CI and lab cards gave different values for some residual movers before (WH-00).</p></section>

<section><h2><span class="n">6</span>Versim on the current tree</h2>
<ul><li><b>Branch</b> <code>nstojictt/mv-final-versim</code>: the code under test plus experiment switches for Versim. <code>LLK_SIM_BARRIER=1</code> uses the barrier start. <code>LLK_SIM_INPROC=1</code> runs the simulator in the test process as the UMD host (core reset crashes in the client role). <code>LLK_SIM_INIT=1</code> and <code>LLK_SIM_WARMUP=1</code> keep the INIT launch and the warm-up pass. <code>LLK_SIM_LOG=1</code> traces the BRISC commands. Register reads, which Versim cannot do, are skipped.</li>
<li><b>Simulator:</b> a Versim build with UMD simulation-server support (fmarkovic's tt-umd-simulators), relinked with the Verilator error limit raised. Without that, an RTL assertion in the L1 atomic adder (<code>tt_int_add.sv:428</code>, "Signed INT32 results not matching") stops the simulation in the second launch, and the host waits on a BRISC counter forever. We did not find which L1 atomic access trips it.</li>
<li><b>Speed:</b> about 3k cycles per minute for one simulation, half that with nine in parallel. One perf_pack_dest_bank run takes about 4 hours; one reduce run 6 hours. The VCD goes through a pipe into the signal extractor, so nothing large is written to disk.</li>
<li><b>Tools</b> (branch <code>nstojictt/wh-perf-issue-drafts</code>, <code>wh_perf_docs/</code>): <code>initwin.py</code> finds a thread's INIT window and aligns two runs instruction by instruction; <code>mpcount.py</code> counts mispredicts per branch; <code>fetchseq.py</code> lists the code-line fetches; <code>lfsr.py</code> prints the branch-type cache LFSR; <code>vrun3.sh</code> runs one Versim case.</li></ul></section>
<p class="small">Status 13 October 2026. Card bgd-lab-t3005; Versim on yyzeon09; results in <code>/proj_sw/user_dev/nstojic/versim-runs/</code> (yyz) and <code>/proj_sw/user_dev/nstojic/mvc/</code> (bgd).</p>
</div>
"""


if __name__ == "__main__":
    open(f"{OUT}/WH-11.html", "w").write(link_shas(body()))
    print("ok")
