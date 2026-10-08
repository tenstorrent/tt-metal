from common import *

D = dict(
    id="WH-03",
    short="Pack Loop First Tiles",
    summary='The first tiles of a pack loop decide the packer rhythm (WH-01) for the whole loop. A change that does no work can change when the pack core reads its last code lines from L1, or when its first instructions reach the packers. Either one can select the slow rhythm. This is the "one nop moves PACK_ISOLATE by up to 46%" effect of #58919.',
    status="Fixed for code outside the loop",
    status_cls="st-ok",
    depends=["WH-01"],
    used_by=["WH-05"],
    problem="no-work change",
    what="""<ul>
<li>One nop in <code>_llk_pack_init_</code> moved 14,750 PACK_ISOLATE points by more than 2% on the full suite (#58068 before the barrier, pack predictor on; 3,415 with the pack predictor off).</li>
<li>matmul 1×3 PACK_ISOLATE: 107,622 cycles with 0 nops, 110,705 with +1 nop, on the card and in Versim.</li>
<li>eltwise_binary: pack pad 1 gives 4,669 cycles, pad 0 gives 4,741.</li>
</ul>""",
    hw=f"""<ul>
<li>The pack TRISC fetches code into its instruction cache from L1, one 16-byte line at a time, through the same L1 ports as the packers' writes ({isa("TensixTile/L1.md", "ISA doc: L1, RISCV instruction fetches")}; {isa("TensixTile/BabyRISCV/InstructionCache.md", "Instruction cache: 2 KiB on TRISC0 and TRISC2")}).</li>
<li>After the first pass a short pack loop always hits in its 2 KiB cache. In Versim: 7–8 line reads in 256 passes (repro), 7 in 1,024 passes (#58068 head).</li>
<li>The pack TRISC sends instructions to the packers through a queue; when the queue is full, the TRISC waits (<code>i_trisc_instrn_buf_req_ready</code> low).</li>
</ul>""",
    how="""<ol class="chain">
<li>The pack loop starts cold: its last code lines are not in the cache yet.</li>
<li>Case A, code fetch: if the loop ends in one more 16-byte line, that line is read from L1 during the first tile, exactly when a packer writes. The packer's write waits; WH-01 step 3 follows.</li>
<li>Case B, instruction timing: the layout changes the cycle at which the first instructions reach the packers, so the four packers start with a different spacing. No extra L1 access is needed.</li>
<li>Either way the rhythm is set in tile 0–2 and holds for the rest of the loop.</li>
</ol>""",
    sketch="the pack core's code lines as boxes on an address line (16 B each), the loop's end crossing one more line with +1 nop, and a time line of tile 0 with the late fetch landing on packer 3's write.",
    versim=f"""<h3>Case A: the pack core's own code fetch (matmul 1×3, PACK_ISOLATE, pack predictor off)</h3>
<div class="legend2"><span><span class="sw" style="background:var(--accent)"></span>pack core reads a code line from L1</span><span><span class="sw" style="background:var(--muted);opacity:.35"></span>packer L1 write accepted</span><span><span class="sw" style="border-color:var(--refc)"></span>refused</span><span>dashed line = first DEST read</span></div>
{fig(13, "0 nops (107,622 cycles): the last loop line 0xf340 arrives 10 cycles before packing starts; every write of the first tile goes through.")}
{fig(14, "+1 nop (110,705 cycles): the loop ends in one more line, 0xf350, read 33 cycles into the first tile; packer 3's write is refused twice.")}
<h3>A resync at tile 0 does not prevent it</h3>
<p>Same pair with <code>TTI_STALLWAIT(STALL_PACK, PACK)</code> before tile 0 only. Card: 107,632 / 110,715. Each column is one tile.</p>
<div class="legend2"><span><span class="sw" style="background:var(--good-soft)"></span>packers 1/2/3 refused ≤ 3 times</span><span><span class="sw" style="background:var(--collc)"></span>more</span><span><span class="sw" style="background:var(--refc)"></span>a packer L1 write waited</span><span>● pack core read a code line</span></div>
{fig(21, "0 nops: tile 0 is the resync; from tile 3 every tile is 35 cycles with 1/1/1 refusals, to the end (2,326 tiles).")}
{fig(22, "+1 nop: in tile 1 the pack core reads a code line and packer 3's write waits; from tile 2 every tile is 36 cycles with 2/2/4, to the end (2,264 tiles).")}
<p>The trigger comes in tile 1, after the resync. Every tile after tile 2 is the same, so the slow rhythm does not come back at block boundaries (this corrects an earlier note).</p>
<h3>Case B: instruction timing (eltwise_binary, one tile per loop pass)</h3>
<div class="legend2"><span><span class="sw" style="background:var(--good)"></span>accepted</span><span><span class="sw" style="background:var(--muted);opacity:.45"></span>instruction blocked (queue full)</span><span><span class="sw" style="background:var(--refc)"></span>DEST read refused</span></div>
{fig(19, "Pad 1 (4,669 cycles): packers 1, 2, 3 refused 139 / 149 / 209 cycles over the run.")}
{fig(20, "Pad 0 (4,741 cycles): the same instructions at slightly different cycles; 253 / 256 / 281. The pack core waits on a full queue in both builds, so it is not the branch predictor (WH-06).")}""",
    card="""<div class="tw"><table><tr><th>Case</th><th class="n">Card</th><th class="n">Versim</th></tr>
<tr><td>matmul 1×3, 0 / +1 nop</td><td class="n">107,622 / 110,705</td><td class="n">107,622 / 110,705</td></tr>
<tr><td>same, resync at tile 0</td><td class="n">107,632 / 110,715</td><td class="n">35 / 36 cycles per tile</td></tr>
<tr><td>eltwise_binary pad 1 / pad 0</td><td class="n">4,669 / 4,741</td><td class="n">4,669 / 4,741</td></tr></table></div>""",
    fix=f"""<p><b>054096d8efa</b> (barrier restart, out-of-line INIT):</p>
<ul>
<li>Each TRISC parks at the start of the measured zone: it stores its restart address and halts with <code>ebreak</code> ({code("tt_metal/tt-llk/tests/helpers/include/barrier.h", 133, "barrier.h park")}). The restart point is on a 512-byte boundary (<code>.balign 512</code>, {code("tt_metal/tt-llk/tests/helpers/include/barrier.h", 148, "barrier.h")}), so the measured loop keeps its address mod 512 when code before it changes.</li>
<li>BRISC serves the barrier ({code("tt_metal/tt-llk/tests/helpers/src/brisc.cpp", 142, "brisc.cpp serve")}): it reads L1 64 times so every L1 bank arbiter last granted BRISC ({code("tt_metal/tt-llk/tests/helpers/src/brisc.cpp", 160, "brisc.cpp")}), invalidates the TRISC instruction caches ({code("tt_metal/tt-llk/tests/helpers/src/brisc.cpp", 165, "brisc.cpp")}), flushes each TRISC through the debug interface ({code("tt_metal/tt-llk/tests/helpers/src/brisc.cpp", 169, "brisc.cpp")}; {isa("TensixTile/BabyRISCV/DebugInterface.md", "ISA doc: debug interface")}), and releases all three together.</li>
<li>INIT and the code after the loop run out of line, each block on 512 bytes ({code("tt_metal/tt-llk/tests/helpers/include/perf.h", 52, "perf.h LLK_INIT_BEGIN")}; {code("tt_metal/tt-llk/tests/helpers/ld/sections.ld", 77, "sections.ld")}), so a change to them cannot move the loop.</li>
</ul>
<p>Result: the same start state and the same layout of the measured loop for every build, whatever changes outside the loop. A change inside the loop still changes its layout, and so its rhythm.</p>""",
    ba="""<div class="tw"><table><tr><th>No-work change (card)</th><th class="n">Before</th><th class="n">After (#58068 head)</th></tr>
<tr><td>4 bytes before every function, all threads (184 test cases, 736 values)</td><td class="n">barrier off: 124 values move</td><td class="n">36 (all from WH-04)</td></tr>
<tr><td>4 bytes before every function, pack thread only (323 test cases, 1,346 values)</td><td class="n">–</td><td class="n">3 (L1_TO_L1 only)</td></tr>
<tr><td>harness code grows by 16 or 384 bytes</td><td class="n">barrier off: 38–43 values</td><td class="n">0</td></tr></table></div>""",
    open="""<ul><li>Case B: we see that the push timing differs, but we did not trace which cycle difference selects which rhythm.</li>
<li>A change inside the measured loop is not covered by any fix; its effect size is set by WH-01.</li></ul>""",
    repro="""<ul><li>Case A: branches <code>nstojictt/exp-b44-z0-sim</code> / <code>-z1-sim</code>, test <code>…matmul_config840-0-1</code>, <code>LLK_PERF_RUN_TYPES=PACK_ISOLATE</code>; first-tile resync: <code>REPRO_PACK_RESYNC=1000000</code>.</li>
<li>At #58068 head: <code>nstojictt/p58-versim</code>, <code>LLK_FN_NOPS=1</code> (all threads) or <code>LLK_FN_NOPS_THREADS=PACK LLK_THREAD_FN_NOPS=1</code>; barrier off: <code>LLK_NO_BARRIER=1</code>.</li>
<li>Waveforms: <code>mm840z0|z1.full.vcd.zst</code>, <code>eb_p0|eb_p1.full.vcd.zst</code>; tile analysis: <code>changelists-2026-10-08/t5z0|t5z1.txt</code> + <code>tilean.py</code>.</li></ul>""",
    refs=f"""<ul class="refs"><li>Code: {code("tt_metal/tt-llk/tests/helpers/include/barrier.h", 148)}, {code("tt_metal/tt-llk/tests/helpers/src/brisc.cpp", 142)}, {code("tt_metal/tt-llk/tests/helpers/include/perf.h", 52)}, {code("tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_pack.h", 435, "_llk_pack_")}.</li>
<li>ISA docs: {isa("TensixTile/BabyRISCV/InstructionCache.md", "Instruction cache")}, {isa("TensixTile/BabyRISCV/DebugInterface.md", "Debug interface")}, {isa("TensixTile/BabyRISCV/PCBufs.md", "PC buffers")}.</li>
<li>Issue #58919 (nop at the top of <code>_llk_pack_init_</code> and <code>_llk_pack_</code>).</li></ul>""",
)
