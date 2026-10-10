import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

HEAD = "24ffcccb24c1d7a3c3be1a531eb460daa5ccd598"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/out"


def c(h):
    return f'<a href="https://github.com/tenstorrent/tt-metal/commit/{h}"><code>{h[:11]}</code></a>'


ROWS = [
    ("52412ea8e67", "Hold the PACK_ISOLATE peers until pack is done", "WH-04", "st-ok", "Works. Keep."),
    ("ed54ddd32f9", "BRISC serve loop in assembly, own 1 KiB section", "WH-07, WH-01", "st-ok", "Works. Keep."),
    ("24ffcccb24c", "INIT measured in its own launch", "INIT values", "st-ok", "Works. Keep."),
    ("22d6769f184", "Layout model fitted to the linked code and the RTL front end", "WH-06, WH-07, WH-08", "st-part", "No failure found. Tested only with 5145805fa38."),
    ("5145805fa38", "INIT inline on unpack and pack", "code shape", "st-part", "No failure found. Tested only with 22d6769f184."),
]


def body():
    rows = "".join(f'<tr><td>{c_}</td><td>{t}</td><td>{m}</td><td><span class="status {cls}">{v}</span></td></tr>' for c_, t, m, cls, v in ROWS)
    return f"""<title>WH-10 New Fixes Check</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>{A['css']}{EXTRA_CSS}</style>
<div class="wrap">
<header><div class="docid">WH-10 · Wormhole LLK perf · <a href="{URLS.get('WH-00')}">WH-00 Index</a></div>
<h1>Check of the new #58068 fixes</h1>
<p class="lede">#58068 got five new perf harness commits on 9 October (head {c(HEAD)}). We tested each one on the bgd-lab-08 card and in Versim. We tried to break each fix with changes that do no work: moved code, a changed BRISC timing, a larger zone helper. None of the five failed. One problem stays open in the run types where all threads work (L1_CONGESTION, L1_TO_L1).</p></header>

<section><h2>Result</h2><div class="tw"><table><tr><th>Commit</th><th>What it does</th><th>Mechanism</th><th>Verdict</th></tr>{rows}</table></div>
<p><b>Test set:</b> 449 cases, 30 from each Wormhole perf module (no SFPU-only modules). 361 run, 88 skip on Wormhole: 1,539 TILE_LOOP values and 1,539 INIT values per arm. <b>Rerun of the same build: 0 of 1,539 values change, in TILE_LOOP and in INIT.</b> So every change below comes from the change we made, not from noise.</p></section>

<section><h2><span class="n">1</span>52412ea8e67: hold the PACK_ISOLATE peers</h2>
<p><b>Claim.</b> In PACK_ISOLATE, unpack and math finish at once and run their exit code during the first tiles of the measured pack loop. Their L1 reads push the packers into the slow rhythm. Now they wait at the end of their TILE_LOOP until pack is done.</p>
<p><b>Test.</b> Move the unpack and math code by 4 bytes (one never-run nop before every function). Do this with the hold and without it (switch <code>LLK_NO_QUIET=1</code> takes the hold out).</p>
<div class="tw"><table><tr><th>Card, 276 PACK_ISOLATE values</th><th class="n">Hold on</th><th class="n">Hold off</th></tr>
<tr><td>Unpack and math code +4 B: values that move &gt; 0.5%</td><td class="n"><b>0</b> (largest 1 cycle)</td><td class="n"><b>54</b>, ±22–28%</td></tr>
<tr><td>Pack code +4 B: values that move &gt; 0.5%</td><td class="n">0</td><td class="n">–</td></tr></table></div>
<div class="tw"><table><tr><th>config12601 PACK_ISOLATE, cycles</th><th class="n">Card</th><th class="n">Versim</th></tr>
<tr><td>Hold on, code as built</td><td class="n">73,850</td><td class="n">73,850</td></tr>
<tr><td>Hold on, unpack and math code +4 B</td><td class="n">73,849</td><td class="n">73,849</td></tr>
<tr><td>Hold off, unpack and math code +4 B</td><td class="n">92,260</td><td class="n">92,260</td></tr>
<tr><td>Hold off, code as built</td><td class="n">92,260</td><td class="n">–</td></tr></table></div>
<p><b>Verdict.</b> The fix works, on the card and in Versim, which agree to the cycle. It replaces our settle fix (<code>LLK_ISO_SETTLE</code>, WH-04) and is better, because it waits for the peers, not for a fixed count.</p>
<p><b>What it changes once.</b> 127 of 276 PACK_ISOLATE values change by more than 0.5% when the hold goes in: 118 get faster (up to 28.6%), 9 slower. These are the values that started in the slow rhythm before.</p>
<p><b>Limit.</b> Only PACK_ISOLATE holds its peers. In UNPACK_ISOLATE and MATH_ISOLATE the idle threads still run their exit code. With all code +4 B, 1 MATH_ISOLATE value moves by more than 0.5% and 0 by more than 2%, so this is not a problem in this set.</p></section>

<section><h2><span class="n">2</span>ed54ddd32f9: BRISC serve loop in assembly</h2>
<p><b>Claim.</b> The BRISC code that releases the TRISCs sets their start skew and their LFSR phase (WH-07). It was compiler output, so a change elsewhere in BRISC could move it. Now it is fixed assembly in its own section.</p>
<div class="tw"><table><tr><th>Card, 1,539 values</th><th class="n">TILE_LOOP &gt; 0.5%</th><th class="n">Largest</th><th class="n">INIT changed</th></tr>
<tr><td>All BRISC code +4 B (one nop before every BRISC function)</td><td class="n"><b>0</b> (all 1,539 identical)</td><td class="n">0</td><td class="n">0</td></tr>
<tr><td>One nop between the release stores (<code>LLK_RELEASE_GAP=1</code>)</td><td class="n"><b>64</b></td><td class="n">12.3%</td><td class="n">0</td></tr></table></div>
<p>The 64: L1_CONGESTION[PACK] 32, L1_CONGESTION[UNPACK] 26, MATH_ISOLATE 4, L1_TO_L1 2.</p>
<p><b>Verdict.</b> Both halves of the claim are true. The release timing matters (one cycle moves 64 values), and the pin keeps all other BRISC changes out (0 values move). <b>Limit:</b> the pin holds the timing fixed; it does not remove the sensitivity. Section 6 has the same sensitivity from TRISC-side changes.</p></section>

<section><h2><span class="n">3</span>24ffcccb24c: INIT in its own launch</h2>
<p><b>Claim.</b> A nop anywhere outside INIT leaves every INIT value the same. BRISC releases the three INITs 1,500 spins apart, so each INIT runs alone.</p>
<div class="tw"><table><tr><th>Card, INIT values</th><th class="n">Changed</th><th class="n">Largest</th></tr>
<tr><th colspan="3">New method (INIT in its own launch)</th></tr>
<tr><td>Rerun</td><td class="n">0 / 1,539</td><td class="n">0</td></tr>
<tr><td>BRISC code +4 B</td><td class="n">0 / 1,539</td><td class="n">0</td></tr>
<tr><td>Release gap +1 nop</td><td class="n">0 / 1,539</td><td class="n">0</td></tr>
<tr><td><code>zone_reserve</code> +1 nop</td><td class="n">0 / 1,539</td><td class="n">0</td></tr>
<tr><td>Unpack and math code +4 B (PACK_ISOLATE)</td><td class="n">0 / 276</td><td class="n">0</td></tr>
<tr><td>Pack code +4 B (PACK_ISOLATE), INIT code moves too</td><td class="n">251 / 276</td><td class="n">15 cycles (2.6%)</td></tr>
<tr><td>All code +4 B, INIT code moves too</td><td class="n">1,336 / 1,539</td><td class="n">24 cycles (4.2%)</td></tr>
<tr><th colspan="3">Old method (<code>LLK_PERF_INIT_LAUNCH=0</code>), same changes</th></tr>
<tr><td><code>zone_reserve</code> +1 nop</td><td class="n">459 / 1,539</td><td class="n">7 cycles (1.5%)</td></tr>
<tr><td>Unpack and math code +4 B (PACK_ISOLATE)</td><td class="n">0 / 276</td><td class="n">0</td></tr>
<tr><td>All code +4 B, INIT code moves too</td><td class="n">1,365 / 1,539</td><td class="n">26 cycles (5.0%)</td></tr></table></div>
<p>The longest INIT in the set is 713 cycles. 1,500 spins on BRISC is several thousand cycles, so the INITs do not overlap.</p>
<p><b>Verdict.</b> The claim holds: no change outside INIT moved an INIT value. With the old method, the <code>zone_reserve</code> change moved 459 INIT values; with the new method, 0. When the INIT code itself moves, INIT moves by up to 24 cycles. That is WH-06 to WH-08 acting on the INIT code, and it is expected. <b>Note:</b> L1_TO_L1 INIT is now the longest of three INITs that run alone, not three INITs that run together. That is a change of meaning, not an error.</p>
<p><b>The INIT numbers change once.</b> New against old method, same build: 1,346 of 1,539 INIT values differ by more than 2%. The median is 3.4% lower. The largest change: sfpu_binop_scalar L1_TO_L1 INIT was 15,301 cycles with the old method and is now 308–315. We did not check why the old value was this large. Tell the people who read the INIT numbers. TILE_LOOP does not change: 0 of 1,539 values differ between the two methods.</p></section>

<section><h2><span class="n">4</span>22d6769f184 and 5145805fa38: layout model and inline INIT</h2>
<p>We have no switch that takes out only one of the two, so we tested them together, with code moves:</p>
<div class="tw"><table><tr><th>Card, TILE_LOOP, values that move &gt; 2%</th><th class="n">Old head 3ff45cbd1c7 (323 cases, 1,346 values)</th><th class="n">New head (449 cases, 1,539 values)</th></tr>
<tr><td>All code +4 B</td><td class="n">75 (50 in isolate run types), up to 28.6%</td><td class="n">14 (0 in isolate run types), up to 12.3%</td></tr>
<tr><td><code>zone_reserve</code> +1 nop</td><td class="n">35 (5 PACK_ISOLATE), up to 17.8%</td><td class="n">14 (0 in isolate run types), up to 16.9%</td></tr></table></div>
<div class="tw"><table><tr><th>config936 MATH_ISOLATE (the WH-08 case), cycles</th><th class="n">Card</th><th class="n">Versim</th></tr>
<tr><td>Code as built</td><td class="n">75,935</td><td class="n">75,935</td></tr>
<tr><td>Math code +4 B</td><td class="n">75,935</td><td class="n">75,935</td></tr></table></div>
<p><b>Verdict.</b> No failure found. Isolate run types do not move with code moves on the new head. We did not check his claim that 8,798 values were more than 2% slower than the old harness before 5145805fa38. That needs the old harness as a reference, and it is a question of accuracy, not stability.</p></section>

<section><h2><span class="n">5</span>WH-05 at the new head</h2>
<p><code>zone_reserve</code> +1 nop: 0 isolate values move by more than 2% (old head: 5). The 14 that still move are L1_CONGESTION and L1_TO_L1 (section 6). We do not need our early-reserve fix for the isolate run types any more.</p></section>

<section><h2><span class="n">6</span>Still open: run types where all threads work</h2>
<p>In L1_CONGESTION and L1_TO_L1, any change of a few cycles at the start can select a different packer rhythm (WH-01). Three different changes, each with no work:</p>
<div class="tw"><table><tr><th>Card, 772 values (L1_CONGESTION 476, L1_TO_L1 296)</th><th class="n">&gt; 0.5%</th><th class="n">&gt; 2%</th><th class="n">Largest</th></tr>
<tr><td>All code +4 B</td><td class="n">38</td><td class="n">14</td><td class="n">12.3%</td></tr>
<tr><td>Release gap +1 nop</td><td class="n">60</td><td class="n">16</td><td class="n">12.3%</td></tr>
<tr><td><code>zone_reserve</code> +1 nop</td><td class="n">48</td><td class="n">14</td><td class="n">16.9%</td></tr>
<tr><td>Any of the three</td><td class="n">95</td><td class="n">–</td><td class="n">–</td></tr></table></div>
<p>The values jump between fixed states. Example: perf_reduce L1_CONGESTION[PACK] (Bfp8_b → Float16) goes from 83,095 to exactly 72,856 with the code move and with the release gap. None of the five commits addresses this. In these run types the threads really run, so the cure the isolate run types got (keep the idle threads quiet) does not apply.</p>
<p><b>Options:</b> (a) accept and document it for L1_CONGESTION, which measures contention by design; (b) for L1_TO_L1, run each case at two or three start offsets and report each state; (c) a periodic packer resync. Section 7 tests (c).</p></section>

<section><h2><span class="n">7</span>Does a packer resync fix it? (10 October)</h2>
<p><b>Test.</b> Switch <code>REPRO_PACK_RESYNC=N</code> (commit 15d74d27621 on <code>nstojictt/p58v2-versim</code>): every N pack calls, the pack thread waits until all four packers are idle (<code>STALLWAIT STALL_PACK</code>), so they start the next call together. It is in <code>_llk_pack_</code>, <code>_llk_pack_rows_</code>, <code>_llk_pack_untilize_</code> and <code>_llk_pack_fast_tilize_block_</code>. Card bgd-lab-17, the same 449 cases, L1_TO_L1 and L1_CONGESTION only (772 values). The rerun is exact (0 of 772 change), and the values are the same as on bgd-lab-08, to the cycle.</p>
<div class="tw"><table><tr><th>Values that move &gt; 2% (of 772)</th><th class="n">No resync</th><th class="n">Resync every 32 calls</th><th class="n">Resync every call</th></tr>
<tr><td>All code +4 B</td><td class="n">14 (max 12.3%)</td><td class="n">12 (max 5.5%)</td><td class="n">3 (max 4.4%)</td></tr>
<tr><td>Release gap +1 nop</td><td class="n">16 (max 12.3%)</td><td class="n">9 (max 7.0%)</td><td class="n">15 (max 15.7%)</td></tr>
<tr><td><code>zone_reserve</code> +1 nop</td><td class="n">14 (max 16.9%)</td><td class="n">3 (max 6.9%)</td><td class="n">13 (max 10.0%)</td></tr>
<tr><td>Any of the three</td><td class="n">29</td><td class="n">19</td><td class="n">28</td></tr>
<tr><td>… of them L1_TO_L1</td><td class="n">7</td><td class="n">6</td><td class="n"><b>0</b></td></tr></table></div>
<div class="tw"><table><tr><th>Cost: change against no resync</th><th class="n">Every 32 calls</th><th class="n">Every call</th></tr>
<tr><td>L1_TO_L1 (296): median / largest / values &gt; 2%</td><td class="n">0.0% / +11.1% / 25</td><td class="n">+0.8% / +41.6% / 126</td></tr>
<tr><td>L1_CONGESTION[PACK] (238)</td><td class="n">+1.3% / +23.4% / 127</td><td class="n">+26.3% / +115.6% / 237</td></tr>
<tr><td>L1_CONGESTION[UNPACK] (238)</td><td class="n">0.0% / +8.6% / 2</td><td class="n">0.0% / +5.1% / 32</td></tr></table></div>
<p><b>Answer: no, a resync does not get everything under 2%.</b> Every 32 calls removes a third of the moves. Every call removes all L1_TO_L1 moves, but it makes 126 L1_TO_L1 values more than 2% slower (up to 41.6%), and L1_CONGESTION[PACK] gets 26% slower in the median. L1_CONGESTION still moves in all three settings.</p>
<h3>Why: two different mechanisms, seen in Versim</h3>
<p>Versim on two of the largest movers, the same ELF as the card. Every Versim value below is the same as the card value, to the cycle.</p>
<div class="tw"><table><tr><th>L1_CONGESTION[PACK], cycles</th><th class="n">Base</th><th class="n">Release gap +1</th><th class="n">Resync every 32</th></tr>
<tr><td>reduce (Bfp8_b → Float16, ReduceScalar Sum)</td><td class="n">83,095</td><td class="n">72,856</td><td class="n">69,821</td></tr>
<tr><td>eltwise_binary (Float16 → Float16_b, Elwmul LoFi)</td><td class="n">17,320</td><td class="n">18,672</td><td class="n">18,125</td></tr></table></div>
<div class="tw"><table><tr><th>Per tile, steady state (Versim)</th><th class="n">Tile lengths</th><th class="n">Packer 1–3 DEST refusals</th><th class="n">Packer 1 waits for L1</th></tr>
<tr><td>reduce, base</td><td class="n">32 / 49 alternating</td><td class="n">2.0 / 4.0 / 7.0</td><td class="n">0.8</td></tr>
<tr><td>reduce, release gap +1</td><td class="n">32 / 35, some 49</td><td class="n">1.3 / 1.8 / 2.7</td><td class="n">1.1</td></tr>
<tr><td>reduce, resync every 32</td><td class="n">32 / 35</td><td class="n">1.1 / 1.2 / 1.3</td><td class="n">1.0</td></tr>
<tr><td>eltwise_binary, base</td><td class="n">72 / 62 alternating</td><td class="n">5.5 / 22.3 / 29.4</td><td class="n">35.1</td></tr>
<tr><td>eltwise_binary, release gap +1</td><td class="n">67 / 79 alternating</td><td class="n">5.0 / 15.2 / 22.5</td><td class="n">40.5</td></tr>
<tr><td>eltwise_binary, resync every 32</td><td class="n">62–81, no fixed pattern</td><td class="n">4.8 / 15.7 / 25.7</td><td class="n">37.6</td></tr></table></div>
<ol>
<li><b>reduce: the packer rhythm (WH-01).</b> The slow state has a slow tile every second tile, with DEST refusals. The resync removes the refusals and the slow tile. Card: with the resync every 32 calls, the release gap moves this value by 0.8% (it was 12.3%).</li>
<li><b>eltwise_binary: packer 1 against the L1 port.</b> Packer 1 waits 35–40 cycles per tile for L1. Its port is shared with unpacker 1 and the scrubber (WH-02), and in L1_CONGESTION the unpacker keeps reading L1. The start skew sets the phase between the unpacker's reads and packer 1's writes, and that phase sets the tile length. The tile length follows packer 1's L1 wait: in the resync run, 62-cycle tiles wait 31 cycles and 81-cycle tiles wait 46. A packer resync does not move the unpacker, so it cannot fix this phase.</li></ol>
<p><b>So:</b> a resync can fix the WH-01 cases at a cost, but not the L1-port cases. For L1_CONGESTION, option (a) (document that it has two or more states) is the realistic choice. For L1_TO_L1, (b) (measure at more than one start offset) costs less than a resync on every call.</p></section>

<section><h2>How to reproduce</h2><ul>
<li>Branch <code>nstojictt/p58v2-versim</code> = #58068 head + Versim harness + switches: <code>LLK_NO_QUIET</code>, <code>LLK_RELEASE_GAP=N</code>, <code>LLK_SERVE_PRE=N</code>, <code>LLK_BRISC_FN_NOPS=N</code>, <code>LLK_FN_NOPS=N</code>, <code>LLK_FN_NOPS_THREADS</code> + <code>LLK_THREAD_FN_NOPS</code>, <code>LLK_ZONE_RESERVE_NOPS=N</code>, <code>LLK_NO_PADS</code>, <code>LLK_FORCE_PADS</code>, <code>LLK_ISO_SETTLE</code>, <code>LLK_TRISC_BP_OFF</code>.</li>
<li>Card results (CSV and logs): <code>/proj_sw/user_dev/nstojic/v2c/</code>. Versim (logs, CSV, signals, VCD): <code>/proj_sw/user_dev/nstojic/versim-runs/q0 q1 q2 m0 m1</code> (bgd). Section 7: card <code>/proj_sw/user_dev/nstojic/v2r/</code> (bgd); Versim <code>/proj_sw/user_dev/nstojic/versim-runs/ye_* yr_*</code> (yyz).</li>
<li>Tests: <code>perf_math_matmul.py::test_perf_math_matmul[MathFidelity.HiFi2-matmul_config12601-5-1]</code> (PACK_ISOLATE), <code>[MathFidelity.LoFi-matmul_config936-0-1]</code> (MATH_ISOLATE).</li></ul></section>
<p class="small">Status 10 October 2026 (section 7: 10 October afternoon, card bgd-lab-17, Versim on yyzeon09). #58068 head {HEAD[:11]}. Card bgd-lab-08 (Wormhole n150), Versim versim-wormhole-b0.</p>
</div>
"""


if __name__ == "__main__":
    noinit = open(sys.argv[1]).read() if len(sys.argv) > 1 else ""
    open(f"{OUT}/WH-10.html", "w").write(body().replace("{NOINIT}", noinit))
    print("ok")
