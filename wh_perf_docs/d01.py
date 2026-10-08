from common import *

D = dict(
    id="WH-01", short="Packer Rhythm",
    summary="The four Wormhole packers read DEST through one crossbar. They can run in a clean rhythm, or in a rhythm with a bank conflict in every tile. Both rhythms repeat exactly, so a pack loop keeps the one it starts in. The slow rhythm costs 1 to 10 cycles per tile. This is the base mechanism behind WH-02 to WH-05.",
    status="Not fixable in software", status_cls="st-part",
    depends=[], used_by=["WH-02", "WH-03", "WH-04", "WH-05"],
    problem="both (re-measure and no-work change)",
    what="""<ul>
<li>The same pack loop gives one of two (sometimes three) values, for example 107,622 or 110,705 cycles for matmul 1×3 PACK_ISOLATE: about 1 cycle per tile.</li>
<li>Which value you get depends on small timing events at the start of the loop or while it runs, not on the work. See WH-02 to WH-05 for those events.</li>
<li>Blackhole does not show it: it has one packer (<code>PACK = PACK0</code> in {bh}), so there is no crossbar between packers.</li>
</ul>""".format(bh=code("tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_instr_params.h", 277, "blackhole ckernel_instr_params.h")),
    hw=f"""<ul>
<li><b>Four packers.</b> Wormhole has <code>PACK0..PACK3</code> and <code>PACK = PACK0 | PACK1 | PACK2 | PACK3</code> ({code("tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/ckernel_instr_params.h", 226)}). They are usually started together; each moves a part of DEST to L1 ({isa("TensixTile/TensixCoprocessor/Packers/README.md", "ISA doc: Packers")}).</li>
<li><b>One DEST read crossbar with 4 banks.</b> {rtl("instrn_path/rtl/tt_instruction_issue.sv", 3281)} <code>localparam DEST_PACK_BANKS = 4</code>; the crossbar is <code>tt_if_xbar</code> ({rtl("instrn_path/rtl/tt_instruction_issue.sv", 3549)}).</li>
<li><b>Bank = row address bits 1:0.</b> {rtl("common/rtl/tt_if_xbar.sv", 69)}: <code>bank_rd_en[b*INTF_CNT + i] = i_rden[i] &amp; (i_addr[...BANK_CNT_LOG2] == b)</code>.</li>
<li><b>Fixed priority per bank.</b> {rtl("common/rtl/tt_if_xbar.sv", 126)}: <code>.FIXED_ARB (1)</code>; {rtl("common/rtl/tt_rr_if_arb_w_retpath.sv", 36)}: "if set to 1, arbitration becomes fixed priority, 0-1-2,etc., instead of round robin". Packer 0 always wins a tie.</li>
<li><b>Which L1 port each packer writes through</b> (WH RTL): packer 0 has its own port ({rtl("tensix/rtl/tt_tensix.sv", 1631)}); packer 1 shares port 1 with the scrubber and unpacker 1 ({rtl("tensix/rtl/tt_tensix.sv", 1211)}); packers 2 and 3 go through the two TDMA round-robin arbiters ({rtl("tdma/rtl/tt_tdma.sv", 3848)}, {rtl("tdma/rtl/tt_tdma.sv", 3855)}), whose outputs share <b>port 2</b> with NCRISC, TRISC0 (unpack) and BRISC ({rtl("tensix/rtl/tt_tensix.sv", 1304)}) and <b>port 3</b> with TRISC1 (math) and TRISC2 (pack) ({rtl("tensix/rtl/tt_tensix.sv", 1398)}). So every L1 access of a TRISC, including its code fetches, competes with packer 2 or packer 3 for a port. The public port diagram is in {isa("TensixTile/L1.md", "ISA doc: L1")}.</li>
<li><b>Shared L1 write path.</b> The packers write L1 through shared access ports ({isa("TensixTile/L1.md", "ISA doc: L1, 16 banks, access ports, round-robin muxes")}). A write can be refused for a cycle when another client uses the same port or bank.</li>
</ul>
{fig(5, "The crossbar: four packers, four banks, fixed priority.")}""",
    how="""<ol class="chain">
<li>Each packer reads DEST rows in order, two reads per row, so it asks for banks 0, 0, 1, 1, 2, 2, 3, 3, 0, …</li>
<li>The four packers run 2 cycles apart. In the clean rhythm they always ask for four different banks, and nobody waits except at the tile start.</li>
<li>If packer 0 loses one cycle (its L1 write is refused once), it falls one cycle behind and lands on packer 1's bank. Packer 0 wins (fixed priority), packer 1 is refused and falls behind, lands on packer 2's bank, and so on to packer 3.</li>
<li>At each tile end a packer pauses its DEST reads and, a few cycles later, its L1 writes. In the clean rhythm these pauses form a staircase 2 cycles apart. In the slow rhythm the refusals split packer 3's pause, the rest of it lands mid-tile, packer 3's restart meets packer 0's write, and step 3 happens again.</li>
<li>Both patterns repeat exactly every tile. The loop stays in its rhythm until something shifts one packer by a cycle (WH-02 to WH-05).</li>
</ol>""",
    sketch="four packer lanes over one tile (35 cycles) with the bank numbers 0,0,1,1,2,2,3,3 shifted by 2 cycles per lane; then the same with packer 0 one cycle late at cycle ~31, and follow the × down the diagonal.",
    versim=f"""<p>Signals: <code>instruction_issue.dma_rdrow_xbar_rden</code>, <code>…_ready</code>, <code>…_rd_addr</code> (bank = bits 1:0), and the packers' <code>o_l1_wren</code> / <code>i_l1_req_ready</code>. matmul 1×3 PACK_ISOLATE, quiet harness, pack predictor off, 76 cycles from tile 1,500. Colours are banks 0–3, × is a refused read, the grey row under each packer is its L1 write.</p>
{fig(6, "Fast (0 nops, 107,622 cycles): one refusal per packer at the tile start, then a clean diagonal.")}
{fig(7, "Slow (+1 nop, 110,705 cycles): packer 0's L1 write is refused once mid-tile; a diagonal of × follows through packers 1, 2, 3.")}
<h3>Why it repeats every tile</h3>
{fig(8, "Fast: the four 3-cycle L1 write pauses form a staircase, 2 cycles apart (offsets 10–12, 12–14, 14–16, 16–18).")}
{fig(9, "Slow: packer 3's pause is split; its leftover lands at offsets 24–25 and its restart meets packer 0's write at offset 31.")}
<h3>A second config, 80 cycles mid-loop</h3>
<div class="legend2"><span><span class="sw" style="background:var(--good)"></span>DEST read accepted</span><span><span class="sw" style="background:var(--refc)"></span>refused</span></div>
{fig(17, "math_matmul config12601 at PR head, fast: 35 cycles per tile.")}
{fig(18, "Same code, slow: 45 cycles per tile, packer 3 refused 10 cycles per tile, all four stop for 12.")}""",
    card="""<p><b>Removing the rhythm removes every effect.</b> Test switch <code>REPRO_PACK_RESYNC=N</code> on the experiment branches: every N tiles the pack thread runs <code>TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::PACK)</code> before the tile, so all four packers start the tile together.</p>
<div class="tw"><table><tr><th>Resync every</th><th class="n">matmul: configs moved by +1 nop (of 41)</th><th class="n">K probe, K = 0 / 150 / 300</th><th class="n">eltwise pads 0–3</th></tr>
<tr><td>never</td><td class="n">24</td><td class="n">9,063 / 9,341 / 9,308</td><td class="n">4,741 / 4,669 / 4,672 / 4,743</td></tr>
<tr><td>1 tile</td><td class="n">0</td><td class="n">14,123 / 14,130 / 14,130</td><td class="n">7,104 / 7,106 / 7,114 / 7,107</td></tr>
<tr><td>32 tiles</td><td class="n">0</td><td class="n">9,627 / 9,627 / 9,627</td><td class="n">4,953 / 4,953 / 4,958 / 4,955</td></tr></table></div>
<p>Cost: +1.8% (matmul) to +6% at N = 32, up to +57% at N = 1. So it proves the cause; it is not a fix.</p>""",
    fix=f"""<p>No software change removes the rhythm itself. #58068 removes or fixes the events that select it (WH-02 to WH-05). Two direct attempts were dropped:</p>
<ul><li>73e829efa99 drained the packers at each section in isolate pack loops; c16630eb8a0 removed it again (up to 60% slower).</li>
<li>A per-block resync in <code>_llk_packer_wait_for_math_done_</code> gave no gain and +8% CI time.</li></ul>
<p>For L1_TO_L1 (the merge gate), where all threads run together, averaging over a few start offsets (#59199) keeps the result inside the gate threshold. The real fix is in hardware: round-robin arbitration in the DEST read crossbar (<code>FIXED_ARB = 0</code>) should stop a lag from repeating.</p>""",
    ba="""<div class="tw"><table><tr><th>Case</th><th class="n">Before</th><th class="n">After</th></tr>
<tr><td>matmul 1×3, 0 / +1 nop, card and Versim</td><td class="n">107,622 / 110,705</td><td class="n">resync every tile: 168,993 / 168,991 (Versim 168,995 both)</td></tr>
<tr><td>Versim, banks of packers 0..3 relative to packer 0, healthy pattern (0,3,2,1)</td><td class="n">79,872 / 79,864 cycles</td><td class="n">plus (0,0,0,0) at every tile start: 6,144 cycles</td></tr></table></div>""",
    open="""<ul><li>Why the slow rhythm in some configs costs 10 cycles per tile (config12601) and in others 1 (matmul 1×3): the size depends on how much the packers stall on L1 after the refusals. Not measured per config.</li>
<li>Packer 0 has its own L1 port, so its one-cycle refusal comes from a bank conflict, not a port conflict; which client takes the bank in that cycle is read from the waveform, not traced.</li></ul>""",
    repro="""<ul><li>Branches <code>nstojictt/exp-b44-z0-sim</code> / <code>-z1-sim</code> (0 / +1 nop), test <code>perf_math_matmul.py::test_perf_math_matmul[MathFidelity.LoFi-matmul_config840-0-1]</code>, <code>LLK_PERF_RUN_TYPES=PACK_ISOLATE</code>, <code>REPRO_PACK_RESYNC=N</code>.</li>
<li>Versim: <code>repro_phase_tools/vrun.sh</code>, then <code>ext.py</code>, <code>to_cycles.py</code>, <code>tilean.py</code> (branch <code>nstojictt/p58-versim</code>).</li>
<li>Waveforms: <code>/proj_sw/user_dev/nstojic/versim-waveforms/mm840z0|z1|rsz0|rsz1.full.vcd.zst</code>, <code>pf0|pf1.full.vcd.zst</code>.</li></ul>""",
    refs=f"""<ul class="refs"><li>RTL: {rtl("instrn_path/rtl/tt_instruction_issue.sv", 3281)}, {rtl("instrn_path/rtl/tt_instruction_issue.sv", 3549)}, {rtl("common/rtl/tt_if_xbar.sv", 69)}, {rtl("common/rtl/tt_if_xbar.sv", 126)}, {rtl("common/rtl/tt_rr_if_arb_w_retpath.sv", 36)}.</li>
<li>Code: {code("tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/ckernel_instr_params.h", 222)}, {code("tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_pack.h", 435, "_llk_pack_")}.</li>
<li>ISA docs: {isa("TensixTile/TensixCoprocessor/Packers/README.md", "Packers")}, {isa("TensixTile/TensixCoprocessor/Dst.md", "Dst")}, {isa("TensixTile/L1.md", "L1")}.</li>
<li>Issues: #58919 (packer counters 32/32/32/32 against 32/34/36/41), #55169.</li></ul>""",
)
