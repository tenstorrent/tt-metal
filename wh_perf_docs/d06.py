from common import *

D = dict(
    id="WH-06", short="Pack Branch Predictor",
    summary="The TRISC branch predictor has 16 entries and no tag. If two branches of the pack loop map to the same entry, both mispredict on every pass, the pack core becomes slower than the packers, and the kernel loses about 1 cycle per tile. Which branches share an entry depends only on their addresses.",
    status="Fixed for code outside the loop", status_cls="st-ok",
    depends=[], used_by=[],
    problem="no-work change",
    what="""<ul>
<li>Repro (PACK_ISOLATE, other threads idle): 9,070 cycles with 0 nops in <code>_llk_pack_init_</code>, 9,317 with +1 nop (card), 9,313 (Versim).</li>
<li>perf_pack pad 0 / pad 9: 3,286 / 3,664 cycles (card = Versim).</li>
<li>Full suite before the barrier: one nop moved 14,750 PACK_ISOLATE points; with the pack predictor off, 3,415.</li>
</ul>""",
    hw=f"""<ul>
<li>16 entries: {rtl("briscv/rtl/tt_bp.sv", 61)} <code>localparam BP_DEPTH = 16</code>.</li>
<li>No tag; the index is a hash of address bits 2–8: {rtl("briscv/rtl/tt_bp.sv", 91)} <code>addr_hash = {{~(a[8]^a[6]), a[7]^a[5], ~(a[4]^a[3]), a[2]}}</code>.</li>
<li>An entry stores a 2-bit counter and the resolved next address, written by any branch with that hash: {rtl("briscv/rtl/tt_bp.sv", 120)}. A wrong stored address resets the counter: {rtl("briscv/rtl/tt_bp.sv", 172)}.</li>
<li>A not-taken branch is also a mispredict when the stored next address is not <code>pc + 4</code>: {rtl("briscv/rtl/tt_ex.sv", 293)} <code>mis_bp_addr_mismatch</code>.</li>
<li>A mispredict costs a 2-cycle bubble ({isa("TensixTile/BabyRISCV/README.md", "ISA doc: Baby RISCV, Integer Unit")}).</li>
<li>Only a reset clears the table (<code>bp_row &lt;= 'd0</code> under <code>!i_reset_n</code>). The predictor can be turned off per TRISC: <code>DISABLE_RISC_BP_Disable_trisc</code> ({code("tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines/cfg_defines.h", 1731, "cfg_defines.h")}).</li>
</ul>""",
    how="""<p><b>The code.</b> The repro is the pack side of <code>math_matmul_test.cpp</code> in PACK_ISOLATE: three nested loops around one <code>_llk_pack_</code> call.</p>
<pre>for (loop = 0; loop &lt; LOOP_FACTOR; loop++)              // outermost loop
    for (block = 0; block &lt; NUM_BLOCKS; ++block)         // "outer" branch below
        for (tile = 0; tile &lt; CT_DIM * RT_DIM; tile++)   // "inner" branch below
            _llk_pack_&lt;...&gt;(DST_INDEX + tile, PERF_ADDRESS(PERF_OUTPUT, tile));</pre>
<p>GCC turns the end of the two inner loops into two <code>bne</code> instructions with one instruction between them, so they are 8 bytes apart. The pack ELF with 0 nops (<code>pack.elf</code> of the repro build, <code>objdump -d</code>):</p>
<pre>    f1ac:  andi   a5,a4,15            &lt;- tile loop starts here
    ...                               (address math and 3 stores for the pack setup)
    f1d0:  ttreg2flop 1,0,0,0,5,12
    f1d4:  ttpacr     2,0,15,0,0,1,0
    f1d8:  sw     a0,0(s9)
    f1dc:  ttmop  1,0,0               &lt;- start the pack MOP for this tile
    f1e0:  addi   a4,a4,1             tile++
    <b>f1e4:  bne    s8,a4,f1ac</b>          <b>inner branch</b>: next tile
    f1e8:  addi   s0,s0,1             block++
    <b>f1ec:  bne    s2,s0,f194</b>          <b>outer branch</b>: next block
    f1f0:  lw     a5,8(sp)            (the outermost loop: loop++)
    ...
    f200:  bne    a4,a5,f188</pre>
<p>With one more nop before the loop, the same code starts 8 bytes later: the inner branch is at <code>0xf1ec</code> and the outer branch at <code>0xf1f4</code>.</p>
<p><b>The math.</b> The predictor has 16 entries and no tag. It picks the entry from 7 address bits, <code>a[8:2]</code>, with this hash (""" + rtl("briscv/rtl/tt_bp.sv", 91) + """):</p>
<pre>entry = { ~(a[8] ^ a[6]),  a[7] ^ a[5],  ~(a[4] ^ a[3]),  a[2] }    // 4 bits: bit 3, 2, 1, 0</pre>
<p>Only the low 9 bits of the address matter (bits 2 to 8). Work it out for the three addresses:</p>
<div class="tw"><table><tr><th>Branch address</th><th>Low 9 bits (a8 … a0)</th><th>bit 3 = ~(a8^a6)</th><th>bit 2 = a7^a5</th><th>bit 1 = ~(a4^a3)</th><th>bit 0 = a2</th><th>Entry</th></tr>
<tr><td class="m">0xf1e4</td><td class="m">1 1110 0100</td><td class="m">~(1^1) = 1</td><td class="m">1^1 = 0</td><td class="m">~(0^0) = 1</td><td class="m">1</td><td class="m">1011 = <b>11</b></td></tr>
<tr><td class="m">0xf1ec</td><td class="m">1 1110 1100</td><td class="m">~(1^1) = 1</td><td class="m">1^1 = 0</td><td class="m">~(0^1) = 0</td><td class="m">1</td><td class="m">1001 = <b>9</b></td></tr>
<tr><td class="m">0xf1f4</td><td class="m">1 1111 0100</td><td class="m">~(1^1) = 1</td><td class="m">1^1 = 0</td><td class="m">~(1^0) = 0</td><td class="m">1</td><td class="m">1001 = <b>9</b></td></tr></table></div>
<ul><li><b>0 nops:</b> inner branch 0xf1e4 → entry 11, outer branch 0xf1ec → entry 9. Each branch has its own entry.</li>
<li><b>+1 nop:</b> inner branch 0xf1ec → entry 9, outer branch 0xf1f4 → entry 9. The two addresses differ only in bits 3 and 4 (0x…ec = …0 1100, 0x…f4 = …1 0100). The hash folds those two bits into one bit, <code>~(a4 ^ a3)</code>, and both flips together give the same result, so both branches land on entry 9.</li></ul>
<p><b>What follows:</b></p>
<ol class="chain">
<li>Each branch overwrites the next address the other one stored. Both mispredict on every pass (2 per pass, 3 cycles each).</li>
<li>The pass takes 36 cycles instead of 35, the pack core can no longer keep the packers' queue full, and the loop is 1 cycle per tile slower.</li>
</ol>""",
    sketch="a 16-row table (the entries) and two arrows from the two branch addresses; for 0 nops they go to rows 11 and 9, for +1 nop both to row 9.",
    versim=f"""<div class="tw"><table><tr><th>Build</th><th>Inner branch</th><th>Outer branch</th><th class="n">Entry writes in 256 passes</th></tr>
<tr><td>0 nops</td><td class="m">0xf1e4 → entry 11</td><td class="m">0xf1ec → entry 9</td><td class="n">1 each</td></tr>
<tr><td>+1 nop</td><td class="m">0xf1ec → entry 9</td><td class="m">0xf1f4 → entry 9</td><td class="n">256 each</td></tr></table></div>
<div class="legend2"><span><span class="sw" style="background:var(--refc)"></span>mispredict</span><span><span class="sw" style="background:var(--bk2)"></span>fetch queue empty</span><span><span class="sw" style="background:var(--good)"></span>instruction to the packers accepted</span><span><span class="sw" style="background:var(--muted);opacity:.45"></span>blocked (queue full)</span><span>dashed = loop pass start</span></div>
{fig(0, "0 nops: 35 cycles per pass, no mispredicts, the core waits on a full queue (it is ahead of the packers).")}
{fig(1, "+1 nop: 36 cycles per pass, two mispredicts per pass, the fetch queue runs empty, the core never fills the queue.")}
<p>Over 240 passes: 0 mispredicts and 5,971 blocked cycles (fast) against 480 mispredicts and 0 blocked cycles (slow). Signals: <code>briscv.ex_bp_mispredict</code>, <code>ifetch.instrn_fifo_empty</code>, <code>o_trisc_instrn_buf_wren</code> / <code>i_trisc_instrn_buf_req_ready</code>, and <code>ifetch.bp.bp_array(0..15)</code> for the entries.</p>""",
    card="""<ul><li>From the branch addresses of 53 builds, the hash predicted the fast or slow result for all 53.</li>
<li>Pack predictor off (<code>DISABLE_RISC_BP_Disable_trisc</code> bit 2): the slow build goes from 9,317 to 9,063. Bits 0 and 1 (unpack, math) have no effect. The bit stays set after a run: write it on every boot.</li></ul>""",
    fix=f"""<p><b>054096d8efa</b> restarts the measured loop from a 512-byte boundary ({code("tt_metal/tt-llk/tests/helpers/include/barrier.h", 148, "barrier.h .balign 512")}). The hash uses bits 2–8, which are the address mod 512, so a change outside the loop cannot change which branches share an entry. The layout pads of 3ff45cbd1c7 are not applied to pack threads ({code("tt_metal/tt-llk/tests/python_tests/helpers/test_config.py", 1862, "test_config.py LAYOUT_THREADS")}).</p>
<p>A change inside the loop can still create a shared entry. Turning the pack predictor off in perf builds would remove it completely, but it shifts many values once and moves the numbers away from production.</p>""",
    ba="""<div class="tw"><table><tr><th>No-work change (card)</th><th class="n">Before</th><th class="n">After</th></tr>
<tr><td>Repro, +1 nop</td><td class="n">9,070 → 9,317</td><td class="n">pack predictor off: 9,063</td></tr>
<tr><td>Full suite, +1 nop, PACK_ISOLATE points moved</td><td class="n">14,750</td><td class="n">pack predictor off: 3,415</td></tr>
<tr><td>#58068 head, pack code +4 B per function</td><td class="n">–</td><td class="n">0 isolate values move</td></tr></table></div>""",
    open="""<ul><li>The table survives the barrier (only a reset clears it), so "every zone starts from the same state" is not exact. We found no value that changes because of it.</li></ul>""",
    repro="""<ul><li>Branch <code>nstojictt/versim-m1-repro</code>: <code>repro_phase.sh hw|sim NOPS=0..4 TAIL=6000</code>; predictor off: <code>REPRO_TRISC_BP_OFF=4</code> (or <code>LLK_TRISC_BP_OFF=4</code> on <code>nstojictt/p58-versim</code>).</li>
<li>Waveforms: <code>n0|n1.full.vcd.zst</code>, <code>bp_fa…bp_n4.full.vcd.zst</code>, <code>pk_p0|pk_p9.full.vcd.zst</code>.</li></ul>""",
    refs=f"""<ul class="refs"><li>RTL: {rtl("briscv/rtl/tt_bp.sv", 91)}, {rtl("briscv/rtl/tt_bp.sv", 120)}, {rtl("briscv/rtl/tt_bp.sv", 194)}, {rtl("briscv/rtl/tt_ex.sv", 293)}.</li>
<li>Code: {code("tt_metal/tt-llk/tests/helpers/include/barrier.h", 148)}, {code("tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines/cfg_defines.h", 1731)}.</li>
<li>ISA docs: {isa("TensixTile/BabyRISCV/README.md", "Baby RISCV")}.</li></ul>""",
)
