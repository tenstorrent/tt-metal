from common import *

D = dict(
    id="WH-09",
    short="Previous Kernel State",
    summary="On an earlier version of #58068 the first kernel of a test depended on the kernel that ran before it on the core, so the result depended on test order and on how pytest split the tests. At #58068 head this no longer happens, and the warm-up pass can be removed.",
    status="Fixed at #58068 head",
    status_cls="st-ok",
    depends=[],
    used_by=[],
    problem="re-measure (test order)",
    what="""<ul>
<li>eltwise_binary X (Float16_b → Float16, Elwmul, HiFi2, dest_acc Yes), L1_TO_L1: 12,347 cycles after a kernel that ran only L1_TO_L1, 10,573 after a full test whose last kernel is L1_CONGESTION (17-commit #58068, warm-up off).</li>
</ul>""",
    hw=f"""<ul>
<li>b7747df1b7d names the cause: "round robin arbiter pointers that only a chip reset clears". L1 access ports are shared through round-robin muxes ({isa("TensixTile/L1.md", "ISA doc: L1")}).</li>
<li>At #58068 head BRISC reads L1 64 times before each release "so every L1 bank arbiter last granted BRISC" ({code("tt_metal/tt-llk/tests/helpers/src/brisc.cpp", 160, "brisc.cpp")}).</li>
</ul>""",
    how="""<ol class="chain">
<li>A kernel ends with the L1 arbiters' round-robin pointers in some state.</li>
<li>The next kernel's first accesses are arbitrated from that state, so the first tiles have a different timing.</li>
<li>That timing selects the packer rhythm or the unpacker/packer L1 order for the rest of the kernel.</li>
</ol>""",
    sketch="two kernels in a row on one time line, with the arbiter pointer as a small dial that the first kernel leaves at different positions.",
    versim="""<p>Not proven in Versim. Four runs of these pairs on the simulator version of the 17-commit tree stopped and never finished; three stopped at exactly the same point in the waveform (8,705,842,323 bytes), whichever test was running. At #58068 head the effect is gone, so there is nothing to simulate there.</p>""",
    card="""<div class="tw"><table><tr><th>What ran before X</th><th class="n">17-commit #58068</th><th class="n">#58068 head, barrier off</th><th class="n">#58068 head</th></tr>
<tr><td>a kernel that ran only L1_TO_L1</td><td class="n">12,347</td><td class="n">12,345</td><td class="n">12,345</td></tr>
<tr><td>a full test (last kernel L1_CONGESTION)</td><td class="n">10,573</td><td class="n">12,344</td><td class="n">12,345</td></tr></table></div>
<p>#58068 head, warm-up off, 323 test cases (1,346 values): the same values as with warm-up, in forward and in reverse test order (0 move by more than 0.5%).</p>""",
    fix=f"""<ul><li><b>b7747df1b7d, 75669afc17a</b>: one unrecorded pass over the run types of each test before measuring ({code("tt_metal/tt-llk/tests/python_tests/helpers/perf/core.py", 843, "core.py warm-up")}). This fixed it on the 17-commit version.</li>
<li>At #58068 head the effect is gone even with the warm-up and the barrier off. The barrier's arbiter reset is a likely reason when it is on; we did not find the reason when it is off.</li></ul>
<p>Recommendation: remove the warm-up pass (about 5% run time).</p>""",
    ba="""<div class="tw"><table><tr><th>Card</th><th class="n">Before (17-commit, no warm-up)</th><th class="n">After (#58068 head, no warm-up)</th></tr>
<tr><td>X after a full test / after L1_TO_L1 only</td><td class="n">10,573 / 12,347</td><td class="n">12,345 / 12,345</td></tr>
<tr><td>323 cases, forward vs reverse order</td><td class="n">–</td><td class="n">0 values move</td></tr></table></div>""",
    open="""<ul><li>Which change between the 17-commit version and the head removes it with the barrier off.</li>
<li>No Versim proof.</li></ul>""",
    repro="""<ul><li>17-commit: branch <code>nstojictt/p58-v17-nowarm</code>, <code>LLK_PERF_NO_WARMUP=1</code>; run P then X in one pytest call, with and without <code>LLK_PERF_RUN_TYPES=L1_TO_L1</code>.</li>
<li>Head: <code>nstojictt/p58-versim</code>, <code>LLK_PERF_NO_WARMUP=1</code>, forward and reversed id lists.</li></ul>""",
    refs=f"""<ul class="refs"><li>Code: {code("tt_metal/tt-llk/tests/helpers/src/brisc.cpp", 160)}, {code("tt_metal/tt-llk/tests/python_tests/helpers/perf/core.py", 843)}.</li>
<li>ISA docs: {isa("TensixTile/L1.md", "L1")}.</li></ul>""",
)
