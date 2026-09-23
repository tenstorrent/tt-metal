"""Generate write_throttle variant kernel dirs (each its own path => fresh JIT compile).

A variant name is a '+'-joined list of tokens; each token becomes a #define at the top of
tilize_writer.cpp and tilize_reader.cpp, and the patched hooks in tilize_stick_reads.hpp /
tilize_reader.cpp read them:
  wsN  - writer: sliding ACK window, at most N tile writes un-ACKED (WT_WS=N)
  wbN  - writer: noc_async_write_barrier after every N tile writes (WT_WB=N)
  wpK  - writer: riscv_wait(K) cycles after every tile write (WT_SPIN=K)
  rsM  - bank-coalesced reader: at most M DRAM reads un-landed (RD_RS=M)
  base - the current kernels, unchanged (ctl: the same, another dir: A/A noise control)
Host knobs (e.g. BANK_COALESCE_STAGE_DEPTH=1) are applied by the test, not here.
"""
import os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "../../kernels")

MACRO = {"ws": "WT_WS", "wb": "WT_WB", "wp": "WT_SPIN", "rs": "RD_RS"}


def patch_once(s, needle, repl):
    assert s.count(needle) == 1, needle
    return s.replace(needle, repl)


WRITE = """                noc_async_write(
                    l1_row_addr + t * out_tile_bytes,
                    accessor.get_noc_addr(row_tile_idx + t, 0, noc),
                    out_tile_bytes,
                    noc);"""
WRITE_HOOKED = (
    """#ifdef WT_WS
                // sliding ACK window: issue only while fewer than WT_WS writes are un-ACKED
                while ((uint32_t)(noc_nonposted_writes_acked[noc] -
                                  NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED)) >= WT_WS) {
                }
#endif
"""
    + WRITE
    + """
#ifdef WT_WB
                if (++wt_issued == WT_WB) {
                    wt_issued = 0;
                    noc_async_write_barrier(noc);
                }
#endif
#ifdef WT_SPIN
                riscv_wait(WT_SPIN);
#endif"""
)

READ = """                noc_async_read(
                    accessor.get_noc_addr(first_stick + j),
                    run_stage + run.offset(j) * stick_page_bytes,
                    run.sticks(j) * stick_page_bytes);"""
READ_HOOKED = (
    """#ifdef RD_RS
                // sliding read window: issue only while fewer than RD_RS reads are un-landed
                while ((uint32_t)(noc_reads_num_issued[noc_index] -
                                  NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED)) >= RD_RS) {
                }
#endif
"""
    + READ
)


def gen(v):
    d = os.path.join(HERE, "kernels_" + v.replace("+", "_"))
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(SRC, d)
    defs = ""
    for tok in v.split("+"):
        if tok in ("base", "ctl"):  # ctl: an A/A copy of base under another path (noise control)
            continue
        defs += f"#define {MACRO[tok[:2]]} {int(tok[2:])}\n"
    h = open(os.path.join(d, "tilize_stick_reads.hpp")).read()
    h = patch_once(h, WRITE, WRITE_HOOKED)
    # wt_issued: the batch-barrier counter, per quantum (N divides the quantum's 4 writes here)
    h = patch_once(
        h,
        """    const uint32_t pages = num_rows * block_width;
    {
        MaybeDeviceZoneScope("writer_wait");  // starved on compute
        cb_wait_front(cb_output_tiles, pages);""",
        """    const uint32_t pages = num_rows * block_width;
#ifdef WT_WB
    uint32_t wt_issued = 0;
#endif
    {
        MaybeDeviceZoneScope("writer_wait");  // starved on compute
        cb_wait_front(cb_output_tiles, pages);""",
    )
    open(os.path.join(d, "tilize_stick_reads.hpp"), "w").write(h)
    r = open(os.path.join(d, "tilize_reader.cpp")).read()
    r = patch_once(r, READ, READ_HOOKED)
    open(os.path.join(d, "tilize_reader.cpp"), "w").write(defs + r)
    w = open(os.path.join(d, "tilize_writer.cpp")).read()
    open(os.path.join(d, "tilize_writer.cpp"), "w").write(defs + w)
    return d


if __name__ == "__main__":
    for v in sys.argv[1:]:
        print(gen(v))
