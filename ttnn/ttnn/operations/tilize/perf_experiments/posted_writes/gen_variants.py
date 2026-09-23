"""Generate the posted_writes bake-off kernel dirs (one dir per variant => fresh JIT per variant).

Every variant is a byte-for-byte copy of the op's current kernels (../../kernels) with ONLY the
store_rows tile-write issue + its completion changed:

  baseline      noc_async_write (any-len path), non-posted; writes_flushed per quantum; write_barrier at end
  onepkt        noc_async_write_one_packet, non-posted (drops the any-len loop; same ack traffic)
  onepkt_posted noc_async_write_one_packet<posted=true>; posted_writes_flushed per quantum + at end
  posted        noc_async_write<posted=true> (any-len path); posted_writes_flushed per quantum + at end
  state         write_set_state once per quantum (NOC_CTRL + length), then per tile only
                coordinate + src + dst + send (4 register writes instead of 6), non-posted
  state_posted  same issue path, posted

store_rows is the only writer path the variants touch; TileStorer (write_ahead knob) is untouched.
"""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "../../kernels")

WRITE_CALL = """                noc_async_write(
                    l1_row_addr + t * out_tile_bytes,
                    accessor.get_noc_addr(row_tile_idx + t, 0, noc),
                    out_tile_bytes,
                    noc);"""
FLUSH = """        noc_async_writes_flushed();
        if constexpr (noc_split != 0) {"""
ISSUE_OPEN = """        MaybeDeviceZoneScope("writer_issue");  // tile-write issue: address gen + command-buffer writes
        uint32_t l1_row_addr = get_read_ptr(cb_output_tiles);"""
END_BARRIER = """    noc_async_write_barrier();
    if constexpr (write_noc_split != 0) {"""

# posted_writes: cheaper issue path. The destination coordinate changes every tile (interleaved
# pages rotate over the DRAM banks), so the stock *_with_state (which never rewrites the
# coordinate) cannot be used as-is: rewrite the coordinate register, then reuse the stock
# ncrisc_noc_write_with_state (src, dst lo, send, counters for either noc_mode).
HELPER = r"""
namespace posted_writes_bench {
// Raw NoC command-buffer path (bypasses noc_async_write): NOC_CTRL + length are programmed once
// per quantum by ncrisc_noc_write_set_state, so a tile write is coordinate + src + dst + send.
template <bool posted>
FORCE_INLINE void write_tile_with_coord(uint32_t src, uint64_t dst, uint8_t noc) {
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dst >> NOC_ADDR_COORD_SHIFT));
    ncrisc_noc_write_with_state<noc_mode, posted, true /*update_counter*/, true /*one_packet*/>(
        noc, write_cmd_buf, src, (uint32_t)dst);
}
}  // namespace posted_writes_bench
"""

VARIANTS = {
    # name: (write replacement, posted, set_state)
    "baseline": (None, False, False),
    "onepkt": ("noc_async_write_one_packet<true, false>", False, False),
    "onepkt_posted": ("noc_async_write_one_packet<true, true>", True, False),
    "posted": ("noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>", True, False),
    "state": ("state", False, True),
    "state_posted": ("state", True, True),
}


def sub(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)


def gen(name, write, posted, set_state):
    d = os.path.join(HERE, "kernels_" + name)
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(SRC, d)
    hp = os.path.join(d, "tilize_stick_reads.hpp")
    h = open(hp).read()
    if write is not None:
        if set_state:
            p = "true" if posted else "false"
            h = sub(h, "namespace tilize_dataflow {", HELPER + "\nnamespace tilize_dataflow {")
            h = sub(
                h,
                ISSUE_OPEN,
                ISSUE_OPEN
                + f"""
        static_assert(noc_split == 0 && out_tile_bytes <= NOC_MAX_BURST_SIZE, "one packet, one NoC");
        ncrisc_noc_write_set_state<{p}, true /*one_packet*/>(
            noc_index, write_cmd_buf, 0 /*coordinate rewritten per tile*/, out_tile_bytes, NOC_UNICAST_WRITE_VC);""",
            )
            h = sub(
                h,
                WRITE_CALL,
                f"""                posted_writes_bench::write_tile_with_coord<{p}>(
                    l1_row_addr + t * out_tile_bytes, accessor.get_noc_addr(row_tile_idx + t, 0, noc), noc);""",
            )
        else:
            if "one_packet" in write:
                h = sub(
                    h,
                    ISSUE_OPEN,
                    ISSUE_OPEN + '\n        static_assert(out_tile_bytes <= NOC_MAX_BURST_SIZE, "one packet");',
                )
            h = sub(h, WRITE_CALL, WRITE_CALL.replace("noc_async_write(", write + "(", 1))
    if posted:
        # The CB slot may be released once the posted writes have LEFT L1 (sent counter), exactly
        # like writes_flushed for non-posted. There is no "landed" counter for posted writes.
        h = sub(h, FLUSH, FLUSH.replace("noc_async_writes_flushed();", "noc_async_posted_writes_flushed();"))
    open(hp, "w").write(h)
    if posted:
        wp = os.path.join(d, "tilize_writer.cpp")
        w = open(wp).read()
        # End of kernel: the strongest completion posted writes have is "sent". Keep the
        # non-posted barrier (other paths) and add the posted sent-wait.
        w = sub(w, END_BARRIER, "    noc_async_posted_writes_flushed();\n" + END_BARRIER)
        open(wp, "w").write(w)


if __name__ == "__main__":
    for n, v in VARIANTS.items():
        gen(n, *v)
    print("generated", list(VARIANTS))
