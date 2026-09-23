"""hop_aware_coread: generate kernels_mask/ from the op's CURRENT kernels (git-ignored, regenerate).

kernels_mask = the op's kernels with ONE change: on the co-read path (CT co_read != 0) the split of
a tile-row's tile_h stick reads between NCRISC (NoC0) and BRISC (NoC1) is a per-core 32-bit STEP
MASK runtime arg instead of the fixed positional cut [0, tile_h - co_read) / [tile_h - co_read,
tile_h). Bit s of the mask set = sequence step s (stick (s + stick_rotation) mod tile_h, the same
per-core rotated order as today) is read by BRISC; clear = by NCRISC. The mask is appended as
reader RT arg 10 and writer RT arg 9 (co-read excludes the padded path, the only user of reader
RT args >= 10). The host (tests/.../test_tilize_perf2_hop_aware_coread.py) picks the mask:
positional (== HEAD's split, byte-identical read order), geometry-preferred, balanced, inverted.
No helper is bypassed: both RISC-Vs still issue plain noc_async_read through the TensorAccessor.
"""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "../../kernels")


def sub(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)


MASKED_FN = """
// hop_aware_coread: the stick reads of ONE tile-row whose sequence steps are the set bits of
// `step_mask` (bit s = step s, stick (s + stick_rotation) mod tile_h), in step order; stops at
// the last set bit. The co-read split: NCRISC passes ~mask, BRISC passes mask.
template <uint32_t tile_h, uint32_t block_stick_bytes, uint32_t page_bytes, uint32_t pages_per_stick, typename Accessor>
FORCE_INLINE void read_tile_row_sticks_masked(
    const Accessor& accessor,
    uint32_t first_stick,
    uint32_t l1_base,
    uint32_t segment_offset,
    uint32_t segment_bytes,
    uint32_t stick_rotation,
    uint32_t step_mask) {
    for (uint32_t s = 0; step_mask != 0; ++s, step_mask >>= 1) {
        if ((step_mask & 1) == 0) {
            continue;
        }
        const uint32_t stick = (s + stick_rotation) & (tile_h - 1);
        const uint32_t l1_dst = l1_base + stick * block_stick_bytes;
        if constexpr (pages_per_stick == 1) {
            noc_async_read(accessor.get_noc_addr(first_stick + stick, segment_offset), l1_dst, segment_bytes);
        } else {
            read_paged_segment<page_bytes, pages_per_stick>(
                accessor, first_stick + stick, segment_offset, l1_dst, segment_bytes);
        }
    }
}

// Co-read (Refinement 8, CO_READ_SHARE)"""


def gen():
    d = os.path.join(HERE, "kernels_mask")
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(SRC, d)

    h = open(os.path.join(d, "tilize_stick_reads.hpp")).read()
    h = sub(h, "\n// Co-read (Refinement 8, CO_READ_SHARE)", MASKED_FN)
    h = sub(
        h,
        "    uint32_t stick_stride_bytes = 0;\n",
        "    uint32_t stick_stride_bytes = 0;\n"
        "    uint32_t co_read_mask = 0;  // hop_aware_coread: steps BRISC reads (bit s = step s)\n",
    )
    h = sub(
        h,
        """        } else {
            read_tile_row_sticks<tile_h, block_stick_bytes, page_bytes, pages_per_stick, noc_split>(
                accessor, first_stick, l1_base, segment_offset, segment_bytes, stick_rotation, 0, tile_h - co_read);
        }""",
        """        } else if constexpr (co_read != 0) {
            read_tile_row_sticks_masked<tile_h, block_stick_bytes, page_bytes, pages_per_stick>(
                accessor, first_stick, l1_base, segment_offset, segment_bytes, stick_rotation,
                ~co_read_mask & (tile_h == 32 ? 0xFFFFFFFFu : ((1u << tile_h) - 1)));
        } else {
            read_tile_row_sticks<tile_h, block_stick_bytes, page_bytes, pages_per_stick, noc_split>(
                accessor, first_stick, l1_base, segment_offset, segment_bytes, stick_rotation, 0, tile_h - co_read);
        }""",
    )
    open(os.path.join(d, "tilize_stick_reads.hpp"), "w").write(h)

    r = open(os.path.join(d, "tilize_reader.cpp")).read()
    r = sub(
        r,
        "    producer.prime(input_accessor, row_start * tile_h, stick_page_bytes);\n",
        "    producer.prime(input_accessor, row_start * tile_h, stick_page_bytes);\n"
        "    if constexpr (co_read != 0) {\n"
        "        producer.co_read_mask = get_arg_val<uint32_t>(10);  // hop_aware_coread step mask\n"
        "    }\n",
    )
    open(os.path.join(d, "tilize_reader.cpp"), "w").write(r)

    w = open(os.path.join(d, "tilize_writer.cpp")).read()
    w = sub(
        w,
        """        tilize_dataflow::read_tile_row_sticks<tile_h, block_width * tile_col_bytes, page_bytes, pages_per_stick>(
            TensorAccessor(input_args, src_addr, stick_page_bytes),
            load_walk.row() * tile_h,
            get_write_ptr(cb_input_sticks),
            load_walk.first_col() * tile_col_bytes,
            load_walk.valid_width() * tile_col_bytes,
            stick_rotation & (tile_h - 1),
            tile_h - co_read,
            tile_h);""",
        """        tilize_dataflow::read_tile_row_sticks_masked<tile_h, block_width * tile_col_bytes, page_bytes, pages_per_stick>(
            TensorAccessor(input_args, src_addr, stick_page_bytes),
            load_walk.row() * tile_h,
            get_write_ptr(cb_input_sticks),
            load_walk.first_col() * tile_col_bytes,
            load_walk.valid_width() * tile_col_bytes,
            stick_rotation & (tile_h - 1),
            get_arg_val<uint32_t>(9));  // hop_aware_coread step mask""",
    )
    open(os.path.join(d, "tilize_writer.cpp"), "w").write(w)
    print("generated", d)


LISTED_FN = """
// hop_aware_coread (kernels_list): the stick reads of ONE tile-row named by an explicit per-RISC-V
// list in the runtime args: arg list_arg = n, then n stick indices (0..tile_h-1) packed one per
// byte, in issue order. Same per-read work as read_tile_row_sticks (one byte load replaces the
// rotation arithmetic), so any split / order costs what the positional one costs.
template <uint32_t tile_h, uint32_t block_stick_bytes, uint32_t page_bytes, uint32_t pages_per_stick, typename Accessor>
FORCE_INLINE void read_tile_row_sticks_listed(
    const Accessor& accessor,
    uint32_t first_stick,
    uint32_t l1_base,
    uint32_t segment_offset,
    uint32_t segment_bytes,
    uint32_t list_arg) {
    const uint32_t n = get_arg_val<uint32_t>(list_arg);
    const tt_l1_ptr uint8_t* list = reinterpret_cast<const tt_l1_ptr uint8_t*>(get_arg_addr(list_arg + 1));
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t stick = list[i];
        const uint32_t l1_dst = l1_base + stick * block_stick_bytes;
        if constexpr (pages_per_stick == 1) {
            noc_async_read(accessor.get_noc_addr(first_stick + stick, segment_offset), l1_dst, segment_bytes);
        } else {
            read_paged_segment<page_bytes, pages_per_stick>(
                accessor, first_stick + stick, segment_offset, l1_dst, segment_bytes);
        }
    }
}

// Co-read (Refinement 8, CO_READ_SHARE)"""


def gen_list():
    d = os.path.join(HERE, "kernels_list")
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(SRC, d)

    h = open(os.path.join(d, "tilize_stick_reads.hpp")).read()
    h = sub(h, "\n// Co-read (Refinement 8, CO_READ_SHARE)", LISTED_FN)
    h = sub(
        h,
        """        } else {
            read_tile_row_sticks<tile_h, block_stick_bytes, page_bytes, pages_per_stick, noc_split>(
                accessor, first_stick, l1_base, segment_offset, segment_bytes, stick_rotation, 0, tile_h - co_read);
        }""",
        """        } else if constexpr (co_read != 0) {
            // hop_aware_coread: NCRISC's share = the list at reader RT arg 10
            read_tile_row_sticks_listed<tile_h, block_stick_bytes, page_bytes, pages_per_stick>(
                accessor, first_stick, l1_base, segment_offset, segment_bytes, 10);
        } else {
            read_tile_row_sticks<tile_h, block_stick_bytes, page_bytes, pages_per_stick, noc_split>(
                accessor, first_stick, l1_base, segment_offset, segment_bytes, stick_rotation, 0, tile_h - co_read);
        }""",
    )
    open(os.path.join(d, "tilize_stick_reads.hpp"), "w").write(h)

    w = open(os.path.join(d, "tilize_writer.cpp")).read()
    w = sub(
        w,
        """        tilize_dataflow::read_tile_row_sticks<tile_h, block_width * tile_col_bytes, page_bytes, pages_per_stick>(
            TensorAccessor(input_args, src_addr, stick_page_bytes),
            load_walk.row() * tile_h,
            get_write_ptr(cb_input_sticks),
            load_walk.first_col() * tile_col_bytes,
            load_walk.valid_width() * tile_col_bytes,
            stick_rotation & (tile_h - 1),
            tile_h - co_read,
            tile_h);""",
        """        tilize_dataflow::read_tile_row_sticks_listed<tile_h, block_width * tile_col_bytes, page_bytes, pages_per_stick>(
            TensorAccessor(input_args, src_addr, stick_page_bytes),
            load_walk.row() * tile_h,
            get_write_ptr(cb_input_sticks),
            load_walk.first_col() * tile_col_bytes,
            load_walk.valid_width() * tile_col_bytes,
            9);  // hop_aware_coread: BRISC's share = the list at writer RT arg 9""",
    )
    open(os.path.join(d, "tilize_writer.cpp"), "w").write(w)
    print("generated", d)


LISTED16_FN = """
// hop_aware_coread (kernels_list16): like read_tile_row_sticks_listed, but the count is a
// compile-time constant n (the balanced split: tile_h - co_read on NCRISC, co_read on BRISC) and
// the packed list words are loaded once per 4 sticks and unpacked in registers, so the loop has the
// positional loop's constant trip count (no per-read L1 byte load, no data-dependent exit).
template <uint32_t n, uint32_t tile_h, uint32_t block_stick_bytes, uint32_t page_bytes, uint32_t pages_per_stick, typename Accessor>
FORCE_INLINE void read_tile_row_sticks_listed_n(
    const Accessor& accessor,
    uint32_t first_stick,
    uint32_t l1_base,
    uint32_t segment_offset,
    uint32_t segment_bytes,
    uint32_t list_arg) {
    static_assert(n % 4 == 0, "packed 4 sticks per word");
    for (uint32_t w = 0; w < n / 4; ++w) {
        uint32_t word = get_arg_val<uint32_t>(list_arg + w);
#pragma GCC unroll 4
        for (uint32_t k = 0; k < 4; ++k, word >>= 8) {
            const uint32_t stick = word & 0xFF;
            const uint32_t l1_dst = l1_base + stick * block_stick_bytes;
            if constexpr (pages_per_stick == 1) {
                noc_async_read(accessor.get_noc_addr(first_stick + stick, segment_offset), l1_dst, segment_bytes);
            } else {
                read_paged_segment<page_bytes, pages_per_stick>(
                    accessor, first_stick + stick, segment_offset, l1_dst, segment_bytes);
            }
        }
    }
}

// Co-read (Refinement 8, CO_READ_SHARE)"""


def gen_list16():
    d = os.path.join(HERE, "kernels_list16")
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(os.path.join(HERE, "kernels_list"), d)
    h = open(os.path.join(d, "tilize_stick_reads.hpp")).read()
    h = sub(h, "\n// Co-read (Refinement 8, CO_READ_SHARE)", LISTED16_FN)
    h = sub(
        h,
        """            read_tile_row_sticks_listed<tile_h, block_stick_bytes, page_bytes, pages_per_stick>(
                accessor, first_stick, l1_base, segment_offset, segment_bytes, 10);""",
        """            read_tile_row_sticks_listed_n<tile_h - co_read, tile_h, block_stick_bytes, page_bytes, pages_per_stick>(
                accessor, first_stick, l1_base, segment_offset, segment_bytes, 11);""",
    )
    open(os.path.join(d, "tilize_stick_reads.hpp"), "w").write(h)
    w = open(os.path.join(d, "tilize_writer.cpp")).read()
    w = sub(
        w,
        "tilize_dataflow::read_tile_row_sticks_listed<tile_h, block_width * tile_col_bytes, page_bytes, pages_per_stick>(",
        "tilize_dataflow::read_tile_row_sticks_listed_n<co_read, tile_h, block_width * tile_col_bytes, page_bytes, pages_per_stick>(",
    )
    w = sub(
        w,
        "            9);  // hop_aware_coread: BRISC's share = the list at writer RT arg 9",
        "            10);  // hop_aware_coread: BRISC's share = the packed list at writer RT arg 10",
    )
    open(os.path.join(d, "tilize_writer.cpp"), "w").write(w)
    print("generated", d)


LISTEDW_FN = """
// hop_aware_coread (kernels_listw, the candidate): the stick reads of ONE tile-row named by an
// explicit per-RISC-V list in the runtime args -- arg list_arg = n, then the n stick indices
// (0..tile_h-1) packed 4 per word (low byte first), in issue order. Any split (balanced or not)
// is expressible; the words are loaded once per 4 sticks and unpacked in registers, so the
// per-read work matches read_tile_row_sticks' (no per-read L1 byte load, 4-way unrolled).
template <uint32_t tile_h, uint32_t block_stick_bytes, uint32_t page_bytes, uint32_t pages_per_stick, typename Accessor>
FORCE_INLINE void read_tile_row_sticks_listed_w(
    const Accessor& accessor,
    uint32_t first_stick,
    uint32_t l1_base,
    uint32_t segment_offset,
    uint32_t segment_bytes,
    uint32_t list_arg) {
    auto read_one = [&](uint32_t stick) {
        const uint32_t l1_dst = l1_base + stick * block_stick_bytes;
        if constexpr (pages_per_stick == 1) {
            noc_async_read(accessor.get_noc_addr(first_stick + stick, segment_offset), l1_dst, segment_bytes);
        } else {
            read_paged_segment<page_bytes, pages_per_stick>(
                accessor, first_stick + stick, segment_offset, l1_dst, segment_bytes);
        }
    };
    const uint32_t n = get_arg_val<uint32_t>(list_arg);
    const uint32_t full = n >> 2;
    for (uint32_t w = 0; w < full; ++w) {
        uint32_t word = get_arg_val<uint32_t>(list_arg + 1 + w);
#pragma GCC unroll 4
        for (uint32_t k = 0; k < 4; ++k, word >>= 8) {
            read_one(word & 0xFF);
        }
    }
    if ((n & 3) != 0) {
        uint32_t word = get_arg_val<uint32_t>(list_arg + 1 + full);
        for (uint32_t k = 0; k < (n & 3); ++k, word >>= 8) {
            read_one(word & 0xFF);
        }
    }
}

// Co-read (Refinement 8, CO_READ_SHARE)"""


def gen_listw():
    d = os.path.join(HERE, "kernels_listw")
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(os.path.join(HERE, "kernels_list"), d)
    h = open(os.path.join(d, "tilize_stick_reads.hpp")).read()
    h = sub(h, "\n// Co-read (Refinement 8, CO_READ_SHARE)", LISTEDW_FN)
    h = sub(
        h,
        """            read_tile_row_sticks_listed<tile_h, block_stick_bytes, page_bytes, pages_per_stick>(
                accessor, first_stick, l1_base, segment_offset, segment_bytes, 10);""",
        """            read_tile_row_sticks_listed_w<tile_h, block_stick_bytes, page_bytes, pages_per_stick>(
                accessor, first_stick, l1_base, segment_offset, segment_bytes, 10);""",
    )
    open(os.path.join(d, "tilize_stick_reads.hpp"), "w").write(h)
    w = open(os.path.join(d, "tilize_writer.cpp")).read()
    w = sub(
        w,
        "tilize_dataflow::read_tile_row_sticks_listed<tile_h, block_width * tile_col_bytes, page_bytes, pages_per_stick>(",
        "tilize_dataflow::read_tile_row_sticks_listed_w<tile_h, block_width * tile_col_bytes, page_bytes, pages_per_stick>(",
    )
    open(os.path.join(d, "tilize_writer.cpp"), "w").write(w)
    print("generated", d)


SELECT_FN = """
// hop_aware_coread (kernels_listsel): the host cannot see physical Tensix coordinates (the
// bindings expose only virtual / translated ones, and WH row harvesting shifts the physical row),
// so it precomputes the stick lists for every physical row this core can be on (K candidates) and
// the kernel picks by its own physical NoC0 node id. RT layout at `arg`: K, then K candidate
// physical NoC0 y values, then K list blocks of `block_words` words each ([n, packed sticks]).
// Both RISC-Vs read the same NoC0 register, so they always pick the same split. Raw register read
// (NOC_CMD_BUF_READ_REG / NOC_NODE_ID): no helper returns physical coordinates.
template <uint32_t block_words>
FORCE_INLINE uint32_t select_physical_row_list(uint32_t arg) {
    const uint32_t k = get_arg_val<uint32_t>(arg);
    const uint32_t my_phys_y = (NOC_CMD_BUF_READ_REG(0, 0, NOC_NODE_ID) >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    uint32_t j = 0;
    for (uint32_t c = 0; c < k; ++c) {
        if (get_arg_val<uint32_t>(arg + 1 + c) == my_phys_y) {
            j = c;
        }
    }
    return arg + 1 + k + j * block_words;
}

// hop_aware_coread (kernels_listw, the candidate)"""


def gen_listsel():
    d = os.path.join(HERE, "kernels_listsel")
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(os.path.join(HERE, "kernels_listw"), d)
    h = open(os.path.join(d, "tilize_stick_reads.hpp")).read()
    h = sub(h, "\n// hop_aware_coread (kernels_listw, the candidate)", SELECT_FN)
    h = sub(
        h,
        """            read_tile_row_sticks_listed_w<tile_h, block_stick_bytes, page_bytes, pages_per_stick>(
                accessor, first_stick, l1_base, segment_offset, segment_bytes, 10);""",
        """            read_tile_row_sticks_listed_w<tile_h, block_stick_bytes, page_bytes, pages_per_stick>(
                accessor, first_stick, l1_base, segment_offset, segment_bytes, co_read_list_arg);""",
    )
    h = sub(
        h,
        "    uint32_t stick_stride_bytes = 0;\n",
        "    uint32_t stick_stride_bytes = 0;\n"
        "    uint32_t co_read_list_arg = 10;  // hop_aware_coread (listsel): this core's selected list\n",
    )
    open(os.path.join(d, "tilize_stick_reads.hpp"), "w").write(h)
    r = open(os.path.join(d, "tilize_reader.cpp")).read()
    r = sub(
        r,
        "    producer.prime(input_accessor, row_start * tile_h, stick_page_bytes);\n",
        "    producer.prime(input_accessor, row_start * tile_h, stick_page_bytes);\n"
        "    if constexpr (co_read != 0) {\n"
        "        producer.co_read_list_arg = tilize_dataflow::select_physical_row_list<1 + tile_h / 4>(10);\n"
        "    }\n",
    )
    open(os.path.join(d, "tilize_reader.cpp"), "w").write(r)
    w = open(os.path.join(d, "tilize_writer.cpp")).read()
    w = sub(
        w,
        "            9);  // hop_aware_coread: BRISC's share = the list at writer RT arg 9",
        "            tilize_dataflow::select_physical_row_list<1 + tile_h / 4>(9));  // BRISC's share, selected by physical row",
    )
    open(os.path.join(d, "tilize_writer.cpp"), "w").write(w)
    print("generated", d)


if __name__ == "__main__":
    gen()
    gen_list()
    gen_list16()
    gen_listw()
    gen_listsel()
