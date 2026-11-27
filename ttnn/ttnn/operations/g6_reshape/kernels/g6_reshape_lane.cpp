// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_reshape lane: one data-movement RISC-V runs the whole read -> assemble -> write pipeline for its
// own contiguous run of tile-side strips. Lane 0 is NCRISC on NoC0, lane 1 is BRISC on NoC1, built
// from this one source; no CB credit passes between lanes, so the CBs are plain lane-private L1.
//
// No kernel_lib helper applies: there is no compute stage, and the transfer unit is a face-row
// segment inside a tile page (tilize/untilize helpers move whole sticks or tiles through the
// engines; tile_column helpers copy tile columns). Transfers are raw noc_async_read/write via
// TensorAccessor; staged copies are RISC-V word stores.
//
// Exactly one regime define is set by the host:
//   G6_REGIME_GATHER_ALIGNED  tile side = output; map-side face rows read direct when DRAM-congruent, else staged
//   G6_REGIME_GATHER_GENERAL  tile side = output; every map-side piece staged through its DRAM granule
//   G6_REGIME_SCATTER_ALIGNED tile side = input; map-side face rows written straight from the tile images

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_reshape_index_map.hpp"

#if (defined(G6_REGIME_GATHER_ALIGNED) + defined(G6_REGIME_GATHER_GENERAL) + defined(G6_REGIME_SCATTER_ALIGNED)) != 1
#error "exactly one G6_REGIME_* define must be set"
#endif

namespace {

[[maybe_unused]] constexpr uint32_t lane_id = get_compile_time_arg_val(0);
constexpr uint32_t esize = get_compile_time_arg_val(1);
constexpr uint32_t W_in = get_compile_time_arg_val(2);
constexpr uint32_t H_in = get_compile_time_arg_val(3);
constexpr uint32_t Wt_in = get_compile_time_arg_val(4);
constexpr uint32_t Ht_in = get_compile_time_arg_val(5);
constexpr uint32_t W_out = get_compile_time_arg_val(6);
constexpr uint32_t H_out = get_compile_time_arg_val(7);
constexpr uint32_t Wt_out = get_compile_time_arg_val(8);
constexpr uint32_t Ht_out = get_compile_time_arg_val(9);
constexpr uint32_t strip_rows = get_compile_time_arg_val(10);
constexpr uint32_t granule_bytes = get_compile_time_arg_val(11);
constexpr uint32_t block_tiles = get_compile_time_arg_val(12);
constexpr uint32_t slot_depth = get_compile_time_arg_val(13);
constexpr uint32_t staging_slots = get_compile_time_arg_val(14);
constexpr uint32_t copy_entries = get_compile_time_arg_val(15);
constexpr uint32_t cb_tile_images = get_compile_time_arg_val(16);
constexpr uint32_t cb_staging = get_compile_time_arg_val(17);
constexpr auto in_args = TensorAccessorArgs<18>();
constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

constexpr uint32_t face_row_bytes = g6::FACE * esize;
constexpr uint32_t tile_bytes = g6::TILE * g6::TILE * esize;
constexpr uint32_t slot_bytes = block_tiles * tile_bytes;
constexpr uint32_t dram_read_align = NOC_DRAM_READ_ALIGNMENT_BYTES;

static_assert(granule_bytes % dram_read_align == 0, "a staged granule must be a whole DRAM read-alignment unit");
static_assert(granule_bytes == strip_rows * face_row_bytes, "a strip is the rows that share one DRAM granule");
static_assert(slot_depth >= 1 && block_tiles >= 1, "block knobs must be at least 1");

using GeoIn = g6::Geometry<W_in, H_in, Wt_in, Ht_in>;
using GeoOut = g6::Geometry<W_out, H_out, Wt_out, Ht_out>;
#if defined(G6_REGIME_SCATTER_ALIGNED)
using GeoTile = GeoIn;
#else
using GeoTile = GeoOut;
#endif

FORCE_INLINE void store_fence() { asm volatile("fence" ::: "memory"); }

#if !defined(G6_REGIME_SCATTER_ALIGNED)
// Staged map-side reads: each lands one DRAM granule in a ring slot (a NoC read into L1 must be
// congruent with its DRAM source modulo the read alignment), and the copy table records which bytes
// go where once the read barrier has passed. The table sits in the staging CB's tail and holds
// 2 words per entry: dst address, then (src offset in the ring) | (bytes << 16).
class Stager {
public:
    explicit Stager(uint32_t base) :
        ring_base_(base), table_(reinterpret_cast<tt_l1_ptr uint32_t*>(base + staging_slots * granule_bytes)) {}

    FORCE_INLINE void stage(uint64_t src, uint32_t dst, uint32_t bytes) {
        const uint64_t granule = src & ~static_cast<uint64_t>(granule_bytes - 1);
        bool hit = n_staged_ != 0 && granule == memo_;
        if ((!hit && n_staged_ == staging_slots) || n_copies_ == copy_entries) {
            flush();
            hit = false;
        }
        if (!hit) {
            slot_off_ = n_staged_ * granule_bytes;
            noc_async_read(granule, ring_base_ + slot_off_, granule_bytes);
            memo_ = granule;
            ++n_staged_;
        }
        const uint32_t src_off = slot_off_ + (static_cast<uint32_t>(src) & (granule_bytes - 1));
        table_[2 * n_copies_] = dst;
        table_[2 * n_copies_ + 1] = src_off | (bytes << 16);
        ++n_copies_;
    }

    // Retires every read issued so far (direct reads included), then lands the staged bytes.
    FORCE_INLINE void flush() {
        {
            DeviceZoneScopedN("G6_READ_BARRIER");
            noc_async_read_barrier();
        }
        DeviceZoneScopedN("G6_ASSEMBLE");
        for (uint32_t i = 0; i < n_copies_; ++i) {
            const uint32_t dst = table_[2 * i];
            const uint32_t packed = table_[2 * i + 1];
            const uint32_t src = ring_base_ + (packed & 0xFFFF);
            const uint32_t bytes = packed >> 16;
            if (((src | dst | bytes) & 3) == 0) {
                volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
                volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
                for (uint32_t w = 0; w < (bytes >> 2); ++w) {
                    d[w] = s[w];
                }
            } else {
                volatile tt_l1_ptr uint16_t* s = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src);
                volatile tt_l1_ptr uint16_t* d = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst);
                for (uint32_t h = 0; h < (bytes >> 1); ++h) {
                    d[h] = s[h];
                }
            }
        }
        // The NoC writes that source these bytes must not overtake the RISC-V stores.
        store_fence();
        n_staged_ = 0;
        n_copies_ = 0;
    }

private:
    const uint32_t ring_base_;
    tt_l1_ptr uint32_t* const table_;
    uint64_t memo_ = 0;
    uint32_t slot_off_ = 0;
    uint32_t n_staged_ = 0;
    uint32_t n_copies_ = 0;
};
#endif

}  // namespace

void kernel_main() {
    const uint32_t start_tile = get_arg_val<uint32_t>(0);
    const uint32_t start_strip = get_arg_val<uint32_t>(1);
    const uint32_t num_strips = get_arg_val<uint32_t>(2);
    const uint32_t num_blocks = get_arg_val<uint32_t>(3);
    if (num_strips == 0) {
        return;
    }
    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
    const auto in_acc = TensorAccessor(in_args, in_addr, tile_bytes);
    const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);
    const uint32_t images_base = get_write_ptr(cb_tile_images);

#if !defined(G6_REGIME_SCATTER_ALIGNED)
    Stager stager(get_write_ptr(cb_staging));
#endif

    g6::StripCursor cur = g6::cursor_at<GeoTile>(start_tile, start_strip);
    uint32_t remaining = num_strips;

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t slot = images_base + (block_idx % slot_depth) * slot_bytes;
        if (block_idx >= slot_depth) {
            // Writes of an earlier block still source this slot.
            DeviceZoneScopedN("G6_SLOT_REUSE");
            noc_async_writes_flushed();
        }
        const g6::StripCursor block_cur = cur;
        const uint32_t block_remaining = remaining;

#if defined(G6_REGIME_SCATTER_ALIGNED)
        // load_block: the lane's rows of each input tile land at their own intra-tile offsets.
        {
            DeviceZoneScopedN("G6_TILE_ISSUE");
            g6::walk_block<GeoTile, strip_rows, block_tiles>(cur, remaining, [&](const g6::BlockTile& bt) {
                const uint32_t image = slot + bt.k * tile_bytes;
                g6::for_each_tile_run<esize>(
                    bt.r_lo, bt.r_hi, GeoIn::valid_cols(bt.tc) > g6::FACE, [&](uint32_t off, uint32_t bytes) {
                        const uint64_t src = in_acc.get_noc_addr(bt.tile, off);
                        ASSERT(((static_cast<uint32_t>(src) ^ (image + off)) & (dram_read_align - 1)) == 0);
                        noc_async_read(src, image + off, bytes);
                    });
            });
        }
        {
            DeviceZoneScopedN("G6_READ_BARRIER");
            noc_async_read_barrier();
        }
        // scatter_block: every valid input face row goes to the output face row with its linear index.
        {
            DeviceZoneScopedN("G6_MAP_ISSUE");
            g6::StripCursor walk = block_cur;
            uint32_t walk_remaining = block_remaining;
            g6::walk_block<GeoTile, strip_rows, block_tiles>(walk, walk_remaining, [&](const g6::BlockTile& bt) {
                const uint32_t image = slot + bt.k * tile_bytes;
                const uint32_t vc = GeoIn::valid_cols(bt.tc);
                for (uint32_t r = bt.r_lo; r < bt.r_hi; ++r) {
                    const uint32_t g_i = bt.img * H_in + bt.tr * g6::TILE + r;
                    for (uint32_t c = 0; c < vc; c += g6::FACE) {
                        const uint32_t lin = g_i * W_in + bt.tc * g6::TILE + c;
                        const uint32_t g_o = lin / W_out;
                        const uint32_t c_o = lin - g_o * W_out;
                        uint32_t tile, off;
                        g6::locate<GeoOut, esize>(g_o, c_o, tile, off);
                        noc_async_write(
                            image + g6::face_off<esize>(r, c), out_acc.get_noc_addr(tile, off), face_row_bytes);
                    }
                }
            });
        }
#else
        // gather_block: every output segment of the lane's rows is fetched by the index map.
        {
            DeviceZoneScopedN("G6_MAP_ISSUE");
            g6::walk_block<GeoTile, strip_rows, block_tiles>(cur, remaining, [&](const g6::BlockTile& bt) {
                const uint32_t image = slot + bt.k * tile_bytes;
                const uint32_t vc = GeoOut::valid_cols(bt.tc);
                for (uint32_t r = bt.r_lo; r < bt.r_hi; ++r) {
                    const uint32_t g_o = bt.img * H_out + bt.tr * g6::TILE + r;
                    for (uint32_t c = 0; c < vc; c += g6::FACE) {
                        const uint32_t lin = g_o * W_out + bt.tc * g6::TILE + c;
                        const uint32_t dst = image + g6::face_off<esize>(r, c);
                        uint32_t g_i = lin / W_in;
                        uint32_t c_i = lin - g_i * W_in;
#if defined(G6_REGIME_GATHER_ALIGNED)
                        uint32_t tile, off;
                        g6::locate<GeoIn, esize>(g_i, c_i, tile, off);
                        const uint64_t src = in_acc.get_noc_addr(tile, off);
                        if (((static_cast<uint32_t>(src) ^ dst) & (dram_read_align - 1)) == 0) {
                            noc_async_read(src, dst, face_row_bytes);
                        } else {
                            stager.stage(src, dst, face_row_bytes);
                        }
#else
                        // A piece is the part of the segment that is contiguous in one input face row.
                        const uint32_t seg_end = vc - c < g6::FACE ? vc - c : g6::FACE;
                        uint32_t d = dst;
                        for (uint32_t n = seg_end; n != 0;) {
                            uint32_t len = g6::FACE - (c_i % g6::FACE);
                            if (W_in - c_i < len) {
                                len = W_in - c_i;
                            }
                            if (n < len) {
                                len = n;
                            }
                            uint32_t tile, off;
                            g6::locate<GeoIn, esize>(g_i, c_i, tile, off);
                            stager.stage(in_acc.get_noc_addr(tile, off), d, len * esize);
                            d += len * esize;
                            n -= len;
                            c_i += len;
                            if (c_i == W_in) {
                                c_i = 0;
                                ++g_i;
                            }
                        }
#endif
                    }
                }
            });
        }
        // assemble_block
        stager.flush();
        // store_block: one write per merged face run of the lane's rows in each output tile.
        {
            DeviceZoneScopedN("G6_TILE_ISSUE");
            g6::StripCursor walk = block_cur;
            uint32_t walk_remaining = block_remaining;
            g6::walk_block<GeoTile, strip_rows, block_tiles>(walk, walk_remaining, [&](const g6::BlockTile& bt) {
                const uint32_t image = slot + bt.k * tile_bytes;
                g6::for_each_tile_run<esize>(
                    bt.r_lo, bt.r_hi, GeoOut::valid_cols(bt.tc) > g6::FACE, [&](uint32_t off, uint32_t bytes) {
                        noc_async_write(image + off, out_acc.get_noc_addr(bt.tile, off), bytes);
                    });
            });
        }
#endif
    }

    DeviceZoneScopedN("G6_FINAL_WRITE_BARRIER");
    noc_async_write_barrier();
}
