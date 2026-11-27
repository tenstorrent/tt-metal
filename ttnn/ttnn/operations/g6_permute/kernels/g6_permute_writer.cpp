// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_permute writer (rows_tile): exchange_rows_block + store_tiles_block.
// exchange: every live row (a, b) of the block moves from input slot a - 32*ta at tile row
// b mod 32 to staging page b - b0 at tile row a mod 32, as two 32-B face-row segments copied by
// self-aimed NoC reads (bit-exact; no compute engine touches the values).
// store: each staging page leaves as one whole 2048-B output tile.
// cb_output_staging is writer-private scratch: no handshake, base taken once.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "g6_permute_common.hpp"

using namespace g6_permute;

constexpr uint32_t cb_input_rows = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_staging = get_compile_time_arg_val(1);
constexpr uint32_t A = get_compile_time_arg_val(2);
constexpr uint32_t B = get_compile_time_arg_val(3);
constexpr uint32_t tensor_a_tiles = get_compile_time_arg_val(4);
constexpr uint32_t tensor_b_tiles = get_compile_time_arg_val(5);
constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(6);
constexpr uint32_t a_tile_rows = get_compile_time_arg_val(7);
constexpr uint32_t block_b_rows = get_compile_time_arg_val(8);
constexpr uint32_t face_row_bytes = get_compile_time_arg_val(9);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(10);
constexpr auto output_args = TensorAccessorArgs<11>();

using Geometry = TileGeometry<face_row_bytes>;
static_assert(Geometry::tile_bytes == tile_bytes, "tile page must be four 16x16 faces");

using Cursor = BlockCursor<B, tensor_a_tiles, tensor_w_tiles, block_b_rows>;

FORCE_INLINE void issue_row_copies(
    Noc noc, uint32_t slots_base, uint32_t staging_base, const Cursor& cursor, uint32_t run) {
    const uint32_t a_live = cursor.a_live(A);
    const uint32_t r_in0 = cursor.b % TILE_ROWS;
    for (uint32_t j = 0; j < run; ++j) {
        const uint32_t src_row = Geometry::off(r_in0 + j, 0);
        const uint32_t dst_page = staging_base + j * tile_bytes;
        for (uint32_t s = 0; s < a_live; ++s) {
            const uint32_t src = slots_base + s * tile_bytes + src_row;
            const uint32_t dst = dst_page + Geometry::off(s, 0);
            dataflow_kernel_lib::read_with_state<face_row_bytes>(noc, dst, src);
            dataflow_kernel_lib::read_with_state<face_row_bytes>(
                noc, dst + Geometry::face_bytes, src + Geometry::face_bytes);
        }
    }
}

template <typename Accessor>
FORCE_INLINE void issue_tile_writes(const Accessor& output, uint32_t staging_base, const Cursor& cursor, uint32_t run) {
    for (uint32_t j = 0; j < run; ++j) {
        const uint32_t page = ((cursor.n * B + cursor.b + j) * tensor_a_tiles + cursor.ta) * tensor_w_tiles + cursor.tw;
        noc_async_write(staging_base + j * tile_bytes, output.get_noc_addr(page), tile_bytes);
    }
}

template <bool zoned>
FORCE_INLINE void exchange_rows_block(
    Noc noc, uint32_t staging_base, const Cursor& cursor, uint32_t run, bool staging_in_flight) {
    G6_ZONED_STEP(zoned, "PERM-WR-WAIT", cb_wait_front(cb_input_rows, a_tile_rows));
    if (staging_in_flight) {
        // The previous block's tile writes must have left staging before it is refilled.
        G6_ZONED_STEP(zoned, "PERM-WR-FLUSH", noc_async_writes_flushed());
    }
    const uint32_t slots_base = get_read_ptr(cb_input_rows);
    G6_ZONED_STEP(zoned, "PERM-WR-XCHG-ISSUE", issue_row_copies(noc, slots_base, staging_base, cursor, run));
    G6_ZONED_STEP(zoned, "PERM-WR-XCHG-BARRIER", noc_async_read_barrier());
    cb_pop_front(cb_input_rows, a_tile_rows);
}

template <bool zoned, typename Accessor>
FORCE_INLINE void store_tiles_block(const Accessor& output, uint32_t staging_base, const Cursor& cursor, uint32_t run) {
    G6_ZONED_STEP(zoned, "PERM-WR-STORE-ISSUE", issue_tile_writes(output, staging_base, cursor, run));
}

void kernel_main() {
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t core_start_unit = get_arg_val<uint32_t>(0);
    const uint32_t core_num_units = get_arg_val<uint32_t>(1);

    const auto output = TensorAccessor(output_args, output_addr, tile_bytes);
    const uint32_t staging_base = get_write_ptr(cb_output_staging);
    Cursor cursor(core_start_unit, core_num_units);

    // Copy size and source coordinates (this core) are fixed for the kernel; each copy rewrites
    // only the two L1 addresses.
    Noc noc;
    dataflow_kernel_lib::set_read_state<face_row_bytes>(noc, get_read_ptr(cb_input_rows));

    uint32_t block_idx = 0;
    for (; block_idx < ZONED_BLOCKS && !cursor.done(); ++block_idx) {
        const uint32_t run = cursor.extent();
        exchange_rows_block<true>(noc, staging_base, cursor, run, block_idx > 0);
        store_tiles_block<true>(output, staging_base, cursor, run);
        cursor.advance(run);
    }
    if (!cursor.done()) {
        DeviceZoneScopedN("PERM-WR-TAIL");
        while (!cursor.done()) {
            const uint32_t run = cursor.extent();
            exchange_rows_block<false>(noc, staging_base, cursor, run, true);
            store_tiles_block<false>(output, staging_base, cursor, run);
            cursor.advance(run);
        }
    }

    {
        DeviceZoneScopedN("PERM-WR-FINAL-BARRIER");
        noc_async_write_barrier();
    }
}
