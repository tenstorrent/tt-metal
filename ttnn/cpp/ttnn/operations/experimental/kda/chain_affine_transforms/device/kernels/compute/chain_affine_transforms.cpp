// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/matmul_subblock.hpp"

// product = A @ state. Each output subblock accumulates the whole key dimension in DST, k = 0 .. Kt - 1 in order.
template <uint32_t Kt, uint32_t Vt, typename DFBA, typename DFBState, typename DFBProduct>
FORCE_INLINE void multiply(DFBA& a, DFBState& state, DFBProduct& product) {
    constexpr uint32_t subblock_columns = kda::MatmulSubblock<Kt, Vt>::columns;
    constexpr uint32_t subblock_rows = kda::MatmulSubblock<Kt, Vt>::rows;
    const uint32_t a_id = a.get_id();
    const uint32_t state_id = state.get_id();
    const uint32_t product_id = product.get_id();
    product.reserve_back(Kt * Vt);
    reconfig_data_format<SrcOrder::Reverse>(a_id, state_id);
    matmul_block_init(a_id, state_id, false, subblock_columns, subblock_rows, Kt);
    for (uint32_t row_start = 0; row_start < Kt; row_start += subblock_rows) {
        for (uint32_t column_start = 0; column_start < Vt; column_start += subblock_columns) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < Kt; ++k) {
                matmul_block(
                    a_id,
                    state_id,
                    row_start * Kt + k,
                    k * Vt + column_start,
                    0,
                    false,
                    subblock_columns,
                    subblock_rows,
                    Kt);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t row = 0; row < subblock_rows; ++row) {
                for (uint32_t column = 0; column < subblock_columns; ++column) {
                    pack_tile(
                        row * subblock_columns + column, product_id, (row_start + row) * Vt + column_start + column);
                }
            }
            tile_regs_release();
        }
    }
    product.push_back(Kt * Vt);
}

// state = product + b as an SFPU add of two FP32 DST operands; an FPU add would truncate the FP32 product in srcA.
// product unpacks to DST losslessly and BF16 b widens exactly through srcA. Every packed buffer is FP32, so the
// packer configuration from startup stays valid.
template <uint32_t Tiles, typename DFBProduct, typename DFBB, typename DFBState, typename DFBOut>
FORCE_INLINE void add(DFBProduct& product, DFBB& b, DFBState& state, DFBOut* out) {
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    // product and b interleave in DST: tile i uses DST[2i] and DST[2i + 1].
    constexpr uint32_t pair_tiles = dst_tiles / 2;
    static_assert(pair_tiles > 0);
    const uint32_t product_id = product.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t state_id = state.get_id();
    const uint32_t out_id = out == nullptr ? 0 : out->get_id();
    state.reserve_back(Tiles);
    if (out != nullptr) {
        out->reserve_back(Tiles);
    }
    for (uint32_t first = 0; first < Tiles; first += pair_tiles) {
        const uint32_t count = first + pair_tiles <= Tiles ? pair_tiles : Tiles - first;
        tile_regs_acquire();
        reconfig_data_format_srca(product_id);
        copy_init(product_id);
        for (uint32_t i = 0; i < count; ++i) {
            copy_tile(product_id, first + i, 2 * i);
        }
        reconfig_data_format_srca(product_id, b_id);
        copy_init(b_id);
        for (uint32_t i = 0; i < count; ++i) {
            copy_tile(b_id, first + i, 2 * i + 1);
        }
        add_binary_tile_init();
        for (uint32_t i = 0; i < count; ++i) {
            add_binary_tile(2 * i, 2 * i + 1, 2 * i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < count; ++i) {
            pack_tile(2 * i, state_id, first + i);
            if (out != nullptr) {
                pack_tile(2 * i, out_id, first + i);
            }
        }
        tile_regs_release();
    }
    state.push_back(Tiles);
    if (out != nullptr) {
        out->push_back(Tiles);
    }
}

template <uint32_t Kt, uint32_t Vt, uint32_t steps>
TT_KERNEL void compute() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::a, dfb::initial, dfb::product);
    constexpr uint32_t a_tiles = Kt * Kt;
    constexpr uint32_t state_tiles = Kt * Vt;
    DataflowBuffer initial(dfb::initial);
    DataflowBuffer a(dfb::a);
    DataflowBuffer b(dfb::b);
    DataflowBuffer product(dfb::product);
    DataflowBuffer state(dfb::state);
    DataflowBuffer out(dfb::out);

    kda_chronology::Topology topology{};
    {
        DataflowBuffer chronology(dfb::chronology_compute);
        topology = kda_chronology::receive(chronology);
    }
    const uint32_t entry_step = (topology.rank + steps - topology.first_rank) % steps;

    for (uint32_t step = 0; step < steps; ++step) {
        auto& current = step == 0 ? initial : state;
        current.wait_front(state_tiles);
        a.wait_front(a_tiles);
        multiply<Kt, Vt>(a, current, product);
        current.pop_front(state_tiles);
        a.pop_front(a_tiles);
        // b is only needed by the add, so its reads may land while the matmul runs.
        b.wait_front(state_tiles);
        product.wait_front(state_tiles);
        // Publish the carry after step entry_step - 1 (this rank's entry state) and after the last step.
        const bool publish = step + 1 == entry_step || step + 1 == steps;
        add<state_tiles>(product, b, state, publish ? &out : nullptr);
        product.pop_front(state_tiles);
        b.pop_front(state_tiles);
    }
}
