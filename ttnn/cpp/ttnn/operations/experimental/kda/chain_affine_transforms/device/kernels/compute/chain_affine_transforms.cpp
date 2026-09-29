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

// product = A @ state. The whole key dimension accumulates in DST for each output row, as in a matmul whose
// in0 block spans K; each output tile therefore sums k = 0 .. Kt - 1 in order.
template <uint32_t Kt, uint32_t Vc>
FORCE_INLINE void multiply(DataflowBuffer& a, DataflowBuffer& state, DataflowBuffer& product) {
    const uint32_t a_id = a.get_id();
    const uint32_t state_id = state.get_id();
    const uint32_t product_id = product.get_id();
    product.reserve_back(Kt * Vc);
    reconfig_data_format(state_id, a_id);
    matmul_block_init(a_id, state_id, false, Vc, 1, Kt);
    for (uint32_t row = 0; row < Kt; ++row) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < Kt; ++k) {
            matmul_block(a_id, state_id, row * Kt + k, k * Vc, 0, false, Vc, 1, Kt);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t column = 0; column < Vc; ++column) {
            pack_tile(column, product_id, row * Vc + column);
        }
        tile_regs_release();
    }
    product.push_back(Kt * Vc);
}

// state = product + b as an FP32 SFPU add of DST operands, matching a separate FP32 elementwise add. The product
// unpacks losslessly to DST; b widens exactly from its transport format.
template <uint32_t Tiles>
FORCE_INLINE void add(DataflowBuffer& product, DataflowBuffer& b, DataflowBuffer& state, DataflowBuffer* out) {
    constexpr uint32_t pair_tiles = 2;
    const uint32_t product_id = product.get_id();
    const uint32_t b_id = b.get_id();
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
            add_binary_tile<ckernel::DstRoundingMode::NearestEven>(2 * i, 2 * i + 1, 2 * i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < count; ++i) {
            pack_tile(2 * i, state.get_id(), first + i);
            if (out != nullptr) {
                pack_tile(2 * i, out->get_id(), first + i);
            }
        }
        tile_regs_release();
    }
    state.push_back(Tiles);
    if (out != nullptr) {
        out->push_back(Tiles);
    }
}

template <uint32_t Kt, uint32_t Vc, uint32_t steps>
TT_KERNEL void compute() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::a, dfb::initial, dfb::product);
    constexpr uint32_t a_tiles = Kt * Kt;
    constexpr uint32_t state_tiles = Kt * Vc;
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
        DataflowBuffer& current = step == 0 ? initial : state;
        current.wait_front(state_tiles);
        a.wait_front(a_tiles);
        b.wait_front(state_tiles);
        multiply<Kt, Vc>(a, current, product);
        current.pop_front(state_tiles);
        a.pop_front(a_tiles);
        product.wait_front(state_tiles);
        const bool publish = step + 1 == entry_step || step + 1 == steps;
        add<state_tiles>(product, b, state, publish ? &out : nullptr);
        product.pop_front(state_tiles);
        b.pop_front(state_tiles);
    }
}
