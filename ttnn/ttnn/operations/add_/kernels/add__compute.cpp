// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// add_ compute (block operation `add_relu_block`, regime `sharded_resident`).
//
// NEO t of C takes this core's tiles t, t+C, ... (STRIDED rings) and computes a + b on the FPU, with the
// ReLU applied by the packer (PackRelu::Zero) when `pack_relu`: the value stored is the sum rounded once
// to the operand dtype, then clamped. All three rings are borrowed from the caller's shards; cb_result_shard
// is the same bytes as cb_a_shard, so each result tile overwrites the operand tile this NEO has just
// unpacked (unpack i -> math i -> pack i on the same address; no other NEO touches it).
//
// One helper call per NEO over its whole share: `neo_block_tiles` is the IterationShape block size, and
// wait/pop/reserve/push run once per block (PerBlockSize; the chain clamps the block to DEST capacity).
//
// `tag_threads` is a test probe of the internal program builder: odd NEOs subtract instead of add, so a
// test can see which NEO computed which tile (a correct result alone does not show that C NEOs ran).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const uint32_t core_tiles = get_arg(args::core_tiles);
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t neo_block_tiles = get_arg(args::neo_block_tiles);
    constexpr uint32_t pack_relu = get_arg(args::pack_relu);
    constexpr uint32_t tag_threads = get_arg(args::tag_threads);
    const uint32_t t = get_my_thread_id();
    const uint32_t my_tiles = core_tiles / C + (t < core_tiles % C ? 1u : 0u);

    DataflowBuffer dfb_a(dfb::cb_a_shard);
    DataflowBuffer dfb_b(dfb::cb_b_shard);
    DataflowBuffer dfb_result(dfb::cb_result_shard);
    compute_kernel_hw_startup(dfb::cb_a_shard, dfb::cb_b_shard, dfb::cb_result_shard);

    if (my_tiles == 0) {
        return;  // only trailing NEOs of a core with fewer tiles than NEOs; the DM kernels skip them too
    }

    constexpr auto in_a = ckl::input(
        dfb::cb_a_shard, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block);
    constexpr auto in_b = ckl::input(
        dfb::cb_b_shard, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block);
    constexpr auto out = ckl::output(
        dfb::cb_result_shard,
        ckl::ReservePolicy::PerBlockSize,
        ckl::PushPolicy::PerBlockSize,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Direct,
        ckl::DestAccumulation::Disabled,
        ckl::L1Accumulation::Disabled,
        pack_relu ? ckl::PackRelu::Zero : ckl::PackRelu::Disabled);
    const auto shape = ckl::IterationShape::tiles(my_tiles).block_size(neo_block_tiles);

    if constexpr (tag_threads) {
        if (t & 1u) {
            ckl::sub<in_a, in_b, out>(shape, dfb_a, dfb_b, dfb_result);
            return;
        }
    }
    ckl::add<in_a, in_b, out>(shape, dfb_a, dfb_b, dfb_result);
}
