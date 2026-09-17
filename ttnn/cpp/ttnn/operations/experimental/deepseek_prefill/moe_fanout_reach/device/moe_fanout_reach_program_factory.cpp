// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_fanout_reach_program_factory.hpp"

#include <algorithm>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/assert.hpp>

#include "kernels/dataflow/moe_fanout_reach_kernel_interface.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_fanout_reach {

namespace {

// The most cores this op splits a sequence over. Past this the per-core token range is short enough
// that the scan across the cores costs more than the walk it shortens, and the scan's round count --
// and with it the semaphore budget -- is bounded by the same number.
constexpr uint32_t kMaxCores = 1u << mfr::kMaxRounds;

// A balanced contiguous split. Contiguity and core order are load-bearing: the scan reproduces the
// sequential allocator only because core i's tokens are exactly those after every earlier core's.
uint32_t token_start(uint32_t core_idx, uint32_t seq_len, uint32_t num_cores) {
    return static_cast<uint32_t>((static_cast<uint64_t>(core_idx) * seq_len) / num_cores);
}

}  // namespace

tt::tt_metal::WorkloadDescriptor MoeFanoutReachProgramFactory::create_workload_descriptor(
    const MoeFanoutReachParams& args,
    const MoeFanoutReachInputs& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& /*tensor_coords*/) {
    auto* mesh = args.device;
    const uint32_t extent = static_cast<uint32_t>(mesh->shape()[static_cast<int32_t>(args.axis)]);
    const uint32_t hops = extent / 2u + 2u;
    const uint32_t seq_len = static_cast<uint32_t>(tensor_args.indices_tensor.logical_shape()[-2]);

    const auto all_cores = tt::tt_metal::corerange_to_cores(args.worker_core_range_set, std::nullopt, true);
    const uint32_t num_cores =
        std::min<uint32_t>({kMaxCores, static_cast<uint32_t>(all_cores.size()), std::max<uint32_t>(1u, seq_len)});
    std::vector<CoreCoord> cores(all_cores.begin(), all_cores.begin() + num_cores);

    uint32_t rounds = 0;
    while ((1u << rounds) < num_cores) {
        rounds++;
    }
    TT_FATAL(
        rounds <= mfr::kMaxRounds,
        "moe_fanout_reach: {} cores needs {} scan rounds but the runtime-argument block holds {}",
        num_cores,
        rounds,
        mfr::kMaxRounds);

    std::vector<CoreRange> core_ranges;
    core_ranges.reserve(cores.size());
    for (const auto& c : cores) {
        core_ranges.emplace_back(c);
    }
    const CoreRangeSet used_cores(core_ranges);

    auto* idx_buffer = tensor_args.indices_tensor.buffer();
    auto* table_buffer = tensor_args.expert_dispatch_table_tensor.buffer();
    auto* offsets_buffer = tensor_args.global_dispatch_offsets.buffer();
    auto* out_buffer = tensor_return_value.buffer();
    const uint32_t out_page_bytes = mfr::align64(static_cast<uint32_t>(out_buffer->aligned_page_size()));

    const mfr::Geometry geom{
        .num_routed_experts = args.num_routed_experts,
        .topk = args.num_experts_per_tok,
        // The longest range any core walks, since one circular-buffer configuration covers the grid.
        .tokens_per_core = (seq_len + num_cores - 1u) / num_cores,
        .rounds = rounds,
        .out_page_bytes = out_page_bytes};
    const uint32_t carve_bytes = mfr::carve_bytes(geom);

    std::vector<uint32_t> ct_args(mfr::CtArg::kCtCount);
    ct_args[mfr::CtArg::kNumRoutedExperts] = args.num_routed_experts;
    ct_args[mfr::CtArg::kTopk] = args.num_experts_per_tok;
    ct_args[mfr::CtArg::kExtent] = extent;
    ct_args[mfr::CtArg::kCapacity] = args.max_dispatch_buffer_token_size;
    ct_args[mfr::CtArg::kHops] = hops;
    ct_args[mfr::CtArg::kTokensPerCore] = geom.tokens_per_core;
    ct_args[mfr::CtArg::kRounds] = rounds;
    ct_args[mfr::CtArg::kOutPageBytes] = out_page_bytes;
    // The scan owns semaphore ids 0..rounds, one per round because a single counting semaphore cannot
    // say which producer arrived and they do not arrive in round order.
    ct_args[mfr::CtArg::kGatherSemId] = rounds + 1u;
    tt::tt_metal::TensorAccessorArgs(idx_buffer).append_to(ct_args);
    tt::tt_metal::TensorAccessorArgs(table_buffer).append_to(ct_args);
    tt::tt_metal::TensorAccessorArgs(offsets_buffer).append_to(ct_args);
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(ct_args);

    // Precomputed per core: the NoC coordinates of the tree's parent and children, and of each scan
    // round's source and the core it publishes to. One kernel covers the whole grid, so these cannot
    // be compile-time args.
    const auto noc_of = [&](uint32_t i) { return mesh->worker_core_from_logical_core(cores[i]); };

    tt::tt_metal::WorkloadDescriptor workload;
    for (const auto& coord : ttnn::MeshCoordinateRange(mesh->shape())) {
        tt::tt_metal::ProgramDescriptor desc;

        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = carve_bytes,
            .core_ranges = used_cores,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(mfr::kCbCarve),
                .data_format = tt::DataFormat::UInt32,
                .page_size = carve_bytes,
            }}},
        });

        for (uint32_t s = 0; s <= rounds + 1u; s++) {
            desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
                .id = s, .core_type = tt::CoreType::WORKER, .core_ranges = used_cores, .initial_value = 0});
        }

        tt::tt_metal::KernelDescriptor kernel;
        kernel.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fanout_reach/device/kernels/dataflow/"
            "moe_fanout_reach.cpp";
        kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        kernel.core_ranges = used_cores;
        kernel.compile_time_args = ct_args;
        // Every hop in this row is measured from this chip's position on the ring, so the program
        // cannot be replicated across the mesh.
        kernel.compile_time_args[mfr::CtArg::kMyRow] = static_cast<uint32_t>(coord[static_cast<int32_t>(args.axis)]);
        kernel.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
        };

        for (uint32_t i = 0; i < num_cores; i++) {
            std::vector<uint32_t> rt(mfr::kRtCount, mfr::kNoCore);
            rt[mfr::RtArg::kTokStart] = token_start(i, seq_len, num_cores);
            rt[mfr::RtArg::kTokCount] = token_start(i + 1u, seq_len, num_cores) - rt[mfr::RtArg::kTokStart];

            // The same tree masked_bincount reduces over: core i takes children at i + 2^L for as many
            // levels as its index has trailing zeros, and reports to the core that clears its lowest
            // set bit.
            uint32_t num_children = 0;
            for (uint32_t level = 0; (1u << level) < num_cores; level++) {
                const uint32_t stride = 1u << level;
                if ((i % (stride << 1u)) != 0u || (i + stride) >= num_cores) {
                    break;
                }
                const auto child = noc_of(i + stride);
                rt[mfr::RtArg::kChildrenBase + num_children * 2u] = static_cast<uint32_t>(child.x);
                rt[mfr::RtArg::kChildrenBase + num_children * 2u + 1u] = static_cast<uint32_t>(child.y);
                num_children++;
            }
            rt[mfr::RtArg::kNumChildren] = num_children;
            if (i > 0) {
                const auto parent = noc_of(i ^ (i & (~i + 1u)));
                rt[mfr::RtArg::kParentNocX] = static_cast<uint32_t>(parent.x);
                rt[mfr::RtArg::kParentNocY] = static_cast<uint32_t>(parent.y);
            }

            // Round r reads core i - 2^r and publishes to core i + 2^r. The extra round past the last
            // shifts the inclusive totals by one core, which is what turns them into the exclusive
            // prefix the allocator needs, so its stride is 1.
            for (uint32_t r = 0; r <= rounds; r++) {
                const uint32_t stride = (r < rounds) ? (1u << r) : 1u;
                const uint32_t slot = mfr::kScanBase + r * mfr::kScanWordsPerRound;
                if (i >= stride) {
                    const auto src = noc_of(i - stride);
                    rt[slot] = static_cast<uint32_t>(src.x);
                    rt[slot + 1u] = static_cast<uint32_t>(src.y);
                }
                if (i + stride < num_cores) {
                    const auto dst = noc_of(i + stride);
                    rt[slot + 2u] = static_cast<uint32_t>(dst.x);
                    rt[slot + 3u] = static_cast<uint32_t>(dst.y);
                }
            }

            // Buffer* rather than an address: an address describes an allocation, not a program, and a
            // cached program must not carry a stale one.
            tt::tt_metal::KernelDescriptor::RTArgList rt_list;
            rt_list.push_back(idx_buffer);
            rt_list.push_back(table_buffer);
            rt_list.push_back(offsets_buffer);
            rt_list.push_back(out_buffer);
            for (uint32_t w = mfr::RtArg::kTokStart; w < mfr::kRtCount; w++) {
                rt_list.push_back(rt[w]);
            }
            kernel.emplace_runtime_args(cores[i], rt_list);
        }

        desc.kernels.push_back(std::move(kernel));
        workload.programs.push_back({ttnn::MeshCoordinateRange(coord, coord), std::move(desc)});
    }
    return workload;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fanout_reach
