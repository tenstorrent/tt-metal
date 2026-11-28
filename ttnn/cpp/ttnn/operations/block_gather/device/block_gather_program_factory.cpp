// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "block_gather_program_factory.hpp"

#include <array>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "kernels/block_gather_args.hpp"

namespace ttnn::operations::block_gather {

using namespace tt::tt_metal;
using namespace ::block_gather;

namespace {

constexpr const char* kBlockGatherKernel = "ttnn/cpp/ttnn/operations/block_gather/device/kernels/block_gather.cpp";

// The kernel's reads are aligned to this many bytes (Blackhole's DRAM read alignment), and so is every scratch region.
constexpr uint32_t kGatherAlignBytes = 64;
constexpr uint32_t kIdBytes = sizeof(uint32_t);
constexpr uint32_t kScoreBytes = 2;  // bf16

uint32_t gather_rows(const Tensor& scores) {
    const auto& shape = scores.logical_shape();
    uint32_t rows = 1;
    for (uint32_t i = 0; i + 1 < shape.rank(); ++i) {
        rows *= shape[i];
    }
    return rows;
}

}  // namespace

uint32_t gather_parts(uint32_t k) {
    constexpr uint32_t kIdsPerAlignedRun = kGatherAlignBytes / kIdBytes;
    for (const uint32_t p : {kGatherParts, 2u}) {
        if (k % p == 0 && (k / p) % kIdsPerAlignedRun == 0) {
            return p;
        }
    }
    return 1;
}

uint32_t gather_scratch_bytes(uint32_t k, uint32_t block) {
    const uint32_t kp = k / gather_parts(k);
    const uint32_t blk_bytes = kScoreBytes * block;
    const uint32_t span = tt::round_up(kIdBytes * kp, kGatherAlignBytes) +
                          tt::round_up(blk_bytes * kp, kGatherAlignBytes) +
                          (blk_bytes == kGatherAlignBytes ? 0 : kGatherAlignBytes * kp);
    return kGatherAlignBytes + (2 * span);
}

ProgramDescriptor BlockGatherProgramFactory::create_descriptor(
    const BlockGatherParams& attrs, const BlockGatherInputs& t, Tensor& output) {
    TT_FATAL(
        cb::SCRATCH < hal::get_num_dataflow_buffers(),
        "block_gather: circular-buffer id {} exceeds the {} circular buffers of this architecture",
        cb::SCRATCH,
        hal::get_num_dataflow_buffers());
    const uint32_t width = t.scores.logical_shape()[-1];
    const uint32_t k = t.ids.logical_shape()[-1];
    const uint32_t block = attrs.block;
    TT_FATAL(
        block > 0 && width % block == 0, "block_gather: width {} must be a multiple of the block {}", width, block);
    const uint32_t parts = gather_parts(k);
    const uint32_t scratch = gather_scratch_bytes(k, block);
    const uint32_t rows = gather_rows(t.scores);

    const CoreRangeSet cores(CoreRange(CoreCoord(0, 0), CoreCoord(attrs.grid_x - 1, attrs.grid_y - 1)));

    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = scratch,
        .core_ranges = cores,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb::SCRATCH),
            .data_format = tt::DataFormat::UInt8,
            .page_size = scratch,
        }},
    });

    std::vector<uint32_t> ct(ct_arg::COUNT);
    ct[ct_arg::CB] = cb::SCRATCH;
    ct[ct_arg::BLOCK] = block;
    ct[ct_arg::K] = k;
    ct[ct_arg::NBW] = width / block;
    ct[ct_arg::GX] = attrs.grid_x;
    ct[ct_arg::PARTS] = parts;
    ct[ct_arg::NCORES] = attrs.grid_x * attrs.grid_y;
    TensorAccessorArgs(*t.scores.buffer()).append_to(ct);
    TensorAccessorArgs(*t.ids.buffer()).append_to(ct);
    TensorAccessorArgs(*output.buffer()).append_to(ct);

    for (uint32_t half = 0; half < kernel_index::COUNT; ++half) {
        KernelDescriptor kernel;
        kernel.kernel_source = kBlockGatherKernel;
        kernel.core_ranges = cores;
        kernel.compile_time_args = ct;
        kernel.compile_time_args[ct_arg::HALF] = half;
        static_assert(
            common_arg::SCORES_ADDR == 0 && common_arg::IDS_ADDR == 1 && common_arg::OUT_ADDR == 2 &&
            common_arg::ROWS == 3 && common_arg::COUNT == 4);
        KernelDescriptor::RTArgList common;
        common.push_back(t.scores.buffer());
        common.push_back(t.ids.buffer());
        common.push_back(output.buffer());
        common.push_back(rows);
        kernel.emplace_common_runtime_args(common);
        TT_FATAL(
            kernel.common_runtime_args.size() == common_arg::COUNT,
            "block_gather: common runtime argument layout is out of sync with block_gather_args.hpp");
        if (half == kernel_index::HALF0) {
            kernel.config = ReaderConfigDescriptor{};
        } else {
            kernel.config = WriterConfigDescriptor{};
        }
        desc.kernels.push_back(std::move(kernel));
    }
    return desc;
}

void BlockGatherProgramFactory::override_runtime_arguments(
    Program& program,
    const BlockGatherParams& /*attrs*/,
    const BlockGatherInputs& t,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // The key pins ROWS (scores' spec) and every compile-time arg; only the three addresses vary per call.
    const uint32_t scores_addr = t.scores.buffer()->address();
    const uint32_t ids_addr = t.ids.buffer()->address();
    const uint32_t out_addr = output.buffer()->address();
    for (const uint32_t kernel : {kernel_index::HALF0, kernel_index::HALF1}) {
        auto& common = GetCommonRuntimeArgs(program, kernel);
        common[common_arg::SCORES_ADDR] = scores_addr;
        common[common_arg::IDS_ADDR] = ids_addr;
        common[common_arg::OUT_ADDR] = out_addr;
    }
}

}  // namespace ttnn::operations::block_gather
