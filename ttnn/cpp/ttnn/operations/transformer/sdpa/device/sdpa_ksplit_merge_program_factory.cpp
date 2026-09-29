// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_ksplit_merge_device_operation.hpp"

#include <bit>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

// Work item = (batch, head, 32-row tile), dealt round-robin over every worker core (item i on core i % P). On a core
// the items alternate between two data movement lanes (BRISC on NOC0, NCRISC on NOC1): both RISCs issue DRAM reads,
// a single reader tops out near 320 GB/s on these tile reads. Compute holds fp32 DEST in full sync (8 tiles).
// Lane l CBs: max c_(4l), sum c_(1+4l), O c_(2+4l), out c_(16+l); c_3 all-ones tile, c_24 row sums (S > 3),
// c_25 merge coefficients.
ProgramDescriptor SDPAKSplitMergeDeviceOperation::ProgramFactory::create_descriptor(
    const SDPAKSplitMergeParams& args, const SDPAKSplitMergeInputs& tensor_args, Tensor& output) {
    const auto& o = tensor_args.partial_output;
    const auto& st = tensor_args.partial_stats;
    const uint32_t S = args.k_split;
    const auto& os = o.padded_shape();
    const uint32_t B = os[0], NH = os[1] / S, St = os[2] / TILE_HEIGHT, DVt = os[3] / TILE_WIDTH;
    const uint32_t HALFt = st.padded_shape()[2] / (2 * TILE_HEIGHT);
    constexpr uint32_t DEPTH = 3;  // work items buffered per lane

    auto* device = o.device();
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t P = grid.x * grid.y;
    const CoreRangeSet all_cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    const uint32_t bf16_tile = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t fp32_tile = tt::tile_size(tt::DataFormat::Float32);

    ProgramDescriptor desc;
    auto cb = [&](uint32_t index, uint32_t tiles, tt::DataFormat fmt, uint32_t page) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = tiles * page,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index),
                .data_format = fmt,
                .page_size = page,
            }}},
        });
    };
    cb(tt::CBIndex::c_3, 1, tt::DataFormat::Float16_b, bf16_tile);
    cb(tt::CBIndex::c_25, S, tt::DataFormat::Float32, fp32_tile);
    if (S > 3) {
        cb(tt::CBIndex::c_24, S, tt::DataFormat::Float32, fp32_tile);
    }
    for (uint32_t lane = 0; lane < 2; ++lane) {
        cb(4 * lane, DEPTH * S, tt::DataFormat::Float16_b, bf16_tile);
        cb(1 + 4 * lane, DEPTH * S, tt::DataFormat::Float16_b, bf16_tile);
        cb(2 + 4 * lane, DEPTH * S * DVt, tt::DataFormat::Float16_b, bf16_tile);
        cb(16 + lane, DEPTH * DVt, tt::DataFormat::Float16_b, bf16_tile);
    }

    const uint32_t items = B * NH * St;
    for (uint32_t lane = 0; lane < 2; ++lane) {
        std::vector<uint32_t> ct = {S, NH, St, DVt, HALFt, P, lane, items};
        TensorAccessorArgs(*o.buffer()).append_to(ct);
        TensorAccessorArgs(*st.buffer()).append_to(ct);
        TensorAccessorArgs(*output.buffer()).append_to(ct);
        KernelDescriptor dm;
        dm.kernel_source = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ksplit_merge_dm.cpp";
        dm.source_type = KernelDescriptor::SourceType::FILE_PATH;
        dm.core_ranges = all_cores;
        dm.compile_time_args = std::move(ct);
        dm.config = DataMovementConfigDescriptor{
            .processor = lane ? DataMovementProcessor::RISCV_1 : DataMovementProcessor::RISCV_0,
            .noc = lane ? NOC::NOC_1 : NOC::NOC_0,
        };
        for (uint32_t me = 0; me < P; ++me) {
            dm.emplace_runtime_args(
                CoreCoord{me % grid.x, me / grid.x}, {o.buffer(), st.buffer(), output.buffer(), me});
        }
        desc.kernels.push_back(std::move(dm));
    }

    KernelDescriptor compute;
    compute.kernel_source = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/ksplit_merge.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = {S, DVt, std::bit_cast<uint32_t>(args.scale)};
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = true,
    };
    if (S > 3) {  // the row sums round-trip through c_24 in fp32: unpack them straight to DEST (no tf32 srcA)
        auto& modes = std::get<ComputeConfigDescriptor>(compute.config).unpack_to_dest_mode;
        modes.assign(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
        modes[tt::CBIndex::c_24] = UnpackToDestMode::UnpackToDestFp32;
    }
    for (uint32_t me = 0; me < P; ++me) {
        const uint32_t mine = me < items ? (items - me + P - 1) / P : 0;
        compute.emplace_runtime_args(CoreCoord{me % grid.x, me / grid.x}, {mine});
    }
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::prim
