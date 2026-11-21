// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b/argmax_decode/device/argmax_decode_device_operation.hpp"
#include "ttnn/operations/experimental/llama31_8b/argmax_decode/device/argmax_decode_table.hpp"

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::experimental::llama31_8b {

using namespace tt::tt_metal;

namespace {

constexpr uint32_t kRowsPerHalf = 16;
constexpr uint32_t kSlotBytes = kRowsPerHalf * 2 * sizeof(uint32_t);
constexpr uint32_t kBf16TileBytes = 32 * 32 * 2;
constexpr uint32_t kFp32TileBytes = 32 * 32 * 4;

}  // namespace

ProgramDescriptor ArgmaxDecodeProgramFactory::create_descriptor(
    const ArgmaxDecodeParams&, const ArgmaxDecodeInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    const auto* shape = argmax_decode_shape(input);
    TT_FATAL(shape != nullptr, "llama31_8b.argmax_decode: input shape is not in the table");
    const auto& shard = input.memory_config().shard_spec().value();
    const uint32_t workers_n = shape->workers;

    const auto cores = corerange_to_cores(shard.grid, workers_n, true);
    std::set<CoreRange> ranges;
    for (const auto& c : cores) {
        ranges.insert(CoreRange(c, c));
    }
    const CoreRangeSet workers(ranges);

    auto* device = input.device();
    const CoreCoord reduce_core = device->worker_core_from_logical_core(cores.front());

    const uint32_t merge_cb = tt::CBIndex::c_0;
    const uint32_t cb_in = tt::CBIndex::c_1;
    const uint32_t cb_keys = tt::CBIndex::c_2;
    const uint32_t merge_bytes = 2 * workers_n * kSlotBytes + 512;

    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = merge_bytes,
        .core_ranges = workers,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(merge_cb),
            .data_format = tt::DataFormat::UInt32,
            .page_size = merge_bytes,
        }}},
    });
    CBDescriptor in_cb{
        .total_size = (shape->shard_cols / tt::constants::TILE_WIDTH) * kBf16TileBytes,
        .core_ranges = workers,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_in),
            .data_format = tt::DataFormat::UInt16,
            .page_size = kBf16TileBytes,
        }}},
    };
    in_cb.buffer = input.buffer();
    desc.cbs.push_back(std::move(in_cb));
    desc.cbs.push_back(CBDescriptor{
        .total_size = kFp32TileBytes,
        .core_ranges = workers,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_keys),
            .data_format = tt::DataFormat::Float32,
            .page_size = kFp32TileBytes,
        }}},
    });
    const uint32_t sem_ids[2] = {0, 1};
    for (uint32_t id : sem_ids) {
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = workers,
            .initial_value = 0,
        });
    }

    KernelDescriptor c;
    c.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/llama31_8b/argmax_decode/device/kernels/argmax_decode_compute.cpp";
    c.source_type = KernelDescriptor::SourceType::FILE_PATH;
    c.core_ranges = workers;
    c.compile_time_args = {cb_in, cb_keys};
    c.config = ComputeConfigDescriptor{.fp32_dest_acc_en = true};
    for (uint32_t i = 0; i < workers_n; ++i) {
        KernelDescriptor::RTArgList args;
        args.push_back(shape->cores[i].tiles);
        c.emplace_runtime_args(cores[i], args);
    }
    desc.kernels.push_back(std::move(c));

    const uint32_t out_page_bytes = output.logical_shape()[-1] * sizeof(uint32_t);
    for (uint32_t half = 0; half < 2; ++half) {
        std::vector<uint32_t> ct_args = {
            half,
            workers_n,
            shape->group,
            shape->num_groups,
            merge_cb,
            sem_ids[half],
            static_cast<uint32_t>(reduce_core.x),
            static_cast<uint32_t>(reduce_core.y),
            out_page_bytes,
            cb_in,
            cb_keys,
        };
        TensorAccessorArgs(output.buffer()).append_to(ct_args);

        KernelDescriptor k;
        k.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b/argmax_decode/device/kernels/argmax_decode_scan.cpp";
        k.source_type = KernelDescriptor::SourceType::FILE_PATH;
        k.core_ranges = workers;
        k.compile_time_args = std::move(ct_args);
        k.config = DataMovementConfigDescriptor{
            .processor = half == 0 ? DataMovementProcessor::RISCV_0 : DataMovementProcessor::RISCV_1,
            .noc = half == 0 ? NOC::RISCV_0_default : NOC::RISCV_1_default,
        };
        for (uint32_t i = 0; i < workers_n; ++i) {
            const auto& row = shape->cores[i];
            const CoreCoord lead = device->worker_core_from_logical_core(cores[row.lead]);
            KernelDescriptor::RTArgList args;
            args.push_back(output.buffer());
            args.push_back(i);
            args.push_back(i * shape->shard_cols);
            args.push_back(row.tiles);
            args.push_back(row.members);
            args.push_back(row.wait);
            args.push_back(static_cast<uint32_t>(lead.x));
            args.push_back(static_cast<uint32_t>(lead.y));
            k.emplace_runtime_args(cores[i], args);
        }
        desc.kernels.push_back(std::move(k));
    }
    return desc;
}

}  // namespace ttnn::experimental::llama31_8b
