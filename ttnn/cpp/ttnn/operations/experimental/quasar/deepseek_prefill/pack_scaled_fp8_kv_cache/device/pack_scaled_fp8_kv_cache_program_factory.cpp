// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "pack_scaled_fp8_kv_cache_device_operation.hpp"

#include <tt-metalium/data_formats/tile_constants.hpp>
#include <tt-metalium/workload/legacy/host_api.hpp>
#include <tt-metalium/workload/legacy/program_descriptors.hpp>
#include <tt-metalium/workload/legacy/tensor_accessor_args.hpp>
#include <tt-metalium/utils/work_split.hpp>

#include "ttnn/operations/experimental/quasar/deepseek_prefill/pack_scaled_fp8_kv_cache/pack_scaled_fp8_kv_cache.hpp"

namespace ttnn::experimental::prim::qsr::pack_scaled_fp8_kv_cache {

namespace packed = ttnn::operations::experimental::quasar::deepseek_prefill::pack_scaled_fp8_kv_cache;

// Only the four buffer addresses vary between two calls sharing a program-cache key (the default hash keys the whole
// spec of every input, so rows and the work split are fixed by a hit). They are declared as Buffer* bindings, which
// the framework patches on cache hits; no override_runtime_arguments is needed.
tt::tt_metal::ProgramDescriptor PackScaledFp8KvCacheDeviceOperation::create_descriptor(
    const PackScaledFp8KvCacheParams&, const PackScaledFp8KvCacheInputs& args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    Buffer* latent_buffer = args.latent.buffer();
    Buffer* scale_buffer = args.scales.buffer();
    Buffer* rope_buffer = args.rope.buffer();
    Buffer* output_buffer = output.buffer();
    const uint32_t rows = args.latent.logical_volume() / packed::LATENT_WIDTH;

    const auto grid = args.latent.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, group_1, group_2, rows_group_1, rows_group_2] = split_work_to_cores(grid, rows);
    auto cores = corerange_to_cores(all_cores, num_cores, true);

    ProgramDescriptor desc;

    constexpr uint32_t cb_scratch = CBIndex::c_0;
    constexpr uint32_t scratch_bytes = packed::LATENT_WIDTH;
    desc.cbs.push_back(CBDescriptor{
        .total_size = scratch_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_scratch),
            .data_format = DataFormat::UInt8,
            .page_size = scratch_bytes,
        }}},
    });

    std::vector<uint32_t> compile_args = {
        cb_scratch, packed::LATENT_WIDTH, packed::SCALE_WIDTH * sizeof(float), packed::ROPE_WIDTH * sizeof(uint16_t)};
    TensorAccessorArgs(latent_buffer).append_to(compile_args);
    TensorAccessorArgs(scale_buffer).append_to(compile_args);
    TensorAccessorArgs(rope_buffer).append_to(compile_args);
    TensorAccessorArgs(output_buffer).append_to(compile_args);

    KernelDescriptor kernel;
    kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/quasar/deepseek_prefill/pack_scaled_fp8_kv_cache/device/kernels/"
        "dataflow/pack_scaled_fp8_kv_cache.cpp";
    kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = all_cores;
    kernel.compile_time_args = std::move(compile_args);
    kernel.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
    };

    uint32_t start_row = 0;
    for (const auto& core : cores) {
        const uint32_t core_rows = group_1.contains(core) ? rows_group_1 : rows_group_2;
        kernel.emplace_runtime_args(
            core, {latent_buffer, scale_buffer, rope_buffer, output_buffer, start_row, core_rows});
        start_row += core_rows;
    }

    desc.kernels.push_back(std::move(kernel));
    return desc;
}

}  // namespace ttnn::experimental::prim::qsr::pack_scaled_fp8_kv_cache
