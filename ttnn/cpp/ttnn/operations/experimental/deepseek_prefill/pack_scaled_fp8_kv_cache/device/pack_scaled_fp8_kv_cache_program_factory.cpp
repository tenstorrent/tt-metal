// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "pack_scaled_fp8_kv_cache_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"

namespace ttnn::experimental::prim::pack_scaled_fp8_kv_cache {

PackScaledFp8KvCacheProgramFactory::cached_program_t PackScaledFp8KvCacheProgramFactory::create(
    const PackScaledFp8KvCacheParams&, const PackScaledFp8KvCacheInputs& args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    auto* latent_buffer = args.latent.buffer();
    auto* scale_buffer = args.scales.buffer();
    const bool has_rope = args.rope.has_value();
    auto* output_buffer = output.buffer();
    const uint32_t latent_bytes = args.latent.logical_shape()[-1];  // one FP8 byte per value
    const uint32_t scale_bytes = ::sparse_sdpa::scaled_kv_scale_bytes(latent_bytes);
    const uint32_t rope_bytes = has_rope ? args.rope->logical_shape()[-1] * sizeof(uint16_t) : 0u;
    const uint32_t rows = args.latent.logical_volume() / latent_bytes;

    Program program;
    const auto grid = args.latent.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, group_1, group_2, rows_group_1, rows_group_2] = split_work_to_cores(grid, rows);
    auto cores = corerange_to_cores(all_cores, num_cores, true);

    constexpr uint32_t cb_scratch = CBIndex::c_0;
    const uint32_t scratch_bytes = std::max({latent_bytes, scale_bytes, rope_bytes});
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(scratch_bytes, {{cb_scratch, DataFormat::UInt8}})
            .set_page_size(cb_scratch, scratch_bytes));

    // rope_bytes == 0 compiles the RoPE copy out; the RoPE accessor chains last (a placeholder when absent).
    std::vector<uint32_t> compile_args = {cb_scratch, latent_bytes, scale_bytes, rope_bytes};
    TensorAccessorArgs(latent_buffer).append_to(compile_args);
    TensorAccessorArgs(scale_buffer).append_to(compile_args);
    TensorAccessorArgs(output_buffer).append_to(compile_args);
    TensorAccessorArgs(has_rope ? args.rope->buffer() : nullptr).append_to(compile_args);

    const auto kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/pack_scaled_fp8_kv_cache/device/kernels/"
        "dataflow/pack_scaled_fp8_kv_cache.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = compile_args});

    uint32_t start_row = 0;
    for (const auto& core : cores) {
        const uint32_t core_rows = group_1.contains(core) ? rows_group_1 : rows_group_2;
        SetRuntimeArgs(
            program,
            kernel_id,
            core,
            {latent_buffer->address(),
             scale_buffer->address(),
             has_rope ? args.rope->buffer()->address() : 0u,
             output_buffer->address(),
             start_row,
             core_rows});
        start_row += core_rows;
    }

    return cached_program_t{std::move(program), {kernel_id, std::move(cores)}};
}

void PackScaledFp8KvCacheProgramFactory::override_runtime_arguments(
    cached_program_t& cached,
    const PackScaledFp8KvCacheParams&,
    const PackScaledFp8KvCacheInputs& args,
    Tensor& output) {
    for (const auto& core : cached.shared_variables.cores) {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(cached.program, cached.shared_variables.kernel_id, core);
        runtime_args[0] = args.latent.buffer()->address();
        runtime_args[1] = args.scales.buffer()->address();
        runtime_args[2] = args.rope.has_value() ? args.rope->buffer()->address() : 0u;
        runtime_args[3] = output.buffer()->address();
    }
}

}  // namespace ttnn::experimental::prim::pack_scaled_fp8_kv_cache
