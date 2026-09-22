// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_recipe.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <set>

#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/generic/generic_op.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::transformer::sdpa::detail {
using namespace tt::tt_metal;

RecipeSelection select_recipe(ttnn::transformer::SDPAPrecision precision, DataType kv_type) {
    using ttnn::transformer::SDPAPrecision;
    Recipe recipe;
    switch (precision) {
        case SDPAPrecision::FAST: recipe = Recipe::A; break;
        case SDPAPrecision::COMPENSATED: recipe = Recipe::B; break;
        case SDPAPrecision::BALANCED: recipe = Recipe::C; break;
        case SDPAPrecision::ACCURATE: recipe = Recipe::D; break;
        case SDPAPrecision::LOW_PRECISION: recipe = Recipe::E; break;
        default: TT_THROW("Unknown SDPA precision recipe");
    }
    const auto storage = kv_type == DataType::BFLOAT4_B   ? KVStorage::BFP4
                         : kv_type == DataType::BFLOAT8_B ? KVStorage::BFP8
                                                          : KVStorage::BF16;
    return {recipe, storage};
}

ProgramDescriptor recipe_compute_program(const PrecisionPolicy& policy, const CoreRangeSet& grid, uint32_t k_chunks) {
    const bool fp32 = policy.fp32_destination;
    const bool compensated = policy.recurrent_state == RecurrentState::CompensatedBF16;
    const uint32_t stride = compensated ? 2 : 1;
    const auto state_format = fp32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t state_bytes = fp32 ? 4096 : 2048;
    const auto kv_type = policy.selection.kv_storage == KVStorage::BF16   ? DataType::BFLOAT16
                         : policy.selection.kv_storage == KVStorage::BFP8 ? DataType::BFLOAT8_B
                                                                          : DataType::BFLOAT4_B;
    const auto kv_format = datatype_to_dataformat_converter(kv_type);
    const uint32_t kv_bytes = kv_type == DataType::BFLOAT16 ? 2048 : kv_type == DataType::BFLOAT8_B ? 1088 : 576;
    ProgramDescriptor program;
    auto add_cb = [&](uint8_t index, uint32_t count, uint32_t page, tt::DataFormat format) {
        CBDescriptor cb{
            .total_size = count * page,
            .core_ranges = grid,
            .format_descriptors = {{.buffer_index = index, .data_format = format, .page_size = page}}};
        if (index == 6 && fp32) {
            cb.format_descriptors.push_back({.buffer_index = 7, .data_format = format, .page_size = page});
        }
        program.cbs.push_back(std::move(cb));
    };
    // Preserve the selected families' actual buffer depths: Q double buffered;
    // K/V one slot for FP32, two slots for BF16. No hidden geometry retuning.
    add_cb(0, 64, 2048, tt::DataFormat::Float16_b);
    add_cb(1, fp32 ? 64 : 128, kv_bytes, kv_format);
    add_cb(2, fp32 ? 64 : 128, kv_bytes, kv_format);
    add_cb(3, 1, 2048, tt::DataFormat::Float16_b);
    add_cb(4, 1, 2048, tt::DataFormat::Float16_b);
    add_cb(5, 1, state_bytes, state_format);
    add_cb(6, 128, state_bytes, state_format);
    for (uint8_t index : {8, 9}) {
        add_cb(index, 32 * stride, state_bytes, state_format);
    }
    for (uint8_t index : {10, 11}) {
        add_cb(index, 8, 2048, tt::DataFormat::Float16_b);
    }
    for (uint8_t index : {12, 13}) {
        add_cb(index, 8 * stride, state_bytes, state_format);
    }
    add_cb(14, 8, state_bytes, state_format);
    add_cb(16, fp32 ? 8 : 16, 2048, tt::DataFormat::Float16_b);
    ComputeConfigDescriptor compute_config{
        .math_fidelity = policy.pv_fidelity,
        .fp32_dest_acc_en = fp32,
        .dst_full_sync_en = false,
        .math_approx_mode = true};
    if (fp32) {
        compute_config.unpack_to_dest_mode.resize(64, UnpackToDestMode::Default);
        for (uint32_t cb : {5, 7, 8, 9, 12, 13, 14}) {
            compute_config.unpack_to_dest_mode[cb] = UnpackToDestMode::UnpackToDestFp32;
        }
    }
    KernelDescriptor compute{
        .kernel_source = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa_recipe.cpp",
        .core_ranges = grid,
        .compile_time_args = {k_chunks, std::bit_cast<uint32_t>(1.0f / std::sqrt(128.0f)), 8},
        .defines =
            {{"EXP_APPROX_MODE", "1"},
             {"STATS_GRANULARITY", fp32 ? "4" : "8"},
             {"SUB_EXP_GRANULARITY", fp32 ? "4" : "8"},
             {"MUL_BCAST_GRANULARITY", fp32 ? "4" : "8"},
             {"DHT_GRANULARITY", "4"},
             {"REDUCE_GRANULARITY", fp32 ? "2" : "4"}},
        .config = compute_config};
    if (fp32) {
        compute.defines.emplace_back("SDPA_RECIPE_FP32", "1");
    }
    if (policy.selection.recipe == Recipe::D) {
        compute.defines.emplace_back("SDPA_RECIPE_ACCURATE", "1");
    }
    if (policy.selection.recipe == Recipe::E) {
        compute.defines.emplace_back("SDPA_RECIPE_LOFI", "1");
    }
    if (policy.selection.recipe == Recipe::A) {
        compute.defines.emplace_back("SDPA_RECIPE_BASELINE", "1");
    }
    program.kernels.push_back(std::move(compute));
    return program;
}

Tensor run_recipe(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const PrecisionPolicy& policy,
    const std::optional<SDPAProgramConfig>& program_config) {
    for (const Tensor* tensor : {&q, &k, &v}) {
        TT_FATAL(tensor->storage_type() == StorageType::DEVICE, "SDPA recipes require device inputs");
        TT_FATAL(tensor->device() == q.device(), "SDPA recipe inputs must belong to the same device");
        TT_FATAL(tensor->device()->arch() == tt::ARCH::BLACKHOLE, "SDPA recipes currently support Blackhole only");
        TT_FATAL(tensor->device()->num_devices() == 1, "SDPA recipes currently require a single-device mesh");
        TT_FATAL(tensor->layout() == Layout::TILE, "SDPA recipes require tiled inputs");
        TT_FATAL(tensor->memory_config() == DRAM_MEMORY_CONFIG, "SDPA recipes require interleaved DRAM inputs");
        TT_FATAL(tensor->logical_shape() == tensor->padded_shape(), "SDPA recipes do not support padded inputs yet");
        TT_FATAL(tensor->logical_shape().rank() == 4, "SDPA recipes require rank-four inputs");
        TT_FATAL(tensor->tensor_spec().tile() == Tile({32, 32}), "SDPA recipes require standard 32x32 tiles");
    }
    const auto& qs = q.logical_shape();
    const auto& ks = k.logical_shape();
    TT_FATAL(qs[0] == 1 && qs[1] > 0 && qs[3] == 128, "SDPA recipes require Q [1,H,Q,128]");
    TT_FATAL(
        ks == v.logical_shape() && ks[0] == 1 && ks[1] == qs[1] && ks[3] == 128,
        "SDPA recipes require matching K/V [1,H,K,128]; GQA is not supported yet");
    TT_FATAL(
        qs[2] > 0 && qs[2] % 256 == 0 && ks[2] > 0 && ks[2] % 512 == 0,
        "SDPA recipes require Q and K lengths divisible by 256 and 512, respectively");
    const DataType kv_type = policy.selection.kv_storage == KVStorage::BF16   ? DataType::BFLOAT16
                             : policy.selection.kv_storage == KVStorage::BFP8 ? DataType::BFLOAT8_B
                                                                              : DataType::BFLOAT4_B;
    TT_FATAL(
        q.dtype() == DataType::BFLOAT16 && k.dtype() == kv_type && v.dtype() == kv_type,
        "SDPA input types do not match the selected recipe");
    const auto hardware = q.device()->compute_with_storage_grid_size();
    const auto grid_size = program_config ? program_config->compute_with_storage_grid_size : hardware;
    TT_FATAL(
        grid_size.x > 0 && grid_size.y > 0 && grid_size.x <= hardware.x && grid_size.y <= hardware.y,
        "SDPA recipe compute grid must fit the device");
    if (program_config) {
        TT_FATAL(
            program_config->q_chunk_size == 256 && program_config->k_chunk_size == 512,
            "Named SDPA recipes currently require Q256/K512 blocking");
        TT_FATAL(!program_config->sub_core_grids.has_value(), "SDPA recipes do not yet support sub_core_grids");
        TT_FATAL(program_config->max_cores_per_head_batch > 0, "SDPA max_cores_per_head_batch must be positive");
    }
    const uint32_t jobs_per_head = qs[2] / 256;
    const uint32_t k_chunks = ks[2] / 512;
    const uint32_t chain = std::min<uint32_t>(
        {jobs_per_head,
         static_cast<uint32_t>(grid_size.x * grid_size.y / qs[1]),
         program_config ? program_config->max_cores_per_head_batch : 16u});
    TT_FATAL(chain > 0, "SDPA recipes require at least one compute core per head");
    const uint32_t cores = chain * qs[1];
    std::vector<CoreCoord> coordinates;
    std::set<CoreRange> ranges;
    for (uint32_t i = 0; i < cores; ++i) {
        const CoreCoord core(i % grid_size.x, i / grid_size.x);
        coordinates.push_back(core);
        ranges.emplace(core, core);
    }
    const CoreRangeSet grid(ranges);
    auto output = create_device_tensor(q.tensor_spec(), q.device());
    auto program = recipe_compute_program(policy, grid, k_chunks);
    for (uint32_t i = 0; i < 3; ++i) {
        program.semaphores.push_back({.id = i, .core_ranges = grid, .initial_value = i == 2 ? 1u : 0u});
    }
    const std::string prefix = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/";
    KernelDescriptor reader{
        .kernel_source = prefix + "dataflow/reader_recipe.cpp",
        .core_ranges = grid,
        .compile_time_args = {8, k_chunks, jobs_per_head},
        .config = ReaderConfigDescriptor{}};
    for (const Tensor* tensor : {&q, &k, &v}) {
        TensorAccessorArgs(tensor->buffer()).append_to(reader.compile_time_args);
    }
    KernelDescriptor writer{
        .kernel_source = prefix + "dataflow/writer_recipe.cpp",
        .core_ranges = grid,
        .compile_time_args = {8},
        .config = WriterConfigDescriptor{}};
    TensorAccessorArgs(output.buffer()).append_to(writer.compile_time_args);
    auto compute = std::move(program.kernels.front());
    for (uint32_t i = 0; i < cores; ++i) {
        const auto core = coordinates[i];
        const uint32_t head = i / chain, rank = i % chain;
        const uint32_t count = jobs_per_head / chain + (rank < jobs_per_head % chain);
        const uint32_t offset =
            head * jobs_per_head + rank * (jobs_per_head / chain) + std::min(rank, jobs_per_head % chain);
        const auto prev = rank ? q.device()->worker_core_from_logical_core(coordinates[i - 1]) : CoreCoord(0, 0);
        const auto next =
            rank + 1 < chain ? q.device()->worker_core_from_logical_core(coordinates[i + 1]) : CoreCoord(0, 0);
        const uint32_t next_count = rank + 1 < chain ? jobs_per_head / chain + (rank + 1 < jobs_per_head % chain) : 0;
        reader.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                q.buffer()->address(),
                k.buffer()->address(),
                v.buffer()->address(),
                offset,
                count,
                rank,
                chain,
                prev.x,
                prev.y,
                next.x,
                next.y,
                next_count});
        writer.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{output.buffer()->address(), offset, count});
        compute.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{count});
    }
    program.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    return ttnn::generic_op({q, k, v, output}, program);
}

}  // namespace ttnn::operations::transformer::sdpa::detail
