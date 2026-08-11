// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "reduce_affine_transforms_program_factory.hpp"

#include <algorithm>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

tt::tt_metal::ProgramDescriptor ReduceAffineTransformsProgramFactory::create_descriptor(
    const ReduceAffineTransformsParams& attrs, const ReduceAffineTransformsInputs& in, std::vector<Tensor>& outputs) {
    const uint32_t Kt = attrs.key_dim / TILE_WIDTH;
    const uint32_t Vt = attrs.value_dim / TILE_WIDTH;
    const uint32_t G = attrs.groups_per_head;
    const uint32_t group_heads = attrs.batch_heads * G;
    const uint32_t kk = Kt * Kt;
    const uint32_t kv = Kt * Vt;

    auto* device = in.a.device();
    const auto grid = device->compute_with_storage_grid_size();
    constexpr uint32_t kMaxAffineCompositionWorkers = 128;
    TT_FATAL(
        group_heads <= std::min<uint32_t>(grid.x * grid.y, kMaxAffineCompositionWorkers),
        "reduce_affine_transforms supports at most {} group workers, got {}",
        kMaxAffineCompositionWorkers,
        group_heads);
    auto dist = kda_factory_detail::distribute_prep(grid, group_heads, group_heads);
    const auto& cores = dist.core_set;

    ProgramDescriptor desc;
    auto add_cb = [&](uint32_t index, uint32_t tiles, tt::DataFormat format = tt::DataFormat::Float32) {
        const uint32_t tile_size = tt::tile_size(format);
        desc.cbs.push_back(CBDescriptor{
            .total_size = tiles * tile_size,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index), .data_format = format, .page_size = tile_size}}}});
    };
    const auto summary_format = tt::tt_metal::datatype_to_dataformat_converter(in.a.dtype());
    add_cb(0, kk, summary_format);
    add_cb(1, kv, summary_format);
    add_cb(2, kk);
    add_cb(3, kv);
    add_cb(4, kk);
    add_cb(5, kv);
    add_cb(6, kk);
    add_cb(7, kv);
    add_cb(9, 1);
    add_cb(10, kv);
    add_cb(11, 1);

    constexpr uint32_t ready_semaphore_id = 0;
    constexpr uint32_t arrival_semaphore_id = 1;
    constexpr uint32_t release_semaphore_id = 2;
    for (uint32_t id : {ready_semaphore_id, arrival_semaphore_id, release_semaphore_id}) {
        desc.semaphores.push_back(
            SemaphoreDescriptor{.id = id, .core_type = tt::CoreType::WORKER, .core_ranges = cores, .initial_value = 0});
    }

    auto* output_a_buffer = outputs[0].buffer();
    auto* output_b_buffer = outputs[1].buffer();
    std::vector<uint32_t> dataflow_ct = {Kt, Vt, attrs.batch_heads, G};
    TensorAccessorArgs(*in.a.buffer()).append_to(dataflow_ct);
    TensorAccessorArgs(*in.b.buffer()).append_to(dataflow_ct);
    TensorAccessorArgs(*output_a_buffer).append_to(dataflow_ct);
    TensorAccessorArgs(*output_b_buffer).append_to(dataflow_ct);

    KernelDescriptor dataflow;
    dataflow.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/kda/reduce_affine_transforms/device/kernels/dataflow/"
        "reader_writer_reduce_affine_transforms.cpp";
    dataflow.source_type = KernelDescriptor::SourceType::FILE_PATH;
    dataflow.core_ranges = cores;
    dataflow.compile_time_args = dataflow_ct;
    dataflow.config = ReaderConfigDescriptor{};
    dataflow.runtime_args.reserve(group_heads);

    KernelDescriptor compute;
    compute.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/kda/reduce_affine_transforms/device/kernels/compute/"
        "reduce_affine_transforms.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = {Kt, Vt, G};
    compute.config = kda_factory_detail::kda_compute_cfg(device->arch(), attrs.compute_kernel_config);
    compute.runtime_args.reserve(group_heads);

    auto* a_buffer = in.a.buffer();
    auto* b_buffer = in.b.buffer();
    const auto coordinator = device->worker_core_from_logical_core(dist.cores[0]);
    for (uint32_t flat = 0; flat < group_heads; flat++) {
        const auto& core = dist.cores[flat];
        const uint32_t group = flat % G;
        KernelDescriptor::RTArgList args;
        args.reserve(12 + 2 * group_heads);
        args.push_back(flat);
        args.push_back(group);
        args.push_back(group_heads);
        args.push_back(a_buffer);
        args.push_back(b_buffer);
        args.push_back(output_a_buffer);
        args.push_back(output_b_buffer);
        args.push_back(ready_semaphore_id);
        args.push_back(arrival_semaphore_id);
        args.push_back(release_semaphore_id);
        args.push_back(coordinator.x);
        args.push_back(coordinator.y);
        for (const auto& worker : dist.cores) {
            const auto physical = device->worker_core_from_logical_core(worker);
            args.push_back(physical.x);
            args.push_back(physical.y);
        }
        dataflow.emplace_runtime_args(core, std::move(args));
        compute.emplace_runtime_args(core, {group});
    }

    desc.kernels.push_back(std::move(dataflow));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::experimental::prim
