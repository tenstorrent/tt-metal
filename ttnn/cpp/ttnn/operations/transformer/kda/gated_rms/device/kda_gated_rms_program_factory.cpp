// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_gated_rms_program_factory.hpp"

#include <cstring>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "../../device/kda_factory_utils.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor KdaGatedRmsProgramFactory::create_descriptor(
    const KdaGatedRmsParams& attrs, const KdaGatedRmsInputs& in, std::vector<Tensor>& outputs) {
    const uint32_t Mt = attrs.sequence / TILE_HEIGHT;
    const uint32_t Vt = attrs.value_dim / TILE_WIDTH;
    const uint32_t total = attrs.batch * attrs.num_heads * Mt;
    // Use the fewest workers that preserve the all-core maximum items/worker.
    const auto grid = in.input.device()->compute_with_storage_grid_size();
    const uint32_t max_items_per_core = tt::div_up(total, grid.x * grid.y);
    const uint32_t rms_core_limit = tt::div_up(total, max_items_per_core);
    auto dist =
        kda_factory_detail::distribute_prep(in.input.device()->compute_with_storage_grid_size(), total, rms_core_limit);
    const auto& cores = dist.core_set;

    ProgramDescriptor desc;
    auto add_cb = [&](uint32_t idx, uint32_t tiles, tt::DataFormat format, uint32_t buffers = 1) {
        const uint32_t tile_size = tt::tile_size(format);
        desc.cbs.push_back(CBDescriptor{
            .total_size = tiles * buffers * tile_size,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(idx), .data_format = format, .page_size = tile_size}}}});
    };
    add_cb(0, Vt, datatype_to_dataformat_converter(in.input.dtype()));
    add_cb(1, Vt, tt::DataFormat::Float16_b);
    add_cb(2, Vt, tt::DataFormat::Float16_b);
    add_cb(3, Vt, tt::DataFormat::Float32);
    add_cb(4, 1, tt::DataFormat::Float32);
    add_cb(5, 1, tt::DataFormat::Float32);
    add_cb(6, Vt, tt::DataFormat::Float32);
    add_cb(7, Vt, datatype_to_dataformat_converter(attrs.output_dtype), 2);
    add_cb(8, 1, tt::DataFormat::Float32);

    uint32_t eps_bits = 0;
    uint32_t inv_v_bits = 0;
    const float inv_v = 1.0f / static_cast<float>(attrs.value_dim);
    std::memcpy(&eps_bits, &attrs.epsilon, sizeof(float));
    std::memcpy(&inv_v_bits, &inv_v, sizeof(float));

    std::vector<uint32_t> reader_ct = {Vt, attrs.num_heads, Mt};
    TensorAccessorArgs(*in.input.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.gate.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.weight.buffer()).append_to(reader_ct);
    std::vector<uint32_t> writer_ct = {Vt, attrs.num_heads, Mt};
    TensorAccessorArgs(*outputs[0].buffer()).append_to(writer_ct);

    KernelDescriptor reader;
    reader.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/kda/gated_rms/device/kernels/dataflow/reader_kda_gated_rms.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_ct;
    reader.config = ReaderConfigDescriptor{};

    KernelDescriptor writer;
    writer.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/kda/gated_rms/device/kernels/dataflow/writer_kda_gated_rms.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_ct;
    writer.config = WriterConfigDescriptor{};

    KernelDescriptor compute;
    compute.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/kda/gated_rms/device/kernels/compute/kda_gated_rms.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = {Vt, eps_bits, inv_v_bits};
    compute.config = kda_factory_detail::kda_compute_cfg(in.input.device()->arch(), attrs.compute_kernel_config);

    auto* input_buffer = in.input.buffer();
    auto* gate_buffer = in.gate.buffer();
    auto* weight_buffer = in.weight.buffer();
    auto* output_buffer = outputs[0].buffer();
    for (uint32_t i = 0; i < dist.cores.size(); i++) {
        const auto& core = dist.cores[i];
        reader.emplace_runtime_args(
            core, {dist.wi_start[i], dist.wi_count[i], input_buffer, gate_buffer, weight_buffer});
        writer.emplace_runtime_args(core, {dist.wi_start[i], dist.wi_count[i], output_buffer});
        compute.emplace_runtime_args(core, {dist.wi_count[i]});
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::prim
