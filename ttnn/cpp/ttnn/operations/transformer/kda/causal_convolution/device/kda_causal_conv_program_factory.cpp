// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_causal_conv_program_factory.hpp"

#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "../../device/kda_factory_utils.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor KdaCausalConvProgramFactory::create_descriptor(
    const KdaCausalConvParams& attrs, const KdaCausalConvInputs& in, std::vector<Tensor>& outputs) {
    constexpr uint32_t act_rm_cb = tt::CBIndex::c_0;
    constexpr uint32_t act_tile_cb = tt::CBIndex::c_1;
    constexpr uint32_t weights_cb = tt::CBIndex::c_2;
    constexpr uint32_t partial_a_cb = tt::CBIndex::c_3;
    constexpr uint32_t partial_b_cb = tt::CBIndex::c_4;
    constexpr uint32_t output_cb = tt::CBIndex::c_5;
    const uint32_t Mt = attrs.sequence / TILE_HEIGHT;
    const uint32_t Qt = attrs.q_width / TILE_WIDTH;
    const uint32_t Kt = attrs.k_width / TILE_WIDTH;
    const uint32_t Vt = attrs.v_width / TILE_WIDTH;
    const uint32_t Ct = Qt + Kt + Vt;
    const uint32_t channels = attrs.q_width + attrs.k_width + attrs.v_width;
    const uint32_t row_bytes = channels * sizeof(uint16_t);
    // Preserve the single-block TP8 case. Wider local head shards use smaller blocks so the nine CBs coexist in L1.
    uint32_t block_ct = Ct <= 48 ? Ct : 24u;
    while (Ct % block_ct != 0) {
        --block_ct;
    }
    const uint32_t num_blocks = Ct / block_ct;
    auto dist =
        kda_factory_detail::distribute_prep(in.input.device()->compute_with_storage_grid_size(), Mt * num_blocks, ~0u);
    const auto& cores = dist.core_set;

    ProgramDescriptor desc;
    auto add_tile_cb = [&](uint32_t idx, uint32_t tiles) {
        const uint32_t tile_size = tt::tile_size(tt::DataFormat::Float16_b);
        desc.cbs.push_back(CBDescriptor{
            .total_size = tiles * tile_size,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(idx),
                .data_format = tt::DataFormat::Float16_b,
                .page_size = tile_size}}}});
    };
    add_tile_cb(act_rm_cb, block_ct);
    add_tile_cb(act_tile_cb, block_ct);
    add_tile_cb(weights_cb, 4 * block_ct);
    add_tile_cb(partial_a_cb, block_ct);
    add_tile_cb(partial_b_cb, block_ct);
    add_tile_cb(output_cb, block_ct);

    std::vector<uint32_t> reader_ct = {block_ct, channels, row_bytes, num_blocks};
    TensorAccessorArgs(*in.input.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.state.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.tap0.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.tap1.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.tap2.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.tap3.buffer()).append_to(reader_ct);
    std::vector<uint32_t> writer_ct = {Qt, Kt, Vt, block_ct, num_blocks};
    TensorAccessorArgs(*outputs[0].buffer()).append_to(writer_ct);
    TensorAccessorArgs(*outputs[1].buffer()).append_to(writer_ct);
    TensorAccessorArgs(*outputs[2].buffer()).append_to(writer_ct);

    KernelDescriptor reader;
    reader.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/kda/causal_convolution/device/kernels/dataflow/"
        "reader_kda_causal_conv1d.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_ct;
    reader.config = ReaderConfigDescriptor{};
    KernelDescriptor writer;
    writer.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/kda/causal_convolution/device/kernels/dataflow/"
        "writer_kda_causal_conv1d.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_ct;
    writer.config = WriterConfigDescriptor{};
    KernelDescriptor compute;
    compute.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/kda/causal_convolution/device/kernels/compute/kda_causal_conv1d.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = {block_ct, num_blocks};
    compute.config = kda_factory_detail::kda_compute_cfg(in.input.device()->arch(), attrs.compute_kernel_config);

    for (uint32_t i = 0; i < dist.cores.size(); ++i) {
        const auto& core = dist.cores[i];
        reader.emplace_runtime_args(
            core,
            {dist.wi_start[i],
             dist.wi_count[i],
             in.input.buffer(),
             in.state.buffer(),
             in.tap0.buffer(),
             in.tap1.buffer(),
             in.tap2.buffer(),
             in.tap3.buffer()});
        writer.emplace_runtime_args(
            core, {dist.wi_start[i], dist.wi_count[i], outputs[0].buffer(), outputs[1].buffer(), outputs[2].buffer()});
        compute.emplace_runtime_args(core, {dist.wi_count[i]});
    }
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::prim
