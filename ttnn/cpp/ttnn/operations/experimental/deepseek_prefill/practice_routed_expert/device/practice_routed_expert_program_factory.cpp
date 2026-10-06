// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "practice_routed_expert_program_factory.hpp"

#include <cstdint>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::practice_routed_expert {

tt::tt_metal::ProgramDescriptor PracticeRoutedExpertProgramFactory::create_descriptor(
    const PracticeRoutedExpertParams& operation_attributes,
    const PracticeRoutedExpertInputs& tensor_args,
    Tensor& tensor_return_value) {
    tt::tt_metal::ProgramDescriptor desc;

    const auto& [x, w_gate, w_up, w_down] = tensor_args;
    Tensor& output = tensor_return_value;

    const uint32_t m_tiles = x.logical_shape()[0] / tt::constants::TILE_HEIGHT;
    const uint32_t k_tiles = x.logical_shape()[1] / tt::constants::TILE_WIDTH;
    const uint32_t n_tiles = w_gate.logical_shape()[1] / tt::constants::TILE_WIDTH;

    const tt::DataFormat x_format = tt::tt_metal::datatype_to_dataformat_converter(x.dtype());
    const tt::DataFormat w_format = tt::tt_metal::datatype_to_dataformat_converter(w_gate.dtype());
    const tt::DataFormat out_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    // H stays bf16 whatever x is: narrowing it would add a rounding step the reference does not have.
    constexpr tt::DataFormat h_format = tt::DataFormat::Float16_b;

    // A single core walks the whole problem.
    const CoreCoord core{0, 0};
    const CoreRangeSet core_range_set{CoreRange{core, core}};

    constexpr uint32_t cb_x = tt::CBIndex::c_0;
    constexpr uint32_t cb_w_gate = tt::CBIndex::c_1;
    constexpr uint32_t cb_w_up = tt::CBIndex::c_2;
    constexpr uint32_t cb_w_down = tt::CBIndex::c_3;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_h = tt::CBIndex::c_24;

    auto add_cb = [&](uint32_t cb_idx, uint32_t num_tiles, tt::DataFormat data_format) {
        const uint32_t tile_size = tt::tile_size(data_format);
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = num_tiles * tile_size,
            .core_ranges = core_range_set,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_idx),
                .data_format = data_format,
                .page_size = tile_size,
            }}},
        });
    };

    // Two slots, so the reader fetches the next tile while compute uses the current one.
    add_cb(cb_x, 2, x_format);
    add_cb(cb_w_gate, 2, w_format);
    add_cb(cb_w_up, 2, w_format);
    add_cb(cb_w_down, 2, w_format);
    add_cb(cb_out, 2, out_format);
    // A whole row of H: the down matmul needs all of it, and compute both fills and drains it.
    add_cb(cb_h, n_tiles, h_format);

    auto* x_buffer = x.buffer();
    auto* w_gate_buffer = w_gate.buffer();
    auto* w_up_buffer = w_up.buffer();
    auto* w_down_buffer = w_down.buffer();
    auto* output_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args = {
        cb_x,
        cb_w_gate,
        cb_w_up,
        cb_w_down,
        m_tiles,
        k_tiles,
        n_tiles,
    };
    tt::tt_metal::TensorAccessorArgs(x_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(w_gate_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(w_up_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(w_down_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {cb_out, m_tiles * k_tiles};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);

    const std::vector<uint32_t> compute_compile_time_args = {
        cb_x,
        cb_w_gate,
        cb_w_up,
        cb_w_down,
        cb_h,
        cb_out,
        m_tiles,
        k_tiles,
        n_tiles,
    };

    tt::tt_metal::KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/practice_routed_expert/device/kernels/dataflow/"
        "reader_practice_routed_expert.cpp";
    reader_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = core_range_set;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel_desc.config = tt::tt_metal::ReaderConfigDescriptor{};
    reader_kernel_desc.emplace_runtime_args(core, {x_buffer, w_gate_buffer, w_up_buffer, w_down_buffer});

    tt::tt_metal::KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/practice_routed_expert/device/kernels/dataflow/"
        "writer_practice_routed_expert.cpp";
    writer_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = core_range_set;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_kernel_desc.config = tt::tt_metal::WriterConfigDescriptor{};
    writer_kernel_desc.emplace_runtime_args(core, {output_buffer});

    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(x.device()->arch(), operation_attributes.compute_kernel_config);

    tt::tt_metal::KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/practice_routed_expert/device/kernels/compute/"
        "practice_routed_expert_compute.cpp";
    compute_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = core_range_set;
    compute_kernel_desc.compile_time_args = compute_compile_time_args;
    compute_kernel_desc.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    desc.kernels.push_back(std::move(reader_kernel_desc));
    desc.kernels.push_back(std::move(writer_kernel_desc));
    desc.kernels.push_back(std::move(compute_kernel_desc));
    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::practice_routed_expert
