// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "qkv_causal_conv1d_silu_device_operation.hpp"

#include <array>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"
#include "ttnn/operations/experimental/kda/kda_performance_model.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {

constexpr std::string_view operation_name = "qkv_causal_conv1d_silu";

void check_tap_tensors(const QkvCausalConv1dSiluInputs& in) {
    using namespace kda_factory_detail;
    for (const auto& [tensor, name] : std::array{
             std::pair{&in.tap0, "tap0"},
             std::pair{&in.tap1, "tap1"},
             std::pair{&in.tap2, "tap2"},
             std::pair{&in.tap3, "tap3"}}) {
        check_allocated_device_tensor(*tensor, operation_name, name);
        check_layout(*tensor, Layout::TILE, operation_name, name);
        check_dtype(*tensor, DataType::BFLOAT16, operation_name, name);
        check_interleaved(*tensor, operation_name, name);
    }
}

void check_same_device_inputs(const QkvCausalConv1dSiluInputs& in) {
    using namespace kda_factory_detail;
    if (in.history.has_value()) {
        check_same_device(in.input, *in.history, operation_name, "history");
    }
    check_same_device(in.input, in.tap0, operation_name, "tap0");
    check_same_device(in.input, in.tap1, operation_name, "tap1");
    check_same_device(in.input, in.tap2, operation_name, "tap2");
    check_same_device(in.input, in.tap3, operation_name, "tap3");
}

void check_default_tile_shape(const Tensor& tensor, std::string_view tensor_name) {
    const auto tile = tensor.tensor_spec().tile();
    TT_FATAL(
        tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH,
        "{}: {} must use 32x32 tiles, got {}x{}",
        operation_name,
        tensor_name,
        tile.get_height(),
        tile.get_width());
}

// Shape, width, chunk and config checks shared by the ROW_MAJOR and TILE paths.
void check_geometry_and_config(const QkvCausalConv1dSiluParams& attrs, const QkvCausalConv1dSiluInputs& in) {
    using namespace kda_factory_detail;
    TT_FATAL(
        attrs.q_width > 0 && attrs.k_width > 0 && attrs.v_width > 0,
        "qkv_causal_conv1d_silu: Q/K/V widths must be positive");
    TT_FATAL(
        attrs.q_width % tt::constants::TILE_WIDTH == 0 && attrs.k_width % tt::constants::TILE_WIDTH == 0 &&
            attrs.v_width % tt::constants::TILE_WIDTH == 0,
        "qkv_causal_conv1d_silu: Q/K/V widths must be tile aligned");
    const uint64_t channels =
        static_cast<uint64_t>(attrs.q_width) + static_cast<uint64_t>(attrs.k_width) + attrs.v_width;
    TT_FATAL(attrs.channel_chunk_size > 0, "qkv_causal_conv1d_silu: channel_chunk_size must be positive");
    TT_FATAL(
        attrs.channel_chunk_size % tt::constants::TILE_WIDTH == 0,
        "qkv_causal_conv1d_silu: channel_chunk_size must be tile aligned");
    TT_FATAL(
        attrs.channel_chunk_size <= channels, "qkv_causal_conv1d_silu: channel_chunk_size must not exceed Q+K+V width");
    TT_FATAL(
        channels % attrs.channel_chunk_size == 0,
        "qkv_causal_conv1d_silu: channel_chunk_size must divide Q+K+V width exactly");

    const auto& input_shape = in.input.logical_shape();
    TT_FATAL(
        input_shape.rank() == 3 && input_shape[0] == 1 && input_shape[1] == attrs.sequence &&
            input_shape[2] == channels,
        "qkv_causal_conv1d_silu: input must be [1,T,Q+K+V]");
    if (in.history.has_value()) {
        const auto& history_shape = in.history->logical_shape();
        TT_FATAL(
            history_shape.rank() == 3 && history_shape[0] == 1 && history_shape[1] == 3 && history_shape[2] == channels,
            "qkv_causal_conv1d_silu: history must be [1,3,Q+K+V]");
    }
    TT_FATAL(
        attrs.sequence > 0 && attrs.sequence % tt::constants::TILE_HEIGHT == 0,
        "qkv_causal_conv1d_silu: sequence must be positive and tile aligned");

    for (const auto& [tensor, name] : std::array{
             std::pair{&in.tap0, "tap0"},
             std::pair{&in.tap1, "tap1"},
             std::pair{&in.tap2, "tap2"},
             std::pair{&in.tap3, "tap3"}}) {
        TT_FATAL(
            tensor->logical_shape()[-1] == channels,
            "qkv_causal_conv1d_silu: {} last dimension must equal Q+K+V",
            name);
        TT_FATAL(
            tensor->logical_volume() == channels, "qkv_causal_conv1d_silu: {} logical volume must equal Q+K+V", name);
    }
    check_output_interleaved(attrs.output_mem_config, operation_name);
    check_compute_config(attrs.compute_kernel_config, operation_name);
    TT_FATAL(
        !attrs.compute_kernel_config.math_approx_mode,
        "qkv_causal_conv1d_silu: math_approx_mode=true is unsupported because silu_tile always uses precise sigmoid");
}

// ROW_MAJOR path: the checks and their order are the same as before the tiled path existed.
void validate_row_major(const QkvCausalConv1dSiluParams& attrs, const QkvCausalConv1dSiluInputs& in) {
    using namespace kda_factory_detail;
    check_allocated_device_tensor(in.input, operation_name, "input");
    check_layout(in.input, Layout::ROW_MAJOR, operation_name, "input");
    TT_FATAL(!attrs.fused_qk_l2_norm, "qkv_causal_conv1d_silu: fused_qk_l2_norm applies to TILE input only");
    check_dtype(in.input, DataType::BFLOAT16, operation_name, "input");
    check_interleaved(in.input, operation_name, "input");
    TT_FATAL(
        in.history.has_value(),
        "qkv_causal_conv1d_silu: history is required for ROW_MAJOR input (history=None needs TILE input)");
    check_allocated_device_tensor(*in.history, operation_name, "history");
    check_layout(*in.history, Layout::ROW_MAJOR, operation_name, "history");
    check_dtype(*in.history, DataType::BFLOAT16, operation_name, "history");
    check_interleaved(*in.history, operation_name, "history");
    check_tap_tensors(in);
    check_same_device_inputs(in);
    TT_FATAL(
        !attrs.return_conv_state,
        "qkv_causal_conv1d_silu: return_conv_state=True requires TILE input; ROW_MAJOR callers own the history "
        "update");
    check_geometry_and_config(attrs, in);
}

// TILE path (design.md section 6.1): QkvCausalConv1dSiluTiledProgramFactory.
void validate_tiled(const QkvCausalConv1dSiluParams& attrs, const QkvCausalConv1dSiluInputs& in) {
    using namespace kda_factory_detail;
    check_allocated_device_tensor(in.input, operation_name, "input");
    check_layout(in.input, Layout::TILE, operation_name, "input");
    if (attrs.fused_qk_l2_norm) {
        TT_FATAL(
            attrs.channel_chunk_size == 128 && attrs.q_width % 128 == 0 && attrs.k_width % 128 == 0,
            "qkv_causal_conv1d_silu: fused_qk_l2_norm needs channel_chunk_size 128 (one 128-channel head per step) "
            "and q/k widths that are multiples of 128, got channel_chunk_size={}, q_width={}, k_width={}",
            attrs.channel_chunk_size,
            attrs.q_width,
            attrs.k_width);
    }
    check_dtype(in.input, DataType::BFLOAT16, operation_name, "input");
    check_interleaved(in.input, operation_name, "input");
    check_default_tile_shape(in.input, "input");
    if (in.history.has_value()) {
        const Tensor& history = *in.history;
        check_allocated_device_tensor(history, operation_name, "history");
        // A ROW_MAJOR history with a TILE input is a layout mix. Name the input first: that keeps the
        // message of the ROW_MAJOR contract, and it names the other valid option.
        TT_FATAL(
            history.layout() == Layout::TILE,
            "qkv_causal_conv1d_silu: input must use ROW_MAJOR layout, got {}, because history is {}. "
            "A TILE input selects the tiled path, which needs a TILE history or history=None",
            in.input.layout(),
            history.layout());
        check_dtype(history, DataType::BFLOAT16, operation_name, "history");
        check_interleaved(history, operation_name, "history");
        check_default_tile_shape(history, "history");
    }
    check_tap_tensors(in);
    check_same_device_inputs(in);
    check_geometry_and_config(attrs, in);

    const uint32_t block_tiles = attrs.channel_chunk_size / tt::constants::TILE_WIDTH;
    TT_FATAL(
        qkv_causal_conv1d_silu_tiled::is_supported_block_tiles(block_tiles),
        "qkv_causal_conv1d_silu: TILE input needs channel_chunk_size in {{32, 64, 128, 256}} (a block of 1, 2, 4 or "
        "8 tiles; 8 tiles fill one bf16 dest half), got {}",
        attrs.channel_chunk_size);
    const uint32_t dest_tiles = ttnn::get_dest_reg_count(attrs.compute_kernel_config);
    TT_FATAL(
        block_tiles <= dest_tiles,
        "qkv_causal_conv1d_silu: TILE input with channel_chunk_size={} needs {} dest tiles, but the compute config "
        "gives {} (fp32_dest_acc_en halves the dest capacity)",
        attrs.channel_chunk_size,
        block_tiles,
        dest_tiles);
}

}  // namespace

QkvCausalConv1dSiluOperation::program_factory_t QkvCausalConv1dSiluOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t& in) {
    if (in.input.layout() == Layout::TILE) {
        return QkvCausalConv1dSiluTiledProgramFactory{};
    }
    return QkvCausalConv1dSiluProgramFactory{};
}

void QkvCausalConv1dSiluOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    kda_factory_detail::check_allocated_device_tensor(in.input, operation_name, "input");
    if (in.input.layout() == Layout::TILE) {
        validate_tiled(attrs, in);
    } else {
        validate_row_major(attrs, in);
    }
}

QkvCausalConv1dSiluOperation::spec_return_value_t QkvCausalConv1dSiluOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const auto layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), attrs.output_mem_config);
    // fused_qk_l2_norm: q and k are fp32 (the normalized output); v stays bf16.
    const auto qk_layout = attrs.fused_qk_l2_norm
                               ? TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config)
                               : layout;
    spec_return_value_t specs = {
        TensorSpec(Shape({1, attrs.sequence, attrs.q_width}), qk_layout),
        TensorSpec(Shape({1, attrs.sequence, attrs.k_width}), qk_layout),
        TensorSpec(Shape({1, attrs.sequence, attrs.v_width}), layout)};
    if (attrs.return_conv_state) {
        // new_state: rows 0-2 = x[T-3..T-1]; the tile padding rows 3-31 are zero. It is always
        // DRAM interleaved (design.md section 4.5): output_mem_config applies to q/k/v only.
        const auto state_layout = TensorLayout(
            DataType::BFLOAT16,
            PageConfig(Layout::TILE),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM});
        specs.push_back(TensorSpec(Shape({1, 3, attrs.q_width + attrs.k_width + attrs.v_width}), state_layout));
    }
    return specs;
}

QkvCausalConv1dSiluOperation::tensor_return_value_t QkvCausalConv1dSiluOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    tensor_return_value_t outputs;
    outputs.reserve(specs.size());
    for (const auto& spec : specs) {
        outputs.push_back(create_device_tensor(spec, in.input.device()));
    }
    return outputs;
}

tt::tt_metal::operation::OpPerformanceModelGeneral<QkvCausalConv1dSiluOperation::tensor_return_value_t>
QkvCausalConv1dSiluOperation::create_op_performance_model(
    const operation_attributes_t& attrs, const tensor_args_t& in, tensor_return_value_t& outputs) {
    using namespace kda_performance_model;

    const auto& input_shape = in.input.logical_shape();
    const double width = static_cast<double>(attrs.q_width) + attrs.k_width + attrs.v_width;
    const double elements = static_cast<double>(input_shape[0]) * attrs.sequence * width;
    const KdaFpuWork work{
        .fpu_multiply_ops = 4.0 * elements,
        .fpu_add_ops = 3.0 * elements,
    };
    std::vector<const Tensor*> inputs = {&in.input};
    if (in.history.has_value()) {
        inputs.push_back(&*in.history);
    }
    inputs.insert(inputs.end(), {&in.tap0, &in.tap1, &in.tap2, &in.tap3});
    return make_profiler_model(work, inputs, outputs, attrs.compute_kernel_config.math_fidelity);
}

std::vector<Tensor> qkv_causal_conv1d_silu(
    const Tensor& input,
    const std::optional<Tensor>& history,
    const Tensor& tap0,
    const Tensor& tap1,
    const Tensor& tap2,
    const Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    uint32_t channel_chunk_size,
    bool return_conv_state,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool fused_qk_l2_norm) {
    const auto& input_shape = input.logical_shape();
    TT_FATAL(input_shape.rank() == 3, "qkv_causal_conv1d_silu: input must be [1,T,Q+K+V]");
    return ttnn::device_operation::launch<QkvCausalConv1dSiluOperation>(
        QkvCausalConv1dSiluParams{
            .sequence = static_cast<uint32_t>(input_shape[1]),
            .q_width = q_width,
            .k_width = k_width,
            .v_width = v_width,
            .channel_chunk_size = channel_chunk_size,
            .return_conv_state = return_conv_state,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config,
            .fused_qk_l2_norm = fused_qk_l2_norm},
        QkvCausalConv1dSiluInputs{
            .input = input, .history = history, .tap0 = tap0, .tap1 = tap1, .tap2 = tap2, .tap3 = tap3});
}

}  // namespace ttnn::experimental::prim
