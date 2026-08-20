// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "recurrent_chunk_scan_device_operation.hpp"

#include <array>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {
namespace {

void check_protocol_tensor(
    const Tensor& tensor, std::string_view name, bool allow_bf16, std::string_view operation_name) {
    using namespace kda_factory_detail;
    check_allocated_device_tensor(tensor, operation_name, name);
    check_layout(tensor, Layout::TILE, operation_name, name);
    if (allow_bf16) {
        constexpr std::array accepted_dtypes = {DataType::FLOAT32, DataType::BFLOAT16};
        check_dtype_in(tensor, accepted_dtypes, "FLOAT32 or BFLOAT16", operation_name, name);
    } else {
        check_dtype(tensor, DataType::FLOAT32, operation_name, name);
    }
    check_interleaved(tensor, operation_name, name);
}

void check_shape(const Tensor& tensor, const Shape& shape, std::string_view name, std::string_view operation_name) {
    TT_FATAL(tensor.logical_shape() == shape, "{}: {} shape mismatch", operation_name, name);
}

}  // namespace

RecurrentChunkScanOperation::program_factory_t RecurrentChunkScanOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return RecurrentChunkScanProgramFactory{};
}

void RecurrentChunkScanOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace kda_factory_detail;
    const std::string_view operation_name =
        attrs.mode == RecurrentChunkScanMode::RECURRENT ? "recurrent_chunk_scan" : "summarize_chunk_recurrence";
    check_protocol_tensor(in.v_beta, "v_beta", true, operation_name);
    check_protocol_tensor(in.kd, "kd", true, operation_name);
    check_protocol_tensor(in.q_decay, "q_decay", true, operation_name);
    check_protocol_tensor(in.intra, "intra", false, operation_name);
    check_protocol_tensor(in.k_dec_t, "k_dec_t", true, operation_name);
    check_protocol_tensor(in.final_decay, "final_decay", true, operation_name);
    check_protocol_tensor(in.t_inv, "t_inv", false, operation_name);

    for (const auto [tensor, name] : std::array{
             std::pair{&in.kd, "kd"},
             std::pair{&in.q_decay, "q_decay"},
             std::pair{&in.intra, "intra"},
             std::pair{&in.k_dec_t, "k_dec_t"},
             std::pair{&in.final_decay, "final_decay"},
             std::pair{&in.t_inv, "t_inv"}}) {
        check_same_device(in.v_beta, *tensor, operation_name, name);
    }
    if (attrs.mode == RecurrentChunkScanMode::RECURRENT) {
        check_output_interleaved(attrs.output_mem_config, operation_name);
    }
    check_compute_config(attrs.compute_kernel_config, operation_name);
    TT_FATAL(attrs.batch_heads > 0, "{}: batch_heads must be positive", operation_name);
    TT_FATAL(attrs.num_chunks > 0, "{}: num_chunks must be positive", operation_name);
    TT_FATAL(
        attrs.key_dim > 0 && attrs.value_dim > 0 && attrs.key_dim % tt::constants::TILE_WIDTH == 0 &&
            attrs.value_dim % tt::constants::TILE_WIDTH == 0,
        "{}: K and V must be positive and tile aligned",
        operation_name);

    constexpr uint32_t chunk_size = tt::constants::TILE_HEIGHT;
    const auto BH = attrs.batch_heads;
    const auto NC = attrs.num_chunks;
    const auto K = attrs.key_dim;
    const auto V = attrs.value_dim;
    check_shape(in.v_beta, Shape({BH, NC, chunk_size, V}), "v_beta", operation_name);
    check_shape(in.kd, Shape({BH, NC, chunk_size, K}), "kd", operation_name);
    check_shape(in.q_decay, Shape({BH, NC, chunk_size, K}), "q_decay", operation_name);
    check_shape(in.intra, Shape({BH, NC, chunk_size, chunk_size}), "intra", operation_name);
    check_shape(in.k_dec_t, Shape({BH, NC, K, chunk_size}), "k_dec_t", operation_name);
    check_shape(in.final_decay, Shape({BH, NC, K, 1}), "final_decay", operation_name);
    check_shape(in.t_inv, Shape({BH, NC, chunk_size, chunk_size}), "t_inv", operation_name);

    if (attrs.mode == RecurrentChunkScanMode::RECURRENT) {
        TT_FATAL(in.initial_state.has_value(), "{}: initial_state is required", operation_name);
        check_protocol_tensor(*in.initial_state, "initial_state", false, operation_name);
        check_same_device(in.v_beta, *in.initial_state, operation_name, "initial_state");
        check_shape(*in.initial_state, Shape({BH, K, V}), "initial_state", operation_name);
    } else {
        TT_FATAL(!in.initial_state.has_value(), "{}: initial_state is not accepted", operation_name);
        TT_FATAL(K == V, "{}: K must equal V", operation_name);
    }
}

RecurrentChunkScanOperation::spec_return_value_t RecurrentChunkScanOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const bool summary = attrs.mode == RecurrentChunkScanMode::SUMMARY;
    const auto output_dtype = summary ? DataType::FLOAT32 : DataType::BFLOAT16;
    const auto output_layout = TensorLayout(output_dtype, PageConfig(Layout::TILE), attrs.output_mem_config);
    const auto state_layout = TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config);
    const auto first_shape =
        summary ? Shape({attrs.batch_heads, attrs.key_dim, attrs.value_dim})
                : Shape({attrs.batch_heads, attrs.num_chunks, tt::constants::TILE_HEIGHT, attrs.value_dim});
    return {
        TensorSpec(first_shape, output_layout),
        TensorSpec(Shape({attrs.batch_heads, attrs.key_dim, attrs.value_dim}), state_layout)};
}

RecurrentChunkScanOperation::tensor_return_value_t RecurrentChunkScanOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(attrs, in)) {
        outputs.push_back(create_device_tensor(spec, in.v_beta.device()));
    }
    return outputs;
}

std::vector<Tensor> recurrent_chunk_scan(
    const Tensor& v_beta,
    const Tensor& kd,
    const Tensor& q_decay,
    const Tensor& intra,
    const Tensor& k_dec_t,
    const Tensor& final_decay,
    const Tensor& t_inv,
    const std::optional<Tensor>& initial_state,
    RecurrentChunkScanMode mode,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    const auto& value_shape = v_beta.logical_shape();
    const auto& key_shape = kd.logical_shape();
    const std::string_view operation_name =
        mode == RecurrentChunkScanMode::RECURRENT ? "recurrent_chunk_scan" : "summarize_chunk_recurrence";
    TT_FATAL(value_shape.rank() == 4 && key_shape.rank() == 4, "{}: v_beta and kd must be rank 4", operation_name);
    return ttnn::device_operation::launch<RecurrentChunkScanOperation>(
        RecurrentChunkScanParams{
            .batch_heads = value_shape[0],
            .num_chunks = value_shape[1],
            .key_dim = key_shape[3],
            .value_dim = value_shape[3],
            .mode = mode,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config},
        RecurrentChunkScanInputs{
            .v_beta = v_beta,
            .kd = kd,
            .q_decay = q_decay,
            .intra = intra,
            .k_dec_t = k_dec_t,
            .final_decay = final_decay,
            .t_inv = t_inv,
            .initial_state = initial_state});
}

}  // namespace ttnn::experimental::prim
