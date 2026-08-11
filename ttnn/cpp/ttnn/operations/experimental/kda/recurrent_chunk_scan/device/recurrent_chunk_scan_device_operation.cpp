// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "recurrent_chunk_scan_device_operation.hpp"

#include <array>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {
namespace {

void check_protocol_tensor(const Tensor& tensor, const char* name, bool allow_bf16) {
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE && tensor.buffer() != nullptr,
        "recurrent_chunk_scan: {} must be an allocated device tensor",
        name);
    TT_FATAL(tensor.layout() == Layout::TILE, "recurrent_chunk_scan: {} must use TILE layout", name);
    TT_FATAL(
        tensor.dtype() == DataType::FLOAT32 || (allow_bf16 && tensor.dtype() == DataType::BFLOAT16),
        "recurrent_chunk_scan: {} must be FLOAT32{}",
        name,
        allow_bf16 ? " or BFLOAT16" : "");
    TT_FATAL(!tensor.is_sharded(), "recurrent_chunk_scan: {} must use interleaved memory", name);
}

void check_shape(const Tensor& tensor, const Shape& shape, const char* name) {
    TT_FATAL(tensor.logical_shape() == shape, "recurrent_chunk_scan: {} shape mismatch", name);
}

}  // namespace

RecurrentChunkScanOperation::program_factory_t RecurrentChunkScanOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return RecurrentChunkScanProgramFactory{};
}

void RecurrentChunkScanOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    check_protocol_tensor(in.v_beta, "v_beta", true);
    check_protocol_tensor(in.kd, "kd", true);
    check_protocol_tensor(in.q_decay, "q_decay", true);
    check_protocol_tensor(in.intra, "intra", false);
    check_protocol_tensor(in.k_dec_t, "k_dec_t", true);
    check_protocol_tensor(in.final_decay, "final_decay", true);
    check_protocol_tensor(in.t_inv, "t_inv", false);

    const std::array<const Tensor*, 7> protocol_inputs = {
        &in.v_beta, &in.kd, &in.q_decay, &in.intra, &in.k_dec_t, &in.final_decay, &in.t_inv};
    for (const auto* tensor : protocol_inputs) {
        TT_FATAL(tensor->device() == in.v_beta.device(), "recurrent_chunk_scan: all inputs must be on the same device");
    }
    TT_FATAL(
        attrs.mode == RecurrentChunkScanMode::SUMMARY || !attrs.output_mem_config.is_sharded(),
        "recurrent_chunk_scan: output memory must be interleaved");
    TT_FATAL(attrs.batch_heads > 0, "recurrent_chunk_scan: batch_heads must be positive");
    TT_FATAL(attrs.num_chunks > 0, "recurrent_chunk_scan: num_chunks must be positive");
    TT_FATAL(
        attrs.key_dim > 0 && attrs.value_dim > 0 && attrs.key_dim % tt::constants::TILE_WIDTH == 0 &&
            attrs.value_dim % tt::constants::TILE_WIDTH == 0,
        "recurrent_chunk_scan: K and V must be positive and tile aligned");

    constexpr uint32_t chunk_size = tt::constants::TILE_HEIGHT;
    const auto BH = attrs.batch_heads;
    const auto NC = attrs.num_chunks;
    const auto K = attrs.key_dim;
    const auto V = attrs.value_dim;
    check_shape(in.v_beta, Shape({BH, NC, chunk_size, V}), "v_beta");
    check_shape(in.kd, Shape({BH, NC, chunk_size, K}), "kd");
    check_shape(in.q_decay, Shape({BH, NC, chunk_size, K}), "q_decay");
    check_shape(in.intra, Shape({BH, NC, chunk_size, chunk_size}), "intra");
    check_shape(in.k_dec_t, Shape({BH, NC, K, chunk_size}), "k_dec_t");
    check_shape(in.final_decay, Shape({BH, NC, K, 1}), "final_decay");
    check_shape(in.t_inv, Shape({BH, NC, chunk_size, chunk_size}), "t_inv");

    if (attrs.mode == RecurrentChunkScanMode::RECURRENT) {
        TT_FATAL(in.initial_state.has_value(), "recurrent_chunk_scan: initial_state is required");
        check_protocol_tensor(*in.initial_state, "initial_state", false);
        TT_FATAL(
            in.initial_state->device() == in.v_beta.device(),
            "recurrent_chunk_scan: all inputs must be on the same device");
        check_shape(*in.initial_state, Shape({BH, K, V}), "initial_state");
    } else {
        TT_FATAL(!in.initial_state.has_value(), "summarize_chunk_recurrence: initial_state is not accepted");
        TT_FATAL(K == V, "summarize_chunk_recurrence: K must equal V");
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
    TT_FATAL(value_shape.rank() == 4 && key_shape.rank() == 4, "recurrent_chunk_scan: v_beta and kd must be rank 4");
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
