// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_final_scan_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {
namespace {

void check_intermediate(const Tensor& tensor, const char* name, bool allow_bf16 = true) {
    TT_FATAL(tensor.layout() == Layout::TILE, "kda_final_chunk_scan: {} must be TILE layout", name);
    TT_FATAL(
        tensor.dtype() == DataType::FLOAT32 || (allow_bf16 && tensor.dtype() == DataType::BFLOAT16),
        "kda_final_chunk_scan: {} must be FLOAT32{}",
        name,
        allow_bf16 ? " or BFLOAT16" : "");
    TT_FATAL(tensor.buffer() != nullptr, "kda_final_chunk_scan: {} must be on device", name);
}

void check_shape(const Tensor& tensor, const Shape& shape, const char* name) {
    TT_FATAL(tensor.logical_shape() == shape, "kda_final_chunk_scan: {} shape mismatch", name);
}

}  // namespace

KdaFinalChunkScanOperation::program_factory_t KdaFinalChunkScanOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return KdaFinalChunkScanProgramFactory{};
}

void KdaFinalChunkScanOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace tt::constants;
    check_intermediate(in.v_beta, "v_beta");
    check_intermediate(in.kd, "kd");
    check_intermediate(in.q_decay, "q_decay");
    check_intermediate(in.intra, "intra", false);
    check_intermediate(in.k_dec_t, "k_dec_t");
    check_intermediate(in.final_decay, "final_decay");
    check_intermediate(in.t_inv, "t_inv", false);

    const auto BH = attrs.batch_heads;
    const auto NC = attrs.num_chunks;
    const auto C = attrs.chunk_size;
    const auto K = attrs.key_dim;
    const auto V = attrs.value_dim;
    check_shape(in.v_beta, Shape({BH, NC, C, V}), "v_beta");
    check_shape(in.kd, Shape({BH, NC, C, K}), "kd");
    check_shape(in.q_decay, Shape({BH, NC, C, K}), "q_decay");
    check_shape(in.intra, Shape({BH, NC, C, C}), "intra");
    check_shape(in.k_dec_t, Shape({BH, NC, K, C}), "k_dec_t");
    check_shape(in.final_decay, Shape({BH, NC, K, 1}), "final_decay");
    check_shape(in.t_inv, Shape({BH, NC, C, C}), "t_inv");

    if (in.initial_state.has_value()) {
        check_intermediate(*in.initial_state, "initial_state", false);
        check_shape(*in.initial_state, Shape({BH, K, V}), "initial_state");
    }
    if (in.identity_tile.has_value()) {
        check_intermediate(*in.identity_tile, "identity_tile", false);
        check_shape(*in.identity_tile, Shape({1, 1, 32, 32}), "identity_tile");
        TT_FATAL(K == V, "identity initial state requires K == V");
    }
    TT_FATAL(
        !attrs.summary_pair || (attrs.state_only && in.identity_tile.has_value()),
        "summary_pair requires state_only and an identity tile");
    TT_FATAL(!attrs.state_only || attrs.summary_pair, "state_only requires summary_pair");
    TT_FATAL(!(attrs.summary_pair && attrs.output_bf16), "summary_pair does not support BF16 output");
    TT_FATAL(
        !(in.initial_state.has_value() && in.identity_tile.has_value()),
        "initial_state and identity_tile are mutually exclusive");
    TT_FATAL(C % TILE_HEIGHT == 0, "chunk_size must be a multiple of 32");
    TT_FATAL(K % TILE_WIDTH == 0, "key_dim must be a multiple of 32");
    TT_FATAL(V % TILE_WIDTH == 0, "value_dim must be a multiple of 32");
}

KdaFinalChunkScanOperation::spec_return_value_t KdaFinalChunkScanOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const auto output_dtype = attrs.output_bf16 ? DataType::BFLOAT16 : DataType::FLOAT32;
    const auto output_layout = TensorLayout(output_dtype, PageConfig(Layout::TILE), attrs.output_mem_config);
    const auto state_layout = TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config);
    const auto output_shape = attrs.summary_pair
                                  ? Shape({attrs.batch_heads, attrs.key_dim, attrs.value_dim})
                                  : Shape({attrs.batch_heads, attrs.num_chunks, attrs.chunk_size, attrs.value_dim});
    return {
        TensorSpec(output_shape, output_layout),
        TensorSpec(Shape({attrs.batch_heads, attrs.key_dim, attrs.value_dim}), state_layout)};
}

KdaFinalChunkScanOperation::tensor_return_value_t KdaFinalChunkScanOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(attrs, in)) {
        outputs.push_back(create_device_tensor(spec, in.v_beta.device()));
    }
    return outputs;
}

std::vector<Tensor> kda_final_chunk_scan(
    const Tensor& v_beta,
    const Tensor& kd,
    const Tensor& q_decay,
    const Tensor& intra,
    const Tensor& k_dec_t,
    const Tensor& final_decay,
    const Tensor& t_inv,
    const std::optional<Tensor>& initial_state,
    uint32_t chunk_size,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool state_only,
    const std::optional<Tensor>& identity_tile,
    bool summary_pair,
    bool output_bf16) {
    const auto& value_shape = v_beta.logical_shape();
    const auto& key_shape = kd.logical_shape();
    return ttnn::device_operation::launch<KdaFinalChunkScanOperation>(
        KdaFinalChunkScanParams{
            .batch_heads = value_shape[0],
            .num_chunks = value_shape[1],
            .chunk_size = chunk_size,
            .key_dim = key_shape[3],
            .value_dim = value_shape[3],
            .identity_initial_state = identity_tile.has_value(),
            .state_only = state_only,
            .summary_pair = summary_pair,
            .output_bf16 = output_bf16,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config},
        KdaFinalChunkScanInputs{
            .v_beta = v_beta,
            .kd = kd,
            .q_decay = q_decay,
            .intra = intra,
            .k_dec_t = k_dec_t,
            .final_decay = final_decay,
            .t_inv = t_inv,
            .initial_state = initial_state,
            .identity_tile = identity_tile});
}

}  // namespace ttnn::prim
