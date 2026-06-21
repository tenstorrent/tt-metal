// SPDX-License-Identifier: Apache-2.0
#include "deltanet_prefill_chunked_device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::deltanet {

DeltaNetPrefillChunkedDeviceOperation::program_factory_t
DeltaNetPrefillChunkedDeviceOperation::select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
    return DeltaNetPrefillChunkedProgramFactory{};
}

void DeltaNetPrefillChunkedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    TT_FATAL(inputs.k.storage_type() == StorageType::DEVICE, "chunked: k must be on device");
    TT_FATAL(inputs.recurrent_state.storage_type() == StorageType::DEVICE, "chunked: state on device");
    TT_FATAL(inputs.k.layout() == Layout::TILE, "chunked: k TILE");
    TT_FATAL(attrs.k_head_dim % 32 == 0 && attrs.v_head_dim % 32 == 0, "chunked: head dims %32");
    TT_FATAL(attrs.chunk == 32, "chunked: C must be 32");
    TT_FATAL(attrs.n_chunks >= 1, "chunked: n_chunks >= 1");
}

void DeltaNetPrefillChunkedDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    validate_on_program_cache_miss(attrs, inputs);
}

DeltaNetPrefillChunkedDeviceOperation::spec_return_value_t
DeltaNetPrefillChunkedDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    auto mem = attrs.output_memory_config;
    auto dtype = inputs.k.dtype();
    uint32_t Sp = attrs.chunk * attrs.n_chunks;
    uint32_t HSp = attrs.num_heads * Sp;
    // output: [Hv*Sp, Dv] (head-major rows; host reshapes to [1,1,S,Hv*Dv])
    auto out_spec = TensorSpec(Shape({HSp, attrs.v_head_dim}), TensorLayout(dtype, Layout::TILE, mem));
    auto state_spec = TensorSpec(inputs.recurrent_state.logical_shape(), TensorLayout(dtype, Layout::TILE, mem));
    return {out_spec, state_spec};
}

DeltaNetPrefillChunkedDeviceOperation::tensor_return_value_t
DeltaNetPrefillChunkedDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& inputs) {
    auto* device = inputs.recurrent_state.device();
    auto specs = compute_output_specs(attrs, inputs);
    return {create_device_tensor(specs[0], device), create_device_tensor(specs[1], device)};
}

}  // namespace ttnn::operations::experimental::deltanet

namespace ttnn::prim {

std::vector<Tensor> deltanet_prefill_chunked(
    const Tensor& k, const Tensor& q, const Tensor& v, const Tensor& z,
    const Tensor& Kdec, const Tensor& KiT, const Tensor& Qd,
    const Tensor& dcol, const Tensor& betacol, const Tensor& dlast,
    const Tensor& recurrent_state, const Tensor& norm_weight,
    uint32_t num_heads, uint32_t k_head_dim, uint32_t v_head_dim,
    uint32_t chunk, uint32_t n_chunks, uint32_t seq_len,
    const std::optional<MemoryConfig>& output_memory_config) {
    using Op = ttnn::operations::experimental::deltanet::DeltaNetPrefillChunkedDeviceOperation;
    auto mem = output_memory_config.value_or(recurrent_state.memory_config());
    auto attrs = Op::operation_attributes_t{
        .num_heads = num_heads, .k_head_dim = k_head_dim, .v_head_dim = v_head_dim,
        .chunk = chunk, .n_chunks = n_chunks, .seq_len = seq_len, .output_memory_config = mem};
    auto args = Op::tensor_args_t{
        .k = k, .q = q, .v = v, .z = z, .Kdec = Kdec, .KiT = KiT, .Qd = Qd,
        .dcol = dcol, .betacol = betacol, .dlast = dlast,
        .recurrent_state = recurrent_state, .norm_weight = norm_weight};
    return ttnn::device_operation::launch<Op>(attrs, args);
}

}  // namespace ttnn::prim
