// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_prefill_query_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {

void check_shape(const Tensor& t, std::initializer_list<uint32_t> expected, const std::string& name) {
    const auto& s = t.logical_shape();
    TT_FATAL(
        static_cast<size_t>(s.rank()) == expected.size(),
        "{} rank mismatch: got {} expected {}",
        name,
        s.rank(),
        expected.size());
    size_t i = 0;
    for (auto e : expected) {
        TT_FATAL(static_cast<uint32_t>(s[i]) == e, "{} dim[{}] expected {} got {}", name, i, e, s[i]);
        ++i;
    }
}

void check_device_tiled(const Tensor& t, const std::string& name) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(t.buffer() != nullptr, "{} must be allocated", name);
    TT_FATAL(t.layout() == Layout::TILE, "{} must be TILE layout", name);
}

}  // namespace

void GatedDeltaPrefillQueryDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const uint32_t Nk = attrs.num_k_heads;
    const uint32_t Nv = attrs.num_v_heads;
    const uint32_t S = attrs.seq_len;
    const uint32_t d = attrs.head_dim;

    TT_FATAL(Nk > 0 && Nv > 0 && S > 0 && d > 0, "num_k_heads/num_v_heads/seq_len/head_dim must all be > 0");
    TT_FATAL(Nv % Nk == 0, "num_v_heads ({}) must be a multiple of num_k_heads ({}) for GVA expansion", Nv, Nk);
    TT_FATAL(d % TILE_WIDTH == 0, "head_dim ({}) must be a multiple of {}", d, TILE_WIDTH);

    // q is row-major; k/v/gate/decay/state are tiled.
    TT_FATAL(in.q.storage_type() == StorageType::DEVICE, "q must be on device");
    TT_FATAL(in.q.layout() == Layout::ROW_MAJOR, "q must be ROW_MAJOR layout");
    check_device_tiled(in.k, "k");
    check_device_tiled(in.v, "v");
    check_device_tiled(in.gate, "gate");
    check_device_tiled(in.decay, "decay");
    check_device_tiled(in.state, "state");

    TT_FATAL(in.state.dtype() == DataType::FLOAT32, "state must be float32, got {}", in.state.dtype());
    TT_FATAL(in.gate.dtype() == DataType::FLOAT32, "gate must be float32, got {}", in.gate.dtype());
    TT_FATAL(in.decay.dtype() == DataType::FLOAT32, "decay must be float32, got {}", in.decay.dtype());

    check_shape(in.q, {1, 1, Nk, d}, "q");
    check_shape(in.k, {1, Nk, S, d}, "k");
    check_shape(in.v, {1, Nv, S, d}, "v");
    check_shape(in.gate, {1, Nv, S, 1}, "gate");
    check_shape(in.decay, {1, Nv, S, 1}, "decay");
    check_shape(in.state, {1, Nv, d, d}, "state");
}

GatedDeltaPrefillQueryDeviceOperation::spec_return_value_t GatedDeltaPrefillQueryDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& /*in*/) {
    const uint32_t Nv = attrs.num_v_heads;
    const uint32_t d = attrs.head_dim;
    const auto& mc = attrs.output_mem_config;

    // O: first output token, per V-head — [1, 1, Nv, d], bf16, TILE.
    tt::tt_metal::TensorSpec o_spec(
        ttnn::Shape({1, 1, Nv, d}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mc));
    // S': updated recurrent state — [1, Nv, d, d], fp32, TILE.
    tt::tt_metal::TensorSpec state_spec(
        ttnn::Shape({1, Nv, d, d}), TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), mc));
    return {o_spec, state_spec};
}

GatedDeltaPrefillQueryDeviceOperation::tensor_return_value_t
GatedDeltaPrefillQueryDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    auto* device = in.state.device();
    return {
        create_device_tensor(specs[0], device),
        create_device_tensor(specs[1], device),
    };
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> gated_delta_prefill_query(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& gate,
    const Tensor& decay,
    const Tensor& state,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using Op = ttnn::experimental::prim::GatedDeltaPrefillQueryDeviceOperation;

    const auto& k_shape = k.logical_shape();  // [1, Nk, S, d]
    const auto& v_shape = v.logical_shape();  // [1, Nv, S, d]

    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            .num_k_heads = static_cast<uint32_t>(k_shape[1]),
            .num_v_heads = static_cast<uint32_t>(v_shape[1]),
            .seq_len = static_cast<uint32_t>(k_shape[2]),
            .head_dim = static_cast<uint32_t>(k_shape[3]),
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config,
        },
        Op::tensor_args_t{
            .q = q,
            .k = k,
            .v = v,
            .gate = gate,
            .decay = decay,
            .state = state,
        });
}

}  // namespace ttnn::prim
