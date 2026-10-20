// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "dsa_decode_state_device_operation.hpp"

#include <initializer_list>

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

DsaDecodeStateDeviceOperation::program_factory_t DsaDecodeStateDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return DsaDecodeStateProgramFactory{};
}

void DsaDecodeStateDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& in) {
    using namespace tt::constants;
    auto* device = in.stem.device();
    const bool fp32_dest_acc_en = std::get<2>(get_compute_kernel_config_args(device->arch(), a.compute_kernel_config));
    TT_FATAL(fp32_dest_acc_en, "dsa_decode_state: requires fp32_dest_acc_en");
    TT_FATAL(a.B >= 1 && a.B <= 32 * 32, "dsa_decode_state: B ({}) out of range", a.B);
    for (uint32_t v : {a.Lq, a.L, a.D, a.qa_off, a.kva_off, a.k_off, a.g_off}) {
        TT_FATAL(v % TILE_WIDTH == 0, "dsa_decode_state: widths and column offsets must be tile multiples, got {}", v);
    }
    TT_FATAL(a.Lq > 0 && a.L > 0 && a.D > 0, "dsa_decode_state: zero width");
    TT_FATAL(a.P >= 1 && a.P <= TILE_HEIGHT, "dsa_decode_state: P ({}) must be in [1, 32]", a.P);
    TT_FATAL(a.pool_blocks >= 1, "dsa_decode_state: pool_blocks must be >= 1");

    auto check = [&](const Tensor& t, const char* name, DataType dtype, Layout layout, std::initializer_list<int64_t> expected) {
        TT_FATAL(t.storage_type() == StorageType::DEVICE && t.buffer() != nullptr, "dsa_decode_state: {} must be on device", name);
        TT_FATAL(t.device() == device, "dsa_decode_state: {} must be on the same device as stem", name);
        TT_FATAL(!t.is_sharded(), "dsa_decode_state: {} must be interleaved", name);
        TT_FATAL(t.layout() == layout, "dsa_decode_state: {} must be {} layout", name, layout);
        TT_FATAL(t.dtype() == dtype, "dsa_decode_state: {} must be {}, got {}", name, dtype, t.dtype());
        const auto& shape = t.logical_shape();
        TT_FATAL(shape.rank() == expected.size(), "dsa_decode_state: {} must be rank {}, got {}", name, expected.size(), shape.rank());
        size_t i = 0;
        for (int64_t e : expected) {
            // -1: any size (checked separately)
            TT_FATAL(e < 0 || static_cast<int64_t>(shape[i]) == e, "dsa_decode_state: {} dim[{}] must be {}, got {}", name, i, e, shape[i]);
            ++i;
        }
    };
    const int64_t B = a.B;
    check(in.stem, "stem", DataType::BFLOAT16, Layout::TILE, {1, 1, B, -1});
    const auto Ws = in.stem.logical_shape()[3];
    TT_FATAL(a.qa_off + a.Lq <= Ws && a.kva_off + a.L <= Ws && a.k_off + a.D <= Ws && a.g_off + a.D <= Ws,
             "dsa_decode_state: stem column blocks exceed its width {}", Ws);
    check(in.latent_cache, "latent_cache", DataType::BFLOAT16, Layout::TILE, {-1, 1, TILE_HEIGHT, (int64_t)a.L});
    check(in.latent_page_table, "latent_page_table", DataType::INT32, Layout::ROW_MAJOR, {B, -1});
    check(in.positions, "positions", DataType::INT32, Layout::ROW_MAJOR, {B});
    check(in.ring_k, "ring_k", DataType::BFLOAT16, Layout::TILE, {-1, 1, TILE_HEIGHT, (int64_t)a.D});
    check(in.ring_g, "ring_g", DataType::BFLOAT16, Layout::TILE, {-1, 1, TILE_HEIGHT, (int64_t)a.D});
    TT_FATAL(in.ring_k.logical_shape()[0] >= B && in.ring_g.logical_shape()[0] >= B, "dsa_decode_state: rings must hold B blocks");
    check(in.pool_cache, "pool_cache", DataType::BFLOAT16, Layout::TILE, {-1, 1, TILE_HEIGHT, (int64_t)a.D});
    TT_FATAL(in.pool_cache.logical_shape()[0] >= B * (int64_t)a.pool_blocks, "dsa_decode_state: pool cache must hold B * pool_blocks blocks");
    check(in.qa_norm_w, "qa_norm_w", DataType::FLOAT32, Layout::TILE, {1, 1, 1, (int64_t)a.Lq});
    check(in.kva_norm_w, "kva_norm_w", DataType::FLOAT32, Layout::TILE, {1, 1, 1, (int64_t)a.L});
    check(in.k_norm_w, "k_norm_w", DataType::FLOAT32, Layout::TILE, {1, 1, 1, (int64_t)a.D});
    check(in.k_norm_b, "k_norm_b", DataType::FLOAT32, Layout::TILE, {1, 1, 1, (int64_t)a.D});
    check(in.ape, "ape", DataType::FLOAT32, Layout::TILE, {1, 1, TILE_HEIGHT, (int64_t)a.D});
    check(in.selectors, "selectors", DataType::BFLOAT16, Layout::TILE, {33 * TILE_HEIGHT, TILE_WIDTH});
}

DsaDecodeStateDeviceOperation::spec_return_value_t DsaDecodeStateDeviceOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t& in) {
    const auto layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), a.output_mem_config);
    return {
        tt::tt_metal::TensorSpec(ttnn::Shape({1, 1, a.B, a.Lq}), layout),
        in.latent_cache.tensor_spec(),
        in.ring_k.tensor_spec(),
        in.ring_g.tensor_spec(),
        in.pool_cache.tensor_spec()};
}

DsaDecodeStateDeviceOperation::tensor_return_value_t DsaDecodeStateDeviceOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    auto specs = compute_output_specs(a, in);
    return {create_device_tensor(specs[0], in.stem.device()), in.latent_cache, in.ring_k, in.ring_g, in.pool_cache};
}

std::vector<Tensor> dsa_decode_state(
    const DsaDecodeStateInputs& inputs,
    uint32_t qa_off,
    uint32_t kva_off,
    uint32_t k_off,
    uint32_t g_off,
    uint32_t Lq,
    uint32_t L,
    uint32_t D,
    uint32_t P,
    uint32_t pool_blocks,
    float eps_rms,
    float eps_ln,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = DsaDecodeStateDeviceOperation;
    const auto& ss = inputs.stem.logical_shape();
    TT_FATAL(ss.rank() == 4, "dsa_decode_state: stem must be rank 4 [1, 1, B, Ws], got rank {}", ss.rank());
    auto attrs = OperationType::operation_attributes_t{
        .B = static_cast<uint32_t>(ss[2]),
        .Lq = Lq,
        .L = L,
        .D = D,
        .P = P,
        .pool_blocks = pool_blocks,
        .qa_off = qa_off,
        .kva_off = kva_off,
        .k_off = k_off,
        .g_off = g_off,
        .eps_rms = eps_rms,
        .eps_ln = eps_ln,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, inputs);
}

}  // namespace ttnn::prim
