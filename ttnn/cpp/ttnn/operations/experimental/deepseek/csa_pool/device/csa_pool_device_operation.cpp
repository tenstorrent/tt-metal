// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "csa_pool_device_operation.hpp"

#include <cstdint>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/device.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::csa_pool {

using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

constexpr auto kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/csa_pool/device/kernels/dataflow/reader_csa_pool.cpp";
constexpr auto kComputeKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/csa_pool/device/kernels/compute/csa_pool.cpp";

void require_rm_width_sharded(const Tensor& t, const char* name) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "csa_pool_window: {} must be on device", name);
    TT_FATAL(t.layout() == Layout::ROW_MAJOR, "csa_pool_window: {} must be ROW_MAJOR", name);
    TT_FATAL(t.dtype() == DataType::BFLOAT16, "csa_pool_window: {} must be BFLOAT16", name);
    TT_FATAL(
        t.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
            t.memory_config().buffer_type() == BufferType::L1,
        "csa_pool_window: {} must be L1 WIDTH_SHARDED",
        name);
    TT_FATAL(t.memory_config().shard_spec().has_value(), "csa_pool_window: {} needs a shard spec", name);
}

}  // namespace

MemoryConfig default_output_memory_config(const Tensor& prev_kv, uint32_t users, uint32_t /*head_dim*/) {
    const auto& in_spec = prev_kv.memory_config().shard_spec().value();
    const auto cores = corerange_to_cores(in_spec.grid, std::nullopt, /*row_wise=*/true);
    const uint32_t n_out = static_cast<uint32_t>(cores.size() / 2);
    std::vector<CoreRange> ranges;
    ranges.reserve(n_out);
    for (uint32_t i = 0; i < n_out; ++i) {
        ranges.emplace_back(cores[i], cores[i]);
    }
    ShardSpec out_shard(CoreRangeSet(std::move(ranges)), {users, in_spec.shape[1]}, in_spec.orientation);
    return MemoryConfig(TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1, out_shard);
}

CsaPoolDeviceOperation::program_factory_t CsaPoolDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ProgramFactory{};
}

void CsaPoolDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const Tensor* windows[] = {
        &tensor_args.prev_kv, &tensor_args.prev_gate, &tensor_args.win_kv, &tensor_args.win_gate};
    const char* names[] = {"prev_kv", "prev_gate", "win_kv", "win_gate"};
    for (uint32_t i = 0; i < 4; ++i) {
        require_rm_width_sharded(*windows[i], names[i]);
    }
    require_rm_width_sharded(tensor_args.position_bias, "position_bias");

    const auto& prev_kv = tensor_args.prev_kv;
    const auto& bias = tensor_args.position_bias;
    const auto& shape = prev_kv.logical_shape();
    TT_FATAL(shape.rank() == 4, "csa_pool_window: window tensors must be rank-4");
    TT_FATAL(bias.logical_shape().rank() == 4, "csa_pool_window: position_bias must be rank-4");

    const uint32_t feat = static_cast<uint32_t>(shape[-1]);
    const uint32_t rows = static_cast<uint32_t>(shape.volume() / feat);
    TT_FATAL(feat == 2 * args.head_dim, "csa_pool_window: last dim {} must be 2*head_dim={}", feat, args.head_dim);
    TT_FATAL(
        rows == args.users * args.compress_rate,
        "csa_pool_window: window rows {} must equal users {} * compress_rate {}",
        rows,
        args.users,
        args.compress_rate);
    TT_FATAL(
        static_cast<uint32_t>(bias.logical_shape()[-1]) == feat,
        "csa_pool_window: position_bias width {} must match window width {}",
        bias.logical_shape()[-1],
        feat);
    TT_FATAL(
        static_cast<uint32_t>(bias.logical_shape()[-2]) == args.compress_rate,
        "csa_pool_window: position_bias height {} must equal compress_rate {}",
        bias.logical_shape()[-2],
        args.compress_rate);

    for (uint32_t i = 1; i < 4; ++i) {
        TT_FATAL(
            windows[i]->logical_shape() == shape && windows[i]->memory_config() == prev_kv.memory_config(),
            "csa_pool_window: {} must match prev_kv shape and memory config",
            names[i]);
    }

    const auto& shard = prev_kv.memory_config().shard_spec().value();
    const uint32_t num_cores = shard.grid.num_cores();
    const uint32_t shard_width = shard.shape[1];
    const uint32_t shard_height = shard.shape[0];
    TT_FATAL(num_cores % 2 == 0, "csa_pool_window: core count {} must be even (Ca/Cb split)", num_cores);
    TT_FATAL(shard_width % TILE_WIDTH == 0, "csa_pool_window: shard width {} must be tile-aligned", shard_width);
    TT_FATAL(
        shard_width * num_cores == feat,
        "csa_pool_window: shard width {} over {} cores must cover 2*head_dim {}",
        shard_width,
        num_cores,
        feat);
    TT_FATAL(
        args.head_dim % shard_width == 0,
        "csa_pool_window: head_dim {} must be divisible by shard width {} so Ca/Cb do not straddle a core",
        args.head_dim,
        shard_width);
    TT_FATAL(
        shard_height == rows, "csa_pool_window: window shard height {} must equal packed rows {}", shard_height, rows);

    const auto& bias_shard = bias.memory_config().shard_spec().value();
    TT_FATAL(
        bias_shard.grid == shard.grid && bias_shard.shape[1] == shard_width,
        "csa_pool_window: position_bias must share the window core grid and shard width");
    TT_FATAL(
        bias_shard.shape[0] == args.compress_rate,
        "csa_pool_window: position_bias shard height {} must equal compress_rate {}",
        bias_shard.shape[0],
        args.compress_rate);

    const auto& out_mem = args.output_mem_config;
    TT_FATAL(
        out_mem.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED && out_mem.buffer_type() == BufferType::L1 &&
            out_mem.shard_spec().has_value(),
        "csa_pool_window: output must be L1 WIDTH_SHARDED");
    const auto& out_shard = out_mem.shard_spec().value();
    TT_FATAL(
        out_shard.grid.num_cores() == num_cores / 2 && out_shard.shape[1] == shard_width &&
            out_shard.shape[0] == args.users,
        "csa_pool_window: output shard must be {} cores of [{}, {}]",
        num_cores / 2,
        args.users,
        shard_width);
}

CsaPoolDeviceOperation::spec_return_value_t CsaPoolDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& prev_kv = tensor_args.prev_kv;
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, args.users, args.head_dim}),
        tt::tt_metal::TensorLayout(
            prev_kv.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR), args.output_mem_config));
}

CsaPoolDeviceOperation::tensor_return_value_t CsaPoolDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.prev_kv.device());
}

tt::tt_metal::ProgramDescriptor CsaPoolDeviceOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& prev_kv = tensor_args.prev_kv;
    const auto& prev_gate = tensor_args.prev_gate;
    const auto& win_kv = tensor_args.win_kv;
    const auto& win_gate = tensor_args.win_gate;
    const auto& bias = tensor_args.position_bias;

    const tt::DataFormat df = datatype_to_dataformat_converter(prev_kv.dtype());
    const tt::tt_metal::Tile face = rm_face_tile();
    const uint32_t face_bytes = face.get_tile_size(df);
    const std::optional<TileDescriptor> face_desc{TileDescriptor{face}};

    const auto& shard = prev_kv.memory_config().shard_spec().value();
    const uint32_t Wt = shard.shape[1] / TILE_WIDTH;
    const uint32_t in_tiles = args.users * args.compress_rate * Wt;
    const uint32_t bias_tiles = args.compress_rate * Wt;
    const uint32_t out_tiles = args.users * Wt;
    const uint32_t W = 2 * args.compress_rate;

    auto* device = prev_kv.device();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), args.compute_kernel_config);

    const auto cores = corerange_to_cores(shard.grid, std::nullopt, /*row_wise=*/true);
    const uint32_t n_ca = static_cast<uint32_t>(cores.size() / 2);
    std::vector<CoreRange> ca_ranges;
    ca_ranges.reserve(n_ca);
    for (uint32_t i = 0; i < n_ca; ++i) {
        ca_ranges.emplace_back(cores[i], cores[i]);
    }
    const CoreRangeSet ca_cores(std::move(ca_ranges));

    constexpr uint8_t cb_prev_kv = tt::CBIndex::c_0;
    constexpr uint8_t cb_prev_gate = tt::CBIndex::c_1;
    constexpr uint8_t cb_bias_ca = tt::CBIndex::c_2;
    constexpr uint8_t cb_win_kv = tt::CBIndex::c_3;
    constexpr uint8_t cb_win_gate = tt::CBIndex::c_4;
    constexpr uint8_t cb_bias_cb = tt::CBIndex::c_5;
    constexpr uint8_t cb_logits = tt::CBIndex::c_6;
    constexpr uint8_t cb_exp = tt::CBIndex::c_7;
    constexpr uint8_t cb_max = tt::CBIndex::c_8;
    constexpr uint8_t cb_sum = tt::CBIndex::c_9;
    constexpr uint8_t cb_acc = tt::CBIndex::c_24;
    constexpr uint8_t cb_weight = tt::CBIndex::c_25;
    constexpr uint8_t cb_out = tt::CBIndex::c_16;

    auto make_global = [&](uint8_t index, uint32_t tiles, tt::tt_metal::Buffer* buffer) {
        return CBDescriptor{
            .total_size = tiles * face_bytes,
            .core_ranges = ca_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = index, .data_format = df, .page_size = face_bytes, .tile = face_desc}}},
            .buffer = buffer,
        };
    };
    auto make_local = [&](uint8_t index, uint32_t tiles) {
        return CBDescriptor{
            .total_size = tiles * face_bytes,
            .core_ranges = ca_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = index, .data_format = df, .page_size = face_bytes, .tile = face_desc}}},
        };
    };

    tt::tt_metal::ProgramDescriptor desc;
    desc.cbs.push_back(make_global(cb_prev_kv, in_tiles, prev_kv.buffer()));
    desc.cbs.push_back(make_global(cb_prev_gate, in_tiles, prev_gate.buffer()));
    desc.cbs.push_back(make_global(cb_bias_ca, bias_tiles, bias.buffer()));
    desc.cbs.push_back(make_local(cb_win_kv, in_tiles));
    desc.cbs.push_back(make_local(cb_win_gate, in_tiles));
    desc.cbs.push_back(make_local(cb_bias_cb, bias_tiles));
    desc.cbs.push_back(make_local(cb_logits, W));
    desc.cbs.push_back(make_local(cb_exp, W));
    desc.cbs.push_back(make_local(cb_max, 1));
    desc.cbs.push_back(make_local(cb_sum, 1));
    desc.cbs.push_back(make_local(cb_acc, 1));
    desc.cbs.push_back(make_local(cb_weight, 1));
    desc.cbs.push_back(make_global(cb_out, out_tiles, output.buffer()));

    KernelDescriptor::CompileTimeArgs reader_ct = {
        (uint32_t)cb_win_kv,
        (uint32_t)cb_win_gate,
        (uint32_t)cb_bias_cb,
        in_tiles,
        bias_tiles,
        face_bytes,
    };
    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kReaderKernelPath;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = ca_cores;
    reader_desc.compile_time_args = std::move(reader_ct);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor::CompileTimeArgs compute_ct = {
        (uint32_t)cb_prev_kv,
        (uint32_t)cb_prev_gate,
        (uint32_t)cb_bias_ca,
        (uint32_t)cb_win_kv,
        (uint32_t)cb_win_gate,
        (uint32_t)cb_bias_cb,
        (uint32_t)cb_logits,
        (uint32_t)cb_exp,
        (uint32_t)cb_max,
        (uint32_t)cb_sum,
        (uint32_t)cb_acc,
        (uint32_t)cb_weight,
        (uint32_t)cb_out,
        args.users,
        args.compress_rate,
        Wt,
    };
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = kComputeKernelPath;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = ca_cores;
    compute_desc.compile_time_args = std::move(compute_ct);
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };

    reader_desc.runtime_args.reserve(n_ca);
    compute_desc.runtime_args.reserve(n_ca);
    auto* win_kv_buffer = win_kv.buffer();
    auto* win_gate_buffer = win_gate.buffer();
    auto* bias_buffer = bias.buffer();
    for (uint32_t i = 0; i < n_ca; ++i) {
        const auto partner_phys = device->worker_core_from_logical_core(cores[i + n_ca]);
        reader_desc.emplace_runtime_args(
            cores[i],
            {static_cast<uint32_t>(partner_phys.x),
             static_cast<uint32_t>(partner_phys.y),
             win_kv_buffer,
             win_gate_buffer,
             bias_buffer});
        compute_desc.runtime_args.emplace_back(cores[i], KernelDescriptor::CoreRuntimeArgs{});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek::csa_pool

namespace ttnn::prim {

ttnn::Tensor csa_pool_window(
    const ttnn::Tensor& prev_kv,
    const ttnn::Tensor& prev_gate,
    const ttnn::Tensor& win_kv,
    const ttnn::Tensor& win_gate,
    const ttnn::Tensor& position_bias,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::experimental::deepseek::csa_pool::CsaPoolDeviceOperation;

    const auto& shape = prev_kv.logical_shape();
    const uint32_t feat = static_cast<uint32_t>(shape[-1]);
    const uint32_t cr = static_cast<uint32_t>(position_bias.logical_shape()[-2]);
    const uint32_t rows = static_cast<uint32_t>(shape.volume() / feat);
    TT_FATAL(feat % 2 == 0, "csa_pool_window: last dim {} must be even (2*head_dim)", feat);
    TT_FATAL(cr > 0 && rows % cr == 0, "csa_pool_window: packed rows {} must divide by compress_rate {}", rows, cr);
    const uint32_t users = rows / cr;
    const uint32_t head_dim = feat / 2;

    auto kernel_config_val = init_device_compute_kernel_config(
        prev_kv.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/true);

    auto attrs = OperationType::operation_attributes_t{
        .users = users,
        .compress_rate = cr,
        .head_dim = head_dim,
        .output_mem_config = memory_config.value_or(
            ttnn::operations::experimental::deepseek::csa_pool::default_output_memory_config(prev_kv, users, head_dim)),
        .compute_kernel_config = kernel_config_val,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .prev_kv = prev_kv,
        .prev_gate = prev_gate,
        .win_kv = win_kv,
        .win_gate = win_gate,
        .position_bias = position_bias,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
