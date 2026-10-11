// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "engram_gate_device_operation.hpp"

#include <bit>
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

namespace ttnn::operations::experimental::deepseek::engram_gate {

using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

constexpr auto kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/engram_gate/device/kernels/dataflow/reader_engram_gate.cpp";
constexpr auto kWriterKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/engram_gate/device/kernels/dataflow/writer_engram_gate.cpp";
constexpr auto kComputeKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/engram_gate/device/kernels/compute/engram_gate.cpp";

// The cross-core partials of one row block land as rows of a single 32x32 gather tile.
constexpr uint32_t kMaxCores = TILE_HEIGHT;
constexpr uint32_t kExchangeSemaphoreId = 0;
// |dot| floor before the signed square root (the checkpoint's ``Engram.clamp_value``).
constexpr float kGateClamp = 1e-6f;

void require_width_sharded_l1(const Tensor& t, const char* name) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "engram_gate: {} must be on device", name);
    TT_FATAL(
        t.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
            t.memory_config().buffer_type() == BufferType::L1 && t.memory_config().shard_spec().has_value(),
        "engram_gate: {} must be L1 WIDTH_SHARDED",
        name);
}

std::vector<CoreCoord> shard_cores(const ShardSpec& spec) {
    return corerange_to_cores(spec.grid, std::nullopt, spec.orientation == ShardOrientation::ROW_MAJOR);
}

}  // namespace

EngramGateDeviceOperation::program_factory_t EngramGateDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ProgramFactory{};
}

void EngramGateDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& streams = tensor_args.streams;
    const auto& kv = tensor_args.kv;
    const auto& gate = tensor_args.gate_weight;
    require_width_sharded_l1(streams, "streams");
    require_width_sharded_l1(kv, "kv");
    require_width_sharded_l1(gate, "gate_weight");

    TT_FATAL(
        streams.layout() == Layout::ROW_MAJOR && streams.dtype() == DataType::BFLOAT16,
        "engram_gate: streams must be ROW_MAJOR BFLOAT16");
    TT_FATAL(
        kv.layout() == Layout::ROW_MAJOR && kv.dtype() == DataType::BFLOAT16,
        "engram_gate: kv must be ROW_MAJOR BFLOAT16");
    TT_FATAL(
        gate.layout() == Layout::TILE && gate.dtype() == DataType::FLOAT32,
        "engram_gate: gate_weight must be TILE FLOAT32");

    const auto& shape = streams.logical_shape();
    TT_FATAL(shape.rank() == 4 && shape[1] == 1, "engram_gate: streams must be [B, 1, hc, D], got {}", shape);
    const uint32_t b = shape[0], hc = shape[2], d = shape[3];
    TT_FATAL(TILE_HEIGHT % hc == 0, "engram_gate: hc {} must divide {} (gate_weight row pattern)", hc, TILE_HEIGHT);

    const auto& kv_shape = kv.logical_shape();
    TT_FATAL(
        kv_shape.rank() == 4 && kv_shape[0] == 1 && kv_shape[1] == 1 && kv_shape[2] == b && kv_shape[3] == (hc + 1) * d,
        "engram_gate: kv must be [1, 1, {}, {}], got {}",
        b,
        (hc + 1) * d,
        kv_shape);
    const auto& gate_shape = gate.logical_shape();
    TT_FATAL(
        gate_shape.volume() == TILE_HEIGHT * d && gate_shape[-2] == TILE_HEIGHT && gate_shape[-1] == d,
        "engram_gate: gate_weight must be [1, 1, {}, {}], got {}",
        TILE_HEIGHT,
        d,
        gate_shape);

    const auto& shard = streams.memory_config().shard_spec().value();
    const uint32_t streams_cores = shard.grid.num_cores();
    const uint32_t streams_w = shard.shape[1];
    TT_FATAL(
        streams_w * streams_cores == d,
        "engram_gate: streams shards {} x {} must cover D {}",
        streams_cores,
        streams_w,
        d);
    TT_FATAL(
        shard.shape[0] >= b * hc,
        "engram_gate: streams shard height {} must hold B*hc {} rows",
        shard.shape[0],
        b * hc);

    const auto& kv_shard = kv.memory_config().shard_spec().value();
    TT_FATAL(
        kv_shard.shape[1] % TILE_WIDTH == 0,
        "engram_gate: kv shard width {} must be a multiple of {} (NoC alignment)",
        kv_shard.shape[1],
        TILE_WIDTH);
    TT_FATAL(kv_shard.shape[0] >= b, "engram_gate: kv shard height {} must hold all {} users", kv_shard.shape[0], b);

    // The op computes on gate_weight's cores, each owning a w-wide column slice that lies within
    // one streams shard (so the gate grid may split every streams shard in several).
    const auto& gate_shard = gate.memory_config().shard_spec().value();
    const uint32_t num_cores = gate_shard.grid.num_cores();
    const uint32_t w = gate_shard.shape[1];
    TT_FATAL(num_cores <= kMaxCores, "engram_gate: gate_weight spans {} cores, at most {}", num_cores, kMaxCores);
    TT_FATAL(w % TILE_WIDTH == 0, "engram_gate: gate_weight shard width {} must be a multiple of {}", w, TILE_WIDTH);
    TT_FATAL(
        gate_shard.shape[0] == TILE_HEIGHT && w * num_cores == d && streams_w % w == 0,
        "engram_gate: gate_weight must be sharded [{}, w] with w * cores = D {} and w dividing the streams shard "
        "width {}, got [{}, {}] on {} cores",
        TILE_HEIGHT,
        d,
        streams_w,
        gate_shard.shape[0],
        w,
        num_cores);

    TT_FATAL(
        args.output_mem_config == streams.memory_config(),
        "engram_gate: output memory config must equal the streams memory config");

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(streams.device()->arch(), args.compute_kernel_config);
    TT_FATAL(fp32_dest_acc_en, "engram_gate: needs fp32_dest_acc_en (its tiles are fp32)");
}

EngramGateDeviceOperation::spec_return_value_t EngramGateDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& streams = tensor_args.streams;
    return TensorSpec(
        streams.logical_shape(), TensorLayout(streams.dtype(), PageConfig(Layout::ROW_MAJOR), args.output_mem_config));
}

EngramGateDeviceOperation::tensor_return_value_t EngramGateDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.streams.device());
}

ProgramDescriptor EngramGateDeviceOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& streams = tensor_args.streams;
    const auto& kv = tensor_args.kv;
    const auto& gate = tensor_args.gate_weight;
    auto* device = streams.device();

    const auto& shape = streams.logical_shape();
    const uint32_t hc = shape[2], d = shape[3];
    const uint32_t rows = shape[0] * hc;
    const uint32_t Rt = tt::div_up(rows, TILE_HEIGHT);

    // Compute on gate_weight's cores; core c's columns [c*w, (c+1)*w) sit in streams shard c*w / streams_w.
    const auto& gate_shard = gate.memory_config().shard_spec().value();
    const uint32_t w = gate_shard.shape[1];
    const uint32_t Wt = w / TILE_WIDTH;
    const auto cores = shard_cores(gate_shard);
    const uint32_t num_cores = static_cast<uint32_t>(cores.size());
    const CoreRangeSet& all_cores = gate_shard.grid;

    const auto& streams_shard = streams.memory_config().shard_spec().value();
    const uint32_t streams_w = streams_shard.shape[1];
    const auto streams_cores = shard_cores(streams_shard);

    const auto& kv_shard = kv.memory_config().shard_spec().value();
    const auto kv_cores = shard_cores(kv_shard);

    const tt::DataFormat rm_df = datatype_to_dataformat_converter(streams.dtype());
    const tt::DataFormat fp32_df = tt::DataFormat::Float32;
    const uint32_t rm_tile_bytes = tt::tile_size(rm_df);
    const uint32_t fp32_tile_bytes = tt::tile_size(fp32_df);
    const uint32_t elem_bytes = streams.element_size();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), args.compute_kernel_config);

    constexpr uint8_t cb_x_rm = tt::CBIndex::c_0;
    constexpr uint8_t cb_k_rm = tt::CBIndex::c_1;
    constexpr uint8_t cb_v_rm = tt::CBIndex::c_2;
    constexpr uint8_t cb_gate = tt::CBIndex::c_3;
    constexpr uint8_t cb_scaler = tt::CBIndex::c_4;
    constexpr uint8_t cb_xt = tt::CBIndex::c_5;
    constexpr uint8_t cb_kt = tt::CBIndex::c_6;
    constexpr uint8_t cb_vt = tt::CBIndex::c_7;
    constexpr uint8_t cb_prod = tt::CBIndex::c_8;
    constexpr uint8_t cb_col = tt::CBIndex::c_9;
    constexpr uint8_t cb_partial = tt::CBIndex::c_10;
    constexpr uint8_t cb_gather = tt::CBIndex::c_11;
    constexpr uint8_t cb_stats = tt::CBIndex::c_12;
    constexpr uint8_t cb_gate_row = tt::CBIndex::c_13;
    constexpr uint8_t cb_gate_col = tt::CBIndex::c_14;
    constexpr uint8_t cb_out_rm = tt::CBIndex::c_16;

    auto make_cb = [&](uint8_t index, tt::DataFormat df, uint32_t page_bytes, uint32_t pages, Buffer* buffer) {
        return CBDescriptor{
            .total_size = pages * page_bytes,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = index, .data_format = df, .page_size = page_bytes}}},
            .buffer = buffer,
        };
    };

    ProgramDescriptor desc;
    // Row-major blocks are padded to whole 32-row tiles so tilize never reads past the CB.
    desc.cbs.push_back(make_cb(cb_x_rm, rm_df, rm_tile_bytes, Rt * Wt, nullptr));
    desc.cbs.push_back(make_cb(cb_k_rm, rm_df, rm_tile_bytes, Rt * Wt, nullptr));
    desc.cbs.push_back(make_cb(cb_v_rm, rm_df, rm_tile_bytes, Rt * Wt, nullptr));
    desc.cbs.push_back(make_cb(cb_gate, fp32_df, fp32_tile_bytes, Wt, gate.buffer()));
    desc.cbs.push_back(make_cb(cb_scaler, fp32_df, fp32_tile_bytes, 1, nullptr));
    desc.cbs.push_back(make_cb(cb_xt, fp32_df, fp32_tile_bytes, Wt, nullptr));
    desc.cbs.push_back(make_cb(cb_kt, fp32_df, fp32_tile_bytes, Wt, nullptr));
    desc.cbs.push_back(make_cb(cb_vt, fp32_df, fp32_tile_bytes, Wt, nullptr));
    desc.cbs.push_back(make_cb(cb_prod, fp32_df, fp32_tile_bytes, Wt, nullptr));
    desc.cbs.push_back(make_cb(cb_col, fp32_df, fp32_tile_bytes, 1, nullptr));
    desc.cbs.push_back(make_cb(cb_partial, fp32_df, fp32_tile_bytes, 3 * Rt, nullptr));
    desc.cbs.push_back(make_cb(cb_gather, fp32_df, fp32_tile_bytes, 3 * Rt, nullptr));
    desc.cbs.push_back(make_cb(cb_stats, fp32_df, fp32_tile_bytes, 3, nullptr));
    desc.cbs.push_back(make_cb(cb_gate_row, fp32_df, fp32_tile_bytes, 1, nullptr));
    desc.cbs.push_back(make_cb(cb_gate_col, fp32_df, fp32_tile_bytes, 1, nullptr));
    desc.cbs.push_back(make_cb(cb_out_rm, rm_df, rm_tile_bytes, 2 * Wt, nullptr));

    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = kExchangeSemaphoreId,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = all_cores,
        .initial_value = 0,
    });

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kReaderKernelPath;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = {
        cb_x_rm,
        cb_k_rm,
        cb_v_rm,
        cb_scaler,
        rows,
        Rt * Wt,
        w,
        elem_bytes,
        hc,
        d,
        kv_shard.shape[1],
        static_cast<uint32_t>(kv_cores.size()),
        streams_w * elem_bytes,
    };
    reader_desc.config = ReaderConfigDescriptor{};
    for (const auto& core : kv_cores) {
        const auto phys = device->worker_core_from_logical_core(core);
        reader_desc.common_runtime_args.push_back(phys.x);
        reader_desc.common_runtime_args.push_back(phys.y);
    }

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterKernelPath;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = {
        cb_partial,
        cb_gather,
        cb_out_rm,
        Rt,
        Wt,
        rows,
        w * elem_bytes,
        num_cores,
        kExchangeSemaphoreId,
        streams_w * elem_bytes,
    };
    writer_desc.config = WriterConfigDescriptor{};
    for (const auto& core : cores) {
        const auto phys = device->worker_core_from_logical_core(core);
        writer_desc.common_runtime_args.push_back(phys.x);
        writer_desc.common_runtime_args.push_back(phys.y);
    }

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = kComputeKernelPath;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = {
        cb_x_rm,
        cb_k_rm,
        cb_v_rm,
        cb_gate,
        cb_scaler,
        cb_xt,
        cb_kt,
        cb_vt,
        cb_prod,
        cb_col,
        cb_partial,
        cb_gather,
        cb_stats,
        cb_gate_row,
        cb_gate_col,
        cb_out_rm,
        Rt,
        Wt,
        std::bit_cast<uint32_t>(1.0f / static_cast<float>(d)),
        std::bit_cast<uint32_t>(args.eps),
        std::bit_cast<uint32_t>(kGateClamp),
    };
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };

    auto* streams_buffer = streams.buffer();
    auto* kv_buffer = kv.buffer();
    auto* out_buffer = output.buffer();
    for (uint32_t c = 0; c < num_cores; ++c) {
        // The streams / output shard holding core c's columns, and their byte offset in its rows.
        const auto home = device->worker_core_from_logical_core(streams_cores[(c * w) / streams_w]);
        const auto home_x = static_cast<uint32_t>(home.x);
        const auto home_y = static_cast<uint32_t>(home.y);
        const uint32_t home_offset = ((c * w) % streams_w) * elem_bytes;
        reader_desc.emplace_runtime_args(cores[c], {c, streams_buffer, kv_buffer, home_x, home_y, home_offset});
        writer_desc.emplace_runtime_args(cores[c], {c, out_buffer, home_x, home_y, home_offset});
        compute_desc.runtime_args.emplace_back(cores[c], KernelDescriptor::CoreRuntimeArgs{});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek::engram_gate

namespace ttnn::prim {

ttnn::Tensor engram_gate(
    const ttnn::Tensor& streams,
    const ttnn::Tensor& kv,
    const ttnn::Tensor& gate_weight,
    float eps,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::experimental::deepseek::engram_gate::EngramGateDeviceOperation;

    auto kernel_config_val = init_device_compute_kernel_config(
        streams.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);

    auto attrs = OperationType::operation_attributes_t{
        .eps = eps,
        .output_mem_config = memory_config.value_or(streams.memory_config()),
        .compute_kernel_config = kernel_config_val,
    };
    auto tensor_args = OperationType::tensor_args_t{.streams = streams, .kv = kv, .gate_weight = gate_weight};
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
