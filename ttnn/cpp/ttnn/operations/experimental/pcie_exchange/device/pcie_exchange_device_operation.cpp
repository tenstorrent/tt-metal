// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pcie_exchange_device_operation.hpp"

#include <set>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/pcie_p2p.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::pcie_exchange {

namespace {

uint32_t shard_ntiles(const Tensor& t) {
    const auto& ss = t.memory_config().shard_spec().value();
    return (ss.shape[0] / tt::constants::TILE_HEIGHT) * (ss.shape[1] / tt::constants::TILE_WIDTH);
}

// Bytes of one core's shard: whole tiles for TILE layout, rows x cols x element for ROW_MAJOR (gather only).
uint32_t shard_bytes(const Tensor& t) {
    const auto& ss = t.memory_config().shard_spec().value();
    if (t.layout() == Layout::TILE) {
        return shard_ntiles(t) * tt::tile_size(datatype_to_dataformat_converter(t.dtype()));
    }
    return ss.shape[0] * ss.shape[1] * t.element_size();
}

}  // namespace

std::vector<CoreCoord> PcieExchangeDeviceOperation::shard_cores(const Tensor& t) {
    const auto& ss = t.memory_config().shard_spec().value();
    return corerange_to_cores(ss.grid, std::nullopt, ss.orientation == ShardOrientation::ROW_MAJOR);
}

CoreRangeSet PcieExchangeDeviceOperation::gather_output_grid(const Tensor& input, uint32_t num_devices) {
    // Same construction as ttnn.num_cores_to_corerangeset(n, grid, row_wise=True), so the output memory config
    // compares equal to a model-side config built that way (CoreRangeSet equality is by range list).
    const auto grid = input.device()->compute_with_storage_grid_size();
    const uint32_t n = shard_cores(input).size() * num_devices;
    TT_FATAL(n <= grid.x * grid.y, "pcie_exchange: ALL_GATHER needs {} output cores, grid has {}", n, grid.x * grid.y);
    return num_cores_to_corerangeset(n, grid, true);
}

PcieExchangeDeviceOperation::program_factory_t PcieExchangeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return PcieExchangeProgramFactory{};
}

void PcieExchangeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    namespace p2p = tt::tt_metal::experimental::pcie_p2p;
    const auto& in = tensor_args.input;
    const auto& xb = tensor_args.exchange;
    TT_FATAL(p2p::is_setup(), "pcie_exchange: tt_metal pcie_p2p windows are not set up (call pcie_p2p.setup first)");
    TT_FATAL(
        in.storage_type() == StorageType::DEVICE && in.buffer() != nullptr, "pcie_exchange: input must be on device");
    TT_FATAL(
        xb.storage_type() == StorageType::DEVICE && xb.buffer() != nullptr,
        "pcie_exchange: exchange must be on device");
    TT_FATAL(
        args.num_devices == 2, "pcie_exchange: only 2 devices are supported for now, mesh has {}", args.num_devices);
    TT_FATAL(
        in.layout() == Layout::TILE || (in.layout() == Layout::ROW_MAJOR && args.mode == PcieExchangeMode::ALL_GATHER),
        "pcie_exchange: input must be TILE layout (ROW_MAJOR is allowed for ALL_GATHER only)");
    if (args.mode == PcieExchangeMode::ALL_REDUCE) {
        TT_FATAL(
            in.dtype() == DataType::BFLOAT16 || in.dtype() == DataType::BFLOAT8_B,
            "pcie_exchange: ALL_REDUCE input must be bf16 or bfp8 (the add runs on the FPU)");
        TT_FATAL(
            !args.bfp8_payload || (in.dtype() == DataType::BFLOAT16 && in.layout() == Layout::TILE),
            "pcie_exchange: bfp8_payload needs a bf16 TILE input");
    } else {
        TT_FATAL(!args.bfp8_payload, "pcie_exchange: bfp8_payload is only defined for ALL_REDUCE");
        TT_FATAL(
            in.dtype() == DataType::BFLOAT16 || in.dtype() == DataType::BFLOAT8_B ||
                in.dtype() == DataType::BFLOAT4_B || in.dtype() == DataType::UINT32 || in.dtype() == DataType::INT32 ||
                in.dtype() == DataType::FLOAT32,
            "pcie_exchange: ALL_GATHER input must be bf16, bfp8, bfp4, uint32, int32 or float32");
    }
    TT_FATAL(
        xb.dtype() == DataType::BFLOAT16 && xb.layout() == Layout::TILE, "pcie_exchange: exchange must be bf16 TILE");
    TT_FATAL(
        in.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
            in.memory_config().buffer_type() == BufferType::L1,
        "pcie_exchange: input must be L1 width-sharded");
    TT_FATAL(
        xb.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
            xb.memory_config().buffer_type() == BufferType::L1,
        "pcie_exchange: exchange must be L1 width-sharded");
    const auto& iss = in.memory_config().shard_spec().value();
    const auto& xss = xb.memory_config().shard_spec().value();
    TT_FATAL(iss.orientation == ShardOrientation::ROW_MAJOR, "pcie_exchange: input shards must be row-major");
    TT_FATAL(iss.grid == xss.grid, "pcie_exchange: exchange buffer must live on the input's cores");
    TT_FATAL(
        iss.shape[0] == in.padded_shape()[-2] && (in.padded_shape().volume() / in.padded_shape()[-1]) == iss.shape[0],
        "pcie_exchange: input must have exactly one shard per core (shard height {} vs tensor rows {})",
        iss.shape[0],
        in.padded_shape()[-2]);
    if (in.layout() == Layout::TILE) {
        TT_FATAL(
            iss.shape[0] % tt::constants::TILE_HEIGHT == 0 && iss.shape[1] % tt::constants::TILE_WIDTH == 0,
            "pcie_exchange: shard must be tile aligned");
    } else {
        TT_FATAL(
            (iss.shape[1] * in.element_size()) % 16 == 0, "pcie_exchange: row-major shard rows must be 16 B aligned");
    }
    const uint32_t in_shard_bytes = shard_bytes(in);
    const uint32_t xb_shard_bytes = shard_bytes(xb);
    const uint32_t ctrl_bytes = tensor_args.ctrl.has_value() ? 0u : 2048u;
    TT_FATAL(
        xb_shard_bytes >= 2 * in_shard_bytes + ctrl_bytes,
        "pcie_exchange: exchange shard is {} bytes, needs 2 * {} + {} (two landing buffers{})",
        xb_shard_bytes,
        in_shard_bytes,
        ctrl_bytes,
        ctrl_bytes ? " and a control tile" : "");
    if (tensor_args.ctrl.has_value()) {
        const auto& ct = tensor_args.ctrl.value();
        TT_FATAL(
            ct.storage_type() == StorageType::DEVICE && ct.buffer() != nullptr,
            "pcie_exchange: ctrl must be on device");
        TT_FATAL(
            ct.dtype() == DataType::BFLOAT16 && ct.layout() == Layout::TILE, "pcie_exchange: ctrl must be bf16 TILE");
        TT_FATAL(
            ct.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
                ct.memory_config().buffer_type() == BufferType::L1 &&
                ct.memory_config().shard_spec().value().grid == iss.grid,
            "pcie_exchange: ctrl must be L1 width-sharded on the input's cores");
        TT_FATAL(shard_bytes(ct) >= 2048, "pcie_exchange: ctrl shard must be at least one tile");
    }
    const auto cores = shard_cores(in);
    const auto& setup_cores = p2p::cores();
    for (const auto& c : cores) {
        bool found = false;
        for (const auto& s : setup_cores) {
            found |= (s == c);
        }
        TT_FATAL(found, "pcie_exchange: core {} has no PCIe P2P window (not in pcie_p2p.setup cores)", c.str());
    }
    if (args.mode == PcieExchangeMode::ALL_GATHER) {
        // The input cores must be the first N cores of the output grid walk, so device d's shards land on
        // cores [d*N, (d+1)*N) and the result is one row-major width-sharded tensor.
        const auto out_cores = corerange_to_cores(gather_output_grid(in, args.num_devices), std::nullopt, true);
        for (size_t i = 0; i < cores.size(); ++i) {
            TT_FATAL(
                out_cores[i] == cores[i],
                "pcie_exchange: ALL_GATHER input shard {} is on core {}, expected the row-major grid core {}",
                i,
                cores[i].str(),
                out_cores[i].str());
        }
    }
}

void PcieExchangeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

PcieExchangeDeviceOperation::spec_return_value_t PcieExchangeDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& in = tensor_args.input;
    if (args.mode == PcieExchangeMode::ALL_REDUCE) {
        return TensorSpec(in.logical_shape(), TensorLayout(in.dtype(), PageConfig(Layout::TILE), in.memory_config()));
    }
    // Shards are concatenated at tile granularity, so a padded last dim (e.g. [32, 1] in a [32, 32] tile)
    // comes back as [32, 32 * num_devices] with each device's columns starting at its tile boundary.
    auto shape = in.logical_shape();
    shape[-1] = in.padded_shape()[-1] * args.num_devices;
    const auto& iss = in.memory_config().shard_spec().value();
    ShardSpec oss(gather_output_grid(in, args.num_devices), iss.shape, ShardOrientation::ROW_MAJOR);
    MemoryConfig mc(TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1, oss);
    return TensorSpec(shape, TensorLayout(in.dtype(), PageConfig(in.layout()), mc));
}

PcieExchangeDeviceOperation::tensor_return_value_t PcieExchangeDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

ttsl::hash::hash_t PcieExchangeDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<PcieExchangeDeviceOperation>(
        static_cast<uint32_t>(args.mode),
        args.num_devices,
        args.bfp8_payload,
        tensor_args.input.tensor_spec(),
        tensor_args.exchange.tensor_spec(),
        tensor_args.ctrl.has_value(),
        tensor_args.input.device()->shape());
}

}  // namespace ttnn::operations::experimental::pcie_exchange

namespace ttnn::prim {

Tensor pcie_exchange(
    const Tensor& input,
    const Tensor& exchange,
    ttnn::operations::experimental::pcie_exchange::PcieExchangeMode mode,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& ctrl,
    bool bfp8_payload) {
    using Op = ttnn::operations::experimental::pcie_exchange::PcieExchangeDeviceOperation;
    auto* mesh = input.device();
    TT_FATAL(mesh != nullptr, "pcie_exchange: input has no device");
    auto attrs = Op::operation_attributes_t{
        .mode = mode,
        .output_mem_config = output_mem_config.value_or(input.memory_config()),
        .num_devices = static_cast<uint32_t>(mesh->num_devices()),
        .bfp8_payload = bfp8_payload,
    };
    auto tensor_args = Op::tensor_args_t{.input = input, .exchange = exchange, .ctrl = ctrl};
    return ttnn::device_operation::launch<Op>(attrs, tensor_args);
}

}  // namespace ttnn::prim
