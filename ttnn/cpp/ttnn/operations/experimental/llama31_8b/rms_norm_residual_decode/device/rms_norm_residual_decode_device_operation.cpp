// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b/rms_norm_residual_decode/device/rms_norm_residual_decode_device_operation.hpp"
#include "ttnn/operations/experimental/llama31_8b/rms_norm_residual_decode/device/rms_norm_residual_decode_table.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <variant>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation.hpp"

namespace ttnn::prim::llama31_8b_rms_norm_residual_decode {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

namespace m2 = tt::tt_metal::experimental;

const m2::DFBSpecName IN0{"in0"};
const m2::DFBSpecName IN1{"in1"};
const m2::DFBSpecName XMM{"xmm"};
const m2::DFBSpecName HOUT{"hout"};
const m2::TensorParamName INPUT{"input"};
const m2::TensorParamName RESIDUAL{"residual"};
const m2::TensorParamName Y_SRC{"y_src"};
const m2::TensorParamName R_SRC{"res_src"};

constexpr const char* KERNEL_DIR =
    "ttnn/cpp/ttnn/operations/experimental/llama31_8b/rms_norm_residual_decode/device/kernels/";

bool source_ends_with(const m2::KernelSpec& kernel, const std::string& tail) {
    const auto* path = std::get_if<std::filesystem::path>(&kernel.source);
    if (path == nullptr) {
        return false;
    }
    const std::string s = path->generic_string();
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

void bind_dfb(m2::KernelSpec& kernel, const m2::DFBSpecName& dfb, std::string accessor_name, m2::DFBEndpointType role) {
    kernel.dfb_bindings.push_back(
        m2::DFBBinding{.dfb_spec_name = dfb, .accessor_name = std::move(accessor_name), .endpoint_type = role});
}

void bind_tensor(m2::KernelSpec& kernel, const m2::TensorParamName& tensor, std::string accessor_name) {
    kernel.tensor_bindings.push_back(
        m2::TensorBinding{.tensor_parameter_name = tensor, .accessor_name = std::move(accessor_name)});
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

using namespace CMAKE_UNIQUE_NAMESPACE;

namespace {

using table::kRows;
using table::kWidth;

bool width_sharded_on(const MemoryConfig& m, const table::Row& row) {
    const CoreCoord grid{row.grid_x, row.grid_y};
    const std::uint32_t shard_w = row.shard_w;
    return m.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED && m.buffer_type() == BufferType::L1 &&
           m.shard_spec().has_value() && m.shard_spec()->orientation == ShardOrientation::ROW_MAJOR &&
           m.shard_spec()->grid == CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1})) &&
           m.shard_spec()->shape == std::array<std::uint32_t, 2>{tt::constants::TILE_HEIGHT, shard_w};
}

bool decode_tensor(const Tensor& t) {
    return t.storage_type() == StorageType::DEVICE && t.dtype() == DataType::BFLOAT16 && t.layout() == Layout::TILE &&
           t.padded_shape() == ttnn::Shape({1, 1, tt::constants::TILE_HEIGHT, kWidth});
}

}  // namespace

bool in_table(
    const Tensor& input,
    const Tensor& residual,
    const Tensor& weight,
    const LayerNormShardedMultiCoreProgramConfig& program_config,
    const MemoryConfig& memory_config) {
    if (!decode_tensor(input) || !decode_tensor(residual) || input.logical_shape() != residual.logical_shape()) {
        return false;
    }
    const auto* device = input.device();
    const auto device_grid = device->compute_with_storage_grid_size();
    if (device->arch() != tt::ARCH::BLACKHOLE || device_grid.x != 11 || device_grid.y != 10) {
        return false;
    }
    if (weight.storage_type() != StorageType::DEVICE || weight.dtype() != DataType::BFLOAT16 ||
        weight.layout() != Layout::ROW_MAJOR || weight.memory_config() != ttnn::DRAM_MEMORY_CONFIG ||
        weight.padded_shape() != ttnn::Shape({1, 1, kWidth / tt::constants::TILE_WIDTH, tt::constants::TILE_WIDTH})) {
        return false;
    }
    if (!width_sharded_on(residual.memory_config(), kRows[table::kResidualRow])) {
        return false;
    }
    if (std::none_of(kRows.begin(), kRows.end(), [&](const table::Row& row) {
            return width_sharded_on(input.memory_config(), row);
        })) {
        return false;
    }
    return std::any_of(kRows.begin(), kRows.end(), [&](const table::Row& row) {
        return program_config.compute_with_storage_grid_size == CoreCoord{row.grid_x, row.grid_y} &&
               program_config.block_w * tt::constants::TILE_WIDTH == row.shard_w &&
               program_config.subblock_w == row.subblock_w && program_config.block_h == 1 && !program_config.inplace &&
               width_sharded_on(memory_config, row);
    });
}

RmsNormResidualDecodeDeviceOperation::program_factory_t RmsNormResidualDecodeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return RmsNormResidualDecodeProgramFactory{};
}

void RmsNormResidualDecodeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input;
    const auto& r = tensor_args.residual;
    for (const Tensor* t : {&a, &r}) {
        TT_FATAL(
            t->storage_type() == StorageType::DEVICE && t->is_allocated(),
            "rms_norm_residual_decode: inputs must be on device");
        TT_FATAL(t->layout() == Layout::TILE, "rms_norm_residual_decode: inputs must be TILE");
        TT_FATAL(t->dtype() == DataType::BFLOAT16, "rms_norm_residual_decode: inputs must be BFLOAT16");
        TT_FATAL(
            t->is_sharded() && t->buffer()->buffer_type() == BufferType::L1,
            "rms_norm_residual_decode: inputs must be L1 sharded");
    }
    TT_FATAL(
        a.logical_shape() == r.logical_shape() && a.padded_shape() == r.padded_shape(),
        "rms_norm_residual_decode: input and residual shapes differ");
    TT_FATAL(
        a.physical_volume() / a.padded_shape()[-1] == tt::constants::TILE_HEIGHT,
        "rms_norm_residual_decode: input must be one tile row");
    const auto& mc = operation_attributes.output_mem_config;
    TT_FATAL(
        mc.is_sharded() && mc.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
            mc.buffer_type() == BufferType::L1 && mc.shard_spec().has_value(),
        "rms_norm_residual_decode: memory_config must be L1 width sharded with a shard spec");
    const auto& pc = operation_attributes.program_config;
    TT_FATAL(
        pc.block_h == 1 && !pc.inplace,
        "rms_norm_residual_decode: program_config must have block_h 1 and inplace false");
    TT_FATAL(
        pc.block_w * tt::constants::TILE_WIDTH == mc.shard_spec()->shape[1] &&
            mc.shard_spec()->shape[0] == tt::constants::TILE_HEIGHT,
        "rms_norm_residual_decode: program_config block_w does not match memory_config shard width");
    TT_FATAL(
        in_table(a, r, tensor_args.weight, pc, mc),
        "rms_norm_residual_decode: shape, sharding, program_config or weight outside the Llama-3.1-8B P100 table; use "
        "ttnn.add + ttnn.rms_norm");
}

void RmsNormResidualDecodeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    TT_FATAL(
        tensor_args.input.is_allocated() && tensor_args.residual.is_allocated(),
        "rms_norm_residual_decode: inputs must be allocated");
}

RmsNormResidualDecodeDeviceOperation::spec_return_value_t RmsNormResidualDecodeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tt::tt_metal::TensorSpec(
        tensor_args.input.logical_shape(),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), operation_attributes.output_mem_config));
}

RmsNormResidualDecodeDeviceOperation::tensor_return_value_t RmsNormResidualDecodeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

ttnn::device_operation::ProgramArtifacts RmsNormResidualDecodeProgramFactory::create_program_artifacts(
    const RmsNormResidualDecodeParams& operation_attributes,
    const RmsNormResidualDecodeInputs& tensor_args,
    Tensor& tensor_return_value) {
    const LayerNormParams ln_params{
        .norm_type = LayerNormType::RMSNORM,
        .distributed_norm_stage = DistributedLayerNormStage::NOT_DISTRIBUTED,
        .eps = operation_attributes.eps,
        .output_mem_config = operation_attributes.output_mem_config,
        .program_config = operation_attributes.program_config,
        .compute_kernel_config = operation_attributes.compute_kernel_config,
        .dtype = std::nullopt,
        .fused_activation = std::nullopt};
    const LayerNormInputs ln_inputs{
        .input = tensor_return_value, .residual_input_tensor = tensor_return_value, .weight = tensor_args.weight};
    auto artifacts = LayerNormShardedProgramFactory::create_program_artifacts(
        ln_params, ln_inputs, tensor_return_value, std::nullopt);
    auto& spec = artifacts.spec;
    auto& run = artifacts.run_params;

    uint32_t tile_bytes = 0;
    uint32_t block_tiles = 0;
    tt::DataFormat in_format = tt::DataFormat::Float16_b;
    for (auto& dfb : spec.dataflow_buffers) {
        if (dfb.unique_id == IN0) {
            tile_bytes = dfb.entry_size;
            block_tiles = dfb.num_entries;
            in_format = dfb.data_format_metadata.value_or(tt::DataFormat::Float16_b);
        }
    }
    TT_FATAL(
        tile_bytes > 0 && block_tiles == operation_attributes.program_config.block_w,
        "rms_norm_residual_decode: unexpected input buffer");
    const auto& rmc = tensor_args.residual.memory_config();
    const auto same_as_residual = [&](const MemoryConfig& m) {
        return m.memory_layout() == rmc.memory_layout() && m.shard_spec().has_value() &&
               m.shard_spec()->grid == rmc.shard_spec()->grid && m.shard_spec()->shape == rmc.shard_spec()->shape;
    };
    const bool in_place =
        same_as_residual(tensor_args.input.memory_config()) && same_as_residual(operation_attributes.output_mem_config);
    for (auto& dfb : spec.dataflow_buffers) {
        if (dfb.unique_id == IN0) {
            dfb.borrowed_from = in_place ? std::optional<m2::TensorParamName>(Y_SRC) : std::nullopt;
        } else if (dfb.unique_id == IN1) {
            dfb.borrowed_from = in_place ? std::optional<m2::TensorParamName>(R_SRC) : std::nullopt;
        } else if (dfb.unique_id == XMM) {
            const uint32_t total = dfb.entry_size * dfb.num_entries;
            dfb.entry_size = tile_bytes;
            dfb.num_entries = total / tile_bytes;
            dfb.data_format_metadata = in_format;
        }
    }
    spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = HOUT, .entry_size = tile_bytes, .num_entries = block_tiles, .data_format_metadata = in_format});

    std::erase_if(spec.tensor_parameters, [](const m2::TensorParameter& p) {
        return p.unique_id == INPUT || p.unique_id == RESIDUAL;
    });
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = Y_SRC, .spec = tensor_args.input.tensor_spec()});
    spec.tensor_parameters.push_back(
        m2::TensorParameter{.unique_id = R_SRC, .spec = tensor_args.residual.tensor_spec()});

    bool saw_compute = false;
    bool saw_writer = false;
    for (auto& kernel : spec.kernels) {
        if (source_ends_with(kernel, "kernels/compute/layernorm_sharded.cpp")) {
            saw_compute = true;
            kernel.source = std::filesystem::path(std::string(KERNEL_DIR) + "compute.cpp");
            std::erase_if(kernel.dfb_bindings, [](const m2::DFBBinding& b) {
                return (b.dfb_spec_name == IN0 || b.dfb_spec_name == IN1) &&
                       b.endpoint_type == m2::DFBEndpointType::PRODUCER;
            });
            bind_dfb(kernel, HOUT, "hout", m2::DFBEndpointType::PRODUCER);
        } else if (
            source_ends_with(kernel, "writer_unary_sharded_ln.cpp") ||
            source_ends_with(kernel, "writer_unary_sharded_ln_rm_gb.cpp")) {
            const bool rm_gamma = source_ends_with(kernel, "writer_unary_sharded_ln_rm_gb.cpp");
            saw_writer = true;
            kernel.source =
                std::filesystem::path(std::string(KERNEL_DIR) + (rm_gamma ? "writer_rm_gb.cpp" : "writer.cpp"));
            bind_dfb(kernel, IN0, "in0", m2::DFBEndpointType::PRODUCER);
            bind_dfb(kernel, IN1, "in1", m2::DFBEndpointType::PRODUCER);
            bind_dfb(kernel, HOUT, "hout", m2::DFBEndpointType::CONSUMER);
            bind_tensor(kernel, Y_SRC, "y_src");
            bind_tensor(kernel, R_SRC, "res_src");
            kernel.compile_time_args.emplace("in_tile_bytes", tile_bytes);
            if (in_place) {
                kernel.compiler_options.defines.emplace("INPUTS_IN_PLACE", "1");
            }
        }
    }
    TT_FATAL(saw_compute && saw_writer, "rms_norm_residual_decode: sharded RMSNorm program has an unexpected layout");

    run.tensor_args.erase(INPUT);
    run.tensor_args.erase(RESIDUAL);
    run.tensor_args.emplace(Y_SRC, tensor_args.input.mesh_tensor());
    run.tensor_args.emplace(R_SRC, tensor_args.residual.mesh_tensor());
    return artifacts;
}

}  // namespace ttnn::prim::llama31_8b_rms_norm_residual_decode
