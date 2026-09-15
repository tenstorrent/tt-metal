// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ltx_rope_materialize_device_operation.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/mesh_ring_plan.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::experimental::transformer::ltx_rope_materialize {

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

constexpr auto kWriterPath =
    "ttnn/cpp/ttnn/operations/experimental/transformer/ltx_rope_materialize/device/kernels/dataflow/"
    "writer_ltx_rope_materialize.cpp";
constexpr uint32_t kCompactSelfRates = 682;
constexpr uint32_t kCompactCrossRates = 1024;
constexpr uint32_t kMetadataElements = 4;
constexpr uint32_t kScratchEntries = 16;
constexpr uint32_t kMetadataBytes = 16;

const DFBSpecName OUTPUT_DFB{"output"};
const DFBSpecName CACHE_DFB{"cache"};
const DFBSpecName META_DFB{"meta"};
const KernelSpecName WRITER{"writer"};

const TensorParamName SELF_COS_INPUT{"self_cos_input"};
const TensorParamName SELF_SIN_INPUT{"self_sin_input"};
const TensorParamName CROSS_COS_INPUT{"cross_cos_input"};
const TensorParamName CROSS_SIN_INPUT{"cross_sin_input"};
const TensorParamName METADATA{"metadata"};
const TensorParamName SELF_COS_OUTPUT{"self_cos_output"};
const TensorParamName SELF_SIN_OUTPUT{"self_sin_output"};
const TensorParamName CROSS_COS_OUTPUT{"cross_cos_output"};
const TensorParamName CROSS_SIN_OUTPUT{"cross_sin_output"};

std::array<std::pair<const Tensor*, const char*>, 9> named_tensors(
    const LtxRopeMaterializeDeviceOperation::tensor_args_t& a) {
    return {{
        {&a.compact_self_cos, "compact_self_cos"},
        {&a.compact_self_sin, "compact_self_sin"},
        {&a.compact_cross_cos, "compact_cross_cos"},
        {&a.compact_cross_sin, "compact_cross_sin"},
        {&a.metadata, "metadata"},
        {&a.self_cos_output, "self_cos_output"},
        {&a.self_sin_output, "self_sin_output"},
        {&a.cross_cos_output, "cross_cos_output"},
        {&a.cross_sin_output, "cross_sin_output"},
    }};
}

void validate_common(
    const LtxRopeMaterializeDeviceOperation::operation_attributes_t& attrs,
    const LtxRopeMaterializeDeviceOperation::tensor_args_t& a) {
    TT_FATAL(attrs.sp_axis < 2 && attrs.tp_axis < 2 && attrs.sp_axis != attrs.tp_axis,
             "ltx_rope_materialize requires distinct sp_axis and tp_axis values in {{0, 1}}, got {} and {}",
             attrs.sp_axis,
             attrs.tp_axis);
    for (const auto& [tensor, name] : named_tensors(a)) {
        TT_FATAL(tensor->storage_type() == StorageType::DEVICE, "{} must be on device", name);
        TT_FATAL(tensor->buffer() != nullptr, "{} must have an allocated device buffer", name);
        TT_FATAL(tensor->device() == a.self_cos_output.device(), "{} must be on the output mesh device", name);
    }

    const auto& view = a.self_cos_output.device()->get_view();
    TT_FATAL(view.is_mesh_2d(), "ltx_rope_materialize requires a 2D mesh");
}

void validate_output_pair(
    const Tensor& cos,
    const Tensor& sin,
    const char* name,
    uint32_t head_dim,
    uint32_t local_heads,
    uint32_t sp_factor) {
    TT_FATAL(cos.tensor_spec() == sin.tensor_spec(), "{} cos and sin outputs must have identical specs", name);
    TT_FATAL(cos.layout() == Layout::TILE && cos.dtype() == DataType::BFLOAT16,
             "{} outputs must be TILE BF16 tensors",
             name);
    TT_FATAL(!cos.memory_config().is_sharded(), "{} outputs must be interleaved within each device", name);
    const auto& shape = cos.logical_shape();
    TT_FATAL(shape.rank() == 4 && shape[0] == 1 && shape[1] == local_heads && shape[3] == head_dim,
             "{} local output must have shape [1, {}, video_N/{}, {}], got {}",
             name,
             local_heads,
             sp_factor,
             head_dim,
             shape);
    TT_FATAL(shape[2] % TILE_HEIGHT == 0, "{} video_N must be tile-aligned, got {}", name, shape[2]);
}

void validate_compact_pair(
    const Tensor& cos, const Tensor& sin, const char* name, uint32_t axes, uint32_t minimum_rates) {
    TT_FATAL(cos.tensor_spec() == sin.tensor_spec(), "{} compact cos and sin tensors must have identical specs", name);
    TT_FATAL(cos.layout() == Layout::TILE && cos.dtype() == DataType::FLOAT32,
             "{} compact tensors must be TILE FP32",
             name);
    TT_FATAL(!cos.memory_config().is_sharded(), "{} compact tensors must be interleaved within each device", name);
    const auto& shape = cos.logical_shape();
    TT_FATAL(shape.rank() == 4 && shape[0] == 1 && shape[1] == axes && shape[2] > 0 &&
                 shape[3] >= minimum_rates,
             "{} compact tensor must have shape [1, {}, positions, >= {}], got {}",
             name,
             axes,
             minimum_rates,
             shape);
}

void validate_structure(
    const LtxRopeMaterializeDeviceOperation::operation_attributes_t& attrs,
    const LtxRopeMaterializeDeviceOperation::tensor_args_t& a) {
    validate_common(attrs, a);
    const auto& view = a.self_cos_output.device()->get_view();
    const uint32_t sp_factor = attrs.sp_axis == 0 ? view.num_rows() : view.num_cols();
    const uint32_t tp_factor = attrs.tp_axis == 0 ? view.num_rows() : view.num_cols();
    validate_compact_pair(a.compact_self_cos, a.compact_self_sin, "self", 3, kCompactSelfRates);
    validate_compact_pair(a.compact_cross_cos, a.compact_cross_sin, "cross", 1, kCompactCrossRates);
    TT_FATAL(32 % tp_factor == 0, "32 attention heads must be divisible by TP factor {}", tp_factor);
    validate_output_pair(a.self_cos_output, a.self_sin_output, "self", 128, 32 / tp_factor, sp_factor);
    validate_output_pair(a.cross_cos_output, a.cross_sin_output, "cross", 64, 32 / tp_factor, sp_factor);
    TT_FATAL(a.self_cos_output.logical_shape()[2] == a.cross_cos_output.logical_shape()[2],
             "self and cross outputs must have the same global video_N");

    TT_FATAL(a.metadata.dtype() == DataType::UINT32 && a.metadata.layout() == Layout::ROW_MAJOR,
             "metadata must be a UINT32 ROW_MAJOR tensor");
    TT_FATAL(a.metadata.logical_volume() == kMetadataElements,
             "metadata must contain [video_N_real, latent_frames, latent_height, latent_width] ({} elements), got {}",
             kMetadataElements,
             a.metadata.logical_volume());
    TT_FATAL(a.metadata.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
                 a.metadata.memory_config().buffer_type() == BufferType::DRAM,
             "metadata must be an interleaved DRAM tensor");

    for (const Tensor* output : {
             &a.self_cos_output, &a.self_sin_output, &a.cross_cos_output, &a.cross_sin_output}) {
        TT_FATAL(
            ttnn::operations::ccl::common::tensor_dim_shard_factor(*output, 2) == sp_factor,
            "output sequence dimension must be sharded across sp_axis {} (factor {})",
            attrs.sp_axis,
            sp_factor);
        TT_FATAL(
            ttnn::operations::ccl::common::tensor_dim_shard_factor(*output, 1) == tp_factor,
            "output head dimension must be sharded across tp_axis {} (factor {})",
            attrs.tp_axis,
            tp_factor);
    }
    TT_FATAL(
        a.self_cos_output.logical_shape()[2] % TILE_HEIGHT == 0,
        "local video_N shard {} must be divisible by TILE_HEIGHT ({})",
        a.self_cos_output.logical_shape()[2],
        TILE_HEIGHT);

    for (const Tensor* input : {
             &a.compact_self_cos, &a.compact_self_sin, &a.compact_cross_cos, &a.compact_cross_sin, &a.metadata}) {
        for (uint32_t dim = 0; dim < input->logical_shape().rank(); ++dim) {
            TT_FATAL(
                ttnn::operations::ccl::common::tensor_dim_shard_factor(*input, dim) == 1,
                "compact inputs and metadata must be replicated across the mesh");
        }
    }
}

}  // namespace

LtxRopeMaterializeDeviceOperation::program_factory_t LtxRopeMaterializeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MeshWorkloadFactory{};
}

void LtxRopeMaterializeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_structure(attrs, tensor_args);
}

void LtxRopeMaterializeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_common(attrs, tensor_args);
}

LtxRopeMaterializeDeviceOperation::spec_return_value_t LtxRopeMaterializeDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& a) {
    return {
        a.self_cos_output.tensor_spec(),
        a.self_sin_output.tensor_spec(),
        a.cross_cos_output.tensor_spec(),
        a.cross_sin_output.tensor_spec()};
}

LtxRopeMaterializeDeviceOperation::tensor_return_value_t LtxRopeMaterializeDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& a) {
    return {a.self_cos_output, a.self_sin_output, a.cross_cos_output, a.cross_sin_output};
}

ttsl::hash::hash_t LtxRopeMaterializeDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& a) {
    return tt::tt_metal::operation::hash_operation<LtxRopeMaterializeDeviceOperation>(
        attrs.sp_axis,
        attrs.tp_axis,
        a.compact_self_cos.tensor_spec(),
        a.compact_self_sin.tensor_spec(),
        a.compact_cross_cos.tensor_spec(),
        a.compact_cross_sin.tensor_spec(),
        a.metadata.tensor_spec(),
        a.self_cos_output.tensor_spec(),
        a.self_sin_output.tensor_spec(),
        a.cross_cos_output.tensor_spec(),
        a.cross_sin_output.tensor_spec());
}

LtxRopeMaterializeDeviceOperation::MeshWorkloadFactory::cached_program_t
LtxRopeMaterializeDeviceOperation::MeshWorkloadFactory::create_at(
    const operation_attributes_t& attrs,
    const ttnn::MeshCoordinate& coord,
    const tensor_args_t& a,
    tensor_return_value_t& outputs) {
    auto* mesh_device = a.self_cos_output.device();
    const uint32_t sp_coord =
        ::ttnn::ccl::get_linearized_index_from_physical_coord(a.self_cos_output, coord, attrs.sp_axis);
    const uint32_t tp_coord =
        ::ttnn::ccl::get_linearized_index_from_physical_coord(a.self_cos_output, coord, attrs.tp_axis);

    const auto& self_in = a.compact_self_cos.mesh_tensor();
    const auto& cross_in = a.compact_cross_cos.mesh_tensor();
    const auto& self_out = std::get<0>(outputs).mesh_tensor();
    const auto& cross_out = std::get<2>(outputs).mesh_tensor();

    const uint32_t self_position_t = self_in.padded_shape()[2] / TILE_HEIGHT;
    const uint32_t self_rate_t = self_in.padded_shape()[3] / TILE_WIDTH;
    const uint32_t cross_position_t = cross_in.padded_shape()[2] / TILE_HEIGHT;
    const uint32_t cross_rate_t = cross_in.padded_shape()[3] / TILE_WIDTH;

    const uint32_t self_heads = self_out.padded_shape()[1];
    const uint32_t self_seq_t = self_out.padded_shape()[2] / TILE_HEIGHT;
    const uint32_t self_dim_t = self_out.padded_shape()[3] / TILE_WIDTH;
    const uint32_t cross_heads = cross_out.padded_shape()[1];
    const uint32_t cross_seq_t = cross_out.padded_shape()[2] / TILE_HEIGHT;
    const uint32_t cross_dim_t = cross_out.padded_shape()[3] / TILE_WIDTH;
    const uint32_t self_tiles = self_heads * self_seq_t * self_dim_t;
    const uint32_t cross_tiles = cross_heads * cross_seq_t * cross_dim_t;
    const uint32_t total_tiles = 2 * self_tiles + 2 * cross_tiles;

    const auto grid = mesh_device->compute_with_storage_grid_size();
    const uint32_t grid_cores = grid.x * grid.y;
    const uint32_t num_cores = std::min(total_tiles, grid_cores);
    const auto cores = grid_to_cores(num_cores, grid.x, grid.y, true);
    const auto all_cores = num_cores_to_corerangeset(num_cores, grid, true);

    const auto input_format = datatype_to_dataformat_converter(DataType::FLOAT32);
    const auto output_format = datatype_to_dataformat_converter(DataType::BFLOAT16);
    const uint32_t input_tile_bytes = tt::tile_size(input_format);
    const uint32_t output_tile_bytes = tt::tile_size(output_format);

    std::vector<DataflowBufferSpec> dfbs = {
        DataflowBufferSpec{
            .unique_id = OUTPUT_DFB,
            .entry_size = output_tile_bytes,
            .num_entries = 1,
            .data_format_metadata = output_format},
        DataflowBufferSpec{
            .unique_id = CACHE_DFB,
            .entry_size = input_tile_bytes,
            .num_entries = kScratchEntries,
            .data_format_metadata = input_format},
        DataflowBufferSpec{
            .unique_id = META_DFB,
            .entry_size = kMetadataBytes,
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::UInt32},
    };

    const auto& self_cos_out = std::get<0>(outputs).mesh_tensor();
    const auto& self_sin_out = std::get<1>(outputs).mesh_tensor();
    const auto& cross_cos_out = std::get<2>(outputs).mesh_tensor();
    const auto& cross_sin_out = std::get<3>(outputs).mesh_tensor();
    std::vector<TensorParameter> tensors = {
        TensorParameter{.unique_id = SELF_COS_INPUT, .spec = a.compact_self_cos.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = SELF_SIN_INPUT, .spec = a.compact_self_sin.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = CROSS_COS_INPUT, .spec = a.compact_cross_cos.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = CROSS_SIN_INPUT, .spec = a.compact_cross_sin.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = METADATA, .spec = a.metadata.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = SELF_COS_OUTPUT, .spec = self_cos_out.tensor_spec()},
        TensorParameter{.unique_id = SELF_SIN_OUTPUT, .spec = self_sin_out.tensor_spec()},
        TensorParameter{.unique_id = CROSS_COS_OUTPUT, .spec = cross_cos_out.tensor_spec()},
        TensorParameter{.unique_id = CROSS_SIN_OUTPUT, .spec = cross_sin_out.tensor_spec()},
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path{kWriterPath},
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = OUTPUT_DFB, .accessor_name = "output", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = OUTPUT_DFB, .accessor_name = "output", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = CACHE_DFB, .accessor_name = "cache", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = CACHE_DFB, .accessor_name = "cache", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = META_DFB, .accessor_name = "meta", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = META_DFB, .accessor_name = "meta", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = SELF_COS_INPUT, .accessor_name = "self_cos_input"},
                TensorBinding{.tensor_parameter_name = SELF_SIN_INPUT, .accessor_name = "self_sin_input"},
                TensorBinding{.tensor_parameter_name = CROSS_COS_INPUT, .accessor_name = "cross_cos_input"},
                TensorBinding{.tensor_parameter_name = CROSS_SIN_INPUT, .accessor_name = "cross_sin_input"},
                TensorBinding{.tensor_parameter_name = METADATA, .accessor_name = "metadata"},
                TensorBinding{.tensor_parameter_name = SELF_COS_OUTPUT, .accessor_name = "self_cos_output"},
                TensorBinding{.tensor_parameter_name = SELF_SIN_OUTPUT, .accessor_name = "self_sin_output"},
                TensorBinding{.tensor_parameter_name = CROSS_COS_OUTPUT, .accessor_name = "cross_cos_output"},
                TensorBinding{.tensor_parameter_name = CROSS_SIN_OUTPUT, .accessor_name = "cross_sin_output"},
            },
        .compile_time_args =
            {
                {"self_position_t", self_position_t},
                {"self_rate_t", self_rate_t},
                {"cross_position_t", cross_position_t},
                {"cross_rate_t", cross_rate_t},
                {"self_heads", self_heads},
                {"self_seq_t", self_seq_t},
                {"self_dim_t", self_dim_t},
                {"cross_heads", cross_heads},
                {"cross_seq_t", cross_seq_t},
                {"cross_dim_t", cross_dim_t},
                {"self_tiles", self_tiles},
                {"cross_tiles", cross_tiles},
                {"sp_coord", sp_coord},
                {"tp_coord", tp_coord},
                {"scratch_entries", kScratchEntries},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"start_tile", "num_tiles"}},
        .hw_config = create_writer_datamovement_config(mesh_device->arch())};

    KernelRunArgs writer_run{.kernel = WRITER};
    const uint32_t base = total_tiles / num_cores;
    const uint32_t remainder = total_tiles % num_cores;
    uint32_t start = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const uint32_t count = base + (i < remainder ? 1 : 0);
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values, cores[i], {{"start_tile", start}, {"num_tiles", count}});
        start += count;
    }

    ProgramSpec spec{
        .name = "ltx_rope_materialize",
        .kernels = {writer},
        .dataflow_buffers = dfbs,
        .tensor_parameters = tensors,
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {WRITER}, .target_nodes = all_cores}}};

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {writer_run};
    run_args.tensor_args = {
        {SELF_COS_INPUT, TensorArgument{a.compact_self_cos.mesh_tensor()}},
        {SELF_SIN_INPUT, TensorArgument{a.compact_self_sin.mesh_tensor()}},
        {CROSS_COS_INPUT, TensorArgument{a.compact_cross_cos.mesh_tensor()}},
        {CROSS_SIN_INPUT, TensorArgument{a.compact_cross_sin.mesh_tensor()}},
        {METADATA, TensorArgument{a.metadata.mesh_tensor()}},
        {SELF_COS_OUTPUT, TensorArgument{self_cos_out}},
        {SELF_SIN_OUTPUT, TensorArgument{self_sin_out}},
        {CROSS_COS_OUTPUT, TensorArgument{cross_cos_out}},
        {CROSS_SIN_OUTPUT, TensorArgument{cross_sin_out}},
    };

    auto program = MakeProgramFromSpec(*mesh_device, spec);
    SetProgramRunArgs(program, run_args);
    return {std::move(program), SharedVariables{}};
}

LtxRopeMaterializeDeviceOperation::MeshWorkloadFactory::cached_mesh_workload_t
LtxRopeMaterializeDeviceOperation::MeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& attrs,
    const ttnn::MeshCoordinateRangeSet& coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared;
    for (const auto& coord : coords.coords()) {
        auto cached = create_at(attrs, coord, tensor_args, outputs);
        const ttnn::MeshCoordinateRange range(coord);
        workload.add_program(range, std::move(cached.program));
        shared.emplace(range, cached.shared_variables);
    }
    return {std::move(workload), std::move(shared)};
}

void LtxRopeMaterializeDeviceOperation::MeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached,
    const operation_attributes_t&,
    const tensor_args_t& a,
    tensor_return_value_t& outputs) {
    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {SELF_COS_INPUT, TensorArgument{a.compact_self_cos.mesh_tensor()}},
        {SELF_SIN_INPUT, TensorArgument{a.compact_self_sin.mesh_tensor()}},
        {CROSS_COS_INPUT, TensorArgument{a.compact_cross_cos.mesh_tensor()}},
        {CROSS_SIN_INPUT, TensorArgument{a.compact_cross_sin.mesh_tensor()}},
        {METADATA, TensorArgument{a.metadata.mesh_tensor()}},
        {SELF_COS_OUTPUT, TensorArgument{std::get<0>(outputs).mesh_tensor()}},
        {SELF_SIN_OUTPUT, TensorArgument{std::get<1>(outputs).mesh_tensor()}},
        {CROSS_COS_OUTPUT, TensorArgument{std::get<2>(outputs).mesh_tensor()}},
        {CROSS_SIN_OUTPUT, TensorArgument{std::get<3>(outputs).mesh_tensor()}},
    };
    for (auto& [range, program] : cached.workload.get_programs()) {
        UpdateProgramRunArgs(program, run_args);
    }
}

}  // namespace ttnn::operations::experimental::transformer::ltx_rope_materialize

namespace ttnn::prim {

std::tuple<Tensor, Tensor, Tensor, Tensor> ltx_rope_materialize(
    const Tensor& compact_self_cos,
    const Tensor& compact_self_sin,
    const Tensor& compact_cross_cos,
    const Tensor& compact_cross_sin,
    const Tensor& metadata,
    const Tensor& self_cos_output,
    const Tensor& self_sin_output,
    const Tensor& cross_cos_output,
    const Tensor& cross_sin_output,
    uint32_t sp_axis,
    uint32_t tp_axis) {
    using Op = operations::experimental::transformer::ltx_rope_materialize::LtxRopeMaterializeDeviceOperation;
    return device_operation::launch<Op>(
        Op::operation_attributes_t{.sp_axis = sp_axis, .tp_axis = tp_axis},
        Op::tensor_args_t{
            .compact_self_cos = compact_self_cos,
            .compact_self_sin = compact_self_sin,
            .compact_cross_cos = compact_cross_cos,
            .compact_cross_sin = compact_cross_sin,
            .metadata = metadata,
            .self_cos_output = self_cos_output,
            .self_sin_output = self_sin_output,
            .cross_cos_output = cross_cos_output,
            .cross_sin_output = cross_sin_output});
}

}  // namespace ttnn::prim
