// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_indexed_device_operation.hpp"

#include <cstdint>
#include <unordered_map>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/device.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama/device/rotary_embedding_llama_metal2_common.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::rotary_embedding_indexed {

using namespace tt::tt_metal;
using namespace tt::constants;
using namespace tt::tt_metal::experimental;
// Reused-kernel binding vocabulary (CB / tensor names, writer+compute sources) — single-sourced here.
using namespace ttnn::experimental::prim::rope_metal2;

namespace {

// Writer + compute kernels are reused verbatim from the rotary_embedding_llama prefill path (they
// consume cos/sin from the CB and write output indexed by local seq tile -- neither touches the
// cos/sin source index). Only the reader is forked to derive the per-device cos/sin shard offset.
// The shared CB/tensor names and the writer/compute sources come from rope_metal2 (kWriterSource,
// kComputeSource, INPUT_DFB, OUT_DFB, OUTPUT_PARAM, ...) so the reused-kernel binding contract has one
// source of truth; only the reader source and the metadata-path names are local.
constexpr auto kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/kernels/dataflow/"
    "reader_rotary_embedding_indexed_interleaved_start_id.cpp";

// Metadata path only: L1-scratch CB the reader reads the metadata page into. The metadata tensor is a
// dedicated 1-element uint32 tensor holding kv_actual_global directly at element [0]; the reader
// NoC-reads only that one element (4 bytes). kMetadataBytes is the CB page size, kept at the 16-byte
// L1 page-alignment floor -- the read itself is 4 bytes.
constexpr uint32_t kMetadataBytes = 16;  // CB page size (16B L1 alignment floor); only 4B (element [0]) is read

// Metadata-path-only names (everything else comes from rope_metal2).
const DFBSpecName META_DFB{"meta"};
const TensorParamName METADATA_PARAM{"metadata"};

// Structural + per-call checks shared by the cache-miss and cache-hit paths. The structural checks
// (cluster_axis, 2D mesh, chunk height) run on both paths; the kv_actual_global VALUE checks run only
// on the SCALAR path — on the METADATA path kv_actual_global lives in the 1-element device tensor
// (read on-device, so its value can't be checked host-side) and is the caller's responsibility. The
// metadata TENSOR itself is still validated (storage/device/dtype/shape) on both paths below.
void validate_runtime_args(
    const RotaryEmbeddingIndexedDeviceOperation::operation_attributes_t& args,
    const RotaryEmbeddingIndexedDeviceOperation::tensor_args_t& tensor_args) {
    // cluster_axis selects which mesh dim is the SP axis (num_rows vs num_cols); any other value
    // would silently pick the wrong extent and corrupt the per-device sharding math.
    TT_FATAL(args.cluster_axis == 0 || args.cluster_axis == 1, "cluster_axis ({}) must be 0 or 1", args.cluster_axis);

    const auto& input = tensor_args.input;
    const auto& cos = tensor_args.cos;
    const auto& mesh_view = cos.device()->get_view();
    TT_FATAL(mesh_view.is_mesh_2d(), "rotary_embedding_indexed requires a 2D mesh");
    const uint32_t chunk_local_t = input.padded_shape()[-2] / TILE_HEIGHT;
    // chunk_local_t is the per-chip chunk height in tiles and is used by the reader as a
    // divisor/modulus to derive the boundary chip; a zero-height input chunk would divide by zero.
    TT_FATAL(chunk_local_t > 0, "input chunk seq dim ({}) must be at least one tile", input.padded_shape()[-2]);

    if (tensor_args.metadata.has_value()) {
        // Metadata path: kv_actual_global is read on-device from element [0] of the metadata tensor, so
        // its VALUE is the caller's responsibility. But the tensor is bound as tensor::metadata and read
        // on-device as uint32, so validate the tensor itself here (runs on both cache miss and hit,
        // since the metadata tensor can differ per call). dtype is NOT part of the program hash, so
        // without this guard a uint32-then-bf16 sequence would silently reuse the cached program.
        const auto& metadata = tensor_args.metadata.value();
        TT_FATAL(metadata.storage_type() == StorageType::DEVICE, "metadata must be on device");
        TT_FATAL(metadata.buffer() != nullptr, "metadata must be allocated in a buffer on device");
        TT_FATAL(metadata.device() == input.device(), "metadata must be on the same device as input");
        TT_FATAL(
            metadata.dtype() == DataType::UINT32,
            "metadata must be uint32 (holds kv_actual_global, read on-device as uint32), got {}",
            metadata.dtype());
        TT_FATAL(
            metadata.logical_shape().volume() == 1,
            "metadata must be a single-element tensor (kv_actual_global at element [0]), got {} elements",
            metadata.logical_shape().volume());
        return;
    }

    // The reader divides kv_actual_global by TILE_HEIGHT to get its tile offset into the cos/sin
    // shard, so it must be tile-aligned.
    TT_FATAL(
        args.kv_actual_global % TILE_HEIGHT == 0,
        "kv_actual_global ({}) must be tile-aligned (a multiple of {})",
        args.kv_actual_global,
        TILE_HEIGHT);

    // Bound the largest update_idxt any chip reads from by the per-device cos/sin shard height.
    // Mirror the reader kernel's per-chip update_idxt exactly: each chip reads chunk_local_t tiles
    // starting at update_idxt, where chips before the boundary chip jump to the next slab
    // ((boundary_slab+1)*chunk_local_t), the boundary chip starts at boundary_slab*chunk_local_t +
    // offset, and chips after it stay on this slab. The max is the pre-boundary value WHEN a
    // pre-boundary chip exists (boundary_chip > 0); when kv_actual_global is exactly slab-aligned
    // (boundary_chip == 0) no chip jumps ahead, so a flat (+1 slab) bound would be off by a slab.
    const uint32_t sp_factor = (args.cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    const uint32_t kv_actual_global_t = args.kv_actual_global / TILE_HEIGHT;
    const uint32_t cos_shard_Ht = cos.padded_shape()[-2] / TILE_HEIGHT;
    const uint32_t chunk_global_t = sp_factor * chunk_local_t;
    const uint32_t boundary_slab_t = (kv_actual_global_t / chunk_global_t) * chunk_local_t;
    const uint32_t boundary_chip = (kv_actual_global_t / chunk_local_t) % sp_factor;
    const uint32_t boundary_offset_t = kv_actual_global_t % chunk_local_t;
    const uint32_t max_update_idxt =
        (boundary_chip > 0) ? boundary_slab_t + chunk_local_t : boundary_slab_t + boundary_offset_t;
    TT_FATAL(
        max_update_idxt + chunk_local_t <= cos_shard_Ht,
        "kv_actual_global ({} tok) + chunk would index past the per-device cos/sin shard ({} tiles)",
        args.kv_actual_global,
        cos_shard_Ht);
}

}  // namespace

RotaryEmbeddingIndexedDeviceOperation::program_factory_t RotaryEmbeddingIndexedDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return MeshWorkloadFactory{};
}

void RotaryEmbeddingIndexedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& cos = tensor_args.cos;
    const auto& sin = tensor_args.sin;
    const auto& trans_mat = tensor_args.trans_mat;

    TT_FATAL(input.storage_type() == StorageType::DEVICE, "input must be on device");
    TT_FATAL(cos.storage_type() == StorageType::DEVICE, "cos must be on device");
    TT_FATAL(sin.storage_type() == StorageType::DEVICE, "sin must be on device");
    TT_FATAL(trans_mat.storage_type() == StorageType::DEVICE, "trans_mat must be on device");

    // Every operand is bound as a tensor parameter and accessed on the input's mesh device, so all
    // must be allocated and live on that same device.
    TT_FATAL(input.buffer() != nullptr, "input must be allocated in a buffer on device");
    TT_FATAL(cos.buffer() != nullptr, "cos must be allocated in a buffer on device");
    TT_FATAL(sin.buffer() != nullptr, "sin must be allocated in a buffer on device");
    TT_FATAL(trans_mat.buffer() != nullptr, "trans_mat must be allocated in a buffer on device");
    TT_FATAL(cos.device() == input.device(), "cos must be on the same device as input");
    TT_FATAL(sin.device() == input.device(), "sin must be on the same device as input");
    TT_FATAL(trans_mat.device() == input.device(), "trans_mat must be on the same device as input");

    TT_FATAL(input.layout() == Layout::TILE, "input must be TILE layout");
    TT_FATAL(cos.layout() == Layout::TILE, "cos must be TILE layout");
    TT_FATAL(sin.layout() == Layout::TILE, "sin must be TILE layout");
    TT_FATAL(trans_mat.layout() == Layout::TILE, "trans_mat must be TILE layout");

    const auto& input_shape = input.padded_shape();
    const auto& cos_shape = cos.padded_shape();
    const auto& sin_shape = sin.padded_shape();
    const auto& trans_mat_shape = trans_mat.padded_shape();
    TT_FATAL(input_shape.rank() == 4, "input must be 4D (got rank {})", input_shape.rank());
    TT_FATAL(cos_shape.rank() == 4, "cos must be 4D (got rank {})", cos_shape.rank());
    // The reader pushes trans_mat as a single page (page 0) into a one-tile CB, so it must be exactly
    // one tile -- a larger tensor would be silently truncated to its first tile.
    TT_FATAL(
        trans_mat_shape.rank() == 4 && trans_mat_shape[0] == 1 && trans_mat_shape[1] == 1 &&
            trans_mat_shape[-2] == TILE_HEIGHT && trans_mat_shape[-1] == TILE_WIDTH,
        "trans_mat must be a single tile [1, 1, {}, {}] (got {})",
        TILE_HEIGHT,
        TILE_WIDTH,
        trans_mat_shape);
    TT_FATAL(cos.dtype() == sin.dtype(), "cos and sin dtype must match");
    TT_FATAL(cos_shape == sin_shape, "cos and sin must have the same shape");
    TT_FATAL(input_shape[-1] == cos_shape[-1], "input and cos head dim must match");

    const uint32_t input_seq = input_shape[-2];
    TT_FATAL(input_seq % TILE_HEIGHT == 0, "input seq dim ({}) must be tile-aligned", input_seq);

    validate_runtime_args(args, tensor_args);
}

void RotaryEmbeddingIndexedDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // kv_actual_global is not hashed and can differ from the compiled program's call; re-validate
    // every hit. Structural constraints are hashed and so guaranteed unchanged here.
    validate_runtime_args(args, tensor_args);
}

RotaryEmbeddingIndexedDeviceOperation::spec_return_value_t RotaryEmbeddingIndexedDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    return tt::tt_metal::TensorSpec(
        input.logical_shape(),
        tt::tt_metal::TensorLayout(input.dtype(), tt::tt_metal::PageConfig(input.layout()), args.output_mem_config));
}

RotaryEmbeddingIndexedDeviceOperation::tensor_return_value_t
RotaryEmbeddingIndexedDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

ttsl::hash::hash_t RotaryEmbeddingIndexedDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // The cache key must cover every structural spec the program is built from -- in particular every
    // TensorParameter's declared TensorSpec -- because on a cache hit UpdateProgramRunArgs REJECTS (does
    // not recompile) a tensor whose spec doesn't match the one baked at creation. So hash the full input,
    // cos, sin and trans_mat specs, the output spec (via output_mem_config; its dtype/shape follow input),
    // cluster_axis, compute_kernel_config, and metadata.has_value() + the metadata tensor's spec.
    //
    // Only kv_actual_global is excluded: it is a reader runtime arg patched on cache hits, so successive
    // chunks reuse one cached program. Per-device my_sp_coord is a per-coordinate compile-time arg the
    // mesh adapter already folds into the workload hash via the target coordinates. Full shapes (not just
    // volumes) are hashed since the work split and CB sizing derive from specific dimensions.
    const auto& input = tensor_args.input;
    const auto& cos = tensor_args.cos;
    const auto& sin = tensor_args.sin;
    const auto& trans_mat = tensor_args.trans_mat;
    // metadata is an optional TensorParameter; stand in with defaults on the scalar path (already
    // separated by has_value=false) so its spec still participates in the key when present.
    const MemoryConfig metadata_mem_config =
        tensor_args.metadata.has_value() ? tensor_args.metadata->memory_config() : MemoryConfig{};
    const Shape metadata_padded_shape =
        tensor_args.metadata.has_value() ? tensor_args.metadata->padded_shape() : Shape{};
    return tt::tt_metal::operation::hash_operation<RotaryEmbeddingIndexedDeviceOperation>(
        tensor_args.metadata.has_value(),
        metadata_mem_config,
        metadata_padded_shape,
        args.cluster_axis,
        args.compute_kernel_config,
        args.output_mem_config,
        input.dtype(),
        input.memory_config(),
        input.logical_shape(),
        input.padded_shape(),
        input.layout(),
        cos.dtype(),
        cos.memory_config(),
        cos.padded_shape(),
        sin.dtype(),
        sin.memory_config(),
        sin.padded_shape(),
        trans_mat.dtype(),
        trans_mat.memory_config(),
        trans_mat.padded_shape());
}

RotaryEmbeddingIndexedDeviceOperation::MeshWorkloadFactory::cached_program_t
RotaryEmbeddingIndexedDeviceOperation::MeshWorkloadFactory::create_at(
    const operation_attributes_t& args,
    const ttnn::MeshCoordinate& coord,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input = tensor_args.input.mesh_tensor();
    const auto& cos = tensor_args.cos.mesh_tensor();
    const auto& sin = tensor_args.sin.mesh_tensor();
    const auto& trans_mat = tensor_args.trans_mat.mesh_tensor();
    const auto& out = output.mesh_tensor();
    const bool has_metadata = tensor_args.metadata.has_value();

    auto* mesh_device = tensor_args.input.device();

    const tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    const tt::DataFormat cos_cb_data_format = datatype_to_dataformat_converter(cos.dtype());
    const uint32_t cos_single_tile_size = tt::tile_size(cos_cb_data_format);
    const tt::DataFormat sin_cb_data_format = datatype_to_dataformat_converter(sin.dtype());
    const uint32_t sin_single_tile_size = tt::tile_size(sin_cb_data_format);
    const tt::DataFormat trans_mat_cb_data_format = datatype_to_dataformat_converter(trans_mat.dtype());
    const uint32_t trans_mat_single_tile_size = tt::tile_size(trans_mat_cb_data_format);
    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(out.dtype());
    const uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const uint32_t batch = input.padded_shape()[0];
    const uint32_t n_heads = input.padded_shape()[1];
    const uint32_t seq_len_t = input.padded_shape()[2] / TILE_HEIGHT;
    const uint32_t head_dim_t = input.padded_shape()[3] / TILE_WIDTH;
    const uint32_t cos_seq_len_t = cos.padded_shape()[2] / TILE_HEIGHT;
    const uint32_t sin_seq_len_t = sin.padded_shape()[2] / TILE_HEIGHT;
    // cos/sin are the (much taller) per-device shards, so rotary coverage is bounded by the input.
    const uint32_t rotary_seq_len_t = seq_len_t;
    // Flag for whether or not sin/cos vary per head. If false, they will be broadcasted across heads.
    const bool freq_per_head = cos.padded_shape()[1] == n_heads;

    // Per-device cos/sin shard offset inputs, baked into this coordinate's program as compile-time args:
    // sp_factor is the mesh extent along the cluster axis and my_sp_coord is this chip's index along it.
    // Both are structural (per mesh coordinate) and constant across calls, so they are baked as CTAs.
    const auto& mesh_view = mesh_device->get_view();
    const uint32_t sp_factor = (args.cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    const uint32_t my_sp_coord =
        ::ttnn::ccl::get_linearized_index_from_physical_coord(tensor_args.cos, coord, args.cluster_axis);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(mesh_device->arch(), args.compute_kernel_config);

    auto compute_with_storage_grid_size = mesh_device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange all_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    const uint32_t num_input_tiles = 2 * head_dim_t;
    const uint32_t num_output_tiles = num_input_tiles;

    const bool row_major = true;
    const uint32_t num_cores = num_cores_x * num_cores_y;
    const uint32_t batch_parallel_factor = std::min(batch, num_cores);
    const uint32_t seq_parallel_factor = std::min(num_cores / batch_parallel_factor, seq_len_t);
    const uint32_t batch_per_core = (batch + batch_parallel_factor - 1) / batch_parallel_factor;
    const uint32_t seq_per_core = (seq_len_t + seq_parallel_factor - 1) / seq_parallel_factor;

    const uint32_t num_sin_cos_rows_per_core = (seq_len_t + seq_parallel_factor - 1) / seq_parallel_factor;
    const uint32_t num_rows_per_core = num_sin_cos_rows_per_core * n_heads;

    uint32_t num_cos_sin_tiles = 2 * head_dim_t * num_sin_cos_rows_per_core;
    uint32_t input_cb_num_tiles = num_sin_cos_rows_per_core * num_input_tiles;

    const bool use_reload_impl = num_rows_per_core > 8 || freq_per_head;
    if (use_reload_impl) {
        input_cb_num_tiles = num_input_tiles;
        num_cos_sin_tiles = num_input_tiles;
    }

    // ------------------------------------------------------------------ dataflow buffers
    std::vector<DataflowBufferSpec> dfbs = {
        DataflowBufferSpec{
            .unique_id = INPUT_DFB,
            .entry_size = input_single_tile_size,
            .num_entries = input_cb_num_tiles,
            .data_format_metadata = input_cb_data_format},
        DataflowBufferSpec{
            .unique_id = COS_DFB,
            .entry_size = cos_single_tile_size,
            .num_entries = num_cos_sin_tiles,
            .data_format_metadata = cos_cb_data_format},
        DataflowBufferSpec{
            .unique_id = SIN_DFB,
            .entry_size = sin_single_tile_size,
            .num_entries = num_cos_sin_tiles,
            .data_format_metadata = sin_cb_data_format},
        DataflowBufferSpec{
            .unique_id = TRANS_MAT_DFB,
            .entry_size = trans_mat_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = trans_mat_cb_data_format},
        DataflowBufferSpec{
            .unique_id = ROTATED_INTERM_DFB,
            .entry_size = input_single_tile_size,
            .num_entries = head_dim_t,
            .data_format_metadata = input_cb_data_format},
        DataflowBufferSpec{
            .unique_id = COS_INTERM_DFB,
            .entry_size = cos_single_tile_size,
            .num_entries = head_dim_t,
            .data_format_metadata = cos_cb_data_format},
        DataflowBufferSpec{
            .unique_id = SIN_INTERM_DFB,
            .entry_size = sin_single_tile_size,
            .num_entries = head_dim_t,
            .data_format_metadata = sin_cb_data_format},
        DataflowBufferSpec{
            .unique_id = OUT_DFB,
            .entry_size = output_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = output_cb_data_format},
        DataflowBufferSpec{
            .unique_id = ZERO_DFB,
            .entry_size = output_single_tile_size,
            .num_entries = head_dim_t,
            .data_format_metadata = output_cb_data_format},
    };
    if (has_metadata) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = META_DFB,
            .entry_size = kMetadataBytes,
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::UInt32});
    }

    // ------------------------------------------------------------------ tensor parameters
    std::vector<TensorParameter> tensor_params = {
        TensorParameter{.unique_id = INPUT_PARAM, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = COS_PARAM, .spec = cos.tensor_spec()},
        TensorParameter{.unique_id = SIN_PARAM, .spec = sin.tensor_spec()},
        TensorParameter{.unique_id = TRANS_MAT_PARAM, .spec = trans_mat.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT_PARAM, .spec = out.tensor_spec()},
    };
    if (has_metadata) {
        tensor_params.push_back(
            TensorParameter{.unique_id = METADATA_PARAM, .spec = tensor_args.metadata->mesh_tensor().tensor_spec()});
    }

    const ComputeHardwareConfig compute_hw_config =
        ComputeGen1Config{.fpu_math_fidelity = math_fidelity, .enable_32_bit_dest = fp32_dest_acc_en};

    const KernelSpec::CompilerOptions::Defines reload_define{{"RELOAD_IMPL", use_reload_impl ? "1" : "0"}};
    KernelSpec::CompilerOptions::Defines reader_defines = reload_define;
    if (has_metadata) {
        reader_defines.emplace("HAS_METADATA", "1");
    }

    // ------------------------------------------------------------------ reader spec (this op's own)
    std::vector<DFBBinding> reader_dfbs = {
        DFBBinding{.dfb_spec_name = INPUT_DFB, .accessor_name = "input", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = COS_DFB, .accessor_name = "cos", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = SIN_DFB, .accessor_name = "sin", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{
            .dfb_spec_name = TRANS_MAT_DFB, .accessor_name = "trans_mat", .endpoint_type = DFBEndpointType::PRODUCER},
    };
    std::vector<TensorBinding> reader_tensors = {
        TensorBinding{.tensor_parameter_name = INPUT_PARAM, .accessor_name = "input"},
        TensorBinding{.tensor_parameter_name = COS_PARAM, .accessor_name = "cos"},
        TensorBinding{.tensor_parameter_name = SIN_PARAM, .accessor_name = "sin"},
        TensorBinding{.tensor_parameter_name = TRANS_MAT_PARAM, .accessor_name = "trans_mat"},
    };
    if (has_metadata) {
        // meta scratch CB is a single-toucher (reader fills + reads it) → self-loop.
        reader_dfbs.push_back(
            DFBBinding{.dfb_spec_name = META_DFB, .accessor_name = "meta", .endpoint_type = DFBEndpointType::PRODUCER});
        reader_dfbs.push_back(
            DFBBinding{.dfb_spec_name = META_DFB, .accessor_name = "meta", .endpoint_type = DFBEndpointType::CONSUMER});
        reader_tensors.push_back(TensorBinding{.tensor_parameter_name = METADATA_PARAM, .accessor_name = "metadata"});
    }

    KernelSpec::RuntimeArgSchema reader_schema{
        .runtime_arg_names = {"batch_start", "batch_end", "seq_t_start", "seq_t_end"}};
    if (!has_metadata) {
        reader_schema.common_runtime_arg_names = {"kv_actual_global"};
    }

    KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path{kReaderKernelPath},
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings = reader_dfbs,
        .tensor_bindings = reader_tensors,
        .compile_time_args =
            {{"n_heads", n_heads},
             {"Ht", seq_len_t},
             {"Wt", head_dim_t},
             {"freq_per_head", static_cast<uint32_t>(freq_per_head)},
             {"cos_Ht", cos_seq_len_t},
             {"sin_Ht", sin_seq_len_t},
             {"rotary_Ht", rotary_seq_len_t},
             {"tile_height", TILE_HEIGHT},  // reader divides kv_actual_global (tokens) into tiles
             {"my_sp_coord", my_sp_coord},
             {"sp_factor", sp_factor}},
        .runtime_arg_schema = reader_schema,
        .hw_config = create_reader_datamovement_config(mesh_device->arch())};

    // ------------------------------------------------------------------ writer + compute (reused llama)
    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = kWriterSource,
        .compiler_options = {.defines = reload_define},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
             // zero is a single-toucher (writer fills + reads it) -> self-loop (PRODUCER + CONSUMER).
             DFBBinding{.dfb_spec_name = ZERO_DFB, .accessor_name = "zero", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = ZERO_DFB, .accessor_name = "zero", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_PARAM, .accessor_name = "output"}},
        .compile_time_args =
            {{"n_heads", n_heads}, {"Wt", head_dim_t}, {"Ht", seq_len_t}, {"rotary_Ht", rotary_seq_len_t}},
        .runtime_arg_schema = {.runtime_arg_names = {"batch_start", "batch_end", "seq_t_start", "seq_t_end"}},
        .hw_config = create_writer_datamovement_config(mesh_device->arch())};

    KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = kComputeSource,
        .compiler_options = {.defines = reload_define},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = INPUT_DFB, .accessor_name = "input", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = COS_DFB, .accessor_name = "cos", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = SIN_DFB, .accessor_name = "sin", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = TRANS_MAT_DFB,
                 .accessor_name = "trans_mat",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
             // Intermediate CBs: compute is the sole toucher -> each is a self-loop (PRODUCER + CONSUMER).
             DFBBinding{
                 .dfb_spec_name = ROTATED_INTERM_DFB,
                 .accessor_name = "rotated_interm",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = ROTATED_INTERM_DFB,
                 .accessor_name = "rotated_interm",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = COS_INTERM_DFB,
                 .accessor_name = "cos_interm",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = COS_INTERM_DFB,
                 .accessor_name = "cos_interm",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = SIN_INTERM_DFB,
                 .accessor_name = "sin_interm",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = SIN_INTERM_DFB,
                 .accessor_name = "sin_interm",
                 .endpoint_type = DFBEndpointType::CONSUMER}},
        .compile_time_args = {{"Wt", head_dim_t}, {"n_heads", n_heads}, {"rotary_Ht", rotary_seq_len_t}},
        .runtime_arg_schema = {.runtime_arg_names = {"batch_start", "batch_end", "seq_t_start", "seq_t_end"}},
        .hw_config = compute_hw_config};

    // ------------------------------------------------------------------ per-node runtime args
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    struct CoreArgs {
        uint32_t start_batch = 0;
        uint32_t end_batch = 0;
        uint32_t start_seq = 0;
        uint32_t end_seq = 0;
    };
    std::vector<CoreArgs> per_core_args(cores.size());
    for (uint32_t batch_parallel = 0; batch_parallel < batch_parallel_factor; batch_parallel++) {
        for (uint32_t seq_parallel = 0; seq_parallel < seq_parallel_factor; seq_parallel++) {
            uint32_t core_idx = (batch_parallel * seq_parallel_factor) + seq_parallel;
            uint32_t start_batch = batch_parallel * batch_per_core;
            uint32_t end_batch = std::min(start_batch + batch_per_core, batch);
            uint32_t start_seq = seq_parallel * seq_per_core;
            uint32_t end_seq = std::min(start_seq + seq_per_core, seq_len_t);
            if (start_seq >= seq_len_t || start_batch >= batch) {
                continue;
            }
            per_core_args[core_idx] = CoreArgs{start_batch, end_batch, start_seq, end_seq};
        }
    }

    KernelRunArgs reader_run{.kernel = READER};
    KernelRunArgs writer_run{.kernel = WRITER};
    KernelRunArgs compute_run{.kernel = COMPUTE};
    if (!has_metadata) {
        reader_run.common_runtime_arg_values = {{"kv_actual_global", args.kv_actual_global}};
    }
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& a = per_core_args[i];
        const NodeCoord node = cores[i];
        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            node,
            {{"batch_start", a.start_batch},
             {"batch_end", a.end_batch},
             {"seq_t_start", a.start_seq},
             {"seq_t_end", a.end_seq}});
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            node,
            {{"batch_start", a.start_batch},
             {"batch_end", a.end_batch},
             {"seq_t_start", a.start_seq},
             {"seq_t_end", a.end_seq}});
        AddRuntimeArgsForNode(
            compute_run.runtime_arg_values,
            node,
            {{"batch_start", a.start_batch},
             {"batch_end", a.end_batch},
             {"seq_t_start", a.start_seq},
             {"seq_t_end", a.end_seq}});
    }

    // ------------------------------------------------------------------ assemble + compile
    ProgramSpec spec{
        .name = "rotary_embedding_indexed",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = dfbs,
        .tensor_parameters = tensor_params,
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores}}};

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run, compute_run};
    run_args.tensor_args = {
        {INPUT_PARAM, TensorArgument{input}},
        {COS_PARAM, TensorArgument{cos}},
        {SIN_PARAM, TensorArgument{sin}},
        {TRANS_MAT_PARAM, TensorArgument{trans_mat}},
        {OUTPUT_PARAM, TensorArgument{out}}};
    if (has_metadata) {
        run_args.tensor_args.emplace(METADATA_PARAM, TensorArgument{tensor_args.metadata->mesh_tensor()});
    }

    auto program = MakeProgramFromSpec(*mesh_device, spec);
    SetProgramRunArgs(program, run_args);
    return {std::move(program), SharedVariables{}};
}

RotaryEmbeddingIndexedDeviceOperation::MeshWorkloadFactory::cached_mesh_workload_t
RotaryEmbeddingIndexedDeviceOperation::MeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(args, coord, tensor_args, output);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), cached_program.shared_variables);
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

void RotaryEmbeddingIndexedDeviceOperation::MeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // kv_actual_global is not hashed and can change per chunk; per-core RTAs and my_sp_coord are stable
    // (same shapes / same coordinate), so a cache hit only refreshes tensor addresses and, on the scalar
    // path, the kv_actual_global common arg. The metadata path advances the value on-device (the reader
    // re-reads element [0]), so only its tensor address needs refreshing. has_metadata is hashed, so it
    // matches the cached program on every hit -- read it straight from tensor_args. The run args are
    // coordinate-independent, so build them once and apply to every stamped program.
    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {INPUT_PARAM, TensorArgument{tensor_args.input.mesh_tensor()}},
        {COS_PARAM, TensorArgument{tensor_args.cos.mesh_tensor()}},
        {SIN_PARAM, TensorArgument{tensor_args.sin.mesh_tensor()}},
        {TRANS_MAT_PARAM, TensorArgument{tensor_args.trans_mat.mesh_tensor()}},
        {OUTPUT_PARAM, TensorArgument{output.mesh_tensor()}}};
    if (tensor_args.metadata.has_value()) {
        run_args.tensor_args.emplace(METADATA_PARAM, TensorArgument{tensor_args.metadata->mesh_tensor()});
    } else {
        KernelRunArgs reader_run{.kernel = READER};
        reader_run.common_runtime_arg_values = {{"kv_actual_global", args.kv_actual_global}};
        run_args.kernel_run_args = {reader_run};
    }

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        UpdateProgramRunArgs(program, run_args);
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::rotary_embedding_indexed

namespace ttnn::prim {

ttnn::Tensor rotary_embedding_indexed(
    const ttnn::Tensor& input,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& trans_mat,
    const std::optional<ttnn::Tensor>& metadata,
    uint32_t kv_actual_global,
    uint32_t cluster_axis,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::experimental::deepseek_prefill::rotary_embedding_indexed::
        RotaryEmbeddingIndexedDeviceOperation;

    auto arch = input.storage_type() == StorageType::DEVICE ? input.device()->arch() : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        arch, compute_kernel_config, tt::tt_metal::MathFidelity::HiFi4, true, false, false);

    MemoryConfig out_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
    if (input.storage_type() == StorageType::DEVICE) {
        out_mem_config = input.memory_config();
    }
    if (memory_config.has_value()) {
        out_mem_config = memory_config.value();
    }

    auto attrs = OperationType::operation_attributes_t{
        .cluster_axis = cluster_axis,
        .kv_actual_global = kv_actual_global,
        .output_mem_config = out_mem_config,
        .compute_kernel_config = kernel_config_val,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .input = input, .cos = cos, .sin = sin, .trans_mat = trans_mat, .metadata = metadata};
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
