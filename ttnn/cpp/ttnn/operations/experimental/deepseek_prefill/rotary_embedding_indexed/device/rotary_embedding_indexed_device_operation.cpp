// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_indexed_device_operation.hpp"

#include <cstdint>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/device.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::rotary_embedding_indexed {

using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

// Writer + compute kernels are reused verbatim from the rotary_embedding_llama prefill path (they
// consume cos/sin from the CB and write output indexed by local seq tile -- neither touches the
// cos/sin source index). Only the reader is forked to derive the per-device cos/sin shard offset.
constexpr auto kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/kernels/dataflow/"
    "reader_rotary_embedding_indexed_interleaved_start_id.cpp";
constexpr auto kWriterKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/dataflow/"
    "writer_rotary_embedding_llama_interleaved_start_id.cpp";
constexpr auto kComputeKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/"
    "rotary_embedding_llama.cpp";

// kv_actual_global is intentionally kept out of the program hash so successive chunks reuse the
// cached program; it bypasses validate_on_program_cache_miss on the 2nd+ call. This helper is
// invoked from BOTH miss and hit paths to keep the tile-alignment / capacity checks every call.
void validate_runtime_args(
    const RotaryEmbeddingIndexedDeviceOperation::operation_attributes_t& args,
    const RotaryEmbeddingIndexedDeviceOperation::tensor_args_t& tensor_args) {
    TT_FATAL(
        args.kv_actual_global % TILE_HEIGHT == 0, "kv_actual_global ({}) must be tile-aligned", args.kv_actual_global);

    const auto& input = tensor_args.input;
    const auto& cos = tensor_args.cos;
    // sp_factor is just the mesh extent along the cluster axis (mirrors how ring-joint SDPA derives
    // num_devices from cluster_axis).
    const auto& mesh_view = cos.device()->get_view();
    TT_FATAL(mesh_view.is_mesh_2d(), "rotary_embedding_indexed requires a 2D mesh");
    const uint32_t sp_factor = (args.cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    const uint32_t chunk_local_t = input.padded_shape()[-2] / TILE_HEIGHT;
    const uint32_t kv_actual_global_t = args.kv_actual_global / TILE_HEIGHT;
    const uint32_t cos_shard_Ht = cos.padded_shape()[-2] / TILE_HEIGHT;

    // Bound the largest update_idxt any chip reads from by the per-device cos/sin shard height.
    // Mirror the reader kernel's per-chip update_idxt exactly: each chip reads chunk_local_t tiles
    // starting at update_idxt, where
    //   chips before the boundary chip jump to the next slab -> (boundary_slab+1)*chunk_local_t,
    //   the boundary chip (== chip boundary_chip) starts at  boundary_slab*chunk_local_t + offset,
    //   chips after it stay on this slab at                  boundary_slab*chunk_local_t.
    // The max is the pre-boundary value WHEN a pre-boundary chip exists (boundary_chip > 0). When
    // kv_actual_global is exactly slab-aligned (boundary_chip == 0, offset == 0) no chip jumps
    // ahead, so the old (+1 slab) bound was off by a slab and wrongly rejected a read that fits.
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
    return ProgramFactory{};
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

    TT_FATAL(input.layout() == Layout::TILE, "input must be TILE layout");
    TT_FATAL(cos.layout() == Layout::TILE, "cos must be TILE layout");
    TT_FATAL(sin.layout() == Layout::TILE, "sin must be TILE layout");

    const auto& input_shape = input.padded_shape();
    const auto& cos_shape = cos.padded_shape();
    const auto& sin_shape = sin.padded_shape();
    TT_FATAL(input_shape.rank() == 4, "input must be 4D (got rank {})", input_shape.rank());
    TT_FATAL(cos_shape.rank() == 4, "cos must be 4D (got rank {})", cos_shape.rank());
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
    return TensorSpec(
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
    // kv_actual_global is a runtime arg read by the reader kernel and intentionally NOT hashed, so
    // successive chunks reuse the cached program. cluster_axis stays IN (structural -- governs which
    // mesh dim is SP and thus the per-device sharding).
    // Hash the full padded shapes, not just their volumes: the descriptor derives seq/head tile
    // counts, the work split and CB sizing from specific dimensions, so two differently-shaped
    // tensors that happen to share a volume must NOT collide onto the same cached program.
    const auto& input = tensor_args.input;
    const auto& cos = tensor_args.cos;
    return tt::tt_metal::operation::hash_operation<RotaryEmbeddingIndexedDeviceOperation>(
        args.cluster_axis,
        input.dtype(),
        input.memory_config(),
        input.padded_shape(),
        cos.dtype(),
        cos.memory_config(),
        cos.padded_shape());
}

tt::tt_metal::ProgramDescriptor RotaryEmbeddingIndexedDeviceOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "RotaryEmbeddingIndexed::create_descriptor requires a mesh dispatch coordinate");
    const auto& coord = mesh_dispatch_coordinate.value();

    const auto& input = tensor_args.input;
    const auto& cos = tensor_args.cos;
    const auto& sin = tensor_args.sin;
    const auto& trans_mat = tensor_args.trans_mat;

    const tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    const tt::DataFormat cos_cb_data_format = datatype_to_dataformat_converter(cos.dtype());
    const uint32_t cos_single_tile_size = tt::tile_size(cos_cb_data_format);
    const tt::DataFormat sin_cb_data_format = datatype_to_dataformat_converter(sin.dtype());
    const uint32_t sin_single_tile_size = tt::tile_size(sin_cb_data_format);
    const tt::DataFormat trans_mat_cb_data_format = datatype_to_dataformat_converter(trans_mat.dtype());
    const uint32_t trans_mat_single_tile_size = tt::tile_size(trans_mat_cb_data_format);
    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
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

    auto* device = input.device();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), args.compute_kernel_config);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
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

    tt::tt_metal::ProgramDescriptor desc;

    constexpr uint8_t input_cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = input_cb_num_tiles * input_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = input_cb_index, .data_format = input_cb_data_format, .page_size = input_single_tile_size}}},
    });
    constexpr uint8_t cos_cb_index = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cos_sin_tiles * cos_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cos_cb_index, .data_format = cos_cb_data_format, .page_size = cos_single_tile_size}}},
    });
    constexpr uint8_t sin_cb_index = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cos_sin_tiles * sin_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = sin_cb_index, .data_format = sin_cb_data_format, .page_size = sin_single_tile_size}}},
    });
    constexpr uint8_t trans_mat_cb_index = tt::CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * trans_mat_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = trans_mat_cb_index,
            .data_format = trans_mat_cb_data_format,
            .page_size = trans_mat_single_tile_size}}},
    });
    constexpr uint8_t rotated_input_interm_cb_index = tt::CBIndex::c_24;
    desc.cbs.push_back(CBDescriptor{
        .total_size = head_dim_t * input_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = rotated_input_interm_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size}}},
    });
    constexpr uint8_t cos_interm_cb_index = tt::CBIndex::c_25;
    desc.cbs.push_back(CBDescriptor{
        .total_size = head_dim_t * cos_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cos_interm_cb_index,
            .data_format = cos_cb_data_format,
            .page_size = cos_single_tile_size}}},
    });
    constexpr uint8_t sin_interm_cb_index = tt::CBIndex::c_26;
    desc.cbs.push_back(CBDescriptor{
        .total_size = head_dim_t * sin_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = sin_interm_cb_index,
            .data_format = sin_cb_data_format,
            .page_size = sin_single_tile_size}}},
    });
    constexpr uint8_t output_cb_index = tt::CBIndex::c_16;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * output_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size}}},
    });
    constexpr uint8_t zero_cb_index = tt::CBIndex::c_27;
    desc.cbs.push_back(CBDescriptor{
        .total_size = head_dim_t * output_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = zero_cb_index,
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size}}},
    });

    KernelDescriptor::Defines kernel_defines;
    kernel_defines.emplace_back("RELOAD_IMPL", use_reload_impl ? "1" : "0");

    auto* src_buffer = input.buffer();
    auto* cos_buffer = cos.buffer();
    auto* sin_buffer = sin.buffer();
    auto* trans_mat_buffer = trans_mat.buffer();
    auto* dst_buffer = output.buffer();

    KernelDescriptor::CompileTimeArgs reader_compile_time_args = {
        (uint32_t)input_cb_index,
        (uint32_t)cos_cb_index,
        (uint32_t)sin_cb_index,
        (uint32_t)trans_mat_cb_index,
        (uint32_t)n_heads,
        (uint32_t)seq_len_t,
        (uint32_t)head_dim_t,
        (uint32_t)freq_per_head,
        (uint32_t)cos_seq_len_t,
        (uint32_t)sin_seq_len_t,
        (uint32_t)rotary_seq_len_t,
    };
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*cos_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*sin_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*trans_mat_buffer).append_to(reader_compile_time_args);

    KernelDescriptor::CompileTimeArgs writer_compile_time_args = {
        (uint32_t)output_cb_index,
        (uint32_t)zero_cb_index,
        (uint32_t)n_heads,
        (uint32_t)head_dim_t,
        (uint32_t)seq_len_t,
        (uint32_t)rotary_seq_len_t,
    };
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // Per-chip cos/sin shard offset inputs (kept out of the program hash via runtime args).
    // sp_factor is the mesh extent along the cluster axis (validated 2D in validate_runtime_args).
    const auto& mesh_view = device->get_view();
    const uint32_t sp_factor = (args.cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    const uint32_t my_sp_coord = ::ttnn::ccl::get_linearized_index_from_physical_coord(cos, coord, args.cluster_axis);
    const uint32_t kv_actual_global_t = args.kv_actual_global / TILE_HEIGHT;

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kReaderKernelPath;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = CoreRangeSet(all_cores);
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = kernel_defines;
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.emplace_common_runtime_args({kv_actual_global_t, my_sp_coord, sp_factor});

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterKernelPath;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = CoreRangeSet(all_cores);
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.defines = kernel_defines;
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor::CompileTimeArgs compute_kernel_args = {
        (uint32_t)input_cb_index,
        (uint32_t)cos_cb_index,
        (uint32_t)sin_cb_index,
        (uint32_t)trans_mat_cb_index,
        (uint32_t)rotated_input_interm_cb_index,
        (uint32_t)cos_interm_cb_index,
        (uint32_t)sin_interm_cb_index,
        (uint32_t)output_cb_index,
        (uint32_t)head_dim_t,
        (uint32_t)n_heads,
        (uint32_t)rotary_seq_len_t,
    };
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = kComputeKernelPath;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = CoreRangeSet(all_cores);
    compute_desc.compile_time_args = std::move(compute_kernel_args);
    compute_desc.defines = kernel_defines;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };

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

    reader_desc.runtime_args.reserve(cores.size());
    writer_desc.runtime_args.reserve(cores.size());
    compute_desc.runtime_args.reserve(cores.size());
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& a = per_core_args[i];
        reader_desc.emplace_runtime_args(
            cores[i],
            {src_buffer->address(),
             cos_buffer->address(),
             sin_buffer->address(),
             trans_mat_buffer->address(),
             a.start_batch,
             a.end_batch,
             a.start_seq,
             a.end_seq});
        writer_desc.emplace_runtime_args(
            cores[i], {dst_buffer->address(), a.start_batch, a.end_batch, a.start_seq, a.end_seq});
        compute_desc.runtime_args.emplace_back(
            cores[i], KernelDescriptor::CoreRuntimeArgs{a.start_batch, a.end_batch, a.start_seq, a.end_seq});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::rotary_embedding_indexed

namespace ttnn::prim {

ttnn::Tensor rotary_embedding_indexed(
    const ttnn::Tensor& input,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& trans_mat,
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
        .kv_actual_global = kv_actual_global,
        .cluster_axis = cluster_axis,
        .output_mem_config = out_mem_config,
        .compute_kernel_config = kernel_config_val,
    };
    auto tensor_args = OperationType::tensor_args_t{.input = input, .cos = cos, .sin = sin, .trans_mat = trans_mat};
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
