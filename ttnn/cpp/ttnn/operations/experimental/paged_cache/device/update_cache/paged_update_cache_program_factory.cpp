// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "paged_update_cache_program_factory.hpp"

#include "paged_update_cache_device_operation.hpp"
#include "paged_update_cache_device_operation_types.hpp"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;

namespace {

bool enable_fp32_dest(const tt_metal::IDevice* device, const ttnn::DeviceComputeKernelConfig& compute_kernel_config) {
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    return fp32_dest_acc_en;
}

// Per-worker-core cache-write offsets derived from `update_idxs`. These values are excluded from the
// program hash (PagedUpdateCacheDeviceOperation::compute_program_hash) yet baked into runtime args, so
// they are re-derived on every cache hit by re-running create_descriptor (override_runtime_arguments).
// This helper is the single source of truth for the formulas, called only from create_descriptor, so
// the miss and hit paths cannot drift. Returns empty when an index tensor is used: in
// that mode the offsets are 0 here and the real positions are read on-device from the (Buffer-bound,
// re-patched) index tensor.
struct UpdateCachePerCoreOffsets {
    tt_metal::CoreCoord core;
    uint32_t cache_start_id = 0;
    uint32_t tile_update_offset_B = 0;
};

std::vector<UpdateCachePerCoreOffsets> compute_update_cache_offsets(
    const PagedUpdateCacheParams& operation_attributes, const PagedUpdateCacheInputs& tensor_args) {
    if (tensor_args.update_idxs_tensor.has_value()) {
        return {};
    }

    const auto& cache_tensor = tensor_args.cache_tensor;
    const auto& input_tensor = tensor_args.input_tensor;
    const bool fp32_dest_acc_en = enable_fp32_dest(input_tensor.device(), operation_attributes.compute_kernel_config);

    const uint32_t Wt = input_tensor.padded_shape()[-1] / TILE_WIDTH;
    const uint32_t Wbytes = fp32_dest_acc_en ? input_tensor.padded_shape()[-1] * sizeof(float)
                                             : input_tensor.padded_shape()[-1] * 2;  // 2 bytes for bfloat16
    const uint32_t cache_total_num_tiles = cache_tensor.physical_volume() / TILE_HW;
    // share_cache => batch offset is 0 (one shared cache buffer); mirror create_descriptor exactly.
    const uint32_t cache_batch_num_tiles =
        operation_attributes.share_cache ? 0 : cache_total_num_tiles / cache_tensor.padded_shape()[0];

    const std::optional<ShardSpec>& shard_spec = input_tensor.shard_spec();
    const bool row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
    const CoreRangeSet all_cores = shard_spec.value().grid;
    const uint32_t num_cores = all_cores.num_cores();
    const auto& cores = corerange_to_cores(all_cores, num_cores, row_major);

    std::vector<UpdateCachePerCoreOffsets> offsets;
    offsets.reserve(cores.size());
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const uint32_t update_idx = operation_attributes.update_idxs.at(i);
        const uint32_t cache_batch_tile_offset = i * cache_batch_num_tiles;
        const uint32_t cache_start_id = cache_batch_tile_offset + ((update_idx / TILE_HEIGHT) * Wt);
        const uint32_t tile_update_offset_B = update_idx % TILE_HEIGHT * Wbytes;
        offsets.push_back({cores.at(i), cache_start_id, tile_update_offset_B});
    }
    return offsets;
}

}  // namespace

ProgramDescriptor PagedUpdateCacheProgramFactory::create_descriptor(
    const PagedUpdateCacheParams& operation_attributes,
    const PagedUpdateCacheInputs& tensor_args,
    Tensor& /*tensor_return_value*/) {
    ProgramDescriptor desc;

    const auto& cache_tensor = tensor_args.cache_tensor;
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& update_idxs_tensor = tensor_args.update_idxs_tensor;
    const auto& page_table = tensor_args.page_table;

    tt_metal::IDevice* device = input_tensor.device();

    tt::DataFormat cache_cb_data_format = tt_metal::datatype_to_dataformat_converter(cache_tensor.dtype());
    uint32_t cache_single_tile_size = tt::tile_size(cache_cb_data_format);

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    bool fp32_dest_acc_en = enable_fp32_dest(device, operation_attributes.compute_kernel_config);

    tt::DataFormat interm_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t interm_single_tile_size = tt::tile_size(interm_cb_data_format);

    // Index tensor-specific parameters
    bool use_index_tensor = update_idxs_tensor.has_value();
    uint32_t index_tensor_tile_size = 0;
    uint32_t log2_page_size = 0;
    uint32_t index_stick_size = 0;
    tt::DataFormat index_data_format = tt::DataFormat::Int32;
    if (use_index_tensor) {
        index_data_format = tt_metal::datatype_to_dataformat_converter(update_idxs_tensor.value().dtype());
        index_tensor_tile_size = tt::tile_size(index_data_format);
        index_stick_size = update_idxs_tensor.value().buffer()->aligned_page_size();
    }

    // Pagetable-specific parameters.
    bool is_paged_cache = page_table.has_value();
    uint32_t block_size = 0;
    uint32_t block_size_t = 0;
    uint32_t max_blocks_per_seq = 0;
    uint32_t page_table_stick_size = 0;
    uint32_t log2_page_table_stick_size = 0;
    tt::DataFormat page_table_data_format = tt::DataFormat::Int32;
    if (is_paged_cache) {
        const auto& page_table_tensor = page_table.value();

        block_size = operation_attributes.block_size_override.value_or(cache_tensor.padded_shape()[2]);
        block_size_t = block_size / TILE_HEIGHT;
        max_blocks_per_seq = page_table_tensor.padded_shape()[1];
        page_table_stick_size = page_table_tensor.padded_shape()[-1] * page_table_tensor.element_size();

        page_table_data_format = tt_metal::datatype_to_dataformat_converter(page_table_tensor.dtype());
    }

    // Per-call write geometry (head_dim, Wt, Wbytes) comes from the input tensor; the
    // cache shape is only a byte budget. num_heads comes from the call view (via the
    // optional num_kv_heads_override) when sharing one buffer across layer types with
    // asymmetric kv-head counts, otherwise from the cache. St is block_size_t in paged
    // mode, cache seq-len-in-tiles otherwise.
    uint32_t Wt = input_tensor.padded_shape()[-1] / TILE_WIDTH;
    uint32_t St = is_paged_cache ? block_size_t : cache_tensor.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t Wbytes = fp32_dest_acc_en ? input_tensor.padded_shape()[-1] * sizeof(float)
                                       : input_tensor.padded_shape()[-1] * 2;  // 2 bytes for bfloat16
    uint32_t cache_total_num_tiles = cache_tensor.physical_volume() / TILE_HW;
    uint32_t cache_batch_num_tiles =
        operation_attributes.share_cache
            ? 0
            : cache_total_num_tiles /
                  cache_tensor.padded_shape()[0];  // if share cache, we can set cache batch num tiles to 0
                                                   // so batch offset would be 0 in future calculations
    uint32_t B = input_tensor.padded_shape()[1];
    uint32_t num_heads = operation_attributes.num_kv_heads_override.value_or(cache_tensor.padded_shape()[1]);

    log_debug(tt::LogOp, "cache_cb_data_format: {}", cache_cb_data_format);
    log_debug(tt::LogOp, "input_cb_data_format: {}", input_cb_data_format);
    log_debug(tt::LogOp, "interm_cb_data_format: {}", interm_cb_data_format);
    log_debug(tt::LogOp, "Wbytes: {}", Wbytes);
    log_debug(tt::LogOp, "Wt: {}", Wt);
    log_debug(tt::LogOp, "St: {}", St);

    const std::optional<ShardSpec>& shard_spec = input_tensor.shard_spec();
    bool row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
    CoreRangeSet all_cores = shard_spec.value().grid;
    uint32_t num_cores = all_cores.num_cores();
    uint32_t num_input_tiles = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
    auto* in1_buffer = shard_spec.has_value() ? input_tensor.buffer() : nullptr;

    uint32_t num_cache_tiles = 2 * Wt;   // double buffered
    uint32_t num_interm_tiles = 2 * Wt;  // double buffered
    uint32_t num_output_tiles = B * Wt;

    const tt::CBIndex src0_cb_index = CBIndex::c_0;
    const tt::CBIndex src1_cb_index = CBIndex::c_1;
    const tt::CBIndex cb_index_id = CBIndex::c_2;
    const tt::CBIndex cb_pagetable_id = CBIndex::c_3;
    const tt::CBIndex intermed0_cb_index = CBIndex::c_24;
    const tt::CBIndex intermed1_cb_index = CBIndex::c_25;
    const tt::CBIndex intermed2_cb_index = CBIndex::c_26;
    const tt::CBIndex output_cb_index = CBIndex::c_16;

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cache_tiles * cache_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cache_cb_data_format,
            .page_size = cache_single_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * input_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src1_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
        .buffer = in1_buffer,
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_interm_tiles * interm_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{
            CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(intermed0_cb_index),
                .data_format = interm_cb_data_format,
                .page_size = interm_single_tile_size,
            },
            CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(intermed1_cb_index),
                .data_format = interm_cb_data_format,
                .page_size = interm_single_tile_size,
            },
        }},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_interm_tiles * interm_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(intermed2_cb_index),
            .data_format = interm_cb_data_format,
            .page_size = interm_single_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * cache_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = cache_cb_data_format,
            .page_size = cache_single_tile_size,
        }}},
    });

    // used for share cache for signaling when the cache is ready to be read
    const uint32_t in0_sequential_mode_semaphore_id = static_cast<uint32_t>(desc.semaphores.size());
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = in0_sequential_mode_semaphore_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = all_cores,
        .initial_value = 0,
    });

    if (use_index_tensor) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = index_tensor_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_index_id),
                .data_format = index_data_format,
                .page_size = index_tensor_tile_size,
            }}},
        });
    }

    if (is_paged_cache) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = page_table_stick_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_pagetable_id),
                .data_format = page_table_data_format,
                .page_size = page_table_stick_size,
            }}},
        });
    }

    auto* dst_buffer = cache_tensor.buffer();

    // cache_position_modulo: 0 = disabled (legacy), nonzero = wrap update_idx mod this
    // value before page_table lookup. Required when the caller's page_table is sized for
    // a bounded sliding-window cache (vLLM SlidingWindowSpec).
    const uint32_t cache_position_modulo = operation_attributes.cache_position_modulo.value_or(0u);

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src1_cb_index,
        // Index tensor args
        (std::uint32_t)use_index_tensor,
        cb_index_id,
        cache_batch_num_tiles,
        Wt,
        log2_page_size,
        index_stick_size,
        // page_table args
        (std::uint32_t)is_paged_cache,
        (std::uint32_t)num_heads,
        (std::uint32_t)block_size,
        (std::uint32_t)block_size_t,
        (std::uint32_t)max_blocks_per_seq,
        log2_page_table_stick_size,
        page_table_stick_size,
        cb_pagetable_id,
        St,
        in0_sequential_mode_semaphore_id,
        cache_position_modulo,
    };
    TensorAccessorArgs(dst_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(update_idxs_tensor.has_value() ? update_idxs_tensor->buffer() : nullptr)
        .append_to(reader_compile_time_args);
    TensorAccessorArgs(page_table.has_value() ? page_table->buffer() : nullptr).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)output_cb_index,
        (std::uint32_t)intermed0_cb_index,
        (std::uint32_t)intermed1_cb_index,
        (std::uint32_t)intermed2_cb_index,
        // Index tensor args
        (std::uint32_t)use_index_tensor,
        cb_index_id,
        cache_batch_num_tiles,
        Wt,
        Wbytes,
        // page_table args
        (std::uint32_t)is_paged_cache,
        (std::uint32_t)num_heads,
        (std::uint32_t)block_size,
        (std::uint32_t)block_size_t,
        (std::uint32_t)max_blocks_per_seq,
        cb_pagetable_id,
        St,
        in0_sequential_mode_semaphore_id,
        cache_position_modulo,
    };
    TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_args);

    std::vector<uint32_t> compute_kernel_args = {
        src0_cb_index,
        src1_cb_index,
        intermed0_cb_index,
        intermed1_cb_index,
        intermed2_cb_index,
        output_cb_index,
        Wt,
        num_heads,
    };

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
        "reader_update_cache_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
        "writer_update_cache_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/update_cache.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = std::move(compute_kernel_args);
    compute_desc.config = ComputeConfigDescriptor{.fp32_dest_acc_en = fp32_dest_acc_en};

    Buffer* const index_buffer_for_rt = use_index_tensor ? update_idxs_tensor.value().buffer() : nullptr;
    Buffer* const page_table_buffer_for_rt = is_paged_cache ? page_table.value().buffer() : nullptr;

    const auto& cores = corerange_to_cores(all_cores, num_cores, row_major);
    // cache_start_id / tile_update_offset_B are derived from update_idxs (excluded from the program
    // hash) — computed via the shared helper; override_runtime_arguments re-runs this on cache hits to
    // re-apply identical values. Empty in index-tensor mode (offsets read on-device from the index tensor).
    const auto offsets = compute_update_cache_offsets(operation_attributes, tensor_args);
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);
        const uint32_t cache_start_id = use_index_tensor ? 0u : offsets.at(i).cache_start_id;
        const uint32_t tile_update_offset_B = use_index_tensor ? 0u : offsets.at(i).tile_update_offset_B;

        bool wait_to_start, send_signal;
        uint32_t send_core_x, send_core_y;
        if (operation_attributes.share_cache) {
            // Share cache
            wait_to_start = i != 0;
            send_signal = i != num_cores - 1;
            auto next_core = i == num_cores - 1 ? core : cores.at(i + 1);
            auto next_core_physical = device->worker_core_from_logical_core(next_core);
            send_core_x = next_core_physical.x;
            send_core_y = next_core_physical.y;
        } else {
            wait_to_start = false;
            send_signal = false;
            send_core_x = 0;
            send_core_y = 0;
        }

        {
            KernelDescriptor::RTArgList rargs;
            rargs.push_back(dst_buffer);
            rargs.push_back(use_index_tensor ? 0u : cache_start_id);
            if (use_index_tensor) {
                rargs.push_back(index_buffer_for_rt);
            } else {
                rargs.push_back(uint32_t{0});
            }
            rargs.push_back(i);
            if (is_paged_cache) {
                rargs.push_back(page_table_buffer_for_rt);
            } else {
                rargs.push_back(uint32_t{0});
            }
            rargs.push_back(static_cast<uint32_t>(wait_to_start));
            reader_desc.emplace_runtime_args(core, rargs);
        }

        writer_desc.emplace_runtime_args(
            core,
            {
                dst_buffer,
                use_index_tensor ? 0u : cache_start_id,
                use_index_tensor ? 0u : tile_update_offset_B,
                i,
                static_cast<uint32_t>(send_signal),
                send_core_x,
                send_core_y,
            });
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

ProgramDescriptor PagedUpdateCacheMeshWorkloadFactory::create_descriptor(
    const PagedUpdateCacheParams& operation_attributes,
    const PagedUpdateCacheInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    if (operation_attributes.mesh_coords.has_value() && mesh_dispatch_coordinate.has_value()) {
        const auto& mesh_coords_set = operation_attributes.mesh_coords.value();
        if (!mesh_coords_set.contains(mesh_dispatch_coordinate.value())) {
            return ProgramDescriptor{};
        }
    }
    return PagedUpdateCacheProgramFactory::create_descriptor(operation_attributes, tensor_args, tensor_return_value);
}

void PagedUpdateCacheDeviceOperation::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    // Re-derive the descriptor from the single source of truth (create_descriptor) for the current tensors
    // and re-apply its per-core args + tensor-backed CB/buffer addresses. Mirrors select_program_factory
    // (mesh factory emits an empty descriptor for excluded coords); supersedes get_dynamic/resolve_bindings.
    auto desc =
        operation_attributes.mesh_coords.has_value()
            ? PagedUpdateCacheMeshWorkloadFactory::create_descriptor(
                  operation_attributes, tensor_args, tensor_return_value, mesh_dispatch_coordinate)
            : PagedUpdateCacheProgramFactory::create_descriptor(operation_attributes, tensor_args, tensor_return_value);
    tt::tt_metal::apply_descriptor_runtime_args(program, desc);
}

}  // namespace ttnn::experimental::prim
