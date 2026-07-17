// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "paged_row_major_fused_update_cache_program_factory.hpp"

#include "paged_fused_update_cache_device_operation_types.hpp"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;

namespace CMAKE_UNIQUE_NAMESPACE_ROW_MAJOR {

bool enable_fp32_dest_acc(
    const tt_metal::IDevice* device, const ttnn::DeviceComputeKernelConfig& compute_kernel_config) {
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    return fp32_dest_acc_en;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE_ROW_MAJOR

std::vector<PagedRowMajorFusedUpdateCacheProgramFactory::PerIndexOffsets>
PagedRowMajorFusedUpdateCacheProgramFactory::compute_row_major_fused_offsets(
    const PagedFusedUpdateCacheParams& operation_attributes, const PagedFusedUpdateCacheInputs& tensor_args) {
    // cache_start_id / tile_update_offset_B are derived from update_idxs, which is excluded from the
    // program hash (see PagedFusedUpdateCacheDeviceOperation::compute_program_hash) yet baked into runtime
    // args, so they are re-derived on every cache hit: override_runtime_arguments re-runs create_descriptor,
    // which calls this helper (the single source of truth for the formulas). Returns empty when an
    // index tensor is used: in that mode the offsets are 0 here and the real positions are read on-device
    // from the (Buffer-bound, re-patched) index tensor.
    if (tensor_args.update_idxs_tensor.has_value()) {
        return {};
    }

    const auto& cache_tensor1 = tensor_args.cache_tensor1;
    const auto& input_tensor1 = tensor_args.input_tensor1;
    const auto& input_tensor2 = tensor_args.input_tensor2;
    const bool fp32_dest_acc_en = CMAKE_UNIQUE_NAMESPACE_ROW_MAJOR::enable_fp32_dest_acc(
        input_tensor1.device(), operation_attributes.compute_kernel_config);

    const uint32_t Wt = cache_tensor1.padded_shape()[-1] / TILE_WIDTH;
    const uint32_t Wbytes = fp32_dest_acc_en ? cache_tensor1.padded_shape()[-1] * sizeof(float)
                                             : cache_tensor1.padded_shape()[-1] * 2;  // 2 bytes for bfloat16
    const uint32_t cache_total_num_tiles = cache_tensor1.physical_volume() / TILE_HW;
    // share_cache => batch offset is 0 (one shared cache buffer); mirror create_descriptor exactly.
    const uint32_t cache_batch_num_tiles =
        operation_attributes.share_cache ? 0 : cache_total_num_tiles / cache_tensor1.padded_shape()[0];

    const bool row_major = input_tensor1.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR;
    const CoreRangeSet input1_cores = input_tensor1.shard_spec().value().grid;
    const CoreRangeSet input2_cores = input_tensor2.shard_spec().value().grid;
    const auto& cores1 = corerange_to_cores(input1_cores, input1_cores.num_cores(), row_major);
    const auto& cores2 = corerange_to_cores(input2_cores, input2_cores.num_cores(), row_major);

    std::vector<PerIndexOffsets> offsets;
    offsets.reserve(cores1.size());
    for (uint32_t i = 0; i < cores1.size(); ++i) {
        const uint32_t update_idx = operation_attributes.update_idxs.at(i);
        const uint32_t cache_batch_tile_offset = i * cache_batch_num_tiles;
        const uint32_t cache_start_id = cache_batch_tile_offset + ((update_idx / TILE_HEIGHT) * Wt);
        const uint32_t tile_update_offset_B = update_idx % TILE_HEIGHT * Wbytes;
        offsets.push_back({cores1.at(i), cores2.at(i), cache_start_id, tile_update_offset_B});
    }
    return offsets;
}

ProgramDescriptor PagedRowMajorFusedUpdateCacheProgramFactory::create_descriptor(
    const PagedFusedUpdateCacheParams& operation_attributes,
    const PagedFusedUpdateCacheInputs& tensor_args,
    PagedFusedUpdateCacheResult& /*tensor_return_value*/) {
    ProgramDescriptor desc;

    const auto& cache_tensor1 = tensor_args.cache_tensor1;
    const auto& input_tensor1 = tensor_args.input_tensor1;
    const auto& cache_tensor2 = tensor_args.cache_tensor2;
    const auto& input_tensor2 = tensor_args.input_tensor2;
    const auto& update_idxs_tensor = tensor_args.update_idxs_tensor;
    const auto& page_table = tensor_args.page_table;

    tt_metal::IDevice* device = input_tensor1.device();

    const tt::DataFormat cache_cb_data_format = tt_metal::datatype_to_dataformat_converter(cache_tensor1.dtype());
    const uint32_t cache_single_tile_size = tt::tile_size(cache_cb_data_format);

    const tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor1.dtype());
    const uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    const bool fp32_dest_acc_en =
        CMAKE_UNIQUE_NAMESPACE_ROW_MAJOR::enable_fp32_dest_acc(device, operation_attributes.compute_kernel_config);

    const tt::DataFormat interm_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t interm_single_tile_size = tt::tile_size(interm_cb_data_format);

    const uint32_t B = input_tensor1.padded_shape()[1];
    const uint32_t num_heads = cache_tensor1.padded_shape()[1];

    // Index tensor-specific parameters
    const bool use_index_tensor = update_idxs_tensor.has_value();
    const uint32_t log2_page_size = 0;
    uint32_t index_stick_size = 0;
    tt::DataFormat index_data_format = tt::DataFormat::Int32;
    Buffer* index_buffer_ptr = nullptr;
    bool index_is_dram = true;
    if (use_index_tensor) {
        index_buffer_ptr = update_idxs_tensor.value().is_sharded() ? update_idxs_tensor.value().buffer() : nullptr;
        index_data_format = tt_metal::datatype_to_dataformat_converter(update_idxs_tensor.value().dtype());
        index_is_dram = update_idxs_tensor.value().buffer()->buffer_type() == tt_metal::BufferType::DRAM;
        index_stick_size = update_idxs_tensor.value().buffer()->aligned_page_size();
    }

    // Pagetable-specific parameters
    const bool is_paged_cache = page_table.has_value();
    uint32_t block_size = 0;
    uint32_t block_size_t = 0;
    uint32_t max_blocks_per_seq = 0;
    uint32_t page_table_stick_size = 0;
    const uint32_t log2_page_table_stick_size = 0;
    tt::DataFormat page_table_data_format = tt::DataFormat::Int32;
    bool page_table_is_dram = true;
    uint32_t num_pages_page_table = 1;
    Buffer* page_table_buffer_ptr = nullptr;
    if (is_paged_cache) {
        const auto& page_table_tensor = page_table.value();
        page_table_buffer_ptr = page_table.value().is_sharded() ? page_table_tensor.buffer() : nullptr;
        num_pages_page_table = page_table.value().is_sharded() ? B : 1;
        block_size = cache_tensor1.padded_shape()[2];
        block_size_t = block_size / TILE_HEIGHT;
        max_blocks_per_seq = page_table_tensor.padded_shape()[1];
        page_table_stick_size = page_table.value().buffer()->aligned_page_size();
        page_table_data_format = tt_metal::datatype_to_dataformat_converter(page_table_tensor.dtype());
        page_table_is_dram = page_table_tensor.buffer()->buffer_type() == tt_metal::BufferType::DRAM;
    }

    const uint32_t Wt = cache_tensor1.padded_shape()[-1] / TILE_WIDTH;
    const uint32_t St = cache_tensor1.padded_shape()[-2] / TILE_HEIGHT;
    const uint32_t Wbytes = fp32_dest_acc_en ? cache_tensor1.padded_shape()[-1] * sizeof(float)
                                             : cache_tensor1.padded_shape()[-1] * 2;  // 2 bytes for bfloat16
    const uint32_t cache_total_num_tiles = cache_tensor1.physical_volume() / TILE_HW;
    const uint32_t cache_batch_num_tiles =
        operation_attributes.share_cache
            ? 0
            : cache_total_num_tiles /
                  cache_tensor1.padded_shape()[0];  // if share cache, we can set cache batch num tiles to 0
                                                    // so batch offset would be 0 in future calculations

    log_debug(tt::LogOp, "cache_cb_data_format: {}", cache_cb_data_format);
    log_debug(tt::LogOp, "input_cb_data_format: {}", input_cb_data_format);
    log_debug(tt::LogOp, "interm_cb_data_format: {}", interm_cb_data_format);
    log_debug(tt::LogOp, "Wbytes: {}", Wbytes);
    log_debug(tt::LogOp, "Wt: {}", Wt);
    log_debug(tt::LogOp, "St: {}", St);

    const auto& input1_shard_spec_opt = input_tensor1.shard_spec();
    const auto& input2_shard_spec_opt = input_tensor2.shard_spec();

    TT_FATAL(input1_shard_spec_opt.has_value(), "input1_shard_spec is not available");
    TT_FATAL(input2_shard_spec_opt.has_value(), "input2_shard_spec is not available");

    const auto& input1_shard_spec = input1_shard_spec_opt.value();
    const auto& input2_shard_spec = input2_shard_spec_opt.value();

    bool row_major = input1_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    const CoreRangeSet input1_cores = input1_shard_spec.grid;
    const CoreRangeSet input2_cores = input2_shard_spec.grid;
    const CoreRangeSet all_cores = input1_cores.merge(input2_cores);
    const CoreRangeSet all_cores_bb = all_cores.bounding_box();
    const CoreRangeSet unused_cores = all_cores_bb.subtract(all_cores);

    const uint32_t num_input_tiles = input1_shard_spec.shape[0] * input1_shard_spec.shape[1] / TILE_HW;

    auto* const in1_buffer = input_tensor1.buffer();
    auto* const in2_buffer = input_tensor2.buffer();

    const uint32_t num_cache_tiles = 2 * Wt;   // double buffered
    const uint32_t num_interm_tiles = 2 * Wt;  // double buffered
    const uint32_t num_output_tiles = B * Wt;

    const tt::CBIndex cache_cb_index = CBIndex::c_0;
    const tt::CBIndex src1_cb_index = CBIndex::c_1;
    const tt::CBIndex src2_cb_index = CBIndex::c_2;
    const tt::CBIndex cb_index_id = CBIndex::c_3;
    const tt::CBIndex cb_pagetable_id = CBIndex::c_4;
    const tt::CBIndex intermed0_cb_index = CBIndex::c_5;
    const tt::CBIndex intermed1_cb_index = CBIndex::c_6;
    const tt::CBIndex output_cb_index = CBIndex::c_7;

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cache_tiles * cache_single_tile_size,
        .core_ranges = all_cores_bb,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cache_cb_index),
            .data_format = cache_cb_data_format,
            .page_size = cache_single_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * input_single_tile_size,
        .core_ranges = input1_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src1_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
        .buffer = in1_buffer,
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * input_single_tile_size,
        .core_ranges = input2_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src2_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
        .buffer = in2_buffer,
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_interm_tiles * interm_single_tile_size,
        .core_ranges = all_cores_bb,
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
        .total_size = num_output_tiles * cache_single_tile_size,
        .core_ranges = all_cores_bb,
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
        .core_ranges = all_cores_bb,
        .initial_value = 0,
    });

    if (use_index_tensor) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = index_stick_size,
            .core_ranges = all_cores_bb,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_index_id),
                .data_format = index_data_format,
                .page_size = index_stick_size,
            }}},
            .buffer = index_buffer_ptr,
        });
    }

    if (is_paged_cache) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_pages_page_table * page_table_stick_size,
            .core_ranges = all_cores_bb,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_pagetable_id),
                .data_format = page_table_data_format,
                .page_size = page_table_stick_size,
            }}},
            .buffer = page_table_buffer_ptr,
        });
    }

    auto* const dst1_buffer = cache_tensor1.buffer();

    auto* const dst2_buffer = cache_tensor2.buffer();

    std::vector<uint32_t> reader_compile_time_args = {
        src1_cb_index,
        src2_cb_index,
        cache_cb_index,
        // Index tensor args
        use_index_tensor,
        index_is_dram,
        cb_index_id,
        cache_batch_num_tiles,
        Wt,
        log2_page_size,
        index_stick_size,
        // page_table args
        is_paged_cache,
        num_heads,
        block_size,
        block_size_t,
        max_blocks_per_seq,
        log2_page_table_stick_size,
        page_table_stick_size,
        page_table_is_dram,
        cb_pagetable_id,
        St,
        in0_sequential_mode_semaphore_id,
        B};
    TensorAccessorArgs(dst1_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(update_idxs_tensor.has_value() ? update_idxs_tensor->buffer() : nullptr)
        .append_to(reader_compile_time_args);
    TensorAccessorArgs(page_table.has_value() ? page_table->buffer() : nullptr).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        output_cb_index,
        intermed0_cb_index,
        intermed1_cb_index,
        src1_cb_index,
        src2_cb_index,
        // Index tensor args
        use_index_tensor,
        cb_index_id,
        cache_batch_num_tiles,
        Wt,
        Wbytes,
        // page_table args
        is_paged_cache,
        num_heads,
        block_size,
        block_size_t,
        max_blocks_per_seq,
        cb_pagetable_id,
        St,
        in0_sequential_mode_semaphore_id,
        B,
        page_table_stick_size,
        page_table_is_dram};
    TensorAccessorArgs(dst1_buffer).append_to(writer_compile_time_args);

    std::vector<uint32_t> compute_kernel_args = {
        src1_cb_index,
        src2_cb_index,
        cache_cb_index,
        intermed0_cb_index,
        intermed1_cb_index,
        output_cb_index,
        Wt,
        num_heads,
    };

    // Create reader kernel
    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
        "reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores_bb;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    // Create writer kernel
    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
        "writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores_bb;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Create compute kernel
    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/"
        "paged_row_major_fused_update_cache.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores_bb;
    compute_desc.compile_time_args = std::move(compute_kernel_args);
    compute_desc.config = ComputeConfigDescriptor{.fp32_dest_acc_en = fp32_dest_acc_en};

    constexpr bool has_work = true;
    constexpr bool is_input1 = true;

    const auto& cores1 = corerange_to_cores(input1_cores, input1_cores.num_cores(), row_major);
    const auto& cores2 = corerange_to_cores(input2_cores, input2_cores.num_cores(), row_major);

    Buffer* const index_buffer_for_rt = use_index_tensor ? update_idxs_tensor.value().buffer() : nullptr;
    Buffer* const page_table_buffer_for_rt = is_paged_cache ? page_table.value().buffer() : nullptr;

    // cache_start_id / tile_update_offset_B are derived from update_idxs (excluded from the program hash)
    // — computed via the shared helper; override_runtime_arguments re-runs create_descriptor to re-apply
    // them on cache hits. Empty in index-tensor mode (offsets read on-device from the re-patched index tensor).
    const auto offsets = compute_row_major_fused_offsets(operation_attributes, tensor_args);
    for (uint32_t i = 0; i < cores1.size(); ++i) {
        const CoreCoord& core1 = cores1.at(i);
        const CoreCoord& core2 = cores2.at(i);

        // Cache tile info
        const uint32_t cache_start_id = use_index_tensor ? 0u : offsets.at(i).cache_start_id;
        const uint32_t tile_update_offset_B = use_index_tensor ? 0u : offsets.at(i).tile_update_offset_B;

        // Calculate synchronization parameters
        const bool wait_to_start = operation_attributes.share_cache and (i != 0);
        const bool send_signal = operation_attributes.share_cache and (i != cores1.size() - 1);
        uint32_t send_core1_x = 0, send_core1_y = 0;
        uint32_t send_core2_x = 0, send_core2_y = 0;

        if (send_signal) {
            auto next_core = cores1.at(i + 1);
            auto next_core_physical = device->worker_core_from_logical_core(next_core);
            send_core1_x = next_core_physical.x;
            send_core1_y = next_core_physical.y;

            next_core = cores2.at(i + 1);
            next_core_physical = device->worker_core_from_logical_core(next_core);
            send_core2_x = next_core_physical.x;
            send_core2_y = next_core_physical.y;
        }

        // Input1 args
        // Set runtime args for reader
        {
            KernelDescriptor::RTArgList rargs;
            rargs.push_back(static_cast<uint32_t>(has_work));
            rargs.push_back(static_cast<uint32_t>(is_input1));
            rargs.push_back(dst1_buffer);
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
            reader_desc.emplace_runtime_args(core1, rargs);
        }

        // Set runtime args for writer
        writer_desc.emplace_runtime_args(
            core1,
            {
                static_cast<uint32_t>(has_work),
                dst1_buffer,
                use_index_tensor ? 0u : cache_start_id,
                use_index_tensor ? 0u : tile_update_offset_B,
                i,
                static_cast<uint32_t>(send_signal),
                send_core1_x,
                send_core1_y,
                static_cast<uint32_t>(is_input1),
            });

        // Set runtime args for compute
        compute_desc.emplace_runtime_args(
            core1,
            {
                static_cast<uint32_t>(has_work),
                static_cast<uint32_t>(is_input1),
            });

        // Input2 args
        // Set runtime args for reader
        {
            KernelDescriptor::RTArgList rargs;
            rargs.push_back(static_cast<uint32_t>(has_work));
            rargs.push_back(static_cast<uint32_t>(!is_input1));
            rargs.push_back(dst2_buffer);
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
            reader_desc.emplace_runtime_args(core2, rargs);
        }

        // Set runtime args for writer
        writer_desc.emplace_runtime_args(
            core2,
            {
                static_cast<uint32_t>(has_work),
                dst2_buffer,
                use_index_tensor ? 0u : cache_start_id,
                use_index_tensor ? 0u : tile_update_offset_B,
                i,
                static_cast<uint32_t>(send_signal),
                send_core2_x,
                send_core2_y,
                static_cast<uint32_t>(!is_input1),
            });

        // Set runtime args for compute
        compute_desc.emplace_runtime_args(
            core2,
            {
                static_cast<uint32_t>(has_work),
                static_cast<uint32_t>(!is_input1),
            });
    }

    // Set runtime args for unused cores
    for (const auto& core_range : unused_cores.ranges()) {
        for (const auto& core : core_range) {
            reader_desc.emplace_runtime_args(core, {uint32_t{!has_work}});
            writer_desc.emplace_runtime_args(core, {uint32_t{!has_work}});
            compute_desc.emplace_runtime_args(core, {uint32_t{!has_work}});
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

ProgramDescriptor PagedRowMajorFusedUpdateCacheMeshWorkloadFactory::create_descriptor(
    const PagedFusedUpdateCacheParams& operation_attributes,
    const PagedFusedUpdateCacheInputs& tensor_args,
    PagedFusedUpdateCacheResult& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    // When mesh_coords is provided, coords outside that set get an empty program
    // (the legacy mesh path skipped them entirely; with the descriptor framework
    // every dispatched coord receives a descriptor, so an empty one is the
    // closest equivalent).
    if (operation_attributes.mesh_coords.has_value() && mesh_dispatch_coordinate.has_value()) {
        const auto& mesh_coords_set = operation_attributes.mesh_coords.value();
        if (!mesh_coords_set.contains(mesh_dispatch_coordinate.value())) {
            return ProgramDescriptor{};
        }
    }
    return PagedRowMajorFusedUpdateCacheProgramFactory::create_descriptor(
        operation_attributes, tensor_args, tensor_return_value);
}

}  // namespace ttnn::experimental::prim
