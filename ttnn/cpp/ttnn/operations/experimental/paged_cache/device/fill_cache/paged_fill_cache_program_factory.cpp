// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "paged_fill_cache_program_factory.hpp"

#include "paged_fill_cache_device_operation.hpp"
#include "paged_fill_cache_device_operation_types.hpp"

#include <cmath>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;

namespace {

// Worker-core list for the fill_cache work-split. batch_idx_fallback (excluded from the program
// hash, baked into writer runtime args) is the SAME value on every core, so re-patching it on a
// cache hit only needs the core *ordering* — not the per-core block counts. This helper is the
// single source of truth for that ordering: both build_paged_fill_cache_descriptor (cache miss)
// and PagedFillCacheDeviceOperation::get_dynamic_runtime_args (cache hit) call it, so the two
// paths cannot drift in which cores they touch or in what order.
std::vector<tt_metal::CoreCoord> compute_paged_fill_cache_cores(
    const PagedFillCacheParams& /*operation_attributes*/, const PagedFillCacheInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    // num_blocks_of_work mirrors build_paged_fill_cache_descriptor: input_batch * num_heads *
    // input_seq_len_t. block_size / cache geometry does not influence the work-split, so it is
    // intentionally omitted here.
    const uint32_t input_batch = input_tensor.padded_shape()[0];
    const uint32_t num_heads = input_tensor.padded_shape()[1];
    const uint32_t input_seq_len = input_tensor.padded_shape()[2];
    const uint32_t input_seq_len_t = input_seq_len / TILE_HEIGHT;
    const uint32_t num_blocks_of_work = input_batch * num_heads * input_seq_len_t;

    tt_metal::IDevice* device = input_tensor.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    const bool row_major = true;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks_of_work, row_major);

    return grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);
}

// Build the per-coord descriptor.  Shared by the single-device and mesh-workload
// factories; the mesh path passes a possibly-overridden `noop` (true for
// coordinates excluded from operation_attributes.mesh_coords).
ProgramDescriptor build_paged_fill_cache_descriptor(
    const PagedFillCacheParams& operation_attributes, const PagedFillCacheInputs& tensor_args, bool noop) {
    ProgramDescriptor desc;

    const auto& cache_tensor = tensor_args.cache_tensor;
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& page_table_tensor = tensor_args.page_table;
    const auto& batch_idx_tensor = tensor_args.batch_idx_tensor_opt;

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    // input_tensor:      [input_batch, num_heads, input_seq_len, head_dim]
    //   input_batch == 1 on the legacy single-batch path; input_batch == N
    //   on the batched path, where N matches batch_idx_tensor element count.
    // cache_tensor:      [max_num_blocks, num_kv_heads, block_size, head_dim]
    // page_table_tensor: [b, max_num_blocks_per_seq]
    //
    // head_dim comes from the input and block_size honors the override; the cache shape
    // is only a byte budget (per-block byte count enforced in validate).
    const uint32_t input_batch = input_tensor.padded_shape()[0];
    const uint32_t num_heads = input_tensor.padded_shape()[1];
    const uint32_t input_seq_len = input_tensor.padded_shape()[2];

    const uint32_t block_size = operation_attributes.block_size_override.value_or(cache_tensor.padded_shape()[2]);
    const uint32_t head_dim = input_tensor.padded_shape()[3];

    const uint32_t input_seq_len_t = input_seq_len / TILE_HEIGHT;
    const uint32_t Wt = head_dim / TILE_WIDTH;
    const uint32_t block_size_t = block_size / TILE_HEIGHT;

    // Each "block of work" is one (batch, head, seq_tile) triple to write.
    // num_blocks_of_work_per_batch lets the writer kernel recover the batch
    // index for the batched path; on the legacy path input_batch == 1 so
    // num_blocks_of_work == num_blocks_of_work_per_batch.
    const uint32_t num_blocks_of_work_per_batch = num_heads * input_seq_len_t;
    const uint32_t num_blocks_of_work = input_batch * num_blocks_of_work_per_batch;
    const uint32_t num_blocks_of_work_per_head = input_seq_len_t;

    // Pagetable-specific parameters
    uint32_t page_table_stick_size_B = page_table_tensor.buffer()->aligned_page_size();
    TT_FATAL(
        page_table_stick_size_B % 32 == 0,
        "page table page size in bytes must be a multiple of 32 due to address alignment");
    uint32_t log2_page_table_stick_size_B = std::log2(page_table_stick_size_B);
    tt::DataFormat page_table_data_format = tt_metal::datatype_to_dataformat_converter(page_table_tensor.dtype());

    // batch_idx_tensor specific parameters. When provided, the tensor's
    // element count must equal input_batch: one batch_idx per input batch
    // row. The legacy single-batch case (input_batch == 1, tensor.shape ==
    // [1]) falls out naturally.
    const bool use_batch_idx_tensor = batch_idx_tensor.has_value();
    tt::DataFormat batch_idx_data_format = tt::DataFormat::UInt32;
    uint32_t batch_idx_stick_size_B = 4;  // per-element size, e.g. 4 for uint32
    uint32_t batch_idx_num_elements = 1;

    if (use_batch_idx_tensor) {
        const auto& tensor = batch_idx_tensor.value();
        batch_idx_data_format = tt_metal::datatype_to_dataformat_converter(tensor.dtype());
        batch_idx_stick_size_B = tensor.element_size();
        batch_idx_num_elements = tensor.physical_volume();
        TT_FATAL(
            batch_idx_num_elements == input_batch,
            "batch_idx_tensor must contain input_batch ({}) elements, got {}",
            input_batch,
            batch_idx_num_elements);
    } else {
        // No batch_idx_tensor: scalar fallback path writes one batch row,
        // so input_batch must be 1. Previously implicit; explicit FATAL
        // avoids silently dropping rows > 0.
        TT_FATAL(
            input_batch == 1,
            "When no batch_idx_tensor is provided, input_batch must be 1 (got {}); pass a batch_idx_tensor of size "
            "input_batch to fill multiple batch rows in one call.",
            input_batch);
    }

    tt_metal::IDevice* device = input_tensor.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    bool row_major;
    uint32_t num_cores, num_blocks_per_core_group_1, num_blocks_per_core_group_2;

    CoreRangeSet all_cores, core_group_1, core_group_2;

    row_major = true;
    std::tie(
        num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2) =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks_of_work, row_major);
    uint32_t num_input_tiles = Wt * 2;  // double buffered

    tt::CBIndex src0_cb_index = tt::CBIndex::c_0;
    tt::CBIndex page_table_cb_index = tt::CBIndex::c_1;
    tt::CBIndex cb_batch_idx_id = tt::CBIndex::c_2;  // New CB for batch_idx_tensor

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = page_table_stick_size_B,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(page_table_cb_index),
            .data_format = page_table_data_format,
            .page_size = page_table_stick_size_B,
        }}},
    });
    if (use_batch_idx_tensor) {
        // CB holds all `batch_idx_num_elements` entries so the writer kernel
        // can pick the right entry per batch row in the batched case.
        desc.cbs.push_back(CBDescriptor{
            .total_size = batch_idx_stick_size_B * batch_idx_num_elements,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_batch_idx_id),
                .data_format = batch_idx_data_format,
                .page_size = batch_idx_stick_size_B,
            }}},
        });
    }

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = cache_tensor.buffer();
    auto* page_table_buffer = page_table_tensor.buffer();

    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_cb_index, Wt};
    TensorAccessorArgs(src_buffer).append_to(reader_compile_time_args);

    // capacity_t (in TILE rows; 0 = unbounded/legacy) wraps seq_tile_id mod this value
    // before page_table lookup. cache_position_modulo % effective_block_size == 0 is
    // enforced in the validator, so the divide is exact.
    const uint32_t capacity_t = operation_attributes.cache_position_modulo.value_or(0u) / TILE_HEIGHT;

    std::vector<uint32_t> writer_compile_time_args = {
        (uint32_t)src0_cb_index,
        (uint32_t)page_table_cb_index,
        num_heads,
        num_blocks_of_work_per_head,
        block_size_t,
        Wt,
        log2_page_table_stick_size_B,
        page_table_stick_size_B,
        // batch_idx_tensor compile-time args (positions 8..12). Positions 9..12
        // are only meaningful when use_batch_idx_tensor is true.
        (uint32_t)use_batch_idx_tensor,
        cb_batch_idx_id,
        batch_idx_stick_size_B,        // per-element size, e.g. 4 for uint32
        batch_idx_num_elements,        // 1 = legacy single-batch, N = batched
        num_blocks_of_work_per_batch,  // num_heads * input_seq_len_t, for row_id -> batch decode
        capacity_t,
    };
    TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_args);
    TensorAccessorArgs(page_table_buffer).append_to(writer_compile_time_args);
    TensorAccessorArgs(batch_idx_tensor.has_value() ? batch_idx_tensor->buffer() : nullptr)
        .append_to(writer_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/reader_fill_cache_interleaved.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_fill_cache_interleaved.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();

    // Core list shared with get_dynamic_runtime_args (single source of truth for ordering).
    const auto cores = compute_paged_fill_cache_cores(operation_attributes, tensor_args);

    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_blocks_per_core = 0;
        if (i < g1_numcores) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else if (i < g1_numcores + g2_numcores) {
            num_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            num_blocks_per_core = 0;
        }

        reader_desc.emplace_runtime_args(
            core,
            {
                src_buffer,
                num_blocks_written * Wt,  // start_tile_id
                num_blocks_per_core,      // num_rows
                (uint32_t)noop,           // noop flag
            });

        // batch_idx_tensor_addr (Buffer*) or batch_idx_fallback (uint32_t).  Use
        // emplace_runtime_args so the buffer base address is patched on cache hits.
        KernelDescriptor::RTArgList writer_args;
        writer_args.push_back(dst_buffer);
        writer_args.push_back(page_table_buffer);
        writer_args.push_back(num_blocks_written);   // start_row_num
        writer_args.push_back(num_blocks_per_core);  // num_rows
        if (use_batch_idx_tensor) {
            writer_args.push_back(batch_idx_tensor->buffer());  // batch_idx_tensor_addr
        } else {
            writer_args.push_back(operation_attributes.batch_idx_fallback);  // batch_idx_fallback
        }
        writer_args.push_back(static_cast<uint32_t>(noop));  // noop flag
        writer_desc.emplace_runtime_args(core, writer_args);
        num_blocks_written += num_blocks_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace

ProgramDescriptor PagedFillCacheProgramFactory::create_descriptor(
    const PagedFillCacheParams& operation_attributes,
    const PagedFillCacheInputs& tensor_args,
    Tensor& /*tensor_return_value*/) {
    return build_paged_fill_cache_descriptor(operation_attributes, tensor_args, operation_attributes.noop);
}

ProgramDescriptor PagedFillCacheMeshWorkloadFactory::create_descriptor(
    const PagedFillCacheParams& operation_attributes,
    const PagedFillCacheInputs& tensor_args,
    Tensor& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    // When mesh_coords is provided, coordinates outside that set get a noop
    // program (kernels early-exit).  This preserves the legacy behavior of
    // dispatching a "dummy" program to every device in the mesh range so the
    // cached workload covers all coords.
    bool noop = operation_attributes.noop;
    if (operation_attributes.mesh_coords.has_value() && mesh_dispatch_coordinate.has_value()) {
        const auto& mesh_coords_set = operation_attributes.mesh_coords.value();
        if (!mesh_coords_set.contains(mesh_dispatch_coordinate.value())) {
            noop = true;
        }
    }
    return build_paged_fill_cache_descriptor(operation_attributes, tensor_args, noop);
}

std::vector<tt::tt_metal::DynamicRuntimeArg> PagedFillCacheDeviceOperation::get_dynamic_runtime_args(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    // Coords excluded from a mesh dispatch build a noop program (kernels early-exit), so there is
    // nothing meaningful to re-patch there.
    if (operation_attributes.mesh_coords.has_value() && mesh_dispatch_coordinate.has_value() &&
        !operation_attributes.mesh_coords.value().contains(mesh_dispatch_coordinate.value())) {
        return {};
    }

    // batch-idx-tensor mode: the writer pushes the batch_idx tensor's Buffer* (writer arg 4), which
    // the framework already re-patches by buffer base address. Nothing op-specific to re-apply.
    if (tensor_args.batch_idx_tensor_opt.has_value()) {
        return {};
    }

    // Scalar-fallback mode: batch_idx_fallback is excluded from the program hash (so two calls
    // differing only in it cache-hit) yet baked into writer runtime arg index 4 — re-patch it on
    // every dispatch or it freezes at the first cache-miss value. It is the SAME value on every
    // core (operation_attributes.batch_idx_fallback, not per-core), so we emit one arg per core
    // using the shared core-list helper (single source of truth for core ordering).
    //
    // noop is intentionally NOT re-patched: it is derived from the hashed mesh_coords (the mesh
    // factory sets it per coord from coord-membership), so it is stable across cache hits for a
    // fixed coord and is already correct in the cached program.
    //
    // Kernel push order in build_paged_fill_cache_descriptor: reader(0), writer(1).
    // Writer rt args: [0]=dst, [1]=page_table, [2]=start_row_num, [3]=num_rows,
    //                 [4]=batch_idx_fallback (scalar-fallback mode), [5]=noop.
    constexpr uint32_t kWriterKernelIdx = 1;
    constexpr uint32_t kBatchIdxFallbackArgIdx = 4;

    const auto cores = compute_paged_fill_cache_cores(operation_attributes, tensor_args);
    std::vector<tt::tt_metal::DynamicRuntimeArg> dynamic_args;
    dynamic_args.reserve(cores.size());
    for (const auto& core : cores) {
        dynamic_args.push_back(
            {kWriterKernelIdx, core, kBatchIdxFallbackArgIdx, operation_attributes.batch_idx_fallback});
    }
    return dynamic_args;
}

}  // namespace ttnn::experimental::prim
