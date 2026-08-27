// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "paged_fill_cache_program_factory.hpp"

#include "paged_fill_cache_device_operation.hpp"
#include "paged_fill_cache_device_operation_types.hpp"

#include <cmath>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;

namespace {

// Metal 2.0 spec resource names for this factory.  Prefixed to keep the anonymous namespace free of
// collisions when the op's factory .cpp files are unity-built into one translation unit.
const DFBSpecName FC_INPUT_DFB{"input"};
const DFBSpecName FC_PAGE_TABLE_DFB{"page_table"};
const DFBSpecName FC_BATCH_IDX_DFB{"batch_idx"};
const DFBSpecName FC_VALID_SEQ_LEN_DFB{"valid_seq_len"};

const TensorParamName FC_INPUT_TENSOR{"input"};
const TensorParamName FC_CACHE_TENSOR{"cache"};
const TensorParamName FC_PAGE_TABLE_TENSOR{"page_table"};
const TensorParamName FC_BATCH_IDX_TENSOR{"batch_idx"};
const TensorParamName FC_VALID_SEQ_LEN_TENSOR{"valid_seq_len"};

const KernelSpecName FC_READER_KERNEL{"reader"};
const KernelSpecName FC_WRITER_KERNEL{"writer"};

constexpr auto FC_READER_SOURCE =
    "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
    "reader_fill_cache_interleaved_metal2.cpp";
constexpr auto FC_WRITER_SOURCE =
    "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
    "writer_fill_cache_interleaved_metal2.cpp";

// `noop` is the only thing that differs between the single-device and the mesh-workload factory: a
// mesh coordinate outside operation_attributes.mesh_coords gets a noop program (kernels early-exit).
// Single source of truth for that choice, called by both factories' program builds and by their
// cache-hit patches — so the patch mirrors select_program_factory by construction (mesh_coords is
// nullopt on the single-device path, where the coordinate is ignored).
bool paged_fill_cache_noop(
    const PagedFillCacheParams& operation_attributes, const std::optional<ttnn::MeshCoordinate>& coord) {
    if (operation_attributes.mesh_coords.has_value() && coord.has_value() &&
        !operation_attributes.mesh_coords->contains(coord.value())) {
        return true;
    }
    return operation_attributes.noop;
}

// Worker-core list for the fill_cache work-split. Single source of truth for core ordering: called by
// both the program build (cache miss, emitting per-core runtime args) and the cache-hit patch, so the
// two paths cannot drift in which cores they touch or in what order.
std::vector<tt_metal::CoreCoord> compute_paged_fill_cache_cores(
    const PagedFillCacheParams& /*operation_attributes*/, const PagedFillCacheInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    // num_blocks_of_work mirrors the program build: input_batch * num_heads *
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

// ---------------------------------------------------------------------------------------------
// Legacy ProgramDescriptor body.
//
// PagedFillCacheMeshWorkloadFactory still builds a ProgramDescriptor: its per-coordinate `noop`
// value differs across the mesh, and the Metal 2.0 spec factory concepts have no per-coordinate hook
// on the cache-miss path — create_program_artifacts is called once and one ProgramRunArgs is applied
// to every coordinate.  So this body stays and keeps binding the legacy kernel sources.
// ---------------------------------------------------------------------------------------------
ProgramDescriptor build_paged_fill_cache_descriptor(
    const PagedFillCacheParams& operation_attributes, const PagedFillCacheInputs& tensor_args, bool noop) {
    ProgramDescriptor desc;

    const auto& cache_tensor = tensor_args.cache_tensor;
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& page_table_tensor = tensor_args.page_table;
    const auto& batch_idx_tensor = tensor_args.batch_idx_tensor_opt;
    const auto& valid_seq_len_tensor = tensor_args.valid_seq_len_tensor_opt;

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

    // valid_seq_len tensor: optional 1-element int giving the block-aligned real
    // fill length (in tokens). When present, the writer restricts the bounded ring
    // window to end at valid_seq_len instead of the padded input end (see kernel).
    const bool use_valid_seq_len = valid_seq_len_tensor.has_value();
    uint32_t valid_seq_len_stick_size_B = 4;
    if (use_valid_seq_len) {
        valid_seq_len_stick_size_B = valid_seq_len_tensor->element_size();
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
    tt::CBIndex cb_batch_idx_id = tt::CBIndex::c_2;      // New CB for batch_idx_tensor
    tt::CBIndex cb_valid_seq_len_id = tt::CBIndex::c_3;  // CB for valid_seq_len_tensor

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
    if (use_valid_seq_len) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = valid_seq_len_stick_size_B,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_valid_seq_len_id),
                .data_format = tt::DataFormat::UInt32,
                .page_size = valid_seq_len_stick_size_B,
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
        // valid_seq_len_tensor compile-time args (positions 14..16). Position 15..16
        // are only meaningful when use_valid_seq_len is true.
        (uint32_t)use_valid_seq_len,
        cb_valid_seq_len_id,
        valid_seq_len_stick_size_B,
    };
    TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_args);
    TensorAccessorArgs(page_table_buffer).append_to(writer_compile_time_args);
    TensorAccessorArgs(batch_idx_tensor.has_value() ? batch_idx_tensor->buffer() : nullptr)
        .append_to(writer_compile_time_args);
    TensorAccessorArgs(valid_seq_len_tensor.has_value() ? valid_seq_len_tensor->buffer() : nullptr)
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

    // Core list shared with the cache-hit patch (single source of truth for ordering).
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
        // Arg 6: valid_seq_len tensor address (Buffer*, framework re-patches on cache
        // hit / trace replay) or 0 scalar when unused.
        if (use_valid_seq_len) {
            writer_args.push_back(valid_seq_len_tensor->buffer());
        } else {
            writer_args.push_back(static_cast<uint32_t>(0));
        }
        writer_desc.emplace_runtime_args(core, writer_args);
        num_blocks_written += num_blocks_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

// Legacy in-place cache-hit patch, used by PagedFillCacheMeshWorkloadFactory (see the note above
// build_paged_fill_cache_descriptor for why that factory stays on the descriptor concept).
void patch_paged_fill_cache_runtime_args(
    tt::tt_metal::Program& program,
    const PagedFillCacheParams& operation_attributes,
    const PagedFillCacheInputs& tensor_args,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    // Patch the cached program in place. Rebuilding the descriptor here would re-pay the whole
    // cache-MISS host cost on every hit (work-split, CoreRangeSet, TensorAccessorArgs, kernel-source
    // strings, a fresh per-core arg vector for every core) plus a full apply over every kernel x core
    // x arg.
    //
    // Kernel push order in build_paged_fill_cache_descriptor: reader(0), writer(1).
    // reader rt args: [0]=src, [1]=start_tile_id, [2]=num_rows, [3]=noop.
    // writer rt args: [0]=dst, [1]=page_table, [2]=start_row_num, [3]=num_rows,
    //                 [4]=batch_idx_tensor addr | batch_idx_fallback, [5]=noop,
    //                 [6]=valid_seq_len addr | 0.
    constexpr uint32_t kReaderKernelIdx = 0;
    constexpr uint32_t kWriterKernelIdx = 1;

    // Buffer addresses: override supersedes resolve_bindings, so every Buffer* the descriptor
    // emplaced is ours to re-apply. The op is in place (tensor_return_value aliases
    // tensor_args.cache_tensor), so read the same tensors build_paged_fill_cache_descriptor does.
    const auto src_addr = static_cast<uint32_t>(tensor_args.input_tensor.buffer()->address());
    const auto dst_addr = static_cast<uint32_t>(tensor_args.cache_tensor.buffer()->address());
    const auto page_table_addr = static_cast<uint32_t>(tensor_args.page_table.buffer()->address());
    // Optional tensors: an absent one is emplaced as the literal 0 the descriptor pushes.
    const uint32_t valid_seq_len_arg =
        tensor_args.valid_seq_len_tensor_opt.has_value()
            ? static_cast<uint32_t>(tensor_args.valid_seq_len_tensor_opt->buffer()->address())
            : 0u;
    // Writer arg 4 is a buffer address in batch-idx-tensor mode, and otherwise batch_idx_fallback —
    // excluded from the program hash (so calls differing only in it cache-hit) yet baked into the
    // arg, so it freezes at the first miss value unless re-applied here.
    const uint32_t batch_idx_arg = tensor_args.batch_idx_tensor_opt.has_value()
                                       ? static_cast<uint32_t>(tensor_args.batch_idx_tensor_opt->buffer()->address())
                                       : operation_attributes.batch_idx_fallback;
    // noop is hash-excluded too, and on the mesh path depends on the dispatch coordinate.
    const auto noop_arg = static_cast<uint32_t>(paged_fill_cache_noop(operation_attributes, mesh_dispatch_coordinate));

    // Not re-applied: start_tile_id / start_row_num / num_rows. They come from the work split over
    // the input's padded shape and the device grid, both of which the program hash includes, so a
    // cache hit has them identical by construction.
    const auto cores = compute_paged_fill_cache_cores(operation_attributes, tensor_args);
    for (const auto& core : cores) {
        auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx, core);
        reader_args[0] = src_addr;
        reader_args[3] = noop_arg;

        auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx, core);
        writer_args[0] = dst_addr;
        writer_args[1] = page_table_addr;
        writer_args[4] = batch_idx_arg;
        writer_args[5] = noop_arg;
        writer_args[6] = valid_seq_len_arg;
    }
    // No CB addresses to re-point: none of the four CBs is globally allocated (no .buffer/.tensor).
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Metal 2.0 program build.
// ---------------------------------------------------------------------------------------------
ttnn::device_operation::ProgramArtifacts PagedFillCacheProgramFactory::create_program_artifacts(
    const PagedFillCacheParams& operation_attributes,
    const PagedFillCacheInputs& tensor_args,
    Tensor& /*tensor_return_value*/) {
    const auto& cache_tensor = tensor_args.cache_tensor;
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& page_table_tensor = tensor_args.page_table;
    const auto& batch_idx_tensor = tensor_args.batch_idx_tensor_opt;
    const auto& valid_seq_len_tensor = tensor_args.valid_seq_len_tensor_opt;

    // mesh_coords is nullopt on this factory's path (see select_program_factory), so the coordinate
    // is ignored and this resolves to operation_attributes.noop.
    const bool noop = paged_fill_cache_noop(operation_attributes, std::nullopt);

    tt::DataFormat input_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(input_data_format);

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

    // valid_seq_len tensor: optional 1-element int giving the block-aligned real
    // fill length (in tokens). When present, the writer restricts the bounded ring
    // window to end at valid_seq_len instead of the padded input end (see kernel).
    const bool use_valid_seq_len = valid_seq_len_tensor.has_value();
    uint32_t valid_seq_len_stick_size_B = 4;
    if (use_valid_seq_len) {
        valid_seq_len_stick_size_B = valid_seq_len_tensor->element_size();
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

    // capacity_t (in TILE rows; 0 = unbounded/legacy) wraps seq_tile_id mod this value
    // before page_table lookup. cache_position_modulo % effective_block_size == 0 is
    // enforced in the validator, so the divide is exact.
    const uint32_t capacity_t = operation_attributes.cache_position_modulo.value_or(0u) / TILE_HEIGHT;

    ProgramSpec spec;
    spec.name = "paged_fill_cache";

    //-------------------------------------------------------------------------
    // Dataflow buffers
    //-------------------------------------------------------------------------
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = FC_INPUT_DFB,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_data_format,
    });
    // Touched only by the writer: reserved once and then written through a raw pointer, never
    // pushed or popped. The writer is therefore bound as both endpoints (self-loop) — on Gen1 a DFB
    // lowers to a hardware FIFO that one RISC can both fill and drain, so a single-toucher buffer
    // needs no second kernel to be legal.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = FC_PAGE_TABLE_DFB,
        .entry_size = page_table_stick_size_B,
        .num_entries = 1,
        .data_format_metadata = page_table_data_format,
    });
    if (use_batch_idx_tensor) {
        // Holds all `batch_idx_num_elements` entries so the writer kernel can pick the right entry
        // per batch row in the batched case. Writer-only, so likewise a self-loop.
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = FC_BATCH_IDX_DFB,
            .entry_size = batch_idx_stick_size_B,
            .num_entries = batch_idx_num_elements,
            .data_format_metadata = batch_idx_data_format,
        });
    }
    if (use_valid_seq_len) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = FC_VALID_SEQ_LEN_DFB,
            .entry_size = valid_seq_len_stick_size_B,
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::UInt32,
        });
    }

    //-------------------------------------------------------------------------
    // Tensor parameters
    //-------------------------------------------------------------------------
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = FC_INPUT_TENSOR,
        .spec = input_tensor.tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = FC_CACHE_TENSOR,
        .spec = cache_tensor.tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = FC_PAGE_TABLE_TENSOR,
        .spec = page_table_tensor.tensor_spec(),
    });
    if (use_batch_idx_tensor) {
        spec.tensor_parameters.push_back(TensorParameter{
            .unique_id = FC_BATCH_IDX_TENSOR,
            .spec = batch_idx_tensor.value().tensor_spec(),
        });
    }
    if (use_valid_seq_len) {
        spec.tensor_parameters.push_back(TensorParameter{
            .unique_id = FC_VALID_SEQ_LEN_TENSOR,
            .spec = valid_seq_len_tensor.value().tensor_spec(),
        });
    }

    //-------------------------------------------------------------------------
    // Kernels
    //-------------------------------------------------------------------------
    KernelSpec reader{
        .unique_id = FC_READER_KERNEL,
        .source = FC_READER_SOURCE,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = FC_INPUT_DFB,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = FC_INPUT_TENSOR,
                    .accessor_name = "src",
                },
            },
        .compile_time_args = {{"Wt", Wt}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_tile_id", "num_rows", "noop"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // The conditional DFB / tensor bindings are gated kernel-side by these defines rather than by a
    // compile-time arg: `if constexpr` still name-looks-up the discarded branch, so a
    // `dfb::batch_idx` or `tensor::valid_seq_len` the host did not bind would fail to compile.
    KernelSpec::CompilerOptions::Defines writer_defines;
    if (use_batch_idx_tensor) {
        writer_defines.emplace("USE_BATCH_IDX_TENSOR", "1");
    }
    if (use_valid_seq_len) {
        writer_defines.emplace("USE_VALID_SEQ_LEN", "1");
    }

    KernelSpec writer{
        .unique_id = FC_WRITER_KERNEL,
        .source = FC_WRITER_SOURCE,
        .compiler_options = {.defines = writer_defines},
        .compile_time_args =
            {
                {"num_heads", num_heads},
                {"num_blocks_of_work_per_head", num_blocks_of_work_per_head},
                {"block_size_t", block_size_t},
                {"Wt", Wt},
                {"log2_page_table_stick_size", log2_page_table_stick_size_B},
                {"page_table_stick_size", page_table_stick_size_B},
                // Only meaningful when use_batch_idx_tensor is true, but always emitted:
                // batch_idx_num_elements also drives the batched-fill decode on both paths.
                {"batch_idx_stick_size", batch_idx_stick_size_B},
                {"batch_idx_num_elements", batch_idx_num_elements},
                {"num_blocks_per_batch", num_blocks_of_work_per_batch},
                {"capacity_t", capacity_t},
                // Only meaningful when use_valid_seq_len is true.
                {"valid_seq_len_stick_size", valid_seq_len_stick_size_B},
            },
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };
    writer.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = FC_INPUT_DFB,
        .accessor_name = "in",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    // Self-loop: this kernel is the only toucher of the page-table buffer.
    writer.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = FC_PAGE_TABLE_DFB,
        .accessor_name = "page_table",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    writer.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = FC_PAGE_TABLE_DFB,
        .accessor_name = "page_table",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    writer.tensor_bindings.push_back(TensorBinding{
        .tensor_parameter_name = FC_CACHE_TENSOR,
        .accessor_name = "out",
    });
    writer.tensor_bindings.push_back(TensorBinding{
        .tensor_parameter_name = FC_PAGE_TABLE_TENSOR,
        .accessor_name = "page_table",
    });
    writer.runtime_arg_schema.runtime_arg_names = {"start_row_num", "num_rows", "noop"};
    if (use_batch_idx_tensor) {
        // Self-loop, same shape as the page-table buffer.
        writer.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = FC_BATCH_IDX_DFB,
            .accessor_name = "batch_idx",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        writer.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = FC_BATCH_IDX_DFB,
            .accessor_name = "batch_idx",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        writer.tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = FC_BATCH_IDX_TENSOR,
            .accessor_name = "batch_idx",
        });
    } else {
        // The legacy writer carried one arg slot that was either the batch_idx tensor's address or
        // this scalar. Metal 2.0 keeps the two on separate channels, so the scalar exists only on
        // the path that uses it — mirroring the single legacy slot.
        writer.runtime_arg_schema.runtime_arg_names.push_back("batch_idx_fallback");
    }
    if (use_valid_seq_len) {
        writer.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = FC_VALID_SEQ_LEN_DFB,
            .accessor_name = "valid_seq_len",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        writer.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = FC_VALID_SEQ_LEN_DFB,
            .accessor_name = "valid_seq_len",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        writer.tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = FC_VALID_SEQ_LEN_TENSOR,
            .accessor_name = "valid_seq_len",
        });
    }

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));

    spec.work_units.push_back(WorkUnitSpec{
        .name = "paged_fill_cache",
        .kernels = {FC_READER_KERNEL, FC_WRITER_KERNEL},
        .target_nodes = all_cores,
    });

    //-------------------------------------------------------------------------
    // Run args
    //-------------------------------------------------------------------------
    ProgramRunArgs run_args;

    KernelRunArgs reader_run_args{.kernel = FC_READER_KERNEL};
    KernelRunArgs writer_run_args{.kernel = FC_WRITER_KERNEL};

    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();

    // Core list shared with override_runtime_arguments (single source of truth for ordering).
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

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {
                {"start_tile_id", num_blocks_written * Wt},
                {"num_rows", num_blocks_per_core},
                {"noop", static_cast<uint32_t>(noop)},
            });

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {
                {"start_row_num", num_blocks_written},
                {"num_rows", num_blocks_per_core},
                {"noop", static_cast<uint32_t>(noop)},
            });
        if (!use_batch_idx_tensor) {
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"batch_idx_fallback", operation_attributes.batch_idx_fallback}});
        }

        num_blocks_written += num_blocks_per_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    // The op is in place (tensor_return_value aliases tensor_args.cache_tensor), so bind the same
    // tensors the spec above declared.
    run_args.tensor_args.emplace(FC_INPUT_TENSOR, input_tensor.mesh_tensor());
    run_args.tensor_args.emplace(FC_CACHE_TENSOR, cache_tensor.mesh_tensor());
    run_args.tensor_args.emplace(FC_PAGE_TABLE_TENSOR, page_table_tensor.mesh_tensor());
    if (use_batch_idx_tensor) {
        run_args.tensor_args.emplace(FC_BATCH_IDX_TENSOR, batch_idx_tensor.value().mesh_tensor());
    }
    if (use_valid_seq_len) {
        run_args.tensor_args.emplace(FC_VALID_SEQ_LEN_TENSOR, valid_seq_len_tensor.value().mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

ProgramRunArgs PagedFillCacheProgramFactory::override_runtime_arguments(
    const PagedFillCacheParams& operation_attributes,
    const PagedFillCacheInputs& tensor_args,
    Tensor& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    // Runs on EVERY cache hit, and on this concept the framework refreshes nothing on our behalf, so
    // this re-applies every tensor binding (the addresses the legacy patch wrote into arg slots)
    // plus the two values compute_program_hash excludes — batch_idx_fallback and noop — which would
    // otherwise freeze at their cache-miss value.
    //
    // Not re-applied: start_tile_id / start_row_num / num_rows. They come from the work split over
    // the input's padded shape and the device grid, both of which the program hash includes, so a
    // cache hit has them identical by construction — and UpdateProgramRunArgs is a partial update,
    // so anything omitted here keeps its cache-miss value.
    ProgramRunArgs run_args;

    const auto& cache_tensor = tensor_args.cache_tensor;
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& page_table_tensor = tensor_args.page_table;
    const auto& batch_idx_tensor = tensor_args.batch_idx_tensor_opt;
    const auto& valid_seq_len_tensor = tensor_args.valid_seq_len_tensor_opt;

    run_args.tensor_args.emplace(FC_INPUT_TENSOR, input_tensor.mesh_tensor());
    run_args.tensor_args.emplace(FC_CACHE_TENSOR, cache_tensor.mesh_tensor());
    run_args.tensor_args.emplace(FC_PAGE_TABLE_TENSOR, page_table_tensor.mesh_tensor());
    if (batch_idx_tensor.has_value()) {
        run_args.tensor_args.emplace(FC_BATCH_IDX_TENSOR, batch_idx_tensor.value().mesh_tensor());
    }
    if (valid_seq_len_tensor.has_value()) {
        run_args.tensor_args.emplace(FC_VALID_SEQ_LEN_TENSOR, valid_seq_len_tensor.value().mesh_tensor());
    }

    // noop is hash-excluded too, and on the mesh path depends on the dispatch coordinate. (This
    // factory is selected only when mesh_coords is nullopt, so the coordinate is inert here, but the
    // call is kept so the two factories stay behaviourally identical.)
    const auto noop_arg = static_cast<uint32_t>(paged_fill_cache_noop(operation_attributes, mesh_dispatch_coordinate));

    KernelRunArgs reader_run_args{.kernel = FC_READER_KERNEL};
    KernelRunArgs writer_run_args{.kernel = FC_WRITER_KERNEL};

    const auto cores = compute_paged_fill_cache_cores(operation_attributes, tensor_args);
    for (const auto& core : cores) {
        AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"noop", noop_arg}});
        AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"noop", noop_arg}});
        if (!batch_idx_tensor.has_value()) {
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"batch_idx_fallback", operation_attributes.batch_idx_fallback}});
        }
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    return run_args;
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
    return build_paged_fill_cache_descriptor(
        operation_attributes, tensor_args, paged_fill_cache_noop(operation_attributes, mesh_dispatch_coordinate));
}

void PagedFillCacheMeshWorkloadFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const PagedFillCacheParams& operation_attributes,
    const PagedFillCacheInputs& tensor_args,
    Tensor& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    patch_paged_fill_cache_runtime_args(program, operation_attributes, tensor_args, mesh_dispatch_coordinate);
}

}  // namespace ttnn::experimental::prim
