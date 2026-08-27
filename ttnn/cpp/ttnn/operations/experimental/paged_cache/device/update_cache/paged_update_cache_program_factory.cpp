// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "paged_update_cache_program_factory.hpp"

#include "paged_update_cache_device_operation.hpp"
#include "paged_update_cache_device_operation_types.hpp"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
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
const DFBSpecName UC_CACHE_DFB{"cache"};
const DFBSpecName UC_INPUT_DFB{"input"};
const DFBSpecName UC_INDEX_DFB{"index"};
const DFBSpecName UC_PAGE_TABLE_DFB{"page_table"};
const DFBSpecName UC_UNTILIZED_CACHE_DFB{"untilized_cache"};
const DFBSpecName UC_UNTILIZED_CACHE2_DFB{"untilized_cache2"};
const DFBSpecName UC_UNTILIZED_INPUT_DFB{"untilized_input"};
const DFBSpecName UC_OUTPUT_DFB{"output"};

const TensorParamName UC_CACHE_TENSOR{"cache"};
const TensorParamName UC_INPUT_TENSOR{"input"};
const TensorParamName UC_INDEX_TENSOR{"index"};
const TensorParamName UC_PAGE_TABLE_TENSOR{"page_table"};

const SemaphoreSpecName UC_SEQUENTIAL_MODE_SEM{"in0_sequential_mode"};

const KernelSpecName UC_READER_KERNEL{"reader"};
const KernelSpecName UC_WRITER_KERNEL{"writer"};
const KernelSpecName UC_COMPUTE_KERNEL{"compute"};

constexpr auto UC_READER_SOURCE =
    "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
    "reader_update_cache_interleaved_start_id_metal2.cpp";
constexpr auto UC_WRITER_SOURCE =
    "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
    "writer_update_cache_interleaved_start_id_metal2.cpp";
constexpr auto UC_COMPUTE_SOURCE =
    "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/update_cache_metal2.cpp";

bool enable_fp32_dest(const tt_metal::IDevice* device, const ttnn::DeviceComputeKernelConfig& compute_kernel_config) {
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    return fp32_dest_acc_en;
}

// Worker cores in the exact order the factory emplaces per-core runtime args (core i handles
// user i, i.e. update_idxs[i]). Shared by create_program_artifacts (cache miss) and
// override_runtime_arguments (cache hit) so the two cannot drift.
std::vector<CoreCoord> update_cache_cores(const PagedUpdateCacheInputs& tensor_args) {
    const ShardSpec& shard_spec = tensor_args.input_tensor.shard_spec().value();
    return corerange_to_cores(
        shard_spec.grid, shard_spec.grid.num_cores(), shard_spec.orientation == ShardOrientation::ROW_MAJOR);
}

// Per-worker-core cache-write offsets derived from `update_idxs`. These values are excluded from the
// program hash (PagedUpdateCacheDeviceOperation::compute_program_hash) yet baked into runtime args, so
// they must be re-patched on every cache hit via override_runtime_arguments. This helper is the single
// source of truth for the formulas — both the cache-miss build and override_runtime_arguments
// (cache hit) call it, so the two paths cannot drift. Entries are in `cores` order. Returns empty when
// an index tensor is used: in that mode the offsets are 0 here and the real positions are read on-device
// from the (re-patched) index tensor.
struct UpdateCachePerCoreOffsets {
    uint32_t cache_start_id = 0;
    uint32_t tile_update_offset_B = 0;
};

std::vector<UpdateCachePerCoreOffsets> compute_update_cache_offsets(
    const PagedUpdateCacheParams& operation_attributes,
    const PagedUpdateCacheInputs& tensor_args,
    const std::vector<CoreCoord>& cores) {
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
    // share_cache => batch offset is 0 (one shared cache buffer); mirror the program build exactly.
    const uint32_t cache_batch_num_tiles =
        operation_attributes.share_cache ? 0 : cache_total_num_tiles / cache_tensor.padded_shape()[0];

    std::vector<UpdateCachePerCoreOffsets> offsets;
    offsets.reserve(cores.size());
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const uint32_t update_idx = operation_attributes.update_idxs.at(i);
        const uint32_t cache_batch_tile_offset = i * cache_batch_num_tiles;
        const uint32_t cache_start_id = cache_batch_tile_offset + ((update_idx / TILE_HEIGHT) * Wt);
        const uint32_t tile_update_offset_B = update_idx % TILE_HEIGHT * Wbytes;
        offsets.push_back({cache_start_id, tile_update_offset_B});
    }
    return offsets;
}

// ---------------------------------------------------------------------------------------------
// Legacy ProgramDescriptor body.
//
// PagedUpdateCacheMeshWorkloadFactory still builds a ProgramDescriptor: it returns a different
// program per mesh coordinate (an empty descriptor outside operation_attributes.mesh_coords), which
// the Metal 2.0 spec factory concepts cannot express — create_program_artifacts takes no coordinate
// and its spec is stamped identically on every coordinate.  So this body stays, unchanged apart from
// having moved out of PagedUpdateCacheProgramFactory, and keeps binding the legacy kernel sources.
// ---------------------------------------------------------------------------------------------
ProgramDescriptor build_paged_update_cache_descriptor(
    const PagedUpdateCacheParams& operation_attributes, const PagedUpdateCacheInputs& tensor_args) {
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
    // desc.cbs[1]: the only globally-allocated CB (aliases the input shard) — re-pointed on cache hits by
    // override_runtime_arguments.
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

    const auto cores = update_cache_cores(tensor_args);
    // cache_start_id / tile_update_offset_B are derived from update_idxs (excluded from the program
    // hash) — computed via the shared helper so override_runtime_arguments re-patches identical values on
    // cache hits. Empty in index-tensor mode (offsets read on-device from the re-patched index tensor).
    const auto offsets = compute_update_cache_offsets(operation_attributes, tensor_args, cores);
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

// Legacy in-place cache-hit patch, used by PagedUpdateCacheMeshWorkloadFactory (see the note above
// build_paged_update_cache_descriptor for why that factory stays on the descriptor concept).
void patch_paged_update_cache_runtime_args(
    tt::tt_metal::Program& program,
    const PagedUpdateCacheParams& operation_attributes,
    const PagedUpdateCacheInputs& tensor_args,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    // Patch the cached program in place — no descriptor rebuild. This runs on EVERY cache hit, so it must
    // re-derive only per-dispatch state: buffer addresses (this hook supersedes resolve_bindings, so all
    // addresses are ours) and the update_idxs-derived offsets the program hash excludes. Everything else
    // is a function of hashed inputs (shapes/dtypes/memory configs/share_cache/overrides) and is identical
    // by construction on a hit.
    //
    // The only mesh-specific behaviour is the empty descriptor for coords excluded from the dispatch —
    // those programs have no kernels to patch.
    if (operation_attributes.mesh_coords.has_value() && mesh_dispatch_coordinate.has_value() &&
        !operation_attributes.mesh_coords.value().contains(mesh_dispatch_coordinate.value())) {
        return;
    }

    // Kernel push order in the descriptor build: reader(0), writer(1), compute(2). Compute takes no
    // runtime args.
    constexpr uint32_t kReaderKernelIdx = 0;
    constexpr uint32_t kWriterKernelIdx = 1;
    // Reader rt args: [0]=cache, [1]=cache_start_id, [2]=update_idxs tensor, [3]=core index,
    //                 [4]=page_table, [5]=wait_to_start.
    // Writer rt args: [0]=cache, [1]=cache_start_id, [2]=tile_update_offset_B, [3]=core index,
    //                 [4]=send_signal, [5..6]=send core x/y.
    // Position of the input-shard CB in desc.cbs (src0, src1, interm0+1, interm2, output, [index], [pagetable]).
    constexpr uint32_t kInputCbPos = 1;

    // In-place op: tensor_return_value aliases cache_tensor, which is the buffer both kernels write.
    const uint32_t cache_addr = tensor_args.cache_tensor.buffer()->address();
    // Absent optional tensors were emplaced as literal 0 (see the descriptor build), so 0 is the correct
    // patch.
    const uint32_t update_idxs_addr =
        tensor_args.update_idxs_tensor.has_value() ? tensor_args.update_idxs_tensor.value().buffer()->address() : 0u;
    const uint32_t page_table_addr =
        tensor_args.page_table.has_value() ? tensor_args.page_table.value().buffer()->address() : 0u;

    const auto cores = update_cache_cores(tensor_args);
    // Empty in index-tensor mode: the kernels read positions on-device from the index tensor, so the two
    // offset slots stay 0 as emplaced.
    const auto offsets = compute_update_cache_offsets(operation_attributes, tensor_args, cores);

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);

        auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx, core);
        reader_args[0] = cache_addr;
        reader_args[2] = update_idxs_addr;
        reader_args[4] = page_table_addr;

        auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx, core);
        writer_args[0] = cache_addr;

        if (!offsets.empty()) {
            reader_args[1] = offsets.at(i).cache_start_id;
            writer_args[1] = offsets.at(i).cache_start_id;
            writer_args[2] = offsets.at(i).tile_update_offset_B;
        }
    }

    tt::tt_metal::UpdateDynamicCircularBufferAddress(
        program, program.circular_buffers().at(kInputCbPos)->id(), *tensor_args.input_tensor.buffer());
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Metal 2.0 program build.
// ---------------------------------------------------------------------------------------------
ttnn::device_operation::ProgramArtifacts PagedUpdateCacheProgramFactory::create_program_artifacts(
    const PagedUpdateCacheParams& operation_attributes,
    const PagedUpdateCacheInputs& tensor_args,
    Tensor& /*tensor_return_value*/) {
    const auto& cache_tensor = tensor_args.cache_tensor;
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& update_idxs_tensor = tensor_args.update_idxs_tensor;
    const auto& page_table = tensor_args.page_table;

    tt_metal::IDevice* device = input_tensor.device();

    tt::DataFormat cache_data_format = tt_metal::datatype_to_dataformat_converter(cache_tensor.dtype());
    uint32_t cache_single_tile_size = tt::tile_size(cache_data_format);

    tt::DataFormat input_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_data_format);

    bool fp32_dest_acc_en = enable_fp32_dest(device, operation_attributes.compute_kernel_config);

    tt::DataFormat interm_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t interm_single_tile_size = tt::tile_size(interm_data_format);

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

    log_debug(tt::LogOp, "cache_data_format: {}", cache_data_format);
    log_debug(tt::LogOp, "input_data_format: {}", input_data_format);
    log_debug(tt::LogOp, "interm_data_format: {}", interm_data_format);
    log_debug(tt::LogOp, "Wbytes: {}", Wbytes);
    log_debug(tt::LogOp, "Wt: {}", Wt);
    log_debug(tt::LogOp, "St: {}", St);

    const std::optional<ShardSpec>& shard_spec = input_tensor.shard_spec();
    CoreRangeSet all_cores = shard_spec.value().grid;
    uint32_t num_cores = all_cores.num_cores();
    uint32_t num_input_tiles = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;

    uint32_t num_cache_tiles = 2 * Wt;   // double buffered
    uint32_t num_interm_tiles = 2 * Wt;  // double buffered
    uint32_t num_output_tiles = B * Wt;

    // cache_position_modulo: 0 = disabled (legacy), nonzero = wrap update_idx mod this
    // value before page_table lookup. Required when the caller's page_table is sized for
    // a bounded sliding-window cache (vLLM SlidingWindowSpec).
    const uint32_t cache_position_modulo = operation_attributes.cache_position_modulo.value_or(0u);

    ProgramSpec spec;
    spec.name = "paged_update_cache";

    //-------------------------------------------------------------------------
    // Dataflow buffers
    //-------------------------------------------------------------------------
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = UC_CACHE_DFB,
        .entry_size = cache_single_tile_size,
        .num_entries = num_cache_tiles,
        .data_format_metadata = cache_data_format,
    });
    // Borrowed memory: this DFB is a view over the resident input shard rather than its own L1
    // allocation, so the reader publishes it without transferring anything. The backing address is
    // refreshed from the input tensor's TensorArgument on every dispatch.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = UC_INPUT_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_data_format,
        .borrowed_from = UC_INPUT_TENSOR,
    });
    // untilized_cache and untilized_cache2 are two logical buffers over ONE L1 region, and the
    // aliasing is the algorithm: compute publishes an untilized cache block through the first, the
    // writer NoC-writes the new row into that same memory in place, then republishes it through the
    // second for compute to re-tilize. Splitting them into independent DFBs validates and silently
    // produces wrong numerics.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = UC_UNTILIZED_CACHE_DFB,
        .entry_size = interm_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = interm_data_format,
        .advanced_options = {.alias_with = {UC_UNTILIZED_CACHE2_DFB}},
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = UC_UNTILIZED_CACHE2_DFB,
        .entry_size = interm_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = interm_data_format,
        .advanced_options = {.alias_with = {UC_UNTILIZED_CACHE_DFB}},
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = UC_UNTILIZED_INPUT_DFB,
        .entry_size = interm_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = interm_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = UC_OUTPUT_DFB,
        .entry_size = cache_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = cache_data_format,
    });

    if (use_index_tensor) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = UC_INDEX_DFB,
            .entry_size = index_tensor_tile_size,
            .num_entries = 1,
            .data_format_metadata = index_data_format,
        });
    }

    if (is_paged_cache) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = UC_PAGE_TABLE_DFB,
            .entry_size = page_table_stick_size,
            .num_entries = 1,
            .data_format_metadata = page_table_data_format,
        });
    }

    //-------------------------------------------------------------------------
    // Semaphores
    //-------------------------------------------------------------------------
    // used for share cache for signaling when the cache is ready to be read
    spec.semaphores.push_back(SemaphoreSpec{
        .unique_id = UC_SEQUENTIAL_MODE_SEM,
        .target_nodes = all_cores,
    });

    //-------------------------------------------------------------------------
    // Tensor parameters
    //-------------------------------------------------------------------------
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = UC_CACHE_TENSOR,
        .spec = cache_tensor.tensor_spec(),
    });
    // Declared for the borrowed-memory DFB above; no kernel binds it as a TensorAccessor.
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = UC_INPUT_TENSOR,
        .spec = input_tensor.tensor_spec(),
    });
    if (use_index_tensor) {
        spec.tensor_parameters.push_back(TensorParameter{
            .unique_id = UC_INDEX_TENSOR,
            .spec = update_idxs_tensor.value().tensor_spec(),
        });
    }
    if (is_paged_cache) {
        spec.tensor_parameters.push_back(TensorParameter{
            .unique_id = UC_PAGE_TABLE_TENSOR,
            .spec = page_table.value().tensor_spec(),
        });
    }

    //-------------------------------------------------------------------------
    // Kernels
    //-------------------------------------------------------------------------
    // The conditional DFB / tensor bindings are gated kernel-side by these defines rather than by a
    // compile-time arg: `if constexpr` still name-looks-up the discarded branch, so a `dfb::index` or
    // `tensor::page_table` the host did not bind would fail to compile.
    KernelSpec::CompilerOptions::Defines conditional_defines;
    if (use_index_tensor) {
        conditional_defines.emplace("USE_INDEX_TENSOR", "1");
    }
    if (is_paged_cache) {
        conditional_defines.emplace("IS_PAGED_CACHE", "1");
    }

    KernelSpec reader{
        .unique_id = UC_READER_KERNEL,
        .source = UC_READER_SOURCE,
        .compiler_options = {.defines = conditional_defines},
        .semaphore_bindings =
            {
                SemaphoreBinding{
                    .semaphore_spec_name = UC_SEQUENTIAL_MODE_SEM,
                    .accessor_name = "receiver",
                },
            },
        .compile_time_args =
            {
                {"cache_batch_num_tiles", cache_batch_num_tiles},
                {"Wt", Wt},
                {"log_base_2_of_page_size", log2_page_size},
                {"index_stick_size_B", index_stick_size},
                {"num_heads", num_heads},
                {"block_size", block_size},
                {"block_size_t", block_size_t},
                {"max_blocks_per_seq", max_blocks_per_seq},
                {"log2_page_table_stick_size", log2_page_table_stick_size},
                {"page_table_stick_size", page_table_stick_size},
                {"St", St},
                {"cache_position_modulo", cache_position_modulo},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"cache_start_id", "my_batch_idx", "wait_to_start"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };
    reader.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = UC_CACHE_DFB,
        .accessor_name = "cache",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    reader.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = UC_INPUT_DFB,
        .accessor_name = "input",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    reader.tensor_bindings.push_back(TensorBinding{
        .tensor_parameter_name = UC_CACHE_TENSOR,
        .accessor_name = "cache",
    });
    if (use_index_tensor) {
        reader.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UC_INDEX_DFB,
            .accessor_name = "index",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader.tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = UC_INDEX_TENSOR,
            .accessor_name = "index",
        });
    }
    if (is_paged_cache) {
        reader.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UC_PAGE_TABLE_DFB,
            .accessor_name = "page_table",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader.tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = UC_PAGE_TABLE_TENSOR,
            .accessor_name = "page_table",
        });
    }

    KernelSpec writer{
        .unique_id = UC_WRITER_KERNEL,
        .source = UC_WRITER_SOURCE,
        .compiler_options = {.defines = conditional_defines},
        .semaphore_bindings =
            {
                SemaphoreBinding{
                    .semaphore_spec_name = UC_SEQUENTIAL_MODE_SEM,
                    .accessor_name = "receiver",
                },
            },
        .compile_time_args =
            {
                {"cache_batch_num_tiles", cache_batch_num_tiles},
                {"Wt", Wt},
                {"Wbytes", Wbytes},
                {"num_heads", num_heads},
                {"block_size", block_size},
                {"block_size_t", block_size_t},
                {"max_blocks_per_seq", max_blocks_per_seq},
                {"St", St},
                {"cache_position_modulo", cache_position_modulo},
            },
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"cache_start_id",
                  "cache_tile_offset_B",
                  "my_batch_idx",
                  "send_signal",
                  "send_core_x",
                  "send_core_y"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };
    // The writer's `cache` accessor is the OUTPUT DFB: it holds the re-tilized cache block this
    // kernel writes back to the cache tensor. (The cache tiles the reader pulled in reach compute
    // through UC_CACHE_DFB, which the writer never touches.)
    writer.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = UC_OUTPUT_DFB,
        .accessor_name = "cache",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    writer.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = UC_UNTILIZED_CACHE_DFB,
        .accessor_name = "untilized_cache",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    writer.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = UC_UNTILIZED_CACHE2_DFB,
        .accessor_name = "untilized_cache2",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    writer.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = UC_UNTILIZED_INPUT_DFB,
        .accessor_name = "untilized_input",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    writer.tensor_bindings.push_back(TensorBinding{
        .tensor_parameter_name = UC_CACHE_TENSOR,
        .accessor_name = "cache",
    });
    if (use_index_tensor) {
        writer.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UC_INDEX_DFB,
            .accessor_name = "index",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }
    if (is_paged_cache) {
        writer.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UC_PAGE_TABLE_DFB,
            .accessor_name = "page_table",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    // Legacy set only fp32_dest_acc_en on its ComputeConfigDescriptor and left every other field at
    // the descriptor's defaults, which coincide one-for-one with ComputeGen1Config's defaults
    // (HiFi4 / Precise SFPU / Approximate BFP pack / double-buffered dest). So only
    // enable_32_bit_dest carries across; routing the resolved TTNN config through
    // to_compute_hardware_config would substitute that helper's high-performance defaults for the
    // knobs this op never applied.
    ComputeGen1Config compute_hw{.enable_32_bit_dest = fp32_dest_acc_en};
    if (fp32_dest_acc_en) {
        // Metal 2.0 requires an explicit unpack mode for every Float32 DFB a compute kernel consumes
        // while enable_32_bit_dest is set. Legacy left unpack_to_dest_mode empty (all Default), which
        // is UnpackToSrc.
        if (cache_data_format == tt::DataFormat::Float32) {
            compute_hw.unpack_modes.emplace(UC_CACHE_DFB, UnpackMode::UnpackToSrc);
        }
        if (input_data_format == tt::DataFormat::Float32) {
            compute_hw.unpack_modes.emplace(UC_INPUT_DFB, UnpackMode::UnpackToSrc);
        }
        if (interm_data_format == tt::DataFormat::Float32) {
            compute_hw.unpack_modes.emplace(UC_UNTILIZED_CACHE2_DFB, UnpackMode::UnpackToSrc);
        }
    }

    KernelSpec compute{
        .unique_id = UC_COMPUTE_KERNEL,
        .source = UC_COMPUTE_SOURCE,
        // Legacy left KernelDescriptor::opt_level unset, which resolves to O3 on a
        // ComputeConfigDescriptor; Metal 2.0's CompilerOptions defaults to O2, so set it explicitly.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = UC_CACHE_DFB,
                    .accessor_name = "cache",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = UC_INPUT_DFB,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = UC_UNTILIZED_CACHE_DFB,
                    .accessor_name = "untilized_cache",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = UC_UNTILIZED_CACHE2_DFB,
                    .accessor_name = "untilized_cache2",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = UC_UNTILIZED_INPUT_DFB,
                    .accessor_name = "untilized_in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = UC_OUTPUT_DFB,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .compile_time_args =
            {
                {"Wt", Wt},
                {"num_heads", num_heads},
            },
        .hw_config = compute_hw,
    };

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));
    spec.kernels.push_back(std::move(compute));

    spec.work_units.push_back(WorkUnitSpec{
        .name = "paged_update_cache",
        .kernels = {UC_READER_KERNEL, UC_WRITER_KERNEL, UC_COMPUTE_KERNEL},
        .target_nodes = all_cores,
    });

    //-------------------------------------------------------------------------
    // Run args
    //-------------------------------------------------------------------------
    ProgramRunArgs run_args;

    KernelRunArgs reader_run_args{.kernel = UC_READER_KERNEL};
    KernelRunArgs writer_run_args{.kernel = UC_WRITER_KERNEL};

    const auto cores = update_cache_cores(tensor_args);
    // cache_start_id / tile_update_offset_B are derived from update_idxs (excluded from the program
    // hash) — computed via the shared helper so override_runtime_arguments re-patches identical values on
    // cache hits. Empty in index-tensor mode (offsets read on-device from the re-patched index tensor).
    const auto offsets = compute_update_cache_offsets(operation_attributes, tensor_args, cores);
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

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {
                {"cache_start_id", cache_start_id},
                {"my_batch_idx", i},
                {"wait_to_start", static_cast<uint32_t>(wait_to_start)},
            });

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {
                {"cache_start_id", cache_start_id},
                {"cache_tile_offset_B", tile_update_offset_B},
                {"my_batch_idx", i},
                {"send_signal", static_cast<uint32_t>(send_signal)},
                {"send_core_x", send_core_x},
                {"send_core_y", send_core_y},
            });
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    // In-place op: tensor_return_value aliases cache_tensor, which is the buffer both kernels write.
    run_args.tensor_args.emplace(UC_CACHE_TENSOR, cache_tensor.mesh_tensor());
    run_args.tensor_args.emplace(UC_INPUT_TENSOR, input_tensor.mesh_tensor());
    if (use_index_tensor) {
        run_args.tensor_args.emplace(UC_INDEX_TENSOR, update_idxs_tensor.value().mesh_tensor());
    }
    if (is_paged_cache) {
        run_args.tensor_args.emplace(UC_PAGE_TABLE_TENSOR, page_table.value().mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

ProgramRunArgs PagedUpdateCacheProgramFactory::override_runtime_arguments(
    const PagedUpdateCacheParams& operation_attributes,
    const PagedUpdateCacheInputs& tensor_args,
    Tensor& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    // Runs on EVERY cache hit, and on this concept the framework refreshes nothing on our behalf, so
    // this must re-derive all per-dispatch state: every tensor binding — including the input
    // tensor, whose address backs the borrowed-memory input DFB — and the update_idxs-derived
    // offsets the program hash excludes. Everything else is a function of hashed inputs (shapes/dtypes/memory
    // configs/share_cache/overrides) and is identical by construction on a hit — UpdateProgramRunArgs
    // is a partial update, so anything omitted here keeps its cache-miss value.
    ProgramRunArgs run_args;

    // Mirrors the legacy body: a coordinate excluded from the dispatch has nothing to patch. This
    // factory is selected only when mesh_coords is nullopt (see select_program_factory), so the guard
    // is inert here, but it is kept so the two factories stay behaviourally identical.
    if (operation_attributes.mesh_coords.has_value() && mesh_dispatch_coordinate.has_value() &&
        !operation_attributes.mesh_coords.value().contains(mesh_dispatch_coordinate.value())) {
        return run_args;
    }

    const auto& cache_tensor = tensor_args.cache_tensor;
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& update_idxs_tensor = tensor_args.update_idxs_tensor;
    const auto& page_table = tensor_args.page_table;

    run_args.tensor_args.emplace(UC_CACHE_TENSOR, cache_tensor.mesh_tensor());
    run_args.tensor_args.emplace(UC_INPUT_TENSOR, input_tensor.mesh_tensor());
    if (update_idxs_tensor.has_value()) {
        run_args.tensor_args.emplace(UC_INDEX_TENSOR, update_idxs_tensor.value().mesh_tensor());
    }
    if (page_table.has_value()) {
        run_args.tensor_args.emplace(UC_PAGE_TABLE_TENSOR, page_table.value().mesh_tensor());
    }

    const auto cores = update_cache_cores(tensor_args);
    // Empty in index-tensor mode: the kernels read positions on-device from the index tensor, so the two
    // offset slots keep the 0 they were given on the cache miss.
    const auto offsets = compute_update_cache_offsets(operation_attributes, tensor_args, cores);
    if (offsets.empty()) {
        return run_args;
    }

    KernelRunArgs reader_run_args{.kernel = UC_READER_KERNEL};
    KernelRunArgs writer_run_args{.kernel = UC_WRITER_KERNEL};
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"cache_start_id", offsets.at(i).cache_start_id}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {
                {"cache_start_id", offsets.at(i).cache_start_id},
                {"cache_tile_offset_B", offsets.at(i).tile_update_offset_B},
            });
    }
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    return run_args;
}

ProgramDescriptor PagedUpdateCacheMeshWorkloadFactory::create_descriptor(
    const PagedUpdateCacheParams& operation_attributes,
    const PagedUpdateCacheInputs& tensor_args,
    Tensor& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    if (operation_attributes.mesh_coords.has_value() && mesh_dispatch_coordinate.has_value()) {
        const auto& mesh_coords_set = operation_attributes.mesh_coords.value();
        if (!mesh_coords_set.contains(mesh_dispatch_coordinate.value())) {
            return ProgramDescriptor{};
        }
    }
    return build_paged_update_cache_descriptor(operation_attributes, tensor_args);
}

void PagedUpdateCacheMeshWorkloadFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const PagedUpdateCacheParams& operation_attributes,
    const PagedUpdateCacheInputs& tensor_args,
    Tensor& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    patch_paged_update_cache_runtime_args(program, operation_attributes, tensor_args, mesh_dispatch_coordinate);
}

}  // namespace ttnn::experimental::prim
