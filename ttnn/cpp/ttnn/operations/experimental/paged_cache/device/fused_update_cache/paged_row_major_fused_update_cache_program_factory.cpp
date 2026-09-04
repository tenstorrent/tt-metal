// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "paged_row_major_fused_update_cache_program_factory.hpp"

#include "paged_fused_update_cache_device_operation_types.hpp"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

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

// Metal 2.0 spec resource names for this factory.  Prefixed to keep the namespace free of collisions
// when the op's factory .cpp files are unity-built into one translation unit.
const DFBSpecName RMF_CACHE_DFB{"cache"};
const DFBSpecName RMF_SRC1_DFB{"src1"};
const DFBSpecName RMF_SRC2_DFB{"src2"};
const DFBSpecName RMF_INDEX_DFB{"index"};
const DFBSpecName RMF_PAGE_TABLE_DFB{"page_table"};
const DFBSpecName RMF_UNTILIZED_CACHE_DFB{"untilized_cache"};
const DFBSpecName RMF_UNTILIZED_CACHE2_DFB{"untilized_cache2"};
const DFBSpecName RMF_OUTPUT_DFB{"output"};

const TensorParamName RMF_CACHE1_TENSOR{"cache1"};
const TensorParamName RMF_CACHE2_TENSOR{"cache2"};
const TensorParamName RMF_INPUT1_TENSOR{"input1"};
const TensorParamName RMF_INPUT2_TENSOR{"input2"};
const TensorParamName RMF_INDEX_TENSOR{"index"};
const TensorParamName RMF_PAGE_TABLE_TENSOR{"page_table"};

const SemaphoreSpecName RMF_SEQUENTIAL_MODE_SEM{"in0_sequential_mode"};

const KernelSpecName RMF_READER_KERNEL{"reader"};
const KernelSpecName RMF_WRITER_KERNEL{"writer"};
const KernelSpecName RMF_COMPUTE_KERNEL{"compute"};

constexpr auto RMF_READER_SOURCE =
    "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
    "reader_paged_row_major_fused_update_cache_interleaved_start_id_metal2.cpp";
constexpr auto RMF_WRITER_SOURCE =
    "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
    "writer_paged_row_major_fused_update_cache_interleaved_start_id_metal2.cpp";
constexpr auto RMF_COMPUTE_SOURCE =
    "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/"
    "paged_row_major_fused_update_cache_metal2.cpp";
}  // namespace CMAKE_UNIQUE_NAMESPACE_ROW_MAJOR

std::vector<PagedRowMajorFusedUpdateCacheProgramFactory::PerIndexOffsets>
PagedRowMajorFusedUpdateCacheProgramFactory::compute_row_major_fused_offsets(
    const PagedFusedUpdateCacheParams& operation_attributes, const PagedFusedUpdateCacheInputs& tensor_args) {
    // cache_start_id / tile_update_offset_B are derived from update_idxs, which is excluded from the
    // program hash (see PagedFusedUpdateCacheDeviceOperation::compute_program_hash) yet baked into runtime
    // args, so they must be re-applied on every cache hit. This helper is the single source of truth for
    // the formulas — both create_descriptor (cache miss) and override_runtime_arguments (cache hit) call it,
    // so the two paths cannot drift. Returns empty when an index tensor is used: in that mode the offsets
    // are 0 here and the real positions are read on-device from the (re-patched) index tensor.
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

namespace CMAKE_UNIQUE_NAMESPACE_ROW_MAJOR {

// Coordinates this dispatch actually runs on.
//
// The ported-from factory expressed its mesh filter by returning an EMPTY ProgramDescriptor for an
// excluded coordinate, which the descriptor adapter then skipped entirely. On this concept the
// equivalent is simply not emitting a program for that coordinate: the adapter requires every range
// returned to sit inside tensor_coords, but does NOT require the ranges to cover it.
//
// One program per coordinate is also what the ported-from path built -- its create_descriptor took a
// mesh_dispatch_coordinate, and for that shape the descriptor adapter iterates
// tensor_coords.coords() and adds one program per coordinate rather than one per range.
std::vector<ttnn::MeshCoordinate> fused_dispatch_coords(
    const PagedFusedUpdateCacheParams& operation_attributes, const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    std::vector<ttnn::MeshCoordinate> coords;
    for (const auto& coord : tensor_coords.coords()) {
        if (operation_attributes.mesh_coords.has_value() && !operation_attributes.mesh_coords->contains(coord)) {
            continue;
        }
        coords.push_back(coord);
    }
    return coords;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE_ROW_MAJOR

// ---------------------------------------------------------------------------------------------
// Metal 2.0 program build.
// ---------------------------------------------------------------------------------------------
ttnn::device_operation::ProgramArtifacts PagedRowMajorFusedUpdateCacheProgramFactory::create_program_artifacts(
    const PagedFusedUpdateCacheParams& operation_attributes,
    const PagedFusedUpdateCacheInputs& tensor_args,
    PagedFusedUpdateCacheResult& /*tensor_return_value*/) {
    using namespace CMAKE_UNIQUE_NAMESPACE_ROW_MAJOR;

    const auto& cache_tensor1 = tensor_args.cache_tensor1;
    const auto& input_tensor1 = tensor_args.input_tensor1;
    const auto& cache_tensor2 = tensor_args.cache_tensor2;
    const auto& input_tensor2 = tensor_args.input_tensor2;
    const auto& update_idxs_tensor = tensor_args.update_idxs_tensor;
    const auto& page_table = tensor_args.page_table;

    tt_metal::IDevice* device = input_tensor1.device();

    tt::DataFormat cache_data_format = tt_metal::datatype_to_dataformat_converter(cache_tensor1.dtype());
    uint32_t cache_single_tile_size = tt::tile_size(cache_data_format);

    tt::DataFormat input_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor1.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_data_format);

    bool fp32_dest_acc_en = enable_fp32_dest_acc(device, operation_attributes.compute_kernel_config);

    tt::DataFormat interm_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t interm_single_tile_size = tt::tile_size(interm_data_format);

    const uint32_t B = input_tensor1.padded_shape()[1];
    const uint32_t num_heads = cache_tensor1.padded_shape()[1];

    // Index tensor-specific parameters
    bool use_index_tensor = update_idxs_tensor.has_value();
    uint32_t log2_page_size = 0;
    uint32_t index_stick_size = 0;
    tt::DataFormat index_data_format = tt::DataFormat::Int32;
    bool index_is_dram = true;
    // The index DFB borrows the index tensor's L1 memory only when that tensor is L1-sharded. On the
    // DRAM-interleaved path it is an ordinary L1 allocation that the reader fills over the NoC, and
    // the reader's read is compiled out on the sharded path. Both paths must survive, and this flag
    // is what carries the distinction into the spec.
    bool index_is_sharded = false;
    if (use_index_tensor) {
        index_is_sharded = update_idxs_tensor.value().is_sharded();
        index_data_format = tt_metal::datatype_to_dataformat_converter(update_idxs_tensor.value().dtype());
        index_is_dram = update_idxs_tensor.value().buffer()->buffer_type() == tt_metal::BufferType::DRAM;
        index_stick_size = update_idxs_tensor.value().buffer()->aligned_page_size();
    }

    // Pagetable-specific parameters
    bool is_paged_cache = page_table.has_value();
    uint32_t block_size = 0;
    uint32_t block_size_t = 0;
    uint32_t max_blocks_per_seq = 0;
    uint32_t page_table_stick_size = 0;
    uint32_t log2_page_table_stick_size = 0;
    uint32_t num_pages_page_table = 1;
    tt::DataFormat page_table_data_format = tt::DataFormat::Int32;
    bool page_table_is_dram = true;
    // Same conditional-borrow shape as the index DFB above.
    bool page_table_is_sharded = false;
    if (is_paged_cache) {
        const auto& page_table_tensor = page_table.value();
        page_table_is_sharded = page_table_tensor.is_sharded();
        num_pages_page_table = page_table_is_sharded ? B : 1;
        block_size = cache_tensor1.padded_shape()[2];
        block_size_t = block_size / TILE_HEIGHT;
        max_blocks_per_seq = page_table_tensor.padded_shape()[1];
        page_table_stick_size = page_table_tensor.buffer()->aligned_page_size();
        page_table_data_format = tt_metal::datatype_to_dataformat_converter(page_table_tensor.dtype());
        page_table_is_dram = page_table_tensor.buffer()->buffer_type() == tt_metal::BufferType::DRAM;
    }

    uint32_t Wt = cache_tensor1.padded_shape()[-1] / TILE_WIDTH;
    uint32_t St = cache_tensor1.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t Wbytes = fp32_dest_acc_en ? cache_tensor1.padded_shape()[-1] * sizeof(float)
                                       : cache_tensor1.padded_shape()[-1] * 2;  // 2 bytes for bfloat16
    uint32_t cache_total_num_tiles = cache_tensor1.physical_volume() / TILE_HW;
    uint32_t cache_batch_num_tiles =
        operation_attributes.share_cache
            ? 0
            : cache_total_num_tiles /
                  cache_tensor1.padded_shape()[0];  // if share cache, we can set cache batch num tiles to 0
                                                    // so batch offset would be 0 in future calculations

    log_debug(tt::LogOp, "cache_data_format: {}", cache_data_format);
    log_debug(tt::LogOp, "input_data_format: {}", input_data_format);
    log_debug(tt::LogOp, "interm_data_format: {}", interm_data_format);
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

    uint32_t num_cache_tiles = 2 * Wt;   // double buffered
    uint32_t num_interm_tiles = 2 * Wt;  // double buffered
    uint32_t num_output_tiles = B * Wt;

    ProgramSpec spec;
    spec.name = "paged_row_major_fused_update_cache";

    //-------------------------------------------------------------------------
    // Dataflow buffers
    //-------------------------------------------------------------------------
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = RMF_CACHE_DFB,
        .entry_size = cache_single_tile_size,
        .num_entries = num_cache_tiles,
        .data_format_metadata = cache_data_format,
    });
    // Borrowed memory: src1 and src2 are views over the two resident input shards rather than their
    // own L1 allocations, so the reader publishes them without transferring anything. Each backing
    // address is refreshed from its input tensor's TensorArgument on every dispatch.
    //
    // Legacy configured src1 only over input1's shard cores and src2 only over input2's (the two are
    // validated disjoint). Metal 2.0 derives a DFB's placement from its bindings, and every kernel
    // here spans the bounding box of both, so each of these is now configured over the whole grid.
    // That costs no L1 -- a borrowed DFB takes its address from the bound tensor and never touches
    // the allocator -- and on the half of the grid legacy left unconfigured the buffer is never
    // touched, because the kernels' is_input1 arg steers each core to its own input. It is Quasar
    // debt rather than a Gen1 behaviour change: on Gen2 a DFB's hardware footprint varies with its
    // endpoint configuration, and a borrowed DFB over a tensor with no shard on the node is
    // meaningless there.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = RMF_SRC1_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_data_format,
        .borrowed_from = RMF_INPUT1_TENSOR,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = RMF_SRC2_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_data_format,
        .borrowed_from = RMF_INPUT2_TENSOR,
    });
    // untilized_cache and untilized_cache2 are two logical buffers over ONE L1 region, and the
    // aliasing is the algorithm: compute publishes an untilized cache block through the first, the
    // writer NoC-writes the new row into that same memory in place, then republishes it through the
    // second for compute to re-tilize. Splitting them into independent DFBs validates and silently
    // produces wrong numerics.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = RMF_UNTILIZED_CACHE_DFB,
        .entry_size = interm_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = interm_data_format,
        .advanced_options = {.alias_with = {RMF_UNTILIZED_CACHE2_DFB}},
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = RMF_UNTILIZED_CACHE2_DFB,
        .entry_size = interm_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = interm_data_format,
        .advanced_options = {.alias_with = {RMF_UNTILIZED_CACHE_DFB}},
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = RMF_OUTPUT_DFB,
        .entry_size = cache_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = cache_data_format,
    });

    if (use_index_tensor) {
        // Borrowed on the L1-sharded path only, where the reader reads the index straight out of the
        // resident tensor; on the DRAM path this is an ordinary L1 allocation the reader fills.
        DataflowBufferSpec index_dfb{
            .unique_id = RMF_INDEX_DFB,
            .entry_size = index_stick_size,
            .num_entries = 1,
            .data_format_metadata = index_data_format,
        };
        if (index_is_sharded) {
            index_dfb.borrowed_from = RMF_INDEX_TENSOR;
        }
        spec.dataflow_buffers.push_back(std::move(index_dfb));
    }

    if (is_paged_cache) {
        // Same conditional borrow as the index DFB above.
        DataflowBufferSpec page_table_dfb{
            .unique_id = RMF_PAGE_TABLE_DFB,
            .entry_size = page_table_stick_size,
            .num_entries = num_pages_page_table,
            .data_format_metadata = page_table_data_format,
        };
        if (page_table_is_sharded) {
            page_table_dfb.borrowed_from = RMF_PAGE_TABLE_TENSOR;
        }
        spec.dataflow_buffers.push_back(std::move(page_table_dfb));
    }

    //-------------------------------------------------------------------------
    // Semaphores
    //-------------------------------------------------------------------------
    // used for share cache for signaling when the cache is ready to be read
    spec.semaphores.push_back(SemaphoreSpec{
        .unique_id = RMF_SEQUENTIAL_MODE_SEM,
        .target_nodes = all_cores_bb,
    });

    //-------------------------------------------------------------------------
    // Tensor parameters
    //-------------------------------------------------------------------------
    // Both cache tensors are declared and bound on both dataflow kernels. Legacy delivered whichever
    // one a core writes through a single per-core address arg; a TensorBinding is per-KernelSpec, not
    // per-node, so the pair is bound everywhere and the kernels select with the is_input1 arg. This
    // is free on this channel: a binding's base address arrives as a common runtime arg broadcast to
    // every node, so both addresses are already correct everywhere, and nothing is allocated.
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = RMF_CACHE1_TENSOR,
        .spec = cache_tensor1.tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = RMF_CACHE2_TENSOR,
        .spec = cache_tensor2.tensor_spec(),
    });
    // Declared for the borrowed-memory DFBs above; no kernel binds either as a TensorAccessor.
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = RMF_INPUT1_TENSOR,
        .spec = input_tensor1.tensor_spec(),
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = RMF_INPUT2_TENSOR,
        .spec = input_tensor2.tensor_spec(),
    });
    if (use_index_tensor) {
        spec.tensor_parameters.push_back(TensorParameter{
            .unique_id = RMF_INDEX_TENSOR,
            .spec = update_idxs_tensor.value().tensor_spec(),
        });
    }
    if (is_paged_cache) {
        spec.tensor_parameters.push_back(TensorParameter{
            .unique_id = RMF_PAGE_TABLE_TENSOR,
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
        .unique_id = RMF_READER_KERNEL,
        .source = RMF_READER_SOURCE,
        .compiler_options = {.defines = conditional_defines},
        .semaphore_bindings =
            {
                SemaphoreBinding{
                    .semaphore_spec_name = RMF_SEQUENTIAL_MODE_SEM,
                    .accessor_name = "receiver",
                },
            },
        .compile_time_args =
            {
                {"index_is_dram", static_cast<uint32_t>(index_is_dram)},
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
                {"page_table_is_dram", static_cast<uint32_t>(page_table_is_dram)},
                {"St", St},
                {"batch_size", B},
            },
        .runtime_arg_schema =
            {.runtime_arg_names = {"has_work", "is_input1", "cache_start_id", "my_batch_idx", "wait_to_start"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };
    reader.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = RMF_CACHE_DFB,
        .accessor_name = "cache",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    reader.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = RMF_SRC1_DFB,
        .accessor_name = "src1",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    reader.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = RMF_SRC2_DFB,
        .accessor_name = "src2",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    reader.tensor_bindings.push_back(TensorBinding{
        .tensor_parameter_name = RMF_CACHE1_TENSOR,
        .accessor_name = "cache1",
    });
    reader.tensor_bindings.push_back(TensorBinding{
        .tensor_parameter_name = RMF_CACHE2_TENSOR,
        .accessor_name = "cache2",
    });
    if (use_index_tensor) {
        reader.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RMF_INDEX_DFB,
            .accessor_name = "index",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        // Bound whether or not the index tensor is sharded, mirroring legacy, which appended the
        // accessor's compile-time args on both paths and built the accessor unconditionally. Only
        // the read through it is gated (on index_is_dram).
        reader.tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = RMF_INDEX_TENSOR,
            .accessor_name = "index",
        });
    }
    if (is_paged_cache) {
        reader.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RMF_PAGE_TABLE_DFB,
            .accessor_name = "page_table",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader.tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = RMF_PAGE_TABLE_TENSOR,
            .accessor_name = "page_table",
        });
    }

    KernelSpec writer{
        .unique_id = RMF_WRITER_KERNEL,
        .source = RMF_WRITER_SOURCE,
        .compiler_options = {.defines = conditional_defines},
        .semaphore_bindings =
            {
                SemaphoreBinding{
                    .semaphore_spec_name = RMF_SEQUENTIAL_MODE_SEM,
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
                {"batch_size", B},
                {"page_table_stick_size", page_table_stick_size},
                {"page_table_is_dram", static_cast<uint32_t>(page_table_is_dram)},
            },
        // is_input1 is a legacy arg on this kernel (it already chose which input buffer to read); it
        // now also selects which of the two bound cache tensors to write, replacing the per-core
        // cache-address arg slot the port removes.
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"has_work",
                  "cache_start_id",
                  "cache_tile_offset_B",
                  "my_batch_idx",
                  "send_signal",
                  "send_core_x",
                  "send_core_y",
                  "is_input1"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };
    // The writer's `cache` accessor is the OUTPUT DFB: it holds the re-tilized cache block this
    // kernel writes back to the cache tensor. (The cache tiles the reader pulled in reach compute
    // through RMF_CACHE_DFB, which the writer never touches.)
    writer.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = RMF_OUTPUT_DFB,
        .accessor_name = "cache",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    writer.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = RMF_UNTILIZED_CACHE_DFB,
        .accessor_name = "untilized_cache",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    writer.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = RMF_UNTILIZED_CACHE2_DFB,
        .accessor_name = "untilized_cache2",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    // A row-major input arrives already untilized, so the "untilized input" this kernel consumes IS
    // the resident input buffer -- there is no intermediate. Both are bound and the kernel picks one
    // at runtime from is_input1; the accessor names keep the kernel's own vocabulary for the role.
    writer.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = RMF_SRC1_DFB,
        .accessor_name = "untilized_input1",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    writer.dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = RMF_SRC2_DFB,
        .accessor_name = "untilized_input2",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    writer.tensor_bindings.push_back(TensorBinding{
        .tensor_parameter_name = RMF_CACHE1_TENSOR,
        .accessor_name = "cache1",
    });
    writer.tensor_bindings.push_back(TensorBinding{
        .tensor_parameter_name = RMF_CACHE2_TENSOR,
        .accessor_name = "cache2",
    });
    if (use_index_tensor) {
        writer.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RMF_INDEX_DFB,
            .accessor_name = "index",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }
    if (is_paged_cache) {
        writer.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RMF_PAGE_TABLE_DFB,
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
        // The input DFBs are absent from this set: this kernel does not consume them.
        if (cache_data_format == tt::DataFormat::Float32) {
            compute_hw.unpack_modes.emplace(RMF_CACHE_DFB, UnpackMode::UnpackToSrc);
        }
        if (interm_data_format == tt::DataFormat::Float32) {
            compute_hw.unpack_modes.emplace(RMF_UNTILIZED_CACHE2_DFB, UnpackMode::UnpackToSrc);
        }
    }

    KernelSpec compute{
        .unique_id = RMF_COMPUTE_KERNEL,
        .source = RMF_COMPUTE_SOURCE,
        // Legacy left KernelDescriptor::opt_level unset, which resolves to O3 on a
        // ComputeConfigDescriptor; Metal 2.0's CompilerOptions defaults to O2, so set it explicitly.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                // Neither input DFB is bound here: a row-major input needs no untilize step, so this
                // kernel never touches them (the writer is their consumer instead).
                DFBBinding{
                    .dfb_spec_name = RMF_CACHE_DFB,
                    .accessor_name = "cache",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = RMF_UNTILIZED_CACHE_DFB,
                    .accessor_name = "untilized_cache",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = RMF_UNTILIZED_CACHE2_DFB,
                    .accessor_name = "untilized_cache2",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = RMF_OUTPUT_DFB,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .compile_time_args =
            {
                {"Wt", Wt},
                {"num_heads", num_heads},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"has_work", "is_input1"}},
        .hw_config = compute_hw,
    };

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));
    spec.kernels.push_back(std::move(compute));

    spec.work_units.push_back(WorkUnitSpec{
        .name = "paged_row_major_fused_update_cache",
        .kernels = {RMF_READER_KERNEL, RMF_WRITER_KERNEL, RMF_COMPUTE_KERNEL},
        .target_nodes = all_cores_bb,
    });

    //-------------------------------------------------------------------------
    // Run args
    //-------------------------------------------------------------------------
    ProgramRunArgs run_args;

    KernelRunArgs reader_run_args{.kernel = RMF_READER_KERNEL};
    KernelRunArgs writer_run_args{.kernel = RMF_WRITER_KERNEL};
    KernelRunArgs compute_run_args{.kernel = RMF_COMPUTE_KERNEL};

    constexpr bool has_work = true;
    constexpr bool is_input1 = true;

    const auto& cores1 = corerange_to_cores(input1_cores, input1_cores.num_cores(), row_major);
    const auto& cores2 = corerange_to_cores(input2_cores, input2_cores.num_cores(), row_major);

    // cache_start_id / tile_update_offset_B are derived from update_idxs (excluded from the program
    // hash) — computed via the shared helper so override_runtime_arguments re-patches identical values
    // on cache hits. Empty in index-tensor mode (offsets read on-device from the index tensor).
    const auto offsets = compute_row_major_fused_offsets(operation_attributes, tensor_args);
    for (uint32_t i = 0; i < cores1.size(); ++i) {
        const CoreCoord& core1 = cores1.at(i);
        const CoreCoord& core2 = cores2.at(i);

        // Cache tile info
        const uint32_t cache_start_id = use_index_tensor ? 0u : offsets.at(i).cache_start_id;
        const uint32_t tile_update_offset_B = use_index_tensor ? 0u : offsets.at(i).tile_update_offset_B;

        // Calculate synchronization parameters
        bool wait_to_start = operation_attributes.share_cache and (i != 0);
        bool send_signal = operation_attributes.share_cache and (i != cores1.size() - 1);
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

        // Index i handles input1 on core1 (writing cache_tensor1) and input2 on core2 (writing
        // cache_tensor2); both share the same offsets.
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core1,
            {
                {"has_work", static_cast<uint32_t>(has_work)},
                {"is_input1", static_cast<uint32_t>(is_input1)},
                {"cache_start_id", cache_start_id},
                {"my_batch_idx", i},
                {"wait_to_start", static_cast<uint32_t>(wait_to_start)},
            });
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core1,
            {
                {"has_work", static_cast<uint32_t>(has_work)},
                {"cache_start_id", cache_start_id},
                {"cache_tile_offset_B", tile_update_offset_B},
                {"my_batch_idx", i},
                {"send_signal", static_cast<uint32_t>(send_signal)},
                {"send_core_x", send_core1_x},
                {"send_core_y", send_core1_y},
                {"is_input1", static_cast<uint32_t>(is_input1)},
            });
        AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values,
            core1,
            {
                {"has_work", static_cast<uint32_t>(has_work)},
                {"is_input1", static_cast<uint32_t>(is_input1)},
            });

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core2,
            {
                {"has_work", static_cast<uint32_t>(has_work)},
                {"is_input1", static_cast<uint32_t>(!is_input1)},
                {"cache_start_id", cache_start_id},
                {"my_batch_idx", i},
                {"wait_to_start", static_cast<uint32_t>(wait_to_start)},
            });
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core2,
            {
                {"has_work", static_cast<uint32_t>(has_work)},
                {"cache_start_id", cache_start_id},
                {"cache_tile_offset_B", tile_update_offset_B},
                {"my_batch_idx", i},
                {"send_signal", static_cast<uint32_t>(send_signal)},
                {"send_core_x", send_core2_x},
                {"send_core_y", send_core2_y},
                {"is_input1", static_cast<uint32_t>(!is_input1)},
            });
        AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values,
            core2,
            {
                {"has_work", static_cast<uint32_t>(has_work)},
                {"is_input1", static_cast<uint32_t>(!is_input1)},
            });
    }

    // Runtime args for cores in the bounding box that carry no work. Legacy gave these nodes a single
    // arg and let the kernels early-return on it; a runtime_arg_schema is one schema for the whole
    // KernelSpec and SetProgramRunArgs requires every declared name on every node the kernel runs on,
    // so the full named set is supplied with has_work = 0 and don't-care zeros. The kernels return
    // before reading any of the rest, so this is the same program as legacy built.
    for (const auto& core_range : unused_cores.ranges()) {
        for (const auto& core : core_range) {
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {
                    {"has_work", static_cast<uint32_t>(!has_work)},
                    {"is_input1", 0u},
                    {"cache_start_id", 0u},
                    {"my_batch_idx", 0u},
                    {"wait_to_start", 0u},
                });
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {
                    {"has_work", static_cast<uint32_t>(!has_work)},
                    {"cache_start_id", 0u},
                    {"cache_tile_offset_B", 0u},
                    {"my_batch_idx", 0u},
                    {"send_signal", 0u},
                    {"send_core_x", 0u},
                    {"send_core_y", 0u},
                    {"is_input1", 0u},
                });
            AddRuntimeArgsForNode(
                compute_run_args.runtime_arg_values,
                core,
                {
                    {"has_work", static_cast<uint32_t>(!has_work)},
                    {"is_input1", 0u},
                });
        }
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));

    // In-place op: tensor_return_value aliases the two cache tensors, which are the buffers the
    // kernels write.
    run_args.tensor_args.emplace(RMF_CACHE1_TENSOR, cache_tensor1.mesh_tensor());
    run_args.tensor_args.emplace(RMF_CACHE2_TENSOR, cache_tensor2.mesh_tensor());
    run_args.tensor_args.emplace(RMF_INPUT1_TENSOR, input_tensor1.mesh_tensor());
    run_args.tensor_args.emplace(RMF_INPUT2_TENSOR, input_tensor2.mesh_tensor());
    if (use_index_tensor) {
        run_args.tensor_args.emplace(RMF_INDEX_TENSOR, update_idxs_tensor.value().mesh_tensor());
    }
    if (is_paged_cache) {
        run_args.tensor_args.emplace(RMF_PAGE_TABLE_TENSOR, page_table.value().mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

tt::tt_metal::experimental::ProgramRunArgs PagedRowMajorFusedUpdateCacheProgramFactory::override_runtime_arguments(
    const PagedFusedUpdateCacheParams& operation_attributes,
    const PagedFusedUpdateCacheInputs& tensor_args,
    PagedFusedUpdateCacheResult& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    using namespace CMAKE_UNIQUE_NAMESPACE_ROW_MAJOR;

    // Runs on EVERY cache hit, and on this concept the framework refreshes nothing on our behalf, so
    // this must re-derive all per-dispatch state: every tensor binding — including the two input
    // tensors, whose addresses back the borrowed-memory src1 / src2 DFBs, and the index / page-table
    // tensors, which back theirs when sharded — and the update_idxs-derived offsets the program hash
    // excludes. Everything else is a function of hashed inputs (shard grids, share_cache, shapes,
    // dtypes, compute config) and is identical by construction on a hit — UpdateProgramRunArgs is a
    // partial update, so anything omitted here keeps its cache-miss value.
    ProgramRunArgs run_args;

    // Mirrors the legacy body: a coordinate excluded from the dispatch has nothing to patch. This
    // factory is selected only when mesh_coords is nullopt (see select_program_factory), so the guard
    // is inert here, but it is kept so the two factories stay behaviourally identical.
    if (operation_attributes.mesh_coords.has_value() && mesh_dispatch_coordinate.has_value() &&
        !operation_attributes.mesh_coords.value().contains(mesh_dispatch_coordinate.value())) {
        return run_args;
    }

    const auto& update_idxs_tensor = tensor_args.update_idxs_tensor;
    const auto& page_table = tensor_args.page_table;

    run_args.tensor_args.emplace(RMF_CACHE1_TENSOR, tensor_args.cache_tensor1.mesh_tensor());
    run_args.tensor_args.emplace(RMF_CACHE2_TENSOR, tensor_args.cache_tensor2.mesh_tensor());
    run_args.tensor_args.emplace(RMF_INPUT1_TENSOR, tensor_args.input_tensor1.mesh_tensor());
    run_args.tensor_args.emplace(RMF_INPUT2_TENSOR, tensor_args.input_tensor2.mesh_tensor());
    if (update_idxs_tensor.has_value()) {
        run_args.tensor_args.emplace(RMF_INDEX_TENSOR, update_idxs_tensor.value().mesh_tensor());
    }
    if (page_table.has_value()) {
        run_args.tensor_args.emplace(RMF_PAGE_TABLE_TENSOR, page_table.value().mesh_tensor());
    }

    // Empty in index-tensor mode: the kernels read positions on-device from the index tensor, so the
    // offset slots keep the 0 they were given on the cache miss. Nodes outside cores1 / cores2 only
    // ever got has_work = 0, so they need no patching either.
    const auto offsets = compute_row_major_fused_offsets(operation_attributes, tensor_args);
    if (offsets.empty()) {
        return run_args;
    }

    KernelRunArgs reader_run_args{.kernel = RMF_READER_KERNEL};
    KernelRunArgs writer_run_args{.kernel = RMF_WRITER_KERNEL};
    for (const auto& offset : offsets) {
        for (const CoreCoord& core : {offset.core1, offset.core2}) {
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values, core, {{"cache_start_id", offset.cache_start_id}});
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {
                    {"cache_start_id", offset.cache_start_id},
                    {"cache_tile_offset_B", offset.tile_update_offset_B},
                });
        }
    }
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    return run_args;
}

// ---------------------------------------------------------------------------------------------
// Metal 2.0 mesh-workload build (MeshWorkloadSpecFactoryConcept).
//
// This factory runs a DIFFERENT set of programs per mesh coordinate: a coordinate outside
// operation_attributes.mesh_coords gets none at all.  That is what kept it on the descriptor
// concept until MeshWorkloadSpecFactoryConcept landed.
// ---------------------------------------------------------------------------------------------
ttnn::device_operation::MeshWorkloadArtifacts
PagedRowMajorFusedUpdateCacheMeshWorkloadFactory::create_mesh_workload_artifacts(
    const PagedFusedUpdateCacheParams& operation_attributes,
    const PagedFusedUpdateCacheInputs& tensor_args,
    PagedFusedUpdateCacheResult& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    // The spec is identical on every coordinate this op runs on -- same kernels, same DFBs, same
    // bindings, same core ranges, none of which depends on the coordinate.  What varies across the
    // mesh is only *which* coordinates get a program.  So build once and stamp it onto each one.
    auto artifacts = PagedRowMajorFusedUpdateCacheProgramFactory::create_program_artifacts(
        operation_attributes, tensor_args, tensor_return_value);

    const auto coords = CMAKE_UNIQUE_NAMESPACE_ROW_MAJOR::fused_dispatch_coords(operation_attributes, tensor_coords);

    ttnn::device_operation::MeshWorkloadArtifacts workload;
    workload.programs.reserve(coords.size());
    for (const auto& coord : coords) {
        workload.programs.push_back({
            .range = ttnn::MeshCoordinateRange(coord),
            .spec = artifacts.spec,
            .run_params = artifacts.run_params,
        });
    }
    return workload;
}

tt::tt_metal::experimental::ProgramRunArgs PagedRowMajorFusedUpdateCacheMeshWorkloadFactory::override_runtime_arguments(
    const PagedFusedUpdateCacheParams& operation_attributes,
    const PagedFusedUpdateCacheInputs& tensor_args,
    PagedFusedUpdateCacheResult& tensor_return_value,
    const ttnn::MeshCoordinateRange& /*coordinate_range*/) {
    // Called once per range, and every range that exists here is one this dispatch runs on -- an
    // excluded coordinate has no program to refresh.  So the coordinate test the ported-from patch
    // performed on every hit is structural now rather than a runtime check, and the refresh is the
    // single-device one unchanged: the per-dispatch state is the same on every device the op runs on.
    //
    // std::nullopt rather than this range's coordinate, deliberately: the single-device override
    // takes an optional coordinate only to run that same exclusion test, which is a no-op here for
    // every range by construction.
    return PagedRowMajorFusedUpdateCacheProgramFactory::override_runtime_arguments(
        operation_attributes, tensor_args, tensor_return_value, std::nullopt);
}

}  // namespace ttnn::experimental::prim
