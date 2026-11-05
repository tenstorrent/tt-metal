// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "paged_tiled_fused_update_cache_program_factory.hpp"

#include <algorithm>
#include <optional>

#include "paged_fused_update_cache_device_operation.hpp"
#include "paged_fused_update_cache_device_operation_types.hpp"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;

namespace CMAKE_UNIQUE_NAMESPACE_TILED {

bool enable_fp32_dest_acc(
    const tt_metal::distributed::MeshDevice& device, const ttnn::DeviceComputeKernelConfig& compute_kernel_config) {
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device.arch(), compute_kernel_config);

    return fp32_dest_acc_en;
}

// Spec names for the tiled fused program. The reader / writer / compute sources are each instantiated
// twice, once per input shard grid, so their kernel names carry the input they serve.
const KernelSpecName TF_READER1{"reader1"};
const KernelSpecName TF_READER2{"reader2"};
const KernelSpecName TF_WRITER1{"writer1"};
const KernelSpecName TF_WRITER2{"writer2"};
const KernelSpecName TF_COMPUTE1{"compute1"};
const KernelSpecName TF_COMPUTE2{"compute2"};
// Helper triples: extra cores that take a share of a user's heads (see HeadSplit).
const KernelSpecName TF_READER1_HELPER{"reader1_helper"};
const KernelSpecName TF_READER2_HELPER{"reader2_helper"};
const KernelSpecName TF_WRITER1_HELPER{"writer1_helper"};
const KernelSpecName TF_WRITER2_HELPER{"writer2_helper"};
const KernelSpecName TF_COMPUTE1_HELPER{"compute1_helper"};
const KernelSpecName TF_COMPUTE2_HELPER{"compute2_helper"};

const DFBSpecName TF_CACHE_TILES{"cache_tiles"};
const DFBSpecName TF_INPUT1{"input1_shard"};
const DFBSpecName TF_INPUT2{"input2_shard"};
// On a helper core the user's input tiles are copied in over the NoC, into a buffer of the program's own.
const DFBSpecName TF_INPUT_COPY{"input_copy"};
const DFBSpecName TF_INDEX{"index"};
const DFBSpecName TF_PAGE_TABLE{"page_table"};
const DFBSpecName TF_UNTILIZED_CACHE{"untilized_cache"};
const DFBSpecName TF_UNTILIZED_CACHE2{"untilized_cache2"};
const DFBSpecName TF_UNTILIZED_INPUT{"untilized_input"};
const DFBSpecName TF_OUT_TILES{"out_tiles"};
// bfp8_b row patch: the input packed to bfp8_b, and the cache granules the writer patches.
const DFBSpecName TF_PATCH_SRC{"patch_src"};
const DFBSpecName TF_PATCH_SCRATCH{"patch_scratch"};

const SemaphoreSpecName TF_IN0_SEQUENTIAL{"in0_sequential_mode"};

const TensorParamName TF_CACHE1{"cache1"};
const TensorParamName TF_CACHE2{"cache2"};
const TensorParamName TF_INPUT1_T{"input1"};
const TensorParamName TF_INPUT2_T{"input2"};
const TensorParamName TF_INDEX_T{"index"};
const TensorParamName TF_PAGE_TABLE_T{"page_table"};

// The tensor arguments every dispatch must (re)bind. Shared by the cache-miss artifacts build and the
// cache-hit override so the two cannot disagree about which optional tensors are present. The two
// input tensors are here even though no kernel binds them: each backs a borrowed input-shard buffer,
// whose SRAM address resolves from its argument.
Table<TensorParamName, TensorArgument> tiled_fused_tensor_args(const PagedFusedUpdateCacheInputs& tensor_args) {
    Table<TensorParamName, TensorArgument> args;
    args.emplace(TF_CACHE1, tensor_args.cache_tensor1.mesh_tensor());
    args.emplace(TF_CACHE2, tensor_args.cache_tensor2.mesh_tensor());
    args.emplace(TF_INPUT1_T, tensor_args.input_tensor1.mesh_tensor());
    args.emplace(TF_INPUT2_T, tensor_args.input_tensor2.mesh_tensor());
    if (tensor_args.update_idxs_tensor.has_value()) {
        args.emplace(TF_INDEX_T, tensor_args.update_idxs_tensor->mesh_tensor());
    }
    if (tensor_args.page_table.has_value()) {
        args.emplace(TF_PAGE_TABLE_T, tensor_args.page_table->mesh_tensor());
    }
    return args;
}

// How each user's heads are spread over cores. The core holding the user's input shard updates the first head
// range; `groups - 1` helper cores per input update the others. Updating one head is a serial read-modify-write
// of a cache block (DRAM read, untilize, row copy, tilize, DRAM write), so one core walking every head of a user
// is the op's critical path when there are few users (32 heads at batch 1: 28 us on a P150). Helpers take the free
// cores of the compute grid in row-major order: the first `users * (groups - 1)` for input 1, the next for input 2.
// The share_cache mode chains users through a semaphore and keeps one core per user. An index tensor or page
// table sharded into SRAM is read in place from the shard cores' own L1, which helpers do not have, so those
// configurations keep one core per user too.
struct HeadSplit {
    uint32_t groups = 1;
    std::vector<CoreCoord> helpers1;  // [user * (groups - 1) + group - 1]
    std::vector<CoreCoord> helpers2;

    uint32_t head_start(uint32_t num_heads, uint32_t group) const { return group * num_heads / groups; }
    uint32_t num_local_heads(uint32_t num_heads, uint32_t group) const {
        return head_start(num_heads, group + 1) - head_start(num_heads, group);
    }
};

HeadSplit compute_head_split(
    const PagedFusedUpdateCacheParams& operation_attributes, const PagedFusedUpdateCacheInputs& tensor_args) {
    HeadSplit split;
    const uint32_t num_heads = tensor_args.cache_tensor1.padded_shape()[1];
    const CoreRangeSet input1_cores = tensor_args.input_tensor1.shard_spec().value().grid;
    const CoreRangeSet input2_cores = tensor_args.input_tensor2.shard_spec().value().grid;
    const uint32_t users = input1_cores.num_cores();
    const bool index_is_sharded =
        tensor_args.update_idxs_tensor.has_value() && tensor_args.update_idxs_tensor->is_sharded();
    const bool page_table_is_sharded = tensor_args.page_table.has_value() && tensor_args.page_table->is_sharded();
    if (operation_attributes.share_cache || index_is_sharded || page_table_is_sharded || num_heads <= 1 ||
        users == 0) {
        return split;
    }
    const CoreRangeSet shard_cores = input1_cores.merge(input2_cores);
    const CoreCoord grid = tensor_args.input_tensor1.device()->compute_with_storage_grid_size();
    std::vector<CoreCoord> free_cores;
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            const CoreCoord core{x, y};
            if (!shard_cores.contains(core)) {
                free_cores.push_back(core);
            }
        }
    }
    const uint32_t helpers_per_user = std::min<uint32_t>(num_heads - 1, free_cores.size() / (2 * users));
    split.groups = 1 + helpers_per_user;
    const uint32_t per_input = users * helpers_per_user;
    split.helpers1.assign(free_cores.begin(), free_cores.begin() + per_input);
    split.helpers2.assign(free_cores.begin() + per_input, free_cores.begin() + 2 * per_input);
    return split;
}

CoreRangeSet cores_to_range_set(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> ranges;
    ranges.reserve(cores.size());
    for (const auto& core : cores) {
        ranges.emplace_back(core, core);
    }
    return CoreRangeSet(ranges);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE_TILED

std::vector<PagedTiledFusedUpdateCacheProgramFactory::PerIndexOffsets>
PagedTiledFusedUpdateCacheProgramFactory::compute_tiled_fused_offsets(
    const PagedFusedUpdateCacheParams& operation_attributes, const PagedFusedUpdateCacheInputs& tensor_args) {
    // cache_start_id / tile_update_offset_B are derived from update_idxs, which is excluded from the
    // program hash (see PagedFusedUpdateCacheDeviceOperation::compute_program_hash) yet baked into runtime
    // args, so they must be re-applied on every cache hit. This helper is the single source of truth for
    // the formulas — both create_program_artifacts (cache miss) and override_runtime_arguments (cache hit)
    // call it, so the two paths cannot drift. Returns empty when an index tensor is used: in that mode the
    // offsets are 0 here and the real positions are read on-device from the (re-patched) index tensor.
    if (tensor_args.update_idxs_tensor.has_value()) {
        return {};
    }

    const auto& cache_tensor1 = tensor_args.cache_tensor1;
    const auto& input_tensor1 = tensor_args.input_tensor1;
    const auto& input_tensor2 = tensor_args.input_tensor2;
    const bool fp32_dest_acc_en = CMAKE_UNIQUE_NAMESPACE_TILED::enable_fp32_dest_acc(
                                      *input_tensor1.device(), operation_attributes.compute_kernel_config) ||
                                  cache_tensor1.dtype() == DataType::FLOAT32;

    const uint32_t Wt = cache_tensor1.padded_shape()[-1] / TILE_WIDTH;
    const uint32_t Wbytes = fp32_dest_acc_en ? cache_tensor1.padded_shape()[-1] * sizeof(float)
                                             : cache_tensor1.padded_shape()[-1] * 2;  // 2 bytes for bfloat16
    const uint32_t cache_total_num_tiles = cache_tensor1.physical_volume() / TILE_HW;
    // share_cache => batch offset is 0 (one shared cache buffer); mirror the artifacts build exactly.
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

ttnn::device_operation::ProgramArtifacts PagedTiledFusedUpdateCacheProgramFactory::create_program_artifacts(
    const PagedFusedUpdateCacheParams& operation_attributes,
    const PagedFusedUpdateCacheInputs& tensor_args,
    PagedFusedUpdateCacheResult& /*tensor_return_value*/) {
    using namespace CMAKE_UNIQUE_NAMESPACE_TILED;

    const auto& cache_tensor1 = tensor_args.cache_tensor1;
    const auto& input_tensor1 = tensor_args.input_tensor1;
    const auto& cache_tensor2 = tensor_args.cache_tensor2;
    const auto& input_tensor2 = tensor_args.input_tensor2;
    const auto& update_idxs_tensor = tensor_args.update_idxs_tensor;
    const auto& page_table = tensor_args.page_table;

    tt_metal::distributed::MeshDevice* device = input_tensor1.device();

    tt::DataFormat cache_dfb_data_format = tt_metal::datatype_to_dataformat_converter(cache_tensor1.dtype());
    uint32_t cache_single_tile_size = tt::tile_size(cache_dfb_data_format);

    tt::DataFormat input_dfb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor1.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_dfb_data_format);

    const bool preserve_fp32_cache = cache_tensor1.dtype() == DataType::FLOAT32;
    bool fp32_dest_acc_en =
        enable_fp32_dest_acc(*device, operation_attributes.compute_kernel_config) || preserve_fp32_cache;

    tt::DataFormat interm_dfb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t interm_single_tile_size = tt::tile_size(interm_dfb_data_format);

    const uint32_t B = input_tensor1.padded_shape()[1];
    const uint32_t num_heads = cache_tensor1.padded_shape()[1];

    // Index tensor-specific parameters
    bool use_index_tensor = update_idxs_tensor.has_value();
    uint32_t log2_page_size = 0;
    uint32_t index_stick_size = 0;
    tt::DataFormat index_data_format = tt::DataFormat::Int32;
    bool index_is_dram = true;
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
    bool page_table_is_sharded = false;
    if (is_paged_cache) {
        const auto& page_table_tensor = page_table.value();
        page_table_is_sharded = page_table.value().is_sharded();
        num_pages_page_table = page_table.value().is_sharded() ? B : 1;
        block_size = cache_tensor1.padded_shape()[2];
        block_size_t = block_size / TILE_HEIGHT;
        max_blocks_per_seq = page_table_tensor.padded_shape()[1];
        page_table_stick_size = page_table.value().buffer()->aligned_page_size();
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

    log_debug(tt::LogOp, "cache_dfb_data_format: {}", cache_dfb_data_format);
    log_debug(tt::LogOp, "input_dfb_data_format: {}", input_dfb_data_format);
    log_debug(tt::LogOp, "interm_dfb_data_format: {}", interm_dfb_data_format);
    log_debug(tt::LogOp, "Wbytes: {}", Wbytes);
    log_debug(tt::LogOp, "Wt: {}", Wt);
    log_debug(tt::LogOp, "St: {}", St);

    const std::optional<ShardSpec>& input1_shard_spec = input_tensor1.shard_spec();
    const std::optional<ShardSpec>& input2_shard_spec = input_tensor2.shard_spec();
    bool row_major = input1_shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
    CoreRangeSet input1_cores = input1_shard_spec.value().grid;
    CoreRangeSet input2_cores = input2_shard_spec.value().grid;
    CoreRangeSet all_cores = input1_cores.merge(input2_cores);

    uint32_t num_input_tiles = input1_shard_spec.value().shape[0] * input1_shard_spec.value().shape[1] / TILE_HW;

    // bfp8_b cache: write only the updated row. A bfp8_b tile keeps one shared exponent per 16-value face row, so the
    // row's bytes (2 exponents and 2 x 16 mantissas per tile) are the packed input row's, and the rest of the tile is
    // untouched. The writer reads, patches and writes back three aligned 64-byte granules per tile instead of the
    // reader, compute and writer moving the whole tile through untilize / tilize (P150, 32 users x 32 heads:
    // 13.4 MB of DRAM traffic per call at the DRAM limit).
    const bool row_patch = cache_tensor1.dtype() == DataType::BFLOAT8_B &&
                           cache_tensor1.tensor_spec().tile().get_height() == TILE_HEIGHT &&
                           cache_tensor1.tensor_spec().tile().get_width() == TILE_WIDTH &&
                           input_tensor1.tensor_spec().tile().get_height() == TILE_HEIGHT;
    constexpr uint32_t row_patch_heads_per_group = 4;  // the kernels' row_patch::kHeadsPerGroup
    uint32_t num_cache_tiles = 2 * Wt;   // double buffered
    uint32_t num_interm_tiles = 2 * Wt;  // double buffered
    uint32_t num_output_tiles = B * Wt;

    // ---------------- Dataflow buffers ----------------

    // Each input buffer is built on its own tensor's resident shard, and a buffer lives wherever its
    // bound kernels run. That is why the kernels below are split one triple per shard grid rather than
    // placed over the bounding box of the two: a kernel spanning both grids would extend input1's
    // borrow onto cores where input_tensor1 has no shard at all.
    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = TF_INPUT1,
            .entry_size = input_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = input_dfb_data_format,
            .borrowed_from = TF_INPUT1_T,
        },
        DataflowBufferSpec{
            .unique_id = TF_INPUT2,
            .entry_size = input_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = input_dfb_data_format,
            .borrowed_from = TF_INPUT2_T,
        },
    };
    if (row_patch) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = TF_PATCH_SRC,
            .entry_size = cache_single_tile_size,
            .num_entries = Wt,
            .data_format_metadata = cache_dfb_data_format,
        });
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = TF_PATCH_SCRATCH,
            .entry_size = cache_single_tile_size,  // a tile prefix of up to the whole tile (see the reader)
            .num_entries = 2 * row_patch_heads_per_group * Wt,
            .data_format_metadata = cache_dfb_data_format,
        });
    } else {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = TF_CACHE_TILES,
            .entry_size = cache_single_tile_size,
            .num_entries = num_cache_tiles,
            .data_format_metadata = cache_dfb_data_format,
        });
        // These two buffers share one backing region, so each names the other. The alias group
        // must agree on total size and on the set of kernels binding it; both hold here, since
        // the writer and compute each bind both.
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = TF_UNTILIZED_CACHE,
            .entry_size = interm_single_tile_size,
            .num_entries = num_interm_tiles,
            .data_format_metadata = interm_dfb_data_format,
            .advanced_options = {.alias_with = {TF_UNTILIZED_CACHE2}},
        });
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = TF_UNTILIZED_CACHE2,
            .entry_size = interm_single_tile_size,
            .num_entries = num_interm_tiles,
            .data_format_metadata = interm_dfb_data_format,
            .advanced_options = {.alias_with = {TF_UNTILIZED_CACHE}},
        });
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = TF_UNTILIZED_INPUT,
            .entry_size = interm_single_tile_size,
            .num_entries = num_interm_tiles,
            .data_format_metadata = interm_dfb_data_format,
        });
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = TF_OUT_TILES,
            .entry_size = cache_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = cache_dfb_data_format,
        });
    }
    const HeadSplit head_split = compute_head_split(operation_attributes, tensor_args);
    if (head_split.groups > 1) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = TF_INPUT_COPY,
            .entry_size = input_single_tile_size,
            .num_entries = Wt,
            .data_format_metadata = input_dfb_data_format,
        });
    }
    if (use_index_tensor) {
        // An index tensor sharded into SRAM is read straight out of its own shard; a DRAM one is
        // NoC-read into a buffer of the program's own.
        DataflowBufferSpec index_dfb{
            .unique_id = TF_INDEX,
            .entry_size = index_stick_size,
            .num_entries = 1,
            .data_format_metadata = index_data_format,
        };
        if (index_is_sharded) {
            index_dfb.borrowed_from = TF_INDEX_T;
        }
        dataflow_buffers.push_back(std::move(index_dfb));
    }
    if (is_paged_cache) {
        DataflowBufferSpec page_table_dfb{
            .unique_id = TF_PAGE_TABLE,
            .entry_size = page_table_stick_size,
            .num_entries = num_pages_page_table,
            .data_format_metadata = page_table_data_format,
        };
        if (page_table_is_sharded) {
            page_table_dfb.borrowed_from = TF_PAGE_TABLE_T;
        }
        dataflow_buffers.push_back(std::move(page_table_dfb));
    }

    // The optional buffers and the tensors behind them exist only under their host flag, and the
    // kernels name them from blocks the preprocessor has to remove on the off-path: a discarded
    // `if constexpr` branch still resolves names, so an unbound dfb:: or tensor:: token would fail to
    // compile. Both the reader and the writer read each flag, so both get every define.
    KernelSpec::CompilerOptions::Defines optional_resource_defines;
    if (use_index_tensor) {
        optional_resource_defines.emplace("USE_INDEX_TENSOR", "1");
        if (index_is_dram) {
            optional_resource_defines.emplace("INDEX_IS_DRAM", "1");
        }
    }
    if (row_patch) {
        optional_resource_defines.emplace("ROW_PATCH", "1");
    }
    if (is_paged_cache) {
        optional_resource_defines.emplace("IS_PAGED_CACHE", "1");
        if (page_table_is_dram) {
            optional_resource_defines.emplace("PAGE_TABLE_IS_DRAM", "1");
        }
    }

    // ---------------- Kernels, one triple per input shard grid ----------------

    Group<KernelSpec> kernels;

    // Builds the reader / writer / compute triple that serves one of the two inputs. The pair differs
    // only in which input buffer and which cache tensor it binds, which is what makes the choice
    // compile-time: the legacy kernels picked between the two buffers from a per-core runtime arg.
    const auto add_kernel_triple = [&](const KernelSpecName& reader_id,
                                       const KernelSpecName& writer_id,
                                       const KernelSpecName& compute_id,
                                       const DFBSpecName& input_dfb,
                                       const TensorParamName& cache_tensor_param,
                                       const std::optional<TensorParamName>& remote_input_param) {
        KernelSpec::CompilerOptions::Defines reader_defines = optional_resource_defines;
        if (remote_input_param.has_value()) {
            reader_defines.emplace("REMOTE_INPUT", "1");
        }
        Group<DFBBinding> reader_dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = input_dfb,
                .accessor_name = "input",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
        };
        if (row_patch) {
            reader_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = TF_PATCH_SCRATCH,
                .accessor_name = "patch_scratch",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
        } else {
            reader_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = TF_CACHE_TILES,
                .accessor_name = "cache",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
        }
        Group<TensorBinding> reader_tensor_bindings = {
            TensorBinding{
                .tensor_parameter_name = cache_tensor_param,
                .accessor_name = "cache",
            },
        };
        Group<DFBBinding> writer_dfb_bindings;
        if (row_patch) {
            writer_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = TF_PATCH_SRC,
                .accessor_name = "patch_src",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
            writer_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = TF_PATCH_SCRATCH,
                .accessor_name = "patch_scratch",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        } else {
          writer_dfb_bindings = {
            // The writer's `cache` handle names the *output* buffer, the one compute fills with
            // the retilized block. The reader's handle of the same name is the cache buffer it reads
            // from DRAM; accessor names are per-kernel, and this is where the two are told apart.
            DFBBinding{
                .dfb_spec_name = TF_OUT_TILES,
                .accessor_name = "cache",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            DFBBinding{
                .dfb_spec_name = TF_UNTILIZED_CACHE,
                .accessor_name = "untilized_cache",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            DFBBinding{
                .dfb_spec_name = TF_UNTILIZED_CACHE2,
                .accessor_name = "untilized_cache2",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            DFBBinding{
                .dfb_spec_name = TF_UNTILIZED_INPUT,
                .accessor_name = "untilized_input",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
          };
        }

        if (remote_input_param.has_value()) {
            reader_tensor_bindings.push_back(TensorBinding{
                .tensor_parameter_name = *remote_input_param,
                .accessor_name = "input",
            });
        }
        if (use_index_tensor) {
            reader_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = TF_INDEX,
                .accessor_name = "index",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            writer_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = TF_INDEX,
                .accessor_name = "index",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
            // Only the DRAM configuration reaches the tensor through an accessor; the sharded one
            // reads the borrowed buffer directly, so it neither needs nor binds the tensor.
            if (index_is_dram) {
                reader_tensor_bindings.push_back(TensorBinding{
                    .tensor_parameter_name = TF_INDEX_T,
                    .accessor_name = "index",
                });
            }
        }
        if (is_paged_cache) {
            reader_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = TF_PAGE_TABLE,
                .accessor_name = "page_table",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            writer_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = TF_PAGE_TABLE,
                .accessor_name = "page_table",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
            if (page_table_is_dram) {
                reader_tensor_bindings.push_back(TensorBinding{
                    .tensor_parameter_name = TF_PAGE_TABLE_T,
                    .accessor_name = "page_table",
                });
            }
        }

        kernels.push_back(KernelSpec{
            .unique_id = reader_id,
            .source = "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
                      "reader_paged_fused_update_cache_interleaved_start_id.cpp",
            .compiler_options = {.defines = reader_defines},
            .dfb_bindings = std::move(reader_dfb_bindings),
            .semaphore_bindings =
                {
                    SemaphoreBinding{
                        .semaphore_spec_name = TF_IN0_SEQUENTIAL,
                        .accessor_name = "in0_seq",
                    },
                },
            .tensor_bindings = std::move(reader_tensor_bindings),
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
                    {"batch_size", B},
                },
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"has_work",
                      "cache_start_id",
                      "my_batch_idx",
                      "wait_to_start",
                      "head_start",
                      "num_local_heads",
                      "cache_row"}},
            .hw_config = create_reader_datamovement_config(),
        });

        kernels.push_back(KernelSpec{
            .unique_id = writer_id,
            .source = "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
                      "writer_paged_fused_update_cache_interleaved_start_id.cpp",
            .compiler_options = {.defines = optional_resource_defines},
            .dfb_bindings = std::move(writer_dfb_bindings),
            .semaphore_bindings =
                {
                    SemaphoreBinding{
                        .semaphore_spec_name = TF_IN0_SEQUENTIAL,
                        .accessor_name = "in0_seq",
                    },
                },
            .tensor_bindings =
                {
                    TensorBinding{
                        .tensor_parameter_name = cache_tensor_param,
                        .accessor_name = "cache",
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
                },
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"has_work",
                      "cache_start_id",
                      "cache_tile_offset_B",
                      "my_batch_idx",
                      "send_signal",
                      "send_core_x",
                      "send_core_y",
                      "head_start",
                      "num_local_heads"}},
            .hw_config = create_writer_datamovement_config(),
        });

        // Legacy built a ComputeConfigDescriptor that set only fp32_dest_acc_en, leaving every other
        // knob at its default even when the caller's compute_kernel_config specified one.
        // ComputeHardwareConfig's defaults coincide with those, so setting only enable_32_bit_dest
        // reproduces it exactly.
        ComputeHardwareConfig compute_hw{.enable_32_bit_dest = fp32_dest_acc_en};
        if (fp32_dest_acc_en) {
            // A 32-bit Dest requires an explicit unpack mode for every Float32 buffer the compute
            // kernel consumes. UnpackToDest preserves full precision for FLOAT32 caches; for other caches,
            // retain the legacy UnpackToSrc behavior.
            const auto require_unpack_mode = [&](const DFBSpecName& dfb, tt::DataFormat format) {
                if (format == tt::DataFormat::Float32) {
                    compute_hw.unpack_modes.emplace(
                        dfb, preserve_fp32_cache ? UnpackMode::UnpackToDest : UnpackMode::UnpackToSrc);
                }
            };
            require_unpack_mode(input_dfb, input_dfb_data_format);
            if (!row_patch) {
                require_unpack_mode(TF_CACHE_TILES, cache_dfb_data_format);
                require_unpack_mode(TF_UNTILIZED_CACHE2, interm_dfb_data_format);
            }
        }

        KernelSpec::CompilerOptions::Defines compute_defines;
        if (row_patch) {
            compute_defines.emplace("ROW_PATCH", "1");
        }
        Group<DFBBinding> compute_dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = input_dfb,
                .accessor_name = "in",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
        };
        if (row_patch) {
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = TF_PATCH_SRC,
                .accessor_name = "patch_src",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
        } else {
            for (const auto& b : {
                     DFBBinding{
                         .dfb_spec_name = TF_CACHE_TILES,
                         .accessor_name = "cache",
                         .endpoint_type = DFBEndpointType::CONSUMER,
                     },
                     DFBBinding{
                         .dfb_spec_name = TF_UNTILIZED_CACHE,
                         .accessor_name = "untilized_cache",
                         .endpoint_type = DFBEndpointType::PRODUCER,
                     },
                     DFBBinding{
                         .dfb_spec_name = TF_UNTILIZED_CACHE2,
                         .accessor_name = "untilized_cache2",
                         .endpoint_type = DFBEndpointType::CONSUMER,
                     },
                     DFBBinding{
                         .dfb_spec_name = TF_UNTILIZED_INPUT,
                         .accessor_name = "untilized_in",
                         .endpoint_type = DFBEndpointType::PRODUCER,
                     },
                     DFBBinding{
                         .dfb_spec_name = TF_OUT_TILES,
                         .accessor_name = "out",
                         .endpoint_type = DFBEndpointType::PRODUCER,
                     },
                 }) {
                compute_dfb_bindings.push_back(b);
            }
        }

        kernels.push_back(KernelSpec{
            .unique_id = compute_id,
            .source = "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/"
                      "paged_fused_update_cache.cpp",
            // Legacy compute kernels default to O3; Metal 2.0's type-agnostic default is O2, so it is
            // set explicitly here to keep the compile and link at the level the op has always used.
            .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings = std::move(compute_dfb_bindings),
            .compile_time_args =
                {
                    {"Wt", Wt},
                },
            .runtime_arg_schema = {.runtime_arg_names = {"has_work", "num_local_heads"}},
            .hw_config = compute_hw,
        });
    };

    add_kernel_triple(TF_READER1, TF_WRITER1, TF_COMPUTE1, TF_INPUT1, TF_CACHE1, std::nullopt);
    add_kernel_triple(TF_READER2, TF_WRITER2, TF_COMPUTE2, TF_INPUT2, TF_CACHE2, std::nullopt);
    if (head_split.groups > 1) {
        add_kernel_triple(
            TF_READER1_HELPER, TF_WRITER1_HELPER, TF_COMPUTE1_HELPER, TF_INPUT_COPY, TF_CACHE1, TF_INPUT1_T);
        add_kernel_triple(
            TF_READER2_HELPER, TF_WRITER2_HELPER, TF_COMPUTE2_HELPER, TF_INPUT_COPY, TF_CACHE2, TF_INPUT2_T);
    }

    // ---------------- Tensor parameters ----------------

    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = TF_CACHE1, .spec = cache_tensor1.tensor_spec()},
        TensorParameter{.unique_id = TF_CACHE2, .spec = cache_tensor2.tensor_spec()},
        // Each backs a borrowed input-shard buffer; the helper readers also bind them to copy a user's input.
        TensorParameter{.unique_id = TF_INPUT1_T, .spec = input_tensor1.tensor_spec()},
        TensorParameter{.unique_id = TF_INPUT2_T, .spec = input_tensor2.tensor_spec()},
    };
    if (use_index_tensor) {
        tensor_parameters.push_back(
            TensorParameter{.unique_id = TF_INDEX_T, .spec = update_idxs_tensor->tensor_spec()});
    }
    if (is_paged_cache) {
        tensor_parameters.push_back(TensorParameter{.unique_id = TF_PAGE_TABLE_T, .spec = page_table->tensor_spec()});
    }

    CoreRangeSet semaphore_cores = all_cores;
    if (head_split.groups > 1) {
        semaphore_cores = semaphore_cores.merge(cores_to_range_set(head_split.helpers1))
                              .merge(cores_to_range_set(head_split.helpers2));
    }

    ProgramSpec spec{
        .name = "paged_tiled_fused_update_cache",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        // Used for share cache, for signaling when the cache is ready to be read. Allocated whether or
        // not share_cache is set, matching the ported-from program.
        .semaphores =
            {
                SemaphoreSpec{
                    .unique_id = TF_IN0_SEQUENTIAL,
                    .target_nodes = semaphore_cores,
                },
            },
        .tensor_parameters = std::move(tensor_parameters),
        .work_units =
            {
                WorkUnitSpec{
                    .name = "input1",
                    .kernels = {TF_READER1, TF_WRITER1, TF_COMPUTE1},
                    .target_nodes = input1_cores,
                },
                WorkUnitSpec{
                    .name = "input2",
                    .kernels = {TF_READER2, TF_WRITER2, TF_COMPUTE2},
                    .target_nodes = input2_cores,
                },
            },
    };
    if (head_split.groups > 1) {
        spec.work_units.push_back(WorkUnitSpec{
            .name = "input1_helpers",
            .kernels = {TF_READER1_HELPER, TF_WRITER1_HELPER, TF_COMPUTE1_HELPER},
            .target_nodes = cores_to_range_set(head_split.helpers1),
        });
        spec.work_units.push_back(WorkUnitSpec{
            .name = "input2_helpers",
            .kernels = {TF_READER2_HELPER, TF_WRITER2_HELPER, TF_COMPUTE2_HELPER},
            .target_nodes = cores_to_range_set(head_split.helpers2),
        });
    }

    // ---------------- Run args ----------------

    const auto& cores1 = corerange_to_cores(input1_cores, input1_cores.num_cores(), row_major);
    const auto& cores2 = corerange_to_cores(input2_cores, input2_cores.num_cores(), row_major);

    const auto offsets = compute_tiled_fused_offsets(operation_attributes, tensor_args);

    KernelRunArgs reader1_run_args{.kernel = TF_READER1};
    KernelRunArgs reader2_run_args{.kernel = TF_READER2};
    KernelRunArgs writer1_run_args{.kernel = TF_WRITER1};
    KernelRunArgs writer2_run_args{.kernel = TF_WRITER2};
    KernelRunArgs compute1_run_args{.kernel = TF_COMPUTE1};
    KernelRunArgs compute2_run_args{.kernel = TF_COMPUTE2};
    KernelRunArgs reader1h_run_args{.kernel = TF_READER1_HELPER};
    KernelRunArgs reader2h_run_args{.kernel = TF_READER2_HELPER};
    KernelRunArgs writer1h_run_args{.kernel = TF_WRITER1_HELPER};
    KernelRunArgs writer2h_run_args{.kernel = TF_WRITER2_HELPER};
    KernelRunArgs compute1h_run_args{.kernel = TF_COMPUTE1_HELPER};
    KernelRunArgs compute2h_run_args{.kernel = TF_COMPUTE2_HELPER};

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

        // Every node a work unit covers does work: the cores inside the bounding box of the two shard
        // grids but in neither of them are outside both work units now, so the flag is constant here.
        // The kernels' early return on it is left in place.
        constexpr uint32_t has_work = 1;

        const auto add_index_args = [&](KernelRunArgs& reader_run_args,
                                        KernelRunArgs& writer_run_args,
                                        KernelRunArgs& compute_run_args,
                                        const CoreCoord& core,
                                        uint32_t send_core_x,
                                        uint32_t send_core_y,
                                        uint32_t group) {
            const uint32_t head_start = head_split.head_start(num_heads, group);
            const uint32_t num_local_heads = head_split.num_local_heads(num_heads, group);
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {
                    {"has_work", has_work},
                    {"cache_start_id", cache_start_id},
                    {"my_batch_idx", i},
                    {"wait_to_start", static_cast<uint32_t>(wait_to_start)},
                    {"head_start", head_start},
                    {"num_local_heads", num_local_heads},
                    {"cache_row", use_index_tensor ? 0u : operation_attributes.update_idxs.at(i) % TILE_HEIGHT},
                });
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {
                    {"has_work", has_work},
                    {"cache_start_id", cache_start_id},
                    {"cache_tile_offset_B", tile_update_offset_B},
                    {"my_batch_idx", i},
                    {"send_signal", static_cast<uint32_t>(send_signal)},
                    {"send_core_x", send_core_x},
                    {"send_core_y", send_core_y},
                    {"head_start", head_start},
                    {"num_local_heads", num_local_heads},
                });
            AddRuntimeArgsForNode(
                compute_run_args.runtime_arg_values, core, {{"has_work", has_work}, {"num_local_heads", num_local_heads}});
        };

        add_index_args(reader1_run_args, writer1_run_args, compute1_run_args, core1, send_core1_x, send_core1_y, 0);
        add_index_args(reader2_run_args, writer2_run_args, compute2_run_args, core2, send_core2_x, send_core2_y, 0);
        for (uint32_t g = 1; g < head_split.groups; ++g) {
            const uint32_t h = i * (head_split.groups - 1) + g - 1;
            add_index_args(reader1h_run_args, writer1h_run_args, compute1h_run_args, head_split.helpers1[h], 0, 0, g);
            add_index_args(reader2h_run_args, writer2h_run_args, compute2h_run_args, head_split.helpers2[h], 0, 0, g);
        }
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        std::move(reader1_run_args),
        std::move(reader2_run_args),
        std::move(writer1_run_args),
        std::move(writer2_run_args),
        std::move(compute1_run_args),
        std::move(compute2_run_args),
    };
    if (head_split.groups > 1) {
        run_args.kernel_run_args.push_back(std::move(reader1h_run_args));
        run_args.kernel_run_args.push_back(std::move(reader2h_run_args));
        run_args.kernel_run_args.push_back(std::move(writer1h_run_args));
        run_args.kernel_run_args.push_back(std::move(writer2h_run_args));
        run_args.kernel_run_args.push_back(std::move(compute1h_run_args));
        run_args.kernel_run_args.push_back(std::move(compute2h_run_args));
    }
    run_args.tensor_args = tiled_fused_tensor_args(tensor_args);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

namespace CMAKE_UNIQUE_NAMESPACE_TILED {

// Cache-hit re-derivation, shared by both tiled factories. Re-applies only per-dispatch state: every
// tensor binding (which also re-points the two borrowed input-shard buffers, and the index or
// page-table buffer when either is borrowed) and the update_idxs-derived offsets the program hash
// excludes. Every other arg is derived from hashed inputs (shard grids, share_cache, shapes, dtypes)
// and so is identical by construction on a cache hit.
ProgramRunArgs tiled_fused_run_args(
    const PagedFusedUpdateCacheParams& operation_attributes, const PagedFusedUpdateCacheInputs& tensor_args) {
    ProgramRunArgs params;
    params.tensor_args = tiled_fused_tensor_args(tensor_args);

    const auto offsets =
        PagedTiledFusedUpdateCacheProgramFactory::compute_tiled_fused_offsets(operation_attributes, tensor_args);

    const auto& input1_shard_spec = tensor_args.input_tensor1.shard_spec().value();
    const bool row_major = input1_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    const CoreRangeSet& input1_cores = input1_shard_spec.grid;
    const CoreRangeSet& input2_cores = tensor_args.input_tensor2.shard_spec().value().grid;
    const auto cores1 = corerange_to_cores(input1_cores, input1_cores.num_cores(), row_major);
    const auto cores2 = corerange_to_cores(input2_cores, input2_cores.num_cores(), row_major);

    KernelRunArgs reader1_run_args{.kernel = TF_READER1};
    KernelRunArgs reader2_run_args{.kernel = TF_READER2};
    KernelRunArgs writer1_run_args{.kernel = TF_WRITER1};
    KernelRunArgs writer2_run_args{.kernel = TF_WRITER2};
    KernelRunArgs reader1h_run_args{.kernel = TF_READER1_HELPER};
    KernelRunArgs reader2h_run_args{.kernel = TF_READER2_HELPER};
    KernelRunArgs writer1h_run_args{.kernel = TF_WRITER1_HELPER};
    KernelRunArgs writer2h_run_args{.kernel = TF_WRITER2_HELPER};
    const HeadSplit head_split = compute_head_split(operation_attributes, tensor_args);

    // Empty offsets == index-tensor mode, where the artifacts build bakes 0 and the kernels read the
    // real positions on-device; writing 0 back reproduces that build exactly.
    for (uint32_t i = 0; i < cores1.size(); ++i) {
        const uint32_t cache_start_id = offsets.empty() ? 0 : offsets[i].cache_start_id;
        const uint32_t tile_update_offset_B = offsets.empty() ? 0 : offsets[i].tile_update_offset_B;

        const auto patch_core =
            [&](KernelRunArgs& reader_run_args, KernelRunArgs& writer_run_args, const CoreCoord& core) {
                AddRuntimeArgsForNode(
                    reader_run_args.runtime_arg_values,
                    core,
                    {{"cache_start_id", cache_start_id},
                     {"cache_row", offsets.empty() ? 0u : operation_attributes.update_idxs.at(i) % TILE_HEIGHT}});
                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values,
                    core,
                    {
                        {"cache_start_id", cache_start_id},
                        {"cache_tile_offset_B", tile_update_offset_B},
                    });
            };
        patch_core(reader1_run_args, writer1_run_args, cores1[i]);
        patch_core(reader2_run_args, writer2_run_args, cores2[i]);
        for (uint32_t g = 1; g < head_split.groups; ++g) {
            const uint32_t h = i * (head_split.groups - 1) + g - 1;
            patch_core(reader1h_run_args, writer1h_run_args, head_split.helpers1[h]);
            patch_core(reader2h_run_args, writer2h_run_args, head_split.helpers2[h]);
        }
    }

    params.kernel_run_args = {
        std::move(reader1_run_args),
        std::move(reader2_run_args),
        std::move(writer1_run_args),
        std::move(writer2_run_args),
    };
    if (head_split.groups > 1) {
        params.kernel_run_args.push_back(std::move(reader1h_run_args));
        params.kernel_run_args.push_back(std::move(reader2h_run_args));
        params.kernel_run_args.push_back(std::move(writer1h_run_args));
        params.kernel_run_args.push_back(std::move(writer2h_run_args));
    }
    return params;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE_TILED

ProgramRunArgs PagedTiledFusedUpdateCacheProgramFactory::override_runtime_arguments(
    const PagedFusedUpdateCacheParams& operation_attributes,
    const PagedFusedUpdateCacheInputs& tensor_args,
    PagedFusedUpdateCacheResult& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    return CMAKE_UNIQUE_NAMESPACE_TILED::tiled_fused_run_args(operation_attributes, tensor_args);
}

ProgramRunArgs PagedTiledFusedUpdateCacheMeshWorkloadFactory::override_runtime_arguments(
    const PagedFusedUpdateCacheParams& operation_attributes,
    const PagedFusedUpdateCacheInputs& tensor_args,
    PagedFusedUpdateCacheResult& /*tensor_return_value*/,
    const ttnn::MeshCoordinateRange& /*range*/) {
    return CMAKE_UNIQUE_NAMESPACE_TILED::tiled_fused_run_args(operation_attributes, tensor_args);
}

ttnn::device_operation::MeshWorkloadArtifacts
PagedTiledFusedUpdateCacheMeshWorkloadFactory::create_mesh_workload_artifacts(
    const PagedFusedUpdateCacheParams& operation_attributes,
    const PagedFusedUpdateCacheInputs& tensor_args,
    PagedFusedUpdateCacheResult& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    // One program per coordinate, which is what the ported-from descriptor path built. A coordinate
    // outside mesh_coords contributes no program at all, the way that path returned an empty
    // ProgramDescriptor for it.
    ttnn::device_operation::MeshWorkloadArtifacts artifacts;
    for (const auto& coord : tensor_coords.coords()) {
        if (operation_attributes.mesh_coords.has_value() && !operation_attributes.mesh_coords->contains(coord)) {
            continue;
        }
        auto per_coord = PagedTiledFusedUpdateCacheProgramFactory::create_program_artifacts(
            operation_attributes, tensor_args, tensor_return_value);
        artifacts.programs.push_back({
            .range = ttnn::MeshCoordinateRange(coord),
            .spec = std::move(per_coord.spec),
            .run_params = std::move(per_coord.run_params),
        });
    }
    return artifacts;
}

}  // namespace ttnn::experimental::prim
