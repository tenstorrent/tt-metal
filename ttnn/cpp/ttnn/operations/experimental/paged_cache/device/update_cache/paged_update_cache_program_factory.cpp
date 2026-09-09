// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "paged_update_cache_program_factory.hpp"

#include "paged_update_cache_device_operation.hpp"
#include "paged_update_cache_device_operation_types.hpp"

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

namespace {

// Spec names for the update_cache program. Prefixed because every program-factory .cpp in this op
// shares one unity-build translation unit, which merges their anonymous namespaces.
const KernelSpecName UC_READER{"reader"};
const KernelSpecName UC_WRITER{"writer"};
const KernelSpecName UC_COMPUTE{"compute"};

const DFBSpecName UC_CACHE_TILES{"cache_tiles"};
const DFBSpecName UC_INPUT_SHARD{"input_shard"};
const DFBSpecName UC_INDEX{"index"};
const DFBSpecName UC_PAGE_TABLE{"page_table"};
const DFBSpecName UC_UNTILIZED_CACHE{"untilized_cache"};
const DFBSpecName UC_UNTILIZED_CACHE2{"untilized_cache2"};
const DFBSpecName UC_UNTILIZED_INPUT{"untilized_input"};
const DFBSpecName UC_OUT_TILES{"out_tiles"};

const SemaphoreSpecName UC_IN0_SEQUENTIAL{"in0_sequential_mode"};

const TensorParamName UC_CACHE{"cache"};
const TensorParamName UC_INPUT{"input"};
const TensorParamName UC_INDEX_T{"index"};
const TensorParamName UC_PAGE_TABLE_T{"page_table"};

bool enable_fp32_dest(const tt_metal::IDevice* device, const ttnn::DeviceComputeKernelConfig& compute_kernel_config) {
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    return fp32_dest_acc_en;
}

// Worker cores in the exact order the artifacts build emplaces per-core runtime args (core i handles
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
// source of truth for the formulas — both the artifacts build (cache miss) and override_runtime_arguments
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
    // share_cache => batch offset is 0 (one shared cache buffer); mirror the artifacts build exactly.
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

// The tensor arguments every dispatch must (re)bind. Shared by the cache-miss artifacts build and the
// cache-hit override so the two cannot disagree about which optional tensors are present. `input` is
// here even though no kernel binds it: it backs the borrowed input-shard DFB, whose SRAM address
// resolves from this argument.
Table<TensorParamName, TensorArgument> paged_update_cache_tensor_args(const PagedUpdateCacheInputs& tensor_args) {
    Table<TensorParamName, TensorArgument> args;
    args.emplace(UC_CACHE, tensor_args.cache_tensor.mesh_tensor());
    args.emplace(UC_INPUT, tensor_args.input_tensor.mesh_tensor());
    if (tensor_args.update_idxs_tensor.has_value()) {
        args.emplace(UC_INDEX_T, tensor_args.update_idxs_tensor->mesh_tensor());
    }
    if (tensor_args.page_table.has_value()) {
        args.emplace(UC_PAGE_TABLE_T, tensor_args.page_table->mesh_tensor());
    }
    return args;
}

ttnn::device_operation::ProgramArtifacts build_paged_update_cache_artifacts(
    const PagedUpdateCacheParams& operation_attributes, const PagedUpdateCacheInputs& tensor_args) {
    const auto& cache_tensor = tensor_args.cache_tensor;
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& update_idxs_tensor = tensor_args.update_idxs_tensor;
    const auto& page_table = tensor_args.page_table;

    tt_metal::IDevice* device = input_tensor.device();

    tt::DataFormat cache_dfb_data_format = tt_metal::datatype_to_dataformat_converter(cache_tensor.dtype());
    uint32_t cache_single_tile_size = tt::tile_size(cache_dfb_data_format);

    tt::DataFormat input_dfb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_dfb_data_format);

    bool fp32_dest_acc_en = enable_fp32_dest(device, operation_attributes.compute_kernel_config);

    tt::DataFormat interm_dfb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t interm_single_tile_size = tt::tile_size(interm_dfb_data_format);

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

    log_debug(tt::LogOp, "cache_dfb_data_format: {}", cache_dfb_data_format);
    log_debug(tt::LogOp, "input_dfb_data_format: {}", input_dfb_data_format);
    log_debug(tt::LogOp, "interm_dfb_data_format: {}", interm_dfb_data_format);
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

    // ---------------- Dataflow buffers ----------------

    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = UC_CACHE_TILES,
            .entry_size = cache_single_tile_size,
            .num_entries = num_cache_tiles,
            .data_format_metadata = cache_dfb_data_format,
        },
        // Built on the input tensor's own shard rather than allocating: the reader reserves and
        // pushes without writing, and compute untilizes straight out of it, so the borrowed buffer
        // is the tensor access. Its SRAM address resolves per dispatch from the input tensor argument.
        DataflowBufferSpec{
            .unique_id = UC_INPUT_SHARD,
            .entry_size = input_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = input_dfb_data_format,
            .borrowed_from = UC_INPUT,
        },
        // These two buffers share one backing region, so each names the other. The alias group
        // must agree on total size and on the set of kernels binding it; both hold here, since
        // the writer and compute each bind both.
        DataflowBufferSpec{
            .unique_id = UC_UNTILIZED_CACHE,
            .entry_size = interm_single_tile_size,
            .num_entries = num_interm_tiles,
            .data_format_metadata = interm_dfb_data_format,
            .advanced_options = {.alias_with = {UC_UNTILIZED_CACHE2}},
        },
        DataflowBufferSpec{
            .unique_id = UC_UNTILIZED_CACHE2,
            .entry_size = interm_single_tile_size,
            .num_entries = num_interm_tiles,
            .data_format_metadata = interm_dfb_data_format,
            .advanced_options = {.alias_with = {UC_UNTILIZED_CACHE}},
        },
        DataflowBufferSpec{
            .unique_id = UC_UNTILIZED_INPUT,
            .entry_size = interm_single_tile_size,
            .num_entries = num_interm_tiles,
            .data_format_metadata = interm_dfb_data_format,
        },
        DataflowBufferSpec{
            .unique_id = UC_OUT_TILES,
            .entry_size = cache_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = cache_dfb_data_format,
        },
    };
    if (use_index_tensor) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = UC_INDEX,
            .entry_size = index_tensor_tile_size,
            .num_entries = 1,
            .data_format_metadata = index_data_format,
        });
    }
    if (is_paged_cache) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = UC_PAGE_TABLE,
            .entry_size = page_table_stick_size,
            .num_entries = 1,
            .data_format_metadata = page_table_data_format,
        });
    }

    // The two optional buffers and the tensors behind them exist only under their host flag, and the
    // kernels name them from blocks the preprocessor has to remove on the off-path: a discarded
    // `if constexpr` branch still resolves names, so an unbound dfb:: or tensor:: token would fail to
    // compile. Both the reader and the writer read each flag, so both get both defines.
    KernelSpec::CompilerOptions::Defines optional_resource_defines;
    if (use_index_tensor) {
        optional_resource_defines.emplace("USE_INDEX_TENSOR", "1");
    }
    if (is_paged_cache) {
        optional_resource_defines.emplace("IS_PAGED_CACHE", "1");
    }

    // ---------------- Reader ----------------

    Group<DFBBinding> reader_dfb_bindings = {
        DFBBinding{
            .dfb_spec_name = UC_CACHE_TILES,
            .accessor_name = "cache",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = UC_INPUT_SHARD,
            .accessor_name = "input",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
    };
    Group<TensorBinding> reader_tensor_bindings = {
        TensorBinding{
            .tensor_parameter_name = UC_CACHE,
            .accessor_name = "cache",
        },
    };
    if (use_index_tensor) {
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UC_INDEX,
            .accessor_name = "index",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = UC_INDEX_T,
            .accessor_name = "index",
        });
    }
    if (is_paged_cache) {
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UC_PAGE_TABLE,
            .accessor_name = "page_table",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = UC_PAGE_TABLE_T,
            .accessor_name = "page_table",
        });
    }

    KernelSpec reader{
        .unique_id = UC_READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
            "reader_update_cache_interleaved_start_id.cpp",
        .compiler_options = {.defines = optional_resource_defines},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .semaphore_bindings =
            {
                SemaphoreBinding{
                    .semaphore_spec_name = UC_IN0_SEQUENTIAL,
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
                {"cache_position_modulo", cache_position_modulo},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"cache_start_id", "my_batch_idx", "wait_to_start"}},
        .hw_config = create_reader_datamovement_config(device->arch()),
    };

    // ---------------- Writer ----------------

    // The writer's `cache` handle names the *output* buffer, the one compute fills with the
    // retilized block. The reader's handle of the same name is the cache buffer it reads from DRAM.
    // Accessor names are per-kernel, and this pair of bindings is where the two are told apart.
    Group<DFBBinding> writer_dfb_bindings = {
        DFBBinding{
            .dfb_spec_name = UC_OUT_TILES,
            .accessor_name = "cache",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = UC_UNTILIZED_CACHE,
            .accessor_name = "untilized_cache",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = UC_UNTILIZED_CACHE2,
            .accessor_name = "untilized_cache2",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = UC_UNTILIZED_INPUT,
            .accessor_name = "untilized_input",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
    };
    if (use_index_tensor) {
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UC_INDEX,
            .accessor_name = "index",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }
    if (is_paged_cache) {
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UC_PAGE_TABLE,
            .accessor_name = "page_table",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    KernelSpec writer{
        .unique_id = UC_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
            "writer_update_cache_interleaved_start_id.cpp",
        .compiler_options = {.defines = optional_resource_defines},
        .dfb_bindings = std::move(writer_dfb_bindings),
        .semaphore_bindings =
            {
                SemaphoreBinding{
                    .semaphore_spec_name = UC_IN0_SEQUENTIAL,
                    .accessor_name = "in0_seq",
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = UC_CACHE,
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
        .hw_config = create_writer_datamovement_config(device->arch()),
    };

    // ---------------- Compute ----------------

    // Legacy built a ComputeConfigDescriptor that set only fp32_dest_acc_en, leaving every other knob
    // at its default even when the caller's compute_kernel_config specified one. ComputeGen1Config's
    // defaults coincide with those, so setting only enable_32_bit_dest reproduces it exactly.
    ComputeGen1Config compute_hw{.enable_32_bit_dest = fp32_dest_acc_en};
    if (fp32_dest_acc_en) {
        // A 32-bit Dest requires an explicit unpack mode for every Float32 buffer the compute kernel
        // consumes. Legacy named none, which resolved to unpacking into SrcA/B.
        const auto require_unpack_mode = [&](const DFBSpecName& dfb, tt::DataFormat format) {
            if (format == tt::DataFormat::Float32) {
                compute_hw.unpack_modes.emplace(dfb, UnpackMode::UnpackToSrc);
            }
        };
        require_unpack_mode(UC_CACHE_TILES, cache_dfb_data_format);
        require_unpack_mode(UC_INPUT_SHARD, input_dfb_data_format);
        require_unpack_mode(UC_UNTILIZED_CACHE2, interm_dfb_data_format);
    }

    KernelSpec compute{
        .unique_id = UC_COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/update_cache.cpp",
        // Legacy compute kernels default to O3; Metal 2.0's type-agnostic default is O2, so it is set
        // explicitly here to keep the compile and link at the level the op has always used.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = UC_CACHE_TILES,
                    .accessor_name = "cache",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = UC_INPUT_SHARD,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = UC_UNTILIZED_CACHE,
                    .accessor_name = "untilized_cache",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = UC_UNTILIZED_CACHE2,
                    .accessor_name = "untilized_cache2",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = UC_UNTILIZED_INPUT,
                    .accessor_name = "untilized_in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = UC_OUT_TILES,
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

    // ---------------- Tensor parameters ----------------

    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = UC_CACHE, .spec = cache_tensor.tensor_spec()},
        // Bound by no kernel: it backs the borrowed input-shard buffer, which is what uses it.
        TensorParameter{.unique_id = UC_INPUT, .spec = input_tensor.tensor_spec()},
    };
    if (use_index_tensor) {
        tensor_parameters.push_back(
            TensorParameter{.unique_id = UC_INDEX_T, .spec = update_idxs_tensor->tensor_spec()});
    }
    if (is_paged_cache) {
        tensor_parameters.push_back(TensorParameter{.unique_id = UC_PAGE_TABLE_T, .spec = page_table->tensor_spec()});
    }

    ProgramSpec spec{
        .name = "paged_update_cache",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dataflow_buffers),
        // Used for share cache, for signaling when the cache is ready to be read. Allocated whether or
        // not share_cache is set, matching the ported-from program.
        .semaphores =
            {
                SemaphoreSpec{
                    .unique_id = UC_IN0_SEQUENTIAL,
                    .target_nodes = all_cores,
                },
            },
        .tensor_parameters = std::move(tensor_parameters),
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {UC_READER, UC_WRITER, UC_COMPUTE},
                    .target_nodes = all_cores,
                },
            },
    };

    // ---------------- Run args ----------------

    const auto cores = update_cache_cores(tensor_args);
    // cache_start_id / tile_update_offset_B are derived from update_idxs (excluded from the program
    // hash) — computed via the shared helper so override_runtime_arguments re-patches identical values on
    // cache hits. Empty in index-tensor mode (offsets read on-device from the re-patched index tensor).
    const auto offsets = compute_update_cache_offsets(operation_attributes, tensor_args, cores);

    KernelRunArgs reader_run_args{.kernel = UC_READER};
    KernelRunArgs writer_run_args{.kernel = UC_WRITER};

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

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = paged_update_cache_tensor_args(tensor_args);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

// Cache-hit re-derivation, shared by both factories. Re-applies only per-dispatch state: every tensor
// binding (which also re-points the borrowed input-shard buffer) and the update_idxs-derived offsets
// the program hash excludes. Everything else is a function of hashed inputs (shapes / dtypes / memory
// configs / share_cache / overrides) and is identical by construction on a hit.
ProgramRunArgs paged_update_cache_run_args(
    const PagedUpdateCacheParams& operation_attributes, const PagedUpdateCacheInputs& tensor_args) {
    ProgramRunArgs params;
    params.tensor_args = paged_update_cache_tensor_args(tensor_args);

    const auto cores = update_cache_cores(tensor_args);
    // Empty in index-tensor mode: the kernels read positions on-device from the index tensor, so the two
    // offset slots keep the zeroes the artifacts build set.
    const auto offsets = compute_update_cache_offsets(operation_attributes, tensor_args, cores);
    if (offsets.empty()) {
        return params;
    }

    KernelRunArgs reader_run_args{.kernel = UC_READER};
    KernelRunArgs writer_run_args{.kernel = UC_WRITER};
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
    params.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    return params;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts PagedUpdateCacheProgramFactory::create_program_artifacts(
    const PagedUpdateCacheParams& operation_attributes,
    const PagedUpdateCacheInputs& tensor_args,
    Tensor& /*tensor_return_value*/) {
    return build_paged_update_cache_artifacts(operation_attributes, tensor_args);
}

ProgramRunArgs PagedUpdateCacheProgramFactory::override_runtime_arguments(
    const PagedUpdateCacheParams& operation_attributes,
    const PagedUpdateCacheInputs& tensor_args,
    Tensor& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    return paged_update_cache_run_args(operation_attributes, tensor_args);
}

ttnn::device_operation::MeshWorkloadArtifacts PagedUpdateCacheMeshWorkloadFactory::create_mesh_workload_artifacts(
    const PagedUpdateCacheParams& operation_attributes,
    const PagedUpdateCacheInputs& tensor_args,
    Tensor& /*tensor_return_value*/,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    // One program per coordinate, which is what the ported-from descriptor path built. A coordinate
    // outside mesh_coords contributes no program at all, the way that path returned an empty
    // ProgramDescriptor for it.
    ttnn::device_operation::MeshWorkloadArtifacts artifacts;
    for (const auto& coord : tensor_coords.coords()) {
        if (operation_attributes.mesh_coords.has_value() && !operation_attributes.mesh_coords->contains(coord)) {
            continue;
        }
        auto per_coord = build_paged_update_cache_artifacts(operation_attributes, tensor_args);
        artifacts.programs.push_back({
            .range = ttnn::MeshCoordinateRange(coord),
            .spec = std::move(per_coord.spec),
            .run_params = std::move(per_coord.run_params),
        });
    }
    return artifacts;
}

ProgramRunArgs PagedUpdateCacheMeshWorkloadFactory::override_runtime_arguments(
    const PagedUpdateCacheParams& operation_attributes,
    const PagedUpdateCacheInputs& tensor_args,
    Tensor& /*tensor_return_value*/,
    const ttnn::MeshCoordinateRange& /*range*/) {
    return paged_update_cache_run_args(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental::prim
