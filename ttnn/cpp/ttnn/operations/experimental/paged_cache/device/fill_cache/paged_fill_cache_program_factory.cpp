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

// Spec names for the fill_cache program. Prefixed because every program-factory .cpp in this op
// shares one unity-build translation unit, which merges their anonymous namespaces.
const KernelSpecName FC_READER{"reader"};
const KernelSpecName FC_WRITER{"writer"};

const DFBSpecName FC_IN_TILES{"in_tiles"};
const DFBSpecName FC_PAGE_TABLE{"page_table"};
const DFBSpecName FC_BATCH_IDX{"batch_idx"};
const DFBSpecName FC_VALID_SEQ_LEN{"valid_seq_len"};

const TensorParamName FC_INPUT{"input"};
const TensorParamName FC_CACHE{"cache"};
const TensorParamName FC_PAGE_TABLE_T{"page_table"};
const TensorParamName FC_BATCH_IDX_T{"batch_idx"};
const TensorParamName FC_VALID_SEQ_LEN_T{"valid_seq_len"};

// `noop` is the only thing that differs between the single-device and the mesh-workload factory: a
// mesh coordinate outside operation_attributes.mesh_coords gets a noop program (kernels early-exit).
// Single source of truth for that choice, called by both artifact-building entry points and by
// override_runtime_arguments — so the cache-hit patch mirrors select_program_factory by construction
// (mesh_coords is nullopt on the single-device path, where the coordinate is ignored).
bool paged_fill_cache_noop(
    const PagedFillCacheParams& operation_attributes, const std::optional<ttnn::MeshCoordinate>& coord) {
    if (operation_attributes.mesh_coords.has_value() && coord.has_value() &&
        !operation_attributes.mesh_coords->contains(coord.value())) {
        return true;
    }
    return operation_attributes.noop;
}

// Worker-core list for the fill_cache work-split. Single source of truth for core ordering: called by
// both build_paged_fill_cache_artifacts (cache miss, emitting per-core runtime args) and
// PagedFillCacheProgramFactory::override_runtime_arguments (cache hit, patching them), so the two
// paths cannot drift in which cores they touch or in what order.
std::vector<tt_metal::CoreCoord> compute_paged_fill_cache_cores(
    const PagedFillCacheParams& /*operation_attributes*/, const PagedFillCacheInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    // num_blocks_of_work mirrors build_paged_fill_cache_artifacts: input_batch * num_heads *
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

// The tensor arguments every dispatch must (re)bind. Shared by the cache-miss artifact build and the
// cache-hit override so the two cannot disagree about which optional tensors are present.
Table<TensorParamName, TensorArgument> paged_fill_cache_tensor_args(const PagedFillCacheInputs& tensor_args) {
    Table<TensorParamName, TensorArgument> args;
    args.emplace(FC_INPUT, tensor_args.input_tensor.mesh_tensor());
    args.emplace(FC_CACHE, tensor_args.cache_tensor.mesh_tensor());
    args.emplace(FC_PAGE_TABLE_T, tensor_args.page_table.mesh_tensor());
    if (tensor_args.batch_idx_tensor_opt.has_value()) {
        args.emplace(FC_BATCH_IDX_T, tensor_args.batch_idx_tensor_opt->mesh_tensor());
    }
    if (tensor_args.valid_seq_len_tensor_opt.has_value()) {
        args.emplace(FC_VALID_SEQ_LEN_T, tensor_args.valid_seq_len_tensor_opt->mesh_tensor());
    }
    return args;
}

// Build the per-coord artifacts. Shared by the single-device and mesh-workload
// factories; the mesh path passes a possibly-overridden `noop` (true for
// coordinates excluded from operation_attributes.mesh_coords).
ttnn::device_operation::ProgramArtifacts build_paged_fill_cache_artifacts(
    const PagedFillCacheParams& operation_attributes, const PagedFillCacheInputs& tensor_args, bool noop) {
    const auto& cache_tensor = tensor_args.cache_tensor;
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& page_table_tensor = tensor_args.page_table;
    const auto& batch_idx_tensor = tensor_args.batch_idx_tensor_opt;
    const auto& valid_seq_len_tensor = tensor_args.valid_seq_len_tensor_opt;

    tt::DataFormat dfb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(dfb_data_format);

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

    // ---------------- Dataflow buffers ----------------

    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = FC_IN_TILES,
            .entry_size = single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = dfb_data_format,
        },
        DataflowBufferSpec{
            .unique_id = FC_PAGE_TABLE,
            .entry_size = page_table_stick_size_B,
            .num_entries = 1,
            .data_format_metadata = page_table_data_format,
        },
    };
    if (use_batch_idx_tensor) {
        // Holds all `batch_idx_num_elements` entries so the writer kernel can pick
        // the right entry per batch row in the batched case.
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = FC_BATCH_IDX,
            .entry_size = batch_idx_stick_size_B,
            .num_entries = batch_idx_num_elements,
            .data_format_metadata = batch_idx_data_format,
        });
    }
    if (use_valid_seq_len) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = FC_VALID_SEQ_LEN,
            .entry_size = valid_seq_len_stick_size_B,
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::UInt32,
        });
    }

    // ---------------- Reader ----------------

    KernelSpec reader{
        .unique_id = FC_READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
            "reader_fill_cache_interleaved.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = FC_IN_TILES,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = FC_INPUT,
                    .accessor_name = "input",
                },
            },
        .compile_time_args = {{"Wt", Wt}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_tile_id", "num_rows", "noop"}},
        .hw_config = create_reader_datamovement_config(device->arch()),
    };

    // ---------------- Writer ----------------

    // The three metadata buffers below are touched by the writer alone: it reserves an entry, takes
    // the write pointer, NoC-reads the metadata into it and reads it straight back through an SRAM
    // pointer, never pushing. One toucher cannot present a producer and a consumer on distinct
    // kernels, so the writer binds both endpoints of each.
    Group<DFBBinding> writer_dfb_bindings = {
        DFBBinding{
            .dfb_spec_name = FC_IN_TILES,
            .accessor_name = "in",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = FC_PAGE_TABLE,
            .accessor_name = "page_table",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = FC_PAGE_TABLE,
            .accessor_name = "page_table",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
    };
    Group<TensorBinding> writer_tensor_bindings = {
        TensorBinding{
            .tensor_parameter_name = FC_CACHE,
            .accessor_name = "cache",
        },
        TensorBinding{
            .tensor_parameter_name = FC_PAGE_TABLE_T,
            .accessor_name = "page_table",
        },
    };
    KernelSpec::CompileTimeArgs writer_compile_time_args = {
        {"num_heads", num_heads},
        {"num_blocks_of_work_per_head", num_blocks_of_work_per_head},
        {"block_size_t", block_size_t},
        {"Wt", Wt},
        {"log2_page_table_stick_size", log2_page_table_stick_size_B},
        {"page_table_stick_size", page_table_stick_size_B},
        // 1 = legacy single-batch, N = batched. Read outside the batch-idx-tensor gate (it decides
        // whether row_id has to be decoded into a batch), so it is emitted in both configurations.
        {"batch_idx_num_elements", batch_idx_num_elements},
        {"num_blocks_per_batch", num_blocks_of_work_per_batch},
        {"capacity_t", capacity_t},
        // Per-element sizes of the two optional metadata tensors. The kernel declares both at
        // function scope, outside the gates gating their use, so both are always emitted; each
        // carries a placeholder when its tensor is absent.
        {"batch_idx_stick_size", batch_idx_stick_size_B},
        {"valid_seq_len_stick_size", valid_seq_len_stick_size_B},
    };
    KernelSpec::CompilerOptions::Defines writer_defines;
    Group<std::string> writer_runtime_arg_names = {"start_row_num", "num_rows", "noop"};

    if (use_batch_idx_tensor) {
        writer_defines.emplace("USE_BATCH_IDX_TENSOR", "1");
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = FC_BATCH_IDX,
            .accessor_name = "batch_idx",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = FC_BATCH_IDX,
            .accessor_name = "batch_idx",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        writer_tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = FC_BATCH_IDX_T,
            .accessor_name = "batch_idx",
        });
    } else {
        // Scalar fallback: the batch row to fill. Hash-excluded, so it is re-applied on every
        // cache hit rather than frozen at the first miss value.
        writer_runtime_arg_names.push_back("batch_idx_fallback");
    }

    if (use_valid_seq_len) {
        writer_defines.emplace("USE_VALID_SEQ_LEN", "1");
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = FC_VALID_SEQ_LEN,
            .accessor_name = "valid_seq_len",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = FC_VALID_SEQ_LEN,
            .accessor_name = "valid_seq_len",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        writer_tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = FC_VALID_SEQ_LEN_T,
            .accessor_name = "valid_seq_len",
        });
    }

    KernelSpec writer{
        .unique_id = FC_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
            "writer_fill_cache_interleaved.cpp",
        .compiler_options = {.defines = std::move(writer_defines)},
        .dfb_bindings = std::move(writer_dfb_bindings),
        .tensor_bindings = std::move(writer_tensor_bindings),
        .compile_time_args = std::move(writer_compile_time_args),
        .runtime_arg_schema = {.runtime_arg_names = std::move(writer_runtime_arg_names)},
        .hw_config = create_writer_datamovement_config(device->arch()),
    };

    // ---------------- Tensor parameters ----------------

    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = FC_INPUT, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = FC_CACHE, .spec = cache_tensor.tensor_spec()},
        TensorParameter{.unique_id = FC_PAGE_TABLE_T, .spec = page_table_tensor.tensor_spec()},
    };
    if (use_batch_idx_tensor) {
        tensor_parameters.push_back(
            TensorParameter{.unique_id = FC_BATCH_IDX_T, .spec = batch_idx_tensor->tensor_spec()});
    }
    if (use_valid_seq_len) {
        tensor_parameters.push_back(
            TensorParameter{.unique_id = FC_VALID_SEQ_LEN_T, .spec = valid_seq_len_tensor->tensor_spec()});
    }

    ProgramSpec spec{
        .name = "paged_fill_cache",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {FC_READER, FC_WRITER},
                    .target_nodes = all_cores,
                },
            },
    };

    // ---------------- Run args ----------------

    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();

    // Core list shared with override_runtime_arguments (single source of truth for ordering).
    const auto cores = compute_paged_fill_cache_cores(operation_attributes, tensor_args);

    KernelRunArgs reader_run_args{.kernel = FC_READER};
    KernelRunArgs writer_run_args{.kernel = FC_WRITER};

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

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = paged_fill_cache_tensor_args(tensor_args);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

// Cache-hit re-derivation, shared by both factories. Re-applies every tensor binding plus the values
// the program hash excludes — batch_idx_fallback and noop — which would otherwise freeze at the
// cache-miss value. Not re-applied: start_tile_id / start_row_num / num_rows. They come from the work
// split over the input's padded shape and the device grid, both of which the program hash includes, so
// a cache hit has them identical by construction.
ProgramRunArgs paged_fill_cache_run_args_for_coord(
    const PagedFillCacheParams& operation_attributes,
    const PagedFillCacheInputs& tensor_args,
    const std::optional<ttnn::MeshCoordinate>& coord) {
    ProgramRunArgs params;
    params.tensor_args = paged_fill_cache_tensor_args(tensor_args);

    // noop is hash-excluded, and on the mesh path depends on the dispatch coordinate.
    const auto noop_arg = static_cast<uint32_t>(paged_fill_cache_noop(operation_attributes, coord));
    const bool use_batch_idx_tensor = tensor_args.batch_idx_tensor_opt.has_value();

    KernelRunArgs reader_run_args{.kernel = FC_READER};
    KernelRunArgs writer_run_args{.kernel = FC_WRITER};

    const auto cores = compute_paged_fill_cache_cores(operation_attributes, tensor_args);
    for (const auto& core : cores) {
        AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"noop", noop_arg}});
        AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"noop", noop_arg}});
        if (!use_batch_idx_tensor) {
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"batch_idx_fallback", operation_attributes.batch_idx_fallback}});
        }
    }

    params.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    return params;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts PagedFillCacheProgramFactory::create_program_artifacts(
    const PagedFillCacheParams& operation_attributes,
    const PagedFillCacheInputs& tensor_args,
    Tensor& /*tensor_return_value*/) {
    return build_paged_fill_cache_artifacts(
        operation_attributes, tensor_args, paged_fill_cache_noop(operation_attributes, std::nullopt));
}

ProgramRunArgs PagedFillCacheProgramFactory::override_runtime_arguments(
    const PagedFillCacheParams& operation_attributes,
    const PagedFillCacheInputs& tensor_args,
    Tensor& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    return paged_fill_cache_run_args_for_coord(operation_attributes, tensor_args, mesh_dispatch_coordinate);
}

ttnn::device_operation::MeshWorkloadArtifacts PagedFillCacheMeshWorkloadFactory::create_mesh_workload_artifacts(
    const PagedFillCacheParams& operation_attributes,
    const PagedFillCacheInputs& tensor_args,
    Tensor& /*tensor_return_value*/,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    // One program per coordinate, which is what the ported-from descriptor path built (the descriptor
    // adapter iterates tensor_coords.coords() for a create_descriptor that takes a coordinate). Every
    // coordinate gets a program; coordinates outside mesh_coords get one whose kernels early-exit on
    // `noop`, so the cache slot is still populated for them.
    ttnn::device_operation::MeshWorkloadArtifacts artifacts;
    for (const auto& coord : tensor_coords.coords()) {
        auto per_coord = build_paged_fill_cache_artifacts(
            operation_attributes,
            tensor_args,
            paged_fill_cache_noop(operation_attributes, std::optional<ttnn::MeshCoordinate>(coord)));
        artifacts.programs.push_back({
            .range = ttnn::MeshCoordinateRange(coord),
            .spec = std::move(per_coord.spec),
            .run_params = std::move(per_coord.run_params),
        });
    }
    return artifacts;
}

ProgramRunArgs PagedFillCacheMeshWorkloadFactory::override_runtime_arguments(
    const PagedFillCacheParams& operation_attributes,
    const PagedFillCacheInputs& tensor_args,
    Tensor& /*tensor_return_value*/,
    const ttnn::MeshCoordinateRange& range) {
    // create_mesh_workload_artifacts emits one program per coordinate, so every range is a single
    // coordinate and its start coordinate is the whole of it.
    return paged_fill_cache_run_args_for_coord(
        operation_attributes, tensor_args, std::optional<ttnn::MeshCoordinate>(range.start_coord()));
}

}  // namespace ttnn::experimental::prim
