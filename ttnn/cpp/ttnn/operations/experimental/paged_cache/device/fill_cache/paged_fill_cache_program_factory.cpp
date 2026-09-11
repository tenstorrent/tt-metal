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
    "reader_fill_cache_interleaved.cpp";
constexpr auto FC_WRITER_SOURCE =
    "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
    "writer_fill_cache_interleaved.cpp";

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

// Overwrite the per-coordinate `noop` runtime arg on both dataflow kernels.
//
// `noop` is the ONLY thing that varies across the mesh for this op -- the spec and every other run-arg
// value are coordinate-independent -- so the mesh build produces one base ProgramRunArgs and patches
// this single named value per coordinate.  AddRuntimeArgsForNode assigns, so this overwrites rather
// than appends.
void apply_paged_fill_cache_noop(
    tt::tt_metal::experimental::ProgramRunArgs& run_args,
    const std::vector<tt_metal::CoreCoord>& cores,
    uint32_t noop_arg) {
    for (auto& kernel_run_args : run_args.kernel_run_args) {
        if (kernel_run_args.kernel != FC_READER_KERNEL && kernel_run_args.kernel != FC_WRITER_KERNEL) {
            continue;
        }
        for (const auto& core : cores) {
            AddRuntimeArgsForNode(kernel_run_args.runtime_arg_values, core, {{"noop", noop_arg}});
        }
    }
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

// ---------------------------------------------------------------------------------------------
// Metal 2.0 mesh-workload build (MeshWorkloadSpecFactoryConcept).
//
// Unlike the op's three sibling mesh factories, this one emits a program on EVERY coordinate.  The
// ported-from factory expressed its mesh filter with a `noop` runtime arg whose kernels early-exit
// rather than with an empty descriptor, so an excluded coordinate still receives a program -- and
// that matters: the cache slot is still populated for it.  Preserve that.
//
// What kept this factory off the single-program spec concepts was therefore never per-coordinate
// *programs* -- its spec is identical everywhere -- but per-coordinate run args on the cache MISS.
// The single-program adapter applies one ProgramRunArgs to every coordinate, so the first dispatch
// would have used one `noop` for the whole mesh and filled the cache on a coordinate the caller
// excluded.  (Its cache-HIT path was already correct, since override_runtime_arguments receives the
// coordinate.)
// ---------------------------------------------------------------------------------------------
ttnn::device_operation::MeshWorkloadArtifacts PagedFillCacheMeshWorkloadFactory::create_mesh_workload_artifacts(
    const PagedFillCacheParams& operation_attributes,
    const PagedFillCacheInputs& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    // One base build, then one copy per coordinate with `noop` patched.  The base is built with the
    // single-device factory, whose own `noop` resolves through paged_fill_cache_noop(attrs, nullopt);
    // every program below overwrites that value, so the base's choice never reaches a device.
    auto artifacts =
        PagedFillCacheProgramFactory::create_program_artifacts(operation_attributes, tensor_args, tensor_return_value);
    const auto cores = compute_paged_fill_cache_cores(operation_attributes, tensor_args);

    // One program per coordinate, which is what the ported-from path built (its create_descriptor
    // took a mesh_dispatch_coordinate, and the descriptor adapter iterates tensor_coords.coords() for
    // that shape).  It also makes each range trivially uniform in `noop`, so no range decomposition
    // is needed -- a coarser range would have to be split wherever mesh_coords membership changes
    // inside it, since one Program backs a whole range.
    ttnn::device_operation::MeshWorkloadArtifacts workload;
    const auto coords = tensor_coords.coords();
    workload.programs.reserve(coords.size());
    for (const auto& coord : coords) {
        auto run_params = artifacts.run_params;
        apply_paged_fill_cache_noop(
            run_params,
            cores,
            static_cast<uint32_t>(
                paged_fill_cache_noop(operation_attributes, std::optional<ttnn::MeshCoordinate>(coord))));
        workload.programs.push_back({
            .range = ttnn::MeshCoordinateRange(coord),
            .spec = artifacts.spec,
            .run_params = std::move(run_params),
        });
    }
    return workload;
}

tt::tt_metal::experimental::ProgramRunArgs PagedFillCacheMeshWorkloadFactory::override_runtime_arguments(
    const PagedFillCacheParams& operation_attributes,
    const PagedFillCacheInputs& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRange& coordinate_range) {
    // Every range this factory emits covers exactly one coordinate (see above), so the range's start
    // coordinate IS the coordinate and the `noop` derived from it is exact, not a representative of
    // several devices sharing a Program.  That is what lets this hand straight off to the
    // single-device refresh, which already computes noop from the coordinate it is given.
    return PagedFillCacheProgramFactory::override_runtime_arguments(
        operation_attributes,
        tensor_args,
        tensor_return_value,
        std::optional<ttnn::MeshCoordinate>(coordinate_range.start_coord()));
}

}  // namespace ttnn::experimental::prim
