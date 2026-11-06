// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified (placement-first) matmul factory (GH#41910): one Metal 2.0 program for every placement.
// Kernels address tiles by tile index through the tensor accessor, so operand layout doesn't matter here.

#include "ttnn/operations/experimental/quasar/matmul/device/factory/matmul_unified_program_factory.hpp"

#include <array>
#include <filesystem>
#include <map>
#include <string>

#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/experimental/quasar/matmul/device/config/matmul_unified_plan.hpp"

using namespace tt;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// Names the kernels see: dfb::A_slice / B_slice / C_slice / C_partials and tensor::A / B / C.
const DFBSpecName A_SLICE_DFB{"A_slice"};
const DFBSpecName B_SLICE_DFB{"B_slice"};
const DFBSpecName C_SLICE_DFB{"C_slice"};
const DFBSpecName C_PARTIALS_DFB{"C_partials"};
const DFBSpecName BIAS_DFB{"bias"};

// Multicast handshake semaphores. Each reader thread t has its own set, "<name>_<t>"; all
// QUASAR_NUM_READER_THREADS sets are bound on every arch because the reader names them all, which takes the
// whole per-core semaphore budget (16).
constexpr std::array<const char*, 4> MCAST_SEMAPHORES = {
    "A_receivers_ready", "A_data_ready", "B_receivers_ready", "B_data_ready"};

const TensorParamName A_TENSOR{"A"};
const TensorParamName B_TENSOR{"B"};
const TensorParamName C_TENSOR{"C"};
const TensorParamName BIAS_TENSOR{"bias"};

const KernelSpecName READER_KERNEL{"reader"};
const KernelSpecName COMPUTE_KERNEL{"compute"};
const KernelSpecName WRITER_KERNEL{"writer"};

constexpr const char* KERNEL_DIR = "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/";

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts MatmulUnifiedProgramFactory::create_program_artifacts(
    const MatmulParams& operation_attributes,
    const MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const operations::experimental::quasar::matmul::MatmulUnifiedProgramConfig& config =
        std::get<operations::experimental::quasar::matmul::MatmulUnifiedProgramConfig>(
            operation_attributes.program_config.value());

    const ttnn::Tensor& A_tensor = tensor_args.input_tensors.at(0);
    const ttnn::Tensor& B_tensor = tensor_args.input_tensors.at(1);
    const tt::tt_metal::MeshTensor& A = A_tensor.mesh_tensor();
    const tt::tt_metal::MeshTensor& B = B_tensor.mesh_tensor();
    const tt::tt_metal::MeshTensor& C = tensor_return_value.at(0).mesh_tensor();
    const std::optional<const ttnn::Tensor>& bias_tensor = tensor_args.optional_input_tensors.at(0);
    const tt::tt_metal::distributed::MeshDevice& device = A.device();

    const tt::DataFormat bias_format = bias_tensor.has_value()
                                           ? tt::tt_metal::datatype_to_dataformat_converter(bias_tensor->dtype())
                                           : tt::DataFormat{};
    const MatmulUnifiedPlan plan = plan_unified_matmul(
                                       A_tensor,
                                       B_tensor,
                                       bias_tensor.has_value() ? tt::tile_size(bias_format) : 0,
                                       config,
                                       operation_attributes,
                                       tensor_return_value.at(0),
                                       /*must_fit=*/true)
                                       .value();

    // ---- Tensor parameters: the kernels' tensor accessors are generated from these specs ----
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = A_TENSOR, .spec = A.tensor_spec()},
        TensorParameter{.unique_id = B_TENSOR, .spec = B.tensor_spec()},
        TensorParameter{.unique_id = C_TENSOR, .spec = C.tensor_spec()},
    };
    if (plan.has_bias) {
        tensor_parameters.push_back(
            TensorParameter{.unique_id = BIAS_TENSOR, .spec = bias_tensor->mesh_tensor().tensor_spec()});
    }

    // ---- Dataflow buffers ----
    const tt::tt_metal::Tile C_tile = C.tensor_spec().tile();
    log_debug(
        tt::LogOp,
        "MatmulUnifiedProgramConfig: {} cores in {} batch groups, borrow A={} B={} C={} (C slice {}x{}, subblock "
        "{}x{}, "
        "K chunk {} of {} tiles, {} compute threads, {} reader threads, {} writer threads, {} A / B slices and {} C "
        "slices in flight, A / B multicast to {} / {} cores)",
        plan.cores.size(),
        plan.num_batch_groups,
        plan.borrow_A,
        plan.borrow_B,
        plan.borrow_C,
        plan.C_slice_M_tiles,
        plan.C_slice_N_tiles,
        plan.subblock_M_tiles,
        plan.subblock_N_tiles,
        plan.K_chunk_tiles,
        plan.K_tiles,
        plan.num_compute_threads,
        plan.num_reader_threads,
        plan.num_writer_threads,
        plan.operand_buffer_depth,
        plan.C_buffer_depth,
        plan.A_mcast_num_dests,
        plan.B_mcast_num_dests);
    if (C.is_sharded() && !plan.borrow_C) {
        log_warning(
            tt::LogOp,
            "MatmulUnifiedProgramConfig: sharded C is copied by the writer instead of packed in place. Packing in "
            "place needs an L1 shard grid equal to the active cores, batch 1, one C slice per core, one compute "
            "thread ({} here) and subblock_N_tiles == C_slice_N_tiles (subblock {}x{} for a {}x{} C slice).",
            plan.num_compute_threads,
            plan.subblock_M_tiles,
            plan.subblock_N_tiles,
            plan.C_slice_M_tiles,
            plan.C_slice_N_tiles);
    }
    Group<DataflowBufferSpec> dataflow_buffers;
    {
        DataflowBufferSpec A_slice_dfb{
            .unique_id = A_SLICE_DFB,
            .entry_size = plan.A_entry_bytes,
            .num_entries = plan.A_slice_entries,
            .data_format_metadata = plan.A_format,
            .tile_format_metadata = A.tensor_spec().tile(),
        };
        if (plan.borrow_A) {
            A_slice_dfb.borrowed_from = A_TENSOR;  // the resident A shard is the DFB
        }
        DataflowBufferSpec B_slice_dfb{
            .unique_id = B_SLICE_DFB,
            .entry_size = plan.B_entry_bytes,
            .num_entries = plan.B_slice_entries,
            .data_format_metadata = plan.B_format,
            .tile_format_metadata = B.tensor_spec().tile(),
        };
        if (plan.borrow_B) {
            B_slice_dfb.borrowed_from = B_TENSOR;  // the resident B shard is the DFB
        }
        DataflowBufferSpec C_slice_dfb{
            .unique_id = C_SLICE_DFB,
            .entry_size = plan.C_entry_bytes,
            .num_entries = plan.C_slice_entries,
            .data_format_metadata = plan.C_format,
            .tile_format_metadata = C_tile,
        };
        if (plan.borrow_C) {
            C_slice_dfb.borrowed_from = C_TENSOR;  // finished tiles are packed straight into the C shard
        }
        DataflowBufferSpec C_partials_dfb{
            .unique_id = C_PARTIALS_DFB,
            .entry_size = plan.C_partials_entry_bytes,
            .num_entries = plan.C_partials_entries,
            .data_format_metadata = plan.C_partials_format,
            .tile_format_metadata = C_tile,
        };
        if (plan.alias_C_partials_onto_C_slice) {
            C_slice_dfb.advanced_options.alias_with = {C_PARTIALS_DFB};
            C_partials_dfb.advanced_options.alias_with = {C_SLICE_DFB};
            if (plan.borrow_C) {
                C_partials_dfb.borrowed_from = C_TENSOR;  // partials accumulate in the shard too
            }
        }
        dataflow_buffers.push_back(std::move(A_slice_dfb));
        dataflow_buffers.push_back(std::move(B_slice_dfb));
        dataflow_buffers.push_back(std::move(C_slice_dfb));
        dataflow_buffers.push_back(std::move(C_partials_dfb));
        if (plan.has_bias) {
            dataflow_buffers.push_back(DataflowBufferSpec{
                .unique_id = BIAS_DFB,
                .entry_size = plan.bias_entry_bytes,
                .num_entries = plan.bias_entries,
                .data_format_metadata = bias_format,
                .tile_format_metadata = bias_tensor->tensor_spec().tile(),
            });
        }
    }
    // ---- Reader ----
    std::vector<std::string> mcast_semaphores;
    for (uint32_t thread = 0; thread < QUASAR_NUM_READER_THREADS; ++thread) {
        for (const char* name : MCAST_SEMAPHORES) {
            mcast_semaphores.push_back(std::string(name) + "_" + std::to_string(thread));
        }
    }
    Group<KernelSpec::SemaphoreBinding> reader_semaphore_bindings;
    for (const std::string& name : mcast_semaphores) {
        reader_semaphore_bindings.push_back({.semaphore_spec_name = SemaphoreSpecName{name}, .accessor_name = name});
    }
    KernelSpec reader{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path(std::string(KERNEL_DIR) + "dataflow/unified_matmul_reader.cpp"),
        .num_threads = plan.num_reader_threads,
        .dfb_bindings = {ProducerOf(A_SLICE_DFB, "A_slice"), ProducerOf(B_SLICE_DFB, "B_slice")},
        .semaphore_bindings = std::move(reader_semaphore_bindings),
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = A_TENSOR, .accessor_name = "A"},
                TensorBinding{.tensor_parameter_name = B_TENSOR, .accessor_name = "B"},
            },
        .compile_time_args =
            {
                {"M_tiles", plan.M_tiles},
                {"K_tiles", plan.K_tiles},
                {"N_tiles", plan.N_tiles},
                {"batch_size", plan.batch_size},
                {"B_batch_stride_tiles", plan.B_batch_stride_tiles},
                {"C_slice_M_tiles", plan.C_slice_M_tiles},
                {"C_slice_N_tiles", plan.C_slice_N_tiles},
                {"C_slice_M_padded_tiles", plan.C_slice_M_padded_tiles},
                {"C_slice_N_padded_tiles", plan.C_slice_N_padded_tiles},
                {"K_chunk_tiles", plan.K_chunk_tiles},
                {"K_chunks_per_C_slice", plan.K_chunks_per_C_slice},
                {"K_chunks_per_C_slice_padded", plan.K_chunks_per_C_slice_padded},
                {"A_last_K_tile_valid_columns", plan.A_last_K_tile_valid_columns},
                {"A_borrowed", plan.borrow_A ? 1u : 0u},
                {"B_borrowed", plan.borrow_B ? 1u : 0u},
                {"num_reader_threads", plan.num_reader_threads},
                {"A_mcast_num_dests", plan.A_mcast_num_dests},
                {"B_mcast_num_dests", plan.B_mcast_num_dests},
            },
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"first_batch",
                  "num_batches",
                  "first_C_slice",
                  "num_C_slices",
                  "A_mcast_receiver",
                  "A_mcast_sender_x",
                  "A_mcast_sender_y",
                  "A_mcast_end_x",
                  "A_mcast_end_y",
                  "B_mcast_receiver",
                  "B_mcast_sender_x",
                  "B_mcast_sender_y",
                  "B_mcast_end_x",
                  "B_mcast_end_y"}},
        .hw_config = ttnn::create_reader_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };

    // ---- Writer ----
    KernelSpec writer{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path(std::string(KERNEL_DIR) + "dataflow/unified_matmul_writer.cpp"),
        .num_threads = plan.num_writer_threads,
        .compiler_options = {},
        .dfb_bindings = {ConsumerOf(C_SLICE_DFB, "C_slice")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = C_TENSOR, .accessor_name = "C"}},
        .compile_time_args =
            {
                {"M_tiles", plan.M_tiles},
                {"N_tiles", plan.N_tiles},
                {"batch_size", plan.batch_size},
                {"C_slice_M_tiles", plan.C_slice_M_tiles},
                {"C_slice_N_tiles", plan.C_slice_N_tiles},
                {"C_slice_M_padded_tiles", plan.C_slice_M_padded_tiles},
                {"C_slice_N_padded_tiles", plan.C_slice_N_padded_tiles},
                {"subblock_M_tiles", plan.subblock_M_tiles},
                {"subblock_N_tiles", plan.subblock_N_tiles},
                {"num_compute_threads", plan.num_compute_threads},
                {"num_writer_threads", plan.num_writer_threads},
                {"C_entries_per_thread", plan.C_entries_per_thread},
                {"C_borrowed", plan.borrow_C ? 1u : 0u},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"first_batch", "num_batches", "first_C_slice", "num_C_slices"}},
        .hw_config = ttnn::create_writer_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };

    // FUSE_BIAS is a define, as in the other matmul kernels: the bias DFB and tensor exist only with a bias.
    if (plan.has_bias) {
        reader.compiler_options.defines["FUSE_BIAS"] = "1";
        reader.dfb_bindings.push_back(ProducerOf(BIAS_DFB, "bias"));
        reader.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = BIAS_TENSOR, .accessor_name = "bias"});
    }

    // ---- Compute ----
    std::map<std::string, std::string> compute_defines_map;  // also throttle / stagger
    if (plan.has_bias) {
        compute_defines_map["FUSE_BIAS"] = "1";
    }
    if (config.fused_activation.has_value()) {
        compute_defines_map.merge(ttnn::operations::unary::utils::get_defines(
            config.fused_activation->op_type,
            config.fused_activation->params,
            "ACTIVATION",
            "i",
            operation_attributes.output_dtype));
    }
    const ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level =
        ttnn::get_throttle_level(operation_attributes.compute_kernel_config);
    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device.arch(), plan.cores.size(), compute_defines_map);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device.arch(), plan.cores.size(), compute_defines_map, throttle_level);
    KernelSpec::CompilerOptions::Defines compute_defines(compute_defines_map);

    ComputeHardwareConfig compute_hw_config =
        ttnn::to_compute_hardware_config(operation_attributes.compute_kernel_config.value());
    if (plan.fp32_dest_acc_en) {
        // With a 32-bit DST every consumed 32-bit DFB needs an explicit unpack mode. The partials are
        // reloaded with a data copy into DST, so unpack them straight to DST and keep fp32 precision;
        // fp32 operands feed the FPU and go through SrcA/SrcB.
        auto is_32bit = [](tt::DataFormat format) {
            return format == tt::DataFormat::Float32 || format == tt::DataFormat::Int32 ||
                   format == tt::DataFormat::UInt32;
        };
        ComputeHardwareConfig::ComputeUnpackModes& modes = compute_hw_config.unpack_modes;
        if (is_32bit(plan.C_partials_format)) {
            modes.emplace(C_PARTIALS_DFB, tt::tt_metal::UnpackMode::UnpackToDest);
        }
        if (is_32bit(plan.A_format)) {
            modes.emplace(A_SLICE_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
        }
        if (is_32bit(plan.B_format)) {
            modes.emplace(B_SLICE_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
        }
    }

    // A and B are bound ALL: every compute thread reads every entry, and each reader thread fills a contiguous part
    // of the DFB. The DFBs the compute produces are always STRIDED (each thread owns every N-th entry), the only
    // pattern for producers.
    KernelSpec compute{
        .unique_id = COMPUTE_KERNEL,
        .source = std::filesystem::path(std::string(KERNEL_DIR) + "compute/unified_matmul_compute.cpp"),
        .num_threads = plan.num_compute_threads,
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings =
            {
                AllConsumerOf(A_SLICE_DFB, "A_slice"),
                AllConsumerOf(B_SLICE_DFB, "B_slice"),
                ProducerOf(C_SLICE_DFB, "C_slice"),
                ProducerOf(C_PARTIALS_DFB, "C_partials"),
                ConsumerOf(C_PARTIALS_DFB, "C_partials"),
            },
        .compile_time_args =
            {
                {"K_chunk_tiles", plan.K_chunk_tiles},
                {"K_chunks_per_C_slice", plan.K_chunks_per_C_slice},
                {"K_chunks_per_C_slice_padded", plan.K_chunks_per_C_slice_padded},
                {"C_slice_M_padded_tiles", plan.C_slice_M_padded_tiles},
                {"C_slice_N_padded_tiles", plan.C_slice_N_padded_tiles},
                {"subblock_M_tiles", plan.subblock_M_tiles},
                {"subblock_N_tiles", plan.subblock_N_tiles},
                {"num_compute_threads", plan.num_compute_threads},
                {"C_entries_per_thread", plan.C_entries_per_thread},
                {"packer_l1_acc", plan.packer_l1_acc_en ? 1u : 0u},
                {"accumulate_in_C_slice", plan.accumulate_in_C_slice ? 1u : 0u},
                {"partials_format_differs", plan.C_partials_format != plan.C_format ? 1u : 0u},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_C_slices"}},
        .hw_config = compute_hw_config,
    };
    if (plan.has_bias) {
        compute.dfb_bindings.push_back(AllConsumerOf(BIAS_DFB, "bias"));
    }

    // ---- One work unit over the active cores ----
    const CoreRangeSet active_cores(ttsl::Span<const CoreCoord>(plan.cores));
    Group<WorkUnitSpec> work_units = {WorkUnitSpec{
        .name = "unified_matmul",
        .kernels = {READER_KERNEL, COMPUTE_KERNEL, WRITER_KERNEL},
        .target_nodes = active_cores,
    }};

    // ---- Per-core runtime args ----
    // Batch group g (cores_per_batch_group consecutive cores) takes a contiguous run of the batches, the first
    // (batch_size % num_batch_groups) groups one batch longer. Within a group each core takes a contiguous run of one
    // batch's C slice walk, the first (C_slices_per_batch % cores_per_batch_group) cores one C slice longer.
    ProgramRunArgs::KernelRunArgs reader_run_args{.kernel = READER_KERNEL};
    ProgramRunArgs::KernelRunArgs compute_run_args{.kernel = COMPUTE_KERNEL};
    ProgramRunArgs::KernelRunArgs writer_run_args{.kernel = WRITER_KERNEL};
    const uint32_t num_active_cores = plan.cores.size();
    const uint32_t cores_per_batch_group = num_active_cores / plan.num_batch_groups;
    const uint32_t batches_per_group_floor = plan.batch_size / plan.num_batch_groups;
    const uint32_t groups_with_extra_batch = plan.batch_size % plan.num_batch_groups;
    const uint32_t C_slices_per_core_floor = plan.C_slices_per_batch / cores_per_batch_group;
    const uint32_t cores_with_extra_C_slice = plan.C_slices_per_batch % cores_per_batch_group;
    // A is multicast along a core's row of the core rectangle and B down its column, each led by its first core, or
    // over the whole rectangle from its first core. The reader is on NOC_0 (Quasar has one NoC), which takes a
    // multicast rectangle from its lowest coordinates, the first core's, to its highest.
    const CoreRange core_grid = config.cores.bounding_box();
    for (uint32_t core = 0; core < num_active_cores; ++core) {
        const uint32_t batch_group = core / cores_per_batch_group;
        const uint32_t core_in_group = core % cores_per_batch_group;
        const uint32_t first_batch =
            batch_group * batches_per_group_floor + std::min(batch_group, groups_with_extra_batch);
        const uint32_t num_batches = batches_per_group_floor + (batch_group < groups_with_extra_batch ? 1 : 0);
        const uint32_t first_C_slice =
            core_in_group * C_slices_per_core_floor + std::min(core_in_group, cores_with_extra_C_slice);
        const uint32_t num_C_slices = C_slices_per_core_floor + (core_in_group < cores_with_extra_C_slice ? 1 : 0);
        const CoreCoord logical = plan.cores[core];
        const CoreCoord A_first =
            plan.A_mcast_to_all_cores ? core_grid.start_coord : CoreCoord{core_grid.start_coord.x, logical.y};
        const CoreCoord A_last =
            plan.A_mcast_to_all_cores ? core_grid.end_coord : CoreCoord{core_grid.end_coord.x, logical.y};
        const CoreCoord B_first =
            plan.B_mcast_to_all_cores ? core_grid.start_coord : CoreCoord{logical.x, core_grid.start_coord.y};
        const CoreCoord B_last =
            plan.B_mcast_to_all_cores ? core_grid.end_coord : CoreCoord{logical.x, core_grid.end_coord.y};
        const CoreCoord A_sender = device.worker_core_from_logical_core(A_first);
        const CoreCoord A_end = device.worker_core_from_logical_core(A_last);
        const CoreCoord B_sender = device.worker_core_from_logical_core(B_first);
        const CoreCoord B_end = device.worker_core_from_logical_core(B_last);
        TT_FATAL(
            (plan.A_mcast_num_dests == 0 || (A_sender.x <= A_end.x && A_sender.y <= A_end.y)) &&
                (plan.B_mcast_num_dests == 0 || (B_sender.x <= B_end.x && B_sender.y <= B_end.y)),
            "Multicast needs a group's first core at its lowest NoC coordinates");
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            logical,
            {{"first_batch", first_batch},
             {"num_batches", num_batches},
             {"first_C_slice", first_C_slice},
             {"num_C_slices", num_C_slices},
             {"A_mcast_receiver", plan.A_mcast_num_dests > 0 && logical != A_first ? 1u : 0u},
             {"A_mcast_sender_x", static_cast<uint32_t>(A_sender.x)},
             {"A_mcast_sender_y", static_cast<uint32_t>(A_sender.y)},
             {"A_mcast_end_x", static_cast<uint32_t>(A_end.x)},
             {"A_mcast_end_y", static_cast<uint32_t>(A_end.y)},
             {"B_mcast_receiver", plan.B_mcast_num_dests > 0 && logical != B_first ? 1u : 0u},
             {"B_mcast_sender_x", static_cast<uint32_t>(B_sender.x)},
             {"B_mcast_sender_y", static_cast<uint32_t>(B_sender.y)},
             {"B_mcast_end_x", static_cast<uint32_t>(B_end.x)},
             {"B_mcast_end_y", static_cast<uint32_t>(B_end.y)}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            logical,
            {{"first_batch", first_batch},
             {"num_batches", num_batches},
             {"first_C_slice", first_C_slice},
             {"num_C_slices", num_C_slices}});
        AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values, logical, {{"num_C_slices", num_batches * num_C_slices}});
    }

    Group<SemaphoreSpec> semaphores;
    for (const std::string& name : mcast_semaphores) {
        semaphores.push_back({.unique_id = SemaphoreSpecName{name}, .target_nodes = active_cores});
    }
    ProgramSpec spec{
        .name = "matmul_unified",
        .kernels = {reader, compute, writer},
        .dataflow_buffers = std::move(dataflow_buffers),
        .semaphores = std::move(semaphores),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };
    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run_args), std::move(compute_run_args), std::move(writer_run_args)},
        .tensor_args = {{A_TENSOR, A}, {B_TENSOR, B}, {C_TENSOR, C}},
    };
    if (plan.has_bias) {
        run_args.tensor_args.emplace(BIAS_TENSOR, std::cref(bias_tensor->mesh_tensor()));
    }
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
