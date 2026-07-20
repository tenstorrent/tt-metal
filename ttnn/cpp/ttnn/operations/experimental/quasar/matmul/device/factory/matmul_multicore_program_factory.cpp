// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/matmul/device/factory/matmul_multicore_program_factory.hpp"
#include <filesystem>
#include <map>
#include <string>
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// Spec/binding names. The DFB accessor names surface kernel-side as dfb::in0 / dfb::in1 / dfb::out;
// the tensor accessor names surface as tensor::in0 / tensor::in1 / tensor::out.
const DFBSpecName IN0_DFB{"in0"};
const DFBSpecName IN1_DFB{"in1"};
const DFBSpecName OUT_DFB{"out"};

const TensorParamName IN0_TENSOR{"in0"};
const TensorParamName IN1_TENSOR{"in1"};
const TensorParamName OUT_TENSOR{"out"};

const KernelSpecName READER_KERNEL{"reader"};
const KernelSpecName WRITER_KERNEL{"writer"};
const KernelSpecName COMPUTE_KERNEL_G1{"compute_g1"};
const KernelSpecName COMPUTE_KERNEL_G2{"compute_g2"};

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts MatmulMultiCoreProgramFactory::create_program_artifacts(
    const ttnn::prim::qsr::MatmulParams& operation_attributes,
    const ttnn::prim::qsr::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    using namespace CMAKE_UNIQUE_NAMESPACE;  // resolve the file-local ids below
    if (!tensor_args.optional_input_tensors.empty()) {
        TT_FATAL(!tensor_args.optional_input_tensors[0].has_value(), "Bias is not supported for matmul multi core");
    }

    const auto& a = tensor_args.input_tensors.at(0).mesh_tensor();
    const auto& b = tensor_args.input_tensors.at(1).mesh_tensor();
    const auto& output = tensor_return_value.at(0).mesh_tensor();

    TT_FATAL(operation_attributes.bcast_batch.has_value(), "Error: bcast_batch field should have been populated");
    bool bcast_batch = operation_attributes.bcast_batch.value();

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();

    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t in0_single_tile_size = tt::tile_size(in0_data_format);
    uint32_t in1_single_tile_size = tt::tile_size(in1_data_format);
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    tt::tt_metal::IDevice* device = &a.mutable_device();
    TT_FATAL(operation_attributes.compute_kernel_config.has_value(), "Compute kernel config should have been provided");

    const auto& cshape = output.padded_shape();  // C=A*B, N1MK*11KN->N1MN

    TT_FATAL(
        operation_attributes.program_config.has_value(),
        "program_config must be provided for MatmulMultiCoreProgramFactory");
    auto pc = std::get<operations::experimental::quasar::matmul::MatmulMultiCoreProgramConfig>(
        operation_attributes.program_config.value());
    if (!pc.allowed_worker_cores.has_value()) {
        log_warning(
            tt::LogOp,
            "MatmulMultiCoreProgramFactory: program_config.allowed_worker_cores not populated; auto-populating "
            "from device compute_with_storage_grid_size. Callers that bypass ttnn::prim::qsr::matmul() should invoke "
            "ttnn::operations::experimental::quasar::matmul::normalize_program_config() on the program config first. "
            "This will become "
            "a hard error in a future release.");
        auto device_grid = device->compute_with_storage_grid_size();
        pc.allowed_worker_cores =
            CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(device_grid.x - 1, device_grid.y - 1)));
    }
    auto compute_with_storage_grid_size = pc.allowed_worker_cores.value().bounding_box().grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t c_batch_size = get_batch_size(cshape);
    auto num_output_tiles_total = c_batch_size * cshape[-2] * cshape[-1] / TILE_HW;
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_tiles_per_core_group_1,
         num_output_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles_total);

    // C = A*B*...
    // MN = MK*KN
    uint32_t B = get_batch_size(ashape);
    uint32_t Mt = ashape[-2] / TILE_HEIGHT;
    uint32_t Kt = ashape[-1] / TILE_WIDTH;
    uint32_t Nt = bshape[-1] / TILE_WIDTH;
    uint32_t KtNt = Kt * Nt;
    uint32_t MtKt = Mt * Kt;
    uint32_t MtNt = Mt * Nt;

    // ---- Tensor parameters (replace the legacy buffer-address RTAs + TensorAccessorArgs plumbing) ----
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = IN0_TENSOR, .spec = a.tensor_spec()},
        TensorParameter{.unique_id = IN1_TENSOR, .spec = b.tensor_spec()},
        TensorParameter{.unique_id = OUT_TENSOR, .spec = output.tensor_spec()},
    };

    // ---- Dataflow buffers (1:1 with the legacy double-buffered CBs at indices 0, 1, c_16) ----
    constexpr uint32_t num_input_tiles = 2;
    constexpr uint32_t num_output_tiles = 2;
    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = IN0_DFB,
            .entry_size = in0_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = in0_data_format,
        },
        DataflowBufferSpec{
            .unique_id = IN1_DFB,
            .entry_size = in1_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = in1_data_format,
        },
        DataflowBufferSpec{
            .unique_id = OUT_DFB,
            .entry_size = output_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = output_data_format,
        },
    };

    // ---- Reader kernel ----
    uint32_t last_ktile_w = a.logical_shape()[-1] % TILE_WIDTH;
    uint32_t last_ktile_h = 0;

    KernelSpec reader{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
                                        "reader_bmm_8bank_output_tiles_partitioned.cpp"),
        .compiler_options = {},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN0_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = IN1_DFB, .accessor_name = "in1", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = IN0_TENSOR, .accessor_name = "in0"},
                TensorBinding{.tensor_parameter_name = IN1_TENSOR, .accessor_name = "in1"},
            },
        .compile_time_args =
            {
                {"in0_last_ktile_w", last_ktile_w},
                {"in0_last_ktile_h", last_ktile_h},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"Mt",
                     "Kt",
                     "Nt",
                     "MtKt",
                     "KtNt",
                     "batch",
                     "bcast_B",
                     "output_tile_start_id",
                     "num_output_tiles",
                     "MtNt"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // ---- Writer kernel ----
    KernelSpec writer{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
                                        "writer_unary_interleaved_start_id.cpp"),
        .compiler_options = {},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "out"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_pages", "start_id"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // ---- Compute kernel(s) — one KernelSpec per core group, preserving the per-group tile-count CTA ----
    const auto throttle_level = ttnn::get_throttle_level(operation_attributes.compute_kernel_config);
    std::map<std::string, std::string> mm_kernel_defines;
    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, mm_kernel_defines, throttle_level);
    // Table has no iterator-pair constructor; use the single-argument range constructor over the std::map.
    KernelSpec::CompilerOptions::Defines compute_defines(mm_kernel_defines);

    ComputeHardwareConfig compute_hw_config =
        ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config.value());

    // bmm compute kernel: B, Mt, Nt are just 3 for loops that act as 1 large loop,
    // so only set Nt for simplicity
    auto make_compute = [&](const KernelSpecName& unique_id, const CoreRangeSet& cores, uint32_t num_tiles) {
        (void)cores;  // node assignment lives on the WorkUnitSpec, not the KernelSpec
        return KernelSpec{
            .unique_id = unique_id,
            .source = std::filesystem::path(
                "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/compute/bmm.cpp"),
            .compiler_options = {.defines = compute_defines},
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = IN0_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
                    DFBBinding{
                        .dfb_spec_name = IN1_DFB, .accessor_name = "in1", .endpoint_type = DFBEndpointType::CONSUMER},
                    DFBBinding{
                        .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
                },
            .compile_time_args =
                {
                    {"batch", 1u},
                    {"Mt", 1u},
                    {"Kt", Kt},
                    {"Nt", num_tiles},
                },
            .hw_config = compute_hw_config,
        };
    };

    const bool group_2_present = !core_group_2.ranges().empty();

    Group<KernelSpec> kernels = {
        reader, writer, make_compute(COMPUTE_KERNEL_G1, core_group_1, num_output_tiles_per_core_group_1)};
    if (group_2_present) {
        kernels.push_back(make_compute(COMPUTE_KERNEL_G2, core_group_2, num_output_tiles_per_core_group_2));
    }

    // ---- Work units: reader + writer + the matching compute kernel per core group ----
    Group<WorkUnitSpec> work_units = {
        WorkUnitSpec{
            .name = "wu_g1",
            .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL_G1},
            .target_nodes = core_group_1,
        },
    };
    if (group_2_present) {
        work_units.push_back(WorkUnitSpec{
            .name = "wu_g2",
            .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL_G2},
            .target_nodes = core_group_2,
        });
    }

    // ---- Per-core runtime args for reader and writer ----
    ProgramRunArgs::KernelRunArgs reader_run_args{.kernel = READER_KERNEL};
    ProgramRunArgs::KernelRunArgs writer_run_args{.kernel = WRITER_KERNEL};
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_output_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        ProgramRunArgs::KernelRunArgs::RuntimeArgValues& reader_rtas = reader_run_args.runtime_arg_values;
        AddRuntimeArgsForNode(
            reader_rtas,
            core,
            {
                {"Mt", Mt},
                {"Kt", Kt},
                {"Nt", Nt},
                {"MtKt", MtKt},
                {"KtNt", KtNt},
                {"batch", B},
                {"bcast_B", uint32_t(bcast_batch)},
                {"output_tile_start_id", num_tiles_written},
                {"num_output_tiles", num_output_tiles_per_core},
                {"MtNt", MtNt},
            });
        ProgramRunArgs::KernelRunArgs::RuntimeArgValues& writer_rtas = writer_run_args.runtime_arg_values;
        AddRuntimeArgsForNode(
            writer_rtas,
            core,
            {
                {"num_pages", num_output_tiles_per_core},
                {"start_id", num_tiles_written},
            });
        num_tiles_written += num_output_tiles_per_core;
    }

    // ---- Assemble spec + run args ----
    ProgramSpec spec{
        .name = "matmul_multicore",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)},
        .tensor_args =
            {
                {IN0_TENSOR, a},
                {IN1_TENSOR, b},
                {OUT_TENSOR, output},
            },
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
