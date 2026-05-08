// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Single-core HW reduction program factory, migrated to the Metal 2.0 host API.
//
// HW reduces both H and W axes on a single worker core. Because there is no work split,
// the program contains exactly one WorkUnitSpec targeting the chosen core.

#include "reduce_op_single_core_hw_program_factory.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/host_api.hpp>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "reduce_metal2_factory_helpers.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

namespace ttnn::prim {

namespace m2 = tt::tt_metal::experimental::metal2_host_api;

namespace {

// KernelSpec / WorkUnitSpec / TensorParameter ids — kept local with an `HW_`
// prefix so the symbols don't clash with other factories' analogues under
// Unity builds. DFB ids and the MakeDFB / BindDFB / DefinesFromMap helpers are
// shared and live in reduce_metal2_factory_helpers.hpp.
constexpr const char* HW_READER_KERNEL = "reduce_hw_reader";
constexpr const char* HW_WRITER_KERNEL = "reduce_hw_writer";
constexpr const char* HW_COMPUTE_KERNEL = "reduce_hw_compute";
constexpr const char* HW_WORK_UNIT = "single_core";
constexpr const char* HW_INPUT_TENSOR = "input_tensor";
constexpr const char* HW_OUTPUT_TENSOR = "output_tensor";

m2::ProgramRunParams BuildRunParams(
    const ReduceSingleCoreHwSharedVariables& shared,
    bool negate,
    uint32_t Ht,
    const tt::tt_metal::MeshTensor& input_mt,
    const tt::tt_metal::MeshTensor& output_mt) {
    m2::ProgramRunParams params;

    m2::ProgramRunParams::KernelRunParams reader_params;
    reader_params.kernel_spec_name = HW_READER_KERNEL;
    reader_params.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
        .node = shared.core,
        .args =
            {
                {"num_tiles", shared.num_tensor_tiles},
                {"start_id", 0u},
            },
    });

    m2::ProgramRunParams::KernelRunParams writer_params;
    writer_params.kernel_spec_name = HW_WRITER_KERNEL;
    writer_params.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
        .node = shared.core,
        .args =
            {
                {"num_pages", shared.num_tensor_tiles / shared.out_dim_divider},
                {"start_id", 0u},
            },
    });

    m2::ProgramRunParams::KernelRunParams compute_params;
    compute_params.kernel_spec_name = HW_COMPUTE_KERNEL;
    if (!negate) {
        // Non-negate kernel (reduce.cpp) takes Ht as a per-node runtime arg so a single
        // KernelSpec can serve both the W factory (varying per-core Ht) and the HW factory
        // (constant Ht). The negate kernel (reduce_hw_neg.cpp) takes Ht as compile-time.
        compute_params.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
            .node = shared.core,
            .args = {{"Ht", Ht}},
        });
    }

    params.kernel_run_params = {std::move(reader_params), std::move(writer_params), std::move(compute_params)};
    params.tensor_args = {
        m2::ProgramRunParams::TensorArg{.tensor_parameter_name = HW_INPUT_TENSOR, .tensor = std::cref(input_mt)},
        m2::ProgramRunParams::TensorArg{.tensor_parameter_name = HW_OUTPUT_TENSOR, .tensor = std::cref(output_mt)},
    };
    return params;
}

}  // namespace

ReduceSingleCoreHwProgramFactory::cached_program_t ReduceSingleCoreHwProgramFactory::create(
    const ReduceParams& operation_attributes,
    const tt::tt_metal::Tensor& tensor_args,
    tt::tt_metal::Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& a = tensor_args;
    auto& output = tensor_return_value;

    // The Metal 2.0 TensorAccessor binding currently only supports non-sharded
    // interleaved buffers. Match the W factory's stance.
    TT_FATAL(
        !a.memory_config().is_sharded(),
        "ReduceSingleCoreHwProgramFactory (Metal 2.0): only interleaved input buffers are supported (got "
        "memory_config = {})",
        a.memory_config());
    TT_FATAL(
        !output.memory_config().is_sharded(),
        "ReduceSingleCoreHwProgramFactory (Metal 2.0): only interleaved output buffers are supported (got "
        "memory_config = {})",
        output.memory_config());

    const auto& shape = a.padded_shape();
    const uint32_t W = shape[3];
    const uint32_t H = shape[2];
    const uint32_t NC = shape[1] * shape[0];
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    const uint32_t tile_hw = a.tensor_spec().tile().get_tile_hw();

    const uint32_t Wt = W / tile_width;
    const uint32_t Ht = H / tile_height;
    TT_FATAL(Ht != 0 && Wt != 0, "Height and width in tiles must be non-zero (Ht={}, Wt={}, H={}, W={})", Ht, Wt, H, W);
    TT_FATAL(
        operation_attributes.dim == ReduceOpDim::HW,
        "ReduceSingleCoreHwProgramFactory supports HW dim only, got dim enum value {}",
        static_cast<int>(operation_attributes.dim));
    TT_FATAL(
        H % tile_height == 0 && W % tile_width == 0,
        "Reduce HW expects tile-aligned padded shape H={}, W={}",
        H,
        W);

    const uint32_t num_tensor_tiles = NC * H * W / tile_hw;
    const uint32_t out_dim_divider = Ht * Wt;
    {
        const uint32_t num_tensor_tiles_ht_wt = NC * Ht * Wt;
        TT_FATAL(
            num_tensor_tiles == num_tensor_tiles_ht_wt,
            "Reduce HW tile count mismatch: tile_hw path={} vs Ht*Wt path={}",
            num_tensor_tiles,
            num_tensor_tiles_ht_wt);
    }
    TT_FATAL(
        out_dim_divider != 0 && num_tensor_tiles % out_dim_divider == 0,
        "Reduce HW per-core input tiles {} must be divisible by Ht*Wt={}",
        num_tensor_tiles,
        out_dim_divider);

    // The single-core HW path uses REDUCE_SCALAR mode, which applies the scaler twice
    // internally (once per dimension). Compensate with sqrt(scaler). sqrt of a negative
    // number is NaN, so negative scalers are routed through the two-step W-then-H path
    // upstream (see reduce_op.cpp).
    TT_FATAL(operation_attributes.scaler >= 0, "Scalar must be non-negative");
    const float scaler = std::sqrt(operation_attributes.scaler);

    CoreCoord core_coord{0, 0};
    if (operation_attributes.sub_core_grids.has_value() && !operation_attributes.sub_core_grids->ranges().empty()) {
        const auto& r = operation_attributes.sub_core_grids->ranges().front();
        core_coord = r.start_coord;
        TT_FATAL(
            operation_attributes.sub_core_grids->contains(core_coord),
            "Selected core {} must be contained in provided sub_core_grids {}",
            core_coord,
            *operation_attributes.sub_core_grids);
    }
    const CoreRange core_range(core_coord, core_coord);
    const CoreRangeSet core_set(core_range);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), operation_attributes.compute_kernel_config);
    (void)math_approx_mode;
    (void)packer_l1_acc;
    (void)dst_full_sync_en;

    const tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    const uint32_t src0_single_tile_size = tile_size(src0_cb_data_format);
    const tt::DataFormat scaler_cb_data_format =
        src0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t scaler_single_tile_size = tile_size(scaler_cb_data_format);
    const tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t dst_single_tile_size = tile_size(dst_cb_data_format);

    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device");

    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;
    const uint32_t scaler_bits = std::bit_cast<uint32_t>(scaler);
    const uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    // ---- DFBs ----
    // Quasar's implicit-sync DFB scheduling allocates 2 transaction IDs per side
    // and asserts capacity % num_txn_ids == 0; standardize on 2 entries.
    constexpr uint32_t kNumInputEntries = 2;
    constexpr uint32_t kNumScalerEntries = 2;
    constexpr uint32_t kNumOutputEntries = 2;
    constexpr uint32_t kNumScratchEntries = 2;

    std::vector<m2::DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(
        MakeDFB(INPUT_DFB, src0_single_tile_size, kNumInputEntries, src0_cb_data_format, a.tensor_spec().tile()));
    dataflow_buffers.push_back(
        MakeDFB(SCALER_DFB, scaler_single_tile_size, kNumScalerEntries, scaler_cb_data_format, a.tensor_spec().tile()));
    dataflow_buffers.push_back(
        MakeDFB(OUTPUT_DFB, dst_single_tile_size, kNumOutputEntries, dst_cb_data_format, output.tensor_spec().tile()));
    if (operation_attributes.negate) {
        dataflow_buffers.push_back(MakeIntraDFB(
            ACC_DFB, dst_single_tile_size, kNumScratchEntries, dst_cb_data_format, output.tensor_spec().tile()));
        dataflow_buffers.push_back(MakeIntraDFB(
            INEG_DFB, dst_single_tile_size, kNumScratchEntries, dst_cb_data_format, output.tensor_spec().tile()));
    }

    const std::map<std::string, std::string> reduce_defines_map =
        reduce_op_utils::get_defines(operation_attributes.math_op, ReduceOpDim::HW);
    const auto reduce_defines = DefinesFromMap(reduce_defines_map);

    // ---- Reader ----
    m2::KernelSpec reader;
    reader.unique_id = HW_READER_KERNEL;
    reader.source = m2::KernelSpec::SourceFilePath{
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
        "reader_unary_reduce_universal_start_id.cpp"};
    reader.num_threads = 1;
    reader.compile_time_arg_bindings = {
        {"scaler_bits", scaler_bits},
    };
    reader.runtime_arguments_schema.named_runtime_args = {"num_tiles", "start_id"};
    reader.compiler_options.defines = reduce_defines;
    reader.config_spec = m2::DataMovementConfiguration{
        .gen1_data_movement_config =
            m2::DataMovementConfiguration::Gen1DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
            },
        .gen2_data_movement_config = m2::DataMovementConfiguration::Gen2DataMovementConfig{},
    };
    BindDFB(reader, INPUT_DFB, "input", m2::KernelSpec::DFBEndpointType::PRODUCER);
    BindDFB(reader, SCALER_DFB, "scaler", m2::KernelSpec::DFBEndpointType::PRODUCER);
    reader.tensor_bindings.push_back(
        m2::KernelSpec::TensorBinding{.tensor_parameter_name = HW_INPUT_TENSOR, .accessor_name = "input_tensor"});

    // ---- Writer ----
    m2::KernelSpec writer;
    writer.unique_id = HW_WRITER_KERNEL;
    writer.source = m2::KernelSpec::SourceFilePath{
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/writer_unary_interleaved.cpp"};
    writer.num_threads = 1;
    writer.runtime_arguments_schema.named_runtime_args = {"num_pages", "start_id"};
    writer.compiler_options.defines = reduce_defines;
    writer.config_spec = m2::DataMovementConfiguration{
        .gen1_data_movement_config =
            m2::DataMovementConfiguration::Gen1DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
            },
        .gen2_data_movement_config = m2::DataMovementConfiguration::Gen2DataMovementConfig{},
    };
    BindDFB(writer, OUTPUT_DFB, "output", m2::KernelSpec::DFBEndpointType::CONSUMER);
    writer.tensor_bindings.push_back(
        m2::KernelSpec::TensorBinding{.tensor_parameter_name = HW_OUTPUT_TENSOR, .accessor_name = "output_tensor"});

    // ---- Compute ----
    // Non-negate uses the shared reduce.cpp (Ht is runtime so the same source
    // serves both the W and HW factories). Negate uses the HW-specific reduce_hw_neg.cpp,
    // which takes all dims as compile-time arguments.
    const std::string compute_kernel_path =
        operation_attributes.negate
            ? "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_hw_neg.cpp"
            : "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce.cpp";

    m2::KernelSpec compute;
    compute.unique_id = HW_COMPUTE_KERNEL;
    compute.source = m2::KernelSpec::SourceFilePath{compute_kernel_path};
    compute.num_threads = 1;
    if (operation_attributes.negate) {
        compute.compile_time_arg_bindings = {
            {"Ht", Ht},
            {"Wt", Wt},
            {"NC", NC},
            {"post_mul_scaler_bits", post_mul_scaler_bits},
        };
    } else {
        compute.compile_time_arg_bindings = {
            {"Wt", Wt},
            {"NC", NC},
            {"post_mul_scaler_bits", post_mul_scaler_bits},
        };
        compute.runtime_arguments_schema.named_runtime_args = {"Ht"};
    }
    auto compute_defines = reduce_defines;
    if (use_post_mul) {
        compute_defines.emplace_back("REDUCE_POST_MUL", "1");
    }
    compute.compiler_options.defines = std::move(compute_defines);
    compute.config_spec = m2::ComputeConfiguration{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };
    BindDFB(compute, INPUT_DFB, "input", m2::KernelSpec::DFBEndpointType::CONSUMER);
    BindDFB(compute, SCALER_DFB, "scaler", m2::KernelSpec::DFBEndpointType::CONSUMER);
    BindDFB(compute, OUTPUT_DFB, "output", m2::KernelSpec::DFBEndpointType::PRODUCER);
    if (operation_attributes.negate) {
        BindDFB(compute, ACC_DFB, "acc_w", m2::KernelSpec::DFBEndpointType::PRODUCER);
        BindDFB(compute, ACC_DFB, "acc_r", m2::KernelSpec::DFBEndpointType::CONSUMER);
        BindDFB(compute, INEG_DFB, "ineg_w", m2::KernelSpec::DFBEndpointType::PRODUCER);
        BindDFB(compute, INEG_DFB, "ineg_r", m2::KernelSpec::DFBEndpointType::CONSUMER);
    }

    // ---- Single-core work unit ----
    m2::WorkUnitSpec work_unit;
    work_unit.unique_id = HW_WORK_UNIT;
    work_unit.kernels = {HW_READER_KERNEL, HW_WRITER_KERNEL, HW_COMPUTE_KERNEL};
    work_unit.target_nodes = core_set;

    // ---- Assemble + parameterize ----
    m2::ProgramSpec spec;
    spec.program_id = "ttnn::reduce_single_core_hw";
    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    spec.dataflow_buffers = std::move(dataflow_buffers);
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = HW_INPUT_TENSOR, .spec = a.mesh_tensor().tensor_spec()},
        m2::TensorParameter{.unique_id = HW_OUTPUT_TENSOR, .spec = output.mesh_tensor().tensor_spec()},
    };
    spec.work_units = {std::move(work_unit)};

    Program program = m2::MakeProgramFromSpec(*a.device(), spec);

    shared_variables_t shared{
        .core = core_coord,
        .num_tensor_tiles = num_tensor_tiles,
        .out_dim_divider = out_dim_divider,
    };

    auto run_params = BuildRunParams(shared, operation_attributes.negate, Ht, a.mesh_tensor(), output.mesh_tensor());
    m2::SetProgramRunParameters(program, run_params);

    return cached_program_t{std::move(program), std::move(shared)};
}

void ReduceSingleCoreHwProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ReduceParams& operation_attributes,
    const tt::tt_metal::Tensor& tensor_args,
    tt::tt_metal::Tensor& tensor_return_value) {
    const auto& shape = tensor_args.padded_shape();
    const uint32_t H = shape[2];
    const uint32_t tile_height = tensor_args.tensor_spec().tile().get_height();
    const uint32_t Ht = H / tile_height;

    auto run_params = BuildRunParams(
        cached_program.shared_variables,
        operation_attributes.negate,
        Ht,
        tensor_args.mesh_tensor(),
        tensor_return_value.mesh_tensor());
    m2::SetProgramRunParameters(cached_program.program, run_params);
}

}  // namespace ttnn::prim
