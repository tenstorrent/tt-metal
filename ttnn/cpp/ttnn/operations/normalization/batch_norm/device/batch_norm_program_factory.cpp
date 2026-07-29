// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 program factory for batch_norm's BatchNormOperation.
//
// Faithful port of the legacy create_descriptor (see METAL2_PORT_PLAN.md). One config: interleaved,
// reader + writer + compute on every node of compute_with_storage_grid_size, per-node tile counts as RTAs.
// The port is a syntax swap only:
//   - CBDescriptor                             -> DataflowBufferSpec (one per legacy CB index)
//   - magic CB-index CTAs                      -> DFBBindings (census in the plan: 2 self-loops, or 3 on
//                                                 the output-typecast path)
//   - TensorAccessorArgs + buffer-address RTAs -> TensorParameter / TensorBinding (6, all Case 1)
//   - positional scalar CTAs                   -> named compile_time_args (sized per selected compute source)
//   - positional per-core RTAs                 -> named runtime_arg_schema + per-node runtime_arg_values
// Numerics, placement, and hardware config are unchanged.

#include "batch_norm_device_operation.hpp"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <bit>
#include <cmath>

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using namespace ttnn::operations::normalization;
namespace m2 = tt::tt_metal::experimental;

// ---- Kernel names ----
const m2::KernelSpecName READER{"reader"};
const m2::KernelSpecName WRITER{"writer"};
const m2::KernelSpecName COMPUTE{"compute"};

// ---- DataflowBuffer names (one per legacy CB index) ----
const m2::DFBSpecName INPUT_DFB{"input"};                // legacy c_0
const m2::DFBSpecName BATCH_MEAN_DFB{"batch_mean"};      // legacy c_1
const m2::DFBSpecName OUTPUT_DFB{"output"};              // legacy c_2 — compute output (FP32 staging when typecast)
const m2::DFBSpecName BATCH_VAR_DFB{"batch_var"};        // legacy c_3
const m2::DFBSpecName EPS_DFB{"eps"};                    // legacy c_4
const m2::DFBSpecName WEIGHT_DFB{"weight"};              // legacy c_5
const m2::DFBSpecName BIAS_DFB{"bias"};                  // legacy c_6
const m2::DFBSpecName DEN_DFB{"den"};                    // legacy c_7 — 1/(sqrt(batch_var + eps))
const m2::DFBSpecName TEMP_1_DFB{"temp_1"};              // legacy c_8 — (input - batch_mean)/(sqrt(batch_var + eps))
const m2::DFBSpecName OUTPUT_FINAL_DFB{"output_final"};  // legacy c_9 — writer-facing output; typecast config only

// ---- Tensor parameter names ----
const m2::TensorParamName INPUT_T{"input"};
const m2::TensorParamName BATCH_MEAN_T{"batch_mean"};
const m2::TensorParamName BATCH_VAR_T{"batch_var"};
const m2::TensorParamName WEIGHT_T{"weight"};
const m2::TensorParamName BIAS_T{"bias"};
const m2::TensorParamName OUTPUT_T{"output"};

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> extract_shape_dims(const ttnn::Tensor& x) {
    const auto& shape = x.padded_shape();
    const auto& tile = x.tensor_spec().tile();
    return {shape[-4], shape[-3], shape[-2] / tile.get_height(), shape[-1] / tile.get_width()};
}

void populate_runtime_arguments(
    m2::KernelRunArgs& reader_run_args,
    m2::KernelRunArgs& writer_run_args,
    m2::KernelRunArgs& compute_run_args,
    tt::tt_metal::CoreCoord compute_with_storage_grid_size,
    bool any_float32,
    const BatchNormOperation::operation_attributes_t& operation_attributes,
    const BatchNormOperation::tensor_args_t& tensor_args,
    BatchNormOperation::tensor_return_value_t& c) {
    const auto& [input_tensor, batch_mean_tensor, batch_var_tensor, weight_tensor, bias_tensor, _] = tensor_args;
    const auto eps = operation_attributes.eps;

    const auto [aN, aC, aHt, aWt] = extract_shape_dims(input_tensor);
    const auto [bN, bC, bHt, bWt] = extract_shape_dims(batch_mean_tensor);
    const auto [cN, cC, cHt, cWt] = extract_shape_dims(c);

    uint32_t num_output_tiles = c.physical_volume() / c.tensor_spec().tile().get_tile_hw();

    constexpr bool row_major = true;
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto
        [_unused_num_cores,
         _unused_all_cores,
         core_group_1,
         core_group_2,
         num_tiles_per_core_group_1,
         num_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);

    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        uint32_t num_tiles_per_core = 0;
        bool in_work_group = true;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            in_work_group = false;
        }

        uint32_t cHtWt = cHt * cWt;
        const auto scalar = eps;
        const auto packed_scalar_eps =
            any_float32 ? std::bit_cast<uint32_t>(scalar) : pack_two_bfloat16_into_uint32({scalar, scalar});

        // Nodes in neither work group received an all-zero argument block from the legacy zero-fill.
        // Metal 2.0 requires every named runtime arg to be set on every node the kernel runs on, so idle
        // nodes are emitted through the same named path, with every value zeroed — the same bytes the
        // legacy CoreRuntimeArgs(num_*_args, 0) wrote. (The legacy `cHt` / `cWt` slots are gone: neither
        // dataflow kernel ever read them.)
        m2::AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"eps", in_work_group ? packed_scalar_eps : 0u},
             {"start_tile_id", in_work_group ? start_tile_id : 0u},
             {"num_tiles", num_tiles_per_core},
             {"HtWt", in_work_group ? cHtWt : 0u},
             {"n_stride", in_work_group ? aHt * aWt * aC * static_cast<uint32_t>(aN > 1) : 0u},
             {"c_stride", in_work_group ? aHt * aWt * static_cast<uint32_t>(aC > 1) : 0u},
             {"N", in_work_group ? cN : 0u},
             {"C", in_work_group ? cC : 0u}});

        m2::AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"start_tile_id", in_work_group ? start_tile_id : 0u},
             {"num_tiles", num_tiles_per_core},
             {"HtWt", in_work_group ? cHtWt : 0u},
             {"n_stride", in_work_group ? bHt * bWt * bC * static_cast<uint32_t>(bN > 1) : 0u},
             {"c_stride", in_work_group ? bHt * bWt * static_cast<uint32_t>(bC > 1) : 0u},
             {"N", in_work_group ? cN : 0u},
             {"C", in_work_group ? cC : 0u}});

        m2::AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values,
            core,
            {{"num_tiles", num_tiles_per_core},
             {"tile_freq", in_work_group ? cHtWt : 0u},
             {"tile_start", in_work_group ? start_tile_id % cHtWt : 0u}});

        start_tile_id += num_tiles_per_core;
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::normalization {
ttnn::device_operation::ProgramArtifacts BatchNormOperation::BatchNormFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const auto& [input_tensor, batch_mean_tensor, batch_var_tensor, weight_tensor, bias_tensor, _] = tensor_args;

    auto* device = input_tensor.device();

    const bool weight_has_value = weight_tensor.has_value();
    const bool bias_has_value = bias_tensor.has_value();

    auto a_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    auto b_data_format = datatype_to_dataformat_converter(batch_mean_tensor.dtype());
    auto c_data_format = datatype_to_dataformat_converter(output.dtype());
    auto d_data_format = datatype_to_dataformat_converter(batch_var_tensor.dtype());
    auto e_data_format =
        weight_has_value ? datatype_to_dataformat_converter(weight_tensor->dtype()) : DataFormat::Float16_b;
    auto f_data_format =
        bias_has_value ? datatype_to_dataformat_converter(bias_tensor->dtype()) : DataFormat::Float16_b;

    const bool any_float32 =
        (a_data_format == DataFormat::Float32 || b_data_format == DataFormat::Float32 ||
         c_data_format == DataFormat::Float32 || d_data_format == DataFormat::Float32 ||
         e_data_format == DataFormat::Float32 || f_data_format == DataFormat::Float32);
    auto interm_data_format = any_float32 ? DataFormat::Float32 : a_data_format;

    uint32_t a_single_tile_size = tt::tile_size(a_data_format);
    uint32_t b_single_tile_size = tt::tile_size(b_data_format);
    uint32_t c_single_tile_size = tt::tile_size(c_data_format);
    uint32_t d_single_tile_size = tt::tile_size(d_data_format);
    uint32_t e_single_tile_size = tt::tile_size(e_data_format);
    uint32_t f_single_tile_size = tt::tile_size(f_data_format);
    uint32_t interm_single_tile_size = tt::tile_size(interm_data_format);

    // If accumulation occurs in float32 but output dtype is different, the compute kernel must typecast from
    // float32 to output dtype
    const bool needs_output_typecast =
        (interm_data_format == DataFormat::Float32 && c_data_format != DataFormat::Float32);

    // we parallelize the computation across the output tiles
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRangeSet(CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1}));

    // Number of tiles to store per input DFB (double buffer)
    constexpr uint32_t num_tiles_per_cb = 2;
    uint32_t b_num_tiles_per_cb = num_tiles_per_cb;

    // ---- DataflowBufferSpecs. entry_size / num_entries / data_format are the legacy CBDescriptor's
    //      page_size / (total_size / page_size) / data_format; tile_format_metadata stays unset because no
    //      legacy CBFormatDescriptor set `tile`. Placement is derived from the kernel bindings. ----
    m2::Group<m2::DataflowBufferSpec> dfbs;

    // Input buffers
    dfbs.push_back(m2::DataflowBufferSpec{
        .unique_id = INPUT_DFB,
        .entry_size = a_single_tile_size,
        .num_entries = num_tiles_per_cb,
        .data_format_metadata = a_data_format,
    });  // input
    dfbs.push_back(m2::DataflowBufferSpec{
        .unique_id = BATCH_MEAN_DFB,
        .entry_size = b_single_tile_size,
        .num_entries = b_num_tiles_per_cb,
        .data_format_metadata = b_data_format,
    });  // batch_mean
    dfbs.push_back(m2::DataflowBufferSpec{
        .unique_id = OUTPUT_DFB,
        .entry_size = needs_output_typecast ? interm_single_tile_size : c_single_tile_size,
        .num_entries = num_tiles_per_cb,
        .data_format_metadata = needs_output_typecast ? interm_data_format : c_data_format,
    });  // compute output (staging when typecast)

    // The DFB the writer drains. Without a typecast the writer takes the compute output directly; with one
    // it takes the BF16 stage the compute kernel typecasts into. Legacy expressed this by assigning
    // `writer_output_cb`; here the writer's single "dst" binding just names a different DFB.
    m2::DFBSpecName writer_output_dfb = OUTPUT_DFB;
    if (needs_output_typecast) {
        dfbs.push_back(m2::DataflowBufferSpec{
            .unique_id = OUTPUT_FINAL_DFB,
            .entry_size = c_single_tile_size,
            .num_entries = num_tiles_per_cb,
            .data_format_metadata = c_data_format,
        });  // writer-facing output (BF16)
        writer_output_dfb = OUTPUT_FINAL_DFB;
    }
    dfbs.push_back(m2::DataflowBufferSpec{
        .unique_id = BATCH_VAR_DFB,
        .entry_size = d_single_tile_size,
        .num_entries = b_num_tiles_per_cb,
        .data_format_metadata = d_data_format,
    });  // batch_var
    dfbs.push_back(m2::DataflowBufferSpec{
        .unique_id = EPS_DFB,
        .entry_size = interm_single_tile_size,
        .num_entries = b_num_tiles_per_cb,
        .data_format_metadata = interm_data_format,
    });  // eps
    // weight / bias are allocated unconditionally, exactly as in legacy: both the writer and the compute
    // kernel name them on every path, and only the FIFO traffic is gated on presence.
    dfbs.push_back(m2::DataflowBufferSpec{
        .unique_id = WEIGHT_DFB,
        .entry_size = e_single_tile_size,
        .num_entries = b_num_tiles_per_cb,
        .data_format_metadata = e_data_format,
    });  // weight
    dfbs.push_back(m2::DataflowBufferSpec{
        .unique_id = BIAS_DFB,
        .entry_size = f_single_tile_size,
        .num_entries = b_num_tiles_per_cb,
        .data_format_metadata = f_data_format,
    });  // bias

    // Temporary buffers to store intermediate results
    dfbs.push_back(m2::DataflowBufferSpec{
        .unique_id = DEN_DFB,
        .entry_size = interm_single_tile_size,
        .num_entries = num_tiles_per_cb,
        .data_format_metadata = interm_data_format,
    });  // to store 1/(sqrt(batch_var + eps))
    dfbs.push_back(m2::DataflowBufferSpec{
        .unique_id = TEMP_1_DFB,
        .entry_size = interm_single_tile_size,
        .num_entries = num_tiles_per_cb,
        .data_format_metadata = interm_data_format,
    });

    // ---- Tensor parameters. An absent optional tensor declares no parameter at all (rather than binding a
    //      null tensor); the matching kernel-side accessor is #ifdef'd out by the define below. ----
    m2::Group<m2::TensorParameter> tensor_parameters{
        m2::TensorParameter{.unique_id = INPUT_T, .spec = input_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = BATCH_MEAN_T, .spec = batch_mean_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = BATCH_VAR_T, .spec = batch_var_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = OUTPUT_T, .spec = output.tensor_spec()},
    };
    if (weight_has_value) {
        tensor_parameters.push_back(m2::TensorParameter{.unique_id = WEIGHT_T, .spec = weight_tensor->tensor_spec()});
    }
    if (bias_has_value) {
        tensor_parameters.push_back(m2::TensorParameter{.unique_id = BIAS_T, .spec = bias_tensor->tensor_spec()});
    }

    // ---- READER KERNEL. Produces `input` (DRAM load) and `eps` (scalar fill). ----
    m2::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/reader_batch_norm.cpp",
        .dfb_bindings =
            {m2::DFBBinding{
                 .dfb_spec_name = INPUT_DFB, .accessor_name = "input", .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = EPS_DFB, .accessor_name = "eps", .endpoint_type = m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = INPUT_T, .accessor_name = "input"}},
        .compile_time_args = {{"fill_eps_fp32", static_cast<uint32_t>(any_float32)}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"eps", "start_tile_id", "num_tiles", "HtWt", "n_stride", "c_stride", "N", "C"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // ---- WRITER KERNEL. Produces `batch_mean` / `batch_var` (and `weight` / `bias` when present) and
    //      consumes the compute output. `weight` / `bias` stay bound even when absent — the kernel still
    //      names them for their entry size — but their *tensor* bindings are conditional. ----
    m2::KernelSpec::CompilerOptions::Defines writer_defines;
    if (weight_has_value) {
        writer_defines.insert({"WEIGHT_HAS_VALUE", "1"});
    }
    if (bias_has_value) {
        writer_defines.insert({"BIAS_HAS_VALUE", "1"});
    }

    m2::Group<m2::TensorBinding> writer_tensor_bindings{
        m2::TensorBinding{.tensor_parameter_name = BATCH_MEAN_T, .accessor_name = "batch_mean"},
        m2::TensorBinding{.tensor_parameter_name = BATCH_VAR_T, .accessor_name = "batch_var"},
        m2::TensorBinding{.tensor_parameter_name = OUTPUT_T, .accessor_name = "output"},
    };
    if (weight_has_value) {
        writer_tensor_bindings.push_back(
            m2::TensorBinding{.tensor_parameter_name = WEIGHT_T, .accessor_name = "weight"});
    }
    if (bias_has_value) {
        writer_tensor_bindings.push_back(m2::TensorBinding{.tensor_parameter_name = BIAS_T, .accessor_name = "bias"});
    }

    auto param_data_format =
        weight_has_value ? e_data_format : (bias_has_value ? f_data_format : DataFormat::Float16_b);

    m2::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/writer_batch_norm.cpp",
        .compiler_options = {.defines = std::move(writer_defines)},
        .dfb_bindings =
            {m2::DFBBinding{
                 .dfb_spec_name = BATCH_MEAN_DFB,
                 .accessor_name = "batch_mean",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = writer_output_dfb,
                 .accessor_name = "dst",
                 .endpoint_type = m2::DFBEndpointType::CONSUMER},
             m2::DFBBinding{
                 .dfb_spec_name = BATCH_VAR_DFB,
                 .accessor_name = "batch_var",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = WEIGHT_DFB,
                 .accessor_name = "weight",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = BIAS_DFB, .accessor_name = "bias", .endpoint_type = m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings = std::move(writer_tensor_bindings),
        .compile_time_args =
            {{"batch_stat_is_fp32", static_cast<uint32_t>(b_data_format == DataFormat::Float32)},
             {"param_is_fp32", static_cast<uint32_t>(param_data_format == DataFormat::Float32)}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"start_tile_id", "num_tiles", "HtWt", "n_stride", "c_stride", "N", "C"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // ---- COMPUTE KERNEL ----
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    // `den` and `temp_1` are compute-only in every config -> self-loop (PRODUCER + CONSUMER). `weight` and
    // `bias` are consumed unconditionally: they are always allocated, and the runtime ternaries selecting
    // `dfb_affine_or_out` / `dfb_scaled_output` can reach `temp_1` or `output` on any path, so both of those
    // stay bound too.
    m2::Group<m2::DFBBinding> compute_dfb_bindings{
        m2::DFBBinding{
            .dfb_spec_name = INPUT_DFB, .accessor_name = "input", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = BATCH_MEAN_DFB,
            .accessor_name = "batch_mean",
            .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = BATCH_VAR_DFB,
            .accessor_name = "batch_var",
            .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = EPS_DFB, .accessor_name = "eps", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = WEIGHT_DFB, .accessor_name = "weight", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = BIAS_DFB, .accessor_name = "bias", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = DEN_DFB, .accessor_name = "den", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = DEN_DFB, .accessor_name = "den", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = TEMP_1_DFB, .accessor_name = "temp_1", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = TEMP_1_DFB, .accessor_name = "temp_1", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = OUTPUT_DFB, .accessor_name = "output", .endpoint_type = m2::DFBEndpointType::PRODUCER},
    };
    m2::KernelSpec::CompilerOptions::Defines compute_defines;
    if (needs_output_typecast) {
        // The writer moved to `output_final`, so compute is now the only toucher of `output`: it packs the
        // FP32 result there and reads it back to typecast. Self-loop it, and add the writer-facing DFB.
        compute_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = OUTPUT_DFB, .accessor_name = "output", .endpoint_type = m2::DFBEndpointType::CONSUMER});
        compute_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = OUTPUT_FINAL_DFB,
            .accessor_name = "output_final",
            .endpoint_type = m2::DFBEndpointType::PRODUCER});
        compute_defines.insert({"NEEDS_OUTPUT_TYPECAST", "1"});
    }

    // The compute kernel *source* is selected by config; the two sources are separate files sharing one
    // argument vocabulary, so the CTA table is sized per source (the FPU source has no typecast path).
    const bool is_sfpu_kernel = (fp32_dest_acc_en || any_float32);
    m2::KernelSpec::CompileTimeArgs compute_compile_time_args{
        {"weight_has_value", static_cast<uint32_t>(weight_has_value)},
        {"bias_has_value", static_cast<uint32_t>(bias_has_value)},
    };
    if (is_sfpu_kernel) {
        compute_compile_time_args.insert({"tc_in_fmt", static_cast<uint32_t>(DataFormat::Float32)});
        compute_compile_time_args.insert(
            {"tc_out_fmt",
             needs_output_typecast ? static_cast<uint32_t>(c_data_format)
                                   : static_cast<uint32_t>(DataFormat::Float32)});
    }

    // Style A: the op resolved a TTNN ComputeKernelConfig, so translate that. The helper carries
    // math_fidelity, the math_approx_mode bool -> Precision enum, fp32_dest_acc_en -> enable_32_bit_dest, and
    // the dst_full_sync_en -> double_buffer_dest *inversion*. bfp_pack_precision_mode stays at its default,
    // which matches the legacy default (the op never set bfp8_pack_precise). packer_l1_acc was unpacked from
    // the legacy tuple and never applied to the descriptor, so it is not translated.
    auto compute_hw_config =
        ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config);
    // Legacy `unpack_to_dest_mode` was a CB-id-indexed vector; Metal 2.0 keys it by DFB name and inverts the
    // sense of the value (UnpackToDestFp32 -> UnpackToDest, Default -> UnpackToSrc, expressed by omission).
    // Translated entry by entry from the legacy list.
    if (fp32_dest_acc_en) {
        auto& gen1_config = std::get<m2::ComputeGen1Config>(compute_hw_config);
        gen1_config.unpack_modes = m2::ComputeUnpackModes{
            {INPUT_DFB, UnpackMode::UnpackToDest},
            {BATCH_MEAN_DFB, UnpackMode::UnpackToDest},
            {BATCH_VAR_DFB, UnpackMode::UnpackToDest},
            {EPS_DFB, UnpackMode::UnpackToDest},
            {DEN_DFB, UnpackMode::UnpackToDest},
            {WEIGHT_DFB, UnpackMode::UnpackToDest},
            {TEMP_1_DFB, UnpackMode::UnpackToDest},
            {BIAS_DFB, UnpackMode::UnpackToDest},
        };
        if (needs_output_typecast) {
            // `output` holds Float32 and is unpacked in the typecast step only, so its entry is gated on the
            // same predicate as its CONSUMER binding. `output_final` deliberately gets no entry: it is only
            // ever a pack destination.
            gen1_config.unpack_modes.insert({OUTPUT_DFB, UnpackMode::UnpackToDest});
        }
    }

    m2::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = std::filesystem::path(fmt::format(
            "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_{}.cpp",
            is_sfpu_kernel ? "sfpu_kernel" : "kernel")),
        .compiler_options = {.defines = std::move(compute_defines)},
        .dfb_bindings = std::move(compute_dfb_bindings),
        .compile_time_args = std::move(compute_compile_time_args),
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "tile_freq", "tile_start"}},
        .hw_config = std::move(compute_hw_config),
    };

    // ---- Runtime argument values (per node) ----
    m2::KernelRunArgs reader_run_args{.kernel = READER};
    m2::KernelRunArgs writer_run_args{.kernel = WRITER};
    m2::KernelRunArgs compute_run_args{.kernel = COMPUTE};
    CMAKE_UNIQUE_NAMESPACE::populate_runtime_arguments(
        reader_run_args,
        writer_run_args,
        compute_run_args,
        compute_with_storage_grid_size,
        any_float32,
        operation_attributes,
        tensor_args,
        output);

    m2::ProgramSpec spec{
        .name = "batch_norm",
        .kernels = {std::move(reader_spec), std::move(writer_spec), std::move(compute_spec)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = {m2::WorkUnitSpec{
            .name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_device_cores}},
    };

    m2::ProgramRunArgs run_params;
    run_params.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)};
    run_params.tensor_args.emplace(INPUT_T, input_tensor.mesh_tensor());
    run_params.tensor_args.emplace(BATCH_MEAN_T, batch_mean_tensor.mesh_tensor());
    run_params.tensor_args.emplace(BATCH_VAR_T, batch_var_tensor.mesh_tensor());
    run_params.tensor_args.emplace(OUTPUT_T, output.mesh_tensor());
    if (weight_has_value) {
        run_params.tensor_args.emplace(WEIGHT_T, weight_tensor->mesh_tensor());
    }
    if (bias_has_value) {
        run_params.tensor_args.emplace(BIAS_T, bias_tensor->mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::operations::normalization
