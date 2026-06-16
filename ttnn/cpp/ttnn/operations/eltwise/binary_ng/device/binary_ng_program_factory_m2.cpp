// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 program factory for binary_ng.
//
// SCOPE: this factory handles the no-broadcast (SubtileBroadcastType::NONE) x tile layout x
// interleaved (not sharded) x no-activation x plain ADD/SUB/MUL x no-typecast cases, across the
// following operand/compute axes (all routed here by select_program_factory()):
//   1. tensor-b present x FPU   (eltwise_binary_no_bcast_m2 compute)
//   2. tensor-b present x SFPU  (eltwise_binary_sfpu_no_bcast_m2 compute)
//   3. scalar-b (no tensor b) x FPU (eltwise_binary_scalar_m2 compute; writer fills the scalar tile)
// Every OTHER path (row-major, all broadcast types, where-op, quant-op, scalar-b-on-SFPU, sharded,
// activations, typecast, non-plain ops) stays on the legacy ProgramFactory::create_descriptor in
// binary_ng_program_factory.cpp. The full set of remaining kernel entry points and the routed-vs-
// legacy matrix are enumerated in METAL2_PORT_REPORT.md.
//
// The legacy factory's logic for these paths is preserved exactly (work-split, per-core tile
// strides, CB sizes, fp32_dest_acc_en, unpack-to-dest); only the host API is converted to
// Metal 2.0 ProgramSpec / ProgramRunArgs, and the kernels it binds are the forked *_m2.cpp ports.

#include "binary_ng_device_operation.hpp"
#include "binary_ng_utils.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include <filesystem>
#include <map>
#include <tuple>
#include <utility>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::operations::binary_ng {

namespace {

// Forked Metal 2.0 kernel sources.
//   tensor-b path (reader reads both src & src_b; writer copies dst):
constexpr const char* READER_AB_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast_m2.cpp";
constexpr const char* WRITER_AB_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast_m2.cpp";
constexpr const char* COMPUTE_FPU_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_no_bcast_m2.cpp";
constexpr const char* COMPUTE_SFPU_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast_m2.cpp";
//   scalar-b path (reader reads only src; writer fills the scalar tile AND copies dst):
constexpr const char* READER_SCALAR_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/reader_interleaved_no_bcast_m2.cpp";
constexpr const char* WRITER_SCALAR_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar_m2.cpp";
constexpr const char* COMPUTE_SCALAR_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_scalar_m2.cpp";

// Mirror of the legacy factory's get_shape_dims for the narrow (tile) path.
std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> get_shape_dims(const Tensor& x) {
    const auto& shape = x.padded_shape();
    const auto& tile = x.tensor_spec().tile();
    return {
        shape.rank() >= 5 ? shape[-5] : 1,
        shape[-4],
        shape[-3],
        tt::div_up(shape[-2], tile.get_height()),
        tt::div_up(shape[-1], tile.get_width())};
}

// For rank > 5 dims will be collapsed into a single dim (mirror of legacy extract_nD_dims).
uint32_t extract_nD_dims(const Tensor& x, const int out_rank) {
    const auto& shape = x.logical_shape();
    uint32_t nD_dim = 1;
    if (out_rank >= 6 && shape.rank() >= 6) {
        for (int i = -6; i >= -out_rank; --i) {
            nD_dim *= shape[i];
        }
    }
    return nD_dim;
}

}  // namespace

// Implements c = a op b, no-broadcast tile interleaved path (FPU/SFPU x tensor-b/scalar-b).
ttnn::device_operation::ProgramArtifacts BinaryNgDeviceOperation::ProgramSpecFactory::create_program_spec(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, tensor_return_value_t& c) {
    const auto& a = tensor_args.input_tensor_a;
    const auto& b_opt = tensor_args.input_tensor_b;
    const bool b_present = b_opt.has_value();
    const bool is_sfpu_op = operation_attributes.is_sfpu;

    const auto a_dtype = a.dtype();
    // Mirror the legacy b_dtype derivation for the routed cases. When b is a scalar, it is packed
    // as bfloat16 for block-float input types; the rhs CB format must match (BFLOAT16) — not the
    // block-float a_dtype. quant/where ops never route here.
    const auto b_dtype = b_present                                  ? b_opt->dtype()
                         : (is_sfpu_op && !is_block_float(a_dtype)) ? a_dtype
                                                                    : DataType::BFLOAT16;
    const auto c_dtype = c.dtype();
    const auto a_data_format = datatype_to_dataformat_converter(a_dtype);
    const auto b_data_format = datatype_to_dataformat_converter(b_dtype);
    const auto c_data_format = datatype_to_dataformat_converter(c_dtype);

    const uint32_t a_single_tile_size = tt::tile_size(a_data_format);
    const uint32_t b_single_tile_size = tt::tile_size(b_data_format);
    const uint32_t c_single_tile_size = tt::tile_size(c_data_format);

    // Compute defines for the binary op. Mirrors the legacy factory's OpConfig path for the
    // no-activation case (no lhs/rhs/post activations -> no PROCESS_* defines beyond the op
    // itself). Only plain ADD/SUB/MUL with no activations and no typecast route here, so OpConfig
    // injects no process_lhs/process_rhs/postprocess (see binary_ng_utils.cpp).
    auto op_config =
        is_sfpu_op ? OpConfig(operation_attributes.binary_op_type, std::in_place_type<OpConfig::SfpuBinaryOp>, a_dtype)
                   : OpConfig(operation_attributes.binary_op_type, std::in_place_type<OpConfig::FpuBinaryOp>, a_dtype);
    auto compute_kernel_defines = op_config.as_defines(a_dtype);
    add_activation_defines(compute_kernel_defines, {}, "LHS", a_dtype);
    add_activation_defines(compute_kernel_defines, {}, "RHS", b_dtype);
    add_activation_defines(compute_kernel_defines, {}, "POST", operation_attributes.input_dtype);
    compute_kernel_defines["BCAST_INPUT"] = "";  // NONE broadcast

    // Dataflow defines (no broadcast, interleaved -> not sharded).
    auto reader_defines = make_dataflow_defines(a_dtype, b_dtype);
    reader_defines["SRC_SHARDED"] = "0";
    reader_defines["SRC_SHARDED_B"] = "0";
    reader_defines["BCAST_LLK"] = "0";
    auto writer_defines = make_dataflow_defines(b_dtype);
    writer_defines["SRC_SHARDED"] = "0";
    writer_defines["DST_SHARDED"] = "0";

    // fp32 dest accumulation, mirroring the legacy factory rule.
    const bool fp32_dest_acc_en = c_data_format == tt::DataFormat::UInt32 || c_data_format == tt::DataFormat::Int32 ||
                                  c_data_format == tt::DataFormat::Float32 ||
                                  a_data_format == tt::DataFormat::Float32 ||
                                  b_data_format == tt::DataFormat::Float32 ||
                                  (a_data_format == tt::DataFormat::Int32 && b_data_format == tt::DataFormat::Int32) ||
                                  (a_data_format == tt::DataFormat::UInt32 && b_data_format == tt::DataFormat::UInt32);

    const auto& all_device_cores = operation_attributes.worker_grid;

    // Work split: parallelize across output tiles (tile path, interleaved -> physical_volume).
    const uint32_t tile_hw = c.tensor_spec().tile().get_tile_hw();
    const uint32_t rt_c_num_tiles = c.physical_volume() / tile_hw;
    const bool row_major = true;  // unsharded path uses row-major core ordering

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(all_device_cores, rt_c_num_tiles, row_major);
    const auto cores = corerange_to_cores(all_device_cores, {}, row_major);

    constexpr uint32_t num_tiles_per_cycle = 1;  // non-sharded default

    // ----------------------------------------------------------------------------------------
    // ProgramSpec resources
    // ----------------------------------------------------------------------------------------
    ProgramSpec spec;
    spec.name = is_sfpu_op ? "binary_ng_no_bcast_tile_sfpu"
                           : (b_present ? "binary_ng_no_bcast_tile_fpu" : "binary_ng_no_bcast_tile_scalar");

    // Tensor parameters (one per distinct accessed tensor). eltwise dataflow kernels declare
    // RuntimeTensorShape on the I/O tensors in the legacy factory, so mirror that relaxation
    // with dynamic_tensor_shape = true (interleaved -> TensorAccessor config unchanged). On the
    // scalar-b path there is no b tensor, so only a and c are declared.
    const TensorParamName A_PARAM{"a"};
    const TensorParamName B_PARAM{"b"};
    const TensorParamName C_PARAM{"c"};
    TensorParameterAdvancedOptions dyn_shape;
    dyn_shape.dynamic_tensor_shape = true;
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = A_PARAM, .spec = a.tensor_spec(), .advanced_options = dyn_shape});
    if (b_present) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = B_PARAM, .spec = b_opt->tensor_spec(), .advanced_options = dyn_shape});
    }
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = C_PARAM, .spec = c.tensor_spec(), .advanced_options = dyn_shape});

    // DFBs (one per legacy CB). c_0 = a (src), c_1 = b/scalar (src_b), c_2 = c (dst).
    // On the scalar-b path the legacy single-tensor reader fills c_0 from src and the writer fills
    // c_1 (single tile) from the packed scalar; both still exist.
    const DFBSpecName SRC_DFB{"src"};
    const DFBSpecName SRC_B_DFB{"src_b"};
    const DFBSpecName DST_DFB{"dst"};
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SRC_DFB,
        .entry_size = a_single_tile_size,
        .num_entries = 2,
        .data_format_metadata = a_data_format});
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SRC_B_DFB,
        .entry_size = b_single_tile_size,
        // Legacy: tensor-b CB is double-buffered (2 pages); scalar-b CB holds a single tile.
        .num_entries = b_present ? 2u : 1u,
        .data_format_metadata = b_data_format});
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = DST_DFB,
        .entry_size = c_single_tile_size,
        .num_entries = 2,
        .data_format_metadata = c_data_format});

    // Kernel names.
    const KernelSpecName READER_K{"reader"};
    const KernelSpecName WRITER_K{"writer"};
    // Per the recipe, preserve work-split multiplicity: one compute KernelSpec per core group.
    const KernelSpecName COMPUTE_K1{"compute_g1"};
    const KernelSpecName COMPUTE_K2{"compute_g2"};

    // op_config.as_defines() / make_dataflow_defines() return std::map<string,string>; the
    // Table range constructor copies them in directly.
    KernelSpec::CompilerOptions::Defines reader_def_table(reader_defines);
    KernelSpec::CompilerOptions::Defines writer_def_table(writer_defines);
    KernelSpec::CompilerOptions::Defines compute_def_table(compute_kernel_defines);

    const char* reader_src = b_present ? READER_AB_M2 : READER_SCALAR_M2;
    const char* writer_src = b_present ? WRITER_AB_M2 : WRITER_SCALAR_M2;
    const char* compute_src = b_present ? (is_sfpu_op ? COMPUTE_SFPU_M2 : COMPUTE_FPU_M2) : COMPUTE_SCALAR_M2;

    // Reader KernelSpec.
    KernelSpec reader_k;
    reader_k.unique_id = READER_K;
    reader_k.source = std::filesystem::path(reader_src);
    reader_k.compiler_options.defines = reader_def_table;
    reader_k.compile_time_args = {{"has_sharding", 0u}};
    if (b_present) {
        reader_k.runtime_arg_schema.runtime_arg_names = {
            "start_tile_id",
            "src_num_tiles",
            "dst_num_tiles",
            "dst_shard_width",
            "nD_stride",
            "d_stride",
            "n_stride",
            "c_stride",
            "D",
            "N",
            "C",
            "Ht",
            "Wt",
            "cND",
            "nD_stride_b",
            "d_stride_b",
            "n_stride_b",
            "c_stride_b",
            "src_num_tiles_b"};
        reader_k.dfb_bindings = {
            DFBBinding{.dfb_spec_name = SRC_DFB, .accessor_name = "src", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{
                .dfb_spec_name = SRC_B_DFB, .accessor_name = "src_b", .endpoint_type = DFBEndpointType::PRODUCER}};
        reader_k.tensor_bindings = {
            TensorBinding{.tensor_parameter_name = A_PARAM, .accessor_name = "src"},
            TensorBinding{.tensor_parameter_name = B_PARAM, .accessor_name = "src_b"}};
    } else {
        // Single-tensor (scalar-b) reader: reads only src into c_0.
        reader_k.runtime_arg_schema.runtime_arg_names = {
            "start_tile_id",
            "src_num_tiles",
            "dst_num_tiles",
            "dst_shard_width",
            "nD_stride",
            "d_stride",
            "n_stride",
            "c_stride",
            "D",
            "N",
            "C",
            "Ht",
            "Wt",
            "cND"};
        reader_k.dfb_bindings = {
            DFBBinding{.dfb_spec_name = SRC_DFB, .accessor_name = "src", .endpoint_type = DFBEndpointType::PRODUCER}};
        reader_k.tensor_bindings = {TensorBinding{.tensor_parameter_name = A_PARAM, .accessor_name = "src"}};
    }
    reader_k.hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER};

    // Writer KernelSpec.
    KernelSpec writer_k;
    writer_k.unique_id = WRITER_K;
    writer_k.source = std::filesystem::path(writer_src);
    writer_k.compiler_options.defines = writer_def_table;
    writer_k.compile_time_args = {{"has_sharding", 0u}};
    if (b_present) {
        writer_k.runtime_arg_schema.runtime_arg_names = {
            "start_tile_id", "dst_num_tiles", "dst_shard_width", "D", "N", "C", "Ht", "Wt", "cND"};
        writer_k.dfb_bindings = {
            DFBBinding{.dfb_spec_name = DST_DFB, .accessor_name = "dst", .endpoint_type = DFBEndpointType::CONSUMER}};
        writer_k.tensor_bindings = {TensorBinding{.tensor_parameter_name = C_PARAM, .accessor_name = "dst"}};
    } else {
        // Scalar-b writer: fills the scalar tile into c_1 (src_b) AND writes c_2 (dst). The packed
        // scalar is the leading named RTA `packed_scalar` (a value, mirroring legacy positional 0).
        writer_k.runtime_arg_schema.runtime_arg_names = {
            "packed_scalar", "start_tile_id", "dst_num_tiles", "dst_shard_width", "D", "N", "C", "Ht", "Wt", "cND"};
        writer_k.dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = SRC_B_DFB, .accessor_name = "src_b", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{.dfb_spec_name = DST_DFB, .accessor_name = "dst", .endpoint_type = DFBEndpointType::CONSUMER}};
        writer_k.tensor_bindings = {TensorBinding{.tensor_parameter_name = C_PARAM, .accessor_name = "dst"}};
    }
    writer_k.hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER};

    // Compute KernelSpec template (one per core group; only the num_tiles RTA is read by the
    // no-bcast/scalar compute kernels). num_tiles_per_cycle is a named CTA.
    // SFPU unpack-to-dest: legacy sets UnpackToDestFp32 on the operand CBs for non-POWER SFPU ops.
    // The Metal 2.0 ComputeHardwareConfig rejects UnpackToDestFp32 unless fp32_dest_acc_en=true; in
    // the legacy-inert case (fp32_dest_acc_en=false) the hardware ignored the mode anyway, so we
    // faithfully map it to Default there (only plain ADD/SUB/MUL route here, never POWER).
    auto make_compute_k = [&](const KernelSpecName& id) {
        KernelSpec k;
        k.unique_id = id;
        k.source = std::filesystem::path(compute_src);
        k.compiler_options.defines = compute_def_table;
        k.compile_time_args = {{"num_tiles_per_cycle", num_tiles_per_cycle}};
        k.runtime_arg_schema.runtime_arg_names = {"num_tiles"};
        k.dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = SRC_DFB, .accessor_name = "pre_lhs", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{
                .dfb_spec_name = SRC_B_DFB, .accessor_name = "pre_rhs", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = DST_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}};
        ComputeHardwareConfig hw;
        hw.fp32_dest_acc_en = fp32_dest_acc_en;
        if (is_sfpu_op && fp32_dest_acc_en) {
            hw.unpack_to_dest_mode.insert({SRC_DFB, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32});
            hw.unpack_to_dest_mode.insert({SRC_B_DFB, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32});
        }
        k.hw_config = hw;
        return k;
    };

    spec.kernels.push_back(reader_k);
    spec.kernels.push_back(writer_k);
    spec.kernels.push_back(make_compute_k(COMPUTE_K1));
    const bool has_group_2 = !core_group_2.ranges().empty();
    if (has_group_2) {
        spec.kernels.push_back(make_compute_k(COMPUTE_K2));
    }

    // WorkUnitSpec: reader + writer + compute_g1 over core_group_1; a second WorkUnit for
    // core_group_2 (reader + writer + compute_g2). Each DFB's producer and consumer share the
    // same WorkUnitSpec (local-DFB rule): reader (producer of src[/src_b]) + writer (consumer of
    // dst [and producer of the scalar src_b on the scalar path]) + compute (consumer of src/src_b,
    // producer of dst) all co-located per group.
    {
        WorkUnitSpec wu1;
        wu1.name = "group_1";
        wu1.kernels = {READER_K, WRITER_K, COMPUTE_K1};
        wu1.target_nodes = core_group_1;
        spec.work_units.push_back(wu1);
    }
    if (has_group_2) {
        WorkUnitSpec wu2;
        wu2.name = "group_2";
        wu2.kernels = {READER_K, WRITER_K, COMPUTE_K2};
        wu2.target_nodes = core_group_2;
        spec.work_units.push_back(wu2);
    }

    // ----------------------------------------------------------------------------------------
    // ProgramRunArgs (per-core runtime arguments). Mirrors the legacy factory's inline per-core
    // RTA emission for the no-bcast tile interleaved path, minus the buffer-address slots (now
    // carried by the TensorBindings).
    // ----------------------------------------------------------------------------------------
    ProgramRunArgs run_args;

    KernelRunArgs reader_run{.kernel = READER_K};
    KernelRunArgs writer_run{.kernel = WRITER_K};
    KernelRunArgs compute_run_1{.kernel = COMPUTE_K1};
    KernelRunArgs compute_run_2{.kernel = COMPUTE_K2};

    const auto out_rank = c.logical_shape().rank();
    const auto aND = extract_nD_dims(a, out_rank);
    const auto bND = b_present ? extract_nD_dims(*b_opt, out_rank) : 1u;
    const auto cND = extract_nD_dims(c, out_rank);
    const auto [aD, aN, aC, aHt, aWt] = get_shape_dims(a);
    const auto [bD, bN, bC, bHt, bWt] = b_present ? get_shape_dims(*b_opt) : std::tuple{1u, 1u, 1u, 1u, 1u};
    const auto [cD, cN, cC, cHt, cWt] = get_shape_dims(c);

    // Packed scalar value for the scalar-b path (mirrors the legacy pack_scalar_runtime_arg call).
    const uint32_t packed_scalar =
        b_present ? 0u : pack_scalar_runtime_arg(*operation_attributes.scalar, a.dtype(), /*is_quant_op=*/false);

    for (uint32_t i = 0, start_tile_id = 0; i < num_cores; i++) {
        const auto& core = cores[i];
        uint32_t c_num_tiles_core = 0;
        if (core_group_1.contains(core)) {
            c_num_tiles_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            c_num_tiles_core = num_tiles_per_core_group_2;
        } else {
            continue;  // unused core: emit no per-core args (Metal 2.0 derives node set from WorkUnitSpec)
        }

        // Reader RTAs (legacy slots, minus a/b base addresses now handled by ta:: bindings).
        if (b_present) {
            reader_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
                .node = core,
                .args = {
                    {"start_tile_id", start_tile_id},
                    {"src_num_tiles", 0u},  // a_num_tiles: 0 on the interleaved (unsharded) path
                    {"dst_num_tiles", c_num_tiles_core},
                    {"dst_shard_width", 0u},
                    {"nD_stride", aHt * aWt * aC * aN * aD * (aND > 1)},
                    {"d_stride", aHt * aWt * aC * aN * (aD > 1)},
                    {"n_stride", aHt * aWt * aC * (aN > 1)},
                    {"c_stride", aHt * aWt * (aC > 1)},
                    {"D", cD},
                    {"N", cN},
                    {"C", cC},
                    {"Ht", cHt},
                    {"Wt", cWt},
                    {"cND", cND},
                    {"nD_stride_b", bHt * bWt * bC * bN * bD * (bND > 1)},
                    {"d_stride_b", bHt * bWt * bC * bN * (bD > 1)},
                    {"n_stride_b", bHt * bWt * bC * (bN > 1)},
                    {"c_stride_b", bHt * bWt * (bC > 1)},
                    {"src_num_tiles_b", 0u},  // b_num_tiles: 0 on the interleaved (unsharded) path
                }});
        } else {
            // Single-tensor reader: a strides only (mirrors legacy reader_interleaved_no_bcast).
            reader_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
                .node = core,
                .args = {
                    {"start_tile_id", start_tile_id},
                    {"src_num_tiles", 0u},  // a_num_tiles: 0 on the interleaved (unsharded) path
                    {"dst_num_tiles", c_num_tiles_core},
                    {"dst_shard_width", 0u},
                    {"nD_stride", aHt * aWt * aC * aN * aD * (aND > 1)},
                    {"d_stride", aHt * aWt * aC * aN * (aD > 1)},
                    {"n_stride", aHt * aWt * aC * (aN > 1)},
                    {"c_stride", aHt * aWt * (aC > 1)},
                    {"D", cD},
                    {"N", cN},
                    {"C", cC},
                    {"Ht", cHt},
                    {"Wt", cWt},
                    {"cND", cND},
                }});
        }

        // Writer RTAs (legacy slots, minus the c base address now handled by ta:: binding; on the
        // scalar path the packed scalar value leads).
        if (b_present) {
            writer_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
                .node = core,
                .args = {
                    {"start_tile_id", start_tile_id},
                    {"dst_num_tiles", c_num_tiles_core},
                    {"dst_shard_width", 0u},
                    {"D", cD},
                    {"N", cN},
                    {"C", cC},
                    {"Ht", cHt},
                    {"Wt", cWt},
                    {"cND", cND},
                }});
        } else {
            writer_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
                .node = core,
                .args = {
                    {"packed_scalar", packed_scalar},
                    {"start_tile_id", start_tile_id},
                    {"dst_num_tiles", c_num_tiles_core},
                    {"dst_shard_width", 0u},
                    {"D", cD},
                    {"N", cN},
                    {"C", cC},
                    {"Ht", cHt},
                    {"Wt", cWt},
                    {"cND", cND},
                }});
        }

        // Compute RTA: the no-bcast/scalar compute kernels read only num_tiles (= c_num_tiles_core
        // on the tile path). freq/counter/scalar slots of the legacy 4-arg vector are dead for
        // these kernels and dropped (superfluous named args are a Metal 2.0 validation error).
        auto& compute_run = core_group_1.contains(core) ? compute_run_1 : compute_run_2;
        compute_run.runtime_arg_values.push_back(
            KernelRunArgs::NodeRuntimeArgs{.node = core, .args = {{"num_tiles", c_num_tiles_core}}});

        start_tile_id += c_num_tiles_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run));
    run_args.kernel_run_args.push_back(std::move(writer_run));
    run_args.kernel_run_args.push_back(std::move(compute_run_1));
    if (has_group_2) {
        run_args.kernel_run_args.push_back(std::move(compute_run_2));
    }

    // Tensor arguments: reference the SAME MeshTensors the factory received (matched by identity).
    run_args.tensor_args.insert({A_PARAM, TensorArgument{a.mesh_tensor()}});
    if (b_present) {
        run_args.tensor_args.insert({B_PARAM, TensorArgument{b_opt->mesh_tensor()}});
    }
    run_args.tensor_args.insert({C_PARAM, TensorArgument{c.mesh_tensor()}});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::binary_ng
