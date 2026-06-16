// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 program factory for binary_ng.
//
// SCOPE: this factory handles interleaved (not sharded), no-activation, no-typecast, plain
// ADD/SUB/MUL, non-where, non-quant, NONE-broadcast cases that fully populate at least one tile,
// across the following axes (all routed here by select_program_factory()):
//   TILE layout:
//     1. NONE broadcast x tensor-b x FPU   (eltwise_binary_no_bcast_m2 compute)
//     2. NONE broadcast x tensor-b x SFPU  (eltwise_binary_sfpu_no_bcast_m2 compute)
//     3. NONE broadcast x scalar-b x FPU   (eltwise_binary_scalar_m2 compute; writer fills scalar)
//     4. NONE broadcast x scalar-b x SFPU  (eltwise_binary_sfpu_scalar_m2 compute; writer fills scalar)
//   ROW_MAJOR layout:
//     5. NONE broadcast x tensor-b x FPU   (RM reader/writer forks + eltwise_binary_no_bcast_m2 compute)
//
// The simple-broadcast tile paths (SCALAR/ROW/COL) are NO LONGER routed here: the forked LLK-bcast
// kernels deadlock for multi-tile broadcast operands (test_bf4b_bf8b add hangs) and the degenerate
// SCALAR_B case ([1,2]+[3]) produces wrong values. They (and any "01-volume" / sub-tile tensor on
// the NONE path, which mis-reads under dynamic_tensor_shape) fall back to the legacy
// ProgramFactory. The simple-broadcast forked kernels + spec wiring (cases 5-7 of the prior pass)
// are left in this file for a future debug pass but are unreachable via the routing predicate.
//
// Every OTHER path (sharded, activations, typecast, where-op, quant-op, ALL subtile broadcast
// (scalar/row/col/mixed), RM broadcast, RM scalar-op, and sub-tile / 01-volume tensors) stays on
// the legacy ProgramFactory::create_descriptor in binary_ng_program_factory.cpp. The full
// routed-vs-legacy matrix is enumerated in METAL2_PORT_REPORT.md.
//
// The legacy factory's logic for these paths is preserved exactly (work-split, per-core tile
// strides, CB sizes, fp32_dest_acc_en, unpack-to-dest, use_llk_bcast); only the host API is
// converted to Metal 2.0 ProgramSpec / ProgramRunArgs, and the kernels it binds are the forked
// *_m2.cpp ports.

#include "binary_ng_device_operation.hpp"
#include "binary_ng_utils.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include <filesystem>
#include <map>
#include <tuple>
#include <utility>
#include <variant>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::operations::binary_ng {

namespace {

// ---- Forked Metal 2.0 kernel sources ----------------------------------------------------------
// NONE-broadcast tile, tensor-b path (reader reads both src & src_b; writer copies dst):
constexpr const char* READER_AB_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast_m2.cpp";
constexpr const char* WRITER_AB_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast_m2.cpp";
constexpr const char* COMPUTE_FPU_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_no_bcast_m2.cpp";
constexpr const char* COMPUTE_SFPU_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast_m2.cpp";
// NONE-broadcast tile, scalar-b path (reader reads only src; writer fills the scalar tile AND
// copies dst):
constexpr const char* READER_SCALAR_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/reader_interleaved_no_bcast_m2.cpp";
constexpr const char* WRITER_SCALAR_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar_m2.cpp";
constexpr const char* COMPUTE_SCALAR_FPU_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_scalar_m2.cpp";
constexpr const char* COMPUTE_SCALAR_SFPU_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar_m2.cpp";
// Simple-broadcast tile readers + LLK-bcast computes (FPU):
constexpr const char* READER_SCALAR_BCAST_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_scalar_bcast_m2.cpp";
constexpr const char* READER_ROW_BCAST_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_row_bcast_m2.cpp";
constexpr const char* READER_COL_BCAST_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_col_bcast_m2.cpp";
constexpr const char* COMPUTE_SCALAR_BCAST_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_binary_scalar_bcast_m2.cpp";
constexpr const char* COMPUTE_ROW_BCAST_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_binary_row_bcast_m2.cpp";
constexpr const char* COMPUTE_COL_BCAST_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_binary_col_bcast_m2.cpp";
// Row-major NONE-broadcast tensor-b path (reader/writer forks; compute is the FPU no-bcast kernel):
constexpr const char* READER_RM_AB_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_rm_no_bcast_m2.cpp";
constexpr const char* WRITER_RM_M2 =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_rm_no_bcast_m2.cpp";

// Mirror of the legacy factory's get_shape_dims for the tile path.
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

// Mirror of legacy calculate_compute_kernel_args (drives the compute freq/counter RTAs for the
// broadcast tile path).
std::tuple<uint32_t, uint32_t> calculate_compute_kernel_args(
    SubtileBroadcastType broadcast_type, uint32_t start_tile_id, uint32_t Ht, uint32_t Wt) {
    uint32_t start_t = start_tile_id % (Ht * Wt);
    uint32_t start_tw = start_t % Wt;

    switch (broadcast_type) {
        case SubtileBroadcastType::NONE:
        case SubtileBroadcastType::ROW_A:
        case SubtileBroadcastType::ROW_B: return {1, 0};
        case SubtileBroadcastType::SCALAR_A:
        case SubtileBroadcastType::SCALAR_B: return {Ht * Wt, start_t};
        case SubtileBroadcastType::COL_A:
        case SubtileBroadcastType::ROW_B_COL_A:
        case SubtileBroadcastType::COL_B:
        case SubtileBroadcastType::ROW_A_COL_B: return {Wt, start_tw};
        default: __builtin_unreachable();
    }
}

}  // namespace

// ----------------------------------------------------------------------------------------------
// Shared LLK-bcast routing predicate. Reproduces the legacy factory's `use_llk_bcast` decision
// (binary_ng_program_factory.cpp) EXACTLY for the routed subset (no activations, no exp ops,
// non-where, non-quant, tile layout). select_program_factory() and the factory both consult this
// so the spec path is selected only on the (arch, broadcast, dtype, fp32) tuples the legacy LLK
// path would have compiled. Kept in sync with the legacy gates: BH col bf16->fp32 hang, dtype/arch
// matrix, BH MOVB2D fp32 fallback, UInt16-scalar relational fallback. The exp-op and where-op
// gates are subsumed by the routing predicate restricting to plain ADD/SUB/MUL and non-where.
bool binary_ng_m2_use_llk_bcast(
    SubtileBroadcastType subtile_broadcast_type, DataType a_dtype, DataType b_dtype, bool fp32_dest_acc_en) {
    [[maybe_unused]] const auto a_data_format = datatype_to_dataformat_converter(a_dtype);
    [[maybe_unused]] const auto b_data_format = datatype_to_dataformat_converter(b_dtype);

    const bool is_col_bcast =
        subtile_broadcast_type == SubtileBroadcastType::COL_A || subtile_broadcast_type == SubtileBroadcastType::COL_B;
    const bool has_bf16_input = a_dtype == DataType::BFLOAT16 || b_dtype == DataType::BFLOAT16;
    // BH col bf16->fp32 hang.
    if (tt::tt_metal::hal::get_arch() == tt::ARCH::BLACKHOLE && is_col_bcast && fp32_dest_acc_en && has_bf16_input) {
        return false;
    }

    auto all_match = [&](DataType dt) { return a_dtype == dt && b_dtype == dt; };

    bool use_llk = false;
    if (subtile_broadcast_type == SubtileBroadcastType::ROW_A ||
        subtile_broadcast_type == SubtileBroadcastType::ROW_B ||
        subtile_broadcast_type == SubtileBroadcastType::COL_A ||
        subtile_broadcast_type == SubtileBroadcastType::COL_B ||
        subtile_broadcast_type == SubtileBroadcastType::SCALAR_A ||
        subtile_broadcast_type == SubtileBroadcastType::SCALAR_B) {
        if (all_match(DataType::BFLOAT16) || all_match(DataType::BFLOAT8_B) || all_match(DataType::BFLOAT4_B)) {
            use_llk = true;
        }
        if (all_match(DataType::FLOAT32) || all_match(DataType::INT32) || all_match(DataType::UINT32) ||
            all_match(DataType::UINT16)) {
            if (tt::tt_metal::hal::get_arch() == tt::ARCH::WORMHOLE_B0) {
                use_llk = true;
            }
        }
    }
    if (!use_llk) {
        return false;
    }

    // BH MOVB2D fp32 fallback (SCALAR / ROW use MOVB2D; COL uses ELWADD and is unaffected).
    if (fp32_dest_acc_en && tt::tt_metal::hal::get_arch() == tt::ARCH::BLACKHOLE) {
        const bool uses_movb2d = subtile_broadcast_type == SubtileBroadcastType::SCALAR_A ||
                                 subtile_broadcast_type == SubtileBroadcastType::SCALAR_B ||
                                 subtile_broadcast_type == SubtileBroadcastType::ROW_A ||
                                 subtile_broadcast_type == SubtileBroadcastType::ROW_B;
        if (uses_movb2d) {
            return false;
        }
    }

    // UInt16 scalar relational fallback (#36217): only EQ/NE (postprocess) or LT/GT/LE/GE (SFPU)
    // hit the corruption. The m2 routing restricts to plain ADD/SUB/MUL with no postprocess, so
    // none of those relational ops route here -- but we mirror the gate for faithfulness: a plain
    // arithmetic UInt16 scalar op has no postprocess and is unaffected, so this never fires for the
    // routed subset.
    return true;
}

bool binary_ng_m2_routes_to_spec(
    const BinaryNgDeviceOperation::operation_attributes_t& attributes,
    const BinaryNgDeviceOperation::tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    const bool b_present = b.has_value();

    // where/quant ops never route. SFPU routes only on NONE-broadcast (tensor-b or scalar-b);
    // broadcast paths route only FPU.
    if (attributes.is_where_op || attributes.is_quant_op) {
        return false;
    }
    const bool no_activations =
        attributes.lhs_activations.empty() && attributes.rhs_activations.empty() && attributes.post_activations.empty();
    const bool plain_op = attributes.binary_op_type == BinaryOpType::ADD ||
                          attributes.binary_op_type == BinaryOpType::SUB ||
                          attributes.binary_op_type == BinaryOpType::MUL;
    if (!no_activations || !plain_op) {
        return false;
    }
    const bool sharded = a.memory_config().is_sharded() || (b_present && b->memory_config().is_sharded()) ||
                         attributes.memory_config.is_sharded();
    if (sharded) {
        return false;
    }
    // No typecast (a and output share dtype) so the compute kernel binds no POST activation.
    if (a.dtype() != attributes.get_dtype()) {
        return false;
    }
    // scalar-b path requires a scalar value.
    const bool operand_ok = b_present || attributes.scalar.has_value();
    if (!operand_ok) {
        return false;
    }

    const bool rm = attributes.input_layout_a == Layout::ROW_MAJOR && attributes.output_layout == Layout::ROW_MAJOR &&
                    (!b_present || attributes.input_layout_b == Layout::ROW_MAJOR);
    const bool tile = attributes.input_layout_a == Layout::TILE && attributes.output_layout == Layout::TILE &&
                      (!b_present || attributes.input_layout_b == Layout::TILE);

    const auto sbt = attributes.subtile_broadcast_type;

    // Only NONE-broadcast is routed to the spec factory. The simple-broadcast tile paths
    // (SCALAR/ROW/COL) are NOT routed: the forked LLK-bcast kernels deadlock on this path for
    // multi-tile broadcast operands (e.g. test_bf4b_bf8b add over [5,3,128,64]+[1,3,128,1] hangs),
    // and the degenerate SCALAR_B case ([1,2]+[3]) produces wrong values. The legacy ProgramFactory
    // handles all broadcast cases correctly, so they stay there until the broadcast spec path is
    // debugged. (Reverts the broadcast widening from the prior pass; see METAL2_PORT_REPORT.md.)
    if (sbt != SubtileBroadcastType::NONE) {
        return false;
    }

    // Sub-tile / "01-volume" guard. Degenerate tensors whose logical shape does not fill a full
    // 32x32 tile (e.g. rank-1 [1], [1,2], [2,3]) read back wrong values on the spec path: with
    // dynamic_tensor_shape = true the framework rebuilds the interleaved TensorAccessor page
    // geometry from the LIVE (sub-tile) logical TensorSpec, so the per-tile 2048-byte NoC reads in
    // the forked dataflow kernels no longer line up with the on-device padded-tile layout (a=[1]
    // reads back 0, so [1]+[2] yields 2 instead of 3). Tensors that fully populate at least one tile
    // (logical volume >= tile area) are unaffected; keep the degenerate cases on the legacy factory,
    // which carries an explicit page-size override.
    // (The output tensor is not available here on a cache miss; for NONE broadcast a and b share the
    // broadcasted shape, so guarding on the inputs is sufficient.)
    const uint64_t tile_area =
        static_cast<uint64_t>(a.tensor_spec().tile().get_height()) * a.tensor_spec().tile().get_width();
    const bool sub_tile =
        a.logical_shape().volume() < tile_area || (b_present && b->logical_shape().volume() < tile_area);
    if (sub_tile) {
        return false;
    }

    if (rm) {
        // Row-major: only NONE broadcast x tensor-b x FPU.
        return b_present && !attributes.is_sfpu;
    }
    if (!tile) {
        return false;
    }

    // NONE tile: tensor-b FPU, tensor-b SFPU, scalar-b FPU, scalar-b SFPU.
    return true;
}

// Implements c = a op b for the routed interleaved cases (see SCOPE header).
ttnn::device_operation::ProgramArtifacts BinaryNgDeviceOperation::ProgramSpecFactory::create_program_spec(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, tensor_return_value_t& c) {
    const auto& a = tensor_args.input_tensor_a;
    const auto& b_opt = tensor_args.input_tensor_b;
    const bool b_present = b_opt.has_value();
    const bool is_sfpu_op = operation_attributes.is_sfpu;
    const auto sbt = operation_attributes.subtile_broadcast_type;
    const bool is_none_bcast = sbt == SubtileBroadcastType::NONE;
    const bool inputs_row_major = operation_attributes.input_layout_a == Layout::ROW_MAJOR;

    const auto a_dtype = a.dtype();
    // Mirror the legacy b_dtype derivation for the routed cases. When b is a scalar, it is packed
    // as bfloat16 for block-float input types; the rhs CB format must match (BFLOAT16) -- not the
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

    // fp32 dest accumulation, mirroring the legacy factory rule.
    const bool fp32_dest_acc_en = c_data_format == tt::DataFormat::UInt32 || c_data_format == tt::DataFormat::Int32 ||
                                  c_data_format == tt::DataFormat::Float32 ||
                                  a_data_format == tt::DataFormat::Float32 ||
                                  b_data_format == tt::DataFormat::Float32 ||
                                  (a_data_format == tt::DataFormat::Int32 && b_data_format == tt::DataFormat::Int32) ||
                                  (a_data_format == tt::DataFormat::UInt32 && b_data_format == tt::DataFormat::UInt32);

    // For the simple-broadcast tile path the factory is reached only on the LLK-bcast subset
    // (see binary_ng_m2_routes_to_spec); the software-fallback path is never routed here.
    const bool use_llk_bcast = !is_none_bcast && !inputs_row_major;

    // Compute defines. Mirrors the legacy OpConfig path for the no-activation case (no
    // lhs/rhs/post activations -> no PROCESS_* defines beyond the op itself). Only plain
    // ADD/SUB/MUL with no activations / no typecast route here.
    auto op_config =
        is_sfpu_op ? OpConfig(operation_attributes.binary_op_type, std::in_place_type<OpConfig::SfpuBinaryOp>, a_dtype)
                   : OpConfig(operation_attributes.binary_op_type, std::in_place_type<OpConfig::FpuBinaryOp>, a_dtype);
    auto compute_kernel_defines = op_config.as_defines(a_dtype);
    add_activation_defines(compute_kernel_defines, {}, "LHS", a_dtype);
    add_activation_defines(compute_kernel_defines, {}, "RHS", b_dtype);
    add_activation_defines(compute_kernel_defines, {}, "POST", operation_attributes.input_dtype);

    // BCAST_INPUT: legacy kernel_config.bcast_input_str(). NONE -> "" ; *_A -> "0" ; *_B -> "1".
    std::string bcast_input_str;
    switch (sbt) {
        case SubtileBroadcastType::NONE: bcast_input_str = ""; break;
        case SubtileBroadcastType::SCALAR_A:
        case SubtileBroadcastType::ROW_A:
        case SubtileBroadcastType::COL_A: bcast_input_str = "0"; break;
        case SubtileBroadcastType::SCALAR_B:
        case SubtileBroadcastType::ROW_B:
        case SubtileBroadcastType::COL_B: bcast_input_str = "1"; break;
        default: bcast_input_str = ""; break;
    }
    compute_kernel_defines["BCAST_INPUT"] = bcast_input_str;
    // SRC_BCAST / SRC_BCAST_B on the compute kernel (overwrite_compute_kernel_name_and_defines).
    if (sbt == SubtileBroadcastType::ROW_A || sbt == SubtileBroadcastType::ROW_B ||
        sbt == SubtileBroadcastType::COL_A || sbt == SubtileBroadcastType::COL_B ||
        sbt == SubtileBroadcastType::SCALAR_A || sbt == SubtileBroadcastType::SCALAR_B) {
        compute_kernel_defines["SRC_BCAST"] =
            (sbt == SubtileBroadcastType::ROW_A || sbt == SubtileBroadcastType::COL_A ||
             sbt == SubtileBroadcastType::SCALAR_A)
                ? "1"
                : "0";
        compute_kernel_defines["SRC_BCAST_B"] =
            (sbt == SubtileBroadcastType::ROW_B || sbt == SubtileBroadcastType::COL_B ||
             sbt == SubtileBroadcastType::SCALAR_B)
                ? "1"
                : "0";
    }

    // Dataflow defines (interleaved -> not sharded).
    auto reader_defines = make_dataflow_defines(a_dtype, b_dtype);
    reader_defines["SRC_SHARDED"] = "0";
    reader_defines["SRC_SHARDED_B"] = "0";
    reader_defines["BCAST_LLK"] = use_llk_bcast ? "1" : "0";
    // Reader SRC_BCAST / SRC_BCAST_B (get_reader_kernel_name_and_defines).
    if (!is_none_bcast) {
        reader_defines["SRC_BCAST"] = (sbt == SubtileBroadcastType::ROW_A || sbt == SubtileBroadcastType::COL_A ||
                                       sbt == SubtileBroadcastType::SCALAR_A)
                                          ? "1"
                                          : "0";
        reader_defines["SRC_BCAST_B"] = (sbt == SubtileBroadcastType::ROW_B || sbt == SubtileBroadcastType::COL_B ||
                                         sbt == SubtileBroadcastType::SCALAR_B)
                                            ? "1"
                                            : "0";
    }
    auto writer_defines = make_dataflow_defines(b_dtype);
    writer_defines["SRC_SHARDED"] = "0";
    writer_defines["DST_SHARDED"] = "0";

    const auto& all_device_cores = operation_attributes.worker_grid;

    // ----------------------------------------------------------------------------------------
    // Resource names.
    // ----------------------------------------------------------------------------------------
    const TensorParamName A_PARAM{"a"};
    const TensorParamName B_PARAM{"b"};
    const TensorParamName C_PARAM{"c"};
    const DFBSpecName SRC_DFB{"src"};
    const DFBSpecName SRC_B_DFB{"src_b"};
    const DFBSpecName DST_DFB{"dst"};
    const DFBSpecName LLK_POST_A_DFB{"llk_post_a"};  // legacy c_5 (bcast on A)
    const DFBSpecName LLK_POST_B_DFB{"llk_post_b"};  // legacy c_6 (bcast on B)
    const KernelSpecName READER_K{"reader"};
    const KernelSpecName WRITER_K{"writer"};
    const KernelSpecName COMPUTE_K1{"compute_g1"};
    const KernelSpecName COMPUTE_K2{"compute_g2"};

    // Which extra LLK-bcast CB does the broadcast operand use? legacy adds c_5 when the broadcast
    // operand is A (SCALAR_A/ROW_A/COL_A) and c_6 when it is B.
    const bool bcast_on_a = sbt == SubtileBroadcastType::SCALAR_A || sbt == SubtileBroadcastType::ROW_A ||
                            sbt == SubtileBroadcastType::COL_A;
    const bool bcast_on_b = sbt == SubtileBroadcastType::SCALAR_B || sbt == SubtileBroadcastType::ROW_B ||
                            sbt == SubtileBroadcastType::COL_B;

    ProgramSpec spec;
    spec.name = inputs_row_major ? "binary_ng_rm_no_bcast_fpu"
                : is_none_bcast
                    ? (is_sfpu_op ? (b_present ? "binary_ng_no_bcast_tile_sfpu" : "binary_ng_no_bcast_tile_scalar_sfpu")
                                  : (b_present ? "binary_ng_no_bcast_tile_fpu" : "binary_ng_no_bcast_tile_scalar"))
                    : "binary_ng_simple_bcast_tile_fpu";

    // Tensor parameters. eltwise dataflow kernels declare RuntimeTensorShape on the I/O tensors in
    // the legacy factory, so mirror that relaxation with dynamic_tensor_shape = true (interleaved
    // -> TensorAccessor config unchanged). On the scalar-b path there is no b tensor.
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

    // DFBs (one per legacy CB). c_0 = a (src), c_1 = b/scalar (src_b), c_2 = c (dst). The
    // simple-broadcast LLK path adds c_5 (bcast-on-A) or c_6 (bcast-on-B) as an intermediate.
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
    if (use_llk_bcast && bcast_on_a) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = LLK_POST_A_DFB,
            .entry_size = a_single_tile_size,
            .num_entries = 2,
            .data_format_metadata = a_data_format});
    }
    if (use_llk_bcast && bcast_on_b) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = LLK_POST_B_DFB,
            .entry_size = b_single_tile_size,
            .num_entries = 2,
            .data_format_metadata = b_data_format});
    }

    // op_config.as_defines() / make_dataflow_defines() return std::map<string,string>; the Table
    // range constructor copies them in directly.
    KernelSpec::CompilerOptions::Defines reader_def_table(reader_defines);
    KernelSpec::CompilerOptions::Defines writer_def_table(writer_defines);
    KernelSpec::CompilerOptions::Defines compute_def_table(compute_kernel_defines);

    // Kernel source selection.
    const char* reader_src = nullptr;
    const char* writer_src = nullptr;
    const char* compute_src = nullptr;
    if (inputs_row_major) {
        reader_src = READER_RM_AB_M2;
        writer_src = WRITER_RM_M2;
        compute_src = COMPUTE_FPU_M2;  // RM NONE-bcast uses the FPU no-bcast compute (freq=1,counter=0)
    } else if (is_none_bcast) {
        reader_src = b_present ? READER_AB_M2 : READER_SCALAR_M2;
        writer_src = b_present ? WRITER_AB_M2 : WRITER_SCALAR_M2;
        compute_src = b_present ? (is_sfpu_op ? COMPUTE_SFPU_M2 : COMPUTE_FPU_M2)
                                : (is_sfpu_op ? COMPUTE_SCALAR_SFPU_M2 : COMPUTE_SCALAR_FPU_M2);
    } else {
        // Simple broadcast tile (LLK path), tensor-b FPU.
        switch (sbt) {
            case SubtileBroadcastType::SCALAR_A:
            case SubtileBroadcastType::SCALAR_B:
                reader_src = READER_SCALAR_BCAST_M2;
                compute_src = COMPUTE_SCALAR_BCAST_M2;
                break;
            case SubtileBroadcastType::ROW_A:
            case SubtileBroadcastType::ROW_B:
                reader_src = READER_ROW_BCAST_M2;
                compute_src = COMPUTE_ROW_BCAST_M2;
                break;
            case SubtileBroadcastType::COL_A:
            case SubtileBroadcastType::COL_B:
                reader_src = READER_COL_BCAST_M2;
                compute_src = COMPUTE_COL_BCAST_M2;
                break;
            default: __builtin_unreachable();
        }
        writer_src = WRITER_AB_M2;  // tensor-b writer (no scalar fill)
    }

    // ----------------------------------------------------------------------------------------
    // Reader KernelSpec.
    // ----------------------------------------------------------------------------------------
    KernelSpec reader_k;
    reader_k.unique_id = READER_K;
    reader_k.source = std::filesystem::path(reader_src);
    reader_k.compiler_options.defines = reader_def_table;
    if (inputs_row_major) {
        // Row-major reader: addresses + page-size overrides dropped (ta::src/ta::src_b carry them).
        reader_k.compile_time_args = {};
        reader_k.runtime_arg_schema.runtime_arg_names = {
            "dst_num_tiles",
            "aD",
            "aN",
            "aC",
            "aHt",
            "aND",
            "bD",
            "bN",
            "bC",
            "bHt",
            "bND",
            "cHt",
            "cC",
            "cND",
            "current_block_start",
            "rows_per_tile",
            "row_width_elements",
            "alignment_a",
            "alignment_b",
            "tiles_per_row",
            "stride_size_bytes"};
        reader_k.dfb_bindings = {
            DFBBinding{.dfb_spec_name = SRC_DFB, .accessor_name = "src", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{
                .dfb_spec_name = SRC_B_DFB, .accessor_name = "src_b", .endpoint_type = DFBEndpointType::PRODUCER}};
        reader_k.tensor_bindings = {
            TensorBinding{.tensor_parameter_name = A_PARAM, .accessor_name = "src"},
            TensorBinding{.tensor_parameter_name = B_PARAM, .accessor_name = "src_b"}};
    } else {
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
                DFBBinding{
                    .dfb_spec_name = SRC_DFB, .accessor_name = "src", .endpoint_type = DFBEndpointType::PRODUCER},
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
            reader_k.dfb_bindings = {DFBBinding{
                .dfb_spec_name = SRC_DFB, .accessor_name = "src", .endpoint_type = DFBEndpointType::PRODUCER}};
            reader_k.tensor_bindings = {TensorBinding{.tensor_parameter_name = A_PARAM, .accessor_name = "src"}};
        }
    }
    reader_k.hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER};

    // ----------------------------------------------------------------------------------------
    // Writer KernelSpec.
    // ----------------------------------------------------------------------------------------
    KernelSpec writer_k;
    writer_k.unique_id = WRITER_K;
    writer_k.source = std::filesystem::path(writer_src);
    writer_k.compiler_options.defines = writer_def_table;
    if (inputs_row_major) {
        writer_k.compile_time_args = {};
        writer_k.runtime_arg_schema.runtime_arg_names = {
            "row_width_elements",
            "dst_num_tiles",
            "outD",
            "outN",
            "outC",
            "outHt",
            "outND",
            "current_block_start",
            "rows_per_tile",
            "alignment",
            "tiles_per_row",
            "stride_size_bytes"};
        writer_k.dfb_bindings = {
            DFBBinding{.dfb_spec_name = DST_DFB, .accessor_name = "dst", .endpoint_type = DFBEndpointType::CONSUMER}};
        writer_k.tensor_bindings = {TensorBinding{.tensor_parameter_name = C_PARAM, .accessor_name = "dst"}};
    } else {
        writer_k.compile_time_args = {{"has_sharding", 0u}};
        if (b_present) {
            writer_k.runtime_arg_schema.runtime_arg_names = {
                "start_tile_id", "dst_num_tiles", "dst_shard_width", "D", "N", "C", "Ht", "Wt", "cND"};
            writer_k.dfb_bindings = {DFBBinding{
                .dfb_spec_name = DST_DFB, .accessor_name = "dst", .endpoint_type = DFBEndpointType::CONSUMER}};
            writer_k.tensor_bindings = {TensorBinding{.tensor_parameter_name = C_PARAM, .accessor_name = "dst"}};
        } else {
            // Scalar-b writer: fills the scalar tile into c_1 (src_b) AND writes c_2 (dst). The
            // packed scalar is the leading named RTA `packed_scalar` (mirroring legacy positional 0).
            writer_k.runtime_arg_schema.runtime_arg_names = {
                "packed_scalar", "start_tile_id", "dst_num_tiles", "dst_shard_width", "D", "N", "C", "Ht", "Wt", "cND"};
            writer_k.dfb_bindings = {
                DFBBinding{
                    .dfb_spec_name = SRC_B_DFB, .accessor_name = "src_b", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = DST_DFB, .accessor_name = "dst", .endpoint_type = DFBEndpointType::CONSUMER}};
            writer_k.tensor_bindings = {TensorBinding{.tensor_parameter_name = C_PARAM, .accessor_name = "dst"}};
        }
    }
    writer_k.hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER};

    // ----------------------------------------------------------------------------------------
    // Work split. Tile path parallelizes across output tiles; RM path parallelizes across row
    // blocks (rt_c_num_tiles = total_row_blocks). All other RM geometry (num_rows_per_tile,
    // stride sizes) mirrors the legacy factory.
    // ----------------------------------------------------------------------------------------
    constexpr uint32_t num_tiles_per_cycle = 1;  // non-sharded default
    const bool row_major_cores = true;           // unsharded path uses row-major core ordering

    // RM geometry (only computed / used on the RM path).
    uint32_t rm_num_rows_per_tile = 0;
    uint32_t rm_tiles_per_row_width = 1;
    uint32_t rm_common_row_width_elements = 0;
    uint32_t rm_reader_stride_size_bytes = 0;
    uint32_t rm_writer_stride_size_bytes = 0;
    uint32_t rm_a_alignment = 0;
    uint32_t rm_b_alignment = 0;
    uint32_t rm_c_alignment = 0;

    uint32_t rt_c_num_tiles;
    const uint32_t tile_height = c.tensor_spec().tile().get_height();
    const uint32_t tile_width = c.tensor_spec().tile().get_width();
    const uint32_t tile_hw = tile_height * tile_width;

    const auto out_rank = c.logical_shape().rank();
    const auto aND = extract_nD_dims(a, out_rank);
    const auto bND = b_present ? extract_nD_dims(*b_opt, out_rank) : 1u;
    const auto cND = extract_nD_dims(c, out_rank);
    const auto [aD, aN, aC, aHt, aWt] = get_shape_dims(a);
    const auto [bD, bN, bC, bHt, bWt] = b_present ? get_shape_dims(*b_opt) : std::tuple{1u, 1u, 1u, 1u, 1u};
    const auto [cD, cN, cC, cHt, cWt] = get_shape_dims(c);

    const auto aHt_r = a.padded_shape()[-2];
    const auto aWt_r = a.padded_shape()[-1];
    const auto bHt_r = b_present ? b_opt->padded_shape()[-2] : 0u;
    const auto bWt_r = b_present ? b_opt->padded_shape()[-1] : 0u;
    const auto cHt_r = c.padded_shape()[-2];
    const auto cWt_r = c.padded_shape()[-1];

    if (inputs_row_major) {
        rm_a_alignment = a.buffer()->alignment();
        rm_b_alignment = b_present ? b_opt->buffer()->alignment() : rm_a_alignment;
        rm_c_alignment = c.buffer()->alignment();

        const uint32_t c_aligned_page_size = c.buffer()->aligned_page_size();
        const uint32_t a_aligned_page_size = a.buffer()->aligned_page_size();
        const uint32_t b_aligned_page_size = b_present ? b_opt->buffer()->aligned_page_size() : a_aligned_page_size;

        const uint32_t c_row_width_elements_aligned = c_aligned_page_size / c.element_size();
        const uint32_t a_row_width_elements_aligned = a_aligned_page_size / a.element_size();
        const uint32_t b_row_width_elements_aligned =
            b_present ? (b_aligned_page_size / b_opt->element_size()) : a_row_width_elements_aligned;

        rm_common_row_width_elements = c_row_width_elements_aligned;
        if (aWt_r == cWt_r) {
            rm_common_row_width_elements = std::min(rm_common_row_width_elements, a_row_width_elements_aligned);
        }
        if (b_present && bWt_r == cWt_r) {
            rm_common_row_width_elements = std::min(rm_common_row_width_elements, b_row_width_elements_aligned);
        }
        rm_common_row_width_elements = std::max<uint32_t>(1u, rm_common_row_width_elements);

        rm_num_rows_per_tile = std::max<uint32_t>(1u, tile_hw / rm_common_row_width_elements);
        const bool aligned_for_a =
            (aWt_r == cWt_r) ? ((rm_common_row_width_elements * a.element_size()) == a_aligned_page_size) : true;
        const bool aligned_for_b = (b_present && bWt_r == cWt_r)
                                       ? ((rm_common_row_width_elements * b_opt->element_size()) == b_aligned_page_size)
                                       : true;
        const bool aligned_for_c = (rm_common_row_width_elements * c.element_size()) == c_aligned_page_size;
        if (!aligned_for_a || !aligned_for_b || !aligned_for_c) {
            rm_num_rows_per_tile = 1;
        }

        const uint32_t row_blocks_per_channel = tt::div_up(cHt_r, rm_num_rows_per_tile);
        const uint32_t total_row_blocks = cND * cD * cN * cC * row_blocks_per_channel;
        rm_tiles_per_row_width = tt::div_up(rm_common_row_width_elements, tile_hw);
        const uint32_t a_tile_bytes = tile_hw * a.element_size();
        const uint32_t a_row_width_bytes = rm_common_row_width_elements * a.element_size();
        rm_reader_stride_size_bytes =
            (a_row_width_bytes > a_tile_bytes) ? a_tile_bytes : tt::round_up(a_row_width_bytes, rm_a_alignment);
        const uint32_t c_tile_bytes = tile_hw * c.element_size();
        const uint32_t c_row_width_bytes = rm_common_row_width_elements * c.element_size();
        rm_writer_stride_size_bytes =
            (c_row_width_bytes > c_tile_bytes) ? c_tile_bytes : tt::round_up(c_row_width_bytes, rm_c_alignment);
        rt_c_num_tiles = total_row_blocks;
    } else {
        rt_c_num_tiles = c.physical_volume() / tile_hw;
    }

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(all_device_cores, rt_c_num_tiles, row_major_cores);
    const auto cores = corerange_to_cores(all_device_cores, {}, row_major_cores);
    const bool has_group_2 = !core_group_2.ranges().empty();

    // ----------------------------------------------------------------------------------------
    // Compute KernelSpec (one per core group, preserving work-split multiplicity). The no-bcast,
    // scalar, and RM compute kernels read num_tiles only; the broadcast compute kernels also read
    // tile_freq + tile_start (calculate_compute_kernel_args). The RM compute (FPU no-bcast) reads
    // num_tiles = c_num_tiles_core * tiles_per_row_width.
    // SFPU unpack-to-dest: legacy sets UnpackToDestFp32 on the operand CBs for non-POWER SFPU ops;
    // ComputeHardwareConfig rejects it unless fp32_dest_acc_en, and the legacy-inert case is mapped
    // to Default (only plain ADD/SUB/MUL route here, never POWER).
    const bool compute_has_freq_counter = use_llk_bcast;  // broadcast kernels take 3 RTAs
    auto make_compute_k = [&](const KernelSpecName& id) {
        KernelSpec k;
        k.unique_id = id;
        k.source = std::filesystem::path(compute_src);
        k.compiler_options.defines = compute_def_table;
        k.compile_time_args = {{"num_tiles_per_cycle", num_tiles_per_cycle}};
        if (compute_has_freq_counter) {
            k.runtime_arg_schema.runtime_arg_names = {"num_tiles", "tile_freq", "tile_start"};
        } else {
            k.runtime_arg_schema.runtime_arg_names = {"num_tiles"};
        }
        Group<DFBBinding> bindings = {
            DFBBinding{
                .dfb_spec_name = SRC_DFB, .accessor_name = "pre_lhs", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{
                .dfb_spec_name = SRC_B_DFB, .accessor_name = "pre_rhs", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = DST_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}};
        // The LLK-bcast compute kernel produces+consumes the intermediate c_5/c_6 DFB (self-loop).
        if (use_llk_bcast && bcast_on_a) {
            bindings.push_back(DFBBinding{
                .dfb_spec_name = LLK_POST_A_DFB,
                .accessor_name = "llk_post_a",
                .endpoint_type = DFBEndpointType::PRODUCER});
            bindings.push_back(DFBBinding{
                .dfb_spec_name = LLK_POST_A_DFB,
                .accessor_name = "llk_post_a",
                .endpoint_type = DFBEndpointType::CONSUMER});
        }
        if (use_llk_bcast && bcast_on_b) {
            bindings.push_back(DFBBinding{
                .dfb_spec_name = LLK_POST_B_DFB,
                .accessor_name = "llk_post_b",
                .endpoint_type = DFBEndpointType::PRODUCER});
            bindings.push_back(DFBBinding{
                .dfb_spec_name = LLK_POST_B_DFB,
                .accessor_name = "llk_post_b",
                .endpoint_type = DFBEndpointType::CONSUMER});
        }
        k.dfb_bindings = bindings;
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
    if (has_group_2) {
        spec.kernels.push_back(make_compute_k(COMPUTE_K2));
    }

    // WorkUnitSpec: reader + writer + compute_g{1,2} over their core groups. Each DFB's producer
    // and consumer share the same WorkUnitSpec (local-DFB rule).
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
    // RTA emission, minus the buffer-address slots (now carried by the TensorBindings).
    // ----------------------------------------------------------------------------------------
    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = READER_K};
    KernelRunArgs writer_run{.kernel = WRITER_K};
    KernelRunArgs compute_run_1{.kernel = COMPUTE_K1};
    KernelRunArgs compute_run_2{.kernel = COMPUTE_K2};

    // Packed scalar value for the scalar-b path (mirrors the legacy pack_scalar_runtime_arg call).
    const uint32_t packed_scalar =
        b_present ? 0u : pack_scalar_runtime_arg(*operation_attributes.scalar, a.dtype(), /*is_quant_op=*/false);

    uint32_t current_block = 0;
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

        if (inputs_row_major) {
            // RM reader RTAs (legacy order, minus a/b addresses + page-size overrides).
            reader_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
                .node = core,
                .args = {
                    {"dst_num_tiles", c_num_tiles_core},
                    {"aD", aD},
                    {"aN", aN},
                    {"aC", aC},
                    {"aHt", aHt_r},
                    {"aND", aND},
                    {"bD", b_present ? bD : 1u},
                    {"bN", b_present ? bN : 1u},
                    {"bC", b_present ? bC : 1u},
                    {"bHt", b_present ? bHt_r : 1u},
                    {"bND", bND},
                    {"cHt", cHt_r},
                    {"cC", cC},
                    {"cND", cND},
                    {"current_block_start", current_block},
                    {"rows_per_tile", rm_num_rows_per_tile},
                    {"row_width_elements", rm_common_row_width_elements},
                    {"alignment_a", rm_a_alignment},
                    {"alignment_b", rm_b_alignment},
                    {"tiles_per_row", rm_tiles_per_row_width},
                    {"stride_size_bytes", rm_reader_stride_size_bytes},
                }});
            // RM writer RTAs.
            writer_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
                .node = core,
                .args = {
                    {"row_width_elements", rm_common_row_width_elements},
                    {"dst_num_tiles", c_num_tiles_core},
                    {"outD", cD},
                    {"outN", cN},
                    {"outC", cC},
                    {"outHt", cHt_r},
                    {"outND", cND},
                    {"current_block_start", current_block},
                    {"rows_per_tile", rm_num_rows_per_tile},
                    {"alignment", rm_c_alignment},
                    {"tiles_per_row", rm_tiles_per_row_width},
                    {"stride_size_bytes", rm_writer_stride_size_bytes},
                }});
            // RM compute RTA: num_tiles = c_num_tiles_core * tiles_per_row_width.
            auto& compute_run = core_group_1.contains(core) ? compute_run_1 : compute_run_2;
            compute_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
                .node = core, .args = {{"num_tiles", c_num_tiles_core * rm_tiles_per_row_width}}});

            start_tile_id += c_num_tiles_core;
            current_block += c_num_tiles_core;
            continue;
        }

        // Tile path reader RTAs (legacy slots, minus a/b base addresses now handled by ta::).
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
                    {"src_num_tiles", 0u},
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

        // Tile path writer RTAs (legacy slots, minus the c base address now handled by ta::; on the
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

        // Compute RTA. For the no-bcast / scalar paths, only num_tiles is read (= c_num_tiles_core).
        // For the simple-broadcast paths, tile_freq + tile_start come from
        // calculate_compute_kernel_args (legacy compute_runtime_args = {compute_tiles, freq, counter, ...};
        // the trailing scalar slot is dead for plain ADD/SUB/MUL and dropped).
        auto& compute_run = core_group_1.contains(core) ? compute_run_1 : compute_run_2;
        if (compute_has_freq_counter) {
            auto [freq, counter] = calculate_compute_kernel_args(sbt, start_tile_id, cHt, cWt);
            compute_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
                .node = core, .args = {{"num_tiles", c_num_tiles_core}, {"tile_freq", freq}, {"tile_start", counter}}});
        } else {
            compute_run.runtime_arg_values.push_back(
                KernelRunArgs::NodeRuntimeArgs{.node = core, .args = {{"num_tiles", c_num_tiles_core}}});
        }

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
