#include "mlir_interface_api.hpp"

#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <tuple>

#include "tensor/types.hpp"  // DataType, Lauout, StorageType
#include "tt_metal/common/logger.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"            // BufferType
#include "tt_metal/impl/buffers/buffer_constants.hpp"  // TensorMemoryLayout, ShardOrientation
#include "ttnn/mlir_interface_graph_capture_utils.hpp"
#include "ttnn/mlir_l1_interface.hpp"
#include "ttnn/operations/common/l1_interface_common.hpp"
#include "ttnn/operations/eltwise/binary/binary_constraints.hpp"
#include "ttnn/operations/eltwise/binary/binary_l1_interface.hpp"
#include "ttnn/operations/eltwise/unary/unary_constraints.hpp"
#include "ttnn/operations/eltwise/unary/unary_l1_interface.hpp"
#include "ttnn/operations/matmul/device/matmul_types.hpp"
#include "ttnn/operations/matmul/matmul_constraints.hpp"
#include "ttnn/operations/matmul/matmul_l1_interface.hpp"
#include "ttnn/operations/normalization/softmax/softmax_constraints.hpp"
#include "ttnn/operations/normalization/softmax/softmax_l1_interface.hpp"
#include "types_wrapper.hpp"

namespace ttnn::mlir_interface {

template <typename T>
static bool extract_graph_capture_constraints_check(const T& call) {
    try {
        call();
        return true;
    } catch (const std::exception& ex) {
        tt::log_info("GRAPH_CAPTURE_TRACE_EXCEPTION {}", ex.what());
        return false;
    }
}

// ============= OP constraints API ==============
bool does_binary_op_support_input_output_constraints(
    const std::vector<uint32_t>& _shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::vector<uint32_t>& _shape_b,
    const memory_config_tuple& _memory_config_b,
    const std::string& _data_type_b,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o) {
    auto shape_a = ttnn::types::Shape(ttnn::vector_wrapper::to_shape(_shape_a));
    auto memory_config_a = ttnn::tuple_wrapper::to_memory_config(_memory_config_a);
    if (!memory_config_a.has_value()) {
        return false;
    }
    auto data_type_a = ttnn::str_wrapper::to_data_type(_data_type_a);
    if (!data_type_a.has_value()) {
        return false;
    }
    auto shape_b = ttnn::types::Shape(ttnn::vector_wrapper::to_shape(_shape_b));
    auto memory_config_b = ttnn::tuple_wrapper::to_memory_config(_memory_config_b);
    if (!memory_config_b.has_value()) {
        return false;
    }
    auto data_type_b = ttnn::str_wrapper::to_data_type(_data_type_b);
    if (!data_type_b.has_value()) {
        return false;
    }
    auto memory_config_o = ttnn::tuple_wrapper::to_memory_config(_memory_config_o);
    if (!memory_config_o.has_value()) {
        return false;
    }
    auto data_type_o = ttnn::str_wrapper::to_data_type(_data_type_o);
    if (!data_type_o.has_value()) {
        return false;
    }

    // TODO: Extract to a separate classes with appropriate factories
    if (graph_capture::is_graph_capture_mode_enabled()) {
        return extract_graph_capture_constraints_check([&]() {
            graph_capture::get_binary_op_trace(
                shape_a,
                data_type_a.value(),
                tt::tt_metal::Layout::TILE,
                memory_config_a.value(),
                shape_b,
                data_type_b.value(),
                tt::tt_metal::Layout::TILE,
                memory_config_b.value(),
                data_type_o.value(),
                memory_config_o.value());
        });
    } else {
        auto builder = EltwiseOpConstraintsFactory::Make(
            shape_a,
            memory_config_a.value(),
            shape_b,
            memory_config_b.value(),
            memory_config_o.value(),
            CoreCoord{8, 8});
        return builder && !(*builder)
                               .setDataTypeA(data_type_a.value())
                               .setDataTypeB(data_type_b.value())
                               .setDataTypeO(data_type_o.value())
                               .build_constraints()
                               .empty();
    }
}

bool does_unary_op_support_input_output_constraints(
    const std::string _op_type,
    const std::vector<uint32_t>& _input_shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::vector<uint32_t>& _input_shape_o,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o) {
    auto shape_a = ttnn::types::Shape(ttnn::vector_wrapper::to_shape(_input_shape_a));
    auto memory_config_a = ttnn::tuple_wrapper::to_memory_config(_memory_config_a);
    if (!memory_config_a.has_value()) {
        return false;
    }
    auto data_type_a = ttnn::str_wrapper::to_data_type(_data_type_a);
    if (!data_type_a.has_value()) {
        return false;
    }
    auto shape_o = ttnn::types::Shape(ttnn::vector_wrapper::to_shape(_input_shape_o));
    auto memory_config_o = ttnn::tuple_wrapper::to_memory_config(_memory_config_o);
    if (!memory_config_o.has_value()) {
        return false;
    }
    auto data_type_o = ttnn::str_wrapper::to_data_type(_data_type_o);
    if (!data_type_o.has_value()) {
        return false;
    }

    auto op_type = ttnn::str_wrapper::to_unary_op_type(_op_type);
    if (!op_type.has_value()) {
        return false;
    }

    // ignoring is_supported_arch for now.
    // because it's GS specific, and we dont care about GS today.
    if (graph_capture::is_graph_capture_mode_enabled()) {
        return extract_graph_capture_constraints_check([&]() {
            graph_capture::get_unary_op_trace(
                shape_a,
                data_type_a.value(),
                tt::tt_metal::Layout::TILE,
                memory_config_a.value(),
                memory_config_o.value());
        });
    } else {
        auto builder = ttnn::operations::unary::UnaryOpConstraintsFactory::Make(
            op_type.value(),
            tt::ARCH::WORMHOLE_B0,
            shape_a,
            memory_config_a.value(),
            shape_o,
            memory_config_o.value(),
            CoreCoord{8, 8});
        return builder && !(*builder)
                               .setDataTypeA(data_type_a.value())
                               .setDataTypeO(data_type_o.value())
                               .build_constraints()
                               .empty();
    }
}

bool does_softmax_op_support_input_output_constraints(
    const std::vector<uint32_t>& _input_shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::vector<uint32_t>& _input_shape_o,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o) {
    auto shape_a = ttnn::types::Shape(ttnn::vector_wrapper::to_shape(_input_shape_a));
    auto memory_config_a = ttnn::tuple_wrapper::to_memory_config(_memory_config_a);
    if (!memory_config_a.has_value()) {
        return false;
    }
    auto data_type_a = ttnn::str_wrapper::to_data_type(_data_type_a);
    if (!data_type_a.has_value()) {
        return false;
    }
    auto shape_o = ttnn::types::Shape(ttnn::vector_wrapper::to_shape(_input_shape_o));
    auto memory_config_o = ttnn::tuple_wrapper::to_memory_config(_memory_config_o);
    if (!memory_config_o.has_value()) {
        return false;
    }
    auto data_type_o = ttnn::str_wrapper::to_data_type(_data_type_o);
    if (!data_type_o.has_value()) {
        return false;
    }

    if (graph_capture::is_graph_capture_mode_enabled()) {
        return extract_graph_capture_constraints_check([&]() {
            graph_capture::get_softmax_op_trace(
                shape_a,
                data_type_a.value(),
                tt::tt_metal::Layout::TILE,
                memory_config_a.value(),
                -1 /* dim_arg */,
                memory_config_o.value());
        });
    } else {
        auto builder = SoftmaxOpConstraintsFactory::Make(
            shape_a, memory_config_a.value(), shape_o, memory_config_o.value(), CoreCoord{8, 8});
        return builder && !(*builder)
                               .setDataTypeA(data_type_a.value())
                               .setDataTypeO(data_type_o.value())
                               .build_constraints()
                               .empty();
    }
}

bool does_matmul_multicore_reuse_multicast_support_input_output_constraints(
    const std::vector<uint32_t>& _shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::vector<uint32_t>& _shape_b,
    const memory_config_tuple& _memory_config_b,
    const std::string& _data_type_b,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o,
    const matmul_multicore_reuse_config_tuple _matmul_config,
    const bool transpose_mcast,
    const bool fuse_batch) {
    auto shape_a = ttnn::types::Shape(ttnn::vector_wrapper::to_shape(_shape_a));
    auto memory_config_a = ttnn::tuple_wrapper::to_memory_config(_memory_config_a);
    if (!memory_config_a.has_value()) {
        return false;
    }
    auto data_type_a = ttnn::str_wrapper::to_data_type(_data_type_a);
    if (!data_type_a.has_value()) {
        return false;
    }
    auto shape_b = ttnn::types::Shape(ttnn::vector_wrapper::to_shape(_shape_b));
    auto memory_config_b = ttnn::tuple_wrapper::to_memory_config(_memory_config_b);
    if (!memory_config_b.has_value()) {
        return false;
    }
    auto data_type_b = ttnn::str_wrapper::to_data_type(_data_type_b);
    if (!data_type_b.has_value()) {
        return false;
    }
    auto memory_config_o = ttnn::tuple_wrapper::to_memory_config(_memory_config_o);
    if (!memory_config_o.has_value()) {
        return false;
    }
    auto data_type_o = ttnn::str_wrapper::to_data_type(_data_type_o);
    if (!data_type_o.has_value()) {
        return false;
    }

    auto matmul_config =
        ttnn::tuple_wrapper::to_multicast_matmul_program_config(_matmul_config, transpose_mcast, fuse_batch);

    if (graph_capture::is_graph_capture_mode_enabled()) {
        return extract_graph_capture_constraints_check([&]() {
            graph_capture::get_matmul_op_trace(
                shape_a,
                data_type_a.value(),
                tt::tt_metal::Layout::TILE,
                memory_config_a.value(),
                shape_b,
                data_type_b.value(),
                tt::tt_metal::Layout::TILE,
                memory_config_b.value(),
                data_type_o.value(),
                memory_config_o.value(),
                matmul_config.value());
        });
    } else {
        auto builder = MatmulOpConstraintsFactory::Make(
            shape_a,
            memory_config_a.value(),
            shape_b,
            memory_config_b.value(),
            memory_config_o.value(),
            matmul_config.value(),
            CoreCoord{8, 8});
        return builder && !(*builder)
                               .setDataTypeA(data_type_a.value())
                               .setDataTypeB(data_type_b.value())
                               .setDataTypeO(data_type_o.value())
                               .build_constraints()
                               .empty();
    }
}

bool does_matmul_multicore_reuse_multicast_1d_op_support_input_output_constraints(
    const std::vector<uint32_t>& _shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::vector<uint32_t>& _shape_b,
    const memory_config_tuple& _memory_config_b,
    const std::string& _data_type_b,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o,
    const matmul_multicore_reuse_config_tuple _matmul_config,
    const bool fuse_batch,
    const bool mcast_in0) {
    auto shape_a = ttnn::types::Shape(ttnn::vector_wrapper::to_shape(_shape_a));
    auto memory_config_a = ttnn::tuple_wrapper::to_memory_config(_memory_config_a);
    if (!memory_config_a.has_value()) {
        return false;
    }
    auto data_type_a = ttnn::str_wrapper::to_data_type(_data_type_a);
    if (!data_type_a.has_value()) {
        return false;
    }
    auto shape_b = ttnn::types::Shape(ttnn::vector_wrapper::to_shape(_shape_b));
    auto memory_config_b = ttnn::tuple_wrapper::to_memory_config(_memory_config_b);
    if (!memory_config_b.has_value()) {
        return false;
    }
    auto data_type_b = ttnn::str_wrapper::to_data_type(_data_type_b);
    if (!data_type_b.has_value()) {
        return false;
    }
    auto memory_config_o = ttnn::tuple_wrapper::to_memory_config(_memory_config_o);
    if (!memory_config_o.has_value()) {
        return false;
    }
    auto data_type_o = ttnn::str_wrapper::to_data_type(_data_type_o);
    if (!data_type_o.has_value()) {
        return false;
    }

    auto matmul_config =
        ttnn::tuple_wrapper::to_multicast_1d_matmul_program_config(_matmul_config, fuse_batch, mcast_in0);

    if (graph_capture::is_graph_capture_mode_enabled()) {
        return extract_graph_capture_constraints_check([&]() {
            graph_capture::get_matmul_op_trace(
                shape_a,
                data_type_a.value(),
                tt::tt_metal::Layout::TILE,
                memory_config_a.value(),
                shape_b,
                data_type_b.value(),
                tt::tt_metal::Layout::TILE,
                memory_config_b.value(),
                data_type_o.value(),
                memory_config_o.value(),
                matmul_config.value());
        });
    } else {
        auto builder = MatmulOpConstraintsFactory::Make(
            shape_a,
            memory_config_a.value(),
            shape_b,
            memory_config_b.value(),
            memory_config_o.value(),
            matmul_config.value(),
            CoreCoord{8, 8});
        return builder && !(*builder)
                               .setDataTypeA(data_type_a.value())
                               .setDataTypeB(data_type_b.value())
                               .setDataTypeO(data_type_o.value())
                               .build_constraints()
                               .empty();
    }
}

// ============= L1 interface API ==============

static std::optional<L1InterfaceOperandParams> get_l1_interface_operand_params(
    const std::vector<uint32_t>& _shape,
    const std::string& _data_type,
    const std::string& _layout,
    const memory_config_tuple& _memory_config) {
    auto shape = ttnn::vector_wrapper::to_shape(_shape);

    auto data_type = ttnn::str_wrapper::to_data_type(_data_type);
    if (!data_type.has_value()) {
        return std::nullopt;
    }

    auto layout = ttnn::str_wrapper::to_layout(_layout);
    if (!layout.has_value()) {
        return std::nullopt;
    }

    auto memory_config = ttnn::tuple_wrapper::to_memory_config(_memory_config);
    if (!memory_config.has_value()) {
        return std::nullopt;
    }

    return std::make_tuple(ttnn::Shape(shape), data_type.value(), layout.value(), memory_config.value());
}

static std::vector<uint32_t> extract_cb_allocations(
    const std::vector<std::tuple<uint32_t, uint32_t>>& l1_interface_output) {
    std::vector<uint32_t> circular_buffer_allocations;
    for (const auto& [cb_size, num_cores] : l1_interface_output) {
        circular_buffer_allocations.push_back(cb_size == c_cb_shares_space_with_sharded_operand ? 0 : cb_size);
    }

    return circular_buffer_allocations;
}

static std::unique_ptr<EltwiseOpL1Usage> get_binary_l1_usage_estimator(
    const std::vector<uint32_t>& _shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::string& _layout_a,
    const std::vector<uint32_t>& _shape_b,
    const memory_config_tuple& _memory_config_b,
    const std::string& _data_type_b,
    const std::string& _layout_b,
    const std::vector<uint32_t>& _shape_o,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o,
    const std::string& _layout_o) {
    auto l1_input_a = get_l1_interface_operand_params(_shape_a, _data_type_a, _layout_a, _memory_config_a);
    if (!l1_input_a.has_value()) {
        return nullptr;
    }

    auto l1_input_b = get_l1_interface_operand_params(_shape_b, _data_type_b, _layout_b, _memory_config_b);
    if (!l1_input_b.has_value()) {
        return nullptr;
    }

    auto l1_output = get_l1_interface_operand_params(_shape_o, _data_type_o, _layout_o, _memory_config_o);
    if (!l1_output.has_value()) {
        return nullptr;
    }

    return OpL1UsageAbstractFactory::Make()->get_eltwise_op_l1_usage(
        l1_input_a.value(), l1_input_b.value(), l1_output.value());
}

std::unique_ptr<UnaryOpL1Usage> get_unary_l1_usage_estimator(
    const std::vector<uint32_t>& _shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::string& _layout_a,
    const std::vector<uint32_t>& _shape_o,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o,
    const std::string& _layout_o) {
    auto l1_input_a = get_l1_interface_operand_params(_shape_a, _data_type_a, _layout_a, _memory_config_a);
    if (!l1_input_a.has_value()) {
        return nullptr;
    }

    auto l1_output = get_l1_interface_operand_params(_shape_o, _data_type_o, _layout_o, _memory_config_o);
    if (!l1_output.has_value()) {
        return nullptr;
    }

    return OpL1UsageAbstractFactory::Make()->get_unary_op_l1_usage(l1_input_a.value(), l1_output);
}

static std::unique_ptr<SoftmaxOpL1Usage> get_softmax_l1_usage_estimator(
    const std::vector<uint32_t>& _shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::string& _layout_a,
    const std::vector<uint32_t>& _shape_o,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o,
    const std::string& _layout_o,
    const int dim_arg) {
    auto l1_input_a = get_l1_interface_operand_params(_shape_a, _data_type_a, _layout_a, _memory_config_a);
    if (!l1_input_a.has_value()) {
        return nullptr;
    }

    auto l1_output = get_l1_interface_operand_params(_shape_o, _data_type_o, _layout_o, _memory_config_o);
    if (!l1_output.has_value()) {
        return nullptr;
    }

    return OpL1UsageAbstractFactory::Make()->get_softmax_op_l1_usage(l1_input_a.value(), dim_arg, l1_output);
}

static std::unique_ptr<MatmulOpL1Usage> get_matmul_l1_usage_estimator(
    const std::vector<uint32_t>& _shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::string& _layout_a,
    const std::vector<uint32_t>& _shape_b,
    const memory_config_tuple& _memory_config_b,
    const std::string& _data_type_b,
    const std::string& _layout_b,
    const std::vector<uint32_t>& _shape_o,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o,
    const std::string& _layout_o,
    const operations::matmul::MatmulProgramConfig& matmul_config) {
    auto l1_input_a = get_l1_interface_operand_params(_shape_a, _data_type_a, _layout_a, _memory_config_a);
    if (!l1_input_a.has_value()) {
        return nullptr;
    }

    auto l1_input_b = get_l1_interface_operand_params(_shape_b, _data_type_b, _layout_b, _memory_config_b);
    if (!l1_input_b.has_value()) {
        return nullptr;
    }

    auto l1_output = get_l1_interface_operand_params(_shape_o, _data_type_o, _layout_o, _memory_config_o);
    if (!l1_output.has_value()) {
        return nullptr;
    }

    return OpL1UsageAbstractFactory::Make()->get_matmul_op_l1_usage(
        l1_input_a.value(), l1_input_b.value(), l1_output.value(), matmul_config);
}

static std::optional<std::vector<uint32_t>> get_matmul_circular_buffers_l1_allocations_helper(
    const std::vector<uint32_t>& _shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::string& _layout_a,
    const std::vector<uint32_t>& _shape_b,
    const memory_config_tuple& _memory_config_b,
    const std::string& _data_type_b,
    const std::string& _layout_b,
    const std::vector<uint32_t>& _shape_o,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o,
    const std::string& _layout_o,
    const operations::matmul::MatmulProgramConfig& matmul_config) {
    auto l1_usage = get_matmul_l1_usage_estimator(
        _shape_a,
        _memory_config_a,
        _data_type_a,
        _layout_a,
        _shape_b,
        _memory_config_b,
        _data_type_b,
        _layout_b,
        _shape_o,
        _memory_config_o,
        _data_type_o,
        _layout_o,
        matmul_config);
    if (!l1_usage) {
        return std::nullopt;
    }

    return extract_cb_allocations(l1_usage->get_circular_buffer_l1_allocations_per_core());
}

static std::optional<std::vector<std::tuple<uint32_t, uint32_t>>> get_matmul_tensor_buffers_l1_allocations_helper(
    const std::vector<uint32_t>& _shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::string& _layout_a,
    const std::vector<uint32_t>& _shape_b,
    const memory_config_tuple& _memory_config_b,
    const std::string& _data_type_b,
    const std::string& _layout_b,
    const std::vector<uint32_t>& _shape_o,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o,
    const std::string& _layout_o,
    const operations::matmul::MatmulProgramConfig& matmul_config) {
    auto l1_usage = get_matmul_l1_usage_estimator(
        _shape_a,
        _memory_config_a,
        _data_type_a,
        _layout_a,
        _shape_b,
        _memory_config_b,
        _data_type_b,
        _layout_b,
        _shape_o,
        _memory_config_o,
        _data_type_o,
        _layout_o,
        matmul_config);
    if (!l1_usage) {
        return std::nullopt;
    }

    return l1_usage->get_tensor_l1_allocations_per_core();
}

std::optional<std::vector<uint32_t>> get_binary_circular_buffers_l1_allocations(
    const std::vector<uint32_t>& _shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::string& _layout_a,
    const std::vector<uint32_t>& _shape_b,
    const memory_config_tuple& _memory_config_b,
    const std::string& _data_type_b,
    const std::string& _layout_b,
    const std::vector<uint32_t>& _shape_o,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o,
    const std::string& _layout_o) {
    auto l1_usage = get_binary_l1_usage_estimator(
        _shape_a,
        _memory_config_a,
        _data_type_a,
        _layout_a,
        _shape_b,
        _memory_config_b,
        _data_type_b,
        _layout_b,
        _shape_o,
        _memory_config_o,
        _data_type_o,
        _layout_o);
    if (!l1_usage) {
        return std::nullopt;
    }

    return extract_cb_allocations(l1_usage->get_circular_buffer_l1_allocations_per_core());
}

std::optional<std::vector<std::tuple<uint32_t, uint32_t>>> get_binary_tensor_buffers_l1_allocations(
    const std::vector<uint32_t>& _shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::string& _layout_a,
    const std::vector<uint32_t>& _shape_b,
    const memory_config_tuple& _memory_config_b,
    const std::string& _data_type_b,
    const std::string& _layout_b,
    const std::vector<uint32_t>& _shape_o,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o,
    const std::string& _layout_o) {
    auto l1_usage = get_binary_l1_usage_estimator(
        _shape_a,
        _memory_config_a,
        _data_type_a,
        _layout_a,
        _shape_b,
        _memory_config_b,
        _data_type_b,
        _layout_b,
        _shape_o,
        _memory_config_o,
        _data_type_o,
        _layout_o);
    if (!l1_usage) {
        return std::nullopt;
    }

    return l1_usage->get_tensor_l1_allocations_per_core();
}

std::optional<std::vector<uint32_t>> get_unary_circular_buffers_l1_allocations(
    const std::vector<uint32_t>& _shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::string& _layout_a,
    const std::vector<uint32_t>& _shape_o,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o,
    const std::string& _layout_o) {
    auto l1_usage = get_unary_l1_usage_estimator(
        _shape_a, _memory_config_a, _data_type_a, _layout_a, _shape_o, _memory_config_o, _data_type_o, _layout_o);
    if (!l1_usage) {
        return std::nullopt;
    }

    return extract_cb_allocations(l1_usage->get_circular_buffer_l1_allocations_per_core());
}

std::optional<std::vector<std::tuple<uint32_t, uint32_t>>> get_unary_tensor_buffers_l1_allocations(
    const std::vector<uint32_t>& _shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::string& _layout_a,
    const std::vector<uint32_t>& _shape_o,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o,
    const std::string& _layout_o) {
    auto l1_usage = get_unary_l1_usage_estimator(
        _shape_a, _memory_config_a, _data_type_a, _layout_a, _shape_o, _memory_config_o, _data_type_o, _layout_o);
    if (!l1_usage) {
        return std::nullopt;
    }

    return l1_usage->get_tensor_l1_allocations_per_core();
}

std::optional<std::vector<uint32_t>> get_softmax_circular_buffers_l1_allocations(
    const std::vector<uint32_t>& _shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::string& _layout_a,
    const std::vector<uint32_t>& _shape_o,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o,
    const std::string& _layout_o,
    const int dim_arg) {
    auto l1_usage = get_softmax_l1_usage_estimator(
        _shape_a,
        _memory_config_a,
        _data_type_a,
        _layout_a,
        _shape_o,
        _memory_config_o,
        _data_type_o,
        _layout_o,
        dim_arg);
    if (!l1_usage) {
        return std::nullopt;
    }

    return extract_cb_allocations(l1_usage->get_circular_buffer_l1_allocations_per_core());
}

std::optional<std::vector<std::tuple<uint32_t, uint32_t>>> get_softmax_tensor_buffers_l1_allocations(
    const std::vector<uint32_t>& _shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::string& _layout_a,
    const std::vector<uint32_t>& _shape_o,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o,
    const std::string& _layout_o,
    const int dim_arg) {
    auto l1_usage = get_softmax_l1_usage_estimator(
        _shape_a,
        _memory_config_a,
        _data_type_a,
        _layout_a,
        _shape_o,
        _memory_config_o,
        _data_type_o,
        _layout_o,
        dim_arg);
    if (!l1_usage) {
        return std::nullopt;
    }

    return l1_usage->get_tensor_l1_allocations_per_core();
}

std::optional<std::vector<uint32_t>> get_matmul_multicore_reuse_multicast_circular_buffers_l1_allocations(
    const std::vector<uint32_t>& _shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::string& _layout_a,
    const std::vector<uint32_t>& _shape_b,
    const memory_config_tuple& _memory_config_b,
    const std::string& _data_type_b,
    const std::string& _layout_b,
    const std::vector<uint32_t>& _shape_o,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o,
    const std::string& _layout_o,
    const matmul_multicore_reuse_config_tuple _matmul_config,
    const bool transpose_mcast,
    const bool fuse_batch) {
    auto matmul_config =
        ttnn::tuple_wrapper::to_multicast_matmul_program_config(_matmul_config, transpose_mcast, fuse_batch);
    if (!matmul_config.has_value()) {
        return std::nullopt;
    }

    return get_matmul_circular_buffers_l1_allocations_helper(
        _shape_a,
        _memory_config_a,
        _data_type_a,
        _layout_a,
        _shape_b,
        _memory_config_b,
        _data_type_b,
        _layout_b,
        _shape_o,
        _memory_config_o,
        _data_type_o,
        _layout_o,
        matmul_config.value());
}

std::optional<std::vector<std::tuple<uint32_t, uint32_t>>>
get_matmul_multicore_reuse_multicast_tensor_buffers_l1_allocations(
    const std::vector<uint32_t>& _shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::string& _layout_a,
    const std::vector<uint32_t>& _shape_b,
    const memory_config_tuple& _memory_config_b,
    const std::string& _data_type_b,
    const std::string& _layout_b,
    const std::vector<uint32_t>& _shape_o,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o,
    const std::string& _layout_o,
    const matmul_multicore_reuse_config_tuple _matmul_config,
    const bool transpose_mcast,
    const bool fuse_batch) {
    auto matmul_config =
        ttnn::tuple_wrapper::to_multicast_matmul_program_config(_matmul_config, transpose_mcast, fuse_batch);
    if (!matmul_config.has_value()) {
        return std::nullopt;
    }

    return get_matmul_tensor_buffers_l1_allocations_helper(
        _shape_a,
        _memory_config_a,
        _data_type_a,
        _layout_a,
        _shape_b,
        _memory_config_b,
        _data_type_b,
        _layout_b,
        _shape_o,
        _memory_config_o,
        _data_type_o,
        _layout_o,
        matmul_config.value());
}

std::optional<std::vector<uint32_t>> get_matmul_multicore_reuse_multicast_1d_circular_buffers_l1_allocations(
    const std::vector<uint32_t>& _shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::string& _layout_a,
    const std::vector<uint32_t>& _shape_b,
    const memory_config_tuple& _memory_config_b,
    const std::string& _data_type_b,
    const std::string& _layout_b,
    const std::vector<uint32_t>& _shape_o,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o,
    const std::string& _layout_o,
    const matmul_multicore_reuse_config_tuple _matmul_config,
    const bool fuse_batch,
    const bool mcast_in0) {
    auto matmul_config =
        ttnn::tuple_wrapper::to_multicast_1d_matmul_program_config(_matmul_config, fuse_batch, mcast_in0);
    if (!matmul_config.has_value()) {
        return std::nullopt;
    }

    return get_matmul_circular_buffers_l1_allocations_helper(
        _shape_a,
        _memory_config_a,
        _data_type_a,
        _layout_a,
        _shape_b,
        _memory_config_b,
        _data_type_b,
        _layout_b,
        _shape_o,
        _memory_config_o,
        _data_type_o,
        _layout_o,
        matmul_config.value());
}

std::optional<std::vector<std::tuple<uint32_t, uint32_t>>>
get_matmul_multicore_reuse_multicast_1d_tensor_buffers_l1_allocations(
    const std::vector<uint32_t>& _shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::string& _layout_a,
    const std::vector<uint32_t>& _shape_b,
    const memory_config_tuple& _memory_config_b,
    const std::string& _data_type_b,
    const std::string& _layout_b,
    const std::vector<uint32_t>& _shape_o,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o,
    const std::string& _layout_o,
    const matmul_multicore_reuse_config_tuple _matmul_config,
    const bool fuse_batch,
    const bool mcast_in0) {
    auto matmul_config =
        ttnn::tuple_wrapper::to_multicast_1d_matmul_program_config(_matmul_config, fuse_batch, mcast_in0);
    if (!matmul_config.has_value()) {
        return std::nullopt;
    }

    return get_matmul_tensor_buffers_l1_allocations_helper(
        _shape_a,
        _memory_config_a,
        _data_type_a,
        _layout_a,
        _shape_b,
        _memory_config_b,
        _data_type_b,
        _layout_b,
        _shape_o,
        _memory_config_o,
        _data_type_o,
        _layout_o,
        matmul_config.value());
}

}  // namespace ttnn::mlir_interface
