#include "mlir_interface_api.hpp"
#include "operations/eltwise/unary/common/unary_op_types.hpp"
#include "types_wrapper.hpp"

#include "tensor/types.hpp" // DataType, Lauout, StorageType
#include "tt_metal/impl/buffers/buffer_constants.hpp" // TensorMemoryLayout, ShardOrientation
#include "tt_metal/impl/buffers/buffer.hpp" // BufferType

#include "ttnn/operations/eltwise/binary/binary_constraints.hpp"
#include "ttnn/operations/eltwise/unary/unary_constraints.hpp"
#include "ttnn/operations/matmul/matmul_constraints.hpp"
#include "ttnn/operations/normalization/softmax/softmax_constraints.hpp"

namespace ttnn::mlir_interface
{

int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}

// check if layout is dram interleaved or l1 sharded, returns false otherwise
bool dummy_check(const std::string& tensor_memory_layout_str, const std::string& buffer_type_str) {
    auto tensor_memory_layout = ttnn::str_wrapper::to_tensor_memory_layout(tensor_memory_layout_str);
    auto buffer_type = ttnn::str_wrapper::to_buffer_type(buffer_type_str);
    if (!tensor_memory_layout.has_value() || !buffer_type.has_value()) {
        return false;
    }

    if (tensor_memory_layout.value() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED && buffer_type.value() == tt::tt_metal::BufferType::DRAM) {
        return true;
    } else if (tensor_memory_layout.value() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED && buffer_type.value() == tt::tt_metal::BufferType::L1) {
        return true;
    } else if (tensor_memory_layout.value() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED && buffer_type.value() == tt::tt_metal::BufferType::L1) {
        return true;
    } else if (tensor_memory_layout.value() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED && buffer_type.value() == tt::tt_metal::BufferType::L1) {
        return true;
    }

    return false;
}

bool does_binary_op_support_input_output_constraints(
    const std::vector<uint32_t>& _shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::vector<uint32_t>& _shape_b,
    const memory_config_tuple& _memory_config_b,
    const std::string& _data_type_b,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o) {


    auto shape_a = ttnn::vector_wrapper::to_shape(_shape_a);
    auto memory_config_a = ttnn::tuple_wrapper::to_memory_config(_memory_config_a);
    if (!memory_config_a.has_value()) {
        return false;
    }
    auto data_type_a = ttnn::str_wrapper::to_data_type(_data_type_a);
    if (!data_type_a.has_value()) {
        return false;
    }
    auto shape_b = ttnn::vector_wrapper::to_shape(_shape_b);
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

    auto builder = EltwiseOpConstraintsFactory::Make(ttnn::Shape(shape_a), memory_config_a.value(), ttnn::Shape(shape_b), memory_config_b.value(), memory_config_o.value());
    if (builder) {
        const auto op_constraints =
            (*builder)
                .setDataTypeA(data_type_a.value())
                .setDataTypeB(data_type_b.value())
                .setDataTypeO(data_type_o.value())
                .build_constraints();
        if (op_constraints.size() == 0) {
            return false;
        }
    } else {
        return false;
    }

    return true;
}

bool does_unary_op_support_input_output_constraints(
    const std::string _op_type,
    const std::vector<uint32_t>& _input_shape_a,
    const memory_config_tuple& _memory_config_a,
    const std::string& _data_type_a,
    const std::vector<uint32_t>& _input_shape_o,
    const memory_config_tuple& _memory_config_o,
    const std::string& _data_type_o) {

    auto shape_a = ttnn::vector_wrapper::to_shape(_input_shape_a);
    auto memory_config_a = ttnn::tuple_wrapper::to_memory_config(_memory_config_a);
    if (!memory_config_a.has_value()) {
        return false;
    }
    auto data_type_a = ttnn::str_wrapper::to_data_type(_data_type_a);
    if (!data_type_a.has_value()) {
        return false;
    }
    auto shape_o = ttnn::vector_wrapper::to_shape(_input_shape_o);
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

    auto builder = ttnn::operations::unary::UnaryOpConstraintsFactory::Make(op_type.value(), tt::ARCH::WORMHOLE_B0, ttnn::Shape(shape_a), memory_config_a.value(), ttnn::Shape(shape_o), memory_config_o.value());
    if (builder)
    {
        const auto op_constraints =
            (*builder)
                .setDataTypeA(data_type_a.value())
                .setDataTypeO(data_type_o.value())
                .build_constraints();
        if (op_constraints.size() == 0)
        {
            return false;
        }
    }
    else
    {
        return false;
    }

    return true;
}


} // namespace ttnn_mlir_interface
