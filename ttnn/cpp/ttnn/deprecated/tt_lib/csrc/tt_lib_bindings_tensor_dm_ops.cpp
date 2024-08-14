// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_lib_bindings_tensor.hpp"
#include "tt_lib_bindings_tensor_impl.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/move/move_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/reshape/reshape_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/bcast/bcast_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/copy/copy_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/sharded/sharded_op.hpp"

namespace tt::tt_metal::detail{

    void TensorModuleDMOPs( py::module & m_tensor)
    {
        // bcast enums
        detail::export_enum<BcastOpMath>(m_tensor);
        /** TODO: add these to bcast ops - good to have not required
            .value("GT", BcastOpMath::GT)
            .value("LT", BcastOpMath::LT)
            .value("GE", BcastOpMath::GE)
            .value("LE", BcastOpMath::LE)
            .value("EQ", BcastOpMath::EQ)
            .value("NEQ", BcastOpMath::NE);
        */

        detail::export_enum<BcastOpDim>(m_tensor);

        detail::bind_unary_op<true, true>(m_tensor, "clone", &clone, R"doc(  Returns a new tensor which is a new copy of input tensor ``{0}``.)doc");
        detail::bind_binary_op<false, false, false, false>(m_tensor, "copy", &copy, R"doc(  Copies the elements from ``{0}`` into ``{1}``. ``{1}`` is modified in place.)doc");

        // *** tensor manipulation ***
        m_tensor.def("typecast", &typecast,
            py::arg("input_tensors").noconvert(), py::arg("dtype"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
                Returns a new tensor which is a typecast of input tensor with new datatype``{0}``.

                Input tensors must be on device, in ROW MAJOR or TILE layout, and have matching data type.

                Datatype must be one of the following types BFLOAT16, BFLOAT8_B, BFLOAT4_B, UINT32, INT32, UINT16 and UINT8.

                Output tensor will be on device, in same layout, and have the given data type.

                .. csv-table::
                    :header: "Argument", "Description", "Data type", "Required"

                    "input_tensors", "Input tensors to typecast", "List of Tensors", "Yes"
                    "dtype", "datatype of typecast", "Datatype", "Yes"
                    "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "No"
            )doc"
        );

        m_tensor.def("reshape", &reshape,
            py::arg("input").noconvert(), py::arg("W"), py::arg("Z"), py::arg("Y"), py::arg("X"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns a tensor with the new shape of ``[W, Z, Y, X]``. The X dimension of input and output tensor must have same size.

            Input tensor must be on host device, in TILE layout, and have BFLOAT16 data type.

            Output tensor will be on host device, in TILE layout, and have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Input tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "W", "W dim of output tensor", "int", "", "Yes"
                "Z", "Z dim of output tensor", "int", "", "Yes"
                "Y", "Y dim of output tensor", "int", "", "Yes"
                "X", "X dim of output tensor", "int", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        // *** broadcast and reduce ***
        m_tensor.def("bcast",
        [](const Tensor &input_tensor_a,
            const Tensor &input_tensor_b,
            BcastOpMath bcast_op,
            BcastOpDim bcast_dim,
            const MemoryConfig &mem_config,
            std::optional<Tensor> output_tensor,
            uint8_t queue_id){
            return bcast(queue_id, input_tensor_a, input_tensor_b, bcast_op, bcast_dim, mem_config, output_tensor);
        },
            py::arg("input_a").noconvert(),
            py::arg("input_b").noconvert(),
            py::arg("math_op"),
            py::arg("dim"),
            py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            py::arg("output_tensor").noconvert() = std::nullopt,
            py::arg("queue_id").noconvert() = 0,
            R"doc(
            Perform a binary elementwise operation ``math_op`` between tensors ``input_a`` and ``input_b``, where values from tensor ``input_b`` are broadcast.

            Let tensor ``input_a`` have shape ``[W0, Z0, Y0, X0]`` and tensor ``input_b`` shape ``[W1, Z1, Y1, X1]``. ``dim`` determines the type of broadcast performed.

            For ``dim=BcastOpDim::W`` broadcast is performed on dimension ``X``. ``Y0`` and ``Y1`` must be the same and either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1).

            For ``dim=BcastOpDim::H`` broadcast is performed on dimension  ``Y``. ``X0`` and ``X1`` must be the same and either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1).

            For ``dim=BcastOpDim::HW`` broadcast is performed on dimensions ``X`` and ``Y``. Either (W1=1 and Z1=1) or (W0=W1 and Z0=Z1) must hold for input shapes.

            Both input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input_a", "Input tensor", "Tensor", "Tensor of shape [W0, Z0, Y0, X0]", "Yes"
                "input_b", "Input tensor to broadcast", "Tensor", "Tensor of shape [W1, Z1, Y1, X1]", "Yes"
                "math_op", "Aggregating math operation", " BcastOpMath", "ADD, SUB, MUL", "Yes"
                "dim", "Dimension on which to broadcast", "BcastOpDim", "W, H, HW", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
                "output_tensor", "Optional output tensor", "Tensor", "Default is None", "No"
                "queue_id", "command queue id", "uint8_t", "Default is 0", "No"
        )doc");

        m_tensor.def("move", &move,
            py::arg().noconvert(), py::arg("output_mem_config").noconvert() = std::nullopt, R"doc(
            Moves the elements of the input tensor ``arg0`` to a location in memory with specified memory layout.

            If no memory layout is specified, output memory will be the same as the input tensor memory config.

            +----------+----------------------------+----------------------------+---------------------------------+----------+
            | Argument | Description                | Data type                  | Valid range                     | Required |
            +==========+============================+============================+=================================+==========+
            | arg0     | Tensor to move             | Tensor                     | Tensor of shape [W, Z, Y, X]    | Yes      |
            +----------+----------------------------+----------------------------+---------------------------------+----------+
            | arg1     | MemoryConfig of tensor of  | tt_lib.tensor.MemoryConfig | Default is same as input tensor | No       |
            |          | TT accelerator device      |                            |                                 |          |
            +----------+----------------------------+----------------------------+---------------------------------+----------+
        )doc");

        m_tensor.def("move_sharded", &move_sharded,
            py::arg().noconvert(), py::arg("output_mem_config").noconvert() = std::nullopt, R"doc(
            Moves the elements of the sharded input tensor ``arg0`` to a location in local L1.

            +----------+----------------------------+----------------------------+---------------------------------+----------+
            | Argument | Description                | Data type                  | Valid range                     | Required |
            +==========+============================+============================+=================================+==========+
            | arg0     | Tensor to move             | Tensor                     | Tensor of shape [W, Z, Y, X]    | Yes      |
            +----------+----------------------------+----------------------------+---------------------------------+----------+
            | arg1     | MemoryConfig of tensor of  | tt_lib.tensor.MemoryConfig | Default is same as input tensor | No       |
            |          | TT accelerator device      |                            |                                 |          |
            +----------+----------------------------+----------------------------+---------------------------------+----------+
        )doc");

        // Sharding ops
        m_tensor.def("interleaved_to_sharded", py::overload_cast<const Tensor&, const std::variant<CoreCoord, CoreRangeSet>,  std::array<uint32_t, 2>, const TensorMemoryLayout, const ShardOrientation, const std::optional<const DataType>>(&interleaved_to_sharded),
            py::arg("input"), py::arg("grid"), py::arg("shard_shape"), py::arg("shard_scheme").noconvert(), py::arg("shard_layout").noconvert(), py::arg("output_dtype").noconvert() = std::nullopt,
            R"doc(Converts tensor from interleaved to sharded memory layout)doc"
        );
        m_tensor.def("interleaved_to_sharded", py::overload_cast<const Tensor&, const MemoryConfig &, const std::optional<const DataType>>(&interleaved_to_sharded),
            py::arg("input"), py::arg("sharded_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt,
            R"doc(Converts tensor from interleaved to sharded memory layout)doc"
        );
        m_tensor.def("sharded_to_interleaved", &sharded_to_interleaved,
            py::arg("input"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt,
            R"doc(Converts tensor from sharded_to_interleaved memory layout)doc"
        );
        m_tensor.def("reshard", &reshard,
            py::arg("input"), py::arg("output_mem_config").noconvert(), py::arg("output_tensor").noconvert() = std::nullopt,
            R"doc(Converts a tensor sharded one way to another way)doc"
        );
    }

}
