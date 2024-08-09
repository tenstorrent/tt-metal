// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_lib_bindings_tensor.hpp"
#include "tt_lib_bindings_tensor_impl.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/move/move_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/reshape/reshape_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/fold/fold_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/bcast/bcast_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/reduce/reduce_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/copy/copy_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/indexed_fill/indexed_fill_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/sharded/sharded_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/sharded_partial/sharded_op_partial.hpp"


namespace tt::tt_metal::detail{

    void TensorModuleDMOPs( py::module & m_tensor)
    {

        // reduce enums
        detail::export_enum<ReduceOpMath>(m_tensor);

        detail::export_enum<ReduceOpDim>(m_tensor);

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
        detail::bind_unary_op<true, true>(m_tensor, "assign", py::overload_cast<const Tensor&, const MemoryConfig&, std::optional<const DataType>>(&assign), R"doc(  Returns a new tensor which is a new copy of input tensor ``{0}``.)doc");

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

        m_tensor.def("assign",
        [](const Tensor& input_a, const Tensor& input_b, uint8_t queue_id){
            return assign(queue_id, input_a, input_b); },
            py::arg("input_a").noconvert(),
            py::arg("input_b").noconvert(),
            py::arg("queue_id").noconvert() = 0,
            R"doc(
            Copies input tensor ``input_a`` to ``input_b`` if their
            shapes and memory layouts match, and returns input_b tensor.

            Input tensors can be of any data type.

            Output tensor will be of same data type as Input tensor.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input_a", "Tensor assign is applied to", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "input_b", "Input tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "queue_id", "command queue id", "uint8_t", "Default is 0", "No"
        )doc");

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

        m_tensor.def("fold", &fold,
            py::arg("input").noconvert(), py::arg("stride_h"), py::arg("stride_w"), py::arg("use_transpose_as_fold")=false, py::arg("output_shape")=std::nullopt, py::arg("pad_c")=0, py::arg("pad_h")=0, py::arg("pad_w")=0, R"doc(
            Fold TT Tensor.

            Input tensor must be on TT accelerator device, in ROW_MAJOR.

            Output tensor will be on TT accelerator device, in ROW_MAJOR.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Input tensor", "Tensor", "Tensor of shape [N, H, W, C]", "Yes"
                "stride_h", "Stride along the H-dimension", "int", "", "Yes"
                "stride_w", "Stride along the W-dimension", "int", "", "Yes"
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

        m_tensor.def("reduce", &reduce,
            py::arg("input").noconvert(), py::arg("math_op"), py::arg("dim"), py::arg("scaler"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt, py::arg("compute_kernel_config").noconvert() = std::nullopt, R"doc(
            Perform a reduction of input tensor ``input`` using mathematical operation ``math_op`` on dimension ``dim``.

            For ``arg2=ReduceOpDim::W`` reduce is done on dimension X.

            For ``arg2=ReduceOpDim::H`` reduce is done on dimension Y.

            For ``arg2=ReduceOpDim::HW`` reduce is done on dimensions X and Y.

            Input tensors must have BFLOAT16 data type.

            Output tensor will have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Input tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "math_op", "Aggregating math operation", " ReduceOpMath", "SUM, MAX, MIN", "Yes"
                "dim", "Dimension on which reduction is performed", "ReduceOpDim", "W, H, HW", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
                "output_dtype", "DataType of output tensor", "DataType", "Default is None (use input dtype)", "No"
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

        //MOE ops
        m_tensor.def("indexed_fill", &indexed_fill,
            py::arg("batch_id"), py::arg("input_a"), py::arg("input_b"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("dim") = 0,
            R"doc(Replaces batch of input in input_b denoted by batch_ids into input_a)doc"
        );


        m_tensor.def("interleaved_to_sharded_partial", py::overload_cast<const Tensor &, const std::variant<CoreCoord, CoreRangeSet>, std::array<uint32_t, 2>, const uint32_t, const uint32_t, const TensorMemoryLayout, const ShardOrientation, const std::optional<const DataType>>(&interleaved_to_sharded_partial),
         py::arg("input"), py::arg("grid"), py::arg("shard_shape"), py::arg("num_slices"), py::arg("slice_index"), py::arg("shard_scheme").noconvert(), py::arg("shard_layout").noconvert(), py::arg("output_dtype").noconvert() = std::nullopt,
            R"doc(Converts a part of tensor from interleaved to sharded memory layout)doc");

        m_tensor.def("sharded_to_interleaved_partial", &sharded_to_interleaved_partial,
            py::arg("input"), py::arg("cache_tensor"), py::arg("num_slices"), py::arg("slice_index"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt,
            R"doc(Converts a partial tensor from sharded_to_interleaved memory layout)doc"
        );

    }

}
