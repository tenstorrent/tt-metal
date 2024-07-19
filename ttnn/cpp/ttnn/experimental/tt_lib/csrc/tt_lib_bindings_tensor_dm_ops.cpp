// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_lib_bindings_tensor.hpp"
#include "tt_lib_bindings_tensor_impl.hpp"
#include "ttnn/experimental/tt_dnn/op_library/move/move_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/untilize/untilize_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/reshape/reshape_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/permute/permute_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/fold/fold_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/transpose/transpose_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/fill_rm/fill_rm_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/concat/concat_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/repeat/repeat_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/bcast/bcast_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/reduce/reduce_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/copy/copy_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/indexed_fill/indexed_fill_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/non_zero_indices/non_zero_indices_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/sharded/sharded_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/sharded_partial/sharded_op_partial.hpp"
#include "ttnn/experimental/tt_dnn/op_library/ccl/reduce_scatter/reduce_scatter_op.hpp"


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
        m_tensor.def("concat", &concat,
            py::arg("input_tensors").noconvert(), py::arg("dim") = 0, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Concatenates shape of tensors ``arg0`` and ``arg1`` to new shape ``[W, Z, Y, X]`` along the specified dimension ``arg1``.

            Input tensors must be on device, in ROW MAJOR or TILE layout, and have matching data type.

            Output tensor will be on device, in same layout, and have same data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input_tensors", "Input tensors to concat", "List of Tensors", "Tensors of shape [W, Z, Y, X], where Y or X must be a multiple of 32 if they are the concat dim", "Yes"
                "dim", "dimension of concat", "int", "", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("repeat", &tt::tt_metal::repeat,
            py::arg("input"), py::arg("size"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
                    Returns a new tensor filled with repetition of input ``input`` tensor according to number of times specified in ``size``. The rank of ``size`` should be less than or equal to the rank of tensor ``input_a``.

                    Output tensor will have same data type as input.

                    .. csv-table::
                        :header: "Argument", "Description", "Data type", "Valid range", "Required"

                        "input", "Input tensor for which repetition is computed", "Tensor", "Tensor of any shape", "Yes"
                        "size", "The number of times to repeat this tensor along each dimension", "List[Int]", "Positive repetition values", "Yes"
                        "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
                )doc");

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

        m_tensor.def("permute", &permute,
            py::arg("input").noconvert(), py::arg("dims"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Returns a tensor that is input tensor ``arg0`` with its dimensions permuted to new order ``dims``.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Input tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
                "dims", "The desired ordering of dimensions", "List[int]", "All indices within input tensor rank", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
        )doc");

        m_tensor.def("untilize", &untilize,
            py::arg("input").noconvert(),
            py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            py::arg("use_multicore").noconvert() = true,
            py::arg("use_pack_untilize").noconvert() = true,
            R"doc(
            Changes data layout of input tensor to ROW_MAJOR.

            Input tensor must be on TT accelerator device, in TILE, and have BFLOAT16 data type.

            Output tensor will be on TT accelerator device, in ROW_MAJOR layout, and have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Input tensor", "Tensor", "Tensor of shape [W, Z, Y, X] where Y%32=0 and X%32=0", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
                "use_multicore", "Whether to use multi-core parallelization", "bool", "Default is true", "No"
                "use_pack_untilize", "Whether to use pack untilize", "bool", "Default is true", "No"
        )doc");

        m_tensor.def(
            "untilize_with_halo_v2",
            &untilize_with_halo_v2,
            py::arg("input_tensor").noconvert(),
            py::arg("padding_config").noconvert(),
            py::arg("local_config").noconvert(),
            py::arg("remote_config").noconvert(),
            py::arg("pad_val").noconvert(),
            py::arg("ncores_height").noconvert(),
            py::arg("max_out_nsticks_per_core").noconvert(),
            py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            py::arg("remote_read").noconvert() = false,
            py::arg("transpose_mcast").noconvert() = true,
            R"doc(
                Untilizes input tiled data to row major format and constructs halo'd output shards.
            )doc");

        m_tensor.def("untilize_with_halo", &untilize_with_halo,
            py::arg("input").noconvert(),
            py::arg("pad_val"),
            py::arg("in_b").noconvert(),
            py::arg("in_h").noconvert(),
            py::arg("in_w").noconvert(),
            py::arg("stride") = 1,
            py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            R"doc(
                Untilizes input tiled data to row major format.
            )doc");

        m_tensor.def("untilize_with_unpadding", &untilize_with_unpadding,
            py::arg("input").noconvert(), py::arg("output_tensor_end"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            py::arg("use_multicore").noconvert() = false, py::arg("use_pack_untilize").noconvert() = true,
            R"doc(
            Changes data layout of input tensor to ROW_MAJOR and unpads/removes elements from the tensor.

            Input tensor must be on TT accelerator device, in TILE, and have BFLOAT16 data type.

            Output tensor will be on TT accelerator device, in ROW_MAJOR layout, and have BFLOAT16 data type.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "input", "Input tensor", "Tensor", "Tensor of shape [W, Z, Y, X] where Y%32=0 and X%32=0", "Yes"
                "output_tensor_end", "End indices of input tensor in output tensor", "List[int[4]]", "Values along each dim must be < input_tensor_shape[i]", "Yes"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
                "use_multicore", "Whether to use multi-core parallelization", "bool", "Default is false", "No"
                "use_pack_untilize", "Whether to use pack untilize", "bool", "Default is true", "No"
        )doc");

        m_tensor.def("fold", &fold,
            py::arg("input").noconvert(), py::arg("stride_h"), py::arg("stride_w"), R"doc(
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
            py::arg("input").noconvert(), py::arg("math_op"), py::arg("dim"), py::arg("scaler"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt, R"doc(
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

        // *** experimental operations ***
        m_tensor.def("fill_rm", &fill_rm,
            py::arg("N"), py::arg("C"), py::arg("H"), py::arg("W"), py::arg("hOnes"), py::arg("wOnes"), py::arg("any").noconvert(), py::arg("val_hi"), py::arg("val_lo"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Generates an NCHW row-major tensor and fill it with high values up to
            hOnes, wOnes in each HW tile with the rest padded with high values. So
            for H=2, W=3, hFill=1, wFill=2 the following tensor will be generated:

            .. code-block::

                +------------> W
                | hi hi lo
                | lo lo lo
                |
                v H

            H, W are expected to be multiples of 32.

            The 'any' Tensor arg is only used to pass the device and resulting
            tensor dtype.

            val_hi/lo are expected to be floats.

            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | Argument | Description                                                           | Data type             | Valid range            | Required |
            +==========+=======================================================================+=======================+========================+==========+
            | N        | Batch count of output tensor                                          | int                   | N > 0                  | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | C        | Channel count of output tensor                                        | int                   | C > 0                  | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | H        | Height count of output tensor                                         | int                   | H > 0                  | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | W        | Width count of output tensor                                          | int                   | W > 0                  | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | hOnes    | Height of high values region                                          | int                   | hOnes <= H             | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | wOnes    | Width of high values region                                           | int                   | wOnes <= W             | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | any      | Any input tensor with desired device and data types for output tensor | tt_lib.tensor.Tensor  |                        | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | val_hi   | High value to use                                                     | float                 |                        | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | val_lo   | Low value to use                                                      | float                 |                        | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
        )doc");
        m_tensor.def("fill_ones_rm", &fill_ones_rm,
            py::arg("N"), py::arg("C"), py::arg("H"), py::arg("W"), py::arg("hOnes"), py::arg("wOnes"), py::arg("any").noconvert(), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Same as ``fill_rm``, but ``val_hi`` is set to ``1`` and ``val_lo`` is
            ``0``.

            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | Argument | Description                                                           | Data type             | Valid range            | Required |
            +==========+=======================================================================+=======================+========================+==========+
            | N        | Batch count of output tensor                                          | int                   | N > 0                  | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | C        | Channel count of output tensor                                        | int                   | C > 0                  | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | H        | Height count of output tensor                                         | int                   | H > 0                  | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | W        | Width count of output tensor                                          | int                   | W > 0                  | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | hOnes    | Height of high values region                                          | int                   | hOnes <= H             | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | wOnes    | Width of high values region                                           | int                   | wOnes <= W             | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | any      | Any input tensor with desired device and data types for output tensor | tt_lib.tensor.Tensor  |                        | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
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

        m_tensor.def("transpose", &transpose,
        py::arg("input").noconvert(), py::arg("dim0"), py::arg("dim1"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
        Returns a tensor that is a transposed version of input tensor with shape ``[W, Z, Y, X]``, where dimensions ``arg1`` and ``arg2`` are swapped.

        Input tensor must have BFLOAT16 data type. Second and third input specify the dimensions of tensor to be transposed.

        Output tensor will have BFLOAT16 data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input", "Input tensor", "Tensor", "Tensor of shape [W, Z, Y, X]", "Yes"
            "dim0", "dimension to transpose", "int", "Index within input tensor rank", "Yes"
            "dim1", "dimension to transpose", "int", "Index within input tensor rank", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
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
        m_tensor.def("nonzero", &non_zero_indices,
            py::arg("input"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            R"doc(Returns the number of elements (N) that are non-zero as well as a tensor of the same shape as input where the first N elements are the indices of non-zero elements )doc"
        );


        m_tensor.def("interleaved_to_sharded_partial", py::overload_cast<const Tensor &, const std::variant<CoreCoord, CoreRangeSet>, std::array<uint32_t, 2>, const uint32_t, const uint32_t, const TensorMemoryLayout, const ShardOrientation, const std::optional<const DataType>>(&interleaved_to_sharded_partial),
         py::arg("input"), py::arg("grid"), py::arg("shard_shape"), py::arg("num_slices"), py::arg("slice_index"), py::arg("shard_scheme").noconvert(), py::arg("shard_layout").noconvert(), py::arg("output_dtype").noconvert() = std::nullopt,
            R"doc(Converts a part of tensor from interleaved to sharded memory layout)doc");

        m_tensor.def("sharded_to_interleaved_partial", &sharded_to_interleaved_partial,
            py::arg("input"), py::arg("cache_tensor"), py::arg("num_slices"), py::arg("slice_index"), py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, py::arg("output_dtype").noconvert() = std::nullopt,
            R"doc(Converts a partial tensor from sharded_to_interleaved memory layout)doc"
        );

        // ---------- Multi-Device ops ----------

        // Reduce Scatter
        m_tensor.def("reduce_scatter", &reduce_scatter,
            py::arg("input_tensors"), py::arg("scatter_split_dim"), py::arg("reduce_op"), py::arg("num_links") = 1, py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            R"doc(
                Performs reduce scatter across chips, where the input tensors are sliced along the scatter dim, and pairwise reduced as they propagate and reduce through the cluster.

                For example, a reduce scatter on a ring of rank 8 and input tensor shapes (per rank) of [1,1,1024,8096] and scatter_dim=3, will split each input tensor
                on width into 8 parts of size [1,1,1024,1024]. Each of those parts will reduce with the corresponding chunk from the other ranks. All chips will collectively
                reduce the first incoming [1,1,1024,1024] chunk with their local first [1,1,1024,1024] chunk and be forwarded. The second incoming [1,1,1024,1024] chunk will
                be reduced with the second local [1,1,1024,1024] chunk and be forwarded and so on. Each rank in the ring will start on a different offset into the chunk such
                that by the end, they will finish with a different reduced chunk offset from the original tensor shape.

            .. csv-table::
                :header: "Argument", "Description", "Data type", "Valid range", "Required"

                "scatter_split_dim", "Dimension to evenly slice input tensor along for each rank", "int", "0..3", "Yes"
                "reduce_op", "reduction math operation", " ReduceOpMath", "SUM", "No"
                "num_links", "Number of ethernet links to allow the op to use to send data chip to chip for the operation. Default=1", "int", "1..max_num_links", "No"
                "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "Default is interleaved in DRAM", "No"
            )doc");
    }

}
