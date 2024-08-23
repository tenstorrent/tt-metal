// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_lib_bindings_tensor.hpp"
#include "tt_lib_bindings_tensor_impl.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/bcast/bcast_op.hpp"

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

    }

}
