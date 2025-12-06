// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_hwc_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "convert_to_hwc.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::experimental::cnn::detail {

void bind_convert_to_hwc(nb::module_& mod) {
    using OperationType = decltype(ttnn::experimental::convert_to_hwc);

    const auto doc = R"doc(
    Convert a tensor from CHW channel ordering to HWC channel ordering.

    The input tensor is expected to be in row-major layout and width-sharded in L1 or DRAM.
    The output is a row-major height-sharded tensor.

    Input tensor shape: [1, B, C, HW] where B is the batch size.
    Output tensor shape: [1, 1, B * HW, C] - batches are flattened into the spatial dimension.

    When C is not a multiple of the device alignment requirement, the output shard width is automatically padded up
    to the next multiple of the alignment requirement to satisfy alignment constraints.

    Args:
        input (ttnn.Tensor): Input tensor in CHW format with shape [1, B, C, HW], width-sharded in L1 or DRAM.
        memory_config (Optional[ttnn.MemoryConfig]): Output memory configuration.
                                                     Required only for DRAM inputs. If omitted for L1 inputs, the output memory_config is automatically inferred.
                                                     The output shard width will be rounded up to the next multiple of the alignment requirement for proper memory alignment.
        dtype (Optional[ttnn.DataType]): Output data type (defaults to input dtype)

    Returns:
        ttnn.Tensor: Output tensor in HWC format, height-sharded

    )doc";

    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::convert_to_hwc,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<DataType> dtype) { return self(input, memory_config, dtype); },
            nb::arg("input"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("dtype") = nb::none()});
}

}  // namespace ttnn::operations::experimental::cnn::detail
