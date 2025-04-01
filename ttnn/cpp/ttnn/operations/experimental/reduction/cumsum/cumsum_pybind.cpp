#include "cumsum_pybind.hpp"
#include <optional>
#include "pybind11/decorators.hpp"
#include "ttnn/operations/experimental/reduction/cumsum/device/cumsum_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::reduction {

void bind_cumsum_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::prim::cumsum,
        "ttnn.cumsum()",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::prim::cumsum)& self,
               const ttnn::Tensor& input_tensor,
               int64_t dim,
               std::optional<tt::tt_metal::DataType> dtype,
               std::optional<Tensor> preallocated_tensor) {
                return self(input_tensor, dim, dtype, preallocated_tensor);
            },
            py::arg("input_tensor"),
            py::arg("dim"),
            py::arg("dtype") = std::nullopt,
            py::arg("preallocated_tensor") = std::nullopt});
}

}  // namespace ttnn::operations::experimental::reduction
