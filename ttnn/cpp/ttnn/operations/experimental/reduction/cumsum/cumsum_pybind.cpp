#include "cumsum_pybind.hpp"
#include "pybind11/decorators.hpp"
#include "ttnn/operations/experimental/reduction/cumsum/device/cumsum_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::reduction {

void bind_cumsum_operation(py::module& module) {
    decorators::bind_registered_operation(
        module,
        ttnn::prim::cumsum,
        "",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::prim::cumsum)& self, const ttnn::Tensor& input_tensor) {
                return self(input_tensor);
            },
            py::arg("input_tensor")});
}

}  // namespace ttnn::operations::experimental::reduction
