#pragma once

#include <pybind11/pybind11.h>

namespace ttnn {
namespace operations::conv {
namespace conv3d {

void py_bind_conv3d(pybind11::module& module);

}  // namespace conv3d
}  // namespace operations::conv
}  // namespace ttnn
