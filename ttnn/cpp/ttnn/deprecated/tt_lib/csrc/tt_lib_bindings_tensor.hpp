// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
namespace py = pybind11;

namespace tt::tt_metal {

namespace detail{

void TensorModulePyTensorTypes(py::module &m_tensor);
void TensorModulePyTensor(py::module &m_tensor);
}  // namespace detail

void TensorModuleTypes(py::module &m_tensor);
void TensorModule(py::module &m_tensor);

}  // namespace tt::tt_metal
