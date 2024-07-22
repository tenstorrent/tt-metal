// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <string>
#include <unordered_map>

namespace tt {

namespace tt_metal {

void dump_tensor(const std::string& file_name, const Tensor& tensor, const std::unordered_map<std::string, std::string>& strategy);

template <typename T>
Tensor load_tensor(const std::string& file_name, T device = nullptr);

}  // namespace tt_metalls

}  // namespace tt
