// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <string>

#include "unary_op_types.hpp"

namespace tt::tt_metal {
enum class DataType;
}

namespace ttnn::operations::unary::utils {

UnaryWithParam string_to_unary_with_param(const std::string& name);

bool get_op_approx_mode(UnaryOpType op_type);

std::pair<std::string, std::string> get_op_init_and_func(UnaryOpType op_type, const std::vector<float>& params = {}, const std::string& idst = "0");

std::map<std::string, std::string> get_defines(
    UnaryOpType op_type, const std::optional<std::vector<float>>& params = std::nullopt, const std::string& id = "0", const std::string& idst = "0");

std::map<std::string, std::string> get_block_defines(
    const std::vector<UnaryWithParam>& op_chain, const std::string& block_id = "0", const std::string& idst = "0");

}
