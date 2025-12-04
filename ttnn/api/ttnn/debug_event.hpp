// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string_view>

namespace ttnn {

namespace debug_event {

// Log relation between workflow and operation information
void log_operation_info(uint32_t workflow_id, std::string_view operation_name, std::string_view operation_attributes);

// Log relation between workflow and particular excetution of operation
void log_runtime_id(uint32_t workflow_id, uint32_t runtime_id);

}  // namespace debug_event
}  // namespace ttnn
