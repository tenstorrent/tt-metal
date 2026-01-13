// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ttnn::graph {

// Helper function to sanitize strings for JSON serialization
// Replaces invalid UTF-8 sequences with replacement character (U+FFFD) or hex escape sequences
std::string sanitize_utf8(const std::string& str);

}  // namespace ttnn::graph
