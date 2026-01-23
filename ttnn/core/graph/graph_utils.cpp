// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/graph/graph_utils.hpp"

namespace ttnn::graph {

// Helper function to sanitize strings for JSON serialization
// Replaces invalid UTF-8 sequences with replacement character (U+FFFD)
std::string sanitize_utf8(const std::string& str) {
    std::string result;
    result.reserve(str.size());

    for (size_t i = 0; i < str.size();) {
        unsigned char c = str[i];

        // ASCII character (0x00-0x7F)
        if (c < 0x80) {
            result.push_back(c);
            i++;
        }
        // 2-byte UTF-8 (0xC0-0xDF)
        else if ((c >= 0xC0) && (c < 0xE0)) {
            if (i + 1 < str.size() && (str[i + 1] & 0xC0) == 0x80) {
                result.push_back(c);
                result.push_back(str[i + 1]);
                i += 2;
            } else {
                // Invalid sequence, use replacement character
                result.append("\\uFFFD");
                i++;
            }
        }
        // 3-byte UTF-8 (0xE0-0xEF)
        else if ((c >= 0xE0) && (c < 0xF0)) {
            if (i + 2 < str.size() && (str[i + 1] & 0xC0) == 0x80 && (str[i + 2] & 0xC0) == 0x80) {
                result.push_back(c);
                result.push_back(str[i + 1]);
                result.push_back(str[i + 2]);
                i += 3;
            } else {
                // Invalid sequence, use replacement character
                result.append("\\uFFFD");
                i++;
            }
        }
        // 4-byte UTF-8 (0xF0-0xF7)
        else if ((c >= 0xF0) && (c < 0xF8)) {
            if (i + 3 < str.size() && (str[i + 1] & 0xC0) == 0x80 && (str[i + 2] & 0xC0) == 0x80 &&
                (str[i + 3] & 0xC0) == 0x80) {
                result.push_back(c);
                result.push_back(str[i + 1]);
                result.push_back(str[i + 2]);
                result.push_back(str[i + 3]);
                i += 4;
            } else {
                // Invalid sequence, use replacement character
                result.append("\\uFFFD");
                i++;
            }
        }
        // Invalid UTF-8 start byte (0x80-0xBF, 0xF8-0xFF)
        else {
            // Replace with hex escape sequence for debugging
            char hex[8];
            snprintf(hex, sizeof(hex), "\\x%02X", c);
            result.append(hex);
            i++;
        }
    }

    return result;
}

}  // namespace ttnn::graph
