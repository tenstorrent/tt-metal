// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// A/B switches for the flat expert features under evaluation on Galaxy (GALAXY_FEATURE_TESTING.md). Every feature is
// OFF by default (the op as on the stack tip); a feature is on when
//   FLAT_FEATURES contains its key, or "all"   (a comma list: FLAT_FEATURES=cmm,rowmop)
// and FLAT_<KEY>=1 / FLAT_<KEY>=0 forces it on / off over FLAT_FEATURES (FLAT_FEATURES=all FLAT_NOC1=0).
// Read when a program is built: one setting per process (the program cache keeps what it built).
// Keys: cmm (gate/up matmul on custom_mm), rowmop (down matmul on the Blackhole row MOP with row-major y), lean (the
// combine hand-off's per-row diet), trailers (trailers spread over the y writers), noc1 (x snakes on NOC1).
#pragma once

#include <cctype>
#include <cstdlib>
#include <string>

namespace ttnn::operations::experimental::moe_block::flat_routed_expert::flat_features {

inline bool on(const std::string& key) {
    std::string upper = "FLAT_";
    for (const char c : key) {
        upper.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    }
    if (const char* e = std::getenv(upper.c_str()); e != nullptr && e[0] != 0) {
        return e[0] == '1';
    }
    const char* f = std::getenv("FLAT_FEATURES");
    if (f == nullptr) {
        return false;
    }
    const std::string list(f);
    size_t i = 0;
    while (i <= list.size()) {
        size_t j = list.find(',', i);
        if (j == std::string::npos) {
            j = list.size();
        }
        std::string item = list.substr(i, j - i);
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        if (item == key || item == "all") {
            return true;
        }
        i = j + 1;
    }
    return false;
}

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert::flat_features
