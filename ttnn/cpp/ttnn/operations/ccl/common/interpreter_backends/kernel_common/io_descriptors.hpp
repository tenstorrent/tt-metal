// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/ccl/common/uops/ccl_command.hpp"

#include <cstdint>

struct address_info_t {
    size_t address = 0;
};

struct core_descriptor_info_t {
    union {
        ttnn::ccl::cmd::CclCommandCoreDescriptorTypeNocXY noc_unicast;
        ttnn::ccl::cmd::CclCommandCoreDescriptorTypeMcast noc_multicast;
    } core_desc_args;
};
