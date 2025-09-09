#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * hal/hal.hpp
 *
 * Thin wrapper around TT-Metal's HAL, without requiring a Metal context.
 */

#include <third_party/umd/device/api/umd/device/cluster.h>
#include <llrt/hal.hpp>
#include <llrt/rtoptions.hpp>

std::unique_ptr<tt::tt_metal::Hal> create_hal(const std::unique_ptr<tt::umd::Cluster>& cluster);
