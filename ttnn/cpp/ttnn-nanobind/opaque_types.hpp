// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <nanobind/nanobind.h>

#include <tt-metalium/base_types.hpp>

// Types bound as opaque containers (nb::bind_vector in program_descriptors.cpp). An explicit
// type_caster specialization must be visible in every translation unit that casts the type, and
// several ttnn-nanobind sources share one unity-build blob, so the declarations live here and
// every TU that passes these vectors through nanobind includes this header.
NB_MAKE_OPAQUE(std::vector<tt::tt_metal::UnpackToDestMode>);
NB_MAKE_OPAQUE(std::vector<uint32_t>);
