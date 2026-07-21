// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::test {

namespace nb = nanobind;

void bind_test_dram_prefetcher_consumer(nb::module_& mod);

}  // namespace ttnn::operations::experimental::test
