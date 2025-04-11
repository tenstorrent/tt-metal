// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


namespace nanobind {
class module_;

// we aren't using 'module' because it can interfere with C++20 modules
//using module = module_;

}  // namespace nanobind
