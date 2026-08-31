// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Release FFT device plan caches while GraphTracker and the device are still
// open.  Safe to call from test teardown before close_device().

#pragma once

#include "ttnn/operations/experimental/fft/device/apply_twiddles_host.hpp"
#include "ttnn/operations/experimental/fft/device/apply_twiddles_xl_host.hpp"
#include "ttnn/operations/experimental/fft/device/bluestein_host.hpp"
#include "ttnn/operations/experimental/fft/device/stockham_host.hpp"

namespace ttnn::experimental::prim::fft_cache {

inline void clear_all_device_plan_caches() {
    ttnn::experimental::prim::bluestein_host::clear_cache();
    fft_stockham::clear_batch_plan_cache();
    ttnn::experimental::prim::apply_twiddles_host::clear_cache();
    ttnn::experimental::prim::apply_twiddles_xl_host::clear_cache();
}

}  // namespace ttnn::experimental::prim::fft_cache
