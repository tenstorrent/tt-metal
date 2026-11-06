// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define CKL_ELTWISE_BINARY_SFPU_BASIC
#include "ttnn/cpp/ttnn/operations/distributed_rms_norm/device/kernels/kernel_lib/eltwise/binary/sfpu/sfpu.inl"
#undef CKL_ELTWISE_BINARY_SFPU_BASIC
