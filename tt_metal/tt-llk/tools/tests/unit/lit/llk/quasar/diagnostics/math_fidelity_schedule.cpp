// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DEFINE: %{diagnose} = %{quasar_tensix_diagnose} %{quasar_math_thread}
// RUN: %split-file %s %t
// RUN: %{diagnose} %t/valid.cpp
// RUN: not %{diagnose} %t/invalid-fidelity.cpp 2>&1 | FileCheck %s --check-prefix=FIDELITY

#include "cmath_common.h"

using namespace ckernel;
using namespace ckernel::math;

static_assert(math_fidelity_schedule<MathFidelity::HiFi4>(DataFormat::Tf32, DataFormat::Tf32).phase_count == 4);
static_assert(math_fidelity_schedule<MathFidelity::HiFi4>(DataFormat::Tf32, DataFormat::Tf32).phase_increment == 1);
static_assert(math_fidelity_schedule<MathFidelity::HiFi3>(DataFormat::Float16, DataFormat::Float32).phase_count == 3);
static_assert(math_fidelity_schedule<MathFidelity::HiFi2>(DataFormat::Float16, DataFormat::Float32).phase_count == 2);

static_assert(math_fidelity_schedule<MathFidelity::HiFi4>(DataFormat::Tf32, DataFormat::Float16_b).phase_count == 2);
static_assert(math_fidelity_schedule<MathFidelity::HiFi4>(DataFormat::Float16_b, DataFormat::Tf32).phase_count == 2);
static_assert(math_fidelity_schedule<MathFidelity::HiFi2>(DataFormat::Float16_b, DataFormat::Tf32).phase_count == 1);
static_assert(math_fidelity_schedule<MathFidelity::HiFi4>(DataFormat::Float16_b, DataFormat::Float16_b).phase_count == 1);
static_assert(math_fidelity_schedule<MathFidelity::HiFi4>(DataFormat::Int8, DataFormat::Int8).phase_count == 1);

static_assert(math_fidelity_schedule<MathFidelity::HiFi4>(DataFormat::Float16_b, DataFormat::Tf32).phase_increment == 2);
static_assert(math_fidelity_schedule<MathFidelity::HiFi4>(DataFormat::Tf32, DataFormat::Float16_b).phase_increment == 1);
static_assert(math_fidelity_schedule<MathFidelity::HiFi4>(DataFormat::Float16_b, DataFormat::Float16_b).phase_increment == 0);
static_assert(math_fidelity_schedule<MathFidelity::LoFi>(DataFormat::Tf32, DataFormat::Tf32).phase_count == 1);
static_assert(math_fidelity_schedule<MathFidelity::LoFi>(DataFormat::Tf32, DataFormat::Tf32).phase_increment == 0);

static_assert(dest_src_format<true>(DataFormat::Float16_b) == DataFormat::Tf32);
static_assert(dest_src_format<true>(DataFormat::Int8) == DataFormat::Int8);
static_assert(dest_src_format<false>(DataFormat::Float16_b) == DataFormat::Float16_b);

#include "cmath_common.h"

using namespace ckernel;
using namespace ckernel::math;

constexpr auto invalid = math_fidelity_schedule<static_cast<MathFidelity>(1)>(DataFormat::Tf32, DataFormat::Tf32);
