// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <optional>
#include <string_view>

#include "ttnn/operations/matmul/device/config/matmul_auto_config.hpp"

// The roofline estimate of a candidate, from the nominal rates in HardwareDesc
namespace ttnn::operations::matmul::auto_config {

// Per-core roofline terms (cycles) of a blocked candidate. They depend on the output blocks but not on
// in0_block_w.
struct RooflineTerms {
    double compute = 0;  // the busiest core's tile products
    double noc = 0;      // input bytes the busiest core receives
    double dram = 0;     // bytes read from and written to DRAM, chip-wide
    double cycles() const { return std::max({compute, noc, dram}); }
};
RooflineTerms roofline(
    const MatmulDesc& matmul, const HardwareDesc& hw, Family family, const Blocking& b, bool fuse_batch);

// The roofline estimate (the largest RooflineTerms term), confidence 0: every estimator that has an answer
// outranks it.
class RooflineEstimator final : public Estimator {
public:
    std::string_view name() const override { return "roofline"; }
    std::optional<Estimate> estimate(
        const MatmulDesc& matmul, const HardwareDesc& hw, const Candidate& candidate) const override;
};

}  // namespace ttnn::operations::matmul::auto_config
