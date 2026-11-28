// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::grouped_matmul {

// One core of the bank path (bank.py Slot): its bank, logical core, group, columns, region's first tile, cluster
// and whether it is the cluster's x sender.
struct BankSlot {
    uint32_t bank = 0;
    uint32_t core_x = 0;
    uint32_t core_y = 0;
    uint32_t g = 0;
    uint32_t n0 = 0;
    uint32_t pn = 0;
    uint32_t tile0 = 0;
    uint32_t cluster = 0;
    bool sender = false;
};

// A cluster's logical rectangle (bank.py BankLayout.rect[c]).
struct BankRect {
    uint32_t x0 = 0;
    uint32_t y0 = 0;
    uint32_t x1 = 0;
    uint32_t y1 = 0;
};

struct BankCore {
    uint32_t x = 0;
    uint32_t y = 0;
};

// bank.BankLayout as the binding converts it (the layout is planned in Python by bank.plan; the C++ op takes it
// as the caller's choice).
struct BankLayout {
    uint32_t G = 0;
    uint32_t KT = 0;
    uint32_t NT = 0;
    uint32_t banks = 0;
    uint32_t cores_per_bank = 0;
    std::vector<BankSlot> slots;
    std::vector<BankRect> rect;
    std::vector<BankCore> sender;

    uint32_t tiles_per_bank() const { return G * NT * KT / banks; }
};

// What the public entry resolved. Every field is structural (part of the default key): addresses are the only
// per-call values, rewritten by override_runtime_arguments.
struct GroupedMatmulParams {
    tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::BFLOAT16;
    tt::tt_metal::MemoryConfig output_memory_config;
    // compute_kernel_config's fields as the Python read them (getattr defaults: HiFi2, fp32 DEST on, approx off).
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi2;
    bool fp32_dest_acc_en = true;
    bool math_approx_mode = false;
    // The grid resolved from max_grid: max(1, min(device grid, max_grid)) per axis (the whole grid without it).
    uint32_t grid_x = 0;
    uint32_t grid_y = 0;
    uint32_t block_w = 0;  // 0: none given (the planner picks the widest of 32 / 16 / 8 / 4 / 2 / 1 that fits)
    uint32_t depth = 2;
    uint32_t rt = 1;
    // Bank path (w_layout given); w_inflight, x_ahead and one_group matter only there.
    std::optional<BankLayout> bank_layout;
    uint32_t w_inflight = 1;
    uint32_t x_ahead = 1;
    bool one_group = false;
    // diag stubs (GMM_DIAG_NO_W / NO_X / NO_COMPUTE defines).
    bool diag_no_w = false;
    bool diag_no_x = false;
    bool diag_no_compute = false;
};

struct GroupedMatmulInputs {
    const Tensor& x;
    const Tensor& w;
    const std::optional<Tensor>& output;
};

// The Python op refuses only through `assert`; its refusals reach Python as AssertionError (the bank test expects
// AssertionError "w_layout needs the bank tensor"). The binding's translator raises this type as AssertionError.
struct GroupedMatmulAssertion : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace ttnn::operations::grouped_matmul
