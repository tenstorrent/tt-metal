// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once // Cosmetic change to test CODEOWNERS2

// Functions for encoding and decoding PC buffer writes
#include <cstdint>

namespace ckernel
{

inline std::uint32_t get_pc_buf_cmd(std::uint32_t ckernel_id, std::uint32_t command_addr)
{
    // Ckernel fast launch cmd has MSB set
    return ckernel_id | ((command_addr & 0x7FFFF) << 12) | (1 << 31);
}

inline std::uint32_t get_pc_buf_cmd(std::uint32_t ckernel_id, std::uint32_t iterations, std::uint32_t number_of_extra_params)
{
    // Ckernel ID can be a max of 12 bits.
    return ckernel_id | ((iterations & 0xFFFF) << 16) | ((number_of_extra_params & 0xF) << 12);
}

inline void decode_pc_buf_cmd(std::uint32_t cmd, std::uint32_t &ckernel_id, std::uint32_t &iterations, std::uint32_t &number_of_extra_params)
{
    // Ckernel ID can be a max of 12 bits.
    ckernel_id             = cmd & 0xFFF;
    iterations             = (cmd >> 16) & 0xFFFF;
    number_of_extra_params = (cmd >> 12) & 0xF;
}

} // namespace ckernel
