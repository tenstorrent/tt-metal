// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <optional>
#include <variant>

#include <tt-metalium/constants.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/factory/dram_sharded.hpp"
#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/factory/mcast_1d.hpp"
#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/factory/mcast_2d.hpp"
#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/factory/minimal.hpp"
#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/tables/dram_sharded_plan.hpp"
#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/tables/mcast_1d_plan.hpp"
#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/tables/mcast_2d_plan.hpp"
#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/tables/minimal_plan.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::prim::llama31_8b_matmul_bh {

using dram_sharded::DramShardedPlan;
using dram_sharded::DramShardedReadOptions;
using mcast_1d::Mcast1DPlan;
using mcast_2d::Mcast2DPlan;

struct BhMatmulRoute {
    operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig program_config;
    const Mcast1DPlan* plan;
};

struct BhMatmul2DRoute {
    operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig program_config;
    const Mcast2DPlan* plan;
};

struct BhMatmulDramShardedRoute {
    operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig program_config;
    DramShardedReadOptions read_options;
    const DramShardedPlan* plan;
};

namespace detail {

using tt::tt_metal::DataType;
using tt::tt_metal::MathFidelity;

struct Row {
    uint32_t M, K, N;
    DataType activation_dtype;
    DataType weight_dtype;
    uint32_t in0_block_w;
    MathFidelity math_fidelity;
    bool fp32_dest_acc_en;
    DataType output_dtype;
    uint32_t model_grid_x, model_grid_y;
    const Mcast1DPlan* plan;
};

// clang-format off
inline const std::array<Row, 4>& rows() {
    static const std::array<Row, 4> table = {{
        {128, 4096, 6144, DataType::BFLOAT16, DataType::BFLOAT8_B, 1, MathFidelity::HiFi2, true, DataType::BFLOAT16, 8, 10,
         &mcast_1d::kMcast1DPlanQkvPf128},
        {128, 4096, 4096, DataType::BFLOAT8_B, DataType::BFLOAT8_B, 8, MathFidelity::HiFi2, true, DataType::BFLOAT8_B, 8, 8,
         &mcast_1d::kMcast1DPlanWoPf128},
        {128, 4096, 14336, DataType::BFLOAT16, DataType::BFLOAT4_B, 8, MathFidelity::LoFi, false, DataType::BFLOAT16, 8, 8,
         &mcast_1d::kMcast1DPlanFf13Pf128},
        {128, 14336, 4096, DataType::BFLOAT8_B, DataType::BFLOAT8_B, 8, MathFidelity::HiFi2, false, DataType::BFLOAT16, 8, 8,
         &mcast_1d::kMcast1DPlanFf2Pf128},
    }};
    return table;
}

struct Row2D {
    uint32_t batch, M, K, N;
    DataType activation_dtype;
    DataType weight_dtype;
    uint32_t in0_block_w;
    MathFidelity math_fidelity;
    bool fp32_dest_acc_en;
    DataType output_dtype;
    uint32_t model_grid_x, model_grid_y;
    const Mcast2DPlan* plan;
};

inline const std::array<Row2D, 4>& rows_2d() {
    static const std::array<Row2D, 4> table = {{
        {1, 1024, 4096, 4096, DataType::BFLOAT8_B, DataType::BFLOAT8_B, 8, MathFidelity::HiFi2, true, DataType::BFLOAT8_B,
         8, 8, &mcast_2d::kMcast2DPlanWoPf1024},
        {2, 512, 4096, 14336, DataType::BFLOAT16, DataType::BFLOAT4_B, 4, MathFidelity::LoFi, false, DataType::BFLOAT16,
         8, 8, &mcast_2d::kMcast2DPlanFf13Bf4Pf1024},
        {2, 512, 4096, 14336, DataType::BFLOAT16, DataType::BFLOAT8_B, 4, MathFidelity::HiFi2, false, DataType::BFLOAT16,
         8, 8, &mcast_2d::kMcast2DPlanFf13Bf8Pf1024},
        {2, 512, 4096, 28672, DataType::BFLOAT16, DataType::BFLOAT4_B, 4, MathFidelity::LoFi, false, DataType::BFLOAT8_B,
         8, 8, &mcast_2d::kMcast2DPlanFf13SwigluPf1024},
    }};
    return table;
}

struct UnevenSegment {
    uint16_t storage_core;
    uint16_t tiles;
};

struct UnevenReader {
    uint16_t worker;
    uint16_t first_tile;
    uint16_t seg_begin;
    uint16_t seg_count;
};

struct UnevenTable {
    uint32_t N, K;
    uint32_t per_core_N;
    const UnevenReader* readers;
    uint32_t num_readers;
    const UnevenSegment* segments;
};

inline constexpr UnevenReader kUnevenReaders128x128[] = {{0, 0, 0, 2}, {1, 3, 2, 3}, {2, 2, 5, 2}, {0, 3, 7, 3}, {1, 2, 10, 3}, {2, 1, 13, 2}, {0, 2, 15, 3}, {1, 1, 18, 2}, {2, 0, 20, 2}, {0, 1, 22, 2}, {1, 0, 24, 2}, {2, 3, 26, 2}, {0, 0, 28, 2}, {1, 3, 30, 3}, {2, 2, 33, 2}, {0, 3, 35, 3}, {1, 2, 38, 3}, {2, 1, 41, 2}, {0, 2, 43, 3}, {1, 1, 46, 2}, {2, 0, 48, 0}};
inline constexpr UnevenSegment kUnevenSegments128x128[] = {{0, 4}, {1, 3}, {1, 1}, {2, 4}, {3, 2}, {3, 2}, {4, 3}, {4, 1}, {5, 4}, {6, 2}, {6, 2}, {7, 4}, {8, 1}, {8, 3}, {9, 2}, {9, 2}, {10, 4}, {11, 1}, {11, 3}, {12, 4}, {13, 4}, {14, 1}, {14, 3}, {15, 4}, {16, 4}, {17, 3}, {17, 1}, {18, 4}, {19, 4}, {20, 3}, {20, 1}, {21, 4}, {22, 2}, {22, 2}, {23, 3}, {23, 1}, {24, 4}, {25, 2}, {25, 2}, {26, 4}, {27, 1}, {27, 3}, {28, 2}, {28, 2}, {29, 4}, {30, 1}, {30, 3}, {31, 4}};
inline constexpr UnevenReader kUnevenReaders128x448[] = {{0, 0, 0, 5}, {1, 0, 5, 5}, {0, 1, 10, 6}, {1, 1, 16, 5}, {0, 0, 21, 5}, {1, 0, 26, 5}, {0, 1, 31, 6}, {1, 1, 37, 5}, {0, 0, 42, 5}, {1, 0, 47, 5}, {0, 1, 52, 6}, {1, 1, 58, 5}, {0, 0, 63, 5}, {1, 0, 68, 2}};
inline constexpr UnevenSegment kUnevenSegments128x448[] = {{0, 2}, {1, 2}, {2, 2}, {3, 2}, {4, 2}, {5, 2}, {6, 2}, {7, 2}, {8, 2}, {9, 1}, {9, 1}, {10, 2}, {11, 2}, {12, 2}, {13, 2}, {14, 1}, {14, 1}, {15, 2}, {16, 2}, {17, 2}, {18, 2}, {19, 2}, {20, 2}, {21, 2}, {22, 2}, {23, 2}, {24, 2}, {25, 2}, {26, 2}, {27, 2}, {28, 1}, {28, 1}, {29, 2}, {30, 2}, {31, 2}, {32, 2}, {33, 1}, {33, 1}, {34, 2}, {35, 2}, {36, 2}, {37, 2}, {38, 2}, {39, 2}, {40, 2}, {41, 2}, {42, 2}, {43, 2}, {44, 2}, {45, 2}, {46, 2}, {47, 1}, {47, 1}, {48, 2}, {49, 2}, {50, 2}, {51, 2}, {52, 1}, {52, 1}, {53, 2}, {54, 2}, {55, 2}, {56, 2}, {57, 2}, {58, 2}, {59, 2}, {60, 2}, {61, 2}, {62, 2}, {63, 2}};
inline constexpr UnevenReader kUnevenReaders192x128[] = {{0, 0, 0, 2}, {1, 4, 2, 3}, {2, 2, 5, 2}, {0, 4, 7, 3}, {1, 2, 10, 2}, {2, 0, 12, 2}, {0, 2, 14, 2}, {1, 0, 16, 2}, {2, 4, 18, 2}, {0, 0, 20, 2}, {1, 4, 22, 3}, {2, 2, 25, 2}, {0, 4, 27, 3}, {1, 2, 30, 2}, {2, 0, 32, 2}, {0, 2, 34, 2}, {1, 0, 36, 2}, {2, 4, 38, 2}, {0, 0, 40, 2}, {1, 4, 42, 3}, {2, 2, 45, 1}};
inline constexpr UnevenSegment kUnevenSegments192x128[] = {{0, 6}, {1, 4}, {1, 2}, {2, 6}, {3, 2}, {3, 4}, {4, 4}, {4, 2}, {5, 6}, {6, 2}, {6, 4}, {7, 6}, {8, 6}, {9, 2}, {9, 4}, {10, 6}, {11, 6}, {12, 4}, {12, 2}, {13, 6}, {14, 6}, {15, 4}, {15, 2}, {16, 6}, {17, 2}, {17, 4}, {18, 4}, {18, 2}, {19, 6}, {20, 2}, {20, 4}, {21, 6}, {22, 6}, {23, 2}, {23, 4}, {24, 6}, {25, 6}, {26, 4}, {26, 2}, {27, 6}, {28, 6}, {29, 4}, {29, 2}, {30, 6}, {31, 2}, {31, 4}};
inline constexpr UnevenReader kUnevenReaders448x128[] = {{0, 0, 0, 5}, {1, 4, 5, 6}, {0, 1, 11, 5}, {1, 5, 16, 6}, {0, 2, 22, 5}, {1, 6, 27, 6}, {0, 3, 33, 5}, {1, 0, 38, 5}, {0, 4, 43, 6}, {1, 1, 49, 5}, {0, 5, 54, 6}, {1, 2, 60, 5}, {0, 6, 65, 6}, {1, 3, 71, 5}};
inline constexpr UnevenSegment kUnevenSegments448x128[] = {{0, 7}, {1, 7}, {2, 7}, {3, 7}, {4, 4}, {4, 3}, {5, 7}, {6, 7}, {7, 7}, {8, 7}, {9, 1}, {9, 6}, {10, 7}, {11, 7}, {12, 7}, {13, 5}, {13, 2}, {14, 7}, {15, 7}, {16, 7}, {17, 7}, {18, 2}, {18, 5}, {19, 7}, {20, 7}, {21, 7}, {22, 6}, {22, 1}, {23, 7}, {24, 7}, {25, 7}, {26, 7}, {27, 3}, {27, 4}, {28, 7}, {29, 7}, {30, 7}, {31, 7}, {32, 7}, {33, 7}, {34, 7}, {35, 7}, {36, 4}, {36, 3}, {37, 7}, {38, 7}, {39, 7}, {40, 7}, {41, 1}, {41, 6}, {42, 7}, {43, 7}, {44, 7}, {45, 5}, {45, 2}, {46, 7}, {47, 7}, {48, 7}, {49, 7}, {50, 2}, {50, 5}, {51, 7}, {52, 7}, {53, 7}, {54, 6}, {54, 1}, {55, 7}, {56, 7}, {57, 7}, {58, 7}, {59, 3}, {59, 4}, {60, 7}, {61, 7}, {62, 7}, {63, 7}};
inline constexpr UnevenReader kUnevenReaders501x128[] = {{0, 0, 0, 5}, {1, 4, 5, 5}, {0, 0, 10, 5}, {1, 4, 15, 5}, {0, 0, 20, 5}, {1, 4, 25, 5}, {0, 0, 30, 5}, {1, 4, 35, 5}, {0, 0, 40, 5}, {1, 4, 45, 5}, {0, 0, 50, 5}, {1, 4, 55, 5}, {0, 0, 60, 5}, {1, 4, 65, 5}};
inline constexpr UnevenSegment kUnevenSegments501x128[] = {{0, 8}, {1, 8}, {2, 8}, {3, 8}, {4, 4}, {4, 4}, {5, 8}, {6, 8}, {7, 8}, {8, 8}, {9, 8}, {10, 8}, {11, 8}, {12, 8}, {13, 4}, {13, 4}, {14, 8}, {15, 8}, {16, 8}, {17, 8}, {18, 8}, {19, 8}, {20, 8}, {21, 8}, {22, 4}, {22, 4}, {23, 8}, {24, 8}, {25, 8}, {26, 8}, {27, 8}, {28, 8}, {29, 8}, {30, 8}, {31, 4}, {31, 4}, {32, 8}, {33, 8}, {34, 8}, {35, 8}, {36, 8}, {37, 8}, {38, 8}, {39, 8}, {40, 4}, {40, 4}, {41, 8}, {42, 8}, {43, 8}, {44, 8}, {45, 8}, {46, 8}, {47, 8}, {48, 8}, {49, 4}, {49, 4}, {50, 8}, {51, 8}, {52, 8}, {53, 8}, {54, 8}, {55, 8}, {56, 8}, {57, 8}, {58, 4}, {58, 4}, {59, 8}, {60, 8}, {61, 8}, {62, 5}};

inline constexpr UnevenTable kUnevenTables[] = {
    {128, 128, 7, kUnevenReaders128x128, 21, kUnevenSegments128x128},
    {128, 448, 10, kUnevenReaders128x448, 14, kUnevenSegments128x448},
    {192, 128, 10, kUnevenReaders192x128, 21, kUnevenSegments192x128},
    {448, 128, 32, kUnevenReaders448x128, 14, kUnevenSegments448x128},
    {501, 128, 36, kUnevenReaders501x128, 14, kUnevenSegments501x128},
};

struct DsReaderSlot {
    uint8_t bank, worker, x, y, vc;
};

struct DsReaderTable {
    uint32_t N, K;
    const DsReaderSlot* readers;
    uint32_t num_readers;
    const DsReaderSlot* first;
    uint32_t num_banks;
};

inline constexpr DsReaderSlot kDsReaders128x128[] = {{0, 0, 0, 9, 0}, {0, 1, 0, 8, 1}, {0, 2, 0, 6, 2}, {1, 0, 0, 0, 1}, {1, 1, 10, 0, 2}, {1, 2, 9, 0, 3}, {2, 0, 0, 7, 2}, {2, 1, 0, 5, 3}, {2, 2, 0, 4, 0}, {3, 0, 6, 9, 3}, {3, 1, 6, 8, 0}, {3, 2, 6, 7, 1}, {4, 0, 6, 1, 0}, {4, 1, 6, 5, 1}, {4, 2, 5, 9, 2}, {5, 0, 6, 6, 1}, {5, 1, 5, 6, 2}, {5, 2, 4, 6, 3}, {6, 0, 6, 4, 2}, {6, 1, 5, 4, 3}, {6, 2, 4, 4, 0}};
inline constexpr DsReaderSlot kDsFirst128x128[] = {{0, 0, 0, 9, 0}, {1, 0, 0, 0, 0}, {2, 0, 0, 7, 0}, {3, 0, 6, 9, 0}, {4, 0, 6, 1, 0}, {5, 0, 6, 6, 0}, {6, 0, 6, 4, 0}};
inline constexpr DsReaderSlot kDsReaders128x448[] = {{0, 0, 0, 9, 0}, {0, 1, 0, 8, 1}, {1, 0, 0, 0, 1}, {1, 1, 10, 0, 2}, {2, 0, 0, 7, 2}, {2, 1, 10, 7, 3}, {3, 0, 6, 9, 3}, {3, 1, 6, 8, 0}, {4, 0, 6, 1, 0}, {4, 1, 5, 9, 1}, {5, 0, 6, 6, 1}, {5, 1, 10, 6, 2}, {6, 0, 6, 4, 2}, {6, 1, 4, 9, 3}};
inline constexpr DsReaderSlot kDsFirst128x448[] = {{0, 0, 0, 9, 0}, {1, 0, 0, 0, 0}, {2, 0, 0, 7, 0}, {3, 0, 6, 9, 0}, {4, 0, 6, 1, 0}, {5, 0, 6, 6, 0}, {6, 0, 6, 4, 0}};
inline constexpr DsReaderSlot kDsReaders192x128[] = {{0, 0, 0, 9, 0}, {0, 1, 0, 8, 1}, {0, 2, 0, 6, 2}, {1, 0, 0, 0, 1}, {1, 1, 10, 0, 2}, {1, 2, 9, 0, 3}, {2, 0, 0, 7, 2}, {2, 1, 0, 5, 3}, {2, 2, 0, 4, 0}, {3, 0, 6, 9, 3}, {3, 1, 6, 8, 0}, {3, 2, 6, 7, 1}, {4, 0, 6, 1, 0}, {4, 1, 6, 5, 1}, {4, 2, 5, 9, 2}, {5, 0, 6, 6, 1}, {5, 1, 5, 6, 2}, {5, 2, 4, 6, 3}, {6, 0, 6, 4, 2}, {6, 1, 5, 4, 3}, {6, 2, 4, 4, 0}};
inline constexpr DsReaderSlot kDsFirst192x128[] = {{0, 0, 0, 9, 0}, {1, 0, 0, 0, 0}, {2, 0, 0, 7, 0}, {3, 0, 6, 9, 0}, {4, 0, 6, 1, 0}, {5, 0, 6, 6, 0}, {6, 0, 6, 4, 0}};
inline constexpr DsReaderSlot kDsReaders448x128[] = {{0, 0, 0, 9, 0}, {0, 1, 0, 8, 0}, {1, 0, 0, 0, 1}, {1, 1, 10, 0, 2}, {2, 0, 0, 7, 2}, {2, 1, 10, 7, 3}, {3, 0, 6, 9, 3}, {3, 1, 6, 8, 3}, {4, 0, 6, 1, 0}, {4, 1, 5, 9, 1}, {5, 0, 6, 6, 1}, {5, 1, 10, 6, 2}, {6, 0, 6, 4, 2}, {6, 1, 4, 9, 2}};
inline constexpr DsReaderSlot kDsFirst448x128[] = {{0, 0, 0, 9, 0}, {1, 0, 0, 0, 0}, {2, 0, 0, 7, 0}, {3, 0, 6, 9, 0}, {4, 0, 6, 1, 0}, {5, 0, 6, 6, 0}, {6, 0, 6, 4, 0}};
inline constexpr DsReaderSlot kDsReaders501x128[] = {{0, 0, 0, 9, 0}, {0, 1, 0, 8, 0}, {1, 0, 0, 0, 1}, {1, 1, 10, 0, 2}, {2, 0, 0, 7, 2}, {2, 1, 10, 7, 3}, {3, 0, 6, 9, 3}, {3, 1, 6, 8, 3}, {4, 0, 6, 1, 0}, {4, 1, 5, 9, 1}, {5, 0, 6, 6, 1}, {5, 1, 10, 6, 2}, {6, 0, 6, 4, 2}, {6, 1, 4, 9, 2}};
inline constexpr DsReaderSlot kDsFirst501x128[] = {{0, 0, 0, 9, 0}, {1, 0, 0, 0, 0}, {2, 0, 0, 7, 0}, {3, 0, 6, 9, 0}, {4, 0, 6, 1, 0}, {5, 0, 6, 6, 0}, {6, 0, 6, 4, 0}};

inline constexpr DsReaderTable kDsReaderTables[] = {
    {128, 128, kDsReaders128x128, 21, kDsFirst128x128, 7},
    {128, 448, kDsReaders128x448, 14, kDsFirst128x448, 7},
    {192, 128, kDsReaders192x128, 21, kDsFirst192x128, 7},
    {448, 128, kDsReaders448x128, 14, kDsFirst448x128, 7},
    {501, 128, kDsReaders501x128, 14, kDsFirst501x128, 7},
};

struct DramShardedRow {
    uint32_t M, K, N;
    DataType activation_dtype;
    DataType weight_dtype;
    uint32_t in0_block_w, per_core_N;
    MathFidelity math_fidelity;
    bool fp32_dest_acc_en;
    DataType output_dtype;
    uint32_t workers_per_bank;
    DramShardedReadOptions read_options;
    const DramShardedPlan* plan;
};

inline const std::array<DramShardedRow, 5>& dram_sharded_rows() {
    static const std::array<DramShardedRow, 5> table = {{
        {32, 4096, 16032, DataType::BFLOAT16, DataType::BFLOAT8_B, 2, 8, MathFidelity::HiFi2, false, DataType::BFLOAT8_B, 2,
         {.uneven_readers = true, .streamed_write_back = true}, &dram_sharded::kDsPlanLmHeadDec},
        {32, 4096, 14336, DataType::BFLOAT16, DataType::BFLOAT4_B, 2, 7, MathFidelity::LoFi, false, DataType::BFLOAT16, 2,
         {.uneven_readers = true, .streamed_write_back = true}, &dram_sharded::kDsPlanFf13Dec},
        {32, 14336, 4096, DataType::BFLOAT8_B, DataType::BFLOAT8_B, 7, 2, MathFidelity::HiFi2, false, DataType::BFLOAT16, 2,
         {.uneven_readers = true, .streamed_write_back = true}, &dram_sharded::kDsPlanFf2Dec},
        {32, 4096, 6144, DataType::BFLOAT16, DataType::BFLOAT8_B, 4, 6, MathFidelity::HiFi2, true, DataType::BFLOAT16, 3,
         {.uneven_readers = true, .streamed_write_back = true}, &dram_sharded::kDsPlanQkvDec},
        {32, 4096, 4096, DataType::BFLOAT16, DataType::BFLOAT8_B, 4, 4, MathFidelity::HiFi2, true, DataType::BFLOAT16, 3,
         {.uneven_readers = true, .streamed_write_back = true}, &dram_sharded::kDsPlanWoDec},
    }};
    return table;
}

// clang-format on

inline const UnevenTable* uneven_table(uint32_t N, uint32_t K) {
    for (const auto& t : kUnevenTables) {
        if (t.N == N && t.K == K) {
            return &t;
        }
    }
    return nullptr;
}

inline const DsReaderTable* ds_reader_table(uint32_t N, uint32_t K) {
    for (const auto& t : kDsReaderTables) {
        if (t.N == N && t.K == K) {
            return &t;
        }
    }
    return nullptr;
}

inline bool ds_readers_match_device(const ttnn::Tensor& a, const DsReaderTable& t) {
    auto* device = a.device();
    const auto first = device->get_optimal_dram_bank_to_logical_worker_assignment(
        tt::tt_metal::NOC::NOC_0, tt::tt_metal::distributed::MeshCoordinate::zero_coordinate(device->shape().dims()));
    if (first.size() != t.num_banks) {
        return false;
    }
    for (uint32_t i = 0; i < t.num_banks; ++i) {
        const auto it = first.find(t.first[i].bank);
        if (it == first.end() || it->second != CoreCoord(t.first[i].x, t.first[i].y)) {
            return false;
        }
    }
    return true;
}

inline bool p100_device(const ttnn::Tensor& a) {
    const auto grid = a.device()->compute_with_storage_grid_size();
    return grid.x == 11 && grid.y == 10 && a.device()->num_dram_channels() == 7;
}

inline bool plain_two_input_matmul(const MatmulParams& params, const MatmulInputs& tensors) {
    if (params.untilize_out || params.transpose_a || params.transpose_b || params.user_run_batched ||
        params.global_cb.has_value() || params.sub_device_id.has_value() || !params.prefetcher_pipes.empty() ||
        params.in0_column_offset.has_value() || params.user_fused_activation.has_value() ||
        !params.compute_kernel_config.has_value()) {
        return false;
    }
    if (params.output_tile.has_value() && (params.output_tile->get_tile_shape()[0] != tt::constants::TILE_HEIGHT ||
                                           params.output_tile->get_tile_shape()[1] != tt::constants::TILE_WIDTH)) {
        return false;
    }
    if (tensors.input_tensors.size() != 2) {
        return false;
    }
    for (const auto& t : tensors.optional_input_tensors) {
        if (t.has_value()) {
            return false;
        }
    }
    const auto& a = tensors.input_tensors[0];
    const auto& b = tensors.input_tensors[1];
    return a.device()->arch() == tt::ARCH::BLACKHOLE && p100_device(a) && a.layout() == Layout::TILE &&
           b.layout() == Layout::TILE && b.memory_config().is_dram() &&
           b.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
}

inline bool full_rect_or_unset(const std::optional<tt::tt_metal::CoreRangeSet>& cores, const CoreCoord& grid) {
    return !cores.has_value() ||
           cores.value() == tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(
                                tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(grid.x - 1, grid.y - 1)));
}

struct InterleavedShape {
    uint32_t batch, M, K, N;
};

inline std::optional<InterleavedShape> interleaved_shape(const MatmulParams& params, const MatmulInputs& tensors) {
    const auto& a = tensors.input_tensors[0];
    const auto& b = tensors.input_tensors[1];
    if (a.memory_config().is_sharded() || !a.memory_config().is_dram() || params.output_mem_config.is_sharded()) {
        return std::nullopt;
    }
    const auto& as = a.padded_shape();
    const auto& bs = b.padded_shape();
    if (as.rank() < 2 || bs.rank() < 2 || a.logical_shape() != as || b.logical_shape() != bs) {
        return std::nullopt;
    }
    uint32_t batch = 1;
    for (int i = 0; i < static_cast<int>(as.rank()) - 2; ++i) {
        batch *= as[i];
    }
    for (int i = 0; i < static_cast<int>(bs.rank()) - 2; ++i) {
        if (bs[i] != 1) {
            return std::nullopt;
        }
    }
    if (bs[-2] != as[-1]) {
        return std::nullopt;
    }
    return InterleavedShape{
        batch, static_cast<uint32_t>(as[-2]), static_cast<uint32_t>(as[-1]), static_cast<uint32_t>(bs[-1])};
}

}  // namespace detail

inline std::optional<BhMatmulRoute> bh_matmul_route(const MatmulParams& params, const MatmulInputs& tensors) {
    using namespace detail;
    if (!params.program_config.has_value() || !plain_two_input_matmul(params, tensors)) {
        return std::nullopt;
    }
    const auto* cfg =
        std::get_if<operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>(&params.program_config.value());
    if (cfg == nullptr || cfg->transpose_mcast || cfg->fused_activation.has_value() ||
        !full_rect_or_unset(cfg->allowed_worker_cores, cfg->compute_with_storage_grid_size)) {
        return std::nullopt;
    }
    const auto shape = interleaved_shape(params, tensors);
    if (!shape.has_value() || shape->batch != 1) {
        return std::nullopt;
    }
    const auto& a = tensors.input_tensors[0];
    const auto& b = tensors.input_tensors[1];
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), params.compute_kernel_config.value());
    (void)math_approx_mode;
    if (!packer_l1_acc || dst_full_sync_en) {
        return std::nullopt;
    }
    const uint32_t M = shape->M, K = shape->K, N = shape->N;
    for (const auto& r : rows()) {
        if (r.M != M || r.K != K || r.N != N || r.activation_dtype != a.dtype() || r.weight_dtype != b.dtype() ||
            r.in0_block_w != cfg->in0_block_w || r.math_fidelity != math_fidelity ||
            r.fp32_dest_acc_en != fp32_dest_acc_en || r.output_dtype != params.output_dtype.value_or(a.dtype())) {
            continue;
        }
        if (cfg->compute_with_storage_grid_size.x != r.model_grid_x ||
            cfg->compute_with_storage_grid_size.y != r.model_grid_y) {
            return std::nullopt;
        }
        if (a.memory_config().is_sharded() || params.output_mem_config.is_sharded() ||
            a.tensor_spec().tile().get_tile_shape() != b.tensor_spec().tile().get_tile_shape() ||
            a.tensor_spec().tile().get_tile_shape()[0] != tt::constants::TILE_HEIGHT ||
            a.tensor_spec().tile().get_tile_shape()[1] != tt::constants::TILE_WIDTH) {
            return std::nullopt;
        }
        const Mcast1DPlan& p = *r.plan;
        operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig pc{
            .compute_with_storage_grid_size = tt::tt_metal::CoreCoord(p.grid_x, p.grid_y),
            .in0_block_w = p.in0_block_w,
            .out_subblock_h = p.out_subblock_h,
            .out_subblock_w = p.out_subblock_w,
            .out_block_h = p.per_core_M,
            .out_block_w = p.per_core_N,
            .per_core_M = p.per_core_M,
            .per_core_N = p.per_core_N,
            .fuse_batch = true,
            .fused_activation = std::nullopt,
            .mcast_in0 = true,
            .gather_in0 = false,
            .hop_cores = {},
            .num_global_cb_receivers = 0,
            .untilize_out = false,
            .allowed_worker_cores = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(
                tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(p.grid_x - 1, p.grid_y - 1))),
        };
        return BhMatmulRoute{.program_config = pc, .plan = r.plan};
    }
    return std::nullopt;
}

inline std::optional<BhMatmul2DRoute> bh_matmul_route_2d(const MatmulParams& params, const MatmulInputs& tensors) {
    using namespace detail;
    if (!params.program_config.has_value() || !plain_two_input_matmul(params, tensors)) {
        return std::nullopt;
    }
    const auto* cfg =
        std::get_if<operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>(&params.program_config.value());
    if (cfg == nullptr || cfg->transpose_mcast || cfg->fused_activation.has_value() ||
        !full_rect_or_unset(cfg->allowed_worker_cores, cfg->compute_with_storage_grid_size)) {
        return std::nullopt;
    }
    const auto shape = interleaved_shape(params, tensors);
    if (!shape.has_value()) {
        return std::nullopt;
    }
    const auto& a = tensors.input_tensors[0];
    const auto& b = tensors.input_tensors[1];
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), params.compute_kernel_config.value());
    (void)math_approx_mode;
    if (!packer_l1_acc || dst_full_sync_en) {
        return std::nullopt;
    }
    for (const auto& r : rows_2d()) {
        if (r.batch != shape->batch || r.M != shape->M || r.K != shape->K || r.N != shape->N ||
            r.activation_dtype != a.dtype() || r.weight_dtype != b.dtype() || r.in0_block_w != cfg->in0_block_w ||
            r.math_fidelity != math_fidelity || r.fp32_dest_acc_en != fp32_dest_acc_en ||
            r.output_dtype != params.output_dtype.value_or(a.dtype())) {
            continue;
        }
        if (cfg->compute_with_storage_grid_size.x != r.model_grid_x ||
            cfg->compute_with_storage_grid_size.y != r.model_grid_y) {
            return std::nullopt;
        }
        const Mcast2DPlan& p = *r.plan;
        operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig pc{
            .compute_with_storage_grid_size = tt::tt_metal::CoreCoord(p.grid_x, p.grid_y),
            .in0_block_w = p.in0_block_w,
            .out_subblock_h = p.out_subblock_h,
            .out_subblock_w = p.out_subblock_w,
            .out_block_h = p.out_block_h,
            .out_block_w = p.out_block_w,
            .per_core_M = p.per_core_M,
            .per_core_N = p.per_core_N,
            .transpose_mcast = true,
            .fused_activation = std::nullopt,
            .fuse_batch = true,
        };
        return BhMatmul2DRoute{.program_config = pc, .plan = r.plan};
    }
    return std::nullopt;
}

inline std::optional<BhMatmulDramShardedRoute> bh_matmul_route_dram_sharded(
    const MatmulParams& params, const MatmulInputs& tensors) {
    using namespace detail;
    if (!params.program_config.has_value() || !plain_two_input_matmul(params, tensors)) {
        return std::nullopt;
    }
    const auto* cfg = std::get_if<operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>(
        &params.program_config.value());
    if (cfg == nullptr || cfg->num_workers_per_dram_bank != 1 || cfg->use_multicore_pipeline ||
        cfg->fused_activation.has_value() || cfg->per_core_M != 1 || !params.output_mem_config.is_sharded()) {
        return std::nullopt;
    }
    const auto& a = tensors.input_tensors[0];
    const auto& b = tensors.input_tensors[1];
    if (!a.memory_config().is_sharded() || a.memory_config().is_dram()) {
        return std::nullopt;
    }
    const auto& as = a.padded_shape();
    const auto& bs = b.padded_shape();
    if (as.rank() < 2 || bs.rank() < 2) {
        return std::nullopt;
    }
    for (int i = 0; i < static_cast<int>(as.rank()) - 2; ++i) {
        if (as[i] != 1) {
            return std::nullopt;
        }
    }
    for (int i = 0; i < static_cast<int>(bs.rank()) - 2; ++i) {
        if (bs[i] != 1) {
            return std::nullopt;
        }
    }
    const uint32_t M = as[-2], K = as[-1], N = b.logical_shape()[-1];
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), params.compute_kernel_config.value());
    (void)math_approx_mode;
    (void)packer_l1_acc;
    if (dst_full_sync_en) {
        return std::nullopt;
    }
    for (const auto& r : dram_sharded_rows()) {
        if (r.M != M || r.K != K || r.N != N || r.activation_dtype != a.dtype() || r.weight_dtype != b.dtype() ||
            r.in0_block_w != cfg->in0_block_w || r.per_core_N != cfg->per_core_N || r.math_fidelity != math_fidelity ||
            r.fp32_dest_acc_en != fp32_dest_acc_en || r.output_dtype != params.output_dtype.value_or(a.dtype())) {
            continue;
        }
        const auto& in0_shard = a.shard_spec();
        if (!in0_shard.has_value() || in0_shard->shape[1] != r.in0_block_w * tt::constants::TILE_WIDTH ||
            in0_shard->grid.num_cores() != r.plan->in0_storage_cores) {
            return std::nullopt;
        }
        const auto* readers = ds_reader_table(N / tt::constants::TILE_WIDTH, K / tt::constants::TILE_WIDTH);
        if (readers == nullptr || !ds_readers_match_device(a, *readers)) {
            return std::nullopt;
        }
        auto pc = *cfg;
        pc.num_workers_per_dram_bank = r.workers_per_bank;
        return BhMatmulDramShardedRoute{.program_config = pc, .read_options = r.read_options, .plan = r.plan};
    }
    return std::nullopt;
}

struct MinimalRow {
    uint32_t M, K, N;
    tt::tt_metal::DataType in0_dtype, in1_dtype, out_dtype;
    tt::tt_metal::MathFidelity math_fidelity;
    bool fp32_dest_acc_en;
    uint32_t grid_x, grid_y, M_block_size, K_block_size, N_block_size;
    uint32_t routed_grid_x, routed_grid_y, routed_M_block_size, routed_N_block_size, subblock_h, subblock_w;
    const ttnn::experimental::prim::llama31_8b_matmul_bh::MinimalPlan* plan;
};

inline const std::array<MinimalRow, 2>& minimal_rows() {
    using tt::tt_metal::DataType;
    using tt::tt_metal::MathFidelity;
    static const std::array<MinimalRow, 2> rows = {{
        {.M = 1024,
         .K = 4096,
         .N = 6144,
         .in0_dtype = DataType::BFLOAT16,
         .in1_dtype = DataType::BFLOAT8_B,
         .out_dtype = DataType::BFLOAT16,
         .math_fidelity = MathFidelity::HiFi2,
         .fp32_dest_acc_en = true,
         .grid_x = 8,
         .grid_y = 10,
         .M_block_size = 8,
         .K_block_size = 8,
         .N_block_size = 8,
         .routed_grid_x = 11,
         .routed_grid_y = 10,
         .routed_M_block_size = 3,
         .routed_N_block_size = 4,
         .subblock_h = 1,
         .subblock_w = 4,
         .plan = &ttnn::experimental::prim::llama31_8b_matmul_bh::kMinimalPlanQkvPf1024},
        {.M = 1024,
         .K = 14336,
         .N = 4096,
         .in0_dtype = DataType::BFLOAT8_B,
         .in1_dtype = DataType::BFLOAT8_B,
         .out_dtype = DataType::BFLOAT8_B,
         .math_fidelity = MathFidelity::HiFi2,
         .fp32_dest_acc_en = false,
         .grid_x = 8,
         .grid_y = 8,
         .M_block_size = 8,
         .K_block_size = 8,
         .N_block_size = 8,
         .routed_grid_x = 11,
         .routed_grid_y = 10,
         .routed_M_block_size = 3,
         .routed_N_block_size = 7,
         .subblock_h = 3,
         .subblock_w = 1,
         .plan = &ttnn::experimental::prim::llama31_8b_matmul_bh::kMinimalPlanFf2Pf1024},
    }};
    return rows;
}

namespace detail {

inline bool minimal_row_matches(
    const MinimalRow& r,
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    tt::tt_metal::DataType out_dtype,
    const DeviceComputeKernelConfig& kernel_config) {
    const uint32_t K = input_tensor.padded_shape()[-1];
    const uint32_t M = input_tensor.physical_volume() / K;
    const uint32_t N = weight_tensor.padded_shape()[-1];
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input_tensor.device()->arch(), kernel_config);
    const auto& w = weight_tensor.memory_config();
    return r.M == M && r.K == K && r.N == N && r.in0_dtype == input_tensor.dtype() &&
           r.in1_dtype == weight_tensor.dtype() && r.out_dtype == out_dtype && r.math_fidelity == math_fidelity &&
           r.fp32_dest_acc_en == fp32_dest_acc_en &&
           w.memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED &&
           w.buffer_type() == tt::tt_metal::BufferType::DRAM && w.shard_spec().has_value() &&
           w.shard_spec()->shape[1] == r.plan->bulk_width * tt::constants::TILE_WIDTH;
}

}  // namespace detail

inline std::optional<ttnn::experimental::prim::MinimalMatmulConfig> minimal_route(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config,
    tt::tt_metal::DataType out_dtype,
    const DeviceComputeKernelConfig& kernel_config) {
    auto* device = input_tensor.device();
    if (!config.has_value() || device->arch() != tt::ARCH::BLACKHOLE || !detail::p100_device(input_tensor)) {
        return std::nullopt;
    }
    const auto& cfg = config.value();
    if (cfg.subblock_h != 1 || cfg.subblock_w != 1) {
        return std::nullopt;
    }
    for (const auto& r : minimal_rows()) {
        if (detail::minimal_row_matches(r, input_tensor, weight_tensor, out_dtype, kernel_config) &&
            r.grid_x == cfg.compute_with_storage_grid_size.x && r.grid_y == cfg.compute_with_storage_grid_size.y &&
            r.M_block_size == cfg.M_block_size && r.K_block_size == cfg.K_block_size &&
            r.N_block_size == cfg.N_block_size) {
            auto routed = cfg;
            routed.compute_with_storage_grid_size = CoreCoord(r.routed_grid_x, r.routed_grid_y);
            routed.M_block_size = r.routed_M_block_size;
            routed.N_block_size = r.routed_N_block_size;
            routed.subblock_h = r.subblock_h;
            routed.subblock_w = r.subblock_w;
            return routed;
        }
    }
    return std::nullopt;
}

inline const ttnn::experimental::prim::llama31_8b_matmul_bh::MinimalPlan* minimal_plan(
    const ttnn::experimental::prim::MinimalMatmulParams& attributes,
    const ttnn::experimental::prim::MinimalMatmulInputs& tensor_args) {
    if (!attributes.config.has_value()) {
        return nullptr;
    }
    const auto& cfg = attributes.config.value();
    const auto& input_tensor = tensor_args.input_tensor;
    const auto out_dtype = attributes.output_dtype.value_or(input_tensor.dtype());
    for (const auto& r : minimal_rows()) {
        if (detail::minimal_row_matches(
                r, input_tensor, tensor_args.weight_tensor, out_dtype, attributes.compute_kernel_config) &&
            r.routed_grid_x == cfg.compute_with_storage_grid_size.x &&
            r.routed_grid_y == cfg.compute_with_storage_grid_size.y && r.routed_M_block_size == cfg.M_block_size &&
            r.K_block_size == cfg.K_block_size && r.routed_N_block_size == cfg.N_block_size &&
            r.subblock_h == cfg.subblock_h && r.subblock_w == cfg.subblock_w) {
            return r.plan;
        }
    }
    return nullptr;
}

}  // namespace ttnn::prim::llama31_8b_matmul_bh
