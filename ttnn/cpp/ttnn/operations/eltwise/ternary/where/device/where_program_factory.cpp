// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "where_device_operation.hpp"
#include "where_utils.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include <cmath>

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using namespace ttnn::operations::ternary;

// For rank > 5 dims will be collapsed into a single dim
uint32_t extract_nD_dims(const ttnn::Tensor& x, const int out_rank) {
    const auto& shape = x.logical_shape();
    uint32_t nD_dim = 1;
    if (out_rank >= 6 && shape.rank() >= 6) {
        for (int i = -6; i >= -out_rank; --i) {
            auto dim = shape[i];
            nD_dim *= dim;
        }
    }
    return nD_dim;
}

template <typename F>
void set_or_update_runtime_arguments(
    tt::tt_metal::Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    tt::tt_metal::KernelHandle compute_kernel_id,
    CoreCoord compute_with_storage_grid_size,
    const WhereDeviceOperation::operation_attributes_t& operation_attributes,
    const WhereDeviceOperation::tensor_args_t& tensor_args,
    WhereDeviceOperation::tensor_return_value_t& output,
    WhereBroadcastType broadcast_type,
    F handle_args) {
    const auto& [predicate_tensor, value_true_tensor, value_false_tensor, optional_output_tensor] = tensor_args;

    WhereVariant variant = operation_attributes.where_variant;

    uint32_t num_output_tiles = output.physical_volume() / output.tensor_spec().tile().get_tile_hw();

    constexpr bool row_major = true;
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);
    constexpr size_t num_reader_args = 27;
    constexpr size_t num_writer_args = 3;
    constexpr size_t num_kernel_args = 4;

    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            handle_args(program, reader_kernel_id, core, std::array<uint32_t, num_reader_args>{0});
            handle_args(program, writer_kernel_id, core, std::array<uint32_t, num_writer_args>{0});
            handle_args(program, compute_kernel_id, core, std::array<uint32_t, num_kernel_args>{0});
            continue;
        }

        // Declare variables for TTT column broadcast case that will be used by both reader and writer
        uint32_t aD = 1, aN = 1, aC = 1, aHt = 1, aWt = 1, aND = 1;
        uint32_t bD = 1, bN = 1, bC = 1, bHt = 1, bWt = 1, bND = 1, b_num_tiles = 0;  // Initialize to 0 like binary_ng
        uint32_t fD = 1, fN = 1, fC = 1, fHt = 1, fWt = 1, fND = 1, f_num_tiles = 0;  // false tensor vars
        uint32_t cD = 1, cN = 1, cC = 1, cHt = 1, cWt = 1, cND = 1,
                 c_current_shard_width = 0;  // Initialize to 0 like binary_ng
        uint32_t a_num_tiles = 0;            // Initialize to 0 like binary_ng

        const auto out_rank = output.logical_shape().rank();
        aND = extract_nD_dims(predicate_tensor, out_rank);  // predicate nD
        cND = extract_nD_dims(output, out_rank);            // output nD

        const auto predicate_shape = predicate_tensor.padded_shape();
        const auto& output_shape = output.padded_shape();
        const auto& tile = output.tensor_spec().tile();

        // Get shape dims (D, N, C, Ht, Wt) for predicate (a)
        aD = predicate_shape.rank() >= 5 ? predicate_shape[-5] : 1;
        aN = predicate_shape[-4];
        aC = predicate_shape[-3];
        aHt = predicate_shape[-2] / tile.get_height();
        aWt = predicate_shape[-1] / tile.get_width();

        // Get shape dims for output (c)
        cD = output_shape.rank() >= 5 ? output_shape[-5] : 1;
        cN = output_shape[-4];
        cC = output_shape[-3];
        cHt = output_shape[-2] / tile.get_height();
        cWt = output_shape[-1] / tile.get_width();

        // Define has_sharding at higher scope so it's accessible in both variable init and reader args blocks
        bool has_sharding = false;
        // calculate has_sharding when support is added
        // has_sharding = predicate_tensor.memory_config().is_sharded() ||
        //                 value_true_tensor.value().memory_config().is_sharded() ||
        //                 value_false_tensor.value().memory_config().is_sharded() ||
        //                 output.memory_config().is_sharded();

        if (variant == WhereVariant::TTT) {
            // Initialize binary_ng style variables for TTT column broadcast
            bND = extract_nD_dims(value_true_tensor.value(), out_rank);   // value_true nD
            fND = extract_nD_dims(value_false_tensor.value(), out_rank);  // value_false nD

            // Extract shape dimensions using binary_ng approach
            const auto value_true_shape = value_true_tensor.value().padded_shape();
            const auto value_false_shape = value_false_tensor.value().padded_shape();

            // Get shape dims for value_true (b) - broadcast tensor
            bD = value_true_shape.rank() >= 5 ? value_true_shape[-5] : 1;
            bN = value_true_shape[-4];
            bC = value_true_shape[-3];
            bHt = value_true_shape[-2] / tile.get_height();
            bWt = value_true_shape[-1] / tile.get_width();

            // Get shape dims for value_false (f) - using actual false tensor shape
            fD = value_false_shape.rank() >= 5 ? value_false_shape[-5] : 1;
            fN = value_false_shape[-4];
            fC = value_false_shape[-3];
            fHt = value_false_shape[-2] / tile.get_height();
            fWt = value_false_shape[-1] / tile.get_width();

            // Match binary_ng logic: only set tile counts if sharding is enabled
            // For non-sharded (interleaved) mode, these remain 0 like binary_ng
            if (has_sharding) {
                a_num_tiles = aHt * aWt;  // predicate tiles per core
                b_num_tiles = bHt * bWt;  // value_true tiles per core
                f_num_tiles = fHt * fWt;  // value_false tiles per core
                c_current_shard_width = cWt;
            }
            // If not sharded, a_num_tiles, b_num_tiles, f_num_tiles, c_current_shard_width remain 0 (like binary_ng)
        }

        // Set reader runtime arguments based on variant
        if (variant == WhereVariant::TTS) {
            // TTS: predicate (arg 0) + value_true tensor (arg 1)
            // For column broadcast, read predicate to CB0 and true tensor to CB1
            if (broadcast_type == WhereBroadcastType::COL_BCAST) {
                has_sharding = predicate_tensor.memory_config().is_sharded() ||
                               value_true_tensor.value().memory_config().is_sharded() ||
                               output.memory_config().is_sharded();
            }

            // Initialize dimensions for TTS
            bND = extract_nD_dims(value_true_tensor.value(), out_rank);  // value_true nD

            // Extract shape dimensions for value_true (b) - TTS specific
            const auto value_true_shape = value_true_tensor.value().padded_shape();

            // Get shape dims for value_true (b) - TTS specific
            bD = value_true_shape.rank() >= 5 ? value_true_shape[-5] : 1;
            bN = value_true_shape[-4];
            bC = value_true_shape[-3];
            bHt = value_true_shape[-2] / tile.get_height();
            bWt = value_true_shape[-1] / tile.get_width();

            // Only set tile counts if sharding is enabled
            // For non-sharded (interleaved) mode, these remain 0
            if (has_sharding) {
                a_num_tiles = aHt * aWt;  // predicate tiles per core
                b_num_tiles = bHt * bWt;  // value_true tiles per core
                c_current_shard_width = cWt;
            }

            // Standard first 5 args + extended args for broadcast
            std::array<uint32_t, num_reader_args> reader_runtime_args{};  // zero-initialized

            // Standard first 5 arguments
            reader_runtime_args[0] = predicate_tensor.buffer()->address();           // 0: src0_addr (predicate)
            reader_runtime_args[1] = value_true_tensor.value().buffer()->address();  // 1: src1_addr (true tensor)
            reader_runtime_args[2] = 0u;                                             // 2: src2_addr (false tensor)
            reader_runtime_args[3] = num_tiles_per_core;                             // 3: num_tiles (per core)
            reader_runtime_args[4] = start_tile_id;                                  // 4: start_id

            // Extended broadcast arguments
            if (broadcast_type != WhereBroadcastType::NONE) {
                reader_runtime_args[5] = aHt * aWt * aC * aN * aD * (aND > 1);   // 5: nD_stride
                reader_runtime_args[6] = aHt * aWt * aC * aN * (aD > 1);         // 6: d_stride
                reader_runtime_args[7] = aHt * aWt * aC * (aN > 1);              // 7: n_stride
                reader_runtime_args[8] = aHt * aWt * (aC > 1);                   // 8: c_stride
                reader_runtime_args[9] = cD;                                     // 9: D
                reader_runtime_args[10] = cN;                                    // 10: N
                reader_runtime_args[11] = cC;                                    // 11: C
                reader_runtime_args[12] = cHt;                                   // 12: Ht
                reader_runtime_args[13] = cWt;                                   // 13: Wt
                reader_runtime_args[14] = cND;                                   // 14: cND
                reader_runtime_args[15] = bHt * bWt * bC * bN * bD * (bND > 1);  // 15: true_nD_stride
                reader_runtime_args[16] = bHt * bWt * bC * bN * (bD > 1);        // 16: true_d_stride
                reader_runtime_args[17] = bHt * bWt * bC * (bN > 1);             // 17: true_n_stride
                reader_runtime_args[18] = bHt * bWt * (bC > 1);                  // 18: true_c_stride
                reader_runtime_args[19] = b_num_tiles;                           // 19: true_num_tiles

                // False is scalar, so no strides needed
                reader_runtime_args[20] = 0u;                     // 20: false_nD_stride
                reader_runtime_args[21] = 0u;                     // 21: false_d_stride
                reader_runtime_args[22] = 0u;                     // 22: false_n_stride
                reader_runtime_args[23] = 0u;                     // 23: false_c_stride
                reader_runtime_args[24] = 0u;                     // 24: false_num_tiles
                reader_runtime_args[25] = c_current_shard_width;  // 25: dst_shard_width
                reader_runtime_args[26] = a_num_tiles;            // 26: src_num_tiles (predicate)
            }
            handle_args(program, reader_kernel_id, core, reader_runtime_args);
        } else if (variant == WhereVariant::TST) {
            // TST: predicate (arg 0) + value_false tensor (arg 1, maps to c_1)
            if (broadcast_type == WhereBroadcastType::COL_BCAST) {
                has_sharding = predicate_tensor.memory_config().is_sharded() ||
                               value_false_tensor.value().memory_config().is_sharded() ||
                               output.memory_config().is_sharded();
            }

            fND = extract_nD_dims(value_false_tensor.value(), out_rank);  // value_false nD
            const auto value_false_shape = value_false_tensor.value().padded_shape();

            // Get shape dims for value_false (f) - using false_tensor's shape
            fD = value_false_shape.rank() >= 5 ? value_false_shape[-5] : 1;
            fN = value_false_shape[-4];
            fC = value_false_shape[-3];
            fHt = value_false_shape[-2] / tile.get_height();
            fWt = value_false_shape[-1] / tile.get_width();

            if (has_sharding) {
                a_num_tiles = aHt * aWt;  // predicate tiles per core
                f_num_tiles = fHt * fWt;  // value_false tiles per core
                c_current_shard_width = cWt;
            }

            // Standard first 5 args + extended args for broadcast
            std::array<uint32_t, num_reader_args> reader_runtime_args{};  // zero-initialized

            // Standard first 5 arguments
            reader_runtime_args[0] = predicate_tensor.buffer()->address();            // 0: src0_addr (predicate)
            reader_runtime_args[1] = value_false_tensor.value().buffer()->address();  // 1: src1_addr (false tensor)
            reader_runtime_args[2] = 0u;                                              // 2: src2_addr
            reader_runtime_args[3] = num_tiles_per_core;                              // 3: num_tiles (per core)
            reader_runtime_args[4] = start_tile_id;                                   // 4: start_id

            // Extended broadcast arguments
            if (broadcast_type != WhereBroadcastType::NONE) {
                reader_runtime_args[5] = aHt * aWt * aC * aN * aD * (aND > 1);   // 5: nD_stride
                reader_runtime_args[6] = aHt * aWt * aC * aN * (aD > 1);         // 6: d_stride
                reader_runtime_args[7] = aHt * aWt * aC * (aN > 1);              // 7: n_stride
                reader_runtime_args[8] = aHt * aWt * (aC > 1);                   // 8: c_stride
                reader_runtime_args[9] = cD;                                     // 9: D
                reader_runtime_args[10] = cN;                                    // 10: N
                reader_runtime_args[11] = cC;                                    // 11: C
                reader_runtime_args[12] = cHt;                                   // 12: Ht
                reader_runtime_args[13] = cWt;                                   // 13: Wt
                reader_runtime_args[14] = cND;                                   // 14: cND
                reader_runtime_args[15] = fHt * fWt * fC * fN * fD * (fND > 1);  // 15: false_nD_stride
                reader_runtime_args[16] = fHt * fWt * fC * fN * (fD > 1);        // 16: false_d_stride
                reader_runtime_args[17] = fHt * fWt * fC * (fN > 1);             // 17: false_n_stride
                reader_runtime_args[18] = fHt * fWt * (fC > 1);                  // 18: false_c_stride
                reader_runtime_args[19] = f_num_tiles;                           // 19: false_num_tiles
                reader_runtime_args[20] = 0u;                                    // 20: true_nD_stride (zero for TST)
                reader_runtime_args[21] = 0u;                                    // 21: true_d_stride (zero for TST)
                reader_runtime_args[22] = 0u;                                    // 22: true_n_stride (zero for TST)
                reader_runtime_args[23] = 0u;                                    // 23: true_c_stride (zero for TST)
                reader_runtime_args[24] = 0u;                                    // 24: true_num_tiles (zero for TST)
                reader_runtime_args[25] = c_current_shard_width;                 // 25: dst_shard_width
                reader_runtime_args[26] = a_num_tiles;                           // 26: src_num_tiles (predicate)
            }
            handle_args(program, reader_kernel_id, core, reader_runtime_args);
        } else if (variant == WhereVariant::TTT) {
            uint32_t c_start_id = 0;
            if (has_sharding) {
                // Match binary_ng sharding logic for c_start_id calculation
                // NOTE: This requires shard shape info that may need to be implemented
                c_start_id = start_tile_id;  // For now, fallback to start_tile_id until shard logic is implemented
            } else {
                c_start_id = start_tile_id;
            }

            // Standard first 5 args + extended args broadcast
            std::array<uint32_t, num_reader_args> reader_runtime_args{};  // zero-initialized

            // Standard first 5 arguments
            reader_runtime_args[0] = predicate_tensor.buffer()->address();            // 0: src0_addr (predicate)
            reader_runtime_args[1] = value_true_tensor.value().buffer()->address();   // 1: src1_addr (true tensor)
            reader_runtime_args[2] = value_false_tensor.value().buffer()->address();  // 2: src2_addr (false tensor)
            reader_runtime_args[3] = num_tiles_per_core;                              // 3: num_tiles (per core)
            reader_runtime_args[4] = c_start_id;                                      // 4: start_id

            // Extended broadcast arguments (only when broadcast != NONE)
            if (broadcast_type != WhereBroadcastType::NONE) {
                reader_runtime_args[5] = aHt * aWt * aC * aN * aD * (aND > 1);   // 5: nD_stride
                reader_runtime_args[6] = aHt * aWt * aC * aN * (aD > 1);         // 6: d_stride
                reader_runtime_args[7] = aHt * aWt * aC * (aN > 1);              // 7: n_stride
                reader_runtime_args[8] = aHt * aWt * (aC > 1);                   // 8: c_stride
                reader_runtime_args[9] = cD;                                     // 9: D
                reader_runtime_args[10] = cN;                                    // 10: N
                reader_runtime_args[11] = cC;                                    // 11: C
                reader_runtime_args[12] = cHt;                                   // 12: Ht
                reader_runtime_args[13] = cWt;                                   // 13: Wt
                reader_runtime_args[14] = cND;                                   // 14: cND
                reader_runtime_args[15] = bHt * bWt * bC * bN * bD * (bND > 1);  // 15: true_nD_stride
                reader_runtime_args[16] = bHt * bWt * bC * bN * (bD > 1);        // 16: true_d_stride
                reader_runtime_args[17] = bHt * bWt * bC * (bN > 1);             // 17: true_n_stride
                reader_runtime_args[18] = bHt * bWt * (bC > 1);                  // 18: true_c_stride
                reader_runtime_args[19] = b_num_tiles;                           // 19: true_num_tiles
                reader_runtime_args[20] = fHt * fWt * fC * fN * fD * (fND > 1);  // 20: false_nD_stride
                reader_runtime_args[21] = fHt * fWt * fC * fN * (fD > 1);        // 21: false_d_stride
                reader_runtime_args[22] = fHt * fWt * fC * (fN > 1);             // 22: false_n_stride
                reader_runtime_args[23] = fHt * fWt * (fC > 1);                  // 23: false_c_stride
                reader_runtime_args[24] = f_num_tiles;                           // 24: false_num_tiles
                reader_runtime_args[25] = c_current_shard_width;                 // 25: dst_shard_width
                reader_runtime_args[26] = a_num_tiles;                           // 26: src_num_tiles (predicate)
            }

            handle_args(program, reader_kernel_id, core, reader_runtime_args);
        } else {
            TT_THROW("Unsupported Where variant in WhereDeviceOperation. Supported: TTS, TST, TTT");
        }

        // Writer runtime args - use simple unary format (3 args: dst_addr, num_tiles, start_id)
        std::array writer_runtime_args = {
            output.buffer()->address(),  // dst_addr
            num_tiles_per_core,          // num_tiles
            start_tile_id                // start_id
        };
        handle_args(program, writer_kernel_id, core, writer_runtime_args);

        // Compute runtime args - binary_ng style for TTT column broadcast
        if (variant == WhereVariant::TTT) {
            if (broadcast_type == WhereBroadcastType::COL_BCAST) {
                // Calculate freq and counter like binary_ng for column broadcast
                uint32_t start_t = start_tile_id % (cHt * cWt);
                uint32_t start_tw = start_t % cWt;
                uint32_t freq = cWt;          // Column broadcast frequency
                uint32_t counter = start_tw;  // Column broadcast counter

                std::array compute_runtime_args = {num_tiles_per_core, freq, counter, 0u};
                handle_args(program, compute_kernel_id, core, compute_runtime_args);
            } else if (broadcast_type == WhereBroadcastType::SCALAR_BCAST) {
                // Calculate freq and counter like binary_ng for scalar broadcast
                uint32_t start_t = start_tile_id % (cHt * cWt);
                uint32_t freq = cHt * cWt;   // scalar broadcast frequency
                uint32_t counter = start_t;  // scalar broadcast counter

                std::array compute_runtime_args = {num_tiles_per_core, freq, counter, 0u};
                handle_args(program, compute_kernel_id, core, compute_runtime_args);
            } else {
                // TTT variant without subtile bcast and ROW_BCAST is also handled here
                std::array compute_runtime_args = {num_tiles_per_core, 0u, 0u, 0u};
                handle_args(program, compute_kernel_id, core, compute_runtime_args);
            }

        } else if (variant == WhereVariant::TTS) {

            auto bit_cast_scalar =
                pack_scalar_runtime_arg(operation_attributes.value_false_scalar.value(), output.dtype());

            if (broadcast_type == WhereBroadcastType::COL_BCAST) {
                // For TTS column broadcast, calculate freq and counter for dedicated kernel
                uint32_t start_t = start_tile_id % (cHt * cWt);
                uint32_t start_tw = start_t % cWt;

                uint32_t freq = cWt;
                uint32_t counter = start_tw;

                std::array compute_runtime_args = {num_tiles_per_core, freq, counter, bit_cast_scalar};
                handle_args(program, compute_kernel_id, core, compute_runtime_args);
            } else if (
                broadcast_type == WhereBroadcastType::SCALAR_A_BCAST ||
                broadcast_type == WhereBroadcastType::SCALAR_B_BCAST) {
                // Calculate freq and counter like binary_ng for scalar broadcast
                auto HtWt = cHt * cWt;
                uint32_t start_t = start_tile_id % HtWt;
                uint32_t freq = HtWt;        // scalar broadcast frequency
                uint32_t counter = start_t;  // scalar broadcast counter

                std::array compute_runtime_args = {num_tiles_per_core, freq, counter, bit_cast_scalar};
                handle_args(program, compute_kernel_id, core, compute_runtime_args);
            } else {
                std::array compute_runtime_args = {num_tiles_per_core, bit_cast_scalar, 0u, 0u};
                handle_args(program, compute_kernel_id, core, compute_runtime_args);
            }
        } else if (variant == WhereVariant::TST) {
            auto bit_cast_scalar =
                    pack_scalar_runtime_arg(operation_attributes.value_true_scalar.value(), output.dtype());
            if (broadcast_type == WhereBroadcastType::COL_BCAST) {
                // For TST column broadcast, calculate freq and counter for dedicated kernel
                uint32_t start_t = start_tile_id % (cHt * cWt);
                uint32_t start_tw = start_t % cWt;

                uint32_t freq = cWt;
                uint32_t counter = start_tw;

                std::array compute_runtime_args = {num_tiles_per_core, freq, counter, bit_cast_scalar};
                handle_args(program, compute_kernel_id, core, compute_runtime_args);
            } else if (
                broadcast_type == WhereBroadcastType::SCALAR_A_BCAST ||
                broadcast_type == WhereBroadcastType::SCALAR_B_BCAST) {
                // Calculate freq and counter like binary_ng for scalar broadcast
                auto HtWt = cHt * cWt;
                uint32_t start_t = start_tile_id % HtWt;
                uint32_t freq = HtWt;        // scalar broadcast frequency
                uint32_t counter = start_t;  // scalar broadcast counter
                std::array compute_runtime_args = {num_tiles_per_core, freq, counter, bit_cast_scalar};
                handle_args(program, compute_kernel_id, core, compute_runtime_args);
            } else {
                std::array compute_runtime_args = {num_tiles_per_core, bit_cast_scalar, 0u, 0u};
                handle_args(program, compute_kernel_id, core, compute_runtime_args);
            }
        }
        start_tile_id += num_tiles_per_core;
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::ternary {
WhereDeviceOperation::WhereProgramFactory::cached_program_t WhereDeviceOperation::WhereProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& [predicate_tensor, value_true_tensor, value_false_tensor, optional_output_tensor] = tensor_args;

    // Declare broadcast detection variable at function level
    bool is_height_bcast = false;  // Track if this is height broadcasting for TTS

    WhereVariant variant = operation_attributes.where_variant;
    WhereBroadcastType broadcast_type = operation_attributes.broadcast_type;
    bool pred_is_bcast = false, true_is_bcast = false, false_is_bcast = false;

    // Use WhereKernelConfig to get the appropriate kernel names
    WhereKernelConfig kernel_config(variant, broadcast_type);

    auto program = CreateProgram();

    auto* device = predicate_tensor.device();

    auto predicate_data_format = datatype_to_dataformat_converter(predicate_tensor.dtype());
    // (predicate_tensor.dtype() == DataType::BFLOAT16) ? DataType::UINT16 : predicate_tensor.dtype());

    // Handle data formats based on variant and tensor availability
    DataFormat value_true_data_format, value_false_data_format;
    if (variant == WhereVariant::TTS) {
        // TTS: only value_true tensor exists
        value_true_data_format = datatype_to_dataformat_converter(value_true_tensor.value().dtype());

        // the bfloat16 impl of where_llk uses UINT16 instr set.
        // If the bfloat16 inputs' CBs are set to UINT16 dataformat this will enable us to get 'NaN' for bfloat16 dtype
        // We need to test the impact of this on the composite ops that use where op and on the models, since bfloat16
        // packs nan as inf in all other ops.

        // (value_true_tensor.value().dtype() == DataType::BFLOAT16) ? DataType::UINT16
        //                                                           : value_true_tensor.value().dtype());

        // Use predicate format as fallback for value_false
        value_false_data_format = predicate_data_format;
    } else if (variant == WhereVariant::TST) {
        // TST: only value_false tensor exists
        value_false_data_format = datatype_to_dataformat_converter(value_false_tensor.value().dtype());
        // (value_false_tensor.value().dtype() == DataType::BFLOAT16) ? DataType::UINT16
        //                                                            : value_false_tensor.value().dtype());
        // Use predicate format as fallback for value_true
        value_true_data_format = predicate_data_format;
    } else {
        // TTT: both tensors exist
        value_true_data_format = datatype_to_dataformat_converter(value_true_tensor.value().dtype());
        // (value_true_tensor.value().dtype() == DataType::BFLOAT16) ? DataType::UINT16
        //                                                           : value_true_tensor.value().dtype());
        value_false_data_format = datatype_to_dataformat_converter(value_false_tensor.value().dtype());
        // (value_false_tensor.value().dtype() == DataType::BFLOAT16) ? DataType::UINT16
        //                                                            : value_false_tensor.value().dtype());
    }

    auto output_data_format = datatype_to_dataformat_converter(output.dtype());
    // datatype_to_dataformat_converter((output.dtype() == DataType::BFLOAT16) ? DataType::UINT16 : output.dtype());

    uint32_t predicate_single_tile_size = tt_metal::detail::TileSize(predicate_data_format);
    uint32_t value_true_single_tile_size = tt_metal::detail::TileSize(value_true_data_format);
    uint32_t value_false_single_tile_size = tt_metal::detail::TileSize(value_false_data_format);
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_data_format);

    // we parallelize the computation across the output tiles
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // Number of tiles to store per input CB (double buffer)
    constexpr uint32_t num_tiles_per_cb = 2;

    // Input buffers - Create predicate CB (always c_0)
    auto [predicate_tensor_cb, predicate_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_0,
        program,
        all_device_cores,
        predicate_single_tile_size,
        num_tiles_per_cb,
        predicate_data_format);  // predicate_tensor

    // Create c_1 based on variant - this is the primary tensor CB
    uint32_t value_true_tensor_cb = 0;
    tt::tt_metal::CBHandle value_true_tensor_cb_handle;
    uint32_t value_false_tensor_cb = 0;
    tt::tt_metal::CBHandle value_false_tensor_cb_handle;

    if (variant == WhereVariant::TTS) {
        // TTS: c_1 = value_true tensor (value_false is scalar)
        auto [cb, cb_handle] = create_cb(
            tt::CBIndex::c_1,
            program,
            all_device_cores,
            value_true_single_tile_size,
            num_tiles_per_cb,
            value_true_data_format);
        value_true_tensor_cb = cb;
        value_true_tensor_cb_handle = cb_handle;
    } else if (variant == WhereVariant::TST) {
        // TST: c_1 = value_false tensor (value_true is scalar)
        auto [cb, cb_handle] = create_cb(
            tt::CBIndex::c_1,
            program,
            all_device_cores,
            value_false_single_tile_size,
            num_tiles_per_cb,
            value_false_data_format);
        value_false_tensor_cb = cb;
        value_false_tensor_cb_handle = cb_handle;
    } else if (variant == WhereVariant::TTT) {
        auto [cb1, cb1_handle] = create_cb(
            tt::CBIndex::c_1,
            program,
            all_device_cores,
            value_true_single_tile_size,
            num_tiles_per_cb,
            value_true_data_format);
        value_true_tensor_cb = cb1;
        value_true_tensor_cb_handle = cb1_handle;

        // Create CB for value_false (using actual false tensor)
        auto [cb2, cb2_handle] = create_cb(
            tt::CBIndex::c_2,
            program,
            all_device_cores,
            value_false_single_tile_size,  // Using actual false tensor size
            num_tiles_per_cb,
            value_false_data_format);  // Using actual false tensor format
        value_false_tensor_cb = cb2;
        value_false_tensor_cb_handle = cb2_handle;
    } else {
        TT_THROW("Unsupported Where variant in WhereDeviceOperation. Supported: TTS, TST, TTT");
    }

    // Output buffer - use c_3 for all cases now
    auto output_cb_index = tt::CBIndex::c_3;
    auto [output_tensor_cb, output_tensor_cb_handle] = create_cb(
        output_cb_index,
        program,
        all_device_cores,
        output_single_tile_size,
        num_tiles_per_cb,
        output_data_format);  // output

    // Handle DRAM flags based on variant and tensor availability
    uint32_t value_true_is_dram = 0, value_false_is_dram = 0;
    if (variant == WhereVariant::TTS) {
        value_true_is_dram =
            static_cast<uint32_t>(value_true_tensor.value().buffer()->buffer_type() == tt_metal::BufferType::DRAM);
    } else if (variant == WhereVariant::TST) {
        value_false_is_dram =
            static_cast<uint32_t>(value_false_tensor.value().buffer()->buffer_type() == tt_metal::BufferType::DRAM);
    } else {
        value_true_is_dram =
            static_cast<uint32_t>(value_true_tensor.value().buffer()->buffer_type() == tt_metal::BufferType::DRAM);
        value_false_is_dram =
            static_cast<uint32_t>(value_false_tensor.value().buffer()->buffer_type() == tt_metal::BufferType::DRAM);
    }

    // BROADCAST DETECTION - Common for both reader and compute kernels
    // Variables are declared at function level, just initialize them here
    pred_is_bcast = false;
    true_is_bcast = false;
    false_is_bcast = false;
    is_height_bcast = false;  // Initialize to false for broadcast detection
    if (broadcast_type == WhereBroadcastType::COL_BCAST) {
        // Determine which tensor is actually broadcast based on logical shapes (not padded)
        auto pred_shape = predicate_tensor.logical_shape();

        if (variant == WhereVariant::TTS) {
            // For TTS: only predicate and true tensor, false is scalar
            auto true_shape = value_true_tensor.value().logical_shape();
            auto pred_w = pred_shape[pred_shape.rank() - 1];
            auto true_w = true_shape[true_shape.rank() - 1];
            auto pred_h = pred_shape[pred_shape.rank() - 2];
            auto true_h = true_shape[true_shape.rank() - 2];

            // Check for width or height broadcasting
            pred_is_bcast = (pred_w == 1 && true_w > 1) || (pred_h == 1 && true_h > 1);
            true_is_bcast = (true_w == 1 && pred_w > 1) || (true_h == 1 && pred_h > 1);

            // Determine if this is height broadcasting
            is_height_bcast = true_is_bcast && (true_h == 1 && pred_h > 1);

            // Check for multi-dimensional broadcasting: if ranks differ, true tensor needs broadcasting
            if (pred_shape.rank() != true_shape.rank()) {
                true_is_bcast = true;
            }

            false_is_bcast = false;  // False is scalar for TTS
        } else if (variant == WhereVariant::TST) {
            // For TST: only predicate and false tensor, true is scalar
            auto false_shape = value_false_tensor.value().logical_shape();
            auto pred_w = pred_shape[pred_shape.rank() - 1];
            auto false_w = false_shape[false_shape.rank() - 1];
            auto pred_h = pred_shape[pred_shape.rank() - 2];
            auto false_h = false_shape[false_shape.rank() - 2];

            // Check for width or height broadcasting
            pred_is_bcast = (pred_w == 1 && false_w > 1) || (pred_h == 1 && false_h > 1);
            false_is_bcast = (false_w == 1 && pred_w > 1) || (false_h == 1 && pred_h > 1);

            // Check for multi-dimensional broadcasting: if ranks differ, false tensor needs broadcasting
            if (pred_shape.rank() != false_shape.rank()) {
                false_is_bcast = true;
            }

            true_is_bcast = false;  // True is scalar for TST
        } else {
            // TTT case
            auto true_shape = value_true_tensor.value().logical_shape();
            auto false_shape = value_false_tensor.value().logical_shape();
            auto pred_w = pred_shape[pred_shape.rank() - 1];
            auto true_w = true_shape[true_shape.rank() - 1];
            auto false_w = false_shape[false_shape.rank() - 1];

            pred_is_bcast = (pred_w == 1 && (true_w > 1 || false_w > 1));
            true_is_bcast = (true_w == 1 && (pred_w > 1 || false_w > 1));
            false_is_bcast = (false_w == 1 && (pred_w > 1 || true_w > 1));
        }
    } else if (broadcast_type == WhereBroadcastType::ROW_BCAST) {
        // Determine which tensor is actually broadcast based on logical shapes (not padded)
        auto pred_shape = predicate_tensor.logical_shape();
        auto true_shape = value_true_tensor.value().logical_shape();

        if (variant == WhereVariant::TTT) {
            // TTT case - check height dimension (second-to-last)
            auto false_shape = value_false_tensor.value().logical_shape();
            auto pred_h = pred_shape[pred_shape.rank() - 2];
            auto true_h = true_shape[true_shape.rank() - 2];
            auto false_h = false_shape[false_shape.rank() - 2];

            pred_is_bcast = (pred_h == 1 && (true_h > 1 || false_h > 1));
            true_is_bcast = (true_h == 1 && (pred_h > 1 || false_h > 1));
            false_is_bcast = (false_h == 1 && (pred_h > 1 || true_h > 1));
        } else {
            // TTS case - not currently supported for row broadcast
            pred_is_bcast = false;
            true_is_bcast = false;
            false_is_bcast = false;
        }
    } else if (broadcast_type == WhereBroadcastType::SCALAR_BCAST) {
        // Determine which tensor is actually broadcast based on logical shapes (not padded)
        auto pred_shape = predicate_tensor.logical_shape();
        auto true_shape = value_true_tensor.value().logical_shape();
        auto false_shape = value_false_tensor.value().logical_shape();

        auto pred_w = pred_shape[-1];
        auto true_w = true_shape[-1];
        auto false_w = false_shape[-1];

        auto pred_h = pred_shape[-2];  // height (second-to-last dimension)
        auto true_h = true_shape[-2];
        auto false_h = false_shape[-2];

        auto max_h = std::max({pred_h, true_h, false_h});
        auto max_w = std::max({pred_w, true_w, false_w});

        bool pred_row_bcast = (pred_h == 1 && max_h > 1);
        bool true_row_bcast = (true_h == 1 && max_h > 1);
        bool false_row_bcast = (false_h == 1 && max_h > 1);

        bool pred_col_bcast = (pred_w == 1 && max_w > 1);
        bool true_col_bcast = (true_w == 1 && max_w > 1);
        bool false_col_bcast = (false_w == 1 && max_w > 1);

        pred_is_bcast = (pred_row_bcast && pred_col_bcast);
        true_is_bcast = (true_row_bcast && true_col_bcast);
        false_is_bcast = (false_row_bcast && false_col_bcast);
    }

    // READER KERNEL - Use kernel path from utils
    // Create dataflow defines for column broadcast kernels like binary_ng
    std::map<std::string, std::string> reader_defines;
    if (variant == WhereVariant::TTT &&
        (broadcast_type == WhereBroadcastType::COL_BCAST || broadcast_type == WhereBroadcastType::SCALAR_BCAST)) {
        // Use binary_ng style dataflow defines with predicate and value_true dtypes
        reader_defines = make_dataflow_defines(
            predicate_tensor.dtype(),
            value_true_tensor.value().dtype(),
            value_false_tensor.value().dtype());  // For predicate (a) and value_true (b)

        // Add binary_ng style sharding defines
        bool predicate_sharded = predicate_tensor.memory_config().is_sharded();
        bool value_true_sharded = value_true_tensor.value().memory_config().is_sharded();
        bool value_false_sharded = value_false_tensor.value().memory_config().is_sharded();
        reader_defines["SRC_SHARDED_PREDICATE"] = predicate_sharded ? "1" : "0";
        reader_defines["SRC_SHARDED_TRUE"] = value_true_sharded ? "1" : "0";
        reader_defines["SRC_SHARDED_FALSE"] = value_false_sharded ? "1" : "0";

        // Set broadcast defines based on actual detection
        reader_defines["SRC_BCAST_PREDICATE"] = pred_is_bcast ? "1" : "0";
        reader_defines["SRC_BCAST_TRUE"] = true_is_bcast ? "1" : "0";
        reader_defines["SRC_BCAST_FALSE"] = false_is_bcast ? "1" : "0";

        // Add BCAST_LLK define (set to 0 for now, can be optimized later)
        reader_defines["BCAST_LLK"] = "0";
    } else if (broadcast_type == WhereBroadcastType::ROW_BCAST) {
        // ROW_BCAST: need dataflow defines for FILL_TILE_WITH_FIRST_ROW_B etc.
        reader_defines = make_dataflow_defines(
            predicate_tensor.dtype(),
            value_true_tensor.value().dtype(),
            value_false_tensor.value().dtype());  // For predicate (a) and value_true (b)

        bool predicate_sharded = predicate_tensor.memory_config().is_sharded();
        bool value_true_sharded = value_true_tensor.value().memory_config().is_sharded();
        bool value_false_sharded = value_false_tensor.value().memory_config().is_sharded();
        reader_defines["SRC_SHARDED_A"] = predicate_sharded ? "1" : "0";    // CB0 sharding
        reader_defines["SRC_SHARDED_B"] = value_true_sharded ? "1" : "0";   // CB1 sharding
        reader_defines["SRC_SHARDED_C"] = value_false_sharded ? "1" : "0";  // CB2 sharding

        // Set broadcast defines to match ternary reader kernel expectations
        // CB0 = predicate, CB1 = true tensor, CB2 = false tensor
        reader_defines["SRC_BCAST_A"] = pred_is_bcast ? "1" : "0";   // First tensor (CB0)
        reader_defines["SRC_BCAST_B"] = true_is_bcast ? "1" : "0";   // Second tensor (CB1)
        reader_defines["SRC_BCAST_C"] = false_is_bcast ? "1" : "0";  // Third tensor (CB2)

        reader_defines["BCAST_LLK"] = "0";
    } else if (variant == WhereVariant::TTS && broadcast_type == WhereBroadcastType::COL_BCAST) {
        // TTS column broadcast
        reader_defines = make_dataflow_defines(
            predicate_tensor.dtype(),
            value_true_tensor.value().dtype(),  // CB1 uses true tensor dtype
            predicate_tensor.dtype());          // For predicate (a), true (b), false is scalar

        // Add basic sharding defines
        bool predicate_sharded = predicate_tensor.memory_config().is_sharded();
        bool true_sharded = value_true_tensor.value().memory_config().is_sharded();
        reader_defines["SRC_SHARDED_PREDICATE"] = predicate_sharded ? "1" : "0";
        reader_defines["SRC_SHARDED_TRUE"] = true_sharded ? "1" : "0";
        reader_defines["SRC_SHARDED_FALSE"] = "0";  // False is scalar for TTS

        // Set broadcast defines for TTS column broadcast
        reader_defines["SRC_BCAST_PREDICATE"] = pred_is_bcast ? "1" : "0";
        reader_defines["SRC_BCAST_TRUE"] = true_is_bcast ? "1" : "0";  // CB1 uses true tensor
        reader_defines["SRC_BCAST_FALSE"] = "0";                       // False is scalar

        // Add BCAST_LLK define
        reader_defines["BCAST_LLK"] = "0";
    } else if (variant == WhereVariant::TST && broadcast_type == WhereBroadcastType::COL_BCAST) {
        // TST column broadcast
        reader_defines = make_dataflow_defines(
            predicate_tensor.dtype(),
            value_false_tensor.value().dtype(),  // For predicate (a), true is scalar
            predicate_tensor.dtype());           // CB1 uses false tensor dtype

        // Add basic sharding defines
        bool predicate_sharded = predicate_tensor.memory_config().is_sharded();
        bool false_sharded = value_false_tensor.value().memory_config().is_sharded();
        reader_defines["SRC_SHARDED_PREDICATE"] = predicate_sharded ? "1" : "0";
        reader_defines["SRC_SHARDED_TRUE"] = "0";  // True is scalar for TST
        reader_defines["SRC_SHARDED_FALSE"] = false_sharded ? "1" : "0";

        // Set broadcast defines for TST column broadcast
        reader_defines["SRC_BCAST_PREDICATE"] = pred_is_bcast ? "1" : "0";
        reader_defines["SRC_BCAST_TRUE"] = "0";                          // True is scalar
        reader_defines["SRC_BCAST_FALSE"] = false_is_bcast ? "1" : "0";  // CB1 uses false tensor

        // Add BCAST_LLK define
        reader_defines["BCAST_LLK"] = "0";
    }

    if (variant == WhereVariant::TTT && (broadcast_type == WhereBroadcastType::OUTER_BCAST)) {
        // TODO: Use the sharding config from the tensor args when sharding support is added
        bool predicate_sharded = predicate_tensor.memory_config().is_sharded();
        bool value_true_sharded = value_true_tensor.value().memory_config().is_sharded();
        bool value_false_sharded = value_false_tensor.value().memory_config().is_sharded();
        reader_defines["SRC_SHARDED_A"] = predicate_sharded ? "1" : "0";
        reader_defines["SRC_SHARDED_B"] = value_true_sharded ? "1" : "0";
        reader_defines["SRC_SHARDED_C"] = value_false_sharded ? "1" : "0";
    }
    if (variant == WhereVariant::TTS && broadcast_type == WhereBroadcastType::OUTER_BCAST) {
        // TODO: Use the sharding config from the tensor args when sharding support is added
        bool predicate_sharded = predicate_tensor.memory_config().is_sharded();
        bool value_true_sharded = value_true_tensor.value().memory_config().is_sharded();
        reader_defines["SRC_SHARDED_A"] = predicate_sharded ? "1" : "0";
        reader_defines["SRC_SHARDED_B"] = value_true_sharded ? "1" : "0";
    }
    if (variant == WhereVariant::TST && broadcast_type == WhereBroadcastType::OUTER_BCAST) {
        // TODO: Use the sharding config from the tensor args when sharding support is added
        bool predicate_sharded = predicate_tensor.memory_config().is_sharded();
        bool value_false_sharded = value_false_tensor.value().memory_config().is_sharded();
        reader_defines["SRC_SHARDED_A"] = predicate_sharded ? "1" : "0";
        reader_defines["SRC_SHARDED_B"] = value_false_sharded ? "1" : "0";
    }
    if (variant == WhereVariant::TTS && (broadcast_type == WhereBroadcastType::SCALAR_A_BCAST ||
                                         broadcast_type == WhereBroadcastType::SCALAR_B_BCAST)) {
        // TODO: Use the sharding config from the tensor args when sharding support is added
        reader_defines = make_dataflow_defines(predicate_tensor.dtype(), value_true_tensor.value().dtype());

        bool predicate_sharded = predicate_tensor.memory_config().is_sharded();
        bool value_true_sharded = value_true_tensor.value().memory_config().is_sharded();
        reader_defines["SRC_SHARDED_A"] = predicate_sharded ? "1" : "0";
        reader_defines["SRC_SHARDED_B"] = value_true_sharded ? "1" : "0";
        reader_defines["SRC_BCAST_A"] =
            (broadcast_type == WhereBroadcastType::SCALAR_A_BCAST) ? "1" : "0";  // First tensor (CB0)
        reader_defines["SRC_BCAST_B"] =
            (broadcast_type == WhereBroadcastType::SCALAR_B_BCAST) ? "1" : "0";  // Second tensor (CB1)
    }
    if (variant == WhereVariant::TST && (broadcast_type == WhereBroadcastType::SCALAR_A_BCAST ||
                                         broadcast_type == WhereBroadcastType::SCALAR_B_BCAST)) {
        // TODO: Use the sharding config from the tensor args when sharding support is added
        reader_defines = make_dataflow_defines(predicate_tensor.dtype(), value_false_tensor.value().dtype());

        bool predicate_sharded = predicate_tensor.memory_config().is_sharded();
        bool value_false_sharded = value_false_tensor.value().memory_config().is_sharded();
        reader_defines["SRC_SHARDED_A"] = predicate_sharded ? "1" : "0";
        reader_defines["SRC_SHARDED_B"] = value_false_sharded ? "1" : "0";
        reader_defines["SRC_BCAST_A"] =
            (broadcast_type == WhereBroadcastType::SCALAR_A_BCAST) ? "1" : "0";  // First tensor (CB0)
        reader_defines["SRC_BCAST_B"] =
            (broadcast_type == WhereBroadcastType::SCALAR_B_BCAST) ? "1" : "0";  // Second tensor (CB1)
    }

    tt_metal::ReaderDataMovementConfig reader_config;

    if (variant == WhereVariant::TTS) {
        // TTS: c_0 = predicate, c_1 = value_true tensor
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t)predicate_tensor_cb, (std::uint32_t)value_true_tensor_cb};

        TensorAccessorArgs(*predicate_tensor.buffer()).append_to(reader_compile_time_args);
        TensorAccessorArgs(*value_true_tensor.value().buffer()).append_to(reader_compile_time_args);
        reader_config = tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines);

    } else if (variant == WhereVariant::TST) {
        // TST: c_0 = predicate, c_1 = value_false tensor
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t)predicate_tensor_cb, (std::uint32_t)value_false_tensor_cb};
        TensorAccessorArgs(*predicate_tensor.buffer()).append_to(reader_compile_time_args);
        TensorAccessorArgs(*value_false_tensor.value().buffer()).append_to(reader_compile_time_args);
        reader_config = tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines);
    } else if (variant == WhereVariant::TTT) {
        // TTT: c_0 = predicate, c_1 = value_true, c_2 = value_false
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t)predicate_tensor_cb,
            (std::uint32_t)value_true_tensor_cb,
            (std::uint32_t)value_false_tensor_cb};
        TensorAccessorArgs(*predicate_tensor.buffer()).append_to(reader_compile_time_args);
        TensorAccessorArgs(*value_true_tensor.value().buffer()).append_to(reader_compile_time_args);
        TensorAccessorArgs(*value_false_tensor.value().buffer()).append_to(reader_compile_time_args);
        reader_config = tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines);
    }

    auto reader_kernel_id = tt_metal::CreateKernel(
        program, get_kernel_file_path(kernel_config.reader_kernel), all_device_cores, reader_config);

    // Use unary writer config for all cases (consistent with other writer variants)
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_tensor_cb};
    tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);
    tt_metal::WriterDataMovementConfig writer_config = tt_metal::WriterDataMovementConfig(writer_compile_time_args);

    auto writer_kernel_id = tt_metal::CreateKernel(
        program, get_kernel_file_path(kernel_config.writer_kernel), all_device_cores, writer_config);

    // COMPUTE KERNEL - Use kernel path from utils
    bool fp32_dest_acc_en = output_data_format == tt::DataFormat::UInt32 ||
                            output_data_format == tt::DataFormat::Int32 ||
                            output_data_format == tt::DataFormat::Float32;

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);

    // c_0 is always predicate
    unpack_to_dest_mode[tt::CBIndex::c_0] = (predicate_tensor.dtype() == DataType::FLOAT32)
                                                ? UnpackToDestMode::UnpackToDestFp32
                                                : UnpackToDestMode::Default;

    // c_1 assignment depends on variant
    if (variant == WhereVariant::TTS) {
        // TTS: c_1 = value_true tensor
        unpack_to_dest_mode[tt::CBIndex::c_1] = (value_true_tensor.value().dtype() == DataType::FLOAT32)
                                                    ? UnpackToDestMode::UnpackToDestFp32
                                                    : UnpackToDestMode::Default;
    } else if (variant == WhereVariant::TST) {
        // TST: c_1 = value_false tensor
        unpack_to_dest_mode[tt::CBIndex::c_1] = (value_false_tensor.value().dtype() == DataType::FLOAT32)
                                                    ? UnpackToDestMode::UnpackToDestFp32
                                                    : UnpackToDestMode::Default;
    } else {
        // TTT: c_1 = value_true tensor, c_2 = value_false tensor (including column broadcast)
        unpack_to_dest_mode[tt::CBIndex::c_1] = (value_true_tensor.value().dtype() == DataType::FLOAT32)
                                                    ? UnpackToDestMode::UnpackToDestFp32
                                                    : UnpackToDestMode::Default;
        unpack_to_dest_mode[tt::CBIndex::c_2] = (value_false_tensor.value().dtype() == DataType::FLOAT32)
                                                    ? UnpackToDestMode::UnpackToDestFp32
                                                    : UnpackToDestMode::Default;
    }

    // Output CB depends on variant: c_2 for binary_ng compatibility (TTT col bcast), c_3 for other cases
    unpack_to_dest_mode[output_cb_index] =
        (output.dtype() == DataType::FLOAT32) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;

    constexpr uint32_t num_tiles_per_cycle = 1;  // we produce 1 output tile per read-compute-write cycle

    // All variants use the same compile args now
    std::vector<uint32_t> compute_kernel_args;
    if (variant == WhereVariant::TTS) {
        auto bit_cast_scalar = pack_scalar_runtime_arg(operation_attributes.value_false_scalar.value(), output.dtype());
        compute_kernel_args = {num_tiles_per_cycle, bit_cast_scalar};
    } else if (variant == WhereVariant::TST) {
        auto bit_cast_scalar = pack_scalar_runtime_arg(operation_attributes.value_true_scalar.value(), output.dtype());
        compute_kernel_args = {num_tiles_per_cycle, bit_cast_scalar};
    } else {
        compute_kernel_args = {num_tiles_per_cycle};
    }

    std::map<std::string, std::string> kernel_defines;

    // Add binary_ng style defines for TTT column broadcast case
    if (variant == WhereVariant::TTT &&
        (broadcast_type == WhereBroadcastType::COL_BCAST || broadcast_type == WhereBroadcastType::SCALAR_BCAST)) {
        // 3-tensor broadcast configuration - set defines for each tensor independently
        kernel_defines["BCAST_PRED"] = pred_is_bcast ? "1" : "0";
        kernel_defines["BCAST_TRUE"] = true_is_bcast ? "1" : "0";
        kernel_defines["BCAST_FALSE"] = false_is_bcast ? "1" : "0";
    } else if (variant == WhereVariant::TTS && broadcast_type == WhereBroadcastType::COL_BCAST) {
        // TTS column broadcast configuration
        kernel_defines["BCAST_PRED"] = pred_is_bcast ? "1" : "0";
        kernel_defines["BCAST_TRUE"] = true_is_bcast ? "1" : "0";
        kernel_defines["BCAST_FALSE"] = "0";                       // False is scalar for TTS
    } else if (variant == WhereVariant::TST && broadcast_type == WhereBroadcastType::COL_BCAST) {
        // TST column broadcast configuration
        kernel_defines["BCAST_PRED"] = pred_is_bcast ? "1" : "0";
        kernel_defines["BCAST_TRUE"] = "0";  // True is scalar for TST
        kernel_defines["BCAST_FALSE"] = false_is_bcast ? "1" : "0";
    }
    if ((variant == WhereVariant::TTS) && (broadcast_type == WhereBroadcastType::SCALAR_A_BCAST ||
                                           broadcast_type == WhereBroadcastType::SCALAR_B_BCAST)) {
        // 2-tensor broadcast configuration - set defines for each tensor independently
        kernel_defines["BCAST_PRED"] = (broadcast_type == WhereBroadcastType::SCALAR_A_BCAST) ? "1" : "0";
        kernel_defines["BCAST_TRUE"] = (broadcast_type == WhereBroadcastType::SCALAR_B_BCAST) ? "1" : "0";
    }
    if ((variant == WhereVariant::TST) && (broadcast_type == WhereBroadcastType::SCALAR_A_BCAST ||
                                           broadcast_type == WhereBroadcastType::SCALAR_B_BCAST)) {
        // 2-tensor broadcast configuration - set defines for each tensor independently
        kernel_defines["BCAST_PRED"] = (broadcast_type == WhereBroadcastType::SCALAR_A_BCAST) ? "1" : "0";
        kernel_defines["BCAST_FALSE"] = (broadcast_type == WhereBroadcastType::SCALAR_B_BCAST) ? "1" : "0";
    }

    kernel_defines["WHERE_LLK"] = "where_tile";
    kernel_defines["FILL_LLK"] = "fill_tile";

    // Data type specific defines (common for all variants)
    if (predicate_tensor.dtype() == DataType::FLOAT32) {
        kernel_defines["WHERE_LLK"] = "where_fp32_tile";
    }
    if (predicate_tensor.dtype() == DataType::INT32) {
        kernel_defines["WHERE_LLK"] = "where_int32_tile";
        kernel_defines["FILL_LLK"] = "fill_tile_int";
        kernel_defines["FILL_WITH_VALUE_INT"] = "1";
    } else {
        kernel_defines["FILL_WITH_VALUE_FLOAT"] = "1";
    }

    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        get_kernel_file_path(kernel_config.compute_kernel),
        all_device_cores,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .compile_args = compute_kernel_args,
            .defines = kernel_defines});

    auto set_runtime_args = [](Program& program, KernelHandle kernel_id, CoreCoord core, auto&& args) {
        tt_metal::SetRuntimeArgs(program, kernel_id, core, args);
    };

    CMAKE_UNIQUE_NAMESPACE::set_or_update_runtime_arguments(
        program,
        reader_kernel_id,
        writer_kernel_id,
        compute_kernel_id,
        compute_with_storage_grid_size,
        operation_attributes,
        tensor_args,
        output,
        broadcast_type,
        set_runtime_args);

    return {
        std::move(program), {reader_kernel_id, writer_kernel_id, compute_kernel_id, compute_with_storage_grid_size}};
}

void WhereDeviceOperation::WhereProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto update_args =
        [](tt::tt_metal::Program& program, tt::tt_metal::KernelHandle kernel_id, CoreCoord core, auto&& args) {
            auto& all_args = GetRuntimeArgs(program, kernel_id);
            auto& core_args = all_args.at(core.x).at(core.y);
            std::copy(args.begin(), args.end(), core_args.data());
        };

    // Detect broadcast type for the cached program
    const auto& [predicate_tensor, value_true_tensor, value_false_tensor, optional_output_tensor] = tensor_args;
    WhereBroadcastType broadcast_type = WhereBroadcastType::NONE;
    if (operation_attributes.where_variant == WhereVariant::TTT) {
        broadcast_type = get_broadcast_type(
            predicate_tensor.logical_shape(),
            value_true_tensor.value().logical_shape(),
            value_false_tensor.value().logical_shape());
    }
    if (operation_attributes.where_variant == WhereVariant::TTS) {
        broadcast_type =
            get_broadcast_type(predicate_tensor.logical_shape(), value_true_tensor.value().logical_shape());
    }
    if (operation_attributes.where_variant == WhereVariant::TST) {
        broadcast_type =
            get_broadcast_type(predicate_tensor.logical_shape(), value_false_tensor.value().logical_shape());
    }

    CMAKE_UNIQUE_NAMESPACE::set_or_update_runtime_arguments(
        cached_program.program,
        cached_program.shared_variables.reader_kernel_id,
        cached_program.shared_variables.writer_kernel_id,
        cached_program.shared_variables.compute_kernel_id,
        cached_program.shared_variables.compute_with_storage_grid_size,
        operation_attributes,
        tensor_args,
        output,
        broadcast_type,
        update_args);
}

}  // namespace ttnn::operations::ternary
