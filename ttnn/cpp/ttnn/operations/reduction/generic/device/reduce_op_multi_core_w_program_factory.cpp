// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <bit>
#include <cmath>
#include <limits>
#include <map>

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor ReduceDeviceOperation::ReduceMultiCoreWProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    const auto& a = tensor_args;
    auto& output = tensor_return_value;
    const bool rm_path = operation_attributes.row_major_w_dense_path;
    const auto& shape = a.padded_shape();
    const auto& logical_shape = a.logical_shape();
    uint32_t W = shape[3], H = shape[2], NC = shape[1] * shape[0];
    const uint32_t H_logical = logical_shape[2];
    const uint32_t W_logical = logical_shape[3];
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();

    uint32_t Wt = (W + tile_width - 1) / tile_width;
    uint32_t Ht = (H + tile_height - 1) / tile_height;

    TT_FATAL(
        !rm_path || (a.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
                     output.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED),
        "Reduce W RM path only supports interleaved tensors (input layout {}, output layout {})",
        static_cast<int>(a.memory_config().memory_layout()),
        static_cast<int>(output.memory_config().memory_layout()));
    TT_FATAL(
        !rm_path || operation_attributes.math_op == tt::tt_metal::ReduceOpMath::SUM,
        "Reduce W RM path only supports SUM (mean lowered from AVG), got {}",
        operation_attributes.math_op);
    TT_FATAL(
        !(rm_path && operation_attributes.negate),
        "Reduce W RM path does not currently support 'negate' (CB index c_4/c_5 collision)");

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), operation_attributes.compute_kernel_config);

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);

    tt::DataFormat scaler_cb_data_format =
        src0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    tt_metal::IDevice* device = a.device();

    // RM-only constants. ht_tiles_per_chunk (orthogonal dim) is restricted to 1 for now but exposed
    // as a named knob so the chunk size can be lifted later without re-threading through every site.
    constexpr uint32_t k_rm_ht_tiles_per_chunk = 1;
    constexpr uint32_t k_rm_max_wt_tiles_per_chunk = 8;
    const uint32_t rm_rows_per_tile = tile_height;
    const uint32_t wt_tiles_per_chunk = std::min<uint32_t>(k_rm_max_wt_tiles_per_chunk, std::max(1U, Wt));
    const uint32_t datum_size = tt::datum_size(dst_cb_data_format);
    const uint32_t src_datum_size = tt::datum_size(src0_cb_data_format);
    const uint32_t chunk_row_bytes = wt_tiles_per_chunk * tile_width * src_datum_size;
    const uint32_t rm_staging_page_size = rm_rows_per_tile * chunk_row_bytes;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    // RM splits NC*H_logical row-wise so each core gets contiguous logical rows; tile path
    // keeps the existing NC*Ht slicing.
    const uint32_t num_rows = rm_path ? (NC * H_logical) : (NC * Ht);
    constexpr bool k_split_rows_row_wise = true;
    const bool split_row_wise = rm_path;
    uint32_t num_cores;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_rows_per_core_group_1, num_rows_per_core_group_2;
    if (operation_attributes.sub_core_grids.has_value()) {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(*operation_attributes.sub_core_grids, num_rows, split_row_wise);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows, split_row_wise);
    }
    TT_FATAL(num_cores > 0, "Reduce W requires at least one worker core");

    ProgramDescriptor desc;

    if (rm_path) {
        constexpr uint32_t cb_rm = tt::CBIndex::c_24;
        constexpr uint32_t num_rm_pages = 2;
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_rm_pages * rm_staging_page_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_rm),
                .data_format = src0_cb_data_format,
                .page_size = rm_staging_page_size,
            }}},
        });

        constexpr uint32_t cb_clear_value = tt::CBIndex::c_4;
        desc.cbs.push_back(CBDescriptor{
            .total_size = src0_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_clear_value),
                .data_format = src0_cb_data_format,
                .page_size = src0_single_tile_size,
            }}},
        });

        // reduce_rm.cpp accumulates partial reductions across W chunks into cb_acc (c_5).
        constexpr uint32_t cb_acc = tt::CBIndex::c_5;
        desc.cbs.push_back(CBDescriptor{
            .total_size = k_rm_ht_tiles_per_chunk * dst_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_acc),
                .data_format = dst_cb_data_format,
                .page_size = dst_single_tile_size,
            }}},
        });
    }

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    if (rm_path) {
        num_input_tiles = std::max(num_input_tiles, wt_tiles_per_chunk);
    }
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * src0_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = src0_cb_data_format,
            .page_size = src0_single_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = scaler_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_2),
            .data_format = scaler_cb_data_format,
            .page_size = scaler_single_tile_size,
        }}},
    });

    uint32_t output_cb_index = tt::CBIndex::c_3;
    uint32_t num_output_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * dst_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = dst_cb_data_format,
            .page_size = dst_single_tile_size,
        }}},
    });

    // For min/max with non-unity scalar, the GMPOOL hardware path only respects the scaler's
    // exponent, so the device reduces with scaler=1.0 and the user scalar is applied after the
    // reduction via SFPU mul_unary_tile inside the compute kernel.
    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;
    uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);
    const uint32_t padding_identity_bits =
        dense_rm_padding_identity_bits(src0_cb_data_format, operation_attributes.math_op);

    tt_metal::Buffer* src_buffer = a.buffer();
    std::vector<uint32_t> reader_compile_time_args = {std::bit_cast<uint32_t>(operation_attributes.scaler)};
    if (rm_path) {
        reader_compile_time_args.insert(
            reader_compile_time_args.end(),
            {
                W_logical,
                src_datum_size,
                padding_identity_bits,
                Wt,
                wt_tiles_per_chunk,
                rm_rows_per_tile,
                1,
                /*H_logical unused on W path (reader REDUCE_ROW branch)*/
                0u,
            });
    }
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    tt_metal::Buffer* dst_buffer = output.buffer();
    std::vector<uint32_t> writer_compile_time_args = {rm_path ? datum_size : static_cast<uint32_t>(output_cb_index)};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    if (operation_attributes.negate) {
        uint32_t acc_cb_index = tt::CBIndex::c_4;
        uint32_t num_acc_tiles = 1;
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_acc_tiles * dst_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(acc_cb_index),
                .data_format = dst_cb_data_format,
                .page_size = dst_single_tile_size,
            }}},
        });

        uint32_t inv_cb_index = tt::CBIndex::c_5;
        uint32_t num_inv_tiles = 1;
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_inv_tiles * dst_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(inv_cb_index),
                .data_format = dst_cb_data_format,
                .page_size = dst_single_tile_size,
            }}},
        });
    }

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, ReduceOpDim::W);
    if (use_post_mul) {
        reduce_defines["REDUCE_POST_MUL"] = "1";
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        rm_path ? "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_rm.cpp"
                : "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
                  "reader_unary_reduce_universal_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.defines = {reduce_defines.begin(), reduce_defines.end()};
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        rm_path
            ? "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/writer_reduce_w_rm_scalar.cpp"
            : "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.defines = {reduce_defines.begin(), reduce_defines.end()};
    writer_desc.config = WriterConfigDescriptor{};

    // For RM, per-group counts are logical rows; the compute kernel expects tile-row counts.
    const uint32_t ht_per_core_group_1 =
        rm_path ? (num_rows_per_core_group_1 + rm_rows_per_tile - 1) / rm_rows_per_tile : num_rows_per_core_group_1;
    const uint32_t ht_per_core_group_2 =
        rm_path ? (num_rows_per_core_group_2 + rm_rows_per_tile - 1) / rm_rows_per_tile : num_rows_per_core_group_2;

    // reduce_rm.cpp expects {Ht, Wt, nc_per_reduce, post_mul_bits, wt_chunk, ht_chunk};
    // reduce.cpp / reduce_w_neg.cpp expect {Ht, Wt, NC, post_mul_bits}.
    std::vector<uint32_t> compute_kernel_args_group_1 = {
        ht_per_core_group_1,   // Ht
        Wt,                    // Wt
        1,                     // NC
        post_mul_scaler_bits,  // packed fp32 user scalar (only used if REDUCE_POST_MUL is set)
    };
    if (rm_path) {
        compute_kernel_args_group_1.insert(
            compute_kernel_args_group_1.end(),
            {
                wt_tiles_per_chunk,
                k_rm_ht_tiles_per_chunk,
            });
    }

    const std::string compute_kernel =
        rm_path ? std::string("ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_rm.cpp")
                : std::string("ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce") +
                      (operation_attributes.negate ? "_w_neg" : "") + ".cpp";

    KernelDescriptor compute_desc_g1;
    compute_desc_g1.kernel_source = compute_kernel;
    compute_desc_g1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_g1.core_ranges = core_group_1;
    compute_desc_g1.compile_time_args = compute_kernel_args_group_1;
    compute_desc_g1.defines = {reduce_defines.begin(), reduce_defines.end()};
    compute_desc_g1.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };

    std::optional<KernelDescriptor> compute_desc_g2;
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2 = {
            ht_per_core_group_2,   // Ht
            Wt,                    // Wt
            1,                     // NC
            post_mul_scaler_bits,  // packed fp32 user scalar (only used if REDUCE_POST_MUL is set)
        };
        if (rm_path) {
            compute_kernel_args_group_2.insert(
                compute_kernel_args_group_2.end(),
                {
                    wt_tiles_per_chunk,
                    k_rm_ht_tiles_per_chunk,
                });
        }

        KernelDescriptor d;
        d.kernel_source = compute_kernel;
        d.source_type = KernelDescriptor::SourceType::FILE_PATH;
        d.core_ranges = core_group_2;
        d.compile_time_args = compute_kernel_args_group_2;
        d.defines = {reduce_defines.begin(), reduce_defines.end()};
        d.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
        };
        compute_desc_g2 = std::move(d);
    }

    TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);
    uint32_t out_dim_divider = Wt;
    std::vector<CoreCoord> cores;
    if (rm_path) {
        // Match the row-wise split so core[i] receives the i-th contiguous row chunk.
        cores = corerange_to_cores(all_cores, std::nullopt, k_split_rows_row_wise);
    } else if (operation_attributes.sub_core_grids.has_value()) {
        for (const auto& range : all_cores.ranges()) {
            for (int y = range.start_coord.y; y <= range.end_coord.y; ++y) {
                for (int x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                    cores.emplace_back(x, y);
                }
            }
        }
    } else {
        cores = grid_to_cores(num_cores, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, false);
    }
    TT_FATAL(
        cores.size() == num_cores, "Resolved core list size {} must match split num_cores {}", cores.size(), num_cores);
    TT_FATAL(num_rows == 0 || !cores.empty(), "Non-zero reduce workload requires non-empty core list");
    for (uint32_t i = 0, num_tiles_read = 0, num_rows_read = 0; i < num_cores; i++) {
        const CoreCoord& core = cores[i];
        uint32_t num_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        if (rm_path) {
            // RM split distributes logical rows, so num_rows_per_core IS the logical-row count.
            // Use raw addresses (not Buffer*) so mesh program-cache fast paths re-apply per-core args.
            reader_desc.emplace_runtime_args(
                core,
                {
                    src_buffer->address(),
                    num_rows_per_core,
                    num_rows_read,
                });
            writer_desc.emplace_runtime_args(
                core,
                {
                    dst_buffer->address(),
                    num_rows_per_core,
                    num_rows_read,
                });
            num_rows_read += num_rows_per_core;
        } else {
            uint32_t num_tensor_tiles_per_core = num_rows_per_core * Wt;
            reader_desc.emplace_runtime_args(
                core,
                {
                    a.buffer(),
                    num_tensor_tiles_per_core,
                    num_tiles_read  // tile index of row to start reading from
                });
            writer_desc.emplace_runtime_args(
                core,
                {
                    output.buffer(),
                    num_tensor_tiles_per_core / out_dim_divider,  // number of tiles to write
                    num_tiles_read / out_dim_divider              // output tile start index
                });
            num_tiles_read += num_tensor_tiles_per_core;
        }

        if (i == num_cores - 1) {
            if (rm_path) {
                TT_FATAL(
                    num_rows_read == num_rows,
                    "Reduce W (dense RM) assigned {} logical rows across cores, expected {}",
                    num_rows_read,
                    num_rows);
            } else {
                TT_FATAL(
                    num_tiles_read == num_rows * Wt,
                    "Reduce W assigned {} input tiles, expected {}",
                    num_tiles_read,
                    num_rows * Wt);
            }
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_g1));
    if (compute_desc_g2.has_value()) {
        desc.kernels.push_back(std::move(*compute_desc_g2));
    }

    return desc;
}

}  // namespace ttnn::prim
