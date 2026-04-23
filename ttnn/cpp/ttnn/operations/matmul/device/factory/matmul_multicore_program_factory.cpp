// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/factory/matmul_multicore_program_factory.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::constants;
using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;
using tt::tt_metal::ReaderConfigDescriptor;
using tt::tt_metal::WriterConfigDescriptor;

namespace ttnn::prim {

ProgramDescriptor MatmulMultiCoreProgramFactory::create_descriptor(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const std::optional<CoreRangeSet>& /*core_range_set*/) {
    if (!tensor_args.optional_input_tensors.empty()) {
        TT_FATAL(!tensor_args.optional_input_tensors[0].has_value(), "Bias is not supported for matmul multi core");
    }

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    auto& output = tensor_return_value.at(0);

    TT_FATAL(operation_attributes.bcast_batch.has_value(), "Error: bcast_batch field should have been populated");
    bool bcast_batch = operation_attributes.bcast_batch.value();

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();

    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t in0_single_tile_size = tt::tile_size(in0_data_format);
    uint32_t in1_single_tile_size = tt::tile_size(in1_data_format);
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    tt_metal::Buffer* src0_buffer = a.buffer();
    tt_metal::Buffer* src1_buffer = b.buffer();

    tt::tt_metal::IDevice* device = a.device();

    TT_FATAL(operation_attributes.compute_kernel_config.has_value(), "Compute kernel config should have been provided");
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config.value());
    (void)packer_l1_acc;

    const auto& cshape = output.padded_shape();  // C=A*B, N1MK*11KN->N1MN

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t c_batch_size = get_batch_size(cshape);
    auto num_output_tiles_total = c_batch_size * cshape[-2] * cshape[-1] / TILE_HW;
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_tiles_per_core_group_1,
         num_output_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles_total);

    tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // C = A*B*...
    // MN = MK*KN
    uint32_t B = get_batch_size(ashape);
    uint32_t Mt = ashape[-2] / TILE_HEIGHT;
    uint32_t Kt = ashape[-1] / TILE_WIDTH;
    uint32_t Nt = bshape[-1] / TILE_WIDTH;
    uint32_t KtNt = Kt * Nt;
    uint32_t MtKt = Mt * Kt;
    uint32_t MtNt = Mt * Nt;

    ProgramDescriptor desc;

    // Circular buffers
    uint32_t num_input_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * in0_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = in0_data_format,
            .page_size = in0_single_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * in1_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = 1,
            .data_format = in1_data_format,
            .page_size = in1_single_tile_size,
        }}},
    });
    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * output_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = output_data_format,
            .page_size = output_single_tile_size,
        }}},
    });

    // Reader kernel
    uint32_t last_ktile_w = a.logical_shape()[-1] % TILE_WIDTH;
    uint32_t last_ktile_h = 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)last_ktile_w, (uint32_t)last_ktile_h};
    tt::tt_metal::TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.named_compile_time_args = {{"cb_in0", tt::CBIndex::c_0}, {"cb_in1", tt::CBIndex::c_1}};
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer kernel
    std::vector<uint32_t> writer_compile_time_args = {};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.named_compile_time_args = {{"cb_out", output_cb_index}};
    writer_desc.config = WriterConfigDescriptor{};

    // Per-core runtime args for reader and writer
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_output_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        reader_desc.emplace_runtime_args(
            core,
            {src0_buffer,
             src1_buffer,
             Mt,
             Kt,
             Nt,
             MtKt,
             KtNt,
             B,
             uint32_t(bcast_batch),
             num_tiles_written,
             num_output_tiles_per_core,
             MtNt});
        writer_desc.emplace_runtime_args(core, {dst_buffer, num_output_tiles_per_core, num_tiles_written});
        num_tiles_written += num_output_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    const auto throttle_level = ttnn::get_throttle_level(operation_attributes.compute_kernel_config);
    std::map<std::string, std::string> mm_kernel_defines;
    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, mm_kernel_defines, throttle_level);

    // Compute kernel(s) — one per core group with different tile counts
    // bmm compute kernel: B, Mt, Nt are just 3 for loops that act as 1 large loop,
    // so only set Nt for simplicity
    std::vector<uint32_t> compute_args_group_1 = {1, 1, Kt, num_output_tiles_per_core_group_1};

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp";
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = compute_args_group_1;
    compute_desc_1.named_compile_time_args = {
        {"cb_in0", tt::CBIndex::c_0}, {"cb_in1", tt::CBIndex::c_1}, {"cb_out", tt::CBIndex::c_16}};
    compute_desc_1.defines = {mm_kernel_defines.begin(), mm_kernel_defines.end()};
    compute_desc_1.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode};
    desc.kernels.push_back(std::move(compute_desc_1));

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_args_group_2 = {1, 1, Kt, num_output_tiles_per_core_group_2};

        KernelDescriptor compute_desc_2;
        compute_desc_2.kernel_source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp";
        compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = compute_args_group_2;
        compute_desc_2.named_compile_time_args = {
            {"cb_in0", tt::CBIndex::c_0}, {"cb_in1", tt::CBIndex::c_1}, {"cb_out", tt::CBIndex::c_16}};
        compute_desc_2.defines = {mm_kernel_defines.begin(), mm_kernel_defines.end()};
        compute_desc_2.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode};
        desc.kernels.push_back(std::move(compute_desc_2));
    }

    return desc;
}

}  // namespace ttnn::prim
