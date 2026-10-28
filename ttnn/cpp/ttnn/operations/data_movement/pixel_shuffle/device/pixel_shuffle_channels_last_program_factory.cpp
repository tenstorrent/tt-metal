// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// pixel_shuffle with channels_last: NHWC [N, H, W, C*r^2] ROW_MAJOR height-sharded in L1 -> NHWC
// [N, H*r, W*r, padded_channels] ROW_MAJOR height-sharded in L1. Every core walks the input sticks of its own
// shard (pixel p = pix0 + j = (n, h, w)); the r^2 output pixels of stick p are (n, h*r + rh, w*r + rw), output
// row n*(H*r)*(W*r) + (h*r + rh)*(W*r) + w*r + rw, written through the output's TensorAccessor (height-sharded).
//   SPATIAL_MAJOR, padded_channels == C: the r*C elements of kernel row rh are contiguous in the stick and go to
//     r consecutive output sticks: one NoC write per (pixel, rh) (the run never crosses a shard when the output
//     shard height is a multiple of r). This is the path the stride-r transposed conv uses.
//   otherwise: each output stick is assembled in a scratch slot (CHANNEL_MAJOR: a strided gather, element
//     c*r^2 + rh*r + rw; SPATIAL_MAJOR: a contiguous copy) with its zero padding, then one NoC write.
// The two dataflow RISCs split the sticks of the shard.

#include "pixel_shuffle_device_op.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::data_movement {

using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor PixelShuffle::MultiCoreChannelsLast::create_descriptor(
    const operation_attributes_t& a, const tensor_args_t& t, tensor_return_value_t& output_tensor) {
    const Tensor& input = t.input_tensor;
    const Tensor& output = output_tensor;
    const uint32_t r = a.upscale_factor;
    const auto& s = input.logical_shape();
    const uint32_t N = s[0];
    const uint32_t H = s[1];
    const uint32_t W = s[2];
    const uint32_t c_in = s[3];
    const uint32_t C = c_in / (r * r);
    const uint32_t Cp = a.padded_channels;
    const uint32_t datum = input.element_size();
    const uint32_t in_stick_nbytes = static_cast<uint32_t>(input.buffer()->aligned_page_size());
    const uint32_t out_stick_nbytes = static_cast<uint32_t>(output.buffer()->aligned_page_size());
    TT_FATAL(out_stick_nbytes == Cp * datum, "PixelShuffle(channels_last): output page {} B != padded_channels x datum {} B.", out_stick_nbytes, Cp * datum);
    const bool fast = a.channel_order == ttnn::PixelUnshuffleChannelOrder::SPATIAL_MAJOR && Cp == C;
    const uint32_t total_pixels = N * H * W;

    const auto& in_shard = input.shard_spec().value();
    const uint32_t shard_h = in_shard.shape[0];
    const std::vector<CoreCoord> cores = corerange_to_cores(in_shard.grid, std::nullopt, in_shard.orientation == ShardOrientation::ROW_MAJOR);

    ProgramDescriptor desc;
    {
        CBDescriptor cb_in;  // the input shard, read in place
        cb_in.total_size = static_cast<uint32_t>(input.buffer()->aligned_size_per_bank());
        cb_in.core_ranges = in_shard.grid;
        cb_in.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0), .data_format = datatype_to_dataformat_converter(input.dtype()), .page_size = in_stick_nbytes});
        cb_in.buffer = input.buffer();
        desc.cbs.push_back(std::move(cb_in));
    }
    constexpr uint32_t scratch_slots = 8;  // general path: output sticks in flight per RISC
    const uint32_t cb_scratch[2] = {tt::CBIndex::c_1, tt::CBIndex::c_2};
    if (!fast) {
        for (uint32_t k = 0; k < 2; k++) {
            CBDescriptor cb;
            cb.total_size = scratch_slots * out_stick_nbytes;
            cb.core_ranges = in_shard.grid;
            cb.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_scratch[k]), .data_format = datatype_to_dataformat_converter(output.dtype()), .page_size = out_stick_nbytes});
            desc.cbs.push_back(std::move(cb));
        }
    }

    KernelDescriptor kernels[2];
    for (uint32_t k = 0; k < 2; k++) {
        std::vector<uint32_t> cta = {
            r,                                            // 0
            H,                                            // 1
            W,                                            // 2
            C,                                            // 3
            Cp,                                           // 4
            datum,                                        // 5
            in_stick_nbytes,                              // 6
            out_stick_nbytes,                             // 7
            tt::CBIndex::c_0,                             // 8
            k,                                            // 9 which half of the shard's sticks
            static_cast<uint32_t>(a.channel_order),       // 10 (0=CHANNEL_MAJOR, 1=SPATIAL_MAJOR)
            fast ? 1u : 0u,                               // 11
            fast ? 0u : cb_scratch[k],                    // 12
            scratch_slots,                                // 13
        };
        TensorAccessorArgs(*output.buffer()).append_to(cta);
        kernels[k].kernel_source = "ttnn/cpp/ttnn/operations/data_movement/pixel_shuffle/device/kernels/dataflow/pixel_shuffle_nhwc_sharded.cpp";
        kernels[k].source_type = KernelDescriptor::SourceType::FILE_PATH;
        kernels[k].core_ranges = in_shard.grid;
        kernels[k].compile_time_args = std::move(cta);
    }
    kernels[0].config = ReaderConfigDescriptor{};
    kernels[1].config = WriterConfigDescriptor{};
    for (uint32_t i = 0; i < cores.size(); i++) {
        const uint32_t pix0 = std::min(i * shard_h, total_pixels);
        const uint32_t npix = std::min(shard_h, total_pixels - pix0);
        for (uint32_t k = 0; k < 2; k++) {
            kernels[k].emplace_runtime_args(cores[i], {output.buffer()->address(), pix0, npix});
        }
    }
    desc.kernels.push_back(std::move(kernels[0]));
    desc.kernels.push_back(std::move(kernels[1]));
    return desc;
}

}  // namespace ttnn::operations::data_movement
