// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reuse_mcast_1d_descriptor.hpp"

#include <algorithm>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/tensor/shape/shape.hpp"

using namespace tt;
using namespace tt::constants;

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

namespace ttnn::prim::matmul_new_detail {

namespace {

uint32_t get_preferred_noc(
    const ttnn::CoreCoord src,
    const ttnn::CoreCoord dst,
    const tt_metal::IDevice* device,
    const bool use_dedicated_noc = false) {
    uint32_t src_x = src.x, src_y = src.y;
    uint32_t dst_x = dst.x, dst_y = dst.y;

    uint32_t MAX_X = device->grid_size().x;
    uint32_t MAX_Y = device->grid_size().y;

    uint32_t dist_right = src_x <= dst_x ? dst_x - src_x : MAX_X - src_x + dst_x;
    uint32_t dist_left = src_x < dst_x ? src_x + MAX_X - dst_x : src_x - dst_x;

    uint32_t dist_bottom = src_y <= dst_y ? dst_y - src_y : MAX_Y - src_y + dst_y;
    uint32_t dist_top = src_y < dst_y ? src_y + MAX_Y - dst_y : src_y - dst_y;

    uint32_t dist_noc_0 = dist_right + dist_bottom;
    uint32_t dist_noc_1 = dist_top + dist_left;

    uint32_t noc = dist_noc_0 < dist_noc_1 ? 0 : 1;

    return use_dedicated_noc ? 1 : noc;
}

enum class CORE_TYPE : uint32_t { IDLE_CORE = 0, WORKER_CORE = 1, HOP_CORE = 2 };

}  // namespace

tt::tt_metal::ProgramDescriptor ReuseMcast1DDescriptorFactory::create_descriptor(
    const MatmulParams& operation_attributes,
    const MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    using namespace tt::tt_metal;

    // ========================================================================
    // Extract program config and compute kernel config
    // ========================================================================
    auto program_config = std::get<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(
        operation_attributes.program_config.value());
    DeviceComputeKernelConfig compute_kernel_config = operation_attributes.compute_kernel_config.value();

    // ========================================================================
    // Extract tensors
    // ========================================================================
    const auto& a = tensor_args.input_tensors.at(0);
    auto b_tensors = std::vector<Tensor>{tensor_args.input_tensors.begin() + 1, tensor_args.input_tensors.end()};
    const auto& b = b_tensors[0];
    const auto& bias = tensor_args.optional_input_tensors.at(0);
    auto& output = tensor_return_value.at(0);

    // ========================================================================
    // Program config parameters
    // ========================================================================
    bool bcast_batch = operation_attributes.bcast_batch.value_or(false);
    bool transpose_a = operation_attributes.transpose_a;
    bool transpose_b = operation_attributes.transpose_b;
    bool untilize_out = operation_attributes.untilize_out;
    bool fuse_batch = program_config.fuse_batch;
    bool mcast_in0 = program_config.mcast_in0;
    bool gather_in0 = program_config.gather_in0;
    CoreCoord compute_with_storage_grid_size = program_config.compute_with_storage_grid_size;
    uint32_t in0_block_w = program_config.in0_block_w;
    uint32_t out_subblock_h = program_config.out_subblock_h;
    uint32_t out_subblock_w = program_config.out_subblock_w;
    uint32_t out_block_h = program_config.out_block_h;
    uint32_t out_block_w = program_config.out_block_w;
    uint32_t per_core_M = program_config.per_core_M;
    uint32_t per_core_N = program_config.per_core_N;
    std::optional<UnaryWithParam> fused_activation = program_config.fused_activation;
    CoreRangeSet hop_cores = program_config.hop_cores;
    uint32_t num_global_cb_receivers = program_config.num_global_cb_receivers;
    const auto& global_cb = operation_attributes.global_cb;
    const auto& sub_device_id = operation_attributes.sub_device_id;
    auto throttle_level = ttnn::get_throttle_level(compute_kernel_config);

    // ========================================================================
    // Shape, tile, and data format setup
    // ========================================================================
    const auto& ashape = operations::matmul::utilities::get_matmul_tensor_padded_shape(a, transpose_a);
    const auto& bshape = operations::matmul::utilities::get_matmul_tensor_padded_shape(b, transpose_b);
    auto in0_tile = operations::matmul::utilities::get_matmul_tile(a, transpose_a);
    auto in1_tile = operations::matmul::utilities::get_matmul_tile(b, transpose_b);
    auto output_tile = tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()});

    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    tt_metal::Buffer* bias_buffer = nullptr;
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;
    tt::tt_metal::Tile bias_tile = output_tile;
    if (bias.has_value()) {
        const auto& c = bias.value();
        TT_FATAL(
            c.storage_type() == StorageType::DEVICE,
            "Bias tensor must be on device, got storage type: {}",
            c.storage_type());
        TT_FATAL(a.device() == c.device(), "Operands to matmul need to be on the same device!");
        TT_FATAL(c.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");
        bias_buffer = c.buffer();
        bias_data_format = tt_metal::datatype_to_dataformat_converter(c.dtype());
        bias_tile = c.tensor_spec().tile();
    }

    tt_metal::IDevice* device = a.device();

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    tt_metal::Buffer* in0_buffer = a.buffer();
    tt_metal::Buffer* in1_buffer = b.buffer();
    TT_FATAL(
        in0_buffer->size() % in0_single_tile_size == 0,
        "Input A buffer size ({}) must be divisible by single tile size ({})",
        in0_buffer->size(),
        in0_single_tile_size);
    TT_FATAL(
        in1_buffer->size() % in1_single_tile_size == 0,
        "Input B buffer size ({}) must be divisible by single tile size ({})",
        in1_buffer->size(),
        in1_single_tile_size);
    TT_FATAL(ashape[-1] == bshape[-2], "Dimension K (A.shape[-1] and B.shape[-2]) must match for A and B in bmm_op");
    TT_FATAL(
        ashape[-2] % in0_tile.get_height() == 0,
        "A.shape[-2] ({}) must be divisible by tile height ({})",
        ashape[-2],
        in0_tile.get_height());
    TT_FATAL(
        ashape[-1] % in0_tile.get_width() == 0,
        "A.shape[-1] ({}) must be divisible by tile width ({})",
        ashape[-1],
        in0_tile.get_width());
    TT_FATAL(
        bshape[-2] % in1_tile.get_height() == 0,
        "B.shape[-2] ({}) must be divisible by tile height ({})",
        bshape[-2],
        in1_tile.get_height());
    TT_FATAL(
        bshape[-1] % in1_tile.get_width() == 0,
        "B.shape[-1] ({}) must be divisible by tile width ({})",
        bshape[-1],
        in1_tile.get_width());

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    const auto B = fuse_batch ? 1 : get_batch_size(ashape);
    const auto M = operations::matmul::utilities::get_M_dim(ashape, in0_tile, fuse_batch);
    const auto K = operations::matmul::utilities::get_K_dim(ashape, in0_tile);
    const auto N = operations::matmul::utilities::get_N_dim(bshape, in1_tile);

    TT_FATAL(K % in0_block_w == 0, "Kt ({}) must be divisible by in0_block_w ({})", K, in0_block_w);

    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    uint32_t num_blocks_y = ((M - 1) / per_core_M) + 1;
    uint32_t num_blocks_x = ((N - 1) / per_core_N) + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;

    TT_FATAL(
        num_blocks_total <= num_cores_x * num_cores_y,
        "Number of blocks exceeds number of cores: {} blocks > {} cores",
        num_blocks_total,
        num_cores_x * num_cores_y);

    if (!gather_in0) {
        TT_FATAL(hop_cores.empty(), "Hop cores are not supported for any mode besides gather_in0.");
    }

    tt_metal::Buffer* out_buffer = output.buffer();
    TT_FATAL(out_buffer != nullptr, "Output buffer should be allocated on device!");

    // ========================================================================
    // Build ProgramDescriptor
    // ========================================================================
    ProgramDescriptor desc;

    if (gather_in0) {
        // ====================================================================
        // GATHER_IN0 PATH
        // ====================================================================
        TT_FATAL(!transpose_a, "Transpose A is ({}) not supported for gather_in0", transpose_a);
        TT_FATAL(!transpose_b, "Transpose B is ({}) not supported for gather_in0", transpose_b);
        TT_FATAL(tensor_return_value.size() == b_tensors.size(), "number of outputs must match number of inputs b");

        std::vector<tt_metal::Buffer*> out_buffers;
        out_buffers.reserve(tensor_return_value.size());
        for (const auto& output_tensor : tensor_return_value) {
            out_buffers.push_back(output_tensor.buffer());
        }

        const auto num_output_cb = out_buffers.size();
        const auto batch = b_tensors.size();
        const bool in1_is_dram_interleaved = in1_buffer->is_dram() && !b.is_sharded();
        const bool in1_is_dram_sharded = in1_buffer->is_dram() && b.is_sharded() && !global_cb.has_value();

        constexpr bool row_major = true;
        CoreRangeSet all_worker_cores = a.shard_spec().value().grid;
        CoreRangeSet non_idle_cores = all_worker_cores.merge(hop_cores);
        CoreRangeSet all_cores = non_idle_cores;
        std::vector<CoreRange> non_idle_cores_vec;
        auto subdevice_cores = device->worker_cores(
            tt::tt_metal::HalProgrammableCoreType::TENSIX,
            sub_device_id.has_value() ? *sub_device_id : device->get_sub_device_ids().at(0));
        // restricted_cores is always nullopt in the create path
        for (const auto& cr : subdevice_cores.ranges()) {
            auto intersection = non_idle_cores.intersection(cr);
            if (!intersection.empty()) {
                non_idle_cores_vec.push_back(intersection.bounding_box());
            }
        }
        all_cores = CoreRangeSet(non_idle_cores_vec);
        std::vector<CoreRange> ring_list = all_worker_cores.ranges();
        std::vector<CoreRange> hop_list = hop_cores.ranges();
        ring_list.insert(ring_list.end(), hop_list.begin(), hop_list.end());

        CoreRangeSet ring_cores = CoreRangeSet(ring_list);
        const uint32_t num_cores = all_worker_cores.num_cores();
        const uint32_t ring_size = num_cores;

        uint32_t num_hop_cores = hop_cores.num_cores();
        bool use_hop_cores = num_hop_cores > 0;

        // Inner dim padding
        const uint32_t Kt_pad = in0_buffer->shard_spec().shape()[1] / in0_tile.get_width() * num_cores;
        uint32_t gather_in0_block_w = Kt_pad / num_cores;

        uint32_t num_blocks = Kt_pad / gather_in0_block_w;
        bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

        bool use_global_cb = global_cb.has_value();

        tt::DataFormat interm0_data_format =
            packer_l1_acc_en ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

        uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
        uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

        // in0
        uint32_t in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / in0_tile.get_width();
        uint32_t in0_CB_tiles = per_core_M * in0_shard_width_in_tiles;
        uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;

        // in1
        uint32_t in1_shard_height_in_tiles = 0;
        uint32_t in1_shard_width_in_tiles = 0;
        uint32_t in1_CB_tiles = 0;

        const auto& bshape_gather = operations::matmul::utilities::get_matmul_tensor_padded_shape(b, false);
        uint32_t in1_tensor_width_in_tiles = bshape_gather[-1] / in1_tile.get_width();

        if (in1_is_dram_sharded || in1_is_dram_interleaved) {
            in1_CB_tiles = 2 * in0_shard_width_in_tiles * per_core_N;
        } else {
            in1_shard_height_in_tiles = in1_buffer->shard_spec().shape()[0] / in1_tile.get_height();
            in1_shard_width_in_tiles =
                in1_buffer->shard_spec().shape()[1] / in1_tile.get_width() / num_global_cb_receivers;
            in1_CB_tiles = in1_shard_height_in_tiles * in1_shard_width_in_tiles;
        }
        uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

        uint32_t per_core_N_size_bytes = per_core_N * in1_single_tile_size;
        uint32_t max_packet_size = 8192;
        uint32_t in1_block_page_size =
            per_core_N_size_bytes > max_packet_size ? max_packet_size : per_core_N_size_bytes;
        uint32_t in1_block_page_size_last =
            per_core_N_size_bytes > max_packet_size ? per_core_N_size_bytes % max_packet_size : per_core_N_size_bytes;
        uint32_t in1_block_width_num_pages = (per_core_N_size_bytes + in1_block_page_size - 1) / in1_block_page_size;
        uint32_t in1_shard_width_in_dram = 0;
        if (in1_is_dram_sharded) {
            in1_shard_width_in_dram = in1_buffer->shard_spec().shape()[1] / in1_tile.get_width();
        }

        // in2
        uint32_t in2_single_tile_size = in0_single_tile_size;
        uint32_t in2_CB_tiles = (ring_size - 1) * in0_CB_tiles;
        uint32_t in2_CB_size = in2_CB_tiles * in2_single_tile_size;

        // out
        uint32_t out_block_tiles = per_core_M * per_core_N;
        uint32_t out_CB_tiles = out_block_tiles;
        uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
        uint32_t interm0_CB_size = out_CB_tiles * interm0_single_tile_size;

        uint32_t K_ = K;
        std::vector<uint32_t> unpadded_in0_shard_widths_in_tiles(num_cores, 0);
        for (uint32_t i = 0; i < num_cores && K_ > 0; ++i) {
            unpadded_in0_shard_widths_in_tiles[i] = std::min(K_, (uint32_t)in0_shard_width_in_tiles);
            K_ -= unpadded_in0_shard_widths_in_tiles[i];
        }

        // Semaphore
        uint32_t in0_signal_semaphore_id = 0;
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = in0_signal_semaphore_id,
            .core_ranges = all_cores,
            .initial_value = INVALID,
        });

        uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
        uint32_t in0_block_num_tiles = out_subblock_h * gather_in0_block_w * in0_num_subblocks;
        uint32_t in0_subblock_num_tiles = out_subblock_h * gather_in0_block_w;
        uint32_t in1_num_subblocks = per_core_N / out_subblock_w;
        uint32_t in1_block_height_in_tiles = gather_in0_block_w;
        uint32_t in1_block_num_tiles = out_subblock_w * in1_block_height_in_tiles * in1_num_subblocks;
        uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size;
        uint32_t in1_tensor_size_bytes = in1_block_num_tiles * num_blocks * in1_single_tile_size;
        uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;
        uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

        // ---- Circular Buffers ----
        uint32_t base_cb_index = tt::CBIndex::c_0;
        uint32_t src0_cb_index = base_cb_index;
        desc.cbs.push_back(CBDescriptor{
            .total_size = in0_CB_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(src0_cb_index),
                .data_format = in0_data_format,
                .page_size = in0_single_tile_size,
                .tile = TileDescriptor(in0_tile),
            }}},
            .buffer = in0_buffer,
        });

        uint32_t src1_cb_index = base_cb_index + 1;
        uint32_t remote_cb_index = tt::CBIndex::c_31;
        if (use_global_cb) {
            uint32_t in1_block_size_bytes_local = in1_single_tile_size * in1_block_num_tiles;
            uint32_t gcb_size = (global_cb->size() / in1_block_size_bytes_local) * in1_block_size_bytes_local;
            CBDescriptor cb;
            cb.total_size = gcb_size;
            cb.core_ranges = all_cores;
            cb.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(src1_cb_index),
                .data_format = in1_data_format,
                .page_size = in1_single_tile_size,
            });
            cb.remote_format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(remote_cb_index),
                .data_format = in1_data_format,
                .page_size = in1_block_size_bytes_local,
            });
            cb.global_circular_buffer = &global_cb.value();
            desc.cbs.push_back(std::move(cb));
        } else {
            CBDescriptor cb;
            cb.total_size = in1_CB_size;
            cb.core_ranges = all_cores;
            cb.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(src1_cb_index),
                .data_format = in1_data_format,
                .page_size = in1_single_tile_size,
                .tile = TileDescriptor(in1_tile),
            });
            if (!in1_is_dram_interleaved && !in1_is_dram_sharded) {
                cb.buffer = in1_buffer;
            }
            desc.cbs.push_back(std::move(cb));
        }

        uint32_t src2_cb_index = base_cb_index + 2;
        desc.cbs.push_back(CBDescriptor{
            .total_size = in2_CB_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(src2_cb_index),
                .data_format = in0_data_format,
                .page_size = in2_single_tile_size,
                .tile = TileDescriptor(in0_tile),
            }}},
        });

        uint32_t sync_cb_index = base_cb_index + 3;
        uint32_t sync_cb_size_bytes = 16;
        desc.cbs.push_back(CBDescriptor{
            .total_size = sync_cb_size_bytes,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(sync_cb_index),
                .data_format = DataFormat::UInt16,
                .page_size = sync_cb_size_bytes,
            }}},
        });

        uint32_t sync_cb2_index = base_cb_index + 4;
        uint32_t sync_cb2_size_bytes = 16;
        desc.cbs.push_back(CBDescriptor{
            .total_size = sync_cb2_size_bytes,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(sync_cb2_index),
                .data_format = DataFormat::UInt16,
                .page_size = sync_cb2_size_bytes,
            }}},
        });

        // Output/interm0 CBs
        uint32_t output_cb_index = base_cb_index + 5;
        uint32_t interm0_cb_index = base_cb_index + 6;
        std::vector<uint32_t> output_cb_indices;
        std::vector<uint32_t> interm_cb_indices;

        if ((interm0_data_format != output_data_format) || (untilize_out && (in1_num_subblocks > 1))) {
            // Separate interm0
            desc.cbs.push_back(CBDescriptor{
                .total_size = interm0_CB_size,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(interm0_cb_index),
                    .data_format = interm0_data_format,
                    .page_size = interm0_single_tile_size,
                    .tile = TileDescriptor(output_tile),
                }}},
            });

            for (uint32_t i = 0; i < out_buffers.size(); ++i) {
                const auto& ob = out_buffers[i];
                output_cb_index += i * 2;
                TT_FATAL(
                    output_cb_index <= tt::CBIndex::c_31,
                    "Output circular buffer index {} exceeds maximum value {}",
                    output_cb_index,
                    tt::CBIndex::c_31);
                desc.cbs.push_back(CBDescriptor{
                    .total_size = out_CB_size,
                    .core_ranges = all_cores,
                    .format_descriptors = {{CBFormatDescriptor{
                        .buffer_index = static_cast<uint8_t>(output_cb_index),
                        .data_format = output_data_format,
                        .page_size = output_single_tile_size,
                        .tile = TileDescriptor(output_tile),
                    }}},
                    .buffer = ob,
                });
                output_cb_indices.push_back(output_cb_index);
                interm_cb_indices.push_back(interm0_cb_index);
            }
        } else {
            for (uint32_t i = 0; i < out_buffers.size(); ++i) {
                const auto& ob = out_buffers[i];
                output_cb_index += i * 2;
                interm0_cb_index += i * 2;
                TT_FATAL(
                    output_cb_index <= tt::CBIndex::c_31,
                    "Output circular buffer index {} exceeds maximum value {}",
                    output_cb_index,
                    tt::CBIndex::c_31);
                TT_FATAL(
                    interm0_cb_index <= tt::CBIndex::c_31,
                    "Interm circular buffer index {} exceeds maximum value {}",
                    interm0_cb_index,
                    tt::CBIndex::c_31);
                CBDescriptor cb;
                cb.total_size = out_CB_size;
                cb.core_ranges = all_cores;
                cb.format_descriptors.push_back(CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(output_cb_index),
                    .data_format = output_data_format,
                    .page_size = output_single_tile_size,
                    .tile = TileDescriptor(output_tile),
                });
                cb.format_descriptors.push_back(CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(interm0_cb_index),
                    .data_format = interm0_data_format,
                    .page_size = interm0_single_tile_size,
                    .tile = TileDescriptor(output_tile),
                });
                cb.buffer = ob;
                desc.cbs.push_back(std::move(cb));
                output_cb_indices.push_back(output_cb_index);
                interm_cb_indices.push_back(interm0_cb_index);
            }
        }

        // ---- Compile time args ----
        std::vector<uint32_t> in0_sender_compile_time_args = {
            (std::uint32_t)in0_shard_width_in_tiles,
            (std::uint32_t)per_core_M,
            (std::uint32_t)batch,
            (std::uint32_t)ring_size,
            (std::uint32_t)in0_signal_semaphore_id,
            (std::uint32_t)src0_cb_index,
            (std::uint32_t)src2_cb_index,
        };

        std::vector<uint32_t> in1_sender_writer_compile_time_args = {
            (std::uint32_t)in1_is_dram_interleaved,
            (std::uint32_t)in1_is_dram_sharded,
            (std::uint32_t)in1_block_height_in_tiles,
            (std::uint32_t)per_core_N,
            (std::uint32_t)in1_tensor_width_in_tiles,
            (std::uint32_t)num_blocks,
            (std::uint32_t)batch,
            (std::uint32_t)in1_block_page_size,
            (std::uint32_t)in1_block_page_size_last,
            (std::uint32_t)in1_block_width_num_pages,
            (std::uint32_t)in1_shard_width_in_dram,
            (std::uint32_t)src1_cb_index,
            (std::uint32_t)sync_cb_index,
            (std::uint32_t)sync_cb2_index,
            (std::uint32_t)remote_cb_index,
            (std::uint32_t)false,  // fused_op_signaler.has_value() is always false
        };
        tt::tt_metal::TensorAccessorArgs(*in1_buffer).append_to(in1_sender_writer_compile_time_args);

        // Compute kernel args
        const uint32_t out_block_num_subblocks = out_block_tiles / out_subblock_num_tiles;
        TT_FATAL(
            out_block_num_subblocks == 1 || !untilize_out,
            "untilize_out is not supported for cases that out_block_num_subblocks > 1");
        std::vector<uint32_t> compute_kernel_args = {
            gather_in0_block_w,
            in0_num_subblocks,
            in0_block_num_tiles,
            in0_subblock_num_tiles,
            in1_num_subblocks,
            in1_block_num_tiles,
            in1_block_size_bytes,
            in1_tensor_size_bytes,
            in1_per_core_w,
            num_blocks,
            out_subblock_h,
            out_subblock_w,
            out_subblock_num_tiles,
            (uint32_t)batch,
            out_block_tiles,
            (uint32_t)untilize_out,
            (uint32_t)in1_is_dram_interleaved,
            (uint32_t)in1_is_dram_sharded,
            src0_cb_index,
            src1_cb_index,
            src2_cb_index,
            sync_cb_index,
            sync_cb2_index,
        };
        compute_kernel_args.push_back(compute_kernel_args.size() + 1);
        for (uint32_t i = 0; i < num_output_cb; ++i) {
            compute_kernel_args.push_back(output_cb_indices[i]);
        }
        for (uint32_t i = 0; i < num_output_cb; ++i) {
            compute_kernel_args.push_back(interm_cb_indices[i]);
        }

        // ---- Kernel defines ----
        KernelDescriptor::Defines mm_in1_kernel_defines;
        std::map<std::string, std::string> mm_kernel_defines_map;

        if (use_global_cb) {
            mm_in1_kernel_defines.emplace_back("ENABLE_GLOBAL_CB", "1");
            mm_kernel_defines_map["ENABLE_GLOBAL_CB"] = "1";
        }

        if (fused_activation.has_value()) {
            if (fused_activation.value().op_type == UnaryOpType::RELU) {
                mm_kernel_defines_map["PACK_RELU"] = "1";
            } else {
                using ttnn::operations::unary::utils::get_defines;
                mm_kernel_defines_map.merge(
                    get_defines(fused_activation.value().op_type, fused_activation.value().params, "ACTIVATION", "i"));
            }
        }
        if (packer_l1_acc_en) {
            mm_kernel_defines_map["PACKER_L1_ACC"] = "1";
        }
        if (fp32_dest_acc_en) {
            mm_kernel_defines_map["FP32_DEST_ACC_EN"] = "1";
        }
        ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
            device->arch(), num_cores, mm_kernel_defines_map);
        ttnn::operations::compute_throttle_utils::throttle_mm_perf(
            device->arch(), num_cores, mm_kernel_defines_map, throttle_level);

        KernelDescriptor::Defines compute_defines;
        for (auto& [k, v] : mm_kernel_defines_map) {
            compute_defines.emplace_back(k, v);
        }

        tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
        tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

        bool use_dedicated_noc = true;
        tt_metal::NOC_MODE noc_mode =
            use_dedicated_noc ? tt_metal::NOC_MODE::DM_DEDICATED_NOC : tt_metal::NOC_MODE::DM_DYNAMIC_NOC;

        // ---- Kernel Descriptors ----
        KernelDescriptor in0_kernel_desc;
        in0_kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_ring_all_gather.cpp";
        in0_kernel_desc.core_ranges = all_cores;
        in0_kernel_desc.compile_time_args = in0_sender_compile_time_args;
        in0_kernel_desc.config = DataMovementConfigDescriptor{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
            .noc_mode = noc_mode,
        };

        KernelDescriptor in1_kernel_desc;
        in1_kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp";
        in1_kernel_desc.core_ranges = all_cores;
        in1_kernel_desc.compile_time_args = in1_sender_writer_compile_time_args;
        in1_kernel_desc.defines = mm_in1_kernel_defines;
        in1_kernel_desc.config = DataMovementConfigDescriptor{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
            .noc_mode = noc_mode,
        };

        KernelDescriptor compute_desc;
        compute_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/"
            "bmm_large_block_zm_fused_bias_activation_gathered.cpp";
        compute_desc.core_ranges = all_cores;
        compute_desc.compile_time_args = compute_kernel_args;
        compute_desc.defines = compute_defines;
        compute_desc.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
        };

        // ---- Runtime args ----
        auto all_cores_vec = corerange_to_cores(all_cores, std::nullopt, row_major);
        auto worker_cores_vec = corerange_to_cores(all_worker_cores, std::nullopt, row_major);
        auto hop_cores_vec = corerange_to_cores(hop_cores, std::nullopt, row_major);

        // DRAM bank mapping for in1_is_dram_sharded
        std::map<uint32_t, uint32_t> worker_y_to_dram_bank_first_col;
        std::map<uint32_t, uint32_t> worker_y_to_dram_bank_second_col;
        uint32_t first_col_max_x = device->arch() == tt::ARCH::WORMHOLE_B0 ? 3 : 7;
        uint32_t num_receiver_cores_per_dram = ring_size / in1_buffer->shard_spec().grid().num_cores();
        if (in1_is_dram_sharded) {
            if (device->arch() == tt::ARCH::WORMHOLE_B0) {
                worker_y_to_dram_bank_first_col[0] = 1;
                worker_y_to_dram_bank_first_col[4] = 2;
                worker_y_to_dram_bank_first_col[5] = 3;
                worker_y_to_dram_bank_first_col[9] = 0;

                worker_y_to_dram_bank_second_col[0] = 4;
                worker_y_to_dram_bank_second_col[1] = 6;
                worker_y_to_dram_bank_second_col[2] = 9;
                worker_y_to_dram_bank_second_col[4] = 10;
                worker_y_to_dram_bank_second_col[5] = 11;
                worker_y_to_dram_bank_second_col[6] = 8;
                worker_y_to_dram_bank_second_col[7] = 7;
                worker_y_to_dram_bank_second_col[9] = 5;
            } else {
                auto optimal_dram_workers = device->get_optimal_dram_bank_to_logical_worker_assignment(in1_noc);
                uint32_t num_banks = optimal_dram_workers.size();
                uint32_t banks_in_first_col = num_banks / 2;

                std::vector<std::pair<uint32_t, uint32_t>> first_col_anchors;
                std::vector<std::pair<uint32_t, uint32_t>> second_col_anchors;

                for (uint32_t bank = 0; bank < num_banks; ++bank) {
                    const auto& core = optimal_dram_workers[bank];
                    if (bank < banks_in_first_col) {
                        first_col_anchors.push_back({core.y, bank});
                    } else {
                        second_col_anchors.push_back({core.y, bank});
                    }
                }

                auto sort_by_y = [](const auto& a, const auto& b) { return a.first < b.first; };
                std::sort(first_col_anchors.begin(), first_col_anchors.end(), sort_by_y);
                std::sort(second_col_anchors.begin(), second_col_anchors.end(), sort_by_y);

                auto find_nearest_bank = [](uint32_t y,
                                            const std::vector<std::pair<uint32_t, uint32_t>>& anchors) -> uint32_t {
                    if (anchors.empty()) {
                        return 0;
                    }
                    uint32_t best_bank = anchors[0].second;
                    uint32_t best_dist = std::abs((int)y - (int)anchors[0].first);
                    for (const auto& [anchor_y, bank] : anchors) {
                        uint32_t dist = std::abs((int)y - (int)anchor_y);
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_bank = bank;
                        }
                    }
                    return best_bank;
                };

                auto compute_grid = device->compute_with_storage_grid_size();
                for (uint32_t y = 0; y < compute_grid.y; ++y) {
                    if (!first_col_anchors.empty()) {
                        worker_y_to_dram_bank_first_col[y] = find_nearest_bank(y, first_col_anchors);
                    }
                    if (!second_col_anchors.empty()) {
                        worker_y_to_dram_bank_second_col[y] = find_nearest_bank(y, second_col_anchors);
                    }
                }
            }
        }

        // Idle cores runtime args
        for (auto core : all_cores_vec) {
            auto all_worker_cores_iter = std::find(worker_cores_vec.begin(), worker_cores_vec.end(), core);
            auto hop_cores_iter = std::find(hop_cores_vec.begin(), hop_cores_vec.end(), core);
            bool core_is_in_all_worker_cores = all_worker_cores_iter != worker_cores_vec.end();
            bool core_is_in_hop_cores = hop_cores_iter != hop_cores_vec.end();
            if (!use_hop_cores) {
                core_is_in_hop_cores = false;
            }

            if (!core_is_in_all_worker_cores && !core_is_in_hop_cores) {
                auto core_type = CORE_TYPE::IDLE_CORE;
                in0_kernel_desc.runtime_args.emplace_back(
                    core, KernelDescriptor::CoreRuntimeArgs{(std::uint32_t)core_type});
                KernelDescriptor::CoreRuntimeArgs in1_idle_args = {(std::uint32_t)core_type};
                // fused_op_signaler is nullopt, no extra args
                in1_kernel_desc.runtime_args.emplace_back(core, std::move(in1_idle_args));
                compute_desc.runtime_args.emplace_back(
                    core, KernelDescriptor::CoreRuntimeArgs{(std::uint32_t)core_type});
            }
        }

        // Worker cores runtime args
        uint32_t bank_id = 0;
        for (uint32_t i = 0; i < num_cores; ++i) {
            bool send_to_hop_core = i == 0 && use_hop_cores;
            const auto& core = worker_cores_vec[i];
            const auto& core_noc = device->worker_core_from_logical_core(core);

            auto core_type = CORE_TYPE::WORKER_CORE;
            CoreCoord next_core;
            if (send_to_hop_core) {
                next_core = hop_cores_vec[0];
            } else {
                uint32_t next_i = i == 0 ? num_cores - 1 : i - 1;
                next_core = worker_cores_vec[next_i % num_cores];
            }
            const auto& next_core_noc = device->worker_core_from_logical_core(next_core);
            uint32_t noc = get_preferred_noc(core_noc, next_core_noc, device, use_dedicated_noc);

            KernelDescriptor::CoreRuntimeArgs mm_in0_args = {
                (std::uint32_t)core_type,
                i,
                next_core_noc.x,
                next_core_noc.y,
                noc,
                (std::uint32_t)false,  // end_of_hop
            };
            mm_in0_args.insert(
                mm_in0_args.end(),
                unpadded_in0_shard_widths_in_tiles.begin(),
                unpadded_in0_shard_widths_in_tiles.end());
            in0_kernel_desc.runtime_args.emplace_back(core, std::move(mm_in0_args));

            KernelDescriptor::CoreRuntimeArgs mm_in1_args = {
                (std::uint32_t)core_type,
                in1_buffer->address(),
                i,
            };
            if (in1_is_dram_sharded) {
                if (core.x <= first_col_max_x) {
                    auto it = worker_y_to_dram_bank_first_col.find(core.y);
                    bank_id = it->second;
                } else {
                    auto it = worker_y_to_dram_bank_second_col.find(core.y);
                    bank_id = it->second;
                }

                uint32_t dram_read_offset = 0;
                if (device->arch() == tt::ARCH::WORMHOLE_B0) {
                    if (core.x % 2 == 0) {
                        dram_read_offset = 1;
                    }
                } else {
                    dram_read_offset = i % num_receiver_cores_per_dram;
                }

                uint32_t vc = 0;
                for (uint32_t j = 0; j < i; ++j) {
                    auto core_prev = worker_cores_vec[j];
                    if (core_prev.y == core.y) {
                        vc = (vc + 1) & 0x3;
                    }
                }
                mm_in1_args.push_back((std::uint32_t)bank_id);
                mm_in1_args.push_back((std::uint32_t)vc);
                mm_in1_args.push_back((std::uint32_t)dram_read_offset);
            }
            // fused_op_signaler is nullopt, no extra args
            in1_kernel_desc.runtime_args.emplace_back(core, std::move(mm_in1_args));

            KernelDescriptor::CoreRuntimeArgs mm_compute_args = {
                (std::uint32_t)core_type,
                i,
            };
            mm_compute_args.insert(
                mm_compute_args.end(),
                unpadded_in0_shard_widths_in_tiles.begin(),
                unpadded_in0_shard_widths_in_tiles.end());
            compute_desc.runtime_args.emplace_back(core, std::move(mm_compute_args));
        }

        // Hop cores runtime args
        for (uint32_t i = 0; i < num_hop_cores; ++i) {
            bool end_of_hop = i == num_hop_cores - 1;
            auto core_type = CORE_TYPE::HOP_CORE;
            const auto& core = hop_cores_vec[i];
            const auto& core_noc = device->worker_core_from_logical_core(core);

            CoreCoord next_core = end_of_hop ? worker_cores_vec[num_cores - 1] : hop_cores_vec[i + 1];
            const auto& next_core_noc = device->worker_core_from_logical_core(next_core);
            uint32_t noc = get_preferred_noc(core_noc, next_core_noc, device, use_dedicated_noc);

            in0_kernel_desc.runtime_args.emplace_back(
                core,
                KernelDescriptor::CoreRuntimeArgs{
                    (std::uint32_t)core_type,
                    0u,
                    next_core_noc.x,
                    next_core_noc.y,
                    noc,
                    (std::uint32_t)end_of_hop,
                });

            KernelDescriptor::CoreRuntimeArgs in1_hop_args = {(std::uint32_t)core_type};
            // fused_op_signaler is nullopt, no extra args
            in1_kernel_desc.runtime_args.emplace_back(core, std::move(in1_hop_args));

            compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{(std::uint32_t)core_type});
        }

        desc.kernels.push_back(std::move(in0_kernel_desc));
        desc.kernels.push_back(std::move(in1_kernel_desc));
        desc.kernels.push_back(std::move(compute_desc));

    } else if (mcast_in0) {
        // ====================================================================
        // MCAST_IN0 PATH
        // ====================================================================
        TT_FATAL(tt::CBIndex::c_0 == tt::CBIndex::c_0, "mcast does not support a non-zero start cb index");

        bool in0_is_sharded = a.memory_config().is_sharded();
        bool in1_is_sharded = b.memory_config().is_sharded();
        bool bias_is_sharded = bias.has_value() ? bias->memory_config().is_sharded() : false;
        bool output_is_sharded = output.memory_config().is_sharded();

        bool in0_transpose_tile = in0_tile.get_transpose_of_faces() && in0_tile.get_transpose_within_face();
        bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();

        uint32_t num_blocks = K / in0_block_w;
        bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

        tt::DataFormat interm0_data_format =
            packer_l1_acc_en ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

        uint32_t bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);
        uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
        uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

        bool do_not_inplace_interm0_out_CB = output_is_sharded && (per_core_M != out_block_h);

        uint32_t in0_block_h = out_block_h;
        uint32_t in1_block_w = out_block_w;
        uint32_t in0_num_blocks_y = per_core_M / out_block_h;
        uint32_t in1_num_blocks_x = per_core_N / out_block_w;
        uint32_t out_num_blocks_x = in1_num_blocks_x;
        uint32_t out_num_blocks_y = in0_num_blocks_y;

        uint32_t in0_block_tiles = in0_block_h * in0_block_w;
        uint32_t in0_CB_tiles = in0_block_tiles;
        if (B * num_blocks > 1) {
            in0_CB_tiles *= operations::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
        }
        uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;

        uint32_t in2_block_tiles = 0;
        uint32_t in0_shard_width_in_tiles = 0;
        uint32_t in0_shard_height_in_tiles = 0;
        if (in0_is_sharded) {
            in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / in0_tile.get_width();
            in0_shard_height_in_tiles = in0_buffer->shard_spec().shape()[0] / in0_tile.get_height();
            in2_block_tiles = per_core_M * in0_shard_width_in_tiles;
        }
        uint32_t in2_CB_tiles = in2_block_tiles;
        uint32_t in2_CB_size = in2_CB_tiles * in0_single_tile_size;

        uint32_t in1_block_tiles = out_block_w * in0_block_w;
        uint32_t in1_CB_tiles = in1_block_tiles;
        if (B * num_blocks > 1) {
            in1_CB_tiles *= operations::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
        }
        if (in1_is_sharded) {
            uint32_t in1_shard_height_in_tiles = in1_buffer->shard_spec().shape()[0] / in1_tile.get_height();
            in1_CB_tiles = per_core_N * in1_shard_height_in_tiles;
        }
        uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

        uint32_t out_block_tiles = out_block_h * out_block_w;
        uint32_t out_shard_tiles = per_core_M * per_core_N;
        uint32_t out_CB_tiles = out_block_tiles;
        if (output_is_sharded) {
            out_CB_tiles = out_shard_tiles;
        }
        uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
        uint32_t interm0_CB_tiles = out_block_tiles;
        uint32_t interm0_CB_size = interm0_CB_tiles * interm0_single_tile_size;

        uint32_t in3_block_tiles = out_block_w;
        uint32_t in3_CB_tiles = in3_block_tiles;
        uint32_t in3_CB_size = in3_CB_tiles * bias_single_tile_size;

        // Core range setup
        CoreCoord start_core = {0, 0};
        uint32_t start_core_x = start_core.x;
        uint32_t start_core_y = start_core.y;
        uint32_t num_cores_c = compute_with_storage_grid_size.x;

        uint32_t num_cores_with_work = num_blocks_total;

        uint32_t in0_sender_num_cores = in0_is_sharded ? a.shard_spec().value().grid.num_cores() : 1;
        uint32_t num_cores_local =
            in0_is_sharded ? std::max(num_cores_with_work, in0_sender_num_cores) : num_cores_with_work;

        constexpr bool row_major = true;
        CoreRangeSet all_cores =
            num_cores_to_corerangeset(start_core, num_cores_local, compute_with_storage_grid_size, row_major);

        CoreRangeSet in0_mcast_sender_cores =
            num_cores_to_corerangeset(in0_sender_num_cores, compute_with_storage_grid_size, row_major);
        CoreCoord in0_mcast_sender_cores_grid = in0_mcast_sender_cores.bounding_box().grid_size();

        CoreRangeSet all_cores_with_work =
            num_cores_to_corerangeset(num_cores_with_work, compute_with_storage_grid_size, row_major);
        CoreRange in0_mcast_receiver_cores_bounding_box = all_cores_with_work.bounding_box();
        uint32_t in0_mcast_receiver_num_cores = in0_mcast_receiver_cores_bounding_box.size();
        uint32_t in0_mcast_receiver_num_dests = std::min(in0_mcast_receiver_num_cores, num_cores_local);

        CoreRangeSet in0_mcast_cores_with_work_and_in_receiver_grid;
        CoreRangeSet in0_mcast_cores_without_work_and_in_receiver_grid;
        CoreRangeSet in0_mcast_cores_without_work_and_not_in_receiver_grid;
        CoreRangeSet in0_mcast_receivers;
        std::vector<uint32_t> in0_mcast_noc_x;
        std::vector<uint32_t> in0_mcast_noc_y;
        if (in0_is_sharded) {
            in0_mcast_cores_with_work_and_in_receiver_grid = all_cores_with_work;
            if (in0_mcast_receiver_num_dests > num_cores_with_work) {
                const uint32_t in0_mcast_cores_without_work_and_in_receiver_grid_num_cores =
                    in0_mcast_receiver_num_dests - num_cores_with_work;
                uint32_t core_idx_x = num_cores_with_work % num_cores_c;
                uint32_t core_idx_y = num_cores_with_work / num_cores_c;
                CoreCoord sc = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};
                in0_mcast_cores_without_work_and_in_receiver_grid = num_cores_to_corerangeset(
                    sc,
                    in0_mcast_cores_without_work_and_in_receiver_grid_num_cores,
                    compute_with_storage_grid_size,
                    row_major);
            }
            if (in0_sender_num_cores > in0_mcast_receiver_num_dests) {
                const uint32_t in0_mcast_cores_without_work_and_not_in_receiver_grid_num_cores =
                    in0_sender_num_cores - in0_mcast_receiver_num_dests;
                uint32_t core_idx_x = in0_mcast_receiver_num_dests % num_cores_c;
                uint32_t core_idx_y = in0_mcast_receiver_num_dests / num_cores_c;
                CoreCoord sc = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};
                in0_mcast_cores_without_work_and_not_in_receiver_grid = num_cores_to_corerangeset(
                    sc,
                    in0_mcast_cores_without_work_and_not_in_receiver_grid_num_cores,
                    compute_with_storage_grid_size,
                    row_major);
            }
            in0_mcast_noc_x.reserve(in0_mcast_sender_cores_grid.x);
            in0_mcast_noc_y.reserve(in0_mcast_sender_cores_grid.y);
            for (uint32_t core_idx_x = 0; core_idx_x < in0_mcast_sender_cores_grid.x; ++core_idx_x) {
                in0_mcast_noc_x.push_back(device->worker_core_from_logical_core({core_idx_x, 0}).x);
            }
            for (uint32_t core_idx_y = 0; core_idx_y < in0_mcast_sender_cores_grid.y; ++core_idx_y) {
                in0_mcast_noc_y.push_back(device->worker_core_from_logical_core({0, core_idx_y}).y);
            }
        } else {
            in0_mcast_cores_with_work_and_in_receiver_grid = CoreRangeSet({CoreRange(start_core, start_core)});
            if (in0_mcast_receiver_num_cores > 1) {
                auto receiver_start_core = start_core.x != (compute_with_storage_grid_size.x - 1)
                                               ? CoreCoord{start_core.x + 1, start_core.y}
                                               : CoreCoord{start_core.x, start_core.y + 1};
                in0_mcast_receivers = num_cores_to_corerangeset(
                    receiver_start_core, num_cores_local - 1, compute_with_storage_grid_size, row_major);
            }
        }

        // Semaphores
        uint32_t in0_mcast_sender_semaphore_id = 0;
        uint32_t in0_mcast_receiver_semaphore_id = 1;
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = in0_mcast_sender_semaphore_id,
            .core_ranges = all_cores,
            .initial_value = INVALID,
        });
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = in0_mcast_receiver_semaphore_id,
            .core_ranges = all_cores,
            .initial_value = INVALID,
        });

        CoreCoord top_left_core = in0_mcast_receiver_cores_bounding_box.start_coord;
        CoreCoord bottom_right_core = in0_mcast_receiver_cores_bounding_box.end_coord;
        auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
        auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

        uint32_t in0_num_subblocks = (out_block_h / out_subblock_h);
        uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
        const auto& a_shape_logical = operations::matmul::utilities::get_matmul_tensor_logical_shape(a, transpose_a);
        const auto in0_last_ktile_w = a_shape_logical[-1] % in0_tile.get_width();

        const auto in0_tensor_stride_w = transpose_a ? M : 1;
        const auto in0_tensor_stride_h = transpose_a ? 1 : K;
        const auto in0_tensor_next_block_stride = in0_block_w * in0_tensor_stride_w;
        const auto in0_tensor_next_h_dim_block_stride = in0_block_h * in0_tensor_stride_h;
        const auto in0_tensor_start_tile_id_stride = per_core_M * in0_tensor_stride_h;

        const auto in1_tensor_stride_w = transpose_b ? K : 1;
        const auto in1_tensor_stride_h = transpose_b ? 1 : N;
        const auto in1_tensor_next_block_stride = in0_block_w * in1_tensor_stride_h;
        const auto in1_tensor_next_w_dim_block_stride = in1_block_w * in1_tensor_stride_w;
        const auto in1_tensor_start_tile_id_stride = per_core_N * in1_tensor_stride_w;

        // Compile time args for in0 sender
        std::vector<uint32_t> in0_sender_compile_time_args;
        if (in0_is_sharded) {
            in0_sender_compile_time_args = {
                (std::uint32_t)1,  // core_has_output_block_work
                (std::uint32_t)1,  // core_in_in0_receiver_mcast_grid
                (std::uint32_t)in0_block_num_tiles,
                (std::uint32_t)in0_block_num_tiles * in0_single_tile_size,
                (std::uint32_t)in0_last_ktile_w,
                (std::uint32_t)num_blocks,
                (std::uint32_t)out_num_blocks_x,
                (std::uint32_t)out_num_blocks_y,
                (std::uint32_t)in0_mcast_sender_semaphore_id,
                (std::uint32_t)in0_mcast_receiver_semaphore_id,
                (std::uint32_t)in0_mcast_receiver_num_dests,
                (std::uint32_t)in0_mcast_receiver_num_cores,
                (std::uint32_t)(in0_mcast_sender_cores_grid.x),
                (std::uint32_t)(in0_mcast_sender_cores_grid.y),
                (std::uint32_t)(false),
                (std::uint32_t)(in0_shard_width_in_tiles),
                (std::uint32_t)(in0_shard_height_in_tiles),
                (std::uint32_t)(in0_block_w),
                (std::uint32_t)in0_block_h,
                (std::uint32_t)B,
            };
        } else {
            in0_sender_compile_time_args = {
                (std::uint32_t)in0_tensor_stride_w,
                (std::uint32_t)in0_tensor_stride_h,
                (std::uint32_t)in0_tensor_next_block_stride,
                (std::uint32_t)in0_tensor_next_h_dim_block_stride,
                (std::uint32_t)in0_block_w,
                (std::uint32_t)in0_block_h,
                (std::uint32_t)in0_block_num_tiles,
                (std::uint32_t)in0_last_ktile_w,
                (std::uint32_t)false,
                (std::uint32_t)0,
                (std::uint32_t)0,
                (std::uint32_t)num_blocks,
                (std::uint32_t)out_num_blocks_x,
                (std::uint32_t)out_num_blocks_y,
                (std::uint32_t)in0_mcast_sender_semaphore_id,
                (std::uint32_t)in0_mcast_receiver_semaphore_id,
                (std::uint32_t)num_cores_local - 1,
                (std::uint32_t)in0_mcast_receiver_num_cores - 1,
                (std::uint32_t)M * K,
                (std::uint32_t)B,
                (std::uint32_t)0,
                (std::uint32_t)0,
                (std::uint32_t)true,
                (std::uint32_t)false,
            };
        }
        in0_sender_compile_time_args.push_back((std::uint32_t)false);  // fuse_op && is_all_gather
        tt::tt_metal::TensorAccessorArgs(*in0_buffer).append_to(in0_sender_compile_time_args);
        tt::tt_metal::TensorAccessorArgs().append_to(in0_sender_compile_time_args);

        // Compile time args for in1 sender/writer
        std::vector<uint32_t> in1_sender_writer_compile_time_args = {
            (std::uint32_t)in1_tensor_stride_w,
            (std::uint32_t)in1_tensor_stride_h,
            (std::uint32_t)in1_tensor_next_block_stride,
            (std::uint32_t)in1_tensor_next_w_dim_block_stride,
            (std::uint32_t)in1_block_w,
            (std::uint32_t)in0_block_w,
            (std::uint32_t)in1_block_w * in0_block_w,
            (std::uint32_t)num_blocks,
            (std::uint32_t)out_num_blocks_x,
            (std::uint32_t)out_num_blocks_y,
            (std::uint32_t)0,
            (std::uint32_t)0,
            (std::uint32_t)0,
            (std::uint32_t)0,
            (std::uint32_t)K * N,
            (std::uint32_t)B,
            (std::uint32_t)bcast_batch,
            (std::uint32_t)0,
            (std::uint32_t)0,
            (std::uint32_t)1,
            (std::uint32_t)N,
            (std::uint32_t)out_subblock_w,
            (std::uint32_t)out_subblock_h * N,
            (std::uint32_t)out_block_w,
            (std::uint32_t)out_block_h * N,
            (std::uint32_t)out_subblock_w,
            (std::uint32_t)out_subblock_h,
            (std::uint32_t)(out_subblock_w * out_subblock_h),
            (std::uint32_t)M * N,
        };
        if (bias_buffer != nullptr) {
            in1_sender_writer_compile_time_args.push_back((std::uint32_t)1);
        } else {
            in1_sender_writer_compile_time_args.push_back(0);
        }
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)false);  // fuse_op && is_all_gather
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)false);  // fuse_op && is_reduce_scatter
        tt::tt_metal::TensorAccessorArgs(*in1_buffer).append_to(in1_sender_writer_compile_time_args);
        tt::tt_metal::TensorAccessorArgs().append_to(in1_sender_writer_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(*out_buffer).append_to(in1_sender_writer_compile_time_args);
        if (bias_buffer != nullptr) {
            tt::tt_metal::TensorAccessorArgs(*bias_buffer).append_to(in1_sender_writer_compile_time_args);
        }

        // Compile time args for in0 receiver
        std::vector<uint32_t> in0_receiver_compile_time_args = {
            (std::uint32_t)in0_block_num_tiles,
            (std::uint32_t)num_blocks,
            (std::uint32_t)out_num_blocks_x,
            (std::uint32_t)out_num_blocks_y,
            (std::uint32_t)in0_mcast_sender_semaphore_id,
            (std::uint32_t)in0_mcast_receiver_semaphore_id,
            (std::uint32_t)B,
            (std::uint32_t)false,
        };

        // ---- Defines ----
        std::map<std::string, std::string> mm_kernel_defines_map;
        KernelDescriptor::Defines mm_kernel_in0_sender_writer_defines;
        KernelDescriptor::Defines mm_kernel_in1_sender_writer_defines;
        if (bias_buffer != nullptr) {
            mm_kernel_defines_map["FUSE_BIAS"] = "1";
            mm_kernel_in1_sender_writer_defines.emplace_back("FUSE_BIAS", "1");
        }
        if (fused_activation.has_value()) {
            if (fused_activation.value().op_type == UnaryOpType::RELU) {
                mm_kernel_defines_map["PACK_RELU"] = "1";
            } else {
                using ttnn::operations::unary::utils::get_defines;
                mm_kernel_defines_map.merge(get_defines(
                    fused_activation.value().op_type,
                    fused_activation.value().params,
                    "ACTIVATION",
                    "i",
                    tt_metal::dataformat_to_datatype_converter(output_data_format)));
            }
        }
        if (packer_l1_acc_en) {
            mm_kernel_defines_map["PACKER_L1_ACC"] = "1";
        }
        if (fp32_dest_acc_en) {
            mm_kernel_defines_map["FP32_DEST_ACC_EN"] = "1";
        }
        if (in1_transpose_tile) {
            mm_kernel_defines_map["IN1_TRANSPOSE_TILE"] = "1";
        }
        ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
            device->arch(), num_cores_local, mm_kernel_defines_map);
        ttnn::operations::compute_throttle_utils::throttle_mm_perf(
            device->arch(), num_cores_local, mm_kernel_defines_map, throttle_level);

        KernelDescriptor::Defines compute_defines;
        for (auto& [k, v] : mm_kernel_defines_map) {
            compute_defines.emplace_back(k, v);
        }

        if (in1_is_sharded) {
            mm_kernel_in1_sender_writer_defines.emplace_back("IN1_SHARDED", "1");
        }
        if (bias_is_sharded) {
            mm_kernel_in1_sender_writer_defines.emplace_back("BIAS_SHARDED", "1");
        }
        if (output_is_sharded) {
            mm_kernel_in1_sender_writer_defines.emplace_back("OUT_SHARDED", "1");
        }
        if (in0_mcast_receiver_num_cores == 1) {
            mm_kernel_in0_sender_writer_defines.emplace_back("SKIP_MCAST", "1");
        }
        mm_kernel_in1_sender_writer_defines.emplace_back("SKIP_MCAST", "1");

        // Intermediate CB read (Blackhole alignment workaround)
        bool in0_needs_intermediate_cb_read = false;
        bool in1_needs_intermediate_cb_read = false;
        if (device->arch() == tt::ARCH::BLACKHOLE) {
            in0_needs_intermediate_cb_read = ((in0_single_tile_size % 64) != 0);
            if (in0_needs_intermediate_cb_read) {
                mm_kernel_in0_sender_writer_defines.emplace_back("INTERMEDIATE_CB_READ", "1");
            }
            in1_needs_intermediate_cb_read = ((in1_single_tile_size % 64) != 0);
            if (in1_needs_intermediate_cb_read) {
                mm_kernel_in1_sender_writer_defines.emplace_back("INTERMEDIATE_CB_READ", "1");
            }
        }

        tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
        tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

        // ---- Kernel Descriptors ----
        // in0 sender - with work and in receiver grid
        KernelDescriptor in0_sender_with_work_desc;
        in0_sender_with_work_desc.kernel_source =
            in0_is_sharded ? "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                             "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp"
                           : "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                             "reader_bmm_tile_layout_in0_sender_padding.cpp";
        in0_sender_with_work_desc.core_ranges = in0_mcast_cores_with_work_and_in_receiver_grid;
        in0_sender_with_work_desc.compile_time_args = in0_sender_compile_time_args;
        in0_sender_with_work_desc.defines = mm_kernel_in0_sender_writer_defines;
        in0_sender_with_work_desc.config = DataMovementConfigDescriptor{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
        };

        // in0 sender - without work, in receiver grid (sharded only)
        KernelDescriptor in0_sender_no_work_in_grid_desc;
        bool has_no_work_in_grid = in0_is_sharded && in0_mcast_cores_without_work_and_in_receiver_grid.num_cores() > 0;
        if (has_no_work_in_grid) {
            auto no_work_args = in0_sender_compile_time_args;
            no_work_args[0] = 0;  // core_has_output_block_work
            no_work_args[1] = 1;  // core_in_in0_receiver_mcast_grid
            in0_sender_no_work_in_grid_desc.kernel_source =
                "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp";
            in0_sender_no_work_in_grid_desc.core_ranges = in0_mcast_cores_without_work_and_in_receiver_grid;
            in0_sender_no_work_in_grid_desc.compile_time_args = no_work_args;
            in0_sender_no_work_in_grid_desc.defines = mm_kernel_in0_sender_writer_defines;
            in0_sender_no_work_in_grid_desc.config = DataMovementConfigDescriptor{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = in0_noc,
            };
        }

        // in0 sender - without work, not in receiver grid (sharded only)
        KernelDescriptor in0_sender_no_work_no_grid_desc;
        bool has_no_work_no_grid =
            in0_is_sharded && in0_mcast_cores_without_work_and_not_in_receiver_grid.num_cores() > 0;
        if (has_no_work_no_grid) {
            auto no_work_args = in0_sender_compile_time_args;
            no_work_args[0] = 0;  // core_has_output_block_work
            no_work_args[1] = 0;  // core_in_in0_receiver_mcast_grid
            in0_sender_no_work_no_grid_desc.kernel_source =
                "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp";
            in0_sender_no_work_no_grid_desc.core_ranges = in0_mcast_cores_without_work_and_not_in_receiver_grid;
            in0_sender_no_work_no_grid_desc.compile_time_args = no_work_args;
            in0_sender_no_work_no_grid_desc.defines = mm_kernel_in0_sender_writer_defines;
            in0_sender_no_work_no_grid_desc.config = DataMovementConfigDescriptor{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = in0_noc,
            };
        }

        // in0 receiver (non-sharded only)
        KernelDescriptor in0_receiver_desc;
        bool has_in0_receiver = !in0_is_sharded && in0_mcast_receivers.num_cores() > 0;
        if (has_in0_receiver) {
            in0_receiver_desc.kernel_source =
                "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp";
            in0_receiver_desc.core_ranges = in0_mcast_receivers;
            in0_receiver_desc.compile_time_args = in0_receiver_compile_time_args;
            in0_receiver_desc.config = DataMovementConfigDescriptor{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = in0_noc,
            };
        }

        // in1 sender/writer
        KernelDescriptor in1_sender_writer_desc;
        in1_sender_writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in1_sender_writer_padding.cpp";
        in1_sender_writer_desc.core_ranges = all_cores_with_work;
        in1_sender_writer_desc.compile_time_args = in1_sender_writer_compile_time_args;
        in1_sender_writer_desc.defines = mm_kernel_in1_sender_writer_defines;
        in1_sender_writer_desc.config = DataMovementConfigDescriptor{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
        };

        // Compute kernel
        uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
        uint32_t in1_num_subblocks = (out_block_w / out_subblock_w);
        uint32_t in1_block_num_tiles_compute = out_subblock_w * in0_block_w * in1_num_subblocks;
        uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;
        uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

        std::vector<uint32_t> compute_kernel_args = {
            in0_block_w,
            in0_num_subblocks,
            in0_block_num_tiles,
            in0_subblock_num_tiles,
            in1_num_subblocks,
            in1_block_num_tiles_compute,
            in1_per_core_w,
            num_blocks,
            out_num_blocks_x,
            out_num_blocks_y,
            out_subblock_h,
            out_subblock_w,
            out_subblock_num_tiles,
            B,
            out_block_tiles,
            (uint32_t)untilize_out,
            (uint32_t)false,
            (uint32_t)in0_transpose_tile,
        };

        KernelDescriptor compute_desc;
        compute_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp";
        compute_desc.core_ranges = all_cores_with_work;
        compute_desc.compile_time_args = compute_kernel_args;
        compute_desc.defines = compute_defines;
        compute_desc.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
        };

        // ---- Circular Buffers ----
        // CB src0
        desc.cbs.push_back(CBDescriptor{
            .total_size = in0_CB_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
                .data_format = in0_data_format,
                .page_size = in0_single_tile_size,
                .tile = TileDescriptor(in0_tile),
            }}},
        });
        // CB src1
        {
            CBDescriptor cb;
            cb.total_size = in1_CB_size;
            cb.core_ranges = all_cores;
            cb.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_1),
                .data_format = in1_data_format,
                .page_size = in1_single_tile_size,
                .tile = TileDescriptor(in1_tile),
            });
            if (in1_is_sharded) {
                cb.buffer = in1_buffer;
            }
            desc.cbs.push_back(std::move(cb));
        }
        // CB src2 (sharded in0)
        if (in0_is_sharded) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = in2_CB_size,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_2),
                    .data_format = in0_data_format,
                    .page_size = in0_single_tile_size,
                    .tile = TileDescriptor(in0_tile),
                }}},
                .buffer = in0_buffer,
            });
            // Local L1 temp vars
            desc.cbs.push_back(CBDescriptor{
                .total_size = 32 * 2,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_6),
                    .data_format = tt::DataFormat::Float16_b,
                    .page_size = 32 * 2,
                }}},
            });
        }
        // Output/interm0 CBs
        if (do_not_inplace_interm0_out_CB || (interm0_data_format != output_data_format) ||
            (untilize_out && (in1_num_subblocks > 1))) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = interm0_CB_size,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_5),
                    .data_format = interm0_data_format,
                    .page_size = interm0_single_tile_size,
                    .tile = TileDescriptor(output_tile),
                }}},
            });
            desc.cbs.push_back(CBDescriptor{
                .total_size = out_CB_size,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_4),
                    .data_format = output_data_format,
                    .page_size = output_single_tile_size,
                    .tile = TileDescriptor(output_tile),
                }}},
                .buffer = output_is_sharded ? out_buffer : nullptr,
            });
        } else {
            CBDescriptor cb;
            cb.total_size = out_CB_size;
            cb.core_ranges = all_cores;
            cb.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_4),
                .data_format = output_data_format,
                .page_size = output_single_tile_size,
                .tile = TileDescriptor(output_tile),
            });
            cb.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_5),
                .data_format = interm0_data_format,
                .page_size = interm0_single_tile_size,
                .tile = TileDescriptor(output_tile),
            });
            if (output_is_sharded) {
                cb.buffer = out_buffer;
            }
            desc.cbs.push_back(std::move(cb));
        }
        // CB bias
        if (bias_buffer != nullptr) {
            CBDescriptor cb;
            cb.total_size = in3_CB_size;
            cb.core_ranges = all_cores;
            cb.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
                .data_format = bias_data_format,
                .page_size = bias_single_tile_size,
                .tile = TileDescriptor(bias_tile),
            });
            if (bias_is_sharded) {
                cb.buffer = bias_buffer;
            }
            desc.cbs.push_back(std::move(cb));
        }
        // Intermediate CB read (Blackhole)
        if (in1_needs_intermediate_cb_read) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = in1_single_tile_size,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_9),
                    .data_format = in1_data_format,
                    .page_size = in1_single_tile_size,
                    .tile = TileDescriptor(in1_tile),
                }}},
            });
        }
        if (in0_needs_intermediate_cb_read) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = in0_single_tile_size,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_8),
                    .data_format = in0_data_format,
                    .page_size = in0_single_tile_size,
                    .tile = TileDescriptor(in0_tile),
                }}},
            });
        }
        // Transpose CB
        if (in0_transpose_tile) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = in0_CB_size,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_10),
                    .data_format = in0_data_format,
                    .page_size = in0_single_tile_size,
                    .tile = TileDescriptor(in0_tile),
                }}},
            });
        }

        // ---- Runtime Args ----
        uint32_t last_per_core_N = N % per_core_N == 0 ? per_core_N : N % per_core_N;
        uint32_t last_out_block_w = last_per_core_N % out_block_w == 0 ? out_block_w : last_per_core_N % out_block_w;
        uint32_t last_out_num_blocks_w = ((last_per_core_N - 1) / out_block_w) + 1;
        uint32_t last_block_num_nonzero_subblocks_w = ((last_out_block_w - 1) / out_subblock_w) + 1;
        uint32_t last_subblock_of_last_block_w =
            last_out_block_w % out_subblock_w == 0 ? out_subblock_w : last_out_block_w % out_subblock_w;
        uint32_t last_block_padded_subblock_tiles_addr_skip =
            output_single_tile_size * (out_subblock_w - last_subblock_of_last_block_w);
        uint32_t last_block_padded_block_tiles_w_skip =
            (out_subblock_w * out_subblock_h) * (out_block_w / out_subblock_w - last_block_num_nonzero_subblocks_w);

        CoreCoord start_core_noc = top_left_core_physical;
        CoreCoord end_core_noc = bottom_right_core_physical;
        if (in0_noc == tt::tt_metal::NOC::NOC_1) {
            std::swap(start_core_noc, end_core_noc);
        }

        const auto& cores = corerange_to_cores(all_cores, std::nullopt, row_major);
        for (uint32_t i = 0; i < num_cores_local; ++i) {
            const auto& core = cores[i];
            uint32_t output_idx_x = i % num_blocks_x;
            uint32_t output_idx_y = i / num_blocks_x;

            if (in0_is_sharded) {
                KernelDescriptor::CoreRuntimeArgs mm_in0_sender_args;
                mm_in0_sender_args.reserve(5 + in0_mcast_noc_x.size() + in0_mcast_noc_y.size());
                mm_in0_sender_args.push_back(i);
                mm_in0_sender_args.push_back(start_core_noc.x);
                mm_in0_sender_args.push_back(start_core_noc.y);
                mm_in0_sender_args.push_back(end_core_noc.x);
                mm_in0_sender_args.push_back(end_core_noc.y);
                mm_in0_sender_args.insert(mm_in0_sender_args.end(), in0_mcast_noc_x.begin(), in0_mcast_noc_x.end());
                mm_in0_sender_args.insert(mm_in0_sender_args.end(), in0_mcast_noc_y.begin(), in0_mcast_noc_y.end());

                if (i < num_cores_with_work) {
                    in0_sender_with_work_desc.runtime_args.emplace_back(core, std::move(mm_in0_sender_args));
                } else if (i < in0_mcast_receiver_num_dests) {
                    in0_sender_no_work_in_grid_desc.runtime_args.emplace_back(core, std::move(mm_in0_sender_args));
                } else {
                    in0_sender_no_work_no_grid_desc.runtime_args.emplace_back(core, std::move(mm_in0_sender_args));
                }
            } else if (core == start_core) {
                in0_sender_with_work_desc.runtime_args.emplace_back(
                    core,
                    KernelDescriptor::CoreRuntimeArgs{
                        (std::uint32_t)in0_buffer->address(),
                        (std::uint32_t)in0_tensor_start_tile_id_stride * output_idx_y,
                        (std::uint32_t)start_core_noc.x,
                        (std::uint32_t)start_core_noc.y,
                        (std::uint32_t)end_core_noc.x,
                        (std::uint32_t)end_core_noc.y,
                        (std::uint32_t)out_block_h,
                        (std::uint32_t)0,
                    });
            } else {
                in0_receiver_desc.runtime_args.emplace_back(
                    core,
                    KernelDescriptor::CoreRuntimeArgs{
                        (std::uint32_t)top_left_core_physical.x,
                        (std::uint32_t)top_left_core_physical.y,
                    });
            }

            if (i < num_cores_with_work) {
                KernelDescriptor::CoreRuntimeArgs mm_in1_sender_writer_args = {
                    (std::uint32_t)in1_buffer->address(),
                    (std::uint32_t)in1_tensor_start_tile_id_stride * output_idx_x,
                    (std::uint32_t)0,
                    (std::uint32_t)0,
                    (std::uint32_t)0,
                    (std::uint32_t)0,
                    (std::uint32_t)0,
                    (std::uint32_t)out_buffer->address(),
                    ((std::uint32_t)output_idx_x * per_core_N) + (output_idx_y * per_core_M * N),
                };

                if (output_idx_x == num_blocks_x - 1) {
                    mm_in1_sender_writer_args.push_back(last_out_block_w);
                    mm_in1_sender_writer_args.push_back(out_block_h / out_subblock_h);
                    mm_in1_sender_writer_args.push_back(out_subblock_h);
                    mm_in1_sender_writer_args.push_back(0);
                    mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_sender_writer_args.push_back(last_block_num_nonzero_subblocks_w);
                    mm_in1_sender_writer_args.push_back(last_subblock_of_last_block_w);
                    mm_in1_sender_writer_args.push_back(last_block_padded_subblock_tiles_addr_skip);
                    mm_in1_sender_writer_args.push_back(last_block_padded_block_tiles_w_skip);
                } else {
                    mm_in1_sender_writer_args.push_back(out_block_w);
                    mm_in1_sender_writer_args.push_back(out_block_h / out_subblock_h);
                    mm_in1_sender_writer_args.push_back(out_subblock_h);
                    mm_in1_sender_writer_args.push_back(0);
                    mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_sender_writer_args.push_back(out_subblock_w);
                    mm_in1_sender_writer_args.push_back(0);
                    mm_in1_sender_writer_args.push_back(0);
                }

                mm_in1_sender_writer_args.push_back(bias_buffer ? (std::uint32_t)bias_buffer->address() : 0);
                mm_in1_sender_writer_args.push_back(bias_buffer ? (std::uint32_t)per_core_N * output_idx_x : 0);
                if (!output_is_sharded) {
                    if (output_idx_x == num_blocks_x - 1) {
                        mm_in1_sender_writer_args.push_back(last_out_num_blocks_w);
                    } else {
                        mm_in1_sender_writer_args.push_back(out_num_blocks_x);
                    }
                }

                in1_sender_writer_desc.runtime_args.emplace_back(core, std::move(mm_in1_sender_writer_args));
            }
        }

        desc.kernels.push_back(std::move(in0_sender_with_work_desc));
        if (has_no_work_in_grid) {
            desc.kernels.push_back(std::move(in0_sender_no_work_in_grid_desc));
        }
        if (has_no_work_no_grid) {
            desc.kernels.push_back(std::move(in0_sender_no_work_no_grid_desc));
        }
        if (has_in0_receiver) {
            desc.kernels.push_back(std::move(in0_receiver_desc));
        }
        desc.kernels.push_back(std::move(in1_sender_writer_desc));
        desc.kernels.push_back(std::move(compute_desc));

    } else {
        // ====================================================================
        // MCAST_IN1 PATH
        // ====================================================================
        TT_FATAL(tt::CBIndex::c_0 == tt::CBIndex::c_0, "mcast does not support a non-zero start cb index");

        bool in0_is_sharded = a.memory_config().is_sharded();
        bool output_is_sharded = output.memory_config().is_sharded();

        bool in0_transpose_tile = in0_tile.get_transpose_of_faces() && in0_tile.get_transpose_within_face();
        bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();

        uint32_t num_blocks = K / in0_block_w;
        bool packer_l1_acc_en = packer_l1_acc && (((bias_buffer != nullptr) && num_blocks > 1) || (num_blocks > 2));

        tt::DataFormat interm0_data_format =
            packer_l1_acc_en ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

        uint32_t bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);
        uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
        uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

        bool do_not_inplace_interm0_out_CB = output_is_sharded && (per_core_M != out_block_h);

        uint32_t in0_block_h = out_block_h;
        uint32_t in1_block_w = out_block_w;
        uint32_t in0_num_blocks_y = per_core_M / out_block_h;
        uint32_t in1_num_blocks_x = per_core_N / out_block_w;
        uint32_t out_num_blocks_x = in1_num_blocks_x;
        uint32_t out_num_blocks_y = in0_num_blocks_y;

        uint32_t in0_block_tiles = in0_block_h * in0_block_w;
        uint32_t in0_CB_tiles = in0_block_tiles;
        if (in0_is_sharded) {
            in0_CB_tiles = num_blocks * per_core_M * in0_block_w * B;
        } else if (B * num_blocks > 1) {
            in0_CB_tiles = in0_CB_tiles * 2;
        }
        uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;

        const auto& a_shape_logical = operations::matmul::utilities::get_matmul_tensor_logical_shape(a, transpose_a);
        const auto in0_last_ktile_w = a_shape_logical[-1] % in0_tile.get_width();

        bool extract_shard_sub_blocks = false;
        uint32_t in0_shard_height_in_tiles = 0;
        uint32_t in0_shard_width_in_tiles = 0;
        if (in0_is_sharded) {
            in0_shard_height_in_tiles = in0_buffer->shard_spec().shape()[0] / in0_tile.get_height();
            in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / in0_tile.get_width();
            if (in0_shard_width_in_tiles / in0_block_w > 1) {
                extract_shard_sub_blocks = true;
            }
        }
        uint32_t in2_CB_tiles = in0_block_tiles;
        uint32_t in2_CB_size = in2_CB_tiles * in0_single_tile_size;

        uint32_t in1_block_tiles = out_block_w * in0_block_w;
        uint32_t in1_CB_tiles = in1_block_tiles;
        if (B * num_blocks > 1) {
            in1_CB_tiles = in1_CB_tiles * 2;
        }
        uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

        uint32_t out_block_tiles = out_block_h * out_block_w;
        uint32_t out_shard_tiles = per_core_M * per_core_N;
        uint32_t out_CB_tiles = out_block_tiles;
        if (output_is_sharded) {
            out_CB_tiles = out_shard_tiles;
        }
        uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
        uint32_t interm0_CB_tiles = out_block_tiles;
        uint32_t interm0_CB_size = interm0_CB_tiles * interm0_single_tile_size;

        uint32_t in3_block_tiles = out_block_w;
        uint32_t in3_CB_tiles = in3_block_tiles;
        uint32_t in3_CB_size = in3_CB_tiles * bias_single_tile_size;

        CoreCoord start_core = {0, 0};
        uint32_t num_cores_local = num_blocks_total;

        constexpr bool row_major = true;
        CoreRangeSet all_cores =
            num_cores_to_corerangeset(start_core, num_cores_local, compute_with_storage_grid_size, row_major);
        CoreRange in1_mcast_receiver_cores_bounding_box = all_cores.bounding_box();
        uint32_t in1_mcast_receiver_num_cores = in1_mcast_receiver_cores_bounding_box.size();

        CoreRange in1_mcast_sender(start_core, start_core);
        CoreRangeSet in1_mcast_receivers;
        if (in1_mcast_receiver_num_cores > 1) {
            auto receiver_start_core = start_core.x != (compute_with_storage_grid_size.x - 1)
                                           ? CoreCoord{start_core.x + 1, start_core.y}
                                           : CoreCoord{start_core.x, start_core.y + 1};
            in1_mcast_receivers = num_cores_to_corerangeset(
                receiver_start_core, num_cores_local - 1, compute_with_storage_grid_size, row_major);
        }

        // Semaphores
        uint32_t in1_mcast_sender_semaphore_id = 0;
        uint32_t in1_mcast_receiver_semaphore_id = 1;
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = in1_mcast_sender_semaphore_id,
            .core_ranges = all_cores,
            .initial_value = INVALID,
        });
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = in1_mcast_receiver_semaphore_id,
            .core_ranges = all_cores,
            .initial_value = INVALID,
        });

        CoreCoord top_left_core = in1_mcast_receiver_cores_bounding_box.start_coord;
        CoreCoord bottom_right_core = in1_mcast_receiver_cores_bounding_box.end_coord;
        auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
        auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

        const auto in0_tensor_stride_w = transpose_a ? M : 1;
        const auto in0_tensor_stride_h = transpose_a ? 1 : K;
        const auto in0_tensor_next_block_stride = in0_block_w * in0_tensor_stride_w;
        const auto in0_tensor_next_h_dim_block_stride = in0_block_h * in0_tensor_stride_h;
        const auto in0_tensor_start_tile_id_stride = per_core_M * in0_tensor_stride_h;

        const auto in1_tensor_stride_w = transpose_b ? K : 1;
        const auto in1_tensor_stride_h = transpose_b ? 1 : N;
        const auto in1_tensor_next_block_stride = in0_block_w * in1_tensor_stride_h;
        const auto in1_tensor_next_w_dim_block_stride = in1_block_w * in1_tensor_stride_w;
        const auto in1_tensor_start_tile_id_stride = per_core_N * in1_tensor_stride_w;

        // in0 sender compile time args
        std::vector<uint32_t> in0_sender_compile_time_args = {
            (std::uint32_t)in0_tensor_stride_w,
            (std::uint32_t)in0_tensor_stride_h,
            (std::uint32_t)in0_tensor_next_block_stride,
            (std::uint32_t)in0_tensor_next_h_dim_block_stride,
            (std::uint32_t)in0_block_w,
            (std::uint32_t)in0_block_h,
            (std::uint32_t)in0_block_w * in0_block_h,
            (std::uint32_t)in0_last_ktile_w,
            (std::uint32_t)extract_shard_sub_blocks,
            (std::uint32_t)in0_shard_width_in_tiles,
            (std::uint32_t)in0_shard_height_in_tiles,
            (std::uint32_t)num_blocks,
            (std::uint32_t)out_num_blocks_x,
            (std::uint32_t)out_num_blocks_y,
            (std::uint32_t)0,
            (std::uint32_t)0,
            (std::uint32_t)0,
            (std::uint32_t)0,
            (std::uint32_t)M * K,
            (std::uint32_t)B,
            (std::uint32_t)0,
            (std::uint32_t)0,
            (std::uint32_t)true,
            (std::uint32_t)false,
        };
        in0_sender_compile_time_args.push_back((std::uint32_t)false);  // fuse_op
        tt::tt_metal::TensorAccessorArgs(*in0_buffer).append_to(in0_sender_compile_time_args);
        tt::tt_metal::TensorAccessorArgs().append_to(in0_sender_compile_time_args);

        // in1 sender/writer compile time args
        std::vector<uint32_t> in1_sender_writer_compile_time_args = {
            (std::uint32_t)in1_tensor_stride_w,
            (std::uint32_t)in1_tensor_stride_h,
            (std::uint32_t)in1_tensor_next_block_stride,
            (std::uint32_t)in1_tensor_next_w_dim_block_stride,
            (std::uint32_t)in1_block_w,
            (std::uint32_t)in0_block_w,
            (std::uint32_t)in1_block_w * in0_block_w,
            (std::uint32_t)num_blocks,
            (std::uint32_t)out_num_blocks_x,
            (std::uint32_t)out_num_blocks_y,
            (std::uint32_t)in1_mcast_sender_semaphore_id,
            (std::uint32_t)in1_mcast_receiver_semaphore_id,
            (std::uint32_t)num_cores_local - 1,
            (std::uint32_t)in1_mcast_receiver_num_cores - 1,
            (std::uint32_t)K * N,
            (std::uint32_t)B,
            (std::uint32_t)bcast_batch,
            (std::uint32_t)0,
            (std::uint32_t)0,
            (std::uint32_t)1,
            (std::uint32_t)N,
            (std::uint32_t)out_subblock_w,
            (std::uint32_t)out_subblock_h * N,
            (std::uint32_t)out_block_w,
            (std::uint32_t)out_block_h * N,
            (std::uint32_t)out_subblock_w,
            (std::uint32_t)out_subblock_h,
            (std::uint32_t)(out_subblock_w * out_subblock_h),
            (std::uint32_t)M * N,
        };
        if (bias_buffer != nullptr) {
            in1_sender_writer_compile_time_args.push_back((std::uint32_t)1);
        } else {
            in1_sender_writer_compile_time_args.push_back(0);
        }
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)false);  // fuse_op
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)false);  // fuse_op
        tt::tt_metal::TensorAccessorArgs(*in1_buffer).append_to(in1_sender_writer_compile_time_args);
        tt::tt_metal::TensorAccessorArgs().append_to(in1_sender_writer_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(*out_buffer).append_to(in1_sender_writer_compile_time_args);
        if (bias_buffer != nullptr) {
            tt::tt_metal::TensorAccessorArgs(*bias_buffer).append_to(in1_sender_writer_compile_time_args);
        }

        // in1 receiver/writer compile time args
        std::vector<uint32_t> in1_receiver_writer_compile_time_args = {
            (std::uint32_t)in1_block_w * in0_block_w,
            (std::uint32_t)num_blocks,
            (std::uint32_t)out_num_blocks_x,
            (std::uint32_t)out_num_blocks_y,
            (std::uint32_t)in1_mcast_sender_semaphore_id,
            (std::uint32_t)in1_mcast_receiver_semaphore_id,
            (std::uint32_t)B,
            (std::uint32_t)1,
            (std::uint32_t)N,
            (std::uint32_t)out_subblock_w,
            (std::uint32_t)out_subblock_h * N,
            (std::uint32_t)out_block_w,
            (std::uint32_t)out_block_h * N,
            (std::uint32_t)out_subblock_w,
            (std::uint32_t)out_subblock_h,
            (std::uint32_t)(out_subblock_w * out_subblock_h),
            (std::uint32_t)M * N,
        };
        if (bias_buffer != nullptr) {
            in1_receiver_writer_compile_time_args.push_back((std::uint32_t)in1_block_w);
        } else {
            in1_receiver_writer_compile_time_args.push_back(0);
        }
        in1_receiver_writer_compile_time_args.push_back((std::uint32_t)false);  // fuse_op
        tt::tt_metal::TensorAccessorArgs(*out_buffer).append_to(in1_receiver_writer_compile_time_args);

        // ---- Defines ----
        std::map<std::string, std::string> mm_kernel_defines_map;
        KernelDescriptor::Defines mm_kernel_in0_sender_defines;
        KernelDescriptor::Defines mm_kernel_in1_sender_writer_defines;
        KernelDescriptor::Defines mm_kernel_in1_receiver_writer_defines;
        if (bias_buffer != nullptr) {
            mm_kernel_defines_map["FUSE_BIAS"] = "1";
            mm_kernel_in1_sender_writer_defines.emplace_back("FUSE_BIAS", "1");
            mm_kernel_in1_receiver_writer_defines.emplace_back("FUSE_BIAS", "1");
        }
        if (fused_activation.has_value()) {
            if (fused_activation.value().op_type == UnaryOpType::RELU) {
                mm_kernel_defines_map["PACK_RELU"] = "1";
            } else {
                using ttnn::operations::unary::utils::get_defines;
                mm_kernel_defines_map.merge(
                    get_defines(fused_activation.value().op_type, fused_activation.value().params, "ACTIVATION", "i"));
            }
        }
        if (packer_l1_acc_en) {
            mm_kernel_defines_map["PACKER_L1_ACC"] = "1";
        }
        if (fp32_dest_acc_en) {
            mm_kernel_defines_map["FP32_DEST_ACC_EN"] = "1";
        }
        if (in1_transpose_tile) {
            mm_kernel_defines_map["IN1_TRANSPOSE_TILE"] = "1";
        }
        ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
            device->arch(), num_cores_local, mm_kernel_defines_map);
        ttnn::operations::compute_throttle_utils::throttle_mm_perf(
            device->arch(), num_cores_local, mm_kernel_defines_map, throttle_level);

        KernelDescriptor::Defines compute_defines;
        for (auto& [k, v] : mm_kernel_defines_map) {
            compute_defines.emplace_back(k, v);
        }

        if (in0_is_sharded) {
            mm_kernel_in0_sender_defines.emplace_back("IN0_SHARDED", "1");
        }
        if (output_is_sharded) {
            mm_kernel_in1_sender_writer_defines.emplace_back("OUT_SHARDED", "1");
            mm_kernel_in1_receiver_writer_defines.emplace_back("OUT_SHARDED", "1");
        }
        mm_kernel_in0_sender_defines.emplace_back("SKIP_MCAST", "1");
        if (in1_mcast_receiver_num_cores == 1) {
            mm_kernel_in1_sender_writer_defines.emplace_back("SKIP_MCAST", "1");
        }

        bool in0_needs_intermediate_cb_read = false;
        bool in1_needs_intermediate_cb_read = false;
        if (device->arch() == tt::ARCH::BLACKHOLE) {
            in0_needs_intermediate_cb_read = ((in0_single_tile_size % 64) != 0);
            if (in0_needs_intermediate_cb_read) {
                mm_kernel_in0_sender_defines.emplace_back("INTERMEDIATE_CB_READ", "1");
            }
            in1_needs_intermediate_cb_read = ((in1_single_tile_size % 64) != 0);
            if (in1_needs_intermediate_cb_read) {
                mm_kernel_in1_sender_writer_defines.emplace_back("INTERMEDIATE_CB_READ", "1");
            }
        }

        tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
        tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

        // ---- Kernel Descriptors ----
        KernelDescriptor in0_sender_desc;
        in0_sender_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp";
        in0_sender_desc.core_ranges = all_cores;
        in0_sender_desc.compile_time_args = in0_sender_compile_time_args;
        in0_sender_desc.defines = mm_kernel_in0_sender_defines;
        in0_sender_desc.config = DataMovementConfigDescriptor{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
        };

        KernelDescriptor in1_sender_writer_desc;
        in1_sender_writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in1_sender_writer_padding.cpp";
        in1_sender_writer_desc.core_ranges = in1_mcast_sender;
        in1_sender_writer_desc.compile_time_args = in1_sender_writer_compile_time_args;
        in1_sender_writer_desc.defines = mm_kernel_in1_sender_writer_defines;
        in1_sender_writer_desc.config = DataMovementConfigDescriptor{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
        };

        KernelDescriptor in1_receiver_writer_desc;
        bool has_in1_receivers = in1_mcast_receivers.num_cores() > 0;
        if (has_in1_receivers) {
            in1_receiver_writer_desc.kernel_source =
                "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in1_receiver_writer_padding.cpp";
            in1_receiver_writer_desc.core_ranges = in1_mcast_receivers;
            in1_receiver_writer_desc.compile_time_args = in1_receiver_writer_compile_time_args;
            in1_receiver_writer_desc.defines = mm_kernel_in1_receiver_writer_defines;
            in1_receiver_writer_desc.config = DataMovementConfigDescriptor{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = in1_noc,
            };
        }

        // Compute kernel
        uint32_t in0_num_subblocks = (out_block_h / out_subblock_h);
        uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
        uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
        uint32_t in1_num_subblocks = (out_block_w / out_subblock_w);
        uint32_t in1_block_num_tiles_compute = out_subblock_w * in0_block_w * in1_num_subblocks;
        uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;
        uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

        std::vector<uint32_t> compute_kernel_args = {
            in0_block_w,
            in0_num_subblocks,
            in0_block_num_tiles,
            in0_subblock_num_tiles,
            in1_num_subblocks,
            in1_block_num_tiles_compute,
            in1_per_core_w,
            num_blocks,
            out_num_blocks_x,
            out_num_blocks_y,
            out_subblock_h,
            out_subblock_w,
            out_subblock_num_tiles,
            B,
            out_block_tiles,
            (uint32_t)untilize_out,
            (uint32_t)false,
            (uint32_t)in0_transpose_tile,
        };

        KernelDescriptor compute_desc;
        compute_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp";
        compute_desc.core_ranges = all_cores;
        compute_desc.compile_time_args = compute_kernel_args;
        compute_desc.defines = compute_defines;
        compute_desc.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
        };

        // ---- Circular Buffers ----
        // CB src0
        {
            CBDescriptor cb;
            cb.total_size = in0_CB_size;
            cb.core_ranges = all_cores;
            cb.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
                .data_format = in0_data_format,
                .page_size = in0_single_tile_size,
                .tile = TileDescriptor(in0_tile),
            });
            if (in0_is_sharded && !extract_shard_sub_blocks) {
                cb.buffer = in0_buffer;
            }
            desc.cbs.push_back(std::move(cb));
        }
        // CB src2 (sharded in0 with extract)
        if (in0_is_sharded && extract_shard_sub_blocks) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = in2_CB_size,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_2),
                    .data_format = in0_data_format,
                    .page_size = in0_single_tile_size,
                    .tile = TileDescriptor(in0_tile),
                }}},
                .buffer = in0_buffer,
            });
        }
        // CB src1
        desc.cbs.push_back(CBDescriptor{
            .total_size = in1_CB_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_1),
                .data_format = in1_data_format,
                .page_size = in1_single_tile_size,
                .tile = TileDescriptor(in1_tile),
            }}},
        });
        // Output/interm0 CBs
        if (do_not_inplace_interm0_out_CB || (interm0_data_format != output_data_format) ||
            (untilize_out && (in1_num_subblocks > 1))) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = interm0_CB_size,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_5),
                    .data_format = interm0_data_format,
                    .page_size = interm0_single_tile_size,
                    .tile = TileDescriptor(output_tile),
                }}},
            });
            desc.cbs.push_back(CBDescriptor{
                .total_size = out_CB_size,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_4),
                    .data_format = output_data_format,
                    .page_size = output_single_tile_size,
                    .tile = TileDescriptor(output_tile),
                }}},
                .buffer = output_is_sharded ? out_buffer : nullptr,
            });
        } else {
            CBDescriptor cb;
            cb.total_size = out_CB_size;
            cb.core_ranges = all_cores;
            cb.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_4),
                .data_format = output_data_format,
                .page_size = output_single_tile_size,
                .tile = TileDescriptor(output_tile),
            });
            cb.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_5),
                .data_format = interm0_data_format,
                .page_size = interm0_single_tile_size,
                .tile = TileDescriptor(output_tile),
            });
            if (output_is_sharded) {
                cb.buffer = out_buffer;
            }
            desc.cbs.push_back(std::move(cb));
        }
        // CB bias
        if (bias_buffer != nullptr) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = in3_CB_size,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
                    .data_format = bias_data_format,
                    .page_size = bias_single_tile_size,
                    .tile = TileDescriptor(bias_tile),
                }}},
            });
        }
        // Intermediate CB read (Blackhole)
        if (in1_needs_intermediate_cb_read) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = in1_single_tile_size,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_9),
                    .data_format = in1_data_format,
                    .page_size = in1_single_tile_size,
                    .tile = TileDescriptor(in1_tile),
                }}},
            });
        }
        if (in0_needs_intermediate_cb_read) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = in0_single_tile_size,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_8),
                    .data_format = in0_data_format,
                    .page_size = in0_single_tile_size,
                    .tile = TileDescriptor(in0_tile),
                }}},
            });
        }
        // Transpose CB
        if (in0_transpose_tile) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = in0_CB_size,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_10),
                    .data_format = in0_data_format,
                    .page_size = in0_single_tile_size,
                    .tile = TileDescriptor(in0_tile),
                }}},
            });
        }

        // ---- Runtime Args ----
        uint32_t last_per_core_M = M % per_core_M == 0 ? per_core_M : M % per_core_M;
        uint32_t last_out_block_h = last_per_core_M % out_block_h == 0 ? out_block_h : last_per_core_M % out_block_h;
        uint32_t last_out_num_blocks_h = ((last_per_core_M - 1) / out_block_h) + 1;
        uint32_t last_block_num_nonzero_subblocks_h = ((last_out_block_h - 1) / out_subblock_h) + 1;
        uint32_t last_subblock_of_last_block_h =
            last_out_block_h % out_subblock_h == 0 ? out_subblock_h : last_out_block_h % out_subblock_h;
        uint32_t last_block_padded_block_tiles_h_skip =
            (out_block_h / out_subblock_h - last_block_num_nonzero_subblocks_h) * (out_block_w * out_subblock_h);

        CoreCoord start_core_noc = bottom_right_core_physical;
        CoreCoord end_core_noc = top_left_core_physical;
        if (in1_noc == tt::tt_metal::NOC::NOC_0) {
            std::swap(start_core_noc, end_core_noc);
        }

        const auto& cores = corerange_to_cores(all_cores, std::nullopt, row_major);
        for (uint32_t i = 0; i < num_cores_local; ++i) {
            const auto& core = cores[i];
            uint32_t output_idx_x = i / num_blocks_y;
            uint32_t output_idx_y = i % num_blocks_y;

            // in1 sender or receiver
            if (core == start_core) {
                KernelDescriptor::CoreRuntimeArgs mm_in1_sender_writer_args = {
                    (std::uint32_t)in1_buffer->address(),
                    (std::uint32_t)in1_tensor_start_tile_id_stride * output_idx_x,
                    (std::uint32_t)start_core_noc.x,
                    (std::uint32_t)start_core_noc.y,
                    (std::uint32_t)end_core_noc.x,
                    (std::uint32_t)end_core_noc.y,
                    (std::uint32_t)0,
                    (std::uint32_t)out_buffer->address(),
                    ((std::uint32_t)output_idx_x * per_core_N) + (output_idx_y * per_core_M * N),
                    (std::uint32_t)out_block_w,
                    (std::uint32_t)out_block_h / out_subblock_h,
                    (std::uint32_t)out_subblock_h,
                    (std::uint32_t)0,
                    (std::uint32_t)out_block_w / out_subblock_w,
                    (std::uint32_t)out_block_w / out_subblock_w,
                    (std::uint32_t)out_subblock_w,
                    (std::uint32_t)0,
                    (std::uint32_t)0,
                };
                if (bias_buffer != nullptr) {
                    mm_in1_sender_writer_args.push_back((std::uint32_t)bias_buffer->address());
                    mm_in1_sender_writer_args.push_back((std::uint32_t)per_core_N * output_idx_x);
                } else {
                    mm_in1_sender_writer_args.push_back(0);
                    mm_in1_sender_writer_args.push_back(0);
                }
                if (!output_is_sharded) {
                    mm_in1_sender_writer_args.push_back(out_num_blocks_x);
                }
                in1_sender_writer_desc.runtime_args.emplace_back(core, std::move(mm_in1_sender_writer_args));
            } else {
                KernelDescriptor::CoreRuntimeArgs mm_in1_receiver_writer_args = {
                    (std::uint32_t)top_left_core_physical.x,
                    (std::uint32_t)top_left_core_physical.y,
                    (std::uint32_t)out_buffer->address(),
                    ((std::uint32_t)output_idx_x * per_core_N) + (output_idx_y * per_core_M * N),
                };

                if (output_idx_y == num_blocks_y - 1) {
                    mm_in1_receiver_writer_args.push_back(out_block_h / out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_h);
                    mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_h);
                    mm_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_h_skip);
                    mm_in1_receiver_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(0);
                    mm_in1_receiver_writer_args.push_back(0);
                } else {
                    mm_in1_receiver_writer_args.push_back(out_block_h / out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(out_block_h / out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(0);
                    mm_in1_receiver_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(0);
                    mm_in1_receiver_writer_args.push_back(0);
                }
                if (!output_is_sharded) {
                    if (output_idx_y == num_blocks_y - 1) {
                        mm_in1_receiver_writer_args.push_back(last_out_num_blocks_h);
                        mm_in1_receiver_writer_args.push_back(out_num_blocks_x);
                    } else {
                        mm_in1_receiver_writer_args.push_back(out_num_blocks_y);
                        mm_in1_receiver_writer_args.push_back(out_num_blocks_x);
                    }
                }
                in1_receiver_writer_desc.runtime_args.emplace_back(core, std::move(mm_in1_receiver_writer_args));
            }

            // in0 sender (all cores)
            in0_sender_desc.runtime_args.emplace_back(
                core,
                KernelDescriptor::CoreRuntimeArgs{
                    (std::uint32_t)in0_buffer->address(),
                    (std::uint32_t)in0_tensor_start_tile_id_stride * output_idx_y,
                    (std::uint32_t)0,
                    (std::uint32_t)0,
                    (std::uint32_t)0,
                    (std::uint32_t)0,
                    (std::uint32_t)per_core_M,
                    (std::uint32_t)0,
                });
        }

        desc.kernels.push_back(std::move(in0_sender_desc));
        desc.kernels.push_back(std::move(in1_sender_writer_desc));
        if (has_in1_receivers) {
            desc.kernels.push_back(std::move(in1_receiver_writer_desc));
        }
        desc.kernels.push_back(std::move(compute_desc));
    }

    return desc;
}

}  // namespace ttnn::prim::matmul_new_detail
