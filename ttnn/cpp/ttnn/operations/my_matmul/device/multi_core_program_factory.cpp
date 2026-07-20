#include "my_matmul_device_operation.hpp"
#include "tt-metalium/base_types.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/program_descriptors.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/hal_types.hpp>
#include <algorithm>
#include <tuple>

namespace ttnn::operations::my_matmul {

using namespace tt;
using namespace tt::tt_metal;

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> calculate_sub_block_sizes(
    uint32_t m, uint32_t k, uint32_t n, uint32_t usable_l1_bytes, uint32_t tile_bytes) {
    // Returns 6 values:
    // (sub_block_m, sub_block_k, sub_block_n, num_blocks_m, num_blocks_k, num_blocks_n)
    // m = per_core_Mt, n = per_core_Nt, k = Kt  (all in tiles).

    // --- 1) Output subblock (h x w): the largest DST-fitting tile that divides the ---
    // per-core output block. DST holds up to 16 bf16 tiles (8 when double-buffered).
    // We are assuming bfloat16 here, but generally should parametrize per dtype.
    constexpr std::array<std::tuple<uint32_t, uint32_t>, 20> known_blocks = {{
        {4, 2}, {2, 4}, {8, 1}, {1, 8}, {7, 1}, {1, 7}, {3, 2}, {2, 3}, {6, 1}, {1, 6},
        {5, 1}, {1, 5}, {2, 2}, {4, 1}, {1, 4}, {3, 1}, {1, 3}, {2, 1}, {1, 2}, {1, 1},
    }};
    uint32_t sub_block_m = 1;
    uint32_t sub_block_n = 1;
    for (const auto& [h, w] : known_blocks) {
        if (m % h == 0 && n % w == 0) {
            sub_block_m = h;
            sub_block_n = w;
            break;
        }
    }

    // --- 2) K-chunk width (sub_block_k): the largest divisor of k whose input CBs fit L1. ---
    // L1 footprint for the K-outer / within-core-reuse design:
    //   fixed  : the per-core output accumulator stays resident across the whole K loop
    //              acc = m * n tiles
    //   scales : the two input slices, double-buffered, per unit of sub_block_k
    //              cb_in0 = 2 * m           * sub_block_k tiles
    //              cb_in1 = 2 * sub_block_k * n           tiles
    //            => 2 * (m + n) * sub_block_k tiles
    const uint64_t acc_bytes = static_cast<uint64_t>(m) * n * tile_bytes;
    const uint64_t bytes_per_kt = static_cast<uint64_t>(2) * (m + n) * tile_bytes;

    uint32_t sub_block_k = 1;  // safe fallback (also the value if nothing bigger fits)
    if (usable_l1_bytes > acc_bytes) {
        const uint64_t input_budget = usable_l1_bytes - acc_bytes;
        uint32_t max_kt = static_cast<uint32_t>(input_budget / bytes_per_kt);
        max_kt = std::min(max_kt, k);
        // Largest divisor of k that is <= max_kt (k % 1 == 0 guarantees termination at 1).
        for (uint32_t c = max_kt; c >= 1; c--) {
            if (k % c == 0) {
                sub_block_k = c;
                break;
            }
        }
    }
    const uint32_t num_blocks_k = k / sub_block_k;

    return std::make_tuple(sub_block_m, sub_block_k, sub_block_n, m / sub_block_m, num_blocks_k, n / sub_block_n);
}

ProgramDescriptor MyMatmulDeviceOperation::MultiCore::create_descriptor(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    auto& out = tensor_return_value;
    auto* src0_buffer = a.buffer();
    auto* src1_buffer = b.buffer();
    auto* dst_buffer = out.buffer();

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    const auto& as = a.padded_shape();
    const auto& bs = b.padded_shape();

    uint32_t Mt = as[as.rank() - 2] / tt::constants::TILE_HEIGHT;
    uint32_t Kt = as[as.rank() - 1] / tt::constants::TILE_WIDTH;
    uint32_t Nt = bs[bs.rank() - 1] / tt::constants::TILE_WIDTH;

    CoreCoord grid = a.device()->compute_with_storage_grid_size();
    std::cout << "grid: " << grid.str() << std::endl;

    uint32_t per_core_Mt = div_up(Mt, grid.y);
    uint32_t per_core_Nt = div_up(Nt, grid.x);
    std::cout << "per_core_Mt: " << per_core_Mt << std::endl;
    std::cout << "per_core_Nt: " << per_core_Nt << std::endl;

    uint32_t y_cores = grid.y < Mt ? grid.y : Mt;
    uint32_t x_cores = grid.x < Nt ? grid.x : Nt;
    CoreCoord top_left = CoreCoord{0, 0};
    CoreCoord bot_right = CoreCoord{x_cores - 1, y_cores - 1};

    CoreRangeSet used_cores{CoreRange{top_left, bot_right}};
    std::cout << "used_cores: " << top_left.str() << ", " << bot_right.str() << std::endl;

    // Usable L1 per core = total L1 minus the region the allocator reserves (firmware, kernel args, etc.).
    auto* device = a.device();
    const uint32_t usable_l1 =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    auto [sub_block_m, sub_block_k, sub_block_n, num_blocks_m, num_blocks_k, num_blocks_n] =
        calculate_sub_block_sizes(per_core_Mt, Kt, per_core_Nt, usable_l1, single_tile_size);
    std::cout << "sub_block (m,k,n): (" << sub_block_m << ", " << sub_block_k << ", " << sub_block_n << ")  "
              << "num_blocks (m,k,n): (" << num_blocks_m << ", " << num_blocks_k << ", " << num_blocks_n << ")"
              << std::endl;

    ProgramDescriptor desc;

    // Circular buffers
    constexpr uint32_t in0_cb_index = CBIndex::c_0;
    const uint32_t in0_cb_tiles = 2 * num_blocks_m * sub_block_m * sub_block_k;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in0_cb_tiles * single_tile_size,
        .core_ranges = used_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in0_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    constexpr uint32_t in1_cb_index = CBIndex::c_1;
    const uint32_t in1_cb_tiles = 2 * sub_block_k * num_blocks_n * sub_block_n;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in1_cb_tiles * single_tile_size,
        .core_ranges = used_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in1_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    constexpr uint32_t out_cb_index = CBIndex::c_16;
    const uint32_t out_cb_tiles =
        num_blocks_m * sub_block_m * num_blocks_n * sub_block_n;  // per_core_Mt * per_core_Nt (depth 1)
    desc.cbs.push_back(CBDescriptor{
        .total_size = out_cb_tiles * single_tile_size,
        .core_ranges = used_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = out_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(*src1_buffer).append_to(reader_ct_args);
    KernelDescriptor reader_desc;
    reader_desc.kernel_source = "ttnn/cpp/ttnn/operations/my_matmul/device/kernels/dataflow/reader.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = used_cores;
    reader_desc.compile_time_args = reader_ct_args;
    reader_desc.config = ReaderConfigDescriptor{};
    // reader_desc.emplace_runtime_args(core, {src0_buffer, src1_buffer, Mt, Kt, Nt});

    std::vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
    KernelDescriptor writer_desc;
    writer_desc.kernel_source = "ttnn/cpp/ttnn/operations/my_matmul/device/kernels/dataflow/writer.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = used_cores;
    writer_desc.compile_time_args = writer_ct_args;
    writer_desc.config = WriterConfigDescriptor();
    // writer_desc.emplace_runtime_args(core, {dst_buffer, Mt, Nt});

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = "ttnn/cpp/ttnn/operations/my_matmul/device/kernels/compute/mm.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = used_cores;
    compute_desc.compile_time_args = {Mt, Kt, Nt};
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
    };

    for (uint32_t core_y = 0; core_y < y_cores; core_y++) {
        for (uint32_t core_x = 0; core_x < x_cores; core_x++) {
            uint32_t top = core_y * per_core_Mt;
            uint32_t bot = std::min(((core_y + 1) * per_core_Mt), Mt);
            uint32_t left = core_x * per_core_Nt;
            uint32_t right = std::min((core_x + 1) * per_core_Nt, Nt);

            std::cout << "  (core_y, core_x): (" << core_y << ", " << core_x << ")" << std::endl;
            std::cout << "    top_left: (" << top << ", " << left << ")" << std::endl;
            std::cout << "    bot_right: (" << bot << ", " << right << ")" << std::endl;

            CoreCoord core{core_x, core_y};

            uint32_t block_size_A = (bot - top) * Kt;
            uint32_t block_size_B = Kt * (right - left);

            // sub_block_* / num_blocks_* are computed once above (identical for every core) and reused here.
            uint32_t dst_size = sub_block_m * sub_block_n;

            uint32_t k_block_A = num_blocks_m * sub_block_m * sub_block_k;  // Mt * sub_block_k
            uint32_t k_block_B = sub_block_k * num_blocks_n * sub_block_n;  // sub_block_k * Nt

            reader_desc.emplace_runtime_args(
                core, {src0_buffer, src1_buffer, Mt, Kt, Nt, top, left, bot, right, block_size_A, block_size_B});

            // compute_desc.emplace_runtime_args(core, {top, left, bot, right, k_block_A, k_block_B});
            compute_desc.emplace_runtime_args(
                core,
                {num_blocks_m,
                 num_blocks_k,
                 num_blocks_n,
                 sub_block_m,
                 sub_block_k,
                 sub_block_n,
                 dst_size,
                 k_block_A,
                 k_block_B});

            writer_desc.emplace_runtime_args(core, {dst_buffer, Mt, Nt, top, left, bot, right});
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::my_matmul
