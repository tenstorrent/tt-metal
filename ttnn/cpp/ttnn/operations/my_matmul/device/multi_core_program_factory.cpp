#include "my_matmul_device_operation.hpp"
#include "tt-metalium/base_types.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/program_descriptors.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::my_matmul {

using namespace tt;
using namespace tt::tt_metal;

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

    ProgramDescriptor desc;

    // Circular buffers
    constexpr uint32_t in0_cb_index = CBIndex::c_0;
    const uint32_t in0_cb_tiles = 2 * per_core_Mt * Kt;
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
    const uint32_t in1_cb_tiles = 2 * Kt * per_core_Nt;
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
    constexpr uint32_t num_output_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * single_tile_size,
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

            reader_desc.emplace_runtime_args(
                core, {src0_buffer, src1_buffer, Mt, Kt, Nt, top, left, bot, right, block_size_A, block_size_B});
            writer_desc.emplace_runtime_args(core, {dst_buffer, Mt, Nt, top, left, bot, right});
            compute_desc.emplace_runtime_args(core, {top, left, bot, right, block_size_A, block_size_B});
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::my_matmul
