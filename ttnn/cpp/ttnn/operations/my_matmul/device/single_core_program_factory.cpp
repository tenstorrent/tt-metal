#include "my_matmul_device_operation.hpp"
#include "tt-metalium/base_types.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/program_descriptors.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::my_matmul {

using namespace tt;
using namespace tt::tt_metal;

ProgramDescriptor MyMatmulDeviceOperation::SingleCore::create_descriptor(
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

    CoreCoord core{0, 0};
    CoreRangeSet core_set{CoreRange{core, core}};

    ProgramDescriptor desc;

    // Circular buffers
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    constexpr uint32_t num_input_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * single_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    constexpr uint32_t src1_cb_index = CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * single_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src1_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    constexpr uint32_t out_cb_index = CBIndex::c_16;
    constexpr uint32_t num_output_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * single_tile_size,
        .core_ranges = core_set,
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
    reader_desc.core_ranges = core_set;
    reader_desc.compile_time_args = reader_ct_args;
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.emplace_runtime_args(core, {src0_buffer, src1_buffer, Mt, Kt, Nt});

    std::vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
    KernelDescriptor writer_desc;
    writer_desc.kernel_source = "ttnn/cpp/ttnn/operations/my_matmul/device/kernels/dataflow/writer.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_set;
    writer_desc.compile_time_args = writer_ct_args;
    writer_desc.config = WriterConfigDescriptor();
    writer_desc.emplace_runtime_args(core, {dst_buffer, Mt, Nt});

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = "ttnn/cpp/ttnn/operations/my_matmul/device/kernels/compute/mm.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_set;
    compute_desc.compile_time_args = {Mt, Kt, Nt};
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
    };

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::my_matmul
