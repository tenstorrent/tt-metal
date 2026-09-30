// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.mhc_pre_xing program: kernels/mhc_pre_xing_{reader,writer,compute}.cpp. Work = token tile-rows
// (coefficients only) or y column tiles (with streams), split contiguously over the compute grid.

#include "mhc_pre_xing_device_operation.hpp"

#include <bit>
#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::bringup::mhc_pre_ttnn {

namespace {

using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::DataType;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;

constexpr const char* KERNEL_DIR = "ttnn/ttnn/bringup/mhc_pre_ttnn/kernels/";
constexpr uint32_t TILE_HW = 32;
constexpr size_t NUM_CIRCULAR_BUFFERS = 64;

constexpr uint8_t CB_ROW = 0;       // coefficient source row tile       reader -> writer
constexpr uint8_t CB_SOA = 1;       // coefficient-major raw row          writer -> compute
constexpr uint8_t CB_SOA_OUT = 2;   // coefficient-major hc               compute -> writer
constexpr uint8_t CB_HC_STAGE = 3;  // row-major hc stage                 writer scratch
constexpr uint8_t CB_PB = 4;        // pre-block tile                     writer -> compute
constexpr uint8_t CB_X = 5;         // n stream tiles per y column        reader -> compute
constexpr uint8_t CB_Y = 16;        // y column tiles                     compute -> writer
constexpr uint8_t CB_ACC = 6;       // pack_stats: lane-wise sum x^2 tile compute -> writer
constexpr uint32_t X_DEPTH_COLS = 4;
constexpr uint32_t Y_DEPTH = 4;

constexpr uint32_t READER_KERNEL = 0, WRITER_KERNEL = 1, COMPUTE_KERNEL = 2;
constexpr uint32_t COMPUTE_RT_SCALARS = 2;  // compute RT: start, count, then the scalars + biases

int64_t token_tiles(const ttnn::Shape& padded) {
    int64_t lead = 1;
    for (size_t i = 0; i + 2 < padded.rank(); ++i) {
        lead *= padded[i];
    }
    return lead * (padded[-2] / TILE_HW);
}

uint32_t f32_bits(double v) { return std::bit_cast<uint32_t>(static_cast<float>(v)); }

std::vector<uint32_t> scalar_args(const MhcPreXingParams& p) {
    std::vector<uint32_t> a = {
        f32_bits(p.scale[0]),
        f32_bits(p.scale[1]),
        f32_bits(p.scale[2]),
        f32_bits(p.inv_nc),
        f32_bits(p.norm_eps),
        f32_bits(p.hc_eps),
        f32_bits(p.clamp_min),
        f32_bits(p.clamp_max),
        p.sinkhorn_iters};
    const uint32_t ng = p.n * (p.n + 2);
    for (uint32_t k = 0; k < ng; ++k) {
        a.push_back(k < p.base.size() ? f32_bits(p.base[k]) : 0u);
    }
    return a;
}

CBDescriptor make_cb(uint8_t index, uint32_t page_bytes, uint32_t num_pages, const CoreRangeSet& cores) {
    CBDescriptor cb;
    cb.total_size = num_pages * page_bytes;
    cb.core_ranges = cores;
    cb.format_descriptors.push_back(CBFormatDescriptor{
        .buffer_index = index,
        .data_format = tt::tt_metal::datatype_to_dataformat_converter(DataType::FLOAT32),
        .page_size = page_bytes});
    return cb;
}

KernelDescriptor make_kernel(
    const char* file,
    const CoreRangeSet& cores,
    std::vector<uint32_t> ct,
    KernelDescriptor::RuntimeArgs rt,
    KernelDescriptor::ConfigDescriptor config) {
    KernelDescriptor k;
    k.kernel_source = std::string(KERNEL_DIR) + file;
    k.source_type = KernelDescriptor::SourceType::FILE_PATH;
    k.core_ranges = cores;
    k.compile_time_args = std::move(ct);
    k.runtime_args = std::move(rt);
    k.config = std::move(config);
    return k;
}

struct Addresses {
    uint32_t row, x, hc, y;
};

Addresses addresses(const MhcPreXingParams& p, const MhcPreXingInputs& in, const std::vector<Tensor>& out) {
    const Tensor& row = in.input;
    const Tensor& x = in.streams.has_value() ? *in.streams : in.input;
    const Tensor& hc = (p.compute_coef || p.pack_stats) ? out.at(0) : row;
    const Tensor& y = (in.streams.has_value() && !p.pack_stats) ? out.back() : hc;
    return {row.buffer()->address(), x.buffer()->address(), hc.buffer()->address(), y.buffer()->address()};
}

}  // namespace

ProgramDescriptor create_xing_program_descriptor(
    const MhcPreXingParams& p, const MhcPreXingInputs& in, const std::vector<Tensor>& out) {
    const Tensor& row = in.input;
    const bool has_streams = in.streams.has_value();
    const Tensor& x = has_streams ? *in.streams : row;
    const Tensor& hc = (p.compute_coef || p.pack_stats) ? out.at(0) : row;
    const Tensor& y = (has_streams && !p.pack_stats) ? out.back() : hc;
    auto* device = row.device();
    TT_FATAL(!p.pack_stats || (has_streams && !p.compute_coef), "mhc_pre_xing: pack_stats needs streams");

    const uint32_t n = p.n;
    const int64_t rows = token_tiles(row.padded_shape());
    const uint32_t ct = has_streams ? static_cast<uint32_t>(x.logical_shape()[-1] / n / TILE_HW) : 1;
    const int64_t units = rows * ((has_streams && !p.pack_stats) ? ct : 1);
    // pack_stats: stream tiles per SFPU round (<= 7 = the free fp32 DEST tiles at full sync), a divisor of n * Ct.
    uint32_t chunk = 1;
    for (uint32_t c = 7; c >= 1; --c) {
        if ((n * ct) % c == 0) {
            chunk = c;
            break;
        }
    }
    const uint32_t page = TILE_HW * TILE_HW * 4;
    TT_FATAL(row.buffer()->page_size() == page, "mhc_pre_xing: the input row must be one fp32 tile wide");
    TT_FATAL(!has_streams || x.buffer()->page_size() == page, "mhc_pre_xing: streams must be fp32 tiles");

    const CoreCoord grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, group_1, group_2, units_1, units_2] =
        tt::tt_metal::split_work_to_cores(grid, static_cast<uint32_t>(units), true);

    ProgramDescriptor desc;
    desc.cbs.push_back(make_cb(CB_ROW, page, 2, all_cores));
    if (p.pack_stats) {
        desc.cbs.push_back(make_cb(CB_SOA, page, 1, all_cores));
        desc.cbs.push_back(make_cb(CB_SOA_OUT, page, 1, all_cores));
        desc.cbs.push_back(make_cb(CB_ACC, page, 1, all_cores));
        desc.cbs.push_back(make_cb(CB_X, page, 2 * chunk, all_cores));
    } else if (p.compute_coef) {
        desc.cbs.push_back(make_cb(CB_SOA, page, 2, all_cores));
        desc.cbs.push_back(make_cb(CB_SOA_OUT, page, 2, all_cores));
        desc.cbs.push_back(make_cb(CB_HC_STAGE, page, 1, all_cores));
    }
    if (has_streams && !p.pack_stats) {
        desc.cbs.push_back(make_cb(CB_PB, page, 2, all_cores));
        desc.cbs.push_back(make_cb(CB_X, page, n * X_DEPTH_COLS, all_cores));
        desc.cbs.push_back(make_cb(CB_Y, page, Y_DEPTH, all_cores));
    }

    std::vector<uint32_t> reader_ct = {
        n,
        ct,
        static_cast<uint32_t>(has_streams),
        CB_ROW,
        CB_X,
        page,
        page,
        static_cast<uint32_t>(p.pack_stats),
        chunk};
    for (const Tensor* t : {&row, &x}) {
        const auto a = tt::tt_metal::TensorAccessorArgs(*t->buffer()).get_compile_time_args();
        reader_ct.insert(reader_ct.end(), a.begin(), a.end());
    }
    std::vector<uint32_t> writer_ct = {
        n,
        ct,
        static_cast<uint32_t>(p.compute_coef),
        static_cast<uint32_t>(has_streams),
        static_cast<uint32_t>(p.compute_coef),
        CB_ROW,
        CB_SOA,
        CB_SOA_OUT,
        CB_HC_STAGE,
        CB_PB,
        CB_Y,
        page,
        page,
        n * (n + 2) + 1,
        n * (n + 2),
        static_cast<uint32_t>(p.pack_stats),
        CB_ACC};
    for (const Tensor* t : {&hc, &y}) {
        const auto a = tt::tt_metal::TensorAccessorArgs(*t->buffer()).get_compile_time_args();
        writer_ct.insert(writer_ct.end(), a.begin(), a.end());
    }
    std::vector<uint32_t> compute_ct = {
        n,
        ct,
        static_cast<uint32_t>(p.compute_coef),
        static_cast<uint32_t>(has_streams),
        CB_SOA,
        CB_SOA_OUT,
        CB_PB,
        CB_X,
        CB_Y,
        static_cast<uint32_t>(p.pack_stats),
        chunk,
        CB_ACC};

    const Addresses a = addresses(p, in, out);
    const auto scalars = scalar_args(p);
    KernelDescriptor::RuntimeArgs reader_rt, writer_rt, compute_rt;
    uint32_t start = 0;
    for (const auto& [group, per_core] : {std::pair{group_1, units_1}, std::pair{group_2, units_2}}) {
        for (const auto& core : tt::tt_metal::corerange_to_cores(group, std::nullopt, true)) {
            reader_rt.emplace_back(core, std::vector<uint32_t>{a.row, a.x, start, per_core});
            writer_rt.emplace_back(core, std::vector<uint32_t>{a.hc, a.y, start, per_core});
            std::vector<uint32_t> c = {start, per_core};
            c.insert(c.end(), scalars.begin(), scalars.end());
            compute_rt.emplace_back(core, std::move(c));
            start += per_core;
        }
    }
    TT_FATAL(start == units, "mhc_pre_xing: work split covers {} of {} units", start, units);

    ComputeConfigDescriptor cc;
    cc.math_fidelity = p.compute_config.math_fidelity;
    cc.fp32_dest_acc_en = true;
    cc.math_approx_mode = false;
    cc.dst_full_sync_en = true;  // n + 1 fp32 DEST tiles in the y-mix window
    cc.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint8_t cb : {CB_SOA, CB_PB, CB_X}) {
        cc.unpack_to_dest_mode[cb] = UnpackToDestMode::UnpackToDestFp32;
    }

    desc.kernels.push_back(make_kernel(
        "mhc_pre_xing_reader.cpp", all_cores, reader_ct, reader_rt, tt::tt_metal::ReaderConfigDescriptor{}));
    desc.kernels.push_back(make_kernel(
        "mhc_pre_xing_writer.cpp", all_cores, writer_ct, writer_rt, tt::tt_metal::WriterConfigDescriptor{}));
    desc.kernels.push_back(make_kernel("mhc_pre_xing_compute.cpp", all_cores, compute_ct, compute_rt, cc));
    return desc;
}

ProgramDescriptor MhcPreXingProgramFactory::create_descriptor(
    const MhcPreXingParams& operation_attributes, const MhcPreXingInputs& tensor_args, std::vector<Tensor>& outputs) {
    return create_xing_program_descriptor(operation_attributes, tensor_args, outputs);
}

void MhcPreXingProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const MhcPreXingParams& operation_attributes,
    const MhcPreXingInputs& tensor_args,
    std::vector<Tensor>& outputs,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const Addresses a = addresses(operation_attributes, tensor_args, outputs);
    const auto scalars = scalar_args(operation_attributes);
    for (auto& col : tt::tt_metal::GetRuntimeArgs(program, READER_KERNEL)) {
        for (auto& args : col) {
            if (args.size() >= 4) {
                args[0] = a.row;
                args[1] = a.x;
            }
        }
    }
    for (auto& col : tt::tt_metal::GetRuntimeArgs(program, WRITER_KERNEL)) {
        for (auto& args : col) {
            if (args.size() >= 4) {
                args[0] = a.hc;
                args[1] = a.y;
            }
        }
    }
    for (auto& col : tt::tt_metal::GetRuntimeArgs(program, COMPUTE_KERNEL)) {
        for (auto& args : col) {
            if (args.size() >= COMPUTE_RT_SCALARS + scalars.size()) {
                for (size_t k = 0; k < scalars.size(); ++k) {
                    args[COMPUTE_RT_SCALARS + k] = scalars[k];
                }
            }
        }
    }
}

}  // namespace ttnn::operations::bringup::mhc_pre_ttnn
