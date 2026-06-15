// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/sliding_window/halo/device/untilize_with_halo_program_factory.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace m2 = tt::tt_metal::experimental;

namespace {

// TODO: Look into increasing this to tradeoff some L1 for performance (#19980)
constexpr int UNTILIZE_BLOCK_SIZE = 32;
constexpr bool ENABLE_UNTILIZE_DOUBLE_BUFFERING = true;
constexpr uint32_t EMPTY_PADDING_CONFIG_BUFFER_SIZE = 4;

constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/dataflow/halo_gather.cpp";
constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/compute/pack_untilize.cpp";

// DFB unique-ids (kernels reference these via dfb::<name>).
constexpr const char* DFB_SRC = "halo_src";
constexpr const char* DFB_OUT = "halo_out";
constexpr const char* DFB_PAD0 = "halo_pad0";
constexpr const char* DFB_PAD1 = "halo_pad1";
constexpr const char* DFB_PAD_CFG0 = "halo_padding_config0";
constexpr const char* DFB_PAD_CFG1 = "halo_padding_config1";
constexpr const char* DFB_GATHER_CFG0 = "halo_gather_config0";
constexpr const char* DFB_GATHER_CFG1 = "halo_gather_config1";
constexpr const char* DFB_UNTILIZE_OUT0 = "halo_untilize_out0";
constexpr const char* DFB_UNTILIZE_OUT1 = "halo_untilize_out1";

// Tensor-parameter names (kernels access via ta::<name>; borrowed DFBs reference them by name).
constexpr const char* TP_INPUT = "input";
constexpr const char* TP_OUTPUT = "output";
constexpr const char* TP_PAD_CFG0 = "padding_config0";
constexpr const char* TP_PAD_CFG1 = "padding_config1";
constexpr const char* TP_GATHER_CFG0 = "gather_config0";
constexpr const char* TP_GATHER_CFG1 = "gather_config1";

// Everything the spec / run-args / owned-tensor builders share, computed from the request. All inputs
// (attrs.config, shard specs, layouts) are part of the cache key, so this is a pure function of the key.
struct HaloGeometry {
    // The four host-built config tensors (uploaded to device by create_owned_tensors).
    Tensor pad_config0;
    Tensor pad_config1;
    Tensor gather_config0;
    Tensor gather_config1;
    std::vector<uint16_t> number_of_blocks_per_core;

    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores;
    bool is_rm_orientation = false;

    bool skip_untilize = false;
    bool config_tensors_in_dram = false;
    bool is_block_sharded = false;
    bool is_height_sharded = false;
    bool is_width_sharded = false;

    tt::DataFormat in_df = tt::DataFormat::Invalid;
    tt::DataFormat out_df = tt::DataFormat::Invalid;
    uint32_t in_page_size = 0;
    uint32_t input_npages = 0;
    uint32_t out_cb_pagesize = 0;
    uint32_t out_cb_npages = 0;
    uint32_t pad_cb_pagesize = 0;
    uint32_t out_tile_size = 0;
    uint32_t ntiles_per_block = 0;
    uint32_t clamped_block_size_height = 0;
    uint32_t untilize_out_cb_num_pages = 0;
    uint32_t aligned_stick_nbytes = 0;
    uint32_t pad_val = 0;
};

HaloGeometry compute_geometry(const HaloParams& attrs, const Tensor& input, const Tensor& output) {
    using namespace ttnn::operations;
    HaloGeometry g;
    auto* device = input.device();

    g.config_tensors_in_dram = attrs.config_tensors_in_dram;
    g.is_block_sharded = input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    g.is_height_sharded = output.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;
    g.is_width_sharded = input.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;
    g.pad_val = attrs.pad_val;
    const bool is_in_tiled = input.layout() == Layout::TILE;
    const bool remote_read = attrs.remote_read;
    const bool transpose_mcast = attrs.transpose_mcast;

    // --- host metadata + the four config tensors (uploaded to device) ---
    auto pad_metadata = sliding_window::generate_pad_metadata(attrs.config);
    auto shard_boundaries = sliding_window::generate_shard_boundaries(attrs.config);
    const uint32_t input_shard_height = input.memory_config().shard_spec()->shape[0];
    auto tensor_metadata = sliding_window::generate_tensor_metadata(pad_metadata, attrs.config, input_shard_height);
    const uint32_t num_cores_x = input.memory_config().shard_spec()->grid.bounding_box().grid_size().x;
    auto kernel_config = sliding_window::generate_halo_kernel_config_tensors(
        tensor_metadata,
        shard_boundaries,
        g.is_block_sharded,
        transpose_mcast,
        remote_read,
        device,
        num_cores_x,
        is_in_tiled,
        UNTILIZE_BLOCK_SIZE);

    // Build the four config tensors on HOST only; create_owned_tensors performs the single device upload.
    // create_program_spec derives their device specs via config_device_spec (no upload), and
    // create_invariant_run_args needs only number_of_blocks_per_core below — so nothing here materializes.
    auto on_host = [&](const auto& cfg) {
        return sliding_window::construct_on_host_config_tensor(cfg, attrs.parallel_config, g.config_tensors_in_dram);
    };
    g.pad_config0 = on_host(kernel_config.pad_config0);
    g.pad_config1 = on_host(kernel_config.pad_config1);
    g.gather_config0 = on_host(kernel_config.gather_config0);
    g.gather_config1 = on_host(kernel_config.gather_config1);
    g.number_of_blocks_per_core =
        sliding_window::remap_nhw_scalar_argument_across_full_grid(kernel_config.number_of_blocks_per_core, attrs.parallel_config);

    // --- CB / work geometry (mirrors build_halo_program) ---
    const uint32_t ncores_nhw = attrs.config.num_cores_nhw;
    g.in_df = datatype_to_dataformat_converter(input.dtype());
    g.out_df = datatype_to_dataformat_converter(output.dtype());
    const uint32_t out_nbytes = datum_size(g.out_df);

    g.all_cores = output.shard_spec().value().grid;
    const ShardOrientation shard_orientation = output.shard_spec().value().orientation;
    g.is_rm_orientation = shard_orientation == ShardOrientation::ROW_MAJOR;
    g.cores = corerange_to_cores(g.all_cores, std::nullopt, g.is_rm_orientation);

    const auto& input_shape = input.padded_shape();
    const auto input_shard_shape = input.shard_spec().value().shape;
    const auto output_shard_shape = output.shard_spec().value().shape;

    const uint32_t input_nhw_height = input_shape[0] * input_shape[1] * input_shape[2];
    const uint32_t remapped_input_shard_shape_for_output_grid = tt::div_up(input_nhw_height, ncores_nhw);
    g.ntiles_per_block = tt::div_up(input_shard_shape[1], TILE_WIDTH);
    const uint32_t input_nblocks_per_core = tt::div_up(remapped_input_shard_shape_for_output_grid, TILE_HEIGHT);
    g.input_npages = g.ntiles_per_block * input_nblocks_per_core;
    g.in_page_size = tt::tile_size(g.in_df);

    const uint32_t stick_nbytes = output_shard_shape[1] * out_nbytes;
    g.aligned_stick_nbytes = stick_nbytes;
    if (stick_nbytes % input.buffer()->alignment() != 0) {
        g.aligned_stick_nbytes = tt::round_up(stick_nbytes, input.buffer()->alignment());
    }
    g.out_tile_size = tt::tile_size(g.out_df);

    g.skip_untilize = input.layout() == Layout::ROW_MAJOR;
    if (g.skip_untilize) {
        g.in_page_size = g.aligned_stick_nbytes;
        g.input_npages = input_shard_shape[0];
    }

    g.out_cb_pagesize = g.aligned_stick_nbytes;
    g.out_cb_npages = attrs.max_out_nsticks_per_core;
    g.pad_cb_pagesize = g.aligned_stick_nbytes;

    g.clamped_block_size_height =
        std::min(static_cast<uint32_t>(UNTILIZE_BLOCK_SIZE), input_nblocks_per_core * TILE_HEIGHT);
    if (!g.skip_untilize) {
        const uint32_t output_ntiles = (g.clamped_block_size_height / TILE_HEIGHT) * g.ntiles_per_block;
        g.untilize_out_cb_num_pages = ENABLE_UNTILIZE_DOUBLE_BUFFERING ? 2 * output_ntiles : output_ntiles;
    }
    return g;
}

// A scratch (own-L1) DFB.
m2::DataflowBufferSpec scratch_dfb(const char* id, tt::DataFormat df, uint32_t entry_size, uint32_t num_entries) {
    return m2::DataflowBufferSpec{
        .unique_id = m2::DFBSpecName{id},
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = df,
    };
}

// A DFB borrowed from a tensor parameter (its backing buffer is the bound tensor).
m2::DataflowBufferSpec borrowed_dfb(
    const char* id, const char* tensor_param, tt::DataFormat df, uint32_t entry_size, uint32_t num_entries) {
    auto dfb = scratch_dfb(id, df, entry_size, num_entries);
    dfb.borrowed_from = m2::TensorParamName{tensor_param};
    return dfb;
}

// Device memory config a sliding-window config tensor WOULD get on upload — mirrors the memcfg in
// sliding_window.cpp::move_config_tensor_to_device (which create_owned_tensors calls to do the actual upload).
// This lets create_program_spec derive the device TensorSpec + page size WITHOUT materializing the tensor, so
// the config tensors are uploaded exactly once (in create_owned_tensors) instead of once per factory method.
MemoryConfig config_device_memcfg(
    const Tensor& host, const ttnn::operations::sliding_window::ParallelConfig& p_config, bool is_block_sharded) {
    std::array<uint32_t, 2> shard_shape{1, static_cast<uint32_t>(host.logical_shape()[-1])};
    auto orientation = is_block_sharded ? (p_config.shard_orientation == ShardOrientation::COL_MAJOR
                                               ? ShardOrientation::ROW_MAJOR
                                               : ShardOrientation::COL_MAJOR)
                                        : ShardOrientation::ROW_MAJOR;
    ShardSpec shard_spec(p_config.grid, shard_shape, orientation);
    return MemoryConfig{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1_SMALL, shard_spec};
}

// The device TensorSpec a host config tensor would have once uploaded (DRAM-interleaved or L1_SMALL sharded).
TensorSpec config_device_spec(
    const Tensor& host,
    const ttnn::operations::sliding_window::ParallelConfig& p_config,
    bool is_block_sharded,
    bool in_dram) {
    MemoryConfig memcfg = in_dram ? MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}
                                  : config_device_memcfg(host, p_config, is_block_sharded);
    // with_memory_config is exactly the transform to_device applies to the spec (data moves, layout
    // preserved, memory_config swapped), so the declared spec matches the uploaded tensor's spec exactly.
    return host.tensor_spec().with_memory_config(memcfg);
}

}  // namespace

// create_owned_tensors — the four host-built sliding-window config tensors. The framework keeps them alive
// for the cached program's lifetime and binds each (via its TensorParameter) to the reader's config CB /
// accessor. They are a pure function of the cache key (the sliding-window config), so they're invariant.
m2::Table<m2::TensorParamName, Tensor> UntilizeWithHaloProgramFactory::create_owned_tensors(
    const HaloParams& attrs, const Tensor& input, Tensor& output) {
    const auto g = compute_geometry(attrs, input, output);
    // The single device upload of the host-built config tensors (the only materialization).
    auto upload = [&](const Tensor& host) {
        return ttnn::operations::sliding_window::move_config_tensor_to_device(
            host, attrs.parallel_config, g.is_block_sharded, input.device(), g.config_tensors_in_dram);
    };
    m2::Table<m2::TensorParamName, Tensor> owned;
    owned.emplace(m2::TensorParamName{TP_PAD_CFG0}, upload(g.pad_config0));
    owned.emplace(m2::TensorParamName{TP_PAD_CFG1}, upload(g.pad_config1));
    owned.emplace(m2::TensorParamName{TP_GATHER_CFG0}, upload(g.gather_config0));
    owned.emplace(m2::TensorParamName{TP_GATHER_CFG1}, upload(g.gather_config1));
    return owned;
}

// create_program_spec — the immutable blueprint: the DFBs (src/out borrowed from the sharded I/O tensors,
// the four config CBs borrowed from the owned config tensors, scratch pad/untilize CBs) and the kernels
// (the split halo_gather reader on RISCV0/1, plus the pack_untilize compute when the input is tiled).
m2::ProgramSpec UntilizeWithHaloProgramFactory::create_program_spec(
    const HaloParams& attrs, const Tensor& input, Tensor& output) {
    const auto g = compute_geometry(attrs, input, output);
    const tt::DataFormat cfg_df = tt::DataFormat::RawUInt16;

    // Device specs for the op-owned config tensors, derived from the host tensors WITHOUT uploading
    // (create_owned_tensors performs the single upload). compute_page_size_bytes() equals the device buffer's
    // page size, so the borrowed-DFB sizing and the empty-padding sentinel below are unchanged.
    const auto spec_of = [&](const Tensor& host) {
        return config_device_spec(host, attrs.parallel_config, g.is_block_sharded, g.config_tensors_in_dram);
    };
    const auto pad_cfg0_spec = spec_of(g.pad_config0);
    const auto pad_cfg1_spec = spec_of(g.pad_config1);
    const auto gather_cfg0_spec = spec_of(g.gather_config0);
    const auto gather_cfg1_spec = spec_of(g.gather_config1);

    const bool enable_padding = g.config_tensors_in_dram ||
                                pad_cfg0_spec.compute_page_size_bytes() != EMPTY_PADDING_CONFIG_BUFFER_SIZE ||
                                pad_cfg1_spec.compute_page_size_bytes() != EMPTY_PADDING_CONFIG_BUFFER_SIZE;

    m2::ProgramSpec spec;
    spec.name = "untilize_with_halo";

    // Tensor parameters: I/O + the four config tensors. The config tensors are op-owned (create_owned_tensors)
    // and bound once at program-build time; mark them enqueue-invariant so the cache-hit path may omit them
    // from the per-enqueue args and retain the bound tensor. I/O are re-specified on every enqueue.
    const m2::TensorParameterAdvancedOptions invariant_tp{.enqueue_invariant = true};
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{TP_INPUT}, .spec = input.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{TP_OUTPUT}, .spec = output.tensor_spec()},
        m2::TensorParameter{
            .unique_id = m2::TensorParamName{TP_PAD_CFG0}, .spec = pad_cfg0_spec, .advanced_options = invariant_tp},
        m2::TensorParameter{
            .unique_id = m2::TensorParamName{TP_PAD_CFG1}, .spec = pad_cfg1_spec, .advanced_options = invariant_tp},
        m2::TensorParameter{
            .unique_id = m2::TensorParamName{TP_GATHER_CFG0},
            .spec = gather_cfg0_spec,
            .advanced_options = invariant_tp},
        m2::TensorParameter{
            .unique_id = m2::TensorParamName{TP_GATHER_CFG1},
            .spec = gather_cfg1_spec,
            .advanced_options = invariant_tp},
    };

    // DFBs. src/out are borrowed from the sharded I/O tensors; the four config CBs are borrowed from the
    // owned config tensors when they live in L1 (DRAM config is read via ta:: instead, no CB backing).
    std::vector<m2::DataflowBufferSpec> dfbs;
    dfbs.push_back(borrowed_dfb(DFB_SRC, TP_INPUT, g.in_df, g.in_page_size, g.input_npages));
    dfbs.push_back(borrowed_dfb(DFB_OUT, TP_OUTPUT, g.out_df, g.out_cb_pagesize, g.out_cb_npages));
    dfbs.push_back(scratch_dfb(DFB_PAD0, g.out_df, g.pad_cb_pagesize, 1));
    dfbs.push_back(scratch_dfb(DFB_PAD1, g.out_df, g.pad_cb_pagesize, 1));
    if (g.config_tensors_in_dram) {
        dfbs.push_back(scratch_dfb(DFB_PAD_CFG0, cfg_df, pad_cfg0_spec.compute_page_size_bytes(), 1));
        dfbs.push_back(scratch_dfb(DFB_PAD_CFG1, cfg_df, pad_cfg1_spec.compute_page_size_bytes(), 1));
        dfbs.push_back(scratch_dfb(DFB_GATHER_CFG0, cfg_df, gather_cfg0_spec.compute_page_size_bytes(), 1));
        dfbs.push_back(scratch_dfb(DFB_GATHER_CFG1, cfg_df, gather_cfg1_spec.compute_page_size_bytes(), 1));
    } else {
        dfbs.push_back(borrowed_dfb(DFB_PAD_CFG0, TP_PAD_CFG0, cfg_df, pad_cfg0_spec.compute_page_size_bytes(), 1));
        dfbs.push_back(borrowed_dfb(DFB_PAD_CFG1, TP_PAD_CFG1, cfg_df, pad_cfg1_spec.compute_page_size_bytes(), 1));
        dfbs.push_back(
            borrowed_dfb(DFB_GATHER_CFG0, TP_GATHER_CFG0, cfg_df, gather_cfg0_spec.compute_page_size_bytes(), 1));
        dfbs.push_back(
            borrowed_dfb(DFB_GATHER_CFG1, TP_GATHER_CFG1, cfg_df, gather_cfg1_spec.compute_page_size_bytes(), 1));
    }
    if (!g.skip_untilize) {
        dfbs.push_back(scratch_dfb(DFB_UNTILIZE_OUT0, g.out_df, g.out_tile_size, g.untilize_out_cb_num_pages));
        dfbs.push_back(scratch_dfb(DFB_UNTILIZE_OUT1, g.out_df, g.out_tile_size, g.untilize_out_cb_num_pages));
    }
    spec.dataflow_buffers = std::move(dfbs);

    // The reader writes into the untilize-out CB (tiled) or straight to src (row-major).
    const char* input_to_writer0 = g.skip_untilize ? DFB_SRC : DFB_UNTILIZE_OUT0;
    const char* input_to_writer1 = g.skip_untilize ? DFB_SRC : DFB_UNTILIZE_OUT1;

    std::vector<m2::KernelSpec> kernels;

    // Compute kernel (tiled input only): untilizes src into the two untilize-out CBs.
    if (!g.skip_untilize) {
        auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
            ttnn::get_compute_kernel_config_args(input.device()->arch(), attrs.compute_kernel_config);
        m2::KernelSpec::CompilerOptions copts;
        copts.defines.emplace("TILES_PER_ROW", std::to_string(g.ntiles_per_block));
        copts.defines.emplace("BLOCK_SIZE", std::to_string(g.clamped_block_size_height / TILE_HEIGHT));
        kernels.push_back(m2::KernelSpec{
            .unique_id = m2::KernelSpecName{"compute"},
            .source = std::filesystem::path{COMPUTE_KERNEL_PATH},
            .compiler_options = std::move(copts),
            .dfb_bindings =
                {m2::ConsumerOf(m2::DFBSpecName{DFB_SRC}, "src"),
                 m2::ProducerOf(m2::DFBSpecName{DFB_UNTILIZE_OUT0}, "untilize_out0"),
                 m2::ProducerOf(m2::DFBSpecName{DFB_UNTILIZE_OUT1}, "untilize_out1")},
            .runtime_arg_schema = {.runtime_arg_names = {"total_blocks"}},
            .hw_config =
                m2::ComputeHardwareConfig{
                    .math_fidelity = math_fidelity,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .dst_full_sync_en = dst_full_sync_en,
                    .math_approx_mode = math_approx_mode,
                },
            .advanced_options = m2::KernelAdvancedOptions{.enqueue_invariant_runtime_args = {"total_blocks"}},
        });
    }

    // The two split-reader kernels (RISCV0 / RISCV1). Compile-time selectors become defines; the per-core
    // config_read_index (DRAM config only) is an enqueue-invariant named arg.
    auto reader_defines = [&](bool is_reader1) {
        m2::KernelSpec::CompilerOptions copts;
        auto def = [&](const char* k, uint32_t v) { copts.defines.emplace(k, std::to_string(v)); };
        def("PAD_VAL", g.pad_val);
        def("IN_NSTICKS", g.input_npages);
        def("ALIGNED_STICK_NBYTES", g.aligned_stick_nbytes);
        def("IS_BLOCK_SHARDED", g.is_block_sharded ? 1 : 0);
        def("REMOTE_READ", attrs.remote_read ? 1 : 0);
        def("IS_COL_MAJOR", attrs.transpose_mcast ? 1 : 0);
        def("IS_WIDTH_SHARDED", g.is_width_sharded ? 1 : 0);
        def("SKIP_UNTILIZE", g.skip_untilize ? 1 : 0);
        def("BLOCK_SIZE_HEIGHT", g.clamped_block_size_height);
        def("BLOCK_SIZE_WIDTH_TILES", g.ntiles_per_block);
        def("BLOCK_START_OFFSET", is_reader1 ? 1u : 0u);
        def("BLOCK_STRIDE", 2u);
        def("ENABLE_PADDING", enable_padding ? 1 : 0);
        if (g.config_tensors_in_dram) {
            copts.defines.emplace("CONFIG_TENSOR_IN_DRAM", "1");
            def("PADDING_CONFIG_PAGE_SIZE",
                (is_reader1 ? pad_cfg1_spec : pad_cfg0_spec).compute_page_size_bytes());
            def("GATHER_CONFIG_PAGE_SIZE",
                (is_reader1 ? gather_cfg1_spec : gather_cfg0_spec).compute_page_size_bytes());
        }
        return copts;
    };
    auto reader_dfb_bindings = [&](bool is_reader1) {
        const char* in_dfb = is_reader1 ? input_to_writer1 : input_to_writer0;
        const char* pad_dfb = is_reader1 ? DFB_PAD1 : DFB_PAD0;
        const char* gather_dfb = is_reader1 ? DFB_GATHER_CFG1 : DFB_GATHER_CFG0;
        const char* padcfg_dfb = is_reader1 ? DFB_PAD_CFG1 : DFB_PAD_CFG0;
        // The reader streams the untilized (or row-major) input from `in`. src is pushed by reader0 and out
        // is written by both readers — assign reader0 the producer role and reader1 the consumer role so
        // each borrowed DFB has one of each (the role only shapes the dep graph; both readers still write).
        // The scratch pad CB and the config CBs are both filled and read by this same reader.
        std::vector<m2::KernelSpec::DFBBinding> b{
            m2::ConsumerOf(m2::DFBSpecName{in_dfb}, "in"),
            is_reader1 ? m2::ConsumerOf(m2::DFBSpecName{DFB_OUT}, "out")
                       : m2::ProducerOf(m2::DFBSpecName{DFB_OUT}, "out"),
            m2::ProducerOf(m2::DFBSpecName{pad_dfb}, "pad"),
            m2::ConsumerOf(m2::DFBSpecName{pad_dfb}, "pad"),
            m2::ProducerOf(m2::DFBSpecName{gather_dfb}, "gather_config"),
            m2::ConsumerOf(m2::DFBSpecName{gather_dfb}, "gather_config")};
        // src is pushed only by the block_start_offset==0 reader, which holds the sole producer binding.
        // The other reader never binds src separately: in tiled mode it doesn't touch src; in row-major
        // skip-untilize src == in, so its `in` consumer binding already covers DFB_SRC (the kernel aliases
        // src to dfb::in there). A second src binding on that reader would be a duplicate consumer.
        if (!is_reader1) {
            b.push_back(m2::ProducerOf(m2::DFBSpecName{DFB_SRC}, "src"));
        }
        if (enable_padding) {
            b.push_back(m2::ProducerOf(m2::DFBSpecName{padcfg_dfb}, "padding_config"));
            b.push_back(m2::ConsumerOf(m2::DFBSpecName{padcfg_dfb}, "padding_config"));
        }
        return b;
    };
    auto reader_tensor_bindings = [&](bool is_reader1) {
        std::vector<m2::TensorBinding> tb;
        if (g.config_tensors_in_dram) {
            tb.push_back(m2::TensorBinding{
                .tensor_parameter_name = m2::TensorParamName{is_reader1 ? TP_PAD_CFG1 : TP_PAD_CFG0},
                .accessor_name = "padding_config_dram"});
            tb.push_back(m2::TensorBinding{
                .tensor_parameter_name = m2::TensorParamName{is_reader1 ? TP_GATHER_CFG1 : TP_GATHER_CFG0},
                .accessor_name = "gather_config_dram"});
        }
        return tb;
    };

    kernels.push_back(m2::KernelSpec{
        .unique_id = m2::KernelSpecName{"reader0"},
        .source = std::filesystem::path{READER_KERNEL_PATH},
        .compiler_options = reader_defines(false),
        .dfb_bindings = reader_dfb_bindings(false),
        .tensor_bindings = reader_tensor_bindings(false),
        .runtime_arg_schema = {.runtime_arg_names = {"config_read_index"}},
        .hw_config = m2::DataMovementHardwareConfig{
            .gen1_config = m2::DataMovementHardwareConfig::Gen1Config{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}},
        .advanced_options = m2::KernelAdvancedOptions{.enqueue_invariant_runtime_args = {"config_read_index"}},
    });
    kernels.push_back(m2::KernelSpec{
        .unique_id = m2::KernelSpecName{"reader1"},
        .source = std::filesystem::path{READER_KERNEL_PATH},
        .compiler_options = reader_defines(true),
        .dfb_bindings = reader_dfb_bindings(true),
        .tensor_bindings = reader_tensor_bindings(true),
        .runtime_arg_schema = {.runtime_arg_names = {"config_read_index"}},
        .hw_config = m2::DataMovementHardwareConfig{
            .gen1_config = m2::DataMovementHardwareConfig::Gen1Config{
                .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}},
        .advanced_options = m2::KernelAdvancedOptions{.enqueue_invariant_runtime_args = {"config_read_index"}},
    });

    spec.kernels = std::move(kernels);

    // Tiled splits across reader0/reader1 (each consuming its own untilize_out half) plus the compute kernel.
    // Row-major skip-untilize uses the same two readers, but both consume the single borrowed src CB on the
    // same core — a same-core broadcast the local-DFB model can't yet express, so the skip path is blocked
    // pending metal support for multiple same-core consumers on a local DFB.
    std::vector<m2::KernelSpecName> work_kernels = {m2::KernelSpecName{"reader0"}, m2::KernelSpecName{"reader1"}};
    if (!g.skip_untilize) {
        work_kernels.insert(work_kernels.begin(), m2::KernelSpecName{"compute"});
    }
    spec.work_units = {m2::WorkUnitSpec{.name = "halo_work", .kernels = work_kernels, .target_nodes = g.all_cores}};
    return spec;
}

// create_invariant_run_args — the per-core work scalars: the compute kernel's per-core block count, and
// (DRAM config only) the per-core config_read_index used to index the shared config tensors.
m2::ProgramRunArgs UntilizeWithHaloProgramFactory::create_invariant_run_args(
    const HaloParams& attrs, const Tensor& input, Tensor& output) {
    const auto g = compute_geometry(attrs, input, output);
    m2::ProgramRunArgs args;

    if (!g.skip_untilize) {
        m2::ProgramRunArgs::KernelRunArgs compute_args{.kernel = m2::KernelSpecName{"compute"}};
        for (size_t i = 0; i < g.cores.size(); ++i) {
            compute_args.runtime_arg_values.push_back(
                {g.cores[i], {{"total_blocks", g.number_of_blocks_per_core[i]}}});
        }
        args.kernel_run_args.push_back(std::move(compute_args));
    }

    auto reader_index = [&](const char* kernel) {
        m2::ProgramRunArgs::KernelRunArgs ra{.kernel = m2::KernelSpecName{kernel}};
        for (size_t i = 0; i < g.cores.size(); ++i) {
            uint32_t idx = 0;
            if (g.config_tensors_in_dram) {
                if (g.is_height_sharded) {
                    idx = static_cast<uint32_t>(i);
                } else if (g.is_block_sharded) {
                    idx = g.is_rm_orientation ? g.cores[i].y : g.cores[i].x;
                }
            }
            ra.runtime_arg_values.push_back({g.cores[i], {{"config_read_index", idx}}});
        }
        return ra;
    };
    args.kernel_run_args.push_back(reader_index("reader0"));
    args.kernel_run_args.push_back(reader_index("reader1"));
    return args;
}

// create_per_enqueue_args — the per-call tensor bindings: the sharded input and output shards. The config
// tensors are op-owned (create_owned_tensors) and bound invariant, so they are not re-applied here.
m2::ProgramRunArgs UntilizeWithHaloProgramFactory::create_per_enqueue_args(
    const HaloParams&, const Tensor& input, Tensor& output, const std::optional<ttnn::MeshCoordinate>&) {
    m2::ProgramRunArgs args;
    args.tensor_args.emplace(
        m2::TensorParamName{TP_INPUT}, m2::ProgramRunArgs::TensorArgument{std::cref(input.mesh_tensor())});
    args.tensor_args.emplace(
        m2::TensorParamName{TP_OUTPUT}, m2::ProgramRunArgs::TensorArgument{std::cref(output.mesh_tensor())});
    return args;
}

}  // namespace ttnn::prim
