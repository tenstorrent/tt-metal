// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_builder.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa_recipe.hpp"

#include <array>
#include <cstdint>
#include <optional>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::prim {

namespace {

namespace ring_recipes = ttnn::operations::transformer::sdpa::detail;
using ring_joint_sdpa::ComputeCb;

// Named precision recipes B/C/D/E on ring joint SDPA. The recipe compute owns CB indices 0-16 (the layout of
// recipe_compute_program); the ring dataflow CBs start at 19. CB17/CB18 carry the raw state checkpoint
// request/ack between compute and writer for multi-Q workers, backed by the internal state tensor (output 3).
class RecipeRingJointCompute final : public ring_joint_sdpa::ComputeVariant {
public:
    static constexpr uint32_t kCheckpointRequestCb = 17;
    static constexpr uint32_t kCheckpointAckCb = 18;
    static constexpr uint32_t kFirstDataflowCb = 19;
    static constexpr size_t kStateOutputIdx = 3;

    ring_joint_sdpa::KernelSources kernel_sources() const override {
        return {
            "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_reader_recipe.cpp",
            "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_writer_recipe.cpp",
            "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/ring_joint_sdpa_recipe.cpp",
        };
    }

    void configure(
        const RingJointSDPAParams& args,
        const RingJointSDPAInputs& tensor_args,
        tt::tt_metal::ProgramDescriptor& desc,
        const tt::tt_metal::CoreRangeSet& grid,
        uint32_t Sq_chunk_t,
        uint32_t Sk_chunk_t,
        uint32_t DHt) override {
        TT_FATAL(
            args.precision && *args.precision != ttnn::transformer::SDPAPrecision::FAST,
            "The ring recipe program factory serves the named recipes B-E; FAST uses the ring joint compute");
        program_ = ring_recipes::recipe_compute_program(
            ring_recipes::resolve_precision_policy(
                ring_recipes::select_recipe(*args.precision, tensor_args.input_k.dtype())),
            grid,
            1,
            Sq_chunk_t,
            Sk_chunk_t,
            DHt);
        desc.cbs = program_.cbs;
        grid_ = grid;
        Sq_chunk_t_ = Sq_chunk_t;
        Sk_chunk_t_ = Sk_chunk_t;
    }

    std::optional<uint32_t> fixed_subblock_h(bool fp32_dest_acc_en) const override {
        return fp32_dest_acc_en ? 1u : 2u;
    }

    bool resident_ring_state() const override { return true; }

    uint32_t first_dataflow_cb_index() const override { return kFirstDataflowCb; }

    std::optional<uint32_t> fixed_cb_index(ComputeCb role) const override {
        switch (role) {
            case ComputeCb::Q: return 0;
            case ComputeCb::K: return 1;
            case ComputeCb::V: return 2;
            case ComputeCb::ReduceScaler: return 3;
            case ComputeCb::ColumnIdentity: return 4;
            case ComputeCb::RecipScratch: return 5;
            case ComputeCb::QkIm: return 6;
            case ComputeCb::OutImA: return 8;
            case ComputeCb::OutImB: return 9;
            case ComputeCb::MaxA: return 10;
            case ComputeCb::MaxB: return 11;
            case ComputeCb::SumA: return 12;
            case ComputeCb::SumB: return 13;
            case ComputeCb::ExpMaxDiff: return 14;
            case ComputeCb::Out: return 16;
        }
        return std::nullopt;
    }

    void finalize_cbs(
        tt::tt_metal::ProgramDescriptor& desc,
        tt::tt_metal::IDevice* device,
        uint32_t cb_q_in,
        uint32_t q_chunk_bytes) const override {
        for (uint8_t index : {kCheckpointRequestCb, kCheckpointAckCb}) {
            desc.cbs.push_back(tt::tt_metal::CBDescriptor{
                .total_size = 4096,
                .core_ranges = grid_,
                .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                    .buffer_index = index, .data_format = tt::DataFormat::UInt32, .page_size = 4096}}}});
        }
        // Every recipe and ring CB spans the whole worker grid; reject before program creation.
        auto cb_total = [&] {
            uint64_t bytes = 0;
            for (const auto& cb : desc.cbs) {
                bytes += cb.total_size;
            }
            return bytes;
        };
        uint64_t cb_bytes = cb_total();
        const uint64_t available =
            device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
        if (cb_bytes > available) {
            // The recipe double-buffers Q. The ring reader reserves one Q chunk at a time and
            // compute pops it when done, so a single slot is correct; it only gives up Q prefetch.
            for (auto& cb : desc.cbs) {
                if (!cb.format_descriptors.empty() && cb.format_descriptors.front().buffer_index == cb_q_in &&
                    cb.total_size == 2 * q_chunk_bytes) {
                    cb.total_size /= 2;
                    log_debug(tt::LogOp, "Named ring recipe: single-slot Q to fit L1");
                }
            }
            cb_bytes = cb_total();
        }
        TT_FATAL(
            cb_bytes <= available,
            "Named ring SDPA recipe needs {} bytes of L1 per core at Q{}/K{}, but only {} are available; use a smaller "
            "Q or K chunk",
            cb_bytes,
            Sq_chunk_t_ * tt::constants::TILE_HEIGHT,
            Sk_chunk_t_ * tt::constants::TILE_HEIGHT,
            available);
    }

    void append_writer_compile_time_args(
        std::vector<uint32_t>& args, const RingJointSDPAResult& output_tensors) const override {
        tt::tt_metal::TensorAccessorArgs(output_tensors.at(kStateOutputIdx).buffer()).append_to(args);
    }

    void append_writer_common_runtime_args(
        tt::tt_metal::KernelDescriptor& writer, const RingJointSDPAResult& output_tensors) const override {
        tt::tt_metal::KernelDescriptor::RTArgList state_args;
        state_args.push_back(output_tensors.at(kStateOutputIdx).buffer());
        writer.emplace_common_runtime_args(state_args);
    }

    void append_defines(tt::tt_metal::KernelDescriptor::Defines& defines) const override {
        const auto& recipe_defines = program_.kernels.front().defines;
        defines.insert(defines.end(), recipe_defines.begin(), recipe_defines.end());
    }

    std::optional<tt::tt_metal::KernelDescriptor::ConfigDescriptor> compute_config() const override {
        return program_.kernels.front().config;
    }

private:
    tt::tt_metal::ProgramDescriptor program_;
    tt::tt_metal::CoreRangeSet grid_;
    uint32_t Sq_chunk_t_ = 0;
    uint32_t Sk_chunk_t_ = 0;
};

}  // namespace

tt::tt_metal::WorkloadDescriptor RingJointSDPARecipeProgramFactory::create_workload_descriptor(
    const RingJointSDPAParams& args,
    const RingJointSDPAInputs& tensor_args,
    RingJointSDPAResult& output_tensors,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    RecipeRingJointCompute variant;
    return ring_joint_sdpa::build_workload_descriptor(args, tensor_args, output_tensors, tensor_coords, variant);
}

RingJointSDPARecipeMeshWorkloadFactory::cached_mesh_workload_t
RingJointSDPARecipeMeshWorkloadFactory::create_mesh_workload(
    const RingJointSDPAParams& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const RingJointSDPAInputs& tensor_args,
    RingJointSDPAResult& output_tensors) {
    return descriptor_adapter_t::create_mesh_workload(args, tensor_coords, tensor_args, output_tensors);
}

void RingJointSDPARecipeMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const RingJointSDPAParams& args,
    const RingJointSDPAInputs& tensor_args,
    RingJointSDPAResult& output_tensors) {
    descriptor_adapter_t::apply_descriptor(cached_workload, args, tensor_args, output_tensors);
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        ring_joint_sdpa::apply_scalar_runtime_args(program, args, tensor_args, coordinate_range.start_coord());
    }
}

}  // namespace ttnn::prim
