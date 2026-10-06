// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <umd/device/types/arch.hpp>

#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_buffers.hpp"
#include "ttnn/operations/matmul/device/matmul_desc.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/matmul_validation.hpp"

// Default program config selection for matmul (issue #57884), enabled by ttnn.CONFIG.matmul_auto_config_v2.
// It runs only when no program config is given; a measured registry entry (#54943), when present, takes
// precedence because it sets the program config before this is reached.
//
// The selection is a pure function of the matmul's specs (MatmulSpecs, and the MatmulDesc describing them) and a
// HardwareDesc (grid, L1), so it can be exercised for any architecture without a device. A Selector combines
//  - candidate sources (CandidateSource), which propose legal candidates: the factory blocking source
//    (factory_blocking_source.hpp) derives them from each factory's blocking rules; and
//  - estimators (Estimator), which rank the proposals: the roofline (roofline_estimator.hpp) by default.
// select() ranks the proposals with the estimators and returns the best that passes check() (the device op's
// validation, plus what the factories need beyond it). A measured registry would plug in as another source and
// estimator. Problems it has no config for return nullopt with the reason; matmul reports them as an error (they are
// inputs the factories can't run).
namespace ttnn::operations::matmul::auto_config {

// Hardware facts the selector depends on. Tests can describe other architectures directly.
struct HardwareDesc {
    tt::ARCH arch = tt::ARCH::WORMHOLE_B0;
    CoreCoord grid;                // worker grid available to this matmul
    CoreCoord origin{0, 0};        // its first core (a sub-device's worker grid need not start at (0, 0))
    bool pinned_origin = false;    // the configs must name the grid's cores (a sub-device's) explicitly
    uint32_t l1_cb_budget = 0;     // per-core bytes available for circular buffers
    uint32_t dram_alignment = 32;  // bytes; tiles read from DRAM are padded to this
    // Nominal rates for the roofline estimate, per core clock cycle (tech_reports/GEMM_FLOPS and
    // tech_reports/FlashAttention): matrix engine FLOPs at LoFi (8x16 x 16x16 per cycle), divided by the math
    // fidelity; one NoC link into a core; the chip's DRAM bandwidth.
    uint32_t matmul_flops_per_cycle = 4096;
    uint32_t noc_bytes_per_cycle = 32;
    double dram_bytes_per_cycle = 288.0;  // 288 GB/s at 1 GHz

    static HardwareDesc for_arch(tt::ARCH arch, CoreCoord grid, uint32_t l1_cb_budget);
};

enum class Family { Mcast2D, Mcast1DIn0, Mcast1DIn1, Reuse };

struct Candidate {
    Family family;
    Blocking blocking;
    uint32_t cores = 0;                     // cores with work
    CoreCoord grid;                         // compute_with_storage_grid_size of the config
    std::optional<CoreRange> worker_cores;  // allowed_worker_cores (sharded layouts: the shard grid)
    bool transpose_mcast = false;           // 2D on a column-major block-sharded A
    bool fuse_batch = true;                 // mcast families: the batch folded into M, else looped over
};

// Per-core L1 bytes the factory for `family` needs with this blocking: the buffers it allocates (sized by the
// factory's own functions in matmul_buffers.hpp), plus a sharded output's shard, which is not allocated yet.
// `fuse_batch`: the batch folded into M (per_core_M and out_block_h count rows of all batches), else looped over.
uint32_t circular_buffer_bytes(
    const MatmulDesc& matmul, const HardwareDesc& hw, Family family, const Blocking& b, bool fuse_batch);

// The program config of a candidate.
MatmulProgramConfig to_program_config(const MatmulDesc& matmul, const Candidate& candidate);

// What the factories need of a config beyond the device op's validation, or empty: subblocks within the DST
// capacity they compute correctly with, a single K block for Reuse with block-float B and A tiles under 16 rows,
// and buffers that fit hw's L1 budget.
std::string factory_limit_error(const MatmulDesc& matmul, const HardwareDesc& hw, const MatmulProgramConfig& config);

// Whether matmul can run `config`: empty if so, else the first rule it breaks, from the device op's validation
// (program_config_error) and then factory_limit_error. `matmul` describes `specs`.
std::string check(
    const ttnn::prim::MatmulSpecs& specs,
    const MatmulDesc& matmul,
    const HardwareDesc& hw,
    const MatmulProgramConfig& config);

// A cost estimate of a candidate, from one Estimator.
struct Estimate {
    double cycles = 0;      // estimated device time
    double confidence = 0;  // how far to trust it for this problem; the most confident estimate is used
    std::string_view source;
};

// Estimates a candidate's device time, or returns nullopt when it has no estimate for this problem. Estimators
// rank candidates; they never create them, so they can't make an illegal choice.
class Estimator {
public:
    virtual ~Estimator() = default;
    virtual std::string_view name() const = 0;
    virtual std::optional<Estimate> estimate(
        const MatmulDesc& matmul, const HardwareDesc& hw, const Candidate& candidate) const = 0;
};

// Proposes candidates for a matmul; select() keeps those that pass check(). A source proposes nothing for
// matmuls it doesn't handle.
class CandidateSource {
public:
    virtual ~CandidateSource() = default;
    virtual std::string_view name() const = 0;
    virtual std::vector<Candidate> propose(const MatmulDesc& matmul, const HardwareDesc& hw) const = 0;
};

// Where candidates come from and how they are ranked
struct Selector {
    std::vector<std::shared_ptr<const CandidateSource>> sources;
    std::vector<std::shared_ptr<const Estimator>> estimators;
};

// The factory blocking source with the heuristic policies, ranked by the roofline estimate
const Selector& default_selector();

// `options` best first by estimate: per option the most confident estimate (the earlier estimator on ties), then
// the lowest cycles (the earlier option on ties). Options without an estimate come after those with one.
std::vector<const Candidate*> rank_by_estimate(
    const MatmulDesc& matmul,
    const HardwareDesc& hw,
    std::span<const Candidate> options,
    std::span<const std::shared_ptr<const Estimator>> estimators);

// The best of the sources' proposals (in source order) by estimate that passes check(); nullopt if none does.
// Proposals are checked in rank order, so usually only the best one is. `matmul` describes `specs`.
std::optional<Candidate> select(
    const ttnn::prim::MatmulSpecs& specs,
    const MatmulDesc& matmul,
    const HardwareDesc& hw,
    const Selector& selector = default_selector());

// The program config for the chosen candidate, or nullopt for inputs it has no config for, with the reason in
// `unsupported` when given.
std::optional<MatmulProgramConfig> select_program_config(
    const ttnn::prim::MatmulSpecs& specs, const HardwareDesc& hw, std::string* unsupported = nullptr);

// Builds the MatmulSpecs and HardwareDesc from matmul's inputs and selects a config (as above).
std::optional<MatmulProgramConfig> select_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias,
    const ttnn::prim::MatmulParams& attributes,
    std::string* unsupported = nullptr);

// For sweeps (testing/benchmark only): a config for a matmul call, where it came from ("heuristic": the config
// select_program_config chooses; "enumerated": another of EnumeratingSource's proposals), and why the device op
// rejects it (check(); empty if it doesn't).
struct EnumeratedConfig {
    MatmulProgramConfig config;
    std::string origin;
    std::string error;
};

// The config select_program_config chooses for matmul's inputs, then every other config EnumeratingSource proposes;
// empty, with the reason in `unsupported` when given, for inputs the selector has no config for.
std::vector<EnumeratedConfig> enumerate_program_configs(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias,
    const ttnn::prim::MatmulParams& attributes,
    std::string* unsupported = nullptr);

// While on (per thread), each matmul that chooses its own config also records enumerate_program_configs for its
// inputs, read back with last_enumerated_configs.
void set_record_enumerated_configs(bool on);
void record_enumerated_configs(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias,
    const ttnn::prim::MatmulParams& attributes);
// The configs last recorded on this thread, and why there are none (unsupported inputs); with reset, cleared after
std::pair<std::vector<EnumeratedConfig>, std::string> last_enumerated_configs(bool reset);

}  // namespace ttnn::operations::matmul::auto_config
