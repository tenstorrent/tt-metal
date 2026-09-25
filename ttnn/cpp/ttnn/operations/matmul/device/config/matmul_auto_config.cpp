// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/matmul_auto_config.hpp"

#include <algorithm>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/hal.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::matmul::auto_config {

namespace {

constexpr uint32_t TILE_DIM = 32;
// Headroom kept free below the L1 budget, for allocator alignment and small factory-side buffers
constexpr uint32_t L1_HEADROOM_BYTES = 16 * 1024;

uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
uint32_t align_up(uint32_t a, uint32_t alignment) { return div_up(a, alignment) * alignment; }

// Largest in0_block_w (see MAX_IN0_BLOCK_W and MAX_SELF_READ_TILES_PER_K_STEP). With a single K block the
// mcast factories single-buffer the inputs, so reading the next block can't overlap math on the current one:
// keep two blocks when K allows it. The reuse factory always double-buffers.
uint32_t max_in0_block_w(uint32_t Kt, Family family, uint32_t out_block_h, uint32_t out_block_w) {
    const uint32_t two_blocks = (family != Family::Reuse && Kt >= 2) ? Kt / 2 : Kt;
    // Tiles per K step of the operand(s) each core reads itself rather than receiving by multicast
    uint32_t self_read = 0;
    switch (family) {
        case Family::Mcast2D: self_read = 0; break;
        case Family::Mcast1DIn0: self_read = out_block_w; break;
        case Family::Mcast1DIn1: self_read = out_block_h; break;
        case Family::Reuse: self_read = out_block_h + out_block_w; break;
    }
    const uint32_t self_read_limit =
        self_read == 0 ? MAX_IN0_BLOCK_W : std::max(1u, MAX_SELF_READ_TILES_PER_K_STEP / self_read);
    return std::min({MAX_IN0_BLOCK_W, two_blocks, self_read_limit});
}

// Rows of output tiles the mcast families split across cores: all batches when fused, else one batch.
uint32_t output_rows(const Problem& p, bool fuse_batch) { return fuse_batch ? p.batch_a * p.Mt : p.Mt; }

// Divisors of n, largest first.
std::vector<uint32_t> divisors_desc(uint32_t n) {
    std::vector<uint32_t> small;
    std::vector<uint32_t> large;
    for (uint32_t d = 1; d * d <= n; ++d) {
        if (n % d == 0) {
            small.push_back(d);
            if (d != n / d) {
                large.push_back(n / d);
            }
        }
    }
    // large holds n/1, n/2, ... (descending); append the small ones in descending order
    large.insert(large.end(), small.rbegin(), small.rend());
    return large;
}

// Tiles held in the destination register for one subblock.
uint32_t max_subblock_area(const Problem& p, Family family) {
    uint32_t area = p.dst_full_sync_en ? 16 : 8;
    if (p.fp32_dest_acc_en) {
        area /= 2;
        // The reuse factory caps fp32-accumulating subblocks at 4 even with full-sync dest
        if (family == Family::Reuse) {
            area = std::min(area, 4u);
        }
    }
    return area;
}

// Largest-area subblock dividing (block_h, block_w); ties prefer the wider one. `h_divides` restricts
// the height further (Reuse with batched inputs needs out_subblock_h | Mt).
std::pair<uint32_t, uint32_t> choose_subblock(
    uint32_t block_h, uint32_t block_w, uint32_t max_area, uint32_t h_divides = 0) {
    std::pair<uint32_t, uint32_t> best{1, 1};
    for (uint32_t h = 1; h <= std::min(block_h, max_area); ++h) {
        if (block_h % h != 0 || (h_divides != 0 && h_divides % h != 0)) {
            continue;
        }
        for (uint32_t w = std::min(block_w, max_area / h); w >= 1; --w) {
            if (block_w % w != 0) {
                continue;
            }
            const auto area = h * w;
            const auto best_area = best.first * best.second;
            if (area > best_area || (area == best_area && w > best.second)) {
                best = {h, w};
            }
            break;  // widest w for this h found
        }
    }
    return best;
}

bool packer_l1_acc_enabled(const Problem& p, Family family, uint32_t num_k_blocks) {
    if (!p.packer_l1_acc) {
        return false;
    }
    switch (family) {
        case Family::Mcast1DIn0: return num_k_blocks > 1;
        case Family::Reuse: return num_k_blocks > 2;
        case Family::Mcast2D:
        case Family::Mcast1DIn1: return (p.bias_tile_bytes != 0 && num_k_blocks > 1) || num_k_blocks > 2;
    }
    return false;
}

}  // namespace

HardwareDesc HardwareDesc::for_arch(tt::ARCH arch, CoreCoord grid, uint32_t l1_cb_budget) {
    HardwareDesc hw;
    hw.arch = arch;
    hw.grid = grid;
    hw.l1_cb_budget = l1_cb_budget;
    hw.dram_alignment = arch == tt::ARCH::BLACKHOLE ? 64 : 32;
    return hw;
}

uint32_t circular_buffer_bytes(const Problem& p, const HardwareDesc& hw, Family family, const Blocking& b) {
    const uint32_t in0_tile = align_up(tt::tile_size(p.in0_format), hw.dram_alignment);
    const uint32_t in1_tile = align_up(tt::tile_size(p.in1_format), hw.dram_alignment);
    const uint32_t out_tile = tt::tile_size(p.out_format);
    const uint32_t num_k_blocks = p.Kt / b.in0_block_w;

    const bool l1_acc = packer_l1_acc_enabled(p, family, num_k_blocks);
    const tt::DataFormat interm_format =
        p.fp32_dest_acc_en ? tt::DataFormat::Float32 : (l1_acc ? tt::DataFormat::Float16_b : p.out_format);
    const bool interm_shares_out = interm_format == p.out_format;

    uint32_t in0_bytes = 0;
    uint32_t in1_bytes = 0;
    uint32_t out_tiles = 0;
    uint32_t bias_bytes = 0;
    if (family == Family::Reuse) {
        const uint32_t per_batch_M = std::min(b.per_core_M, p.Mt);
        in0_bytes = per_batch_M * b.in0_block_w * 2 * in0_tile;
        in1_bytes = b.per_core_N * b.in0_block_w * 2 * in1_tile;
        out_tiles = b.per_core_M * b.per_core_N;
        bias_bytes = per_batch_M * b.per_core_N * p.bias_tile_bytes;
    } else {
        const bool looped_batches = !b.fuse_batch && p.batch_a > 1;
        const uint32_t buffering = (num_k_blocks > 1 || looped_batches) ? utilities::MCAST_INPUT_BUFFERING_DEPTH : 1;
        in0_bytes = b.out_block_h * b.in0_block_w * buffering * in0_tile;
        in1_bytes = b.out_block_w * b.in0_block_w * buffering * in1_tile;
        out_tiles = b.out_block_h * b.out_block_w;
        bias_bytes = p.bias_tile_bytes == 0 ? 0 : b.out_block_w * align_up(p.bias_tile_bytes, hw.dram_alignment);
    }
    uint32_t total = in0_bytes + in1_bytes + out_tiles * out_tile + bias_bytes;
    if (!interm_shares_out) {
        total += out_tiles * tt::tile_size(interm_format);
    }
    if (p.transpose_a) {
        total += in0_bytes;  // CB holding the transposed in0 block
    }
    return total;
}

namespace {

// 2D mcast (issue #57884 heuristic 1): largest in0_block_w * out_block_h * out_block_w that fits L1.
std::optional<Blocking> block_2d(
    const Problem& p, const HardwareDesc& hw, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch) {
    const auto k_options = divisors_desc(p.Kt);
    const uint32_t k_max = max_in0_block_w(p.Kt, Family::Mcast2D, per_core_M, per_core_N);
    std::optional<Blocking> best;
    uint64_t best_product = 0;
    uint64_t best_area = 0;
    for (uint32_t h : divisors_desc(per_core_M)) {
        for (uint32_t w : divisors_desc(per_core_N)) {
            const uint64_t area = static_cast<uint64_t>(h) * w;
            if (area * k_max < best_product) {
                break;  // narrower blocks for this h can't win
            }
            for (uint32_t k : k_options) {
                if (k > k_max) {
                    continue;
                }
                Blocking b{per_core_M, per_core_N, k, h, w, 0, 0, fuse_batch};
                if (circular_buffer_bytes(p, hw, Family::Mcast2D, b) > hw.l1_cb_budget) {
                    continue;
                }
                const uint64_t product = area * k;
                if (product > best_product || (product == best_product && area > best_area)) {
                    best = b;
                    best_product = product;
                    best_area = area;
                }
                break;  // largest fitting k for this output block
            }
        }
    }
    return best;
}

// 1D mcast (issue #57884 heuristic 2): keep the full per-core extent along the multicast dimension, shrink
// the other one only if needed; in0_block_w is the largest that fits, up to MAX_IN0_BLOCK_W.
std::optional<Blocking> block_1d(
    const Problem& p,
    const HardwareDesc& hw,
    Family family,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch) {
    const bool is_tall = family == Family::Mcast1DIn1;
    const uint32_t fixed = is_tall ? per_core_N : per_core_M;
    const uint32_t cheap_full = is_tall ? per_core_M : per_core_N;
    const uint32_t M_rows = output_rows(p, fuse_batch);
    std::optional<Blocking> best;
    uint64_t best_product = 0;
    for (uint32_t cheap : divisors_desc(cheap_full)) {
        const uint32_t out_block_h = is_tall ? cheap : fixed;
        const uint32_t out_block_w = is_tall ? fixed : cheap;
        // in1-mcast with a single block row: Mt % out_block_h == 0 or one output block per core
        if (is_tall && div_up(M_rows, per_core_M) == 1 && M_rows % out_block_h != 0 && per_core_M != out_block_h) {
            continue;
        }
        const uint32_t k_limit = max_in0_block_w(p.Kt, family, out_block_h, out_block_w);
        for (uint32_t k : divisors_desc(p.Kt)) {
            if (k > k_limit) {
                continue;
            }
            Blocking b{per_core_M, per_core_N, k, out_block_h, out_block_w, 0, 0, fuse_batch};
            if (circular_buffer_bytes(p, hw, family, b) > hw.l1_cb_budget) {
                continue;
            }
            const uint64_t product = static_cast<uint64_t>(out_block_h) * out_block_w * k;
            const uint64_t area = static_cast<uint64_t>(out_block_h) * out_block_w;
            const uint64_t best_area = best ? static_cast<uint64_t>(best->out_block_h) * best->out_block_w : 0;
            if (product > best_product || (product == best_product && area > best_area)) {
                best = b;
                best_product = product;
            }
            break;  // largest fitting k for this output block
        }
        if (best && (best->in0_block_w > 1 || best_product == static_cast<uint64_t>(per_core_M) * per_core_N)) {
            break;  // full block already fits at k >= 1; don't split further
        }
    }
    return best;
}

// Reuse (batched B): per_core_N = Nt, and each core block is a whole batch matrix (per_core_M = Mt) unless
// the batch alone leaves cores idle. Then each matrix is split into row slices, as many as still give every
// core at most one block: the reuse factory leaves output unwritten when a core gets several blocks that are
// partial batch matrices (#57954). in0_block_w is the largest divisor of Kt up to MAX_IN0_BLOCK_W that fits L1.
std::optional<Blocking> block_reuse(const Problem& p, const HardwareDesc& hw) {
    const uint32_t cores = hw.grid.x * hw.grid.y;
    std::optional<Blocking> best;
    for (uint32_t per_core_M : divisors_desc(p.Mt)) {
        if (per_core_M < p.Mt && p.batch_a * (p.Mt / per_core_M) > cores) {
            break;  // smaller slices would put several partial blocks on a core
        }
        for (uint32_t k : divisors_desc(p.Kt)) {
            if (k > max_in0_block_w(p.Kt, Family::Reuse, std::min(per_core_M, p.Mt), p.Nt)) {
                continue;
            }
            Blocking b{per_core_M, p.Nt, k, per_core_M, p.Nt, 0, 0};
            if (circular_buffer_bytes(p, hw, Family::Reuse, b) <= hw.l1_cb_budget) {
                best = b;  // fits; keep looking for a finer split that is still one block per core
                break;
            }
        }
    }
    return best;
}

// Input tiles per K tile each core reads: its rows of A plus its columns of B
uint32_t per_core_input_tiles(const Blocking& b) { return b.per_core_M + b.per_core_N; }

uint32_t cores_used(const Problem& p, const HardwareDesc& hw, Family family, const Blocking& b) {
    if (family == Family::Reuse) {
        const uint32_t blocks = p.batch_a * p.Mt / b.per_core_M;
        return std::min(blocks, static_cast<uint32_t>(hw.grid.x * hw.grid.y));
    }
    return div_up(output_rows(p, b.fuse_batch), b.per_core_M) * div_up(p.Nt, b.per_core_N);
}

void set_subblock(const Problem& p, Family family, Blocking& b) {
    const bool reuse = family == Family::Reuse;
    // Reuse with batched A and B requires out_subblock_h | Mt
    const auto [h, w] = choose_subblock(
        reuse ? b.per_core_M : b.out_block_h,
        reuse ? b.per_core_N : b.out_block_w,
        max_subblock_area(p, family),
        reuse ? p.Mt : 0);
    b.out_subblock_h = h;
    b.out_subblock_w = w;
}

}  // namespace

std::vector<Candidate> candidates(const Problem& p, const HardwareDesc& hw) {
    std::vector<Candidate> result;
    const uint32_t cores = hw.grid.x * hw.grid.y;
    auto add = [&](Family family, std::optional<Blocking> b) {
        if (!b) {
            return;
        }
        set_subblock(p, family, *b);
        result.push_back({family, *b, cores_used(p, hw, family, *b)});
    };
    // Batched B can't be fused into M: the mcast families then loop over the batch
    const bool fuse_batch = p.batch_b == 1;
    if (!fuse_batch) {
        add(Family::Reuse, block_reuse(p, hw));
    }
    const uint32_t M = output_rows(p, fuse_batch);
    add(Family::Mcast2D, block_2d(p, hw, div_up(M, hw.grid.y), div_up(p.Nt, hw.grid.x), fuse_batch));
    add(Family::Mcast1DIn0, block_1d(p, hw, Family::Mcast1DIn0, M, div_up(p.Nt, cores), fuse_batch));
    add(Family::Mcast1DIn1, block_1d(p, hw, Family::Mcast1DIn1, div_up(M, cores), p.Nt, fuse_batch));
    return result;
}

std::optional<Candidate> choose_candidate(const Problem& p, const HardwareDesc& hw) {
    const auto all = candidates(p, hw);
    if (all.empty()) {
        return std::nullopt;
    }
    auto find = [&](Family family) -> const Candidate* {
        for (const auto& c : all) {
            if (c.family == family) {
                return &c;
            }
        }
        return nullptr;
    };
    const Candidate* two_d = find(Family::Mcast2D);
    const Candidate* in0 = find(Family::Mcast1DIn0);
    const Candidate* in1 = find(Family::Mcast1DIn1);
    if (const auto* reuse = find(Family::Reuse)) {
        // Reuse unless a batch-looping mcast layout keeps clearly more cores busy
        const Candidate* widest = nullptr;
        for (const auto* c : {two_d, in0, in1}) {
            if (c && (!widest || c->cores > widest->cores)) {
                widest = c;
            }
        }
        return widest && widest->cores >= ONE_D_CORE_ADVANTAGE * reuse->cores ? *widest : *reuse;
    }
    // 2D unless a 1D layout keeps clearly more cores busy; in0-mcast first when both do
    const double threshold = two_d ? ONE_D_CORE_ADVANTAGE * two_d->cores : 0;
    const Candidate* one_d = nullptr;
    if (in0 && in0->cores >= threshold && (!in1 || in0->cores >= in1->cores)) {
        one_d = in0;
    } else if (in1 && in1->cores >= threshold) {
        one_d = in1;
    } else if (!two_d) {
        one_d = in0 ? in0 : in1;
    } else if (
        in0 && in0->cores >= two_d->cores &&
        per_core_input_tiles(in0->blocking) < per_core_input_tiles(two_d->blocking)) {
        // As many cores busy and each reads less input (a taller, squarer per-core block): in0-mcast
        one_d = in0;
    }
    return one_d ? *one_d : *two_d;
}

std::optional<MatmulProgramConfig> select_program_config(const Problem& p, const HardwareDesc& hw) {
    if (p.Mt == 0 || p.Kt == 0 || p.Nt == 0 || hw.grid.x == 0 || hw.grid.y == 0) {
        return std::nullopt;
    }
    const auto chosen = choose_candidate(p, hw);
    if (!chosen) {
        return std::nullopt;
    }
    const Family family = chosen->family;
    const Blocking& b = chosen->blocking;
    switch (family) {
        case Family::Mcast2D:
            return MatmulMultiCoreReuseMultiCastProgramConfig{
                .compute_with_storage_grid_size = hw.grid,
                .in0_block_w = b.in0_block_w,
                .out_subblock_h = b.out_subblock_h,
                .out_subblock_w = b.out_subblock_w,
                .out_block_h = b.out_block_h,
                .out_block_w = b.out_block_w,
                .per_core_M = b.per_core_M,
                .per_core_N = b.per_core_N,
                .transpose_mcast = false,
                .fused_activation = p.activation,
                .fuse_batch = b.fuse_batch,
            };
        case Family::Mcast1DIn0:
        case Family::Mcast1DIn1:
            return MatmulMultiCoreReuseMultiCast1DProgramConfig{
                .compute_with_storage_grid_size = hw.grid,
                .in0_block_w = b.in0_block_w,
                .out_subblock_h = b.out_subblock_h,
                .out_subblock_w = b.out_subblock_w,
                .out_block_h = b.out_block_h,
                .out_block_w = b.out_block_w,
                .per_core_M = b.per_core_M,
                .per_core_N = b.per_core_N,
                .fuse_batch = b.fuse_batch,
                .fused_activation = p.activation,
                .mcast_in0 = family == Family::Mcast1DIn0,
            };
        case Family::Reuse:
            return MatmulMultiCoreReuseProgramConfig{
                .compute_with_storage_grid_size = hw.grid,
                .in0_block_w = b.in0_block_w,
                .out_subblock_h = b.out_subblock_h,
                .out_subblock_w = b.out_subblock_w,
                .per_core_M = b.per_core_M,
                .per_core_N = b.per_core_N,
            };
    }
    return std::nullopt;
}

std::optional<MatmulProgramConfig> select_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const bool transpose_a,
    const bool transpose_b,
    const uint32_t bias_single_tile_size,
    const ttnn::prim::MatmulParams& attributes) {
    // Scope of the new selector so far: interleaved operands and output with 32x32 tiles on one grid
    const auto& output_mem_config = attributes.output_mem_config;
    if (input_tensor_a.is_sharded() || input_tensor_b.is_sharded() || output_mem_config.is_sharded() ||
        attributes.global_cb.has_value() || attributes.sub_device_id.has_value()) {
        return std::nullopt;
    }
    const auto is_32x32 = [](const tt::tt_metal::Tile& tile) {
        return tile.get_height() == TILE_DIM && tile.get_width() == TILE_DIM;
    };
    if (!is_32x32(input_tensor_a.tensor_spec().tile()) || !is_32x32(input_tensor_b.tensor_spec().tile()) ||
        !is_32x32(attributes.output_tile.value_or(tt::tt_metal::Tile()))) {
        return std::nullopt;
    }

    const auto a_shape = utilities::get_matmul_tensor_padded_shape(input_tensor_a, transpose_a);
    const auto b_shape = utilities::get_matmul_tensor_padded_shape(input_tensor_b, transpose_b);
    if (a_shape.rank() < 2 || b_shape.rank() < 2) {
        return std::nullopt;
    }
    Problem p;
    p.batch_a = a_shape.volume() / (a_shape[-2] * a_shape[-1]);
    p.batch_b = b_shape.volume() / (b_shape[-2] * b_shape[-1]);
    p.Mt = a_shape[-2] / TILE_DIM;
    p.Kt = a_shape[-1] / TILE_DIM;
    p.Nt = b_shape[-1] / TILE_DIM;
    // Batched B needs a matching A batch (A batch 1 against batched B is left to the legacy path)
    if (p.batch_b > 1 && p.batch_a != p.batch_b) {
        return std::nullopt;
    }
    // transpose_a can't fuse a batch of multi-tile-row matrices into M
    if (transpose_a && p.batch_b == 1 && p.batch_a > 1 && p.Mt > 1) {
        return std::nullopt;
    }

    if (!attributes.compute_kernel_config.has_value()) {
        return std::nullopt;
    }
    auto* device = input_tensor_a.device();
    const auto arch = device->arch();
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, attributes.compute_kernel_config.value());
    p.in0_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_a.dtype());
    p.in1_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_b.dtype());
    p.out_format =
        tt::tt_metal::datatype_to_dataformat_converter(attributes.output_dtype.value_or(input_tensor_a.dtype()));
    p.bias_tile_bytes = bias_single_tile_size;
    p.transpose_a = transpose_a;
    p.math_fidelity = math_fidelity;
    p.fp32_dest_acc_en = fp32_dest_acc_en;
    p.packer_l1_acc = packer_l1_acc;
    p.dst_full_sync_en = dst_full_sync_en;
    p.activation = attributes.user_fused_activation;

    auto grid = device->compute_with_storage_grid_size();
    if (attributes.user_core_coord.has_value()) {
        const auto& user = attributes.user_core_coord.value();
        if (user.x > 0 && user.y > 0) {
            grid = CoreCoord(std::min(user.x, grid.x), std::min(user.y, grid.y));
        }
    }

    // L1 left for CBs: free space above the lowest L1 buffer, less this op's own L1 output (not allocated yet)
    uint32_t budget = utilities::get_max_l1_space(input_tensor_a);
    if (output_mem_config.buffer_type() == tt::tt_metal::BufferType::L1) {
        const uint64_t out_tiles = static_cast<uint64_t>(p.batch_a) * p.Mt * p.Nt;
        const uint32_t num_banks = device->allocator()->get_num_banks(tt::tt_metal::BufferType::L1);
        const uint64_t out_per_bank = div_up(out_tiles, num_banks) * static_cast<uint64_t>(tt::tile_size(p.out_format));
        budget = out_per_bank >= budget ? 0 : budget - static_cast<uint32_t>(out_per_bank);
    }
    budget = budget > L1_HEADROOM_BYTES ? budget - L1_HEADROOM_BYTES : 0;

    return select_program_config(p, HardwareDesc::for_arch(arch, grid, budget));
}

}  // namespace ttnn::operations::matmul::auto_config
