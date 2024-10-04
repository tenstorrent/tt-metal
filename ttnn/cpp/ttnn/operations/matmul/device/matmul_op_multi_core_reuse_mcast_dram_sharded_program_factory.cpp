// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "hostdevcommon/common_values.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"

using namespace tt;
using namespace tt::constants;

namespace reuse_dram_sharded_optimized_helpers {
using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

void get_dram_reader_core_coords_grayskull(
    tt::tt_metal::Device* device, CoreRangeSet& all_cores, std::vector<CoreCoord>& all_cores_ordered) {
    // hardcoded for grayskull
    uint32_t full_grid_size_y = 12;

    // get all the logical coord
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // get dram banks and coords
    uint32_t num_banks = device->num_dram_channels();
    uint32_t max_bank_id = num_banks - 1;
    std::vector<CoreCoord> dram_coord_phy;
    for (int i = 0; i < num_banks; ++i) {
        dram_coord_phy.push_back(device->dram_core_from_dram_channel(i));
    }

    // get worker logical coords
    std::vector<CoreCoord> all_worker_cores_logical;
    for (int i = 0; i < num_cores_x; ++i) {
        for (int j = 0; j < num_cores_y; ++j) {
            all_worker_cores_logical.push_back(CoreCoord(i, j));
        }
    }

    // get y coords of the workers
    std::vector<uint32_t> all_worker_cores_y_physical;
    uint32_t max_worker_y_physical = 0;
    uint32_t min_worker_y_physical = 10000;
    for (int i = 0; i < num_cores_y; ++i) {
        auto core_phy = device->worker_core_from_logical_core(CoreCoord(0, i));
        all_worker_cores_y_physical.push_back(core_phy.y);
        if (core_phy.y > max_worker_y_physical) {
            max_worker_y_physical = core_phy.y;
        }
        if (core_phy.y < min_worker_y_physical) {
            min_worker_y_physical = core_phy.y;
        }
    }

    // get the harvested rows, we treat dram and eth cores as harvested as well
    std::vector<uint32_t> harvested_rows;
    for (int i = 0; i < full_grid_size_y; ++i) {
        auto y = i;

        if (std::find(all_worker_cores_y_physical.begin(), all_worker_cores_y_physical.end(), y) ==
            all_worker_cores_y_physical.end()) {
            harvested_rows.push_back(y);
        }
    }

    // get the ajacent cores of DRAM banks
    std::vector<CoreCoord> adj_core_physical;
    for (int i = 0; i < num_banks; ++i) {
        auto dram_core = dram_coord_phy[i];
        uint32_t adj_core_x = dram_core.x;
        uint32_t adj_core_y = dram_core.y + 1;
        adj_core_physical.push_back(CoreCoord(adj_core_x, adj_core_y));
    }

    // move worker if they are in the harvested rows
    for (auto& coord : adj_core_physical) {
        auto y = coord.y;

        // if row is harvested, move core down by 1
        while (std::find(harvested_rows.begin(), harvested_rows.end(), y) != harvested_rows.end() and
               y < (full_grid_size_y - 1)) {
            y += 1;
        }

        coord.y = y;
    }

    // find the logical coord from physical coord
    std::vector<CoreCoord> adj_core_logical_realloc;
    for (int i = 0; i < adj_core_physical.size(); ++i) {
        for (int j = 0; j < all_worker_cores_logical.size(); ++j) {
            auto core = device->worker_core_from_logical_core(all_worker_cores_logical[j]);
            if (adj_core_physical[i] == core) {
                adj_core_logical_realloc.push_back(all_worker_cores_logical[j]);
            }
        }
    }

    // create sets
    std::set<CoreRange> all_cores_set;
    for (int i = 0; i < num_banks; ++i) {
        all_cores_set.insert(CoreRange(adj_core_logical_realloc[i]));
    }
    all_cores = CoreRangeSet(all_cores_set);
    all_cores_ordered = adj_core_logical_realloc;
}

void get_dram_reader_core_coords_wormhole_b0(
    tt::tt_metal::Device* device, CoreRangeSet& all_cores, std::vector<CoreCoord>& all_cores_ordered) {
    // hardcoded for wh_b0
    uint32_t full_grid_size_y = 12;
    uint32_t x_step = 3;

    // get all the logical coord
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // get dram banks and coords
    uint32_t num_banks = device->num_dram_channels();
    uint32_t max_bank_id = num_banks - 1;
    std::vector<CoreCoord> dram_coord_phy;
    dram_coord_phy.reserve(num_banks);
    for (int i = 0; i < num_banks; ++i) {
        dram_coord_phy.push_back(device->dram_core_from_dram_channel(i));
    }

    // get worker logical coords
    std::vector<CoreCoord> all_worker_cores_logical;
    all_worker_cores_logical.reserve(num_cores_x * num_cores_y);
    for (int i = 0; i < num_cores_x; ++i) {
        for (int j = 0; j < num_cores_y; ++j) {
            all_worker_cores_logical.push_back(CoreCoord(i, j));
        }
    }

    // get y coords of the workers
    std::vector<uint32_t> all_worker_cores_y_physical;
    all_worker_cores_y_physical.reserve(num_cores_y);
    uint32_t max_worker_y_physical = 0;
    uint32_t min_worker_y_physical = 10000;
    for (int i = 0; i < num_cores_y; ++i) {
        auto core_phy = device->worker_core_from_logical_core(CoreCoord(0, i));
        all_worker_cores_y_physical.push_back(core_phy.y);
        if (core_phy.y > max_worker_y_physical) {
            max_worker_y_physical = core_phy.y;
        }
        if (core_phy.y < min_worker_y_physical) {
            min_worker_y_physical = core_phy.y;
        }
    }

    // get the harvested rows, we treat dram and eth cores as harvested as well
    std::vector<uint32_t> harvested_rows;
    for (int i = 0; i < full_grid_size_y; ++i) {
        auto y = i;

        if (std::find(all_worker_cores_y_physical.begin(), all_worker_cores_y_physical.end(), y) ==
            all_worker_cores_y_physical.end()) {
            harvested_rows.push_back(y);
        }
    }

    // get the ajacent cores of DRAM banks
    std::vector<CoreCoord> adj_core_physical;
    adj_core_physical.reserve(num_banks);
    for (int i = 0; i < num_banks; ++i) {
        auto dram_core = dram_coord_phy[i];
        uint32_t adj_core_x = dram_core.x + 1;
        uint32_t adj_core_y = dram_core.y;
        adj_core_physical.push_back(CoreCoord(adj_core_x, adj_core_y));
    }

    // split the adjacent coords into two groups, because DRAM banks has two cols
    std::vector<CoreCoord> adj_core_physical_g1;
    adj_core_physical_g1.reserve(num_banks);
    std::vector<size_t> adj_core_physical_y_g1;
    adj_core_physical_y_g1.reserve(num_banks);
    std::vector<CoreCoord> adj_core_physical_g2;
    adj_core_physical_g2.reserve(num_banks);
    std::vector<size_t> adj_core_physical_y_g2;
    adj_core_physical_y_g2.reserve(num_banks);
    for (auto core : adj_core_physical) {
        if (core.x == adj_core_physical.front().x) {
            adj_core_physical_g1.push_back(core);
        } else {
            adj_core_physical_g2.push_back(core);
        }
    }
    std::vector<int> indices_g1(adj_core_physical_g1.size());
    std::vector<int> indices_g2(adj_core_physical_g2.size());
    std::iota(indices_g1.begin(), indices_g1.end(), 0);
    std::iota(indices_g2.begin(), indices_g2.end(), 0);
    std::sort(indices_g1.begin(), indices_g1.end(), [&adj_core_physical_g1](int i1, int i2) {
        return adj_core_physical_g1[i1].y < adj_core_physical_g1[i2].y;
    });
    std::sort(indices_g2.begin(), indices_g2.end(), [&adj_core_physical_g2](int i1, int i2) {
        return adj_core_physical_g2[i1].y < adj_core_physical_g2[i2].y;
    });
    std::rotate(indices_g1.begin(), indices_g1.end() - 1, indices_g1.end());
    std::rotate(indices_g2.begin(), indices_g2.end() - 1, indices_g2.end());

    std::vector<int> indices_g1_realloc(adj_core_physical_g1.size());
    std::vector<int> indices_g2_realloc(adj_core_physical_g2.size());
    for (int new_index = 0; new_index < indices_g1.size(); ++new_index) {
        indices_g1_realloc[indices_g1[new_index]] = new_index;
    }
    for (int new_index = 0; new_index < indices_g2.size(); ++new_index) {
        indices_g2_realloc[indices_g2[new_index]] = new_index;
    }

    std::sort(adj_core_physical_g1.begin(), adj_core_physical_g1.end(), [](const CoreCoord& a, const CoreCoord& b) {
        return a.y < b.y;
    });
    std::sort(adj_core_physical_g2.begin(), adj_core_physical_g2.end(), [](const CoreCoord& a, const CoreCoord& b) {
        return a.y < b.y;
    });
    std::rotate(adj_core_physical_g1.begin(), adj_core_physical_g1.end() - 1, adj_core_physical_g1.end());
    std::rotate(adj_core_physical_g2.begin(), adj_core_physical_g2.end() - 1, adj_core_physical_g2.end());

    for (auto core : adj_core_physical_g1) {
        adj_core_physical_y_g1.push_back(core.y);
    }
    for (auto core : adj_core_physical_g2) {
        adj_core_physical_y_g2.push_back(core.y);
    }

    // move the workers, if they are on harvested rows
    auto process_group = [&](std::vector<CoreCoord>& group, std::vector<size_t>& group_y, uint32_t x_step) {
        for (auto& coord : group) {
            auto y = coord.y;

            if (std::find(harvested_rows.begin(), harvested_rows.end(), y) != harvested_rows.end() ||
                std::count(group_y.begin(), group_y.end(), y) >= 2) {
                auto adjust_coord = [&](int start, int end, int step) {
                    bool found_new_row = false;
                    for (int j = start; step > 0 ? j <= end : j >= end; j += step) {
                        if (std::find(harvested_rows.begin(), harvested_rows.end(), j) == harvested_rows.end() &&
                            std::count(group_y.begin(), group_y.end(), j) == 0) {
                            coord.y = j;
                            coord.x += x_step;
                            x_step--;
                            found_new_row = true;
                            break;
                        }
                    }
                    if (not found_new_row) {
                        for (int j = start; step > 0 ? j <= end : j >= end; j += step) {
                            if (std::find(harvested_rows.begin(), harvested_rows.end(), j) == harvested_rows.end()) {
                                coord.y = j;
                                coord.x += x_step;
                                x_step--;
                                found_new_row = true;
                                break;
                            }
                        }
                    }
                };

                if (y >= max_bank_id) {
                    adjust_coord(max_worker_y_physical, min_worker_y_physical, -1);
                } else {
                    adjust_coord(min_worker_y_physical, max_worker_y_physical, 1);
                }
            }
        }
    };
    // move the workers, if they are on harvested rows
    process_group(adj_core_physical_g1, adj_core_physical_y_g1, x_step);
    process_group(adj_core_physical_g2, adj_core_physical_y_g2, x_step);

    // merge two group into one
    std::vector<CoreCoord> adj_core_physical_realloc;
    adj_core_physical_realloc.reserve(num_banks);
    for (int i = 0; i < indices_g1_realloc.size(); ++i) {
        adj_core_physical_realloc.push_back(adj_core_physical_g1[indices_g1_realloc[i]]);
    }
    for (int i = 0; i < indices_g2_realloc.size(); ++i) {
        adj_core_physical_realloc.push_back(adj_core_physical_g2[indices_g2_realloc[i]]);
    }

    // find the logical coord from physical coord
    std::vector<CoreCoord> adj_core_logical_realloc;
    adj_core_logical_realloc.reserve(num_banks);
    for (int i = 0; i < adj_core_physical_realloc.size(); ++i) {
        for (int j = 0; j < all_worker_cores_logical.size(); ++j) {
            auto core = device->worker_core_from_logical_core(all_worker_cores_logical[j]);
            if (adj_core_physical_realloc[i] == core) {
                adj_core_logical_realloc.push_back(all_worker_cores_logical[j]);
            }
        }
    }

    // create sets
    std::set<CoreRange> all_cores_set;
    for (int i = 0; i < num_banks; ++i) {
        all_cores_set.insert(CoreRange(adj_core_logical_realloc[i]));
    }
    all_cores = CoreRangeSet(all_cores_set);
    all_cores_ordered = adj_core_logical_realloc;
}

void get_dram_reader_core_coords_blackhole(
    tt_metal::Device* device, CoreRangeSet& all_cores, std::vector<CoreCoord>& all_cores_ordered) {

    // hardcoded for blackhole
    uint32_t full_grid_size_x = 17;

    // get all the logical coord
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // get dram banks and coords
    uint32_t num_banks = device->num_dram_channels();
    uint32_t max_bank_id = num_banks - 1;
    std::vector<CoreCoord> dram_coord_phy;
    for (int i = 0; i < num_banks; ++i) {
        dram_coord_phy.push_back(device->dram_core_from_dram_channel(i));
    }

    // get worker logical coords
    std::vector<CoreCoord> all_worker_cores_logical;
    for (int i = 0; i < num_cores_x; ++i) {
        for (int j = 0; j < num_cores_y; ++j) {
            all_worker_cores_logical.push_back(CoreCoord(i, j));
        }
    }

    // get x coords of the workers
    std::vector<uint32_t> all_worker_cores_x_physical;
    for (int i = 0; i < num_cores_x; ++i) {
        auto core_phy = device->worker_core_from_logical_core(CoreCoord(i, 0));
        all_worker_cores_x_physical.push_back(core_phy.x);
    }

    // get the harvested cols, we treat dram and eth cores as harvested as well
    std::vector<uint32_t> harvested_cols;
    for (int i = 0; i < full_grid_size_x; ++i) {
        auto x = i;

        if (std::find(all_worker_cores_x_physical.begin(), all_worker_cores_x_physical.end(), x) ==
            all_worker_cores_x_physical.end()) {
            harvested_cols.push_back(x);
        }
    }

    // get the ajacent cores of DRAM banks
    std::vector<CoreCoord> adj_core_physical;
    for (int i = 0; i < num_banks; ++i) {
        auto dram_core = dram_coord_phy[i];
        uint32_t adj_core_x = dram_core.x + 1;
        uint32_t adj_core_y = dram_core.y;
        adj_core_physical.push_back(CoreCoord(adj_core_x, adj_core_y));
    }

    // move worker if they are in the harvested cols
    for (auto& coord : adj_core_physical) {
        auto x = coord.x;

        // if col is harvested, move core right by 1
        while (std::find(harvested_cols.begin(), harvested_cols.end(), x) != harvested_cols.end() and x < (full_grid_size_x - 1)) {
            x += 1;
        }

        coord.x = x;
    }

    // find the logical coord from physical coord
    std::vector<CoreCoord> adj_core_logical_realloc;
    for (int i = 0; i < adj_core_physical.size(); ++i) {
        for (int j = 0; j < all_worker_cores_logical.size(); ++j) {
            auto core = device->worker_core_from_logical_core(all_worker_cores_logical[j]);
            if (adj_core_physical[i] == core) {
                adj_core_logical_realloc.push_back(all_worker_cores_logical[j]);
            }
        }
    }

    // create sets
    std::set<CoreRange> all_cores_set;
    for (int i = 0; i < num_banks; ++i) {
        all_cores_set.insert(CoreRange(adj_core_logical_realloc[i]));
    }
    all_cores = CoreRangeSet(all_cores_set);
    all_cores_ordered = adj_core_logical_realloc;
}

void get_max_page_size_and_num_pages(uint32_t num_tiles, uint32_t tile_size, uint32_t& page_size, uint32_t& num_pages) {
    uint64_t total_size = static_cast<uint64_t>(num_tiles) * tile_size;

    page_size = (8192 / tile_size) * tile_size;
    while (total_size % page_size != 0 && page_size >= tile_size) {
        page_size -= tile_size;
    }
    num_pages = total_size / page_size;
}

void move_common_entries(std::vector<CoreCoord>& v1, std::vector<CoreCoord>& v2, std::vector<CoreCoord>& commons) {
    for (const CoreCoord& item : v2) {
        if (std::find(v1.begin(), v1.end(), item) != v1.end()) {
            commons.push_back(item);
        }
    }

    for (const CoreCoord& item : commons) {
        v2.erase(std::remove(v2.begin(), v2.end(), item), v2.end());
    }
}

operation::ProgramWithCallbacks create_program_dram_sharded(
    tt::tt_metal::Device* device,
    CoreRangeSet all_storage_cores,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    uint32_t B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t in0_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N_storage,
    std::optional<UnaryWithParam> fused_activation,
    tt_metal::Buffer* in0_buffer,
    tt_metal::Buffer* in1_buffer,
    tt_metal::Buffer* bias_buffer,
    tt_metal::Buffer* out_buffer,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& bias_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat bias_data_format,
    tt::DataFormat output_data_format,
    bool untilize_out,
    bool skip_compute,
    bool skip_in0_mcast,
    bool skip_write_back) {
    log_debug("math_fidelity: {}", math_fidelity);
    log_debug("fp32_dest_acc_en: {}", fp32_dest_acc_en);
    log_debug("math_approx_mode: {}", math_approx_mode);
    log_debug("packer_l1_acc: {}", packer_l1_acc);
    log_debug("M: {}, K: {}, N: {}", M, K, N);
    log_debug("per_core_M: {}, per_core_N_storage: {}", per_core_M, per_core_N_storage);

    tt_metal::Program program{};

    // get the dram readers
    CoreRangeSet all_worker_cores = CoreRangeSet{{}};
    std::vector<CoreCoord> all_worker_cores_ordered;

    if (device->arch() == tt::ARCH::WORMHOLE_B0) {
        get_dram_reader_core_coords_wormhole_b0(device, all_worker_cores, all_worker_cores_ordered);
    } else if (device->arch() == tt::ARCH::GRAYSKULL) {
        get_dram_reader_core_coords_grayskull(device, all_worker_cores, all_worker_cores_ordered);
    } else if (device->arch() == tt::ARCH::BLACKHOLE) {
        get_dram_reader_core_coords_blackhole(device, all_worker_cores, all_worker_cores_ordered);
    }

    // dram banks
    uint32_t num_dram_banks = all_worker_cores_ordered.size();
    for (auto core : corerange_to_cores(all_worker_cores)) {
        log_debug("all_worker_cores_log: {}", core);
    }
    for (auto core : all_worker_cores_ordered) {
        log_debug("all_worker_cores_ordered: {}", core);
    }

    uint32_t per_core_N = (N + num_dram_banks - 1) / num_dram_banks;
    uint32_t per_core_N_unpad = per_core_N;
    auto subblock_hw = bmm_op_utils::get_matmul_subblock_params(per_core_M, per_core_N, false, false, fp32_dest_acc_en);
    auto out_subblock_h = std::get<0>(subblock_hw);
    auto out_subblock_w = std::get<1>(subblock_hw);

    uint32_t max_subblock_w = fp32_dest_acc_en ? 4 : 8;
    // it is bad for compute, pad per_core_N
    if (out_subblock_h == 1 and out_subblock_w < max_subblock_w) {
        uint32_t num_subblock_w_per_core_N = per_core_N / out_subblock_w;
        uint32_t num_iter = max_subblock_w - out_subblock_w;
        uint32_t new_out_subblock_w = out_subblock_w;
        uint32_t preferred_out_subblock_w = out_subblock_w;

        for (uint32_t i = 0; i < num_iter; ++i) {
            new_out_subblock_w += 1;
            uint32_t new_num_subblock_w_per_core_N = (per_core_N + new_out_subblock_w - 1) / new_out_subblock_w;

            if (new_num_subblock_w_per_core_N < num_subblock_w_per_core_N) {
                num_subblock_w_per_core_N = new_num_subblock_w_per_core_N;
                preferred_out_subblock_w = new_out_subblock_w;
            }
        }
        out_subblock_w = preferred_out_subblock_w;
        per_core_N = out_subblock_w * num_subblock_w_per_core_N;
    }

    log_debug("per_core_M: {}, per_core_N: {}", per_core_M, per_core_N);
    log_debug("out_subblock_h: {}, out_subblock_w: {}", out_subblock_h, out_subblock_w);

    uint32_t num_blocks = K / in0_block_w;
    // Only enable packer l1 accumulation when there are spills, otherwise
    // unnecessary overhead for reconfigs are added
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    // if fp32 enabled then we pack fp32 in l1, if not, then we pack fp16 in l1
    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);
    interm0_data_format = tt::DataFormat::Float16_b;

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles;
    if (B * num_blocks > 1) {
        in0_CB_tiles = in0_CB_tiles * 2;  // double buffer
    }
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;
    uint32_t in1_block_tiles = per_core_N_unpad * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles;
    if (B * num_blocks > 1) {
        in1_CB_tiles = in1_CB_tiles * 3;  // tripple buffer
    }
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_size = out_CB_tiles * interm0_single_tile_size;

    uint32_t out_reshard_block_tiles = per_core_M * per_core_N_storage;
    uint32_t out_reshard_CB_tiles = out_reshard_block_tiles;  // No double buffer
    uint32_t out_reshard_CB_size = out_reshard_CB_tiles * output_single_tile_size;

    uint32_t in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / in0_tile.get_tile_shape()[1];
    uint32_t in0_shard_height_in_tiles = in0_buffer->shard_spec().shape()[0] / in0_tile.get_tile_shape()[0];
    uint32_t in2_block_tiles = per_core_M * in0_shard_width_in_tiles;
    uint32_t in2_CB_tiles = in2_block_tiles;
    uint32_t in2_CB_size = in2_CB_tiles * in0_single_tile_size;

    uint32_t in3_block_tiles = per_core_N_unpad;
    uint32_t in3_CB_tiles = in3_block_tiles;  // No double buffer
    uint32_t in3_CB_size = in3_CB_tiles * bias_single_tile_size;

    // get the max page size based on num tiles
    uint32_t in1_buffer_page_size, in1_buffer_num_pages;
    get_max_page_size_and_num_pages(in1_block_tiles, in1_single_tile_size, in1_buffer_page_size, in1_buffer_num_pages);

    uint32_t bias_buffer_page_size, bias_buffer_num_pages;
    get_max_page_size_and_num_pages(
        in3_block_tiles, bias_single_tile_size, bias_buffer_page_size, bias_buffer_num_pages);

    uint32_t num_worker_cores = num_dram_banks;
    uint32_t num_mcast_cores = num_worker_cores;

    // move conflict coord from mcast receiver to mcast sender
    std::vector<CoreCoord> all_storage_cores_vec = corerange_to_cores(all_storage_cores);
    std::vector<CoreCoord> all_worker_cores_vec = corerange_to_cores(all_worker_cores);
    std::vector<CoreCoord> storage_worker_common;
    move_common_entries(all_storage_cores_vec, all_worker_cores_vec, storage_worker_common);

    std::vector<CoreRange> all_storage_cores_range;
    all_storage_cores_range.reserve(all_storage_cores_vec.size());
    std::transform(
        all_storage_cores_vec.begin(),
        all_storage_cores_vec.end(),
        std::back_inserter(all_storage_cores_range),
        [](const CoreCoord& coord) { return CoreRange(coord); });

    std::vector<CoreRange> all_worker_cores_range;
    all_worker_cores_range.reserve(all_worker_cores_vec.size());
    std::transform(
        all_worker_cores_vec.begin(),
        all_worker_cores_vec.end(),
        std::back_inserter(all_worker_cores_range),
        [](const CoreCoord& coord) { return CoreRange(coord); });

    std::set<CoreRange> all_storage_cores_set(all_storage_cores_range.begin(), all_storage_cores_range.end());
    std::set<CoreRange> all_worker_cores_set(all_worker_cores_range.begin(), all_worker_cores_range.end());
    CoreRangeSet mcast_senders = CoreRangeSet(all_storage_cores_set);
    CoreRangeSet mcast_receivers = CoreRangeSet(all_worker_cores_set);

    for (auto core : corerange_to_cores(mcast_senders)) {
        log_debug("mcast_senders: {}", core);
    }
    for (auto core : corerange_to_cores(mcast_receivers)) {
        log_debug("mcast_receivers: {}", core);
    }

    // all cores
    std::set<CoreRange> all_cores_set;
    all_cores_set.insert(mcast_senders.ranges().begin(), mcast_senders.ranges().end());
    all_cores_set.insert(mcast_receivers.ranges().begin(), mcast_receivers.ranges().end());
    CoreRangeSet all_cores = CoreRangeSet(all_cores_set);

    for (auto core : corerange_to_cores(all_cores)) {
        log_debug("all_cores: {}", core);
    }

    // grid bounding box
    CoreRange bounding_box = all_cores.bounding_box();
    std::set<CoreRange> bounding_box_set;
    bounding_box_set.insert(bounding_box);
    CoreRangeSet all_cores_in_rect_grid(bounding_box_set);
    std::vector<CoreCoord> all_cores_in_rect_grid_vec = corerange_to_cores(all_cores_in_rect_grid);
    log_debug("bounding_box: {}", bounding_box);

    // Mcast args
    auto in0_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores_in_rect_grid, INVALID);
    auto in0_mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores_in_rect_grid, INVALID);
    auto in0_mcast_sender_valid_semaphore_id = tt_metal::CreateSemaphore(program, all_cores_in_rect_grid, VALID);

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    CoreCoord top_left_core = {(std::size_t)start_core_x, (std::size_t)start_core_y};
    CoreCoord bottom_right_core = {
        (std::size_t)start_core_x + compute_with_storage_grid_size.x - 1,
        (std::size_t)start_core_y + compute_with_storage_grid_size.y - 1};
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    bool in0_is_dram = false;
    bool in1_is_dram = true;
    bool in3_is_dram = true;

    uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = detail::GetPreferredNOCForDRAMWrite(device->arch());
    tt_metal::NOC in1_noc = detail::GetPreferredNOCForDRAMRead(device->arch());

    CoreCoord start_core_noc = top_left_core_physical;
    CoreCoord end_core_noc = bottom_right_core_physical;
    if (in0_noc == NOC::NOC_1) {
        std::swap(start_core_noc, end_core_noc);
    }

    uint32_t num_blocks_per_shard = num_blocks / all_storage_cores_vec.size();
    log_debug("num_blocks_per_shard: {}", num_blocks_per_shard);
    if (per_core_M > 1) {
        TT_ASSERT(
            num_blocks_per_shard == 1,
            "currently not support per_core_M larger than 1, while split one shard into multiple blocks");
    }

    std::vector<uint32_t> in0_sender_compile_time_args = {
        (std::uint32_t)in0_block_num_tiles,                         // in0_block_num_tiles
        (std::uint32_t)in0_block_num_tiles * in0_single_tile_size,  // in0_block_size_bytes
        // in0 mcast args
        (std::uint32_t)in0_mcast_sender_semaphore_id,
        (std::uint32_t)in0_mcast_receiver_semaphore_id,
        (std::uint32_t)num_worker_cores,  // in0_mcast_num_dests
        (std::uint32_t)num_mcast_cores,   // in0_mcast_num_cores
        // block
        (std::uint32_t)num_blocks,
        // mcast noc coords
        (std::uint32_t)start_core_noc.x,
        (std::uint32_t)start_core_noc.y,
        (std::uint32_t)end_core_noc.x,
        (std::uint32_t)end_core_noc.y,
        // semahpre valid
        (std::uint32_t)in0_mcast_sender_valid_semaphore_id,
        //
        (std::uint32_t)num_blocks_per_shard};

    std::vector<uint32_t> in1_sender_writer_compile_time_args = {
        (std::uint32_t)in1_buffer_page_size,
        (std::uint32_t)in1_buffer_num_pages,
        // in1 block args
        (std::uint32_t)per_core_N_unpad,                // in1_block_w
        (std::uint32_t)per_core_N_unpad * in0_block_w,  // in1_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)num_blocks,                                    // num_blocks
        (std::uint32_t)out_block_tiles,                               // out_block_num_tiles
        (std::uint32_t)per_core_N * output_single_tile_size,          // out_tensor_stride_w_bytes
        (std::uint32_t)per_core_N_storage * output_single_tile_size,  // out_reshard_tensor_stride_w_bytes
        (std::uint32_t)per_core_M};
    if (bias_buffer != nullptr) {
        in1_sender_writer_compile_time_args.push_back(bias_buffer_page_size);
        in1_sender_writer_compile_time_args.push_back(bias_buffer_num_pages);
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)1);
    }

    std::map<string, string> mm_kernel_defines;
    std::map<string, string> mm_kernel_in0_sender_define;
    std::map<string, string> mm_kernel_in1_sender_writer_defines;
    if (bias_buffer != nullptr) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_sender_writer_defines["FUSE_BIAS"] = "1";
    }
    if (fused_activation.has_value()) {
        if (fused_activation.value().op_type == UnaryOpType::RELU) {
            mm_kernel_defines["PACK_RELU"] = "1";
        } else {
            using ttnn::operations::unary::utils::get_defines;
            mm_kernel_defines.merge(
                get_defines(fused_activation.value().op_type, fused_activation.value().params, "ACTIVATION", "i"));
        }
    }
    if (packer_l1_acc_en) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }
    mm_kernel_in1_sender_writer_defines["OUT_SHARDED"] = "1";
    mm_kernel_in1_sender_writer_defines["SKIP_MCAST"] = "1";

    if (skip_compute) {
        mm_kernel_defines["SKIP_COMPUTE"] = "1";
    }
    if (skip_in0_mcast) {
        mm_kernel_in0_sender_define["SKIP_MCAST"] = "1";
    }
    if (skip_write_back) {
        mm_kernel_in1_sender_writer_defines["SKIP_WRITE_BACK"] = "1";
    }
    mm_kernel_defines["MATMUL_DRAM_SHARDED"] = "1";

    auto mm_kernel_in0_sender_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp",
        all_cores_in_rect_grid,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
            .compile_args = in0_sender_compile_time_args,
            .defines = mm_kernel_in0_sender_define});

    auto mm_kernel_in1_sender_writer_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp",
        all_cores_in_rect_grid,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
            .compile_args = in1_sender_writer_compile_time_args,
            .defines = mm_kernel_in1_sender_writer_defines});

    // Compute kernel compile time args
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (per_core_N / out_subblock_w);
    uint32_t in1_per_core_w = per_core_N_unpad;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    vector<uint32_t> compute_kernel_args = {
        in0_block_w,             // in0_block_w
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_num_tiles,     // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles

        in1_num_subblocks,  // in1_num_subblocks
        in1_block_tiles,    // in1_block_num_tiles
        in1_per_core_w,     // in1_per_core_w

        num_blocks,  // num_blocks

        out_subblock_h,          // out_subblock_h
        out_subblock_w,          // out_subblock_w
        out_subblock_num_tiles,  // out_subblock_num_tiles
        B,                       // batch
        out_block_tiles,         // out_block_num_tiles

        untilize_out  // untilize_out
    };

    // Create compute kernel
    auto mm_kernel = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp",
        // all_worker_cores,
        all_cores_in_rect_grid,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = mm_kernel_defines});

    log_debug(LogOp, "in1_single_tile_size: {}", in1_single_tile_size);

    // Create circular buffers
    uint32_t src0_cb_index = 0;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size)
            .set_tile_dims(src0_cb_index, in0_tile);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores_in_rect_grid, src0_cb_config);
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src0_cb_index,
        in0_single_tile_size,
        in0_CB_size / in0_single_tile_size,
        in0_CB_size);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig src1_cb_config =
        tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
            .set_page_size(src1_cb_index, in1_single_tile_size)
            .set_tile_dims(src1_cb_index, in1_tile);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores_in_rect_grid, src1_cb_config);
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src1_cb_index,
        in1_single_tile_size,
        in1_CB_size / in1_single_tile_size,
        in1_CB_size);

    uint32_t src2_cb_index = 2;
    tt_metal::CircularBufferConfig src2_cb_config =
        tt_metal::CircularBufferConfig(in2_CB_size, {{src2_cb_index, in0_data_format}})
            .set_page_size(src2_cb_index, in0_single_tile_size)
            .set_tile_dims(src2_cb_index, in0_tile)
            .set_globally_allocated_address(*in0_buffer);
    auto cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores_in_rect_grid, src2_cb_config);
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src2_cb_index,
        in0_single_tile_size,
        in2_CB_size / in0_single_tile_size,
        in2_CB_size);

    uint32_t output_cb_index = 16;  // output operands start at index 16
    uint32_t interm0_cb_index = 24;
    tt_metal::CircularBufferConfig interm0_cb_config =
        tt_metal::CircularBufferConfig(0, {{interm0_cb_index, interm0_data_format}});
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(0, {{output_cb_index, output_data_format}});

    if ((interm0_data_format != output_data_format) || (untilize_out && (in1_num_subblocks > 1))) {
        // output
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
            {output_cb_index, output_data_format},
        };
        output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                               .set_page_size(output_cb_index, output_single_tile_size)
                               .set_tile_dims(output_cb_index, output_tile);
        // interm0
        std::map<uint8_t, tt::DataFormat> interm0_cb_data_format_spec{
            {interm0_cb_index, interm0_data_format},
        };
        interm0_cb_config = tt_metal::CircularBufferConfig(interm0_CB_size, interm0_cb_data_format_spec)
                                .set_page_size(interm0_cb_index, interm0_single_tile_size)
                                .set_tile_dims(interm0_cb_index, output_tile);

        auto cb_interm0 = tt_metal::CreateCircularBuffer(program, all_cores_in_rect_grid, interm0_cb_config);
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            interm0_cb_index,
            interm0_single_tile_size,
            interm0_CB_size / interm0_single_tile_size,
            interm0_CB_size);
    } else {
        log_debug(LogOp, "inplace interm and outout cb");
        // share buffer
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
            {output_cb_index, output_data_format}, {interm0_cb_index, interm0_data_format}};
        output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                               .set_page_size(output_cb_index, output_single_tile_size)
                               .set_page_size(interm0_cb_index, interm0_single_tile_size)
                               .set_tile_dims(output_cb_index, output_tile)
                               .set_tile_dims(interm0_cb_index, output_tile);
    }
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores_in_rect_grid, output_cb_config);
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        output_cb_index,
        output_single_tile_size,
        out_CB_size / output_single_tile_size,
        out_CB_size);

    // resharded output
    uint32_t output_reshard_cb_index = 17;
    std::map<uint8_t, tt::DataFormat> output_reshard_cb_data_format_spec{
        {output_reshard_cb_index, output_data_format},
    };
    tt_metal::CircularBufferConfig output_reshard_cb_config =
        tt_metal::CircularBufferConfig(out_reshard_CB_size, output_reshard_cb_data_format_spec)
            .set_page_size(output_reshard_cb_index, output_single_tile_size)
            .set_tile_dims(output_reshard_cb_index, output_tile);
    output_reshard_cb_config = output_reshard_cb_config.set_globally_allocated_address(*out_buffer);
    auto cb_output_reshard = tt_metal::CreateCircularBuffer(program, all_cores_in_rect_grid, output_reshard_cb_config);

    if (bias_buffer != nullptr) {
        uint32_t src3_cb_index = 3;
        tt_metal::CircularBufferConfig cb_src3_config =
            tt_metal::CircularBufferConfig(in3_CB_size, {{src3_cb_index, bias_data_format}})
                .set_page_size(src3_cb_index, bias_single_tile_size)
                .set_tile_dims(src3_cb_index, bias_tile);
        auto cb_src3 = tt_metal::CreateCircularBuffer(program, all_cores_in_rect_grid, cb_src3_config);
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            src3_cb_index,
            bias_single_tile_size,
            in3_CB_size / bias_single_tile_size,
            in3_CB_size);
    }

    // Parameters for last row, col, or block
    uint32_t last_block_h = M % per_core_M == 0 ? per_core_M : M % per_core_M;
    uint32_t last_block_w = N % per_core_N == 0 ? per_core_N : N % per_core_N;
    uint32_t last_block_num_nonzero_subblocks_h = (last_block_h - 1) / out_subblock_h + 1;
    uint32_t last_block_num_nonzero_subblocks_w = (last_block_w - 1) / out_subblock_w + 1;
    uint32_t last_subblock_of_last_block_h =
        last_block_h % out_subblock_h == 0 ? out_subblock_h : last_block_h % out_subblock_h;
    uint32_t last_subblock_of_last_block_w =
        last_block_w % out_subblock_w == 0 ? out_subblock_w : last_block_w % out_subblock_w;
    uint32_t last_block_padded_subblock_tiles_addr_skip =
        output_single_tile_size * (out_subblock_w - last_subblock_of_last_block_w);
    uint32_t last_block_padded_block_tiles_w_skip =
        (out_subblock_w * out_subblock_h) * (per_core_N / out_subblock_w - last_block_num_nonzero_subblocks_w);
    uint32_t last_block_padded_block_tiles_h_skip =
        (per_core_M / out_subblock_h - last_block_num_nonzero_subblocks_h) * (per_core_N * out_subblock_h);

    std::vector<KernelHandle> reader_kernel_ids;
    std::vector<KernelHandle> writer_kernel_ids;

    std::vector<uint32_t> in0_mcast_sender_noc_x;
    std::vector<uint32_t> in0_mcast_sender_noc_y;
    std::vector<CoreCoord> mcast_senders_coords = corerange_to_cores(mcast_senders);
    std::sort(mcast_senders_coords.begin(), mcast_senders_coords.end(), [](const CoreCoord& a, const CoreCoord& b) {
        if (a.y != b.y) {
            return a.y < b.y;
        }
        return a.x < b.x;
    });
    for (auto core : mcast_senders_coords) {
        in0_mcast_sender_noc_x.push_back((std::uint32_t)device->worker_core_from_logical_core(core).x);
    }
    for (auto core : mcast_senders_coords) {
        in0_mcast_sender_noc_y.push_back((std::uint32_t)device->worker_core_from_logical_core(core).y);
    }

    uint32_t sender_id = 0;
    for (auto core : mcast_senders_coords) {
        std::vector<uint32_t> mm_in0_sender_args;

        // mcast sender - 1, mcast sender + compute core - 2
        uint32_t worker_core_type;
        if (find(storage_worker_common.begin(), storage_worker_common.end(), core) != storage_worker_common.end()) {
            worker_core_type = 2;
        } else {
            worker_core_type = 1;
        }

        mm_in0_sender_args.push_back((std::uint32_t)worker_core_type);
        mm_in0_sender_args.push_back((std::uint32_t)sender_id);
        mm_in0_sender_args.insert(
            mm_in0_sender_args.end(), in0_mcast_sender_noc_x.begin(), in0_mcast_sender_noc_x.end());
        mm_in0_sender_args.insert(
            mm_in0_sender_args.end(), in0_mcast_sender_noc_y.begin(), in0_mcast_sender_noc_y.end());

        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_sender_id, core, mm_in0_sender_args);
        reader_kernel_ids.push_back(mm_kernel_in0_sender_id);

        sender_id++;
    }

    std::vector<CoreCoord> mcast_receiver_coords = corerange_to_cores(mcast_receivers);
    for (uint32_t i = 0; i < mcast_receiver_coords.size(); ++i) {
        auto core = mcast_receiver_coords[i];

        // in0 receivers rt args
        std::vector<uint32_t> mm_in0_receiver_args;
        // mcast receiver - 3
        uint32_t worker_core_type = 3;
        mm_in0_receiver_args.push_back((std::uint32_t)worker_core_type);
        mm_in0_receiver_args.push_back((std::uint32_t)0);
        mm_in0_receiver_args.insert(
            mm_in0_receiver_args.end(), in0_mcast_sender_noc_x.begin(), in0_mcast_sender_noc_x.end());
        mm_in0_receiver_args.insert(
            mm_in0_receiver_args.end(), in0_mcast_sender_noc_y.begin(), in0_mcast_sender_noc_y.end());

        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_sender_id, core, mm_in0_receiver_args);
        reader_kernel_ids.push_back(mm_kernel_in0_sender_id);
    }

    for (auto core : all_cores_in_rect_grid_vec) {
        if (std::find(mcast_senders_coords.begin(), mcast_senders_coords.end(), core) == mcast_senders_coords.end() and
            std::find(mcast_receiver_coords.begin(), mcast_receiver_coords.end(), core) ==
                mcast_receiver_coords.end()) {
            // in0 receivers rt args
            std::vector<uint32_t> mm_in0_idle_args;
            // idle core - 0
            uint32_t worker_core_type = 0;
            mm_in0_idle_args.push_back((std::uint32_t)worker_core_type);

            tt_metal::SetRuntimeArgs(program, mm_kernel_in0_sender_id, core, mm_in0_idle_args);
        }
    }

    uint32_t bank_id = 0;
    std::vector<uint32_t> bank_ids;
    uint32_t curr_storage_core_idx = 0;
    uint32_t per_core_N_storage_curr_stride = 0;

    uint32_t worker_core_stride = 0;
    uint32_t storage_core_stride = 0;
    uint32_t curr_worker_core = 0;
    uint32_t curr_storage_core = 0;

    // for all the cores in the rect grid, we send one rt arg to determine if they are worker core
    for (uint32_t i = 0; i < all_cores_in_rect_grid_vec.size(); ++i) {
        auto core = all_cores_in_rect_grid_vec[i];

        if (all_worker_cores.ranges().find(core) == all_worker_cores.ranges().end()) {  // not worker
            // in1 reader rt args
            bool is_worker_core = false;
            std::vector<uint32_t> mm_in1_sender_writer_args;
            mm_in1_sender_writer_args.push_back((std::uint32_t)is_worker_core);

            tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_in1_sender_writer_args);

            // compute rt args
            std::vector<uint32_t> mm_compute_args;
            mm_compute_args.push_back((std::uint32_t)is_worker_core);

            tt_metal::SetRuntimeArgs(program, mm_kernel, core, mm_compute_args);
        } else {
            // compute rt args
            bool is_worker_core = true;
            std::vector<uint32_t> mm_compute_args;
            mm_compute_args.push_back((std::uint32_t)is_worker_core);

            tt_metal::SetRuntimeArgs(program, mm_kernel, core, mm_compute_args);
        }
    }

    for (uint32_t i = 0; i < all_worker_cores_ordered.size(); ++i) {
        auto core = all_worker_cores_ordered[i];

        // in1 reader rt args
        bool is_worker_core = true;
        std::vector<uint32_t> mm_in1_sender_writer_args;
        mm_in1_sender_writer_args.push_back((std::uint32_t)is_worker_core);
        mm_in1_sender_writer_args.push_back(in1_buffer->address());
        if (bias_buffer != nullptr) {
            mm_in1_sender_writer_args.push_back(bias_buffer->address());
        } else {
            mm_in1_sender_writer_args.push_back(0);
        }

        uint32_t vc = bank_id & 0x3;
        bank_ids.push_back(bank_id);
        for (uint32_t j = 0; j < i; ++j) {
            auto core_prev = all_worker_cores_ordered[j];

            if (core_prev.y == core.y and ((bank_id & 0x3) == (bank_ids[j] & 0x3))) {  // same vc and same row
                vc = (vc + 1) & 0x3;
                break;
            }
        }
        mm_in1_sender_writer_args.push_back((std::uint32_t)bank_id);
        mm_in1_sender_writer_args.push_back((std::uint32_t)vc);

        bank_id = (bank_id + 1) % num_dram_banks;

        if (per_core_N_unpad < per_core_N_storage) {
            if (curr_storage_core_idx < all_storage_cores_vec.size()) {
                uint32_t remaining_per_core_N_storage = (per_core_N_storage - per_core_N_storage_curr_stride);
                uint32_t per_core_N_reshard_1 =
                    (remaining_per_core_N_storage > per_core_N_unpad) ? per_core_N_unpad : remaining_per_core_N_storage;
                uint32_t per_core_N_reshard_2 = per_core_N_unpad - per_core_N_reshard_1;

                if (per_core_N_reshard_2 != 0 and (curr_storage_core_idx + 1) < all_storage_cores_vec.size()) {
                    mm_in1_sender_writer_args.push_back(2);
                } else {
                    mm_in1_sender_writer_args.push_back(1);
                }

                mm_in1_sender_writer_args.push_back(
                    per_core_N_storage_curr_stride * output_single_tile_size);  // reshard_tensor_start_offset
                mm_in1_sender_writer_args.push_back(
                    per_core_N_reshard_1 * output_single_tile_size);  // per_core_N_reshard_bytes_1
                mm_in1_sender_writer_args.push_back(
                    in0_mcast_sender_noc_x[curr_storage_core_idx]);  // in0_mcast_sender_noc_x
                mm_in1_sender_writer_args.push_back(
                    in0_mcast_sender_noc_y[curr_storage_core_idx]);  // in0_mcast_sender_noc_y

                if (per_core_N_reshard_2 != 0 and (curr_storage_core_idx + 1) < all_storage_cores_vec.size()) {
                    mm_in1_sender_writer_args.push_back(
                        per_core_N_reshard_2 * output_single_tile_size);  // per_core_N_reshard_bytes_2
                    mm_in1_sender_writer_args.push_back(
                        in0_mcast_sender_noc_x[curr_storage_core_idx + 1]);  // in0_mcast_sender_noc_x
                    mm_in1_sender_writer_args.push_back(
                        in0_mcast_sender_noc_y[curr_storage_core_idx + 1]);  // in0_mcast_sender_noc_y
                }

                curr_storage_core_idx += (per_core_N_storage_curr_stride + per_core_N_unpad) / per_core_N_storage;
                per_core_N_storage_curr_stride =
                    (per_core_N_storage_curr_stride + per_core_N_unpad) % per_core_N_storage;
            }
        } else {
            uint32_t num_iter = 0;

            if (curr_storage_core < all_storage_cores_vec.size()) {
                num_iter++;

                log_debug(
                    "curr worker core: {}, send back to storage core: {}, coord: {}",
                    curr_worker_core,
                    curr_storage_core,
                    mcast_senders_coords[curr_storage_core]);

                worker_core_stride = per_core_N_storage - storage_core_stride;

                mm_in1_sender_writer_args.push_back(
                    storage_core_stride * output_single_tile_size);  // reshard_tensor_start_offset
                mm_in1_sender_writer_args.push_back(
                    worker_core_stride * output_single_tile_size);  // per_core_N_reshard
                mm_in1_sender_writer_args.push_back(
                    in0_mcast_sender_noc_x[curr_storage_core]);  // in0_mcast_sender_noc_x
                mm_in1_sender_writer_args.push_back(
                    in0_mcast_sender_noc_y[curr_storage_core]);  // in0_mcast_sender_noc_y

                curr_storage_core += (storage_core_stride + worker_core_stride) / per_core_N_storage;
                storage_core_stride = (storage_core_stride + worker_core_stride) % per_core_N_storage;

                if (worker_core_stride >= per_core_N_unpad) {
                    curr_worker_core += 1;
                }

                while (curr_worker_core <= i and curr_storage_core < all_storage_cores_vec.size()) {
                    num_iter++;

                    log_debug(
                        "curr worker core: {}, send back to storage core: {}, coord: {}",
                        curr_worker_core,
                        curr_storage_core,
                        mcast_senders_coords[curr_storage_core]);

                    uint32_t stride = worker_core_stride + per_core_N_storage;
                    if (stride >= per_core_N_unpad) {
                        stride = per_core_N_unpad;
                        curr_worker_core += 1;
                    }

                    mm_in1_sender_writer_args.push_back(
                        (stride - worker_core_stride) * output_single_tile_size);  // per_core_N_reshard
                    mm_in1_sender_writer_args.push_back(
                        in0_mcast_sender_noc_x[curr_storage_core]);  // in0_mcast_sender_noc_x
                    mm_in1_sender_writer_args.push_back(
                        in0_mcast_sender_noc_y[curr_storage_core]);  // in0_mcast_sender_noc_y

                    storage_core_stride = (stride - worker_core_stride) % per_core_N_storage;
                    curr_storage_core += (stride - worker_core_stride) / per_core_N_storage;
                    worker_core_stride = stride;
                }
            }

            mm_in1_sender_writer_args.insert(mm_in1_sender_writer_args.begin() + 5, num_iter);
        }

        tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_in1_sender_writer_args);
        writer_kernel_ids.push_back(mm_kernel_in1_sender_writer_id);
    }

    auto override_runtime_arguments_callback =
        [writer_kernel_ids, all_worker_cores_ordered, cb_src2, cb_output_reshard](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            TT_FATAL(input_tensors.size() + optional_input_tensors.size() == 3, "Error");
            TT_FATAL(output_tensors.size() == 1, "Error");

            auto src_buffer_a = input_tensors.at(0).buffer();
            auto src_buffer_b = input_tensors.at(1).buffer();
            auto bias_tensor = optional_input_tensors.at(0);

            auto dst_buffer = output_tensors.at(0).buffer();

            UpdateDynamicCircularBufferAddress(program, cb_src2, *src_buffer_a);
            UpdateDynamicCircularBufferAddress(program, cb_output_reshard, *dst_buffer);

            for (uint32_t i = 0; i < all_worker_cores_ordered.size(); ++i) {
                auto core = all_worker_cores_ordered[i];
                auto writer_kernel_id = writer_kernel_ids[i];
                auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                writer_runtime_args[1] = src_buffer_b->address();
                if (bias_tensor.has_value()) {
                    writer_runtime_args[2] = bias_tensor.value().buffer()->address();
                } else {
                    writer_runtime_args[2] = 0;
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}
}  // namespace reuse_dram_sharded_optimized_helpers

namespace ttnn {

namespace operations {

namespace matmul {

operation::ProgramWithCallbacks matmul_multi_core_reuse_dram_sharded_optimized_(
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor> bias,
    Tensor& output,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    bool untilize_out,
    bool skip_compute,
    bool skip_in0_mcast,
    bool skip_write_back) {
    const auto &ashape = a.get_legacy_shape(), bshape = b.get_legacy_shape();
    auto in0_tile = a.get_tile();
    auto in1_tile = b.get_tile();
    // cannot use the output tensor tile directly as that might be changed by user override
    auto output_tile = tt::tt_metal::Tile({in0_tile.get_tile_shape()[0], in1_tile.get_tile_shape()[1]});
    auto in0_tile_shape = a.get_tile().get_tile_shape();
    auto in1_tile_shape = b.get_tile().get_tile_shape();

    // CB dataformats
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());          // in0
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.get_dtype());          // in1
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());  // output

    tt_metal::Buffer* bias_buffer = nullptr;
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;  // bias; doesn't matter if bias=nullptr
    if (bias.has_value()) {
        auto& c = bias.value();
        TT_FATAL(c.storage_type() == StorageType::DEVICE, "Error");
        TT_FATAL(a.device() == c.device(), "Operands to matmul need to be on the same device!");
        TT_FATAL(c.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");

        bias_buffer = c.buffer();

        bias_data_format = tt_metal::datatype_to_dataformat_converter(c.get_dtype());
    }

    tt::tt_metal::Device* device = a.device();

    TT_FATAL(a.shard_spec().has_value() && output.shard_spec().has_value(), "Error");
    CoreRangeSet all_cores_storage = a.shard_spec().value().grid;

    uint32_t in0_single_tile_size = a.get_tile().get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = b.get_tile().get_tile_size(in1_data_format);
    tt_metal::Buffer* in0_buffer = a.buffer();
    tt_metal::Buffer* in1_buffer = b.buffer();
    TT_FATAL(in0_buffer->size() % in0_single_tile_size == 0, "Error");
    TT_FATAL(in1_buffer->size() % in1_single_tile_size == 0, "Error");

    TT_FATAL(
        ashape[-1] == bshape[-2],
        "Dimension K (A.shape[-1] and B.shape[-2]) must match for A and B in bmm_op");  // A.K == B.K
    TT_FATAL(ashape[-2] % in0_tile_shape[0] == 0, "Error");
    TT_FATAL(ashape[-1] % in0_tile_shape[1] == 0, "Error");
    TT_FATAL(bshape[-2] % in1_tile_shape[0] == 0, "Error");
    TT_FATAL(bshape[-1] % in1_tile_shape[1] == 0, "Error");

    MathFidelity math_fidelity;
    bool math_approx_mode;
    bool fp32_dest_acc_en;
    bool packer_l1_acc;

    std::visit(
        [&](auto&& compute_kernel_config) {
            using T = std::decay_t<decltype(compute_kernel_config)>;
            if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
                TT_FATAL(device->arch() == ARCH::GRAYSKULL, "kernel config is not for graykull");
                math_fidelity = compute_kernel_config.math_fidelity;
                math_approx_mode = compute_kernel_config.math_approx_mode;
                fp32_dest_acc_en = false;
                packer_l1_acc = false;
            } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
                TT_FATAL(ttnn::device::is_wormhole_or_blackhole(device->arch()), "kernel config is not for wormhole_b0 or blackhole");
                math_fidelity = compute_kernel_config.math_fidelity;
                math_approx_mode = compute_kernel_config.math_approx_mode;
                fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en;
                packer_l1_acc = compute_kernel_config.packer_l1_acc;
            } else {
                TT_THROW("arch not supported");
            }
        },
        compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                      Matmul Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // NOTE: Pads matmul input dims to 512 x 512 multiples (ie. multiples of 16*32 x 16*32)
    // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])
    uint32_t B = 1;
    uint32_t Mt = get_batch_size(ashape) * ashape[-2] / in0_tile_shape[0];
    uint32_t Kt = ashape[-1] / in0_tile_shape[1];
    uint32_t Nt = bshape[-1] / in1_tile_shape[1];

    TT_FATAL(Kt % in0_block_w == 0, "Error");

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Buffer* out_buffer = output.buffer();
    TT_FATAL(out_buffer != nullptr, "Output buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    return reuse_dram_sharded_optimized_helpers::create_program_dram_sharded(
        device,
        all_cores_storage,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode,
        packer_l1_acc,
        B,
        Mt,
        Nt,
        Kt,
        in0_block_w,
        per_core_M,
        per_core_N,
        fused_activation,
        in0_buffer,
        in1_buffer,
        bias_buffer,
        out_buffer,
        in0_tile,
        in1_tile,
        bias.has_value() ? bias->get_tile() : output_tile,
        output_tile,
        in0_data_format,
        in1_data_format,
        bias_data_format,
        output_data_format,
        untilize_out,
        skip_compute,
        skip_in0_mcast,
        skip_write_back);
}

operation::ProgramWithCallbacks matmul_multi_core_reuse_dram_sharded_optimized(
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor> bias,
    Tensor& output_tensor,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    bool untilize_out,
    bool skip_compute,
    bool skip_in0_mcast,
    bool skip_write_back) {
    return matmul_multi_core_reuse_dram_sharded_optimized_(
        a,
        b,
        bias,
        output_tensor,
        compute_kernel_config,
        in0_block_w,
        per_core_M,
        per_core_N,
        fused_activation,
        untilize_out,
        skip_compute,
        skip_in0_mcast,
        skip_write_back);
}

}  // namespace matmul

}  // namespace operations

}  // namespace ttnn
