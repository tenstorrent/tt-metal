// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace ttnn::operations::experimental::high_bw_all_gather::scheduler {

// These helpers assume interleaved DRAM page p maps to allocator bank
// p % num_dram_banks at bank-local page offset p / num_dram_banks.
struct BankOwnedSlice {
    uint32_t bank;
    uint32_t input_page_start;
    uint32_t page_count;
};

// Assign the flat (link, worker) set round-robin across logical DRAM banks.
// When workers do not divide evenly across banks, a bank may be shared by
// multiple links; each worker still receives one disjoint, contiguous range
// within exactly one bank.
inline BankOwnedSlice derive_bank_owned_slice(
    uint32_t num_input_pages,
    uint32_t num_links,
    uint32_t workers_per_direction,
    uint32_t num_dram_banks,
    uint32_t link,
    uint32_t worker) {
    if (num_links == 0 || workers_per_direction == 0 || num_dram_banks == 0 || link >= num_links ||
        worker >= workers_per_direction) {
        return {};
    }
    const uint32_t total_workers = num_links * workers_per_direction;
    if (total_workers < num_dram_banks) {
        return {};
    }
    uint32_t bank;
    uint32_t worker_in_bank;
    uint32_t workers_for_bank;
    if (num_dram_banks % num_links == 0 && workers_per_direction % (num_dram_banks / num_links) == 0) {
        // Preserve the original placement exactly for evenly divisible devices:
        // each link owns interleaved banks and adjacent workers split a bank.
        const uint32_t banks_per_link = num_dram_banks / num_links;
        workers_for_bank = workers_per_direction / banks_per_link;
        bank = link + (worker / workers_for_bank) * num_links;
        worker_in_bank = worker % workers_for_bank;
    } else {
        // Interleave links in the flat order so harvested configurations retain
        // the same link-to-bank pattern as closely as possible.
        const uint32_t global_worker = worker * num_links + link;
        bank = global_worker % num_dram_banks;
        worker_in_bank = global_worker / num_dram_banks;
        workers_for_bank = total_workers <= bank ? 0 : 1 + (total_workers - 1 - bank) / num_dram_banks;
    }

    // Interleaved logical page IDs owned by `bank` are:
    //   bank, bank + num_dram_banks, bank + 2 * num_dram_banks, ...
    // Divide that bank-local sequence evenly between its workers. The first
    // `remainder` workers receive one extra page, so arbitrary tensor sizes
    // remain bank-owned without dropping or duplicating a tail.
    const uint32_t bank_page_count = num_input_pages <= bank ? 0 : 1 + (num_input_pages - 1 - bank) / num_dram_banks;
    const uint32_t pages_per_worker = bank_page_count / workers_for_bank;
    const uint32_t remainder = bank_page_count % workers_for_bank;
    const uint32_t worker_page_offset = worker_in_bank * pages_per_worker + std::min(worker_in_bank, remainder);
    const uint32_t worker_page_count = pages_per_worker + (worker_in_bank < remainder ? 1u : 0u);

    return {
        bank,
        bank + worker_page_offset * num_dram_banks,
        worker_page_count,
    };
}

inline bool can_partition_workers_by_bank(
    uint32_t num_input_pages, uint32_t num_links, uint32_t workers_per_direction, uint32_t num_dram_banks) {
    if (num_links == 0 || workers_per_direction == 0 || num_dram_banks == 0) {
        return false;
    }
    const uint32_t total_workers = num_links * workers_per_direction;
    return total_workers >= num_dram_banks && num_input_pages >= total_workers;
}

inline uint32_t workers_per_direction_to_cover_banks(uint32_t num_links, uint32_t num_dram_banks) {
    if (num_links == 0) {
        return 0;
    }
    return (num_dram_banks + num_links - 1) / num_links;
}

inline bool worker_count_fits(uint32_t workers_per_direction, uint32_t num_links, uint32_t available_worker_cores) {
    if (workers_per_direction == 0 || num_links == 0) {
        return false;
    }
    const uint32_t cores_per_direction = workers_per_direction + (workers_per_direction > 1 ? 1u : 0u);
    return num_links * 2 * cores_per_direction <= available_worker_cores;
}

template <std::size_t NumTiers>
inline uint32_t select_fitting_worker_count(
    uint32_t preferred_workers_per_direction,
    uint32_t num_links,
    uint32_t available_worker_cores,
    const std::array<uint32_t, NumTiers>& descending_tiers) {
    for (const uint32_t candidate : descending_tiers) {
        if (candidate <= preferred_workers_per_direction &&
            worker_count_fits(candidate, num_links, available_worker_cores)) {
            return candidate;
        }
    }
    // Let the caller's capacity check report the minimum one-worker failure.
    return 1;
}

}  // namespace ttnn::operations::experimental::high_bw_all_gather::scheduler
