// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file blocked_range.h
 * @brief A range object to iterate over a sequence in blocks
 *
 * @details The BlockedRange class allows for iterating over a
 * sequence of objects in blocks. For example, one can iterate
 * over an array of 10 objects with a block size of 3, which will
 * result in 3 blocks of 3 and 1 block of 1. For each block, one
 * can also iterate within the block.
 */

#pragma once

#include <algorithm>

namespace norm::kernel_util::generic {

template <typename SizeT = uint32_t>
class BlockedRange {
public:
    class Block {
    public:
        class Iterator {
        public:
            Iterator(SizeT val) : val_(val) {}
            SizeT operator*() const { return val_; }
            Iterator& operator++() {
                ++val_;
                return *this;
            }
            bool operator!=(const Iterator& other) const { return val_ != other.val_; }

        private:
            SizeT val_;
        };

        class GlobalRange {
        public:
            GlobalRange(SizeT start, SizeT end) : start_(start), end_(end) {}
            Iterator begin() const { return Iterator(start_); }
            Iterator end() const { return Iterator(end_); }
            SizeT size() const { return end_ - start_; }

        private:
            SizeT start_;
            SizeT end_;
        };

        class LocalRange {
        public:
            LocalRange(SizeT size) : size_(size) {}
            [[nodiscard]] Iterator begin() const { return Iterator(SizeT(0)); }
            [[nodiscard]] Iterator end() const { return Iterator(size_); }
            [[nodiscard]] SizeT size() const { return size_; }

        private:
            SizeT size_;
        };

        Block(SizeT start, SizeT end, SizeT block_size) : start_(start), end_(end), block_size_(block_size) {}

        [[nodiscard]] SizeT size() const { return end_ - start_; }
        [[nodiscard]] bool is_full() const { return size() == block_size_; }
        [[nodiscard]] bool is_partial() const { return size() < block_size_; }
        [[nodiscard]] bool is_first() const { return start_ == 0; }
        [[nodiscard]] SizeT start() const { return start_; }
        [[nodiscard]] SizeT last() const { return end_; }
        [[nodiscard]] SizeT remainder() const { return block_size_ - size(); }
        [[nodiscard]] SizeT full_block_size() const { return block_size_; }

        [[nodiscard]] GlobalRange global() const { return GlobalRange{start_, end_}; }
        [[nodiscard]] LocalRange local() const { return LocalRange{size()}; }

        [[nodiscard]] SizeT to_global(SizeT local) const { return start_ + local; }

        // Default iteration is global
        [[nodiscard]] Iterator begin() const { return Iterator(start_); }
        [[nodiscard]] Iterator end() const { return Iterator(end_); }

    private:
        SizeT start_;
        SizeT end_;
        SizeT block_size_;
    };

    class Iterator {
    public:
        Iterator(SizeT pos, SizeT block_size, SizeT total) : pos_(pos), block_size_(block_size), total_(total) {}

        Block operator*() const { return Block{pos_, std::min(pos_ + block_size_, total_), block_size_}; }

        Iterator& operator++() {
            pos_ += block_size_;
            return *this;
        }

        bool operator!=(const Iterator& other) const { return pos_ < other.pos_; }

    private:
        SizeT pos_;
        SizeT block_size_;
        SizeT total_;
    };

    BlockedRange(SizeT total, SizeT block_size) : total_(total), block_size_(block_size) {}

    [[nodiscard]] Iterator begin() const { return Iterator(0, block_size_, total_); }
    [[nodiscard]] Iterator end() const { return Iterator(total_, block_size_, total_); }
    [[nodiscard]] SizeT num_blocks() const { return (total_ + block_size_ - 1) / block_size_; }
    [[nodiscard]] SizeT block_size() const { return block_size_; }
    [[nodiscard]] Block front() const { return Block(0, std::min(block_size_, total_), block_size_); }
    [[nodiscard]] Block back() const {
        SizeT last_start = (num_blocks() - 1) * block_size_;
        return Block(last_start, total_, block_size_);
    }
    [[nodiscard]] SizeT total_with_remainder() const { return total_ + back().remainder(); }

private:
    SizeT total_;
    SizeT block_size_;
};

/**
 * @brief Convenience function to create a BlockedRange object
 */
template <typename SizeT = uint32_t>
inline BlockedRange<SizeT> blocks(SizeT total, SizeT block_size) {
    return BlockedRange<SizeT>(total, block_size);
}

}  // namespace norm::kernel_util::generic
