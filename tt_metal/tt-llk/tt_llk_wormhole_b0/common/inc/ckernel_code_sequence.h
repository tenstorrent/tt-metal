// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "lltt.h"

// Code sequences: driving an execution unit from a table instead of from branches.
//
// When an op picks a different instruction per work item from runtime data, branching on that data puts
// the RISC in the critical path. Instead, name each instruction with one character and record the
// resulting "code sequence" into the replay buffer at init (@ref sequence::load). A compile-time table
// then maps runtime data to a replay handle: an encoded REPLAY instruction naming a (start, length) run
// of that buffer (@ref make_table), which the op pushes at runtime.
//
// Every fragment a table needs must appear in the sequence as a contiguous substring, so the sequence is
// an overlap-packing of the required fragment set and has to fit its slot in the replay buffer.
//
// Letter case is the skip convention: lower case skips a work item, upper case does it, and a has-work
// mask picks one item pattern's fragment out of one template, e.g. case_encode("ni", 0b10) resolves "nI".

namespace ckernel
{
namespace code_seq
{

/**
 * @brief A compile-time code sequence bound to its slot in the replay buffer.
 *
 * Declare via @ref make_sequence as a namespace-scope `static constexpr auto`: @ref make_table takes the
 * sequence as a template argument, so it must be a constant expression with linkage.
 *
 * @tparam CodeLen: Number of instructions in the sequence, deduced from the literal.
 */
template <std::size_t CodeLen>
class sequence
{
public:
    /**
     * @brief Construct from a code-sequence literal; prefer @ref make_sequence, which deduces CodeLen.
     */
    constexpr sequence(const char (&codes)[CodeLen + 1], const std::uint32_t replay_base) : m_codes(codes), m_replay_base(replay_base)
    {
    }

    /**
     * @brief Resolve an instruction fragment to a replay handle over this sequence.
     *
     * @param needle: The fragment to replay, as characters of the sequence.
     * @note Returns 0 when unrepresentable; lltt::replay_insn always sets an opcode bit, so 0 is never a
     *       real handle. @ref make_table rejects those, a standalone handle needs its own static_assert.
     */
    constexpr std::uint32_t fragment(const char* needle) const
    {
        std::uint32_t needle_len = 0;
        while (needle[needle_len] != '\0')
        {
            ++needle_len;
        }
        if (needle_len == 0)
        {
            return 0; // a zero-length replay is not an instruction, and would encode as a real handle
        }
        for (std::uint32_t i = 0; i + needle_len <= CodeLen; ++i)
        {
            std::uint32_t k = 0;
            for (; k < needle_len; ++k)
            {
                if (m_codes[i + k] != needle[k])
                {
                    break;
                }
            }
            if (k == needle_len)
            {
                return lltt::replay_insn(m_replay_base + i, needle_len);
            }
        }
        return 0; // not a substring of this sequence
    }

    /**
     * @brief Resolve the fragment a has-work mask selects out of a lower-case template.
     *
     * @param tmpl: All-lower-case template, one character per work item.
     * @param mask: Bit i set upper-cases character i, e.g. case_encode("ni", 0b10) resolves "nI".
     */
    template <std::size_t N>
    constexpr std::uint32_t case_encode(const char (&tmpl)[N], const std::uint32_t mask) const
    {
        std::array<char, N> encoded {};
        std::size_t n = 0;
        for (; tmpl[n] != '\0'; ++n)
        {
            encoded[n] = ((mask >> n) & 1u) ? static_cast<char>(tmpl[n] - ('a' - 'A')) : tmpl[n];
        }
        encoded[n] = '\0';
        return fragment(encoded.data());
    }

    /**
     * @brief Record the sequence into the replay buffer, one instruction per character.
     *
     * @tparam Emit: Callable taking one code character and issuing that character's instruction.
     * @param emit: The per-character instruction emitter, called once per character in order.
     * @note Emit must issue exactly ONE instruction per character, or the recorded length will not match
     *       the handles @ref fragment hands out.
     */
    template <typename Emit>
    [[gnu::always_inline, gnu::flatten]] inline void load(Emit&& emit) const
    {
        const char* codes = m_codes;
        auto emit_all     = [codes, &emit]
        {
            auto step = [&](auto self, auto idx) -> void
            {
                constexpr std::size_t i = decltype(idx)::value;
                if constexpr (i < CodeLen)
                {
                    emit(codes[i]);
                    self(self, std::integral_constant<std::size_t, i + 1> {});
                }
            };
            step(step, std::integral_constant<std::size_t, 0> {});
        };

        // The only arch-divergent code in this file: Blackhole disables instruction gathering around the
        // record window via ckernel::load_replay_buf; Wormhole has no gathering and records directly.
        lltt::record(m_replay_base, CodeLen);
        emit_all();
    }

private:
    const char* m_codes;
    std::uint32_t m_replay_base;
};

namespace detail
{

template <std::size_t N>
constexpr bool all_resolved(const std::array<std::uint32_t, N>& table)
{
    for (const std::uint32_t entry : table)
    {
        if (entry == 0)
        {
            return false;
        }
    }
    return true;
}

template <const auto& Seq, std::size_t N, typename Entry>
constexpr std::array<std::uint32_t, N> build_table(Entry&& entry)
{
    const auto find        = [](const char* needle) { return Seq.fragment(needle); };
    const auto case_encode = [](const auto& tmpl, const std::uint32_t mask) { return Seq.case_encode(tmpl, mask); };

    std::array<std::uint32_t, N> table {};
    for (std::uint32_t i = 0; i < N; ++i)
    {
        if constexpr (std::is_invocable_v<Entry&, std::uint32_t, decltype(find), decltype(case_encode)>)
        {
            table[i] = entry(i, find, case_encode);
        }
        else if constexpr (std::is_invocable_v<Entry&, std::uint32_t, decltype(find)>)
        {
            table[i] = entry(i, find);
        }
        else
        {
            table[i] = entry(i);
        }
    }
    return table;
}

} // namespace detail

/**
 * @brief Bind a code-sequence literal to its slot in the replay buffer.
 *
 * A function rather than a constructor: class template argument deduction is all-or-nothing, so only a
 * function template can take MaxLen explicitly and still deduce CodeLen from the literal.
 *
 * @tparam ReplayBase: Offset of the sequence within the replay buffer.
 * @tparam MaxLen: Instructions the sequence may occupy. The caller states it, since splitting the replay
 *                 buffer between users is a per-thread convention (ckernel::math::replay_buf_offset).
 * @param codes: One character per instruction, NUL-terminated.
 */
template <std::uint32_t ReplayBase, std::size_t MaxLen, std::size_t N>
constexpr sequence<N - 1> make_sequence(const char (&codes)[N])
{
    static_assert(N - 1 <= MaxLen, "code sequence is longer than its slot in the replay buffer");
    static_assert(ReplayBase + MaxLen <= REPLAY_BUF_SIZE, "code sequence slot does not fit the replay buffer");
    return {codes, ReplayBase};
}

/**
 * @brief Build a runtime-index -> instruction decode table over a code sequence.
 *
 * The entry builder gets @ref sequence::fragment and @ref sequence::case_encode bound to Seq, so a table
 * body names fragments directly: find("N"), case_encode("ni", mask).
 *
 * @tparam Seq: The @ref sequence to resolve against. A template argument rather than a parameter, so the
 *              built table is a constant expression here and the check below can be a static_assert.
 * @tparam N: Table size; the index handed to entry runs over [0, N).
 * @tparam Entry: Callable returning the instruction for one index, invoked as entry(index),
 *                entry(index, find) or entry(index, find, case_encode) -- whichever it accepts, and
 *                stateless, which a namespace-scope lambda always is.
 */
template <const auto& Seq, std::size_t N, typename Entry>
constexpr std::array<std::uint32_t, N> make_table(Entry&& entry)
{
    constexpr auto table = detail::build_table<Seq, N>(entry);
    static_assert(detail::all_resolved(table), "code sequence: a table fragment is not a substring of the sequence");
    return table;
}

} // namespace code_seq
} // namespace ckernel
