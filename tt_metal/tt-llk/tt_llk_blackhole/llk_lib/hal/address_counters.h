// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h" // ckernel::instrn_buffer

namespace hal
{
/**
 * Enum class for address counter client selection
 */
enum class AddressCounterClient : std::uint8_t
{
    Unpacker0 = 0x1,
    Unpacker1 = 0x2,
    Packers   = 0x4
};

/**
 * @brief Selects one of the two hardware channels within an X/Y/Z/W address counter.
 *
 * Channel meaning depends on the @ref AddressClient and the counter (X/Y/Z/W):
 *
 * Unpackers (Unpacker0/Unpacker1 each own their X/Y/Z/W):
 *   | Dim     | Channel0                                             | Channel1                                                                 |
 *   |---------|------------------------------------------------------|--------------------------------------------------------------------------|
 *   |    X    | Input address gen (L1 address + part of datum count) | Input address gen (part of datum count)                                  |
 *   | Y, Z, W | Decompressor row seek within L1                      | Output address gen (Dst/SrcA address for Unacper0, or SrcB on Unpacker1) |
 *
 * Packers (all four packers share one X/Y/Z/W):
 *   | Dim     | Channel0                                              | Channel1                                |
 *   |---------|-------------------------------------------------------|-----------------------------------------|
 *   |    X    | Input address gen (Dst address + part of datum count) | Input address gen (part of datum count) |
 *   | Y, Z, W | Input address gen (Dst address)                       | Output address gen (L1 address)         |
 *
 * @note Z/W split the same way as X/Y
 */
enum class AddressChannel : std::uint8_t
{
    Channel0 = 0x1,
    Channel1 = 0x2
};

/**
 * @brief Selects which type of get_operation() method is returning
 */
enum class GetOpType : std::uint8_t
{
    SETTER    = 0x0,
    INCREMENT = 0x1
};

/**
 * @brief Compile-time builder for the Tensix address counters (SETADC / INCADC family).
 *
 * Drive it as a fluent chain starting from the global @ref address_counters instance. Each
 * selector/setter returns a *new* ADC type with its choice folded into the template parameters,
 * so the whole description is assembled at compile time with no runtime state:
 *
 *   1. client<...>()      — one or more @ref AddressCounterClient (Unpacker0/Unpacker1/Packers).
 *   2. channel<Channel>() — select a single @ref AddressChannel to configure next.
 *   3. X/Y/Z/W<value>()   — assign values to counters for the currently selected channel.
 *
 * A given X/Y/Z/W applies only to the channel selected in step 2. To program both channels,
 * finish one channel's counters, then select the other channel and set its counters:
 *   channel<Channel0>() → X/Y/Z/W ... → channel<Channel1>() → X/Y/Z/W ...
 *
 * Then terminate with an emitter:
 *   - apply()                      — emit the minimal SETADC* instruction sequence that programs the
 *                                    assigned counters for the selected client(s)/channel(s).
 *   - increment()                  — emit the matching INCADC* sequence instead of absolute SETs.
 *   - get_operation<GetOpType>()   — return a single encoded instruction word (e.g. to embed in a MOP
 *                                    or replay buffer) rather than issuing it inline.
 *
 * @code
 *   // Program Unpacker0's X and Y counters on both channels, one channel at a time:
 *   address_counters
 *       .client<AddressCounterClient::Unpacker0>()
 *       .channel<AddressChannel::Channel0>()
 *       .X<face_r_dim>()
 *       .Y<0>()
 *       .channel<AddressChannel::Channel1>()
 *       .X<face_r_dim>()
 *       .Y<0>()
 *       .apply();
 * @endcode
 *
 * @tparam SetMask     compile-time bitmask recording which per-channel counter slots have been assigned.
 *                     Each counter owns a 2-bit field (one bit per channel); a bit is set when X/Y/Z/W<value>()
 *                     wrote that channel's slot.
 * @tparam X/X1, Y/Y1, Z/Z1, W/W1  Per-counter Channel0/Channel1 values; X/Y/Z/W<value>() writes only the
 *                     currently selected channel (the one from the most recent channel<...>()).
 * @tparam ClientMask  compile-time bitmask of the selected @ref AddressCounterClient(s).
 * @tparam ChannelMask compile-time bitmask of every selected @ref AddressChannel (cumulative).
 * @tparam CurrentChannel  the single channel most recently selected; X/Y/Z/W<value>() targets it.
 *
 * @note At least one client and one channel must be selected before any emitter is called
 *       (enforced by static_assert). The emitter chooses the fewest SETADC / INCADC instructions
 *       that cover the assigned counters for the chosen client(s)/channel(s).
 */
template <
    std::uint8_t SetMask        = 0,
    std::uint8_t ClientMask     = 0,
    std::uint8_t ChannelMask    = 0,
    std::uint8_t CurrentChannel = 0,
    std::uint32_t X_Channel0    = 0,
    std::uint32_t X_Channel1    = 0,
    std::uint32_t Y_Channel0    = 0,
    std::uint32_t Y_Channel1    = 0,
    std::uint8_t Z_Channel0     = 0,
    std::uint8_t Z_Channel1     = 0,
    std::uint8_t W_Channel0     = 0,
    std::uint8_t W_Channel1     = 0>
struct ADC
{
    /*
     * Each counter owns a non-overlapping 2-bit field in SetMask
     * bit0 -> Represents Channel 0
     * bit1 -> Represents Channel 1
     */
    static constexpr std::uint8_t X_CHANNEL0 = 0b01 << 0;
    static constexpr std::uint8_t X_CHANNEL1 = 0b10 << 0;

    static constexpr std::uint8_t Y_CHANNEL0 = 0b01 << 2;
    static constexpr std::uint8_t Y_CHANNEL1 = 0b10 << 2;

    static constexpr std::uint8_t Z_CHANNEL0 = 0b01 << 4;
    static constexpr std::uint8_t Z_CHANNEL1 = 0b10 << 4;

    static constexpr std::uint8_t W_CHANNEL0 = 0b01 << 6;
    static constexpr std::uint8_t W_CHANNEL1 = 0b10 << 6;

    template <AddressCounterClient... clients>
    static constexpr std::uint8_t client_bits()
    {
        return (std::uint8_t {0} | ... | ckernel::to_underlying(clients));
    }

    template <AddressCounterClient... clients>
    constexpr auto client() const
    {
        return ADC<
            SetMask,
            ClientMask | client_bits<clients...>(),
            ChannelMask,
            CurrentChannel,
            X_Channel0,
            X_Channel1,
            Y_Channel0,
            Y_Channel1,
            Z_Channel0,
            Z_Channel1,
            W_Channel0,
            W_Channel1> {};
    }

    template <AddressChannel Channel>
    constexpr auto channel() const
    {
        static_assert(!(ChannelMask & ckernel::to_underlying(Channel)), "channel already selected — channel<...>() called twice for the same channel");

        // ChannelMask accumulates every selected channel (used by the emitter and the double-select guard);
        // CurrentChannel is replaced so subsequent X/Y/Z/W<value>() write only this channel.
        return ADC<
            SetMask,
            ClientMask,
            ChannelMask | ckernel::to_underlying(Channel),
            ckernel::to_underlying(Channel),
            X_Channel0,
            X_Channel1,
            Y_Channel0,
            Y_Channel1,
            Z_Channel0,
            Z_Channel1,
            W_Channel0,
            W_Channel1> {};
    }

    template <std::uint32_t value>
    constexpr auto X() const
    {
        return _set_counter<Counter::X, value>();
    }

    template <std::uint32_t value>
    constexpr auto Y() const
    {
        return _set_counter<Counter::Y, value>();
    }

    template <std::uint8_t value>
    constexpr auto Z() const
    {
        return _set_counter<Counter::Z, value>();
    }

    template <std::uint8_t value>
    constexpr auto W() const
    {
        return _set_counter<Counter::W, value>();
    }

private:
    // True when any bit of the given field mask is set in SetMask.
    static constexpr bool _is_set(std::uint8_t mask)
    {
        return SetMask & mask;
    }

    // Internal enum helper. Values are shifter as the CHANNEL0/CHANNEL1 mask shift for that counter.
    enum class Counter : std::uint8_t
    {
        X = 0x0,
        Y = 0x2,
        Z = 0x4,
        W = 0x6
    };

    // ADC with `value` written into the given counter's slot(s) for whichever channel(s) CurrentChannel selects,
    // and the matching channel bits OR'd into SetMask.
    template <Counter counter, auto value>
    static constexpr auto _set_counter()
    {
        constexpr bool selected_channel0 = CurrentChannel & ckernel::to_underlying(AddressChannel::Channel0);
        constexpr bool selected_channel1 = CurrentChannel & ckernel::to_underlying(AddressChannel::Channel1);

        // Check which channel is selected
        constexpr std::uint8_t counter_channel0_mask = 0b01 << ckernel::to_underlying(counter);
        constexpr std::uint8_t counter_channle1_mask = 0b10 << ckernel::to_underlying(counter);

        // Build new set mask
        constexpr std::uint8_t new_set_mask = SetMask | (selected_channel0 ? counter_channel0_mask : 0) | (selected_channel1 ? counter_channle1_mask : 0);

        // Choose if we set new value or we stay with default one
        constexpr std::uint32_t new_X0 = (counter == Counter::X && selected_channel0) ? value : X_Channel0;
        constexpr std::uint32_t new_X1 = (counter == Counter::X && selected_channel1) ? value : X_Channel1;

        constexpr std::uint32_t new_Y0 = (counter == Counter::Y && selected_channel0) ? value : Y_Channel0;
        constexpr std::uint32_t new_Y1 = (counter == Counter::Y && selected_channel1) ? value : Y_Channel1;

        constexpr std::uint8_t new_Z0 = (counter == Counter::Z && selected_channel0) ? value : Z_Channel0;
        constexpr std::uint8_t new_Z1 = (counter == Counter::Z && selected_channel1) ? value : Z_Channel1;

        constexpr std::uint8_t new_W0 = (counter == Counter::W && selected_channel0) ? value : W_Channel0;
        constexpr std::uint8_t new_W1 = (counter == Counter::W && selected_channel1) ? value : W_Channel1;

        return ADC<new_set_mask, ClientMask, ChannelMask, CurrentChannel, new_X0, new_X1, new_Y0, new_Y1, new_Z0, new_Z1, new_W0, new_W1> {};
    }

    static constexpr std::uint32_t _single_counter_op()
    {
        constexpr bool selected_channel0 = ChannelMask & ckernel::to_underlying(AddressChannel::Channel0);
        constexpr bool selected_channel1 = ChannelMask & ckernel::to_underlying(AddressChannel::Channel1);

        constexpr bool is_X = _is_set(X_CHANNEL0 | X_CHANNEL1);
        constexpr bool is_Y = _is_set(Y_CHANNEL0 | Y_CHANNEL1);
        constexpr bool is_Z = _is_set(Z_CHANNEL0 | Z_CHANNEL1);

        // ChannelIndex: 0 -> Channel0, 1 -> Channel1
        constexpr std::uint32_t channel = selected_channel1 ? 1 : 0;

        // Select which counter is getting changed: 0 -> X, 1 -> Y, 2 -> Z, 3 -> W
        constexpr std::uint32_t counter_selection = is_X ? 0 : is_Y ? 1 : is_Z ? 2 : 3;

        // Choose selected value
        constexpr std::uint32_t X_value = selected_channel0 ? X_Channel0 : X_Channel1;
        constexpr std::uint32_t Y_value = selected_channel0 ? Y_Channel0 : Y_Channel1;
        constexpr std::uint32_t Z_value = selected_channel0 ? Z_Channel0 : Z_Channel1;
        constexpr std::uint32_t W_value = selected_channel0 ? W_Channel0 : W_Channel1;
        constexpr std::uint32_t value   = is_X ? X_value : is_Y ? Y_value : is_Z ? Z_value : W_value;

        return TT_OP_SETADC(ClientMask, channel, counter_selection, value);
    }

    // Encoded SETADCXX instruction, sets both channels
    static constexpr std::uint32_t _xx_op()
    {
        return TT_OP_SETADCXX(ClientMask, X_Channel0, X_Channel1);
    }

    // Encoded SETADCXY instruction word for the current X/Y selection.
    template <GetOpType operation>
    static constexpr std::uint32_t _xy_op()
    {
        if constexpr (operation == GetOpType::SETTER)
        {
            constexpr bool ch0_x              = _is_set(X_CHANNEL0);
            constexpr bool ch1_x              = _is_set(X_CHANNEL1);
            constexpr bool ch0_y              = _is_set(Y_CHANNEL0);
            constexpr bool ch1_y              = _is_set(Y_CHANNEL1);
            constexpr std::uint8_t write_mask = (ch0_x << 0) | (ch0_y << 1) | (ch1_x << 2) | (ch1_y << 3);

            return TT_OP_SETADCXY(ClientMask, Y_Channel1, X_Channel1, Y_Channel0, X_Channel0, write_mask);
        }
        else
        {
            return TT_OP_INCADCXY(ClientMask, Y_Channel1, X_Channel1, Y_Channel0, X_Channel0);
        }
    }

    // Encoded SETADCZW instruction word for the current Z/W selection.
    template <GetOpType operation>
    static constexpr std::uint32_t _zw_op()
    {
        if constexpr (operation == GetOpType::SETTER)
        {
            constexpr bool ch0_z              = _is_set(Z_CHANNEL0);
            constexpr bool ch1_z              = _is_set(Z_CHANNEL1);
            constexpr bool ch0_w              = _is_set(W_CHANNEL0);
            constexpr bool ch1_w              = _is_set(W_CHANNEL1);
            constexpr std::uint8_t write_mask = (ch0_z << 0) | (ch0_w << 1) | (ch1_z << 2) | (ch1_w << 3);

            return TT_OP_SETADCZW(ClientMask, W_Channel1, Z_Channel1, W_Channel0, Z_Channel0, write_mask);
        }
        else
        {
            return TT_OP_INCADCZW(ClientMask, W_Channel1, Z_Channel1, W_Channel0, Z_Channel0);
        }
    }

    // Shared validation for both get_operation() and apply().
    static constexpr void _assert_selection()
    {
        static_assert(ClientMask != 0, "no client selected — call client<...>() first");
        static_assert(ChannelMask != 0, "no channel selected — call channel<...>() first");
    }

public:
    /**
     *  Return single instruction for given builder description
     *  Valid builder groups:
     *  X or Y, any channel
     *  Z or W, any channel
     *  Single counter, single channel (e.g. X counter, Channel 0)
     */
    template <GetOpType operation>
    static constexpr std::uint32_t get_operation()
    {
        _assert_selection();

        if constexpr (operation == GetOpType::SETTER)
        {
            constexpr bool selected_channel0 = ChannelMask & ckernel::to_underlying(AddressChannel::Channel0);
            constexpr bool selected_channel1 = ChannelMask & ckernel::to_underlying(AddressChannel::Channel1);

            constexpr bool is_X = _is_set(X_CHANNEL0 | X_CHANNEL1);
            constexpr bool is_Y = _is_set(Y_CHANNEL0 | Y_CHANNEL1);
            constexpr bool is_Z = _is_set(Z_CHANNEL0 | Z_CHANNEL1);
            constexpr bool is_W = _is_set(W_CHANNEL0 | W_CHANNEL1);

            constexpr bool is_single_counter_op = (selected_channel0 != selected_channel1) && (is_X + is_Y + is_Z + is_W == 1);
            constexpr bool is_xx_op             = is_X && !is_Y && !is_Z && !is_W && selected_channel0 && selected_channel1;
            constexpr bool is_xy_op             = (is_X || is_Y) && !is_Z && !is_W;
            constexpr bool is_zw_op             = (is_Z || is_W) && !is_X && !is_Y;

            if constexpr (is_single_counter_op)
            {
                return _single_counter_op();
            }
            else if constexpr (is_xx_op)
            {
                return _xx_op();
            }
            else if constexpr (is_xy_op)
            {
                return _xy_op<operation>();
            }
            else if constexpr (is_zw_op)
            {
                return _zw_op<operation>();
            }
            else
            {
                static_assert(
                    SetMask != SetMask,
                    "unsupported selection — matches none of the valid op() patterns (single counter/single channel, X/Y any channel, or Z/W any channel)");
            }
        }
        else if constexpr (operation == GetOpType::INCREMENT)
        {
            constexpr bool is_X = _is_set(X_CHANNEL0 | X_CHANNEL1);
            constexpr bool is_Y = _is_set(Y_CHANNEL0 | Y_CHANNEL1);

            constexpr bool is_Z = _is_set(Z_CHANNEL0 | Z_CHANNEL1);
            constexpr bool is_W = _is_set(W_CHANNEL0 | W_CHANNEL1);

            if constexpr (is_X || is_Y)
            {
                return _xy_op<operation>();
            }
            else if constexpr (is_Z || is_W)
            {
                return _zw_op<operation>();
            }
            else
            {
                static_assert(SetMask != SetMask, "misuse - nothing to increment");
            }
        }
        else
        {
            static_assert(operation != operation, "Invalid operation type");
        }
    }

    inline __attribute__((always_inline)) void apply() const
    {
        _assert_selection();

        static constexpr bool selected_channel0 = ChannelMask & ckernel::to_underlying(AddressChannel::Channel0);
        static constexpr bool selected_channel1 = ChannelMask & ckernel::to_underlying(AddressChannel::Channel1);
        static constexpr bool is_single_channel = selected_channel0 + selected_channel1 == 1;

        static constexpr bool is_X = _is_set(X_CHANNEL0 | X_CHANNEL1);
        static constexpr bool is_Y = _is_set(Y_CHANNEL0 | Y_CHANNEL1);

        if constexpr (is_X || is_Y)
        {
            static constexpr bool only_X = is_X && !is_Y;
            static constexpr bool only_Y = is_Y && !is_X;

            if constexpr (only_X && !is_single_channel)
            {
                // X only, both channels -> single SETADCXX.
                INSTRUCTION_WORD(_xx_op());
            }
            else if constexpr ((only_X || only_Y) && is_single_channel)
            {
                // One counter on one channel -> single SETADC.
                INSTRUCTION_WORD(_single_counter_op());
            }
            else
            {
                // X and Y, or a counter across both channels -> SETADCXY.
                INSTRUCTION_WORD(_xy_op<GetOpType::SETTER>());
            }
        }

        static constexpr bool is_Z = _is_set(Z_CHANNEL0 | Z_CHANNEL1);
        static constexpr bool is_W = _is_set(W_CHANNEL0 | W_CHANNEL1);

        if constexpr (is_Z || is_W)
        {
            static constexpr bool single_set_from_ZW = (is_Z + is_W == 1);
            if constexpr (single_set_from_ZW && is_single_channel)
            {
                INSTRUCTION_WORD(_single_counter_op());
            }
            else
            {
                INSTRUCTION_WORD(_zw_op<GetOpType::SETTER>());
            }
        }
    }

    inline __attribute__((always_inline)) void increment() const
    {
        _assert_selection();

        static constexpr bool is_X = _is_set(X_CHANNEL0 | X_CHANNEL1);
        static constexpr bool is_Y = _is_set(Y_CHANNEL0 | Y_CHANNEL1);

        static constexpr bool is_Z = _is_set(Z_CHANNEL0 | Z_CHANNEL1);
        static constexpr bool is_W = _is_set(W_CHANNEL0 | W_CHANNEL1);

        if constexpr (is_X || is_Y)
        {
            INSTRUCTION_WORD(_xy_op<GetOpType::INCREMENT>());
        }

        if constexpr (is_Z || is_W)
        {
            INSTRUCTION_WORD(_zw_op<GetOpType::INCREMENT>());
        }
    }

    // TT_ADDRCRXY and ADDRCRZW were never used and were removed in later arhitecures
    // inline __attribute__((always_inline)) void add_and_reset() const
    // {
    //     static_assert(ClientMask != 0, "add_and_reset(): no client selected — call client<...>() first");
    //     static_assert(ChannelMask != 0, "add_and_reset(): no channel selected — call channel<...>() first");

    //     if constexpr (_is_set(X_SET)) {}
    //     if constexpr (_is_set(Y_SET)) {}
    //     if constexpr (_is_set(Z_SET)) {}
    //     if constexpr (_is_set(W_SET)) {}
    // }
};

/**
 * @brief Compile-time-selected, runtime-valued address-counter builder.
 *
 * Client, channel, and counter selection remain template arguments, exactly as in @ref ADC.
 * X/Y/Z/W values are normal function arguments. Consequently the emitter is selected entirely at
 * compile time and the inlined runtime path contains a TT_* instruction write with no dispatch.
 *
 * @code
 *   runtime_address_counters
 *       .client<AddressCounterClient::Unpacker0>()
 *       .channel<AddressChannel::Channel0>()
 *       .X(face_r_dim)
 *       .Y(0)
 *       .apply();
 * @endcode
 */
template <std::uint8_t SetMask = 0, std::uint8_t ClientMask = 0, std::uint8_t ChannelMask = 0, std::uint8_t CurrentChannel = 0>
struct RuntimeADC
{
private:
    static constexpr std::uint8_t X_CHANNEL0 = 0b01 << 0;
    static constexpr std::uint8_t X_CHANNEL1 = 0b10 << 0;
    static constexpr std::uint8_t Y_CHANNEL0 = 0b01 << 2;
    static constexpr std::uint8_t Y_CHANNEL1 = 0b10 << 2;
    static constexpr std::uint8_t Z_CHANNEL0 = 0b01 << 4;
    static constexpr std::uint8_t Z_CHANNEL1 = 0b10 << 4;
    static constexpr std::uint8_t W_CHANNEL0 = 0b01 << 6;
    static constexpr std::uint8_t W_CHANNEL1 = 0b10 << 6;

    enum class Counter : std::uint8_t
    {
        X = 0x0,
        Y = 0x2,
        Z = 0x4,
        W = 0x6
    };

    std::uint32_t x_channel0 = 0;
    std::uint32_t x_channel1 = 0;
    std::uint32_t y_channel0 = 0;
    std::uint32_t y_channel1 = 0;
    std::uint32_t z_channel0 = 0;
    std::uint32_t z_channel1 = 0;
    std::uint32_t w_channel0 = 0;
    std::uint32_t w_channel1 = 0;

    template <std::uint8_t, std::uint8_t, std::uint8_t, std::uint8_t>
    friend struct RuntimeADC;

    static constexpr bool _is_set(const std::uint8_t mask)
    {
        return SetMask & mask;
    }

    inline __attribute__((always_inline)) static constexpr void _assert_selection()
    {
        static_assert(ClientMask != 0, "no client selected — call client<...>() first");
        static_assert(ChannelMask != 0, "no channel selected — call channel<...>() first");
    }

    template <std::uint8_t NewSetMask, std::uint8_t NewClientMask, std::uint8_t NewChannelMask, std::uint8_t NewCurrentChannel>
    inline __attribute__((always_inline)) constexpr RuntimeADC<NewSetMask, NewClientMask, NewChannelMask, NewCurrentChannel> _rebind() const
    {
        RuntimeADC<NewSetMask, NewClientMask, NewChannelMask, NewCurrentChannel> next;
        next.x_channel0 = x_channel0;
        next.x_channel1 = x_channel1;
        next.y_channel0 = y_channel0;
        next.y_channel1 = y_channel1;
        next.z_channel0 = z_channel0;
        next.z_channel1 = z_channel1;
        next.w_channel0 = w_channel0;
        next.w_channel1 = w_channel1;
        return next;
    }

    template <Counter counter>
    inline __attribute__((always_inline)) constexpr auto _set_counter(const std::uint32_t value) const
    {
        constexpr bool selected_channel0     = CurrentChannel & ckernel::to_underlying(AddressChannel::Channel0);
        constexpr bool selected_channel1     = CurrentChannel & ckernel::to_underlying(AddressChannel::Channel1);
        constexpr std::uint8_t channel0_mask = 0b01 << ckernel::to_underlying(counter);
        constexpr std::uint8_t channel1_mask = 0b10 << ckernel::to_underlying(counter);
        constexpr std::uint8_t new_set_mask  = SetMask | (selected_channel0 ? channel0_mask : 0) | (selected_channel1 ? channel1_mask : 0);

        auto next = _rebind<new_set_mask, ClientMask, ChannelMask, CurrentChannel>();
        if constexpr (counter == Counter::X)
        {
            if constexpr (selected_channel0)
            {
                next.x_channel0 = value;
            }
            if constexpr (selected_channel1)
            {
                next.x_channel1 = value;
            }
        }
        else if constexpr (counter == Counter::Y)
        {
            if constexpr (selected_channel0)
            {
                next.y_channel0 = value;
            }
            if constexpr (selected_channel1)
            {
                next.y_channel1 = value;
            }
        }
        else if constexpr (counter == Counter::Z)
        {
            if constexpr (selected_channel0)
            {
                next.z_channel0 = value;
            }
            if constexpr (selected_channel1)
            {
                next.z_channel1 = value;
            }
        }
        else
        {
            if constexpr (selected_channel0)
            {
                next.w_channel0 = value;
            }
            if constexpr (selected_channel1)
            {
                next.w_channel1 = value;
            }
        }
        return next;
    }

    inline __attribute__((always_inline)) void _emit_single_counter() const
    {
        constexpr bool selected_channel0 = ChannelMask & ckernel::to_underlying(AddressChannel::Channel0);
        constexpr bool selected_channel1 = ChannelMask & ckernel::to_underlying(AddressChannel::Channel1);
        constexpr bool is_x              = _is_set(X_CHANNEL0 | X_CHANNEL1);
        constexpr bool is_y              = _is_set(Y_CHANNEL0 | Y_CHANNEL1);
        constexpr bool is_z              = _is_set(Z_CHANNEL0 | Z_CHANNEL1);
        constexpr std::uint32_t channel  = selected_channel1 ? 1 : 0;
        constexpr std::uint32_t counter  = is_x ? 0 : is_y ? 1 : is_z ? 2 : 3;

        if constexpr (is_x)
        {
            TT_SETADC(ClientMask, channel, counter, selected_channel0 ? x_channel0 : x_channel1);
        }
        else if constexpr (is_y)
        {
            TT_SETADC(ClientMask, channel, counter, selected_channel0 ? y_channel0 : y_channel1);
        }
        else if constexpr (is_z)
        {
            TT_SETADC(ClientMask, channel, counter, selected_channel0 ? z_channel0 : z_channel1);
        }
        else
        {
            TT_SETADC(ClientMask, channel, counter, selected_channel0 ? w_channel0 : w_channel1);
        }
    }

public:
    template <AddressCounterClient... clients>
    inline __attribute__((always_inline)) constexpr auto client() const
    {
        constexpr std::uint8_t clients_mask = (std::uint8_t {0} | ... | ckernel::to_underlying(clients));
        return _rebind<SetMask, ClientMask | clients_mask, ChannelMask, CurrentChannel>();
    }

    template <AddressChannel Channel>
    inline __attribute__((always_inline)) constexpr auto channel() const
    {
        constexpr std::uint8_t channel_bit = ckernel::to_underlying(Channel);
        static_assert(!(ChannelMask & channel_bit), "channel already selected — channel<...>() called twice for the same channel");
        return _rebind<SetMask, ClientMask, ChannelMask | channel_bit, channel_bit>();
    }

    inline __attribute__((always_inline)) constexpr auto X(const std::uint32_t value) const
    {
        return _set_counter<Counter::X>(value);
    }

    inline __attribute__((always_inline)) constexpr auto Y(const std::uint32_t value) const
    {
        return _set_counter<Counter::Y>(value);
    }

    inline __attribute__((always_inline)) constexpr auto Z(const std::uint32_t value) const
    {
        return _set_counter<Counter::Z>(value);
    }

    inline __attribute__((always_inline)) constexpr auto W(const std::uint32_t value) const
    {
        return _set_counter<Counter::W>(value);
    }

    template <GetOpType operation>
    inline __attribute__((always_inline)) std::uint32_t get_operation() const
    {
        _assert_selection();

        constexpr bool selected_channel0 = ChannelMask & ckernel::to_underlying(AddressChannel::Channel0);
        constexpr bool selected_channel1 = ChannelMask & ckernel::to_underlying(AddressChannel::Channel1);
        constexpr bool is_x              = _is_set(X_CHANNEL0 | X_CHANNEL1);
        constexpr bool is_y              = _is_set(Y_CHANNEL0 | Y_CHANNEL1);
        constexpr bool is_z              = _is_set(Z_CHANNEL0 | Z_CHANNEL1);
        constexpr bool is_w              = _is_set(W_CHANNEL0 | W_CHANNEL1);
        constexpr bool has_xy            = is_x || is_y;
        constexpr bool has_zw            = is_z || is_w;

        static_assert(has_xy != has_zw, "runtime get_operation() must encode exactly one TT instruction (select X/Y or Z/W, not both)");

        if constexpr (operation == GetOpType::SETTER)
        {
            constexpr bool single_counter = (selected_channel0 != selected_channel1) && (is_x + is_y + is_z + is_w == 1);
            constexpr bool xx             = is_x && !is_y && selected_channel0 && selected_channel1;

            if constexpr (single_counter)
            {
                constexpr std::uint32_t channel = selected_channel1 ? 1 : 0;
                constexpr std::uint32_t counter = is_x ? 0 : is_y ? 1 : is_z ? 2 : 3;
                if constexpr (is_x)
                {
                    return TT_OP_SETADC(ClientMask, channel, counter, selected_channel0 ? x_channel0 : x_channel1);
                }
                else if constexpr (is_y)
                {
                    return TT_OP_SETADC(ClientMask, channel, counter, selected_channel0 ? y_channel0 : y_channel1);
                }
                else if constexpr (is_z)
                {
                    return TT_OP_SETADC(ClientMask, channel, counter, selected_channel0 ? z_channel0 : z_channel1);
                }
                else
                {
                    return TT_OP_SETADC(ClientMask, channel, counter, selected_channel0 ? w_channel0 : w_channel1);
                }
            }
            else if constexpr (xx)
            {
                return TT_OP_SETADCXX(ClientMask, x_channel0, x_channel1);
            }
            else if constexpr (has_xy)
            {
                constexpr std::uint8_t write_mask =
                    (_is_set(X_CHANNEL0) << 0) | (_is_set(Y_CHANNEL0) << 1) | (_is_set(X_CHANNEL1) << 2) | (_is_set(Y_CHANNEL1) << 3);
                return TT_OP_SETADCXY(ClientMask, y_channel1, x_channel1, y_channel0, x_channel0, write_mask);
            }
            else
            {
                constexpr std::uint8_t write_mask =
                    (_is_set(Z_CHANNEL0) << 0) | (_is_set(W_CHANNEL0) << 1) | (_is_set(Z_CHANNEL1) << 2) | (_is_set(W_CHANNEL1) << 3);
                return TT_OP_SETADCZW(ClientMask, w_channel1, z_channel1, w_channel0, z_channel0, write_mask);
            }
        }
        else
        {
            static_assert(operation == GetOpType::INCREMENT, "invalid operation type");
            if constexpr (has_xy)
            {
                return TT_OP_INCADCXY(ClientMask, y_channel1, x_channel1, y_channel0, x_channel0);
            }
            else
            {
                return TT_OP_INCADCZW(ClientMask, w_channel1, z_channel1, w_channel0, z_channel0);
            }
        }
    }

    inline __attribute__((always_inline)) void apply() const
    {
        _assert_selection();

        constexpr bool selected_channel0 = ChannelMask & ckernel::to_underlying(AddressChannel::Channel0);
        constexpr bool selected_channel1 = ChannelMask & ckernel::to_underlying(AddressChannel::Channel1);
        constexpr bool single_channel    = selected_channel0 + selected_channel1 == 1;
        constexpr bool is_x              = _is_set(X_CHANNEL0 | X_CHANNEL1);
        constexpr bool is_y              = _is_set(Y_CHANNEL0 | Y_CHANNEL1);
        constexpr bool is_z              = _is_set(Z_CHANNEL0 | Z_CHANNEL1);
        constexpr bool is_w              = _is_set(W_CHANNEL0 | W_CHANNEL1);
        constexpr bool has_xy            = is_x || is_y;
        constexpr bool has_zw            = is_z || is_w;

        static_assert(has_xy != has_zw, "runtime apply() must encode exactly one TT instruction (select X/Y or Z/W, not both)");

        if constexpr (has_xy)
        {
            constexpr bool only_x = is_x && !is_y;
            constexpr bool only_y = is_y && !is_x;
            if constexpr (only_x && !single_channel)
            {
                TT_SETADCXX(ClientMask, x_channel0, x_channel1);
            }
            else if constexpr ((only_x || only_y) && single_channel)
            {
                _emit_single_counter();
            }
            else
            {
                constexpr std::uint8_t write_mask =
                    (_is_set(X_CHANNEL0) << 0) | (_is_set(Y_CHANNEL0) << 1) | (_is_set(X_CHANNEL1) << 2) | (_is_set(Y_CHANNEL1) << 3);
                TT_SETADCXY(ClientMask, y_channel1, x_channel1, y_channel0, x_channel0, write_mask);
            }
        }
        else
        {
            constexpr bool one_counter = is_z + is_w == 1;
            if constexpr (one_counter && single_channel)
            {
                _emit_single_counter();
            }
            else
            {
                constexpr std::uint8_t write_mask =
                    (_is_set(Z_CHANNEL0) << 0) | (_is_set(W_CHANNEL0) << 1) | (_is_set(Z_CHANNEL1) << 2) | (_is_set(W_CHANNEL1) << 3);
                TT_SETADCZW(ClientMask, w_channel1, z_channel1, w_channel0, z_channel0, write_mask);
            }
        }
    }

    inline __attribute__((always_inline)) void increment() const
    {
        _assert_selection();

        constexpr bool has_xy = _is_set(X_CHANNEL0 | X_CHANNEL1 | Y_CHANNEL0 | Y_CHANNEL1);
        constexpr bool has_zw = _is_set(Z_CHANNEL0 | Z_CHANNEL1 | W_CHANNEL0 | W_CHANNEL1);
        static_assert(has_xy != has_zw, "runtime increment() must encode exactly one TT instruction (select X/Y or Z/W, not both)");

        if constexpr (has_xy)
        {
            TT_INCADCXY(ClientMask, y_channel1, x_channel1, y_channel0, x_channel0);
        }
        else
        {
            TT_INCADCZW(ClientMask, w_channel1, z_channel1, w_channel0, z_channel0);
        }
    }
};

using _address_counters = ADC<>;
inline constexpr _address_counters address_counters {};
inline constexpr RuntimeADC<> runtime_address_counters {};

} // namespace hal
