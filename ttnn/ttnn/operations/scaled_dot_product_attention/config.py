# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Performance configuration for the fused tiled flash-attention program."""

from dataclasses import dataclass

import ttnn


FLASH_ATTENTION_PROFILE_PHASES = (
    "FA_DIRECT_Q_READ",
    "FA_DIRECT_KV_READ",
    "FA_SENDER_Q_READ",
    "FA_SENDER_KV_RESERVE",
    "FA_SENDER_KV_DRAM",
    "FA_SENDER_KV_MCAST",
    "FA_RECEIVER_Q_READ",
    "FA_RECEIVER_KV_MCAST",
    "FA_WRITER_WAIT",
    "FA_WRITER_DRAM",
    "FA_Q_SCALE",
    "FA_QK_MATMUL",
    "FA_BLOCK_MAX",
    "FA_ONLINE_RESCALE",
    "FA_PROBS_EXP",
    "FA_BLOCK_SUM",
    "FA_PV_MATMUL",
    "FA_STATE_O_UPDATE",
    "FA_FINAL_NORMALIZE",
)


@dataclass(frozen=True)
class FlashAttentionProgramConfig:
    """Compile-time tuning knobs for :func:`ttnn.flash_attention`.

    Tile counts are in 32x32 tiles. The defaults target prefill with at least
    eight heads, sequence lengths of 4096 or more, and head dimensions around
    64--128:

    - four query tile rows per core invocation;
    - sixteen key/value tile rows per online-softmax step;
    - eight-tile matmul output subblocks using BF16 DEST accumulation;
    - two-tile elementwise batches for the softmax/state phases;
    - approximate block-probability exp plus accurate-fast online rescaling;
    - horizontal per-head core groups which read K/V once and multicast them;
    - double-buffered K/V and output streams;
    - NoC0 reads and NoC1 writes.

    ``q_parallel_group_size`` caps the number of cores collaborating on one
    head. ``None`` chooses the widest horizontal segment which fits all heads
    concurrently. Set ``use_kv_multicast=False`` to use an independent DRAM
    reader per core, which can be useful for small problems or topology studies.
    ``spread_kv_readers=True`` rotates the K/V sender position by head instead
    of concentrating all senders at the left edge of their groups.
    """

    query_block_tiles: int | None = None
    key_block_tiles: int | None = None
    qk_output_subblock: tuple[int, int] | None = None
    pv_output_subblock: tuple[int, int] | None = None
    softmax_block_tiles: int | None = None

    num_cores: int | None = None
    q_parallel_group_size: int | None = None
    use_kv_multicast: bool = True
    spread_kv_readers: bool = True

    kv_buffer_depth: int = 2
    output_buffer_depth: int = 2
    read_barrier_tiles: int = 8
    write_barrier_tiles: int = 8
    reader_noc: str = "noc0"
    writer_noc: str = "noc1"

    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    exp_approx_mode: str = "fast"
    rescale_exp_approx_mode: str | None = None
    fp32_dest_acc_en: bool = False
    profile_phase: str | None = None

    def validate_basic(self) -> None:
        positive = {
            "kv_buffer_depth": self.kv_buffer_depth,
            "output_buffer_depth": self.output_buffer_depth,
            "read_barrier_tiles": self.read_barrier_tiles,
            "write_barrier_tiles": self.write_barrier_tiles,
        }
        if self.softmax_block_tiles is not None:
            positive["softmax_block_tiles"] = self.softmax_block_tiles
        if self.query_block_tiles is not None:
            positive["query_block_tiles"] = self.query_block_tiles
        if self.key_block_tiles is not None:
            positive["key_block_tiles"] = self.key_block_tiles
        if self.num_cores is not None:
            positive["num_cores"] = self.num_cores
        if self.q_parallel_group_size is not None:
            positive["q_parallel_group_size"] = self.q_parallel_group_size
        for name, value in positive.items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"flash_attention: {name} must be a positive integer, got {value!r}")

        for name, value in (
            ("use_kv_multicast", self.use_kv_multicast),
            ("spread_kv_readers", self.spread_kv_readers),
            ("fp32_dest_acc_en", self.fp32_dest_acc_en),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"flash_attention: {name} must be a bool, got {value!r}")
        dest_capacity = self.dest_tile_capacity
        if self.resolved_softmax_block_tiles > dest_capacity:
            raise ValueError(
                f"flash_attention: softmax_block_tiles must be <= {dest_capacity} for "
                f"{'FP32' if self.fp32_dest_acc_en else 'BF16'} DEST"
            )
        if self.reader_noc not in ("noc0", "noc1") or self.writer_noc not in ("noc0", "noc1"):
            raise ValueError("flash_attention: reader_noc and writer_noc must be 'noc0' or 'noc1'")
        if self.reader_noc == self.writer_noc:
            raise ValueError("flash_attention: reader_noc and writer_noc must differ so the streams can overlap")
        exp_modes = ("fast", "accurate_fast", "exact")
        if self.exp_approx_mode not in exp_modes:
            raise ValueError("flash_attention: exp_approx_mode must be 'fast', 'accurate_fast', or 'exact'")
        if self.rescale_exp_approx_mode is not None and self.rescale_exp_approx_mode not in exp_modes:
            raise ValueError(
                "flash_attention: rescale_exp_approx_mode must be None, 'fast', 'accurate_fast', or 'exact'"
            )
        if "accurate_fast" in (self.exp_approx_mode, self.resolved_rescale_exp_approx_mode) and self.fp32_dest_acc_en:
            raise ValueError("flash_attention: accurate_fast exponential requires fp32_dest_acc_en=False")
        if self.profile_phase is not None and self.profile_phase not in FLASH_ATTENTION_PROFILE_PHASES:
            raise ValueError(
                f"flash_attention: profile_phase must be None or one of {FLASH_ATTENTION_PROFILE_PHASES}, "
                f"got {self.profile_phase!r}"
            )

        for name, shape in (
            ("qk_output_subblock", self.qk_output_subblock),
            ("pv_output_subblock", self.pv_output_subblock),
        ):
            if shape is None:
                continue
            if not isinstance(shape, tuple) or len(shape) != 2 or any(not isinstance(x, int) or x < 1 for x in shape):
                raise ValueError(f"flash_attention: {name} must be a pair of positive tile counts, got {shape!r}")
            if shape[0] * shape[1] > dest_capacity:
                raise ValueError(
                    f"flash_attention: {name}={shape} exceeds the {dest_capacity}-tile "
                    f"{'FP32' if self.fp32_dest_acc_en else 'BF16'} DEST budget"
                )

    @property
    def dest_tile_capacity(self) -> int:
        """Number of live tiles available in DEST for the selected accumulation mode."""
        return 4 if self.fp32_dest_acc_en else 8

    @property
    def resolved_softmax_block_tiles(self) -> int:
        """Eltwise batch size, tuned independently of the matmul DEST capacity."""
        return min(2, self.dest_tile_capacity) if self.softmax_block_tiles is None else self.softmax_block_tiles

    @property
    def resolved_rescale_exp_approx_mode(self) -> str:
        """Online-max rescale mode, selecting accurate-fast automatically for BF16 DEST."""
        if self.rescale_exp_approx_mode is not None:
            return self.rescale_exp_approx_mode
        return "accurate_fast" if not self.fp32_dest_acc_en else self.exp_approx_mode


def resolve_output_subblock(
    rows: int,
    cols: int,
    requested: tuple[int, int] | None,
    dest_capacity: int = 4,
    *,
    prefer_wide: bool = True,
) -> tuple[int, int]:
    """Resolve an explicit subblock or choose the largest balanced divisor with directional reuse."""
    if requested is not None:
        return requested
    candidates = [
        (height, width)
        for height in range(1, min(rows, dest_capacity) + 1)
        for width in range(1, min(cols, dest_capacity) + 1)
        if height * width <= dest_capacity and rows % height == 0 and cols % width == 0
    ]
    preferred_extent = 1 if prefer_wide else 0
    return max(
        candidates,
        key=lambda shape: (shape[0] * shape[1], -abs(shape[0] - shape[1]), shape[preferred_extent]),
    )


def resolve_block_tiles(total_tiles: int, requested: int | None, default_max: int) -> int:
    """Resolve an explicit block or choose the largest divisor no greater than the tuned default."""
    if requested is not None:
        return requested
    for candidate in range(min(total_tiles, default_max), 0, -1):
        if total_tiles % candidate == 0:
            return candidate
    return 1


DEFAULT_FLASH_ATTENTION_PROGRAM_CONFIG = FlashAttentionProgramConfig()
