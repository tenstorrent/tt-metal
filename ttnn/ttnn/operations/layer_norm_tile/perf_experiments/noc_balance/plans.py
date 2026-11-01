# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-core NoC plans. A plan(input, residual, blocking) returns f(core, core_row_tiles) ->
(reader_noc, writer_noc, residual_on_writer), or None for the op's current placement.

Plan names are '+'-joined rules applied left to right on top of base (reader NoC0, writer NoC1,
blocking.residual_on_writer). Core coords are LOGICAL (x 0..10, y 0..9 on BH p100a).
  rd1_all        every reader on NoC1
  wr0_all        every writer on NoC0
  swap_all       reader NoC1 + writer NoC0 everywhere
  split_all      residual stream on the writer RISC/NoC1 everywhere
  rd1_heavy      reader NoC1 on the cores holding the larger row share (only when the split is uneven)
  split_heavy    residual on the writer RISC/NoC1 on the heavy cores (only when the split is uneven)
  rd1_heavymin   rd1_heavy, only while the heavy group is at most half the active cores
  gate1row       (first) apply the rest only while most active cores own one row
  keepsplit      (last) split_reader cores keep the base placement
  rd1_y<k>       reader NoC1 on logical y < k
  wr0_x<k>       writer NoC0 on logical x >= k
  wr0_y<k>       writer NoC0 on logical y >= k
  wr0_par        writer NoC0 on (x + y) odd (checkerboard)
  wr0_yodd       writer NoC0 on odd logical rows
  wr0_half       writer NoC0 on the bottom half of the compute grid (2y >= grid_y)
"""


def _uneven_heavy(blk):
    counts = {rows for _, _, rows in blk.assignment}
    return max(counts) if len(counts) > 1 else None


def _rule(name, blk):
    heavy = _uneven_heavy(blk)
    if name == "base":
        return lambda core, rows, t: t
    if name == "rd1_all":
        return lambda core, rows, t: (1, t[1], t[2])
    if name == "wr0_all":
        return lambda core, rows, t: (t[0], 0, t[2])
    if name == "swap_all":
        return lambda core, rows, t: (1, 0, t[2])
    if name == "split_all":
        return lambda core, rows, t: (t[0], t[1], True)
    if name == "rd1_heavymin":
        # heavy group on NoC1 only while it is a minority of the active cores
        n_heavy = sum(1 for _, _, rows in blk.assignment if rows == heavy)
        minority = heavy is not None and 2 * n_heavy <= len(blk.assignment)
        return lambda core, rows, t: (1 if minority and rows == heavy else t[0], t[1], t[2])
    if name == "gate1row":
        # (first) whole plan applies only while most active cores own a single row: their reads and
        # writes then fall in separate phases. Otherwise every core stays on the base placement.
        one_row = sum(1 for _, _, rows in blk.assignment if rows == 1)
        on = 2 * one_row > len(blk.assignment)
        return lambda core, rows, t: t if on else ("off",)
    if name == "keepsplit":
        return lambda core, rows, t: (0, 1, True) if t[2] else t
    if name == "rd1_heavy":
        return lambda core, rows, t: (1 if rows == heavy else t[0], t[1], t[2])
    if name == "split_heavy":
        return lambda core, rows, t: (t[0], t[1], t[2] or rows == heavy)
    if name.startswith("rd1_y"):
        k = int(name[5:])
        return lambda core, rows, t: (1 if core.y < k else t[0], t[1], t[2])
    if name.startswith("wr0_x"):
        k = int(name[5:])
        return lambda core, rows, t: (t[0], 0 if core.x >= k else t[1], t[2])
    if name == "wr0_heavy":
        return lambda core, rows, t: (t[0], 0 if rows == heavy else t[1], t[2])
    if name == "wr0_par":
        return lambda core, rows, t: (t[0], 0 if (core.x + core.y) % 2 else t[1], t[2])
    if name == "wr0_yodd":
        return lambda core, rows, t: (t[0], 0 if core.y % 2 else t[1], t[2])
    if name == "wr0_half":
        # bottom half of the grid; derived from the device grid, not a fixed map
        return lambda core, rows, t: (t[0], 0 if 2 * core.y >= blk.grid_y else t[1], t[2])
    if name.startswith("wr0_y"):
        k = int(name[5:])
        return lambda core, rows, t: (t[0], 0 if core.y >= k else t[1], t[2])
    raise KeyError(name)


class _Plans(dict):
    def __missing__(self, name):
        def plan(inp, res, blk):
            object.__setattr__(blk, "grid_y", inp.device().compute_with_storage_grid_size().y)
            if name == "base":
                return None
            rules = [_rule(part, blk) for part in name.split("+")]

            def per_core(core, rows):
                t = (0, 1, blk.residual_on_writer)
                for rule in rules:
                    t = rule(core, rows, t)
                    if t == ("off",):
                        return (0, 1, blk.residual_on_writer)
                if t[2] and t[0] == t[1]:
                    # split_reader core: both RISCs read, so they need different NoCs; the writer's choice wins.
                    t = (1 - t[1], t[1], t[2])
                return t

            return per_core

        return plan


PLANS = _Plans()
