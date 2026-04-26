# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import gdb
import gdb.printing


class ShapePrinter:
    """Pretty printer for tt::tt_metal::Shape

    Shape inherits from ShapeBase which contains a SmallVector<uint32_t> named 'value_'
    that stores the dimensions of the tensor.
    """

    def __init__(self, val):
        """Initialize the printer with a gdb.Value representing a Shape object."""
        self.__val = val

    def to_string(self):
        """Return a string representation of the Shape.

        Format: Shape([d1, d2, ..., dn], rank=n, volume=v)
        where d1..dn are dimensions, n is the rank, and v is the total volume.
        """
        value_ = self.__val["value_"]

        t = value_.type.template_argument(0).pointer()
        begin_ptr = value_["BeginX"].cast(t)
        size = value_["Size"]

        # Extract elements
        dimensions = []
        for i in range(size):
            elem = (begin_ptr + i).dereference()
            dimensions.append(int(elem))

        volume = eval("*".join(map(str, dimensions))) if dimensions else 0

        return "Shape({}, rank={}, volume={})".format(dimensions, len(dimensions), volume)

    def display_hint(self):
        """Return a display hint for GDB.

        This tells GDB how to format the output.
        """
        return "array"


def build_ttmetal_pretty_printer():
    """Build and return the pretty printer collection for TT-Metal types."""
    pp = gdb.printing.RegexpCollectionPrettyPrinter("tt-metal")

    # Register Shape printer
    # Matches: tt::tt_metal::Shape
    pp.add_printer("Shape", "^tt::tt_metal::Shape$", ShapePrinter)

    return pp


def register_ttmetal_printers(obj=None):
    """Register TT-Metal pretty printers with GDB.

    Args:
        obj: The objfile to register printers for. If None, registers globally.
    """
    gdb.printing.register_pretty_printer(obj, build_ttmetal_pretty_printer(), replace=True)


# Auto-register printers when this script is loaded
register_ttmetal_printers()

print("TT-Metal GDB pretty printers loaded successfully")
