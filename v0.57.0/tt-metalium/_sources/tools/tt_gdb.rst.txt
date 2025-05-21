:orphan:

tt-gdb
======
.. raw:: html

    <video width="800" height="600" controls>
        <source src="../_static/DebuggerTutorial.mp4" type="video/mp4">
    <p>Your browser does not support the video tag.</p>
    </video>

Enabling the breakpoint server
------------------------------

To enable the breakpoint server, in your test, you must include `tools/tt_gdb/tt_gdb.hpp`, and call it right before calling your test function.

For example, below is a snippet of enabling tt_gdb in `test_run_datacopy`.

.. code-block:: cpp

    const vector<CoreCoord> cores = {core};
    const vector<string> ops = {op};

    tt_gdb(cluster, chip_id, cores, ops);
    pass &= run_data_copy_multi_tile(cluster, chip_id, core, 2048);

where cluster was defined earlier in the file.

Installing python packages
--------------------------

Within the ``tools/tt_gdb`` directory,
``pip3 install -r requirements.txt``

Compiling under debug mode
--------------------------
To get the most out of `tt_gdb`, you must compile under debug mode. This will use different debug ld files for the RISCs, and will enable debug symbols. To enable under debug mode, use
`DEBUG_MODE=1` when compiling a test.

For example, below is a snippet of compiling with `DEBUG_MODE=1` for compilation of the datacopy op.

``DEBUG_MODE=1 ./build/test/build_kernels_for_riscv/tests/test_compile_datacopy``

Limitations
-----------
So far, debug mode only compiles BRISC with proper debug symbols, therefore the `p` functionality will only work for BRISC; however, the breakpoint function should work for any RISC if your goal is just to hang a program.

Additionally, unit testing needs to be added for the debugger. If you have any issues,
please create an issue under the debugger board.

## Using breakpoints in your code
`breakpoint` is a macro part of `riscv/common`, which means that you don't need to include any special header files in your kernels. It compiles differently for each of brisc, ncrisc, and triscs, which means you can use it generically. Just insert `breakpoint();` into your code wherever you need it.

For example, below is a dummy brisc kernel running on core 1-1:

.. code-block:: cpp

    #include "dataflow_api.h"

    void kernel_main() {
        volatile uint32_t dst_addr  = get_arg_val<uint32_t>(0);
        uint32_t dst_noc_x = get_arg_val<uint32_t>(1);
        uint32_t dst_noc_y = get_arg_val<uint32_t>(2);
        uint32_t num_tiles = get_arg_val<uint32_t>(3);

        constexpr uint32_t cb_id_out0 = 16;

        // single-tile ublocks
        uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);
        uint32_t ublock_size_tiles = 1;




        breakpoint();
        ...


When I run my test, as soon as this breakpoint is reached, it opens up an interactive debugger. You will be greeted with this prompt:

.. image:: ../_static/grid-debugger.png
    :width: 400
    :alt: First login


The blue core represents the core you are currently hovering over, and the red cores represent cores in which a breakpoint was hit. If you are hovering over a core that has a breakpoint, it will appear as grey.

You can move around the grid with the arrow keys, and when you would like to debug a particular core, press enter on that core. For example, if I move to core 1-1 and press enter, I will see this prompt:

.. image:: ../_static/core-view.png
    :width: 400
    :alt: First login

Gray will represent your current cursor, so in this image, we are hovering over trisc0. Just like before, you can use the arrow keys to move around. Since we hit a breakpoint for brisc, you will see a message notifying you that a breakpoint has been hit, as well as which line the breakpoint was on.

TODO(agrebenisan): In the future, would like to make this a hyperlink that brings you directly to the file and line number.

From this screen, we can select a particular thread to debug. For example, if we move to the brisc thread and press enter, we will see this prompt:

.. code-block:: bash

    (tt_gdb)

Here, you can write `help` or `h` to display available commands:

.. code-block:: bash

    (tt_gdb) help
    Documented commands (type help <topic>):
    ========================================

    c       e       h    help    p       q


You can get more info by following the above instructions, however here is one example of printing a local variable:

.. code-block:: bash

    (tt_gdb) p dst_addr
    536870912

You may notice I specifically chose to print out a volatile variable, since volatile variables live in L1. So far, non-L1 variable printing is not supported, so you may need to modify your code to make local variables volatile for the time being.
