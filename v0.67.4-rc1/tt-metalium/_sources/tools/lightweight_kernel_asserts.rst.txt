Lightweight Kernel Asserts
==========================

Overview
--------

Lightweight Kernel Asserts provide a mechanism for assertion checks within kernels. They are designed to be lightweight to minimize performance impact while still offering valuable debugging information.
They are not intended to replace more comprehensive assertion mechanisms provided by the Watcher tool, but to complement them by providing a low-overhead option for critical checks.
The ASSERT macro expands into an if statement followed by a RISC-V instruction ebreak.
This causes the kernel to hang and enter debug mode when an assertion fails. You can attach with a debugger to inspect the state.
After a hang, you can run tt-triage to analyze the state. The script ``dump_lightweight_asserts.py`` will print call stacks for the failed assertions.

Enabling
--------

Configure the Lightweight Kernel Asserts by setting the following environment variable:

.. code-block::

   export TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1  # optional: enable/disable Lightweight Kernel Asserts. Default is `0` (disabled).
