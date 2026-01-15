Tools
=====

.. note::
   Tools are only fully supported on source builds.

.. toctree::
    :maxdepth: 1

    kernel_print

Kernel print is the debug tool for printing tiles, scalers, and strings from device to host.

.. toctree::
    :maxdepth: 1

    watcher

The Watcher is a thread that monitors the status of the TT device to help with
debug.

.. toctree::
    :maxdepth: 1

    tracy_profiler

Tracy profiler is for profiling device-side RISCV code and host-side python and C++ code.

.. toctree::
    :maxdepth: 1

    device_program_profiler

Device program profiler brings visibility to execution of device side programs by providing duration counts on marked portions of the code.

.. toctree::
    :maxdepth: 1

    lightweight_kernel_asserts

Lightweight kernel asserts provide a mechanism for assertion checks within the kernel.

.. toctree::
    :maxdepth: 1

    llk_asserts

LLK asserts provide validation checks within the low-level kernel library infrastructure code.

.. toctree::
    :maxdepth: 1

    inspector

The Inspector is a tool that provides insights into Metal host runtime.

.. toctree::
    :maxdepth: 1

    triage

The tt-triage is a collection of Python scripts for analyzing and debugging Metal workload.

* `tt-smi <https://github.com/tenstorrent/tt-smi>`_

TT-SMI is a command line utility to interact with all Tenstorrent devices on host.
