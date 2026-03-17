What is TT-NN?
##############

:doc:`TT-NN </ttnn/about>` is an open source library of neural network operations built on the ``tt-metal`` programming model.

It can be used in C++ and Python projects, and is designed to feel familiar to developers experienced with `PyTorch <https://pytorch.org/>`_.

Key features of :doc:`TT-NN </ttnn/about>` include:

* More than 200 :doc:`operations </ttnn/api>` (matrix multiplication, convolution, reduction, CCL, fused Transformer operations, etc.)
* A Tensor type that enables `different ways <https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/tensor_layouts/tensor_layouts.md>`_ to represent, distribute and access data on device
* The ability for developers to register custom operations
* Native support for a `mesh of devices <https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md>`_
* Tools to trace and `visualize <https://github.com/tenstorrent/ttnn-visualizer>`_ the computation `graph <https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/graph-tracing.md>`_
* Utilities to cache converted parameters, significantly speeding up model loading on repeated runs
* A `comparison mode <https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/comparison-mode.md>`_ that enables debugging of long operation sequences against a known reference
