EnqueueWriteBuffer
==================

.. doxygenfunction:: tt::tt_metal::EnqueueWriteBuffer(CommandQueue& cq, const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > &buffer, std::vector<DType>&, bool blocking)
.. doxygenfunction:: tt::tt_metal::EnqueueWriteBuffer(CommandQueue& cq, const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > &buffer, HostDataType src, bool blocking)
