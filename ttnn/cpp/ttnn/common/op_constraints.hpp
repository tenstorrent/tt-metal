#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "ttnn/cpp/ttnn/tensor/types.hpp"

class OpConstraint {
   private:
    // First input
    std::optional<tt::tt_metal::DataType> data_type_a;
    std::optional<tt::tt_metal::Layout> tile_layout_a;
    std::optional<tt::tt_metal::StorageType> storage_type_a;
    // Second input
    std::optional<tt::tt_metal::DataType> data_type_b;
    std::optional<tt::tt_metal::Layout> tile_layout_b;
    std::optional<tt::tt_metal::StorageType> storage_type_b;
    // Output
    std::optional<tt::tt_metal::DataType> data_type_o;
    std::optional<tt::tt_metal::Layout> tile_layout_o;
    std::optional<tt::tt_metal::StorageType> storage_type_o;

   public:
    // Constructor
    OpConstraint(
        std::optional<tt::tt_metal::DataType> dataTypeA = std::nullopt,
        std::optional<tt::tt_metal::Layout> tileLayoutA = std::nullopt,
        std::optional<tt::tt_metal::StorageType> storageTypeA = std::nullopt,
        std::optional<tt::tt_metal::DataType> dataTypeB = std::nullopt,
        std::optional<tt::tt_metal::Layout> tileLayoutB = std::nullopt,
        std::optional<tt::tt_metal::StorageType> storageTypeB = std::nullopt,
        std::optional<tt::tt_metal::DataType> dataTypeO = std::nullopt,
        std::optional<tt::tt_metal::Layout> tileLayoutO = std::nullopt,
        std::optional<tt::tt_metal::StorageType> storageTypeO = std::nullopt) :
        data_type_a(dataTypeA),
        tile_layout_a(tileLayoutA),
        storage_type_a(storageTypeA),
        data_type_b(dataTypeB),
        tile_layout_b(tileLayoutB),
        storage_type_b(storageTypeB),
        data_type_o(dataTypeO),
        tile_layout_o(tileLayoutO),
        storage_type_o(storageTypeO) {}

    std::optional<tt::tt_metal::DataType> getDataTypeA() const { return data_type_a; }
    std::optional<tt::tt_metal::Layout> getTileLayoutA() const { return tile_layout_a; }
    std::optional<tt::tt_metal::StorageType> getStorageTypeA() const { return storage_type_a; }

    std::optional<tt::tt_metal::DataType> getDataTypeB() const { return data_type_b; }
    std::optional<tt::tt_metal::Layout> getTileLayoutB() const { return tile_layout_b; }
    std::optional<tt::tt_metal::StorageType> getStorageTypeB() const { return storage_type_b; }

    std::optional<tt::tt_metal::DataType> getDataTypeO() const { return data_type_o; }
    std::optional<tt::tt_metal::Layout> getTileLayoutO() const { return tile_layout_o; }
    std::optional<tt::tt_metal::StorageType> getStorageTypeO() const { return storage_type_o; }
};

class OpConstraintsBuilder {
   private:
    uint32_t get_packed_size_in_bytes(const tt::tt_metal::DataType& data_type, const ttnn::Shape& shape) const;
    bool can_allocate_sharded_buffer(
        const MemoryConfig& memory_config,
        const ttnn::Shape& shape,
        const Layout& layout,
        const DataType& data_type) const;

   protected:
    std::optional<tt::tt_metal::DataType> data_type_a;  // required
    std::optional<tt::tt_metal::Layout> tile_layout_a;
    std::optional<tt::tt_metal::StorageType> storage_type_a;

    std::optional<tt::tt_metal::DataType> data_type_b;  // required
    std::optional<tt::tt_metal::Layout> tile_layout_b;
    std::optional<tt::tt_metal::StorageType> storage_type_b;

    std::optional<tt::tt_metal::DataType> data_type_o;  // required
    std::optional<tt::tt_metal::Layout> tile_layout_o;
    std::optional<tt::tt_metal::StorageType> storage_type_o;

    virtual bool can_build_constraints() const;
    virtual bool is_valid_external_constraint(const OpConstraint& constraint) const;
    virtual bool is_valid_op_constraint(const OpConstraint& constraint) const = 0;

    bool is_sharded_tensor_valid(
        const MemoryConfig& memory_config,
        const ttnn::Shape& shape,
        const Layout& layout,
        const DataType& data_type) const;

   public:
    virtual ~OpConstraintsBuilder() = default;
    virtual std::string get_op_name() const = 0;
    virtual std::vector<OpConstraint> build_constraints() = 0;

    // Setters for parameter a
    OpConstraintsBuilder& setDataTypeA(tt::tt_metal::DataType dataType);

    OpConstraintsBuilder& setTileLayoutA(tt::tt_metal::Layout tileLayout);

    OpConstraintsBuilder& setStorageTypeA(tt::tt_metal::StorageType storageType);

    // Setters for parameter b
    OpConstraintsBuilder& setDataTypeB(tt::tt_metal::DataType dataType);

    OpConstraintsBuilder& setTileLayoutB(tt::tt_metal::Layout tileLayout);

    OpConstraintsBuilder& setStorageTypeB(tt::tt_metal::StorageType storageType);

    // Setters for parameter output
    OpConstraintsBuilder& setDataTypeO(tt::tt_metal::DataType dataType);

    OpConstraintsBuilder& setTileLayoutO(tt::tt_metal::Layout tileLayout);

    OpConstraintsBuilder& setStorageTypeO(tt::tt_metal::StorageType storageType);
};

class OpConstraintsFactory {
   public:
    static const uint32_t Volume(const ttnn::Shape& shape);

    static bool Can_fit_op_on_chip(const MemoryConfig& memory_config, const CoreCoord& chip_grid);
};
