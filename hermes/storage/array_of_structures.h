/// Copyright (c) 2020, FilipeCN.
///
/// The MIT License (MIT)
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.
///
///\file array_of_structures.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-12-10
///
///\brief

#ifndef HERMES_STORAGE_ARRAY_OF_STRUCTURES_H
#define HERMES_STORAGE_ARRAY_OF_STRUCTURES_H

#include <string>
#include <hermes/logging/logging.h>
#include <hermes/storage/memory_block.h>
#include <hermes/storage/array_of_structures_view.h>
#include "struct_descriptor.h"
#include <hermes/logging/memory_dump.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                     ArrayOfStructs
// *********************************************************************************************************************
/// DataArray of Structures
/// This class stores an array of structures that can be defined in runtime
template<MemoryLocation L>
class ArrayOfStructs {
public:
  // *******************************************************************************************************************
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  template<MemoryLocation LL>
  friend class ArrayOfStructs;
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  ArrayOfStructs() = default;
  virtual ~ArrayOfStructs() {
    data_.clear();
  }
  ArrayOfStructs(const ArrayOfStructs<MemoryLocation::HOST> &other) { *this = other; }
  ArrayOfStructs(const ArrayOfStructs<MemoryLocation::DEVICE> &other) { *this = other; }
  ArrayOfStructs(const ArrayOfStructs<MemoryLocation::UNIFIED> &other) { *this = other; }
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       assignment
  ArrayOfStructs &operator=(ArrayOfStructs &&other) noexcept {
    data_ = std::move(other.data_);
    size_ = other.size_;
    struct_descriptor_.struct_size_ = other.struct_descriptor_.struct_size_;
    struct_descriptor_.fields_ = std::move(other.struct_descriptor_.fields_);
    struct_descriptor_.field_id_map_ = std::move(other.struct_descriptor_.field_id_map_);
    return *this;
  }
  ArrayOfStructs &operator=(const ArrayOfStructs &other) {
    data_.clear();
    size_ = other.size_;
    struct_descriptor_.struct_size_ = other.struct_descriptor_.struct_size_;
    struct_descriptor_.fields_ = other.struct_descriptor_.fields_;
    struct_descriptor_.field_id_map_ = other.struct_descriptor_.field_id_map_;
    if (size_ && struct_descriptor_.struct_size_) {
      data_.resize(size_ * struct_descriptor_.struct_size_);
      data_ = other.data_;
    }
    return *this;
  }
  template<typename T>
  ArrayOfStructs &operator=(std::vector<T> &&vector_data) {
    if (L == MemoryLocation::DEVICE) {
      HERMES_NOT_IMPLEMENTED
      return *this;
    }
    /// TODO: move operation is making a copy instead!
    if (!struct_descriptor_.struct_size_) {
      Log::warn("[ArrayOfStructs] Fields must be previously registered.");
      return *this;
    }
    if (vector_data.size() * sizeof(T) % struct_descriptor_.struct_size_ != 0)
      Log::warn("[ArrayOfStructs] Vector data with incompatible size.");
    data_.clear();
    size_ = vector_data.size() * sizeof(T) / struct_descriptor_.struct_size_;
    data_.resize(size_ * struct_descriptor_.struct_size_);
    std::memcpy(data_.ptr(), vector_data.data(), size_ * struct_descriptor_.struct_size_);
    return *this;
  }
  template<MemoryLocation LL>
  ArrayOfStructs &operator=(const ArrayOfStructs<LL> &other) {
    data_ = other.data_;
    size_ = other.size_;
    struct_descriptor_ = other.struct_descriptor_;
    return *this;
  }
  template<MemoryLocation LL>
  ArrayOfStructs &operator=(ArrayOfStructs<LL> &&other) {
    data_ = std::move(other.data_);
    size_ = other.size_;
    struct_descriptor_ = std::move(other.struct_descriptor_);
    return *this;
  }
  // *******************************************************************************************************************
  //                                                                                                          METRICS
  // *******************************************************************************************************************
  [[nodiscard]] inline u64 size() const { return size_; }
  [[nodiscard]] inline u64 memorySizeInBytes() const { return size_ * struct_descriptor_.struct_size_; }
  [[nodiscard]] inline u64 stride() const { return struct_descriptor_.struct_size_; }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  const StructDescriptor &structDescriptor() const { return struct_descriptor_; }
  StructDescriptor &structDescriptor() { return struct_descriptor_; }
  void setStructDescriptor(const StructDescriptor &new_struct_descriptor) {
    struct_descriptor_ = new_struct_descriptor;
  }
  /// \return
  [[nodiscard]] const u8 *data() const { return data_.ptr(); }
  [[nodiscard]] u8 *data() { return data_.ptr(); }
  /// \param new_size in number of elements_
  void resize(u64 new_size) {
    data_.clear();
    size_ = new_size;
    if (!new_size)
      return;
    data_.resize(new_size * struct_descriptor_.struct_size_);
  }
  //                                                                                                           access
  AoSView view() { return AoSView(struct_descriptor_.view(), data_.ptr(), size_); }
  ConstAoSView view() const { return ConstAoSView(struct_descriptor_.view(), data_.ptr(), size_); }
  ConstAoSView constView() const { return ConstAoSView(struct_descriptor_.view(), data_.ptr(), size_); }
  template<typename T>
  AoSFieldView<T> field(u64 field_id) {
    if (field_id >= struct_descriptor_.fields().size()) {
      Log::error("Field with id {} not found.", field_id);
      return AoSFieldView<T>(nullptr, 0, 0, 0);
    }
    return AoSFieldView<T>(data_.ptr(),
                           struct_descriptor_.struct_size_,
                           struct_descriptor_.fields_[field_id].offset,
                           size_);
  }
  template<typename T>
  AoSFieldView<T> field(const std::string &name) {
    auto it = struct_descriptor_.field_id_map_.find(name);
    if (it == struct_descriptor_.field_id_map_.end()) {
      Log::error("Field {} not found.", name);
      return AoSFieldView<T>(nullptr, 0, 0, 0);
    }
    return AoSFieldView<T>(data_.ptr(),
                           struct_descriptor_.struct_size_,
                           struct_descriptor_.fields_[it->second].offset,
                           size_);
  }
  template<typename T>
  ConstAoSFieldView<T> field(u64 field_id) const {
    if (field_id >= struct_descriptor_.fields().size()) {
      Log::error("Field with id {} not found.", field_id);
      return ConstAoSFieldView<T>(nullptr, 0, 0, 0);
    }
    return ConstAoSFieldView<T>(data_.ptr(),
                                struct_descriptor_.struct_size_,
                                struct_descriptor_.fields_[field_id].offset,
                                size_);
  }
  template<typename T>
  ConstAoSFieldView<T> field(const std::string &name) const {
    auto it = struct_descriptor_.field_id_map_.find(name);
    if (it == struct_descriptor_.field_id_map_.end()) {
      Log::error("Field {} not found.", name);
      return ConstAoSFieldView<T>(nullptr, 0, 0, 0);
    }
    return ConstAoSFieldView<T>(data_.ptr(),
                                struct_descriptor_.struct_size_,
                                struct_descriptor_.fields_[it->second].offset,
                                size_);
  }
  //                                                                                                            field
  template<typename T>
  u64 pushField(const std::string &name = "") {
    u64 new_field_id = 0;
    std::string field_name = name;
    if (name.empty())
      field_name = Str::concat("field_", struct_descriptor_.fields_.size());
    // increase buffer if necessary
    if (size_) {
      StructDescriptor desc = struct_descriptor_;
      new_field_id = desc.template pushField<T>(field_name);
      // allocate memory
      MemoryBlock<MemoryLocation::HOST> new_data(desc.struct_size_ * size_);
      auto ptr = data_.ptr();
      if (location == MemoryLocation::DEVICE) {
        // TODO
        HERMES_NOT_IMPLEMENTED
      }
      for (size_t i = 0; i < size_; ++i) {
        // since all fields remain in order, we can copy the entire struct
        auto buffer_offset = struct_descriptor_.addressOffsetOf(0, i);
        auto new_buffer_offset = desc.addressOffsetOf(0, i);
        std::memcpy(new_data.ptr() + new_buffer_offset, ptr + buffer_offset, struct_descriptor_.struct_size_);
      }
      struct_descriptor_ = desc;
      data_ = new_data;
    } else
      new_field_id = struct_descriptor_.pushField<T>(field_name);
    return new_field_id;
  }
  /// \return
  const std::vector<StructDescriptor::Field> &fields() const { return struct_descriptor_.fields_; }
  template<typename T>
  T &valueAt(u64 field_id, u64 i) {
    return *reinterpret_cast<T *>(data_.ptr() + i * struct_descriptor_.struct_size_
        + struct_descriptor_.fields_[field_id].offset);
  }
  template<typename T>
  const T &valueAt(u64 field_id, u64 i) const {
    return *reinterpret_cast<const T *>(data_.ptr() + i * struct_descriptor_.struct_size_
        + struct_descriptor_.fields_[field_id].offset);
  }
  template<typename T>
  const T &back(u64 field_id) const {
    return *reinterpret_cast<const T *>(data_.ptr() + (size_ - 1) * struct_descriptor_.struct_size_
        + struct_descriptor_.fields_[field_id].offset);
  }
  template<typename T>
  T &back(u64 field_id) {
    return *reinterpret_cast< T *>(data_.ptr() + (size_ - 1) * struct_descriptor_.struct_size_
        + struct_descriptor_.fields_[field_id].offset);
  }
  // *******************************************************************************************************************
  //                                                                                                            debug
  // *******************************************************************************************************************
  std::string dumpMemory(memory_dumper_options options = memory_dumper_options::colored_output
      | memory_dumper_options::type_values) const {
    auto layout = MemoryDumper::RegionLayout().withSize(struct_descriptor_.sizeInBytes(), size_);

    for (size_t i = 0; i < struct_descriptor_.fields_.size(); ++i) {
      const auto &f = struct_descriptor_.fields_[i];
      auto field_layout = MemoryDumper::RegionLayout().withSize(
          f.size, f.component_count).withType(f.type);
      layout.pushSubRegion(field_layout
                               .withColor(ConsoleColors::color((i % 3) + 2)));
    }
    return MemoryDumper::dump<u8>(data_.ptr(), data_.sizeInBytes(), 16, layout, options);
  }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  static const MemoryLocation location{L};
private:
  u64 size_{0}; //!< struct count
  StructDescriptor struct_descriptor_;
  MemoryBlock<L> data_;
};

// *********************************************************************************************************************
//                                                                                                                 IO
// *********************************************************************************************************************
inline std::ostream &operator<<(std::ostream &o, const ArrayOfStructs<MemoryLocation::HOST> &aos) {
#define PRINT_FIELD_VALUE(T, Type) \
        if (f.type == DataType::Type) { \
    const T *ptr = reinterpret_cast<const T *>(aos.data() + offset + f.offset); \
    for (u32 j = 0; j < f.component_count; ++j) \
      o << ptr[j] << ((j < f.component_count - 1) ?  " " : ""); \
    o << ") "; \
  }
  o << "ArrayOfStructs (struct count: " << aos.size() << ") (struct size in bytes: "
    << aos.structDescriptor().sizeInBytes() << ")\n";
  o << "fields: ";
  size_t k = 0;
  for (const auto &f : aos.structDescriptor().fields()) {
    o << "field #" << k++ << " (" << f.name << " ): ";
    o << "\tbase data type: " << DataTypes::typeName(f.type) << "\n";
    o << "\tbase data size in bytes: " << f.size << "\n";
    o << "\tcomponent count: " << f.component_count << "\n";
    o << "field values:\n";
    u64 offset = 0;
    for (u64 i = 0; i < aos.size(); ++i) {
      o << "[" << i << "](";
      PRINT_FIELD_VALUE(i8, I8)
      PRINT_FIELD_VALUE(i16, I16)
      PRINT_FIELD_VALUE(i32, I32)
      PRINT_FIELD_VALUE(i64, I64)
      PRINT_FIELD_VALUE(u8, U8)
      PRINT_FIELD_VALUE(u16, U16)
      PRINT_FIELD_VALUE(u32, U32)
      PRINT_FIELD_VALUE(u64, U64)
      PRINT_FIELD_VALUE(f32, F32)
      PRINT_FIELD_VALUE(f64, F64)
      offset += aos.structDescriptor().sizeInBytes();
    }
    o << std::endl;
  }
  return o;
#undef PRINT_FIELD_VALUE
}
inline std::ofstream &operator<<(std::ofstream &o, const ArrayOfStructs<MemoryLocation::HOST> &aos) {
  size_t size = aos.size();
  o.write(reinterpret_cast<const char *>(&size), sizeof(u64));
  o << aos.structDescriptor();
  o.write(reinterpret_cast<const char *>(aos.data()), aos.memorySizeInBytes());
  return o;
}
inline std::ifstream &operator>>(std::ifstream &i, ArrayOfStructs<MemoryLocation::HOST> &aos) {
  aos = ArrayOfStructs<MemoryLocation::HOST>();
  size_t size = 0;
  i.read(reinterpret_cast<char *>(&size), sizeof(u64));
  i >> aos.structDescriptor();
  aos.resize(size);
  i.read(reinterpret_cast<char *>(aos.data()), aos.memorySizeInBytes());
  return i;
}
// *********************************************************************************************************************
//                                                                                                           TYPEDEFS
// *********************************************************************************************************************
using AoS = ArrayOfStructs<MemoryLocation::HOST>;
using DeviceAoS = ArrayOfStructs<MemoryLocation::DEVICE>;
using UnifiedAoS = ArrayOfStructs<MemoryLocation::UNIFIED>;

}

#endif //HERMES_STORAGE_ARRAY_OF_STRUCTURES_H
