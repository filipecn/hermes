/* Copyright (c) 2020, FilipeCN.
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/// \file   aos.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2020-12-10

#pragma once

#include <hermes/base/str.h>
#include <hermes/storage/block.h>

#include <string>

namespace hermes::mem {

// *****************************************************************************
//                                                                         AoS
// *****************************************************************************
/// This class stores an array of structs that can be defined in runtime.
class AoS {
public:
  // ***************************************************************************
  //                                                                    Layout
  // ***************************************************************************
  /// Describes the structure that is stored in an array of structures
  class Layout {
  public:
    /// \brief Field description
    ///
    /// Consider a single field named "field_a" and defined as float[3]. Then
    /// name is "field_a", size is 3 * sizeof(float), offset is 0,
    /// component_count is 3 and type is DataType::F32
    struct Field {
      std::string name;
      u64 size{0};            //!< field size in bytes
      u64 offset{0};          //!< field offset in bytes inside structure
      u32 component_count{1}; //!< component count of data type
      DataType type{DataType::CUSTOM}; //!< data type id
    };

    Layout() = default;
    ~Layout() = default;

    /// Register field of structure.
    /// \note fields are assumed to be stored in the same order they are pushed
    /// \tparam T base data type
    /// \param name field name
    /// \return field push id
    template <typename T> u64 pushField(const std::string &name) {
      field_id_map_[name] = fields_.size();
      Field d = {name, sizeof(T), size_in_bytes_, 1, DataTypes::typeFrom<T>()};
      d.component_count = componentCountOf<T>();
      d.type = dataTypeOf<T>();
      fields_.emplace_back(d);
      size_in_bytes_ += d.size;
      return fields_.size() - 1;
    }

    /// \tparam T
    /// \param data
    /// \param field_id
    /// \param i
    /// \return
    template <typename T> T &valueAt(void *data, u64 field_id, u64 i) const {
      return *reinterpret_cast<T *>(reinterpret_cast<h_byte *>(data) +
                                    i * size_in_bytes_ +
                                    fields_[field_id].offset);
    }

    /// \tparam T
    /// \param data
    /// \param field_id
    /// \param i
    /// \return
    template <typename T>
    const T &valueAt(const void *data, u64 field_id, u64 i) const {
      return *reinterpret_cast<const T *>(
          reinterpret_cast<const h_byte *>(data) + i * size_in_bytes_ +
          fields_[field_id].offset);
    }

    /// \param field_id field push id
    /// \return field name
    const std::string &fieldName(u64 field_id) const;
    bool contains(const std::string &field_name) const;
    const std::vector<Field> &fields() const;
    u64 fieldId(const std::string &field_name) const;
    ptrdiff_t addressOffsetOf(u64 field_id, u64 i) const;
    u64 offsetOf(const std::string &field_name) const;
    u64 offsetOf(u64 field_id) const;
    u64 sizeOf(const std::string &field_name) const;
    u64 sizeOf(u64 field_id) const;
    inline u64 sizeInBytes() const { return size_in_bytes_; }

  private:
    u64 size_in_bytes_{0};
    std::vector<Field> fields_;
    std::unordered_map<std::string, u64> field_id_map_;

    friend class AoS;
    HERMES_TO_STRING_FRIEND(Layout)
  };

  /// Provides access to a single field
  /// \tparam T field data type
  template <typename T> class ConstFieldView {
  public:
    HERMES_CPU_GPU const T &operator[](size_t i) const {
      return *reinterpret_cast<const T *>(data_ + i * stride_ + offset_);
    }
    HERMES_CPU_GPU size_t size() const { return size_; }

  private:
    HERMES_CPU_GPU ConstFieldView(const h_byte *data, u64 stride, u64 offset,
                                  size_t size)
        : data_{data}, stride_{stride}, offset_{offset}, size_{size} {}

    const h_byte *data_{nullptr};
    u64 stride_{0};
    u64 offset_{0};
    size_t size_{0};

    friend class AoS;
  };

  /// Provides access to a single field
  /// \tparam T field data type
  template <typename T> class FieldView {
  public:
    operator ConstFieldView<T>() const {
      return ConstFieldView<T>(data_, stride_, offset_, size_);
    }

    FieldView &operator=(const std::vector<T> &data) {
      for (u64 i = 0; i < data.size(); ++i)
        (*this)[i] = data[i];
      return *this;
    }
    HERMES_CPU_GPU FieldView &operator=(const T *data) {
      for (size_t i = 0; i < size_; ++i)
        (*this)[i] = data[i];
      return *this;
    }

    /// \param i
    /// \return
    HERMES_CPU_GPU T &operator[](size_t i) {
      return *reinterpret_cast<T *>(data_ + i * stride_ + offset_);
    }
    HERMES_CPU_GPU const T &operator[](size_t i) const {
      return *reinterpret_cast<T *>(data_ + i * stride_ + offset_);
    }
    HERMES_CPU_GPU size_t size() const { return size_; }

  private:
    HERMES_CPU_GPU FieldView(h_byte *data, u64 stride, u64 offset, size_t size)
        : data_{data}, stride_{stride}, offset_{offset}, size_{size} {}

    h_byte *data_{nullptr};
    u64 stride_{0};
    u64 offset_{0};
    size_t size_{0};

    friend class AoS;
  };

  class View {
  public:
    void setDataPtr(h_byte *data) { data_ = data; }
    size_t size() const { return size_; }
    //                                                                  access
    template <typename T> const T &valueAt(u64 field_id, u64 i) const {
      return *reinterpret_cast<const T *>(data_ + i * layout.size_in_bytes_ +
                                          layout.fields_[field_id].offset);
    }
    template <typename T> T &valueAt(u64 field_id, u64 i) {
      return *reinterpret_cast<T *>(data_ + i * layout.size_in_bytes_ +
                                    layout.fields_[field_id].offset);
    }

    const Layout &layout;

  private:
    View(const Layout &descriptor, h_byte *data, size_t size)
        : layout{descriptor}, data_{data}, size_{size} {}

    h_byte *data_{nullptr};
    size_t size_{0};

    friend class AoS;
  };

  class ConstView {
  public:
    void setDataPtr(h_byte *data) { data_ = data; }
    size_t size() const { return size_; }
    //                                                                  access
    template <typename T> const T &valueAt(u64 field_id, u64 i) const {
      return *reinterpret_cast<const T *>(data_ + i * layout.size_in_bytes_ +
                                          layout.fields_[field_id].offset);
    }

    const Layout &layout;

  private:
    ConstView(const Layout &descriptor, const h_byte *data, size_t size)
        : layout{descriptor}, data_{data}, size_{size} {}

    const h_byte *data_{nullptr};
    size_t size_{0};

    friend class AoS;
  };

  AoS() = default;
  virtual ~AoS() noexcept;
  AoS(const AoS &rhs);
  AoS(AoS &&rhs) noexcept;

  AoS &operator=(AoS &&other) noexcept;
  AoS &operator=(const AoS &other);

  template <typename T> AoS &operator=(std::vector<T> &&vector_data) {
    /// TODO: move operation is making a copy instead!
    if (!layout_.size_in_bytes_) {
      HERMES_ERROR(
          "[AoS] Fields must be previously registered befor vector assign.");
      return *this;
    }
    if (vector_data.size() * sizeof(T) % layout_.size_in_bytes_ != 0)
      HERMES_WARN("[AoS] Vector data with incompatible size.");
    HERMES_CHECK_HE_RESULT(data_.clear());
    size_ = vector_data.size() * sizeof(T) / layout_.size_in_bytes_;
    HERMES_CHECK_HE_RESULT(data_.resize(size_ * layout_.size_in_bytes_));
    HERMES_CHECK_HE_RESULT(
        data_.copy(vector_data.data(), size_ * layout_.size_in_bytes_));
    return *this;
  }

  HERMES_NODISCARD HeError clear() noexcept;
  u64 size() const;
  u64 dataSize() const;
  u64 stride() const;
  HERMES_NODISCARD HeError setLayout(const Layout &layout);
  HERMES_NODISCARD const Layout &layout() const;
  /// \param new_size in number of elements_
  HERMES_NODISCARD HeError resize(u64 count);
  HERMES_NODISCARD const Block &data() const;
  HERMES_NODISCARD Block &data();

  View view() { return View(layout_, data_.bytes(), size_); }
  ConstView view() const { return ConstView(layout_, data_.bytes(), size_); }
  template <typename T> FieldView<T> field(u64 field_id) {
    if (field_id >= layout_.fields().size()) {
      HERMES_ERROR("Field with id {} not found.", field_id);
      return FieldView<T>(nullptr, 0, 0, 0);
    }
    return FieldView<T>(data_.bytes(), layout_.size_in_bytes_,
                        layout_.fields_[field_id].offset, size_);
  }
  template <typename T> FieldView<T> field(const std::string &name) {
    auto it = layout_.field_id_map_.find(name);
    if (it == layout_.field_id_map_.end()) {
      HERMES_ERROR("Field {} not found.", name);
      return FieldView<T>(nullptr, 0, 0, 0);
    }
    return FieldView<T>(data_.bytes(), layout_.size_in_bytes_,
                        layout_.fields_[it->second].offset, size_);
  }
  template <typename T> ConstFieldView<T> field(u64 field_id) const {
    if (field_id >= layout_.fields().size()) {
      HERMES_ERROR("Field with id {} not found.", field_id);
      return ConstFieldView<T>(nullptr, 0, 0, 0);
    }
    return ConstFieldView<T>(data_.bytes(), layout_.size_in_bytes_,
                             layout_.fields_[field_id].offset, size_);
  }
  template <typename T> ConstFieldView<T> field(const std::string &name) const {
    auto it = layout_.field_id_map_.find(name);
    if (it == layout_.field_id_map_.end()) {
      HERMES_ERROR("Field {} not found.", name);
      return ConstFieldView<T>(nullptr, 0, 0, 0);
    }
    return ConstFieldView<T>(data_.bytes(), layout_.size_in_bytes_,
                             layout_.fields_[it->second].offset, size_);
  }

  template <typename T> u64 pushField(const std::string &name = "") {
    u64 new_field_id = 0;
    std::string field_name = name;
    if (name.empty())
      field_name = cstr::concat("field_", layout_.fields_.size());
    // increase buffer if necessary
    if (size_) {
      Layout desc = layout_;
      new_field_id = desc.template pushField<T>(field_name);
      // allocate memory
      Block new_data =
          Block::Config().setSize(desc.size_in_bytes_ * size_).create().value();
      auto ptr = data_.bytes();
      if (data_.location() == MemoryLocation::DEVICE) {
        // TODO
        HERMES_NOT_IMPLEMENTED
      }
      for (size_t i = 0; i < size_; ++i) {
        // since all fields remain in order, we can copy the entire struct
        auto buffer_offset = layout_.addressOffsetOf(0, i);
        auto new_buffer_offset = desc.addressOffsetOf(0, i);
        std::memcpy(new_data.bytes() + new_buffer_offset, ptr + buffer_offset,
                    layout_.size_in_bytes_);
      }
      layout_ = desc;
      data_ = new_data;
    } else
      new_field_id = layout_.pushField<T>(field_name);
    return new_field_id;
  }
  /// \return
  template <typename T> T &valueAt(u64 field_id, u64 i) {
    return *reinterpret_cast<T *>(data_.bytes() + i * layout_.size_in_bytes_ +
                                  layout_.fields_[field_id].offset);
  }
  template <typename T> const T &valueAt(u64 field_id, u64 i) const {
    return *reinterpret_cast<const T *>(data_.bytes() +
                                        i * layout_.size_in_bytes_ +
                                        layout_.fields_[field_id].offset);
  }
  template <typename T> const T &back(u64 field_id) const {
    return *reinterpret_cast<const T *>(data_.bytes() +
                                        (size_ - 1) * layout_.size_in_bytes_ +
                                        layout_.fields_[field_id].offset);
  }
  template <typename T> T &back(u64 field_id) {
    return *reinterpret_cast<T *>(data_.bytes() +
                                  (size_ - 1) * layout_.size_in_bytes_ +
                                  layout_.fields_[field_id].offset);
  }
  /*
    std::string dumpMemory(memory_dumper_options options =
                               memory_dumper_options::colored_output |
                               memory_dumper_options::type_values) const {
      auto layout =
          MemoryDumper::RegionLayout().withSize(layout_.sizeInBytes(), size_);

      for (size_t i = 0; i < layout_.fields_.size(); ++i) {
        const auto &f = layout_.fields_[i];
        auto field_layout = MemoryDumper::RegionLayout()
                                .withSize(f.size, f.component_count)
                                .withType(f.type);
        layout.pushSubRegion(
            field_layout.withColor(ConsoleColors::color((i % 3) + 2)));
      }
      return MemoryDumper::dump<byte>(data_.ptr(), data_.sizeInBytes(), 16,
    layout, options);
    }
  */
private:
  h_size size_{0}; //!< struct count
  Layout layout_;
  Block data_;

  HERMES_TO_STRING_FRIEND(AoS);
};

} // namespace hermes::mem

namespace hermes {

HERMES_DECLARE_TO_STRING_DEBUG_METHOD(mem::AoS::Layout);
HERMES_DECLARE_TO_STRING_DEBUG_METHOD(mem::AoS);

} // namespace hermes

/*
inline std::ostream &operator<<(std::ostream &o,
                              const AoS<MemoryLocation::HOST> &aos) {
#define PRINT_FIELD_VALUE(T, Type)                                             \
if (f.type == DataType::Type) {                                              \
  const T *ptr =                                                             \
      reinterpret_cast<const T *>(aos.data() + offset + f.offset);           \
  for (u32 j = 0; j < f.component_count; ++j)                                \
    o << ptr[j] << ((j < f.component_count - 1) ? " " : "");                 \
  o << ") ";                                                                 \
}
o << "AoS (struct count: " << aos.size()
  << ") (struct size in bytes: " << aos.structDescriptor().sizeInBytes()
  << ")\n";
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
    PRINT_FIELD_VALUE(byte, byte)
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
inline std::ofstream &operator<<(std::ofstream &o,
                               const AoS<MemoryLocation::HOST> &aos) {
size_t size = aos.size();
o.write(reinterpret_cast<const char *>(&size), sizeof(u64));
o << aos.structDescriptor();
o.write(reinterpret_cast<const char *>(aos.data()), aos.memorySizeInBytes());
return o;
}
inline std::ifstream &operator>>(std::ifstream &i,
                               AoS<MemoryLocation::HOST> &aos) {
aos = AoS<MemoryLocation::HOST>();
size_t size = 0;
i.read(reinterpret_cast<char *>(&size), sizeof(u64));
i >> aos.structDescriptor();
aos.resize(size);
i.read(reinterpret_cast<char *>(aos.data()), aos.memorySizeInBytes());
return i;
}
*/
