/// Copyright (c) 2021, FilipeCN.
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
///\file array_of_structures_view.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-07-14
///
///\brief

#ifndef HERMES_STORAGE_ARRAY_OF_STRUCTURES_VIEW_H
#define HERMES_STORAGE_ARRAY_OF_STRUCTURES_VIEW_H

#include <hermes/storage/struct_descriptor.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                               FORWARD DECLARATIONS
// *********************************************************************************************************************
template<MemoryLocation L> class ArrayOfStructs;

// *********************************************************************************************************************
//                                                                                                       AoSFieldView
// *********************************************************************************************************************
/// Provides access to a single struct field in an array of structs
/// \tparam T field data type
template<typename T>
class AoSFieldView {
public:
  // *******************************************************************************************************************
  //                                                                                                   FRIEND STRUCTS
  // *******************************************************************************************************************
  friend class ArrayOfStructs<MemoryLocation::HOST>;
  friend class ArrayOfStructs<MemoryLocation::DEVICE>;
  friend class ArrayOfStructs<MemoryLocation::UNIFIED>;
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \param data
  /// \param stride
  /// \param offset
  /// \param size struct count
  HERMES_DEVICE_CALLABLE AoSFieldView(u8 *data, u64 stride, u64 offset, size_t size) : data_{data}, stride_{stride},
                                                                                       offset_{offset}, size_{size} {}
  //                                                                                                       assignment
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       assignment
  AoSFieldView &operator=(const std::vector<T> &data) {
    for (u64 i = 0; i < data.size(); ++i)
      (*this)[i] = data[i];
    return *this;
  }
  HERMES_DEVICE_CALLABLE AoSFieldView &operator=(const T *data) {
    for (size_t i = 0; i < size_; ++i)
      (*this)[i] = data[i];
    return *this;
  }
  //                                                                                                           access
  /// \param i
  /// \return
  HERMES_DEVICE_CALLABLE T &operator[](size_t i) { return *reinterpret_cast<T *>(data_ + i * stride_ + offset_); }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE size_t size() const { return size_; }
  /// \param data
  HERMES_DEVICE_CALLABLE void setDataPtr(u8 *data) { data_ = data; }
private:
  u8 *data_{nullptr};
  u64 stride_{0};
  u64 offset_{0};
  size_t size_{0};
};

// *********************************************************************************************************************
//                                                                                                  ConstAoSFieldView
// *********************************************************************************************************************
/// Provides access to a single struct field in an array of structs
/// \tparam T field data type
template<typename T>
class ConstAoSFieldView {
public:
  // *******************************************************************************************************************
  //                                                                                                   FRIEND STRUCTS
  // *******************************************************************************************************************
  friend class ArrayOfStructs<MemoryLocation::HOST>;
  friend class ArrayOfStructs<MemoryLocation::DEVICE>;
  friend class ArrayOfStructs<MemoryLocation::UNIFIED>;
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \param data
  /// \param stride
  /// \param offset
  /// \param size struct count
  HERMES_DEVICE_CALLABLE ConstAoSFieldView(const u8 *data, u64 stride, u64 offset, size_t size) : data_{data},
                                                                                                  stride_{stride},
                                                                                                  offset_{offset},
                                                                                                  size_{size} {}
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                           access
  /// \param i
  /// \return
  HERMES_DEVICE_CALLABLE const T &operator[](size_t i) const {
    return *reinterpret_cast<const T *>(data_ + i * stride_ + offset_);
  }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  [[nodiscard]] HERMES_DEVICE_CALLABLE size_t size() const { return size_; }
  /// \param data
  HERMES_DEVICE_CALLABLE void setDataPtr(const u8 *data) { data_ = data; }
private:
  const u8 *data_{nullptr};
  u64 stride_{0};
  u64 offset_{0};
  size_t size_{0};
};

// *********************************************************************************************************************
//                                                                                                            AoSView
// *********************************************************************************************************************
class AoSView {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \param descriptor
  /// \param data
  /// \param size
  AoSView(const StructDescriptorView &descriptor, u8 *data, size_t size) :
      struct_descriptor{descriptor}, data_{data}, size_{size} {}
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE void setDataPtr(u8 *data) { data_ = data; }
  HERMES_DEVICE_CALLABLE size_t size() const { return size_; }
  //                                                                                                           access
  template<typename T>
  HERMES_DEVICE_CALLABLE const T &valueAt(u64 field_id, u64 i) const {
    return *reinterpret_cast<const T *>(data_ + i * struct_descriptor.struct_size
        + struct_descriptor.fields_[field_id].offset);
  }
  template<typename T>
  HERMES_DEVICE_CALLABLE T &valueAt(u64 field_id, u64 i) {
    return *reinterpret_cast<T *>(data_ + i * struct_descriptor.struct_size
        + struct_descriptor.fields_[field_id].offset);
  }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  const StructDescriptorView struct_descriptor;
private:
  u8 *data_{nullptr};
  size_t size_{0};
};

// *********************************************************************************************************************
//                                                                                                            AoSView
// *********************************************************************************************************************
class ConstAoSView {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \param descriptor
  /// \param data
  /// \param size
  ConstAoSView(const StructDescriptorView &descriptor, const u8 *data, size_t size) :
      struct_descriptor{descriptor}, data_{data}, size_{size} {}
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE void setDataPtr(const u8 *data) { data_ = data; }
  HERMES_DEVICE_CALLABLE size_t size() const { return size_; }
  //                                                                                                           access
  template<typename T>
  HERMES_DEVICE_CALLABLE const T &valueAt(u64 field_id, u64 i) const {
    return *reinterpret_cast<const T *>(data_ + i * struct_descriptor.struct_size
        + struct_descriptor.fields_[field_id].offset);
  }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  const StructDescriptorView struct_descriptor;
private:
  const u8 *data_{nullptr};
  size_t size_{0};
};

}

#endif // HERMES_STORAGE_ARRAY_OF_STRUCTURES_VIEW_H
