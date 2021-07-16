//
// Created by filipecn on 29/06/2021.
//


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
///\file array_view.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-06-29
///
///\brief

#ifndef HERMES_HERMES_STORAGE_ARRAY_VIEW_H
#define HERMES_HERMES_STORAGE_ARRAY_VIEW_H

#include <hermes/common/index.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                               FORWARD DECLARATIONS
// *********************************************************************************************************************
template<typename T, MemoryLocation L> class DataArray;
/// forward declaration of Array1
template<typename T> class Array1;
/// forward declaration of Array2
template<typename T> class Array2;

// *********************************************************************************************************************
//                                                                                                         ARRAY VIEW
// *********************************************************************************************************************
template<typename T>
class ArrayView {
public:
  // *******************************************************************************************************************
  //                                                                                                   FRIEND STRUCTS
  // *******************************************************************************************************************
  friend class DataArray<T, MemoryLocation::HOST>;
  friend class DataArray<T, MemoryLocation::DEVICE>;
  friend class DataArray<T, MemoryLocation::UNIFIED>;
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  //                                                                                                              new
  HERMES_DEVICE_CALLABLE ~ArrayView() {};
  HERMES_DEVICE_CALLABLE ArrayView(const ArrayView<T> &other) : size{other.size}, data_{other.data_},
                                                                pitch_{other.pitch_} {}
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                         1-access
  /// Access as a 1-dimensional array
  /// \param i 1-dimensional index
  /// \return (data ptr)[i]
  HERMES_DEVICE_CALLABLE const T &operator[](size_t i) const {
    return reinterpret_cast<const T *>(data_)[i];
  }
  /// Access as a 1-dimensional array
  /// \param i 1-dimensional index
  /// \return (data ptr)[i]
  HERMES_DEVICE_CALLABLE T &operator[](size_t i) {
    return reinterpret_cast<T *>(data_)[i];
  }
  //                                                                                                         2-access
  /// Access as a 2-dimensional array
  /// \param ij 2-dimensional index
  /// \return (data ptr) + j * pitch + i
  HERMES_DEVICE_CALLABLE const T &operator[](index2 ij) const {
    return reinterpret_cast<const T * >(data_ + ij.j * pitch_ + ij.i * sizeof(T))[0];
  }
  /// Access as a 2-dimensional array
  /// \param ij 2-dimensional index
  /// \return (data ptr) + j * pitch + i
  HERMES_DEVICE_CALLABLE T &operator[](index2 ij) {
    return reinterpret_cast<T * >(data_ + ij.j * pitch_ + ij.i * sizeof(T))[0];
  }
  //                                                                                                         3-access
  /// Access as a 3-dimensional array
  /// \param ijk 3-dimensional index
  /// \return (data ptr) + j * pitch + i
  HERMES_DEVICE_CALLABLE const T &operator[](index3 ijk) const {
    return reinterpret_cast<const T *>( data_ + ijk.k * pitch_ * size.height + ijk.j * pitch_
        + ijk.i * sizeof(T))[0];
  }
  /// Access as a 3-dimensional array
  /// \param ijk 3-dimensional index
  /// \return (data ptr) + j * pitch + i
  HERMES_DEVICE_CALLABLE T &operator[](index3 ijk) {
    return reinterpret_cast<T *>( data_ + ijk.k * pitch_ * size.height + ijk.j * pitch_
        + ijk.i * sizeof(T))[0];
  }
  const size3 size;
private:
  ArrayView(byte *data, size3 size, size_t pitch) : data_{data}, size{size}, pitch_{pitch} {}
  byte *data_{nullptr};
  size_t pitch_{0};
};

// *********************************************************************************************************************
//                                                                                                   CONST ARRAY VIEW
// *********************************************************************************************************************
template<typename T>
class ConstArrayView {
public:
  // *******************************************************************************************************************
  //                                                                                                   FRIEND STRUCTS
  // *******************************************************************************************************************
  friend class DataArray<T, MemoryLocation::HOST>;
  friend class DataArray<T, MemoryLocation::DEVICE>;
  friend class DataArray<T, MemoryLocation::UNIFIED>;
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  //                                                                                                              new
  ~ConstArrayView() = default;
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                         1-access
  /// Access as a 1-dimensional array
  /// \param i 1-dimensional index
  /// \return (data ptr)[i]
  HERMES_DEVICE_CALLABLE const T &operator[](size_t i) const {
    return reinterpret_cast<const T *>(data_)[i];
  }
  //                                                                                                         2-access
  /// Access as a 2-dimensional array
  /// \param ij 2-dimensional index
  /// \return (data ptr) + j * pitch + i
  HERMES_DEVICE_CALLABLE const T &operator[](index2 ij) const {
    return reinterpret_cast<const T * >(data_ + ij.j * pitch_ + ij.i * sizeof(T))[0];
  }
  //                                                                                                         3-access
  /// Access as a 3-dimensional array
  /// \param ijk 3-dimensional index
  /// \return (data ptr) + j * pitch + i
  HERMES_DEVICE_CALLABLE const T &operator[](index3 ijk) const {
    return reinterpret_cast<const T *>( data_ + ijk.k * pitch_ * size.height + ijk.j * pitch_
        + ijk.i * sizeof(T))[0];
  }
  const size3 size;
private:
  ConstArrayView(const byte *data, size3 size, size_t pitch) : data_{data}, size{size}, pitch_{pitch} {}
  const byte *data_{nullptr};
  size_t pitch_{0};
};
// *********************************************************************************************************************
//                                                                                                     Array1Iterator
// *********************************************************************************************************************
/// Auxiliary class to iterate an Array1 inside a for loop.
/// Ex: for(auto e : array) { e.value = x; }
///\tparam T Array1 data type
template<typename T> class Array1Iterator {
public:
  // *******************************************************************************************************************
  //                                                                                                   FRIEND STRUCTS
  // *******************************************************************************************************************
  friend class Array1<T>;
  // *******************************************************************************************************************
  //                                                                                                    Element class
  // *******************************************************************************************************************
  class Element {
    friend class Array1Iterator<T>;
  public:
    T &value;
    HERMES_DEVICE_CALLABLE Element &operator=(const T &v) {
      value = v;
      return *this;
    }
    HERMES_DEVICE_CALLABLE bool operator==(const T &v) const { return value == v; }
    const i32 index;
  private:
    Element(T &v, i32 i) : value(v), index(i) {}
  };
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  [[nodiscard]] u64 size() const { return size_; }
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       arithmetic
  Array1Iterator &operator++() {
    i++;
    if (i >= size_)
      i = -1;
    return *this;
  }
  //                                                                                                           access
  Element operator*() {
    return Element(data_[i], i);
  }
  //                                                                                                          boolean
  bool operator==(const Array1Iterator &other) {
    return size_ == other.size_ && data_ == other.data_ && i == other.i;
  }
  bool operator!=(const Array1Iterator &other) {
    return size_ != other.size_ || data_ != other.data_ || i != other.i;
  }

private:
  Array1Iterator(T *data, size_t size, i32 i) :
      data_(data), size_{size}, i(i) {}
  T *data_{nullptr};
  size_t size_{0};
  int i = 0;
};

// *********************************************************************************************************************
//                                                                                                ConstArray1Iterator
// *********************************************************************************************************************
/// Auxiliary class to iterate an const Array1 inside a for loop.
/// Ex: for(auto e : array) { e.value = x; }
///\tparam T Array1 data type
template<typename T> class ConstArray1Iterator {
public:
  // *******************************************************************************************************************
  //                                                                                                   FRIEND STRUCTS
  // *******************************************************************************************************************
  friend class Array1<T>;
  // *******************************************************************************************************************
  //                                                                                                    Element class
  // *******************************************************************************************************************
  class Element {
    friend class ConstArray1Iterator<T>;
  public:
    bool operator==(const T &v) const { return value == v; }
    const T &value;
    const i32 index;
  private:
    Element(const T &v, i32 i) : value(v), index(i) {}
  };
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  [[nodiscard]] u64 size() const { return size_; }
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       arithmetic
  ConstArray1Iterator &operator++() {
    if (++i >= size_)
      i = -1;
    return *this;
  }
  //                                                                                                           access
  Element operator*() {
    return Element(data_[i], i);
  }
  //                                                                                                          boolean
  bool operator==(const ConstArray1Iterator &other) {
    return size_ == other.size_ && data_ == other.data_ && i == other.i;
  }
  bool operator!=(const ConstArray1Iterator &other) {
    return size_ != other.size_ || data_ != other.data_ || i != other.i;
  }

private:
  ConstArray1Iterator(const T *data, size_t size, i32 i)
      : data_(data), size_{size}, i(i) {}
  const T *data_{nullptr};
  size_t size_{0};
  int i = 0;
};

// *********************************************************************************************************************
//                                                                                                     Array2Iterator
// *********************************************************************************************************************
/// Auxiliary class to iterate an Array2 inside a for loop.
/// Ex: for(auto e : array) { e.value = x; }
///\tparam T Array2 data type
template<typename T> class Array2Iterator {
public:
  // *******************************************************************************************************************
  //                                                                                                   FRIEND STRUCTS
  // *******************************************************************************************************************
  friend class Array2<T>;
  // *******************************************************************************************************************
  //                                                                                                    Element class
  // *******************************************************************************************************************
  class Element {
    friend class Array2Iterator<T>;
  public:
    Element &operator=(const T &v) {
      value = v;
      return *this;
    }
    bool operator==(const T &v) const { return value == v; }
    /// \return j * width + i
    [[nodiscard]] u64 flatIndex() const { return index.j * width_ + index.i; }
    T &value;
    const index2 index;
  private:
    Element(T &v, const index2 &ij, u64 width) : value(v), index(ij), width_{width} {}
    u64 width_{0};
  };
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  [[nodiscard]] size2 size() const { return size_; }
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       arithmetic
  Array2Iterator &operator++() {
    i++;
    if (i >= static_cast<i64>(size_.width)) {
      i = 0;
      j++;
      if (j >= static_cast<i64>(size_.height)) {
        i = j = -1;
      }
    }
    return *this;
  }
  //                                                                                                           access
  Element operator*() {
    return Element((T &) (*((char *) data_ + j * pitch_ + i * sizeof(T))),
                   index2(i, j), size_.width);
  }
  //                                                                                                          boolean
  bool operator==(const Array2Iterator &other) {
    return size_ == other.size_ && data_ == other.data_ && i == other.i &&
        j == other.j && pitch_ == other.pitch_;
  }
  bool operator!=(const Array2Iterator &other) {
    return size_ != other.size_ || data_ != other.data_ || i != other.i ||
        j != other.j || pitch_ != other.pitch_;
  }

private:
  Array2Iterator(T *data, const size2 &size, size_t pitch, const index2 &ij)
      : size_(size), data_(data), pitch_(pitch), i(ij.i), j(ij.j) {}
  size2 size_;
  T *data_ = nullptr;
  size_t pitch_ = 0;
  int i = 0, j = 0;
};

// *********************************************************************************************************************
//                                                                                                ConstArray1Iterator
// *********************************************************************************************************************
/// Auxiliary class to iterate an const Array2 inside a for loop.
/// Ex: for(auto e : array) { e.value = x; }
///\tparam T Array2 data type
template<typename T> class ConstArray2Iterator {
public:
  // *******************************************************************************************************************
  //                                                                                                   FRIEND STRUCTS
  // *******************************************************************************************************************
  friend class Array2<T>;
  // *******************************************************************************************************************
  //                                                                                                    Element class
  // *******************************************************************************************************************
  class Element {
    friend class ConstArray2Iterator<T>;
  public:
    bool operator==(const T &v) const { return value == v; }
    /// \return j * width + i
    [[nodiscard]] u64 flatIndex() const { return index.j * width_ + index.i; }
    const T &value;
    const index2 index;
  private:
    Element(const T &v, const index2 &ij, u64 width) : value(v), index(ij), width_(width) {}
    u64 width_{0};
  };
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  [[nodiscard]] size2 size() const { return size_; }
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       arithmetic
  ConstArray2Iterator &operator++() {
    i++;
    if (i >= static_cast<i64>(size_.width)) {
      i = 0;
      j++;
      if (j >= static_cast<i64>(size_.height)) {
        i = j = -1;
      }
    }
    return *this;
  }
  //                                                                                                           access
  Element operator*() {
    return Element((const T &) (*((const char *) data_ + j * pitch_ + i * sizeof(T))),
                   index2(i, j), size_.width);
  }
  //                                                                                                          boolean
  bool operator==(const ConstArray2Iterator &other) {
    return size_ == other.size_ && data_ == other.data_ && i == other.i &&
        j == other.j && pitch_ == other.pitch_;
  }
  bool operator!=(const ConstArray2Iterator &other) {
    return size_ != other.size_ || data_ != other.data_ || i != other.i ||
        j != other.j || pitch_ != other.pitch_;
  }

private:
  ConstArray2Iterator(const T *data, const size2 &size, size_t pitch, const index2 &ij)
      : size_(size), data_(data), pitch_(pitch), i(ij.i), j(ij.j) {}
  size2 size_;
  const T *data_ = nullptr;
  size_t pitch_ = 0;
  int i = 0, j = 0;
};

}

#endif //HERMES_HERMES_STORAGE_ARRAY_VIEW_H
