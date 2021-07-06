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
///\file array.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-29
///
///\brief

#ifndef HERMES_STORAGE_ARRAY_H
#define HERMES_STORAGE_ARRAY_H

#include <hermes/storage/memory_block.h>
#include <hermes/storage/array_view.h>
#include <hermes/common/index.h>
#include <hermes/common/size.h>
#include <hermes/common/str.h>
#include <iomanip>    // std::setw
#include <ios>        // std::left

namespace hermes {

// *********************************************************************************************************************
//                                                                                                              Array
// *********************************************************************************************************************
/// Holds a linear memory area that can be accessed as a 1-dimensional, 2-dimensional or a 3-dimensional array
/// of elements. The memory can live in host memory or device memory, as set by the template.
/// \note Elements in the array can be conveniently iterated by a foreach loop.
/// \note Elements are stored in first-dimension major
/// \tparam T data type
/// \tparam L memory space
template<typename T, MemoryLocation L>
class Array {
public:
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  // *******************************************************************************************************************
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  //                                                                                                              new
  Array() = default;
  ~Array() = default;
  Array(size_t size_in_elements) { resize(size_in_elements); }
  Array(size2 size_in_elements, size_t pitch = 0) { resize(size_in_elements, pitch); }
  Array(size3 size_in_elements, size_t pitch = 0) { resize(size_in_elements, pitch); }
  //                                                                                                       assignment
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       assignment
  template<MemoryLocation LL>
  Array &operator=(const Array<T, LL> &other) {
    if (*this == other)
      return *this;
    data_ = other.data_;
    size_ = other.size_;
    return *this;
  }
  template<MemoryLocation LL>
  Array &operator=(Array<T, LL> &&other) {
    if (*this == other)
      return *this;
    data_ = std::move(other.data_);
    size_ = other.size_;
    return *this;
  }
  //                                                                                                         1-access
  /// Access as a 1-dimensional array
  /// \param i 1-dimensional index
  /// \return (data ptr)[i]
  const T &operator[](size_t i) const {
    return reinterpret_cast<const T *>(data_.ptr())[i];
  }
  /// Access as a 1-dimensional array
  /// \param i 1-dimensional index
  /// \return (data ptr)[i]
  T &operator[](size_t i) {
    return reinterpret_cast<T *>(data_.ptr())[i];
  }
  //                                                                                                         2-access
  /// Access as a 2-dimensional array
  /// \param ij 2-dimensional index
  /// \return (data ptr) + j * pitch + i
  const T &operator[](index2 ij) const {
    return reinterpret_cast<const T * >(data_.ptr() + ij.j * data_.pitch() + ij.i * sizeof(T))[0];
  }
  /// Access as a 2-dimensional array
  /// \param ij 2-dimensional index
  /// \return (data ptr) + j * pitch + i
  T &operator[](index2 ij) {
    return reinterpret_cast<T * >(data_.ptr() + ij.j * data_.pitch() + ij.i * sizeof(T))[0];
  }
  //                                                                                                         3-access
  /// Access as a 3-dimensional array
  /// \param ijk 3-dimensional index
  /// \return (data ptr) + j * pitch + i
  const T &operator[](index3 ijk) const {
    return reinterpret_cast<const T *>( data_.ptr() + ijk.k * data_.pitch() * size_.height + ijk.j * data_.pitch()
        + ijk.i * sizeof(T))[0];
  }
  /// Access as a 3-dimensional array
  /// \param ijk 3-dimensional index
  /// \return (data ptr) + j * pitch + i
  T &operator[](index3 ijk) {
    return reinterpret_cast<T *>( data_.ptr() + ijk.k * data_.pitch() * size_.height + ijk.j * data_.pitch()
        + ijk.i * sizeof(T))[0];
  }
  //                                                                                                       arithmetic
  //                                                                                                          boolean
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  //                                                                                                             size
  /// \return
  [[nodiscard]] bool empty() const { return size_.total() == 0; }
  /// \return
  [[nodiscard]] u32 dimensions() const {
    if (size_.total() == 0)
      return 0;
    if (size_.depth == 1)
      return size_.height > 1 ? 2 : 1;
    return 3;
  }
  /// \return total allocated memory in bytes
  [[nodiscard]] size_t sizeInBytes() const { return data_.sizeInBytes(); }
  /// \return array size in elements
  [[nodiscard]] size3 size() const { return size_; }
  /// \param new_size_in_bytes
  void resize(size_t new_size) {
    size_ = {static_cast<u32>( new_size), 1, 1};
    data_.resize(new_size * sizeof(T));
  }
  /// \param new_size width in elements
  void resize(size2 new_size, size_t new_pitch = 0) {
    size_ = {new_size.width, new_size.height, 1};
    data_.resize(size2(size_.width * sizeof(T), size_.height), new_pitch);
  }
  /// \param new_size width in elements
  void resize(size3 new_size, size_t new_pitch = 0) {
    size_ = new_size;
    data_.resize(size3(size_.width * sizeof(T), size_.height, size_.depth), new_pitch);
  }
  void clear() {
    size_ = {0, 0, 0};
    data_.clear();
  }
  //                                                                                                           access
  const T *data() const { return reinterpret_cast<T *>( data_.ptr()); }
  T *data() { return reinterpret_cast<T *>(data_.ptr()); }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  const MemoryLocation location{L};
private:
  size3 size_;
  MemoryBlock<L> data_;
};



// *********************************************************************************************************************
//                                                                                                             Array1
// *********************************************************************************************************************
/// Holds a linear memory area representing a 1-dimensional array of
/// ``size`` elements_.
/// - Array1 provides a convenient way to access its elements_:
/// \verbatim embed:rst:leading-slashes
///    .. code-block:: cpp
///
///       for(auto e : my_array1) {
///         e.value = 0; // element value access
///         e.index; // element index
///       }
/// \endverbatim
/// \tparam T data type

///

template<class T> class Array1 {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  Array1() = default;
  /// \param size dimensions (in elements_ count)
  explicit Array1(u64 size) {
    resize(size);
  }
  /// Copy constructor
  /// \param other **[in]** const reference to other Array1 object
  Array1(const Array1 &other) {
    resize(other.size_);
    std::memcpy(data_, other.data_, memorySize());
  }
  Array1(const Array1 &&other) = delete;
  /// Assign constructor
  /// \param other **[in]** temporary Array2 object
  Array1(Array1 &&other) noexcept: data_(other.data_) {
    other.data_ = nullptr;
  }
  /// Constructs an Array1 from a std vector
  /// \param std_vector **[in]** data
  Array1(const std::vector<T> &std_vector) {
    resize(std_vector.size());
    std::memcpy(data_, std_vector.data(), size_ * sizeof(T));
  }
  /// Initialization list constructor
  /// \param list **[in]** data list
  /// \verbatim embed:rst:leading-slashes
  ///    **Example**::
  ///
  ///       hermes::Array1<i32> a = {1,2,3,4};
  /// \endverbatim
  Array1(std::initializer_list<T> list) {
    resize(list.size());
    for (u64 i = 0; i < size_; ++i)
      (*this)[i] = list.begin()[i];
  }
  ///
  virtual ~Array1() {
    delete[] reinterpret_cast<char *>(data_);
  }
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       assignment
  /// Assign operator from raw data
  /// \param std_vector **[in]** data
  /// \return Array1<T>&
  Array1<T> &operator=(const std::vector<T> &std_vector) {
    data_ = std_vector;
  }
  /// Assign operator
  /// \param other **[in]** const reference to other Array1 object
  /// \return Array1<T>&
  Array1<T> &operator=(const Array1<T> &other) {
    data_ = other.data_;
    return *this;
  }
  /// Assign operator
  /// \param other **[in]** temporary Array1 object
  /// \return Array2<T>&
  Array1<T> &operator=(Array1<T> &&other) noexcept {
    data_ = other.data_;
    return *this;
  }
  /// Assign ``value`` to all elements_
  /// \param value assign value
  /// \return *this
  Array1 &operator=(T value) {
    for (size_t i = 0; i < size_; ++i)
      reinterpret_cast<T *>(data_)[i] = value;
    return *this;
  }
  //                                                                                                           access
  /// \verbatim embed:rst:leading-slashes
  ///    .. warning::
  ///       This method does **not** check if ``i`` is out of bounds.
  /// \endverbatim
  /// \param i element index
  /// \return reference to element at ``i`` position
  T &operator[](u64 i) {
    return reinterpret_cast<T *>(data_)[i];
  }
  /// \verbatim embed:rst:leading-slashes
  ///    .. warning::
  ///       This method does **not** check if ``ij`` is out of bounds.
  /// \endverbatim
  /// \param i element index
  /// \return const reference to element at ``i`` position
  const T &operator[](u64 i) const {
    return reinterpret_cast<T *>(data_)[i];
  }
  /// \verbatim embed:rst:leading-slashes
  ///    .. warning::
  ///       This method does **not** check if ``i`` is out of bounds.
  /// \endverbatim
  /// \param i **[in]** index
  /// \return T& reference to element in position ``i``
  T &operator()(u64 i) {
    return reinterpret_cast<T *>(data_)[i];
  }
  /// \verbatim embed:rst:leading-slashes
  ///    .. warning::
  ///       This method does **not** check if ``i`` is out of bounds.
  /// \endverbatim
  /// \param i **[in]** index
  /// \return const reference to element at ``i`` position
  const T &operator()(u64 i) const {
    return reinterpret_cast<T *>(data_)[i];
  }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  //                                                                                                             size
  /// Changes the dimensions
  /// \verbatim embed:rst:leading-slashes
  ///    .. note::
  ///       All previous data is erased.
  /// \endverbatim
  /// \param new_size new row and column counts
  void resize(u64 new_size) {
    delete[](char *) data_;
    size_ = new_size;
    data_ = new T[size_];
  }
  /// Computes actual memory usage
  /// \return memory usage (in bytes)
  [[nodiscard]] u64 memorySize() const { return size_ * sizeof(T); }
  /// \return dimensions (in elements_ count)
  [[nodiscard]] u64 size() const { return size_; }
  //                                                                                                           access
  /// \return const pointer to raw data (**row major**)
  const T *data() const { return data_; }
  /// \return pointer to raw data (**row major**)
  T *data() { return data_; }
  /// Copies data from another Array1
  ///
  /// - This gets resized if necessary.
  /// \param other **[in]**
  void copy(const Array1 &other) {
    data_ = other.data();
  }
  /// Checks if ``i`` is not out of bounds
  /// \param ij position index
  /// \return ``true`` if position can be accessed
  [[nodiscard]] bool stores(i32 i) const {
    return i >= 0 && static_cast<i64>(i) < size_;
  }
  //                                                                                                        iterators
  Array1Iterator<T> begin() {
    return Array1Iterator<T>(reinterpret_cast<T *>( data_), size_, 0);
  }
  Array1Iterator<T> end() {
    return Array1Iterator<T>(reinterpret_cast<T *>( data_), size_, -1);
  }
  ConstArray1Iterator<T> begin() const {
    return ConstArray1Iterator<T>(reinterpret_cast<const T *>( data_), size_, 0);
  }
  ConstArray1Iterator<T> end() const {
    return ConstArray1Iterator<T>(reinterpret_cast<const T *>( data_), size_, -1);
  }

private:
  size_t size_{0};
  T *data_ = nullptr;
};

// *********************************************************************************************************************
//                                                                                                             Array2
// *********************************************************************************************************************
/// Holds a linear memory area representing a 2-dimensional array of
/// ``size.width`` * ``size.height`` elements_.
///
/// - Considering ``size.height`` rows of ``size.width`` elements_, data is
/// laid out in memory in **row major** fashion.
///
/// - It is also possible to set _row level_ memory alignment via a custom size
/// of allocated memory per row, called pitch size. The minimal size of pitch is
/// ``size.width``*``sizeof(T)``.
///
/// - The methods use the convention of ``i`` and ``j`` indices, representing
/// _column_ and _row_ indices respectively. ``i`` accesses the first
/// dimension (``size.width``) and ``j`` accesses the second dimension
/// (``size.height``).
/// \verbatim embed:rst:leading-slashes"
///   .. note::
///     This index convention is the **opposite** of some mathematical forms
///     where matrix elements_ are indexed by the i-th row and j-th column.
/// \endverbatim
/// - Array2 provides a convenient way to access its elements_:
/// \verbatim embed:rst:leading-slashes
///    .. code-block:: cpp
///
///       for(auto e : my_array2) {
///         e.value = 0; // element value access
///         e.index; // element index
///       }
/// \endverbatim
/// \tparam T data type
template<class T> class Array2 {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  Array2() = default;
  /// pitch is set to ``size.width`` * ``sizeof(T)``
  /// \param size dimensions (in elements_ count)
  explicit Array2(const size2 &size) : pitch_(size.width * sizeof(T)), size_(size) {
    data_ = new char[pitch_ * size.height];
  }
  /// \param size dimensions (in elements_ count)
  /// \param pitch memory size occupied by a single row (in bytes)
  explicit Array2(const size2 &size, size_t pitch)
      : pitch_(pitch), size_(size) {
    data_ = new char[pitch_ * size.height];
  }
  /// Copy constructor
  /// \param other **[in]** const reference to other Array2 object
  Array2(const Array2 &other) : Array2(other.size_, other.pitch_) {
    memcpy(data_, other.data_, memorySize());
  }
  Array2(const Array2 &&other) = delete;
  /// Copy constructor
  /// \param other **[in]** reference to other Array2 object
  Array2(Array2 &other) : Array2(other.size_, other.pitch_) {
    memcpy(data_, other.data_, memorySize());
  }
  /// Assign constructor
  /// \param other **[in]** temporary Array2 object
  Array2(Array2 &&other) noexcept
      : pitch_(other.pitch_), size_(other.size_), data_(other.data_) {
    other.data_ = nullptr;
  }
  /// Constructs an Array2 from a std vector matrix
  /// \param linear_vector **[in]** data matrix
  Array2(const std::vector<std::vector<T>> &linear_vector) {
    resize(size2(linear_vector[0].size(), linear_vector.size()));
    for (auto ij : Index2Range<i32>(size_))
      (*this)[ij] = linear_vector[ij.j][ij.i];
  }
  /// Initialization list constructor
  /// - Inner lists represent rows.
  /// \param list **[in]** data list
  /// \verbatim embed:rst:leading-slashes
  ///    **Example**::
  ///
  ///       hermes::Array2<i32> a = {{1,2},{3,4}};
  /// \endverbatim
  Array2(std::initializer_list<std::initializer_list<T>> list) {
    resize(size2(list.begin()[0].size(), list.size()));
    for (auto ij : Index2Range<i32>(size_))
      (*this)[ij] = list.begin()[ij.j].begin()[ij.i];
  }
  ///
  virtual ~Array2() {
    delete[](char *) data_;
  }
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       assignment
  /// Assign operator from raw data
  /// - Inner vectors represent rows.
  /// \param linear_vector **[in]** data matrix
  /// \return Array2<T>&
  Array2<T> &operator=(const std::vector<std::vector<T>> &linear_vector) {
    resize(size2(linear_vector[0].size(), linear_vector.size()));
    for (auto ij : Index2Range<i32>(size_))
      (*this)[ij] = linear_vector[ij.j][ij.i];
  }
  /// Assign operator
  /// \param other **[in]** const reference to other Array2 object
  /// \return Array2<T>&
  Array2<T> &operator=(const Array2<T> &other) {
    size_ = other.size_;
    pitch_ = other.pitch_;
    resize(size_);
    memcpy(data_, other.data_, other.memorySize());
    return *this;
  }
  /// Assign operator
  /// \param other **[in]** temporary Array2 object
  /// \return Array2<T>&
  Array2<T> &operator=(Array2<T> &&other) noexcept {
    size_ = other.size_;
    pitch_ = other.pitch_;
    data_ = other.data_;
    other.data_ = nullptr;
    return *this;
  }
  /// Assign ``value`` to all elements_
  /// \param value assign value
  /// \return *this
  Array2 &operator=(T value) {
    if (!data_)
      data_ = new char[pitch_ * size_.height];
    for (index2 ij : Index2Range<i32>(size_))
      (*this)[ij] = value;
    return *this;
  }
  //                                                                                                           access
  /// \verbatim embed:rst:leading-slashes
  ///    .. warning::
  ///       This method does **not** check if ``ij`` is out of bounds.
  /// \endverbatim
  /// \param ij ``ij.i`` for column and ``ij.j`` for row
  /// \return reference to element at ``ij`` position
  T &operator[](index2 ij) {
    return (T &) (*((char *) data_ + ij.j * pitch_ + ij.i * sizeof(T)));
  }
  /// \verbatim embed:rst:leading-slashes
  ///    .. warning::
  ///       This method does **not** check if ``ij`` is out of bounds.
  /// \endverbatim
  /// \param ij ``ij.i`` for column and ``ij.j`` for row
  /// \return const reference to element at ``ij`` position
  const T &operator[](index2 ij) const {
    return (T &) (*((char *) data_ + ij.j * pitch_ + ij.i * sizeof(T)));
  }
  /// \verbatim embed:rst:leading-slashes
  ///    .. warning::
  ///       This method does **not** check if ``ij`` is out of bounds.
  /// \endverbatim
  /// \param ij expected to be constructed from j * width + i
  /// \return reference to element at ``ij`` position
  T &operator[](u64 ij) {
    auto j = ij / size_.width;
    auto i = ij % size_.width;
    return (T &) (*((char *) data_ + j * pitch_ + i * sizeof(T)));
  }
  /// \verbatim embed:rst:leading-slashes
  ///    .. warning::
  ///       This method does **not** check if ``ij`` is out of bounds.
  /// \endverbatim
  /// \param ij expected to be constructed from j * width + i
  /// \return reference to element at ``ij`` position
  const T &operator[](u64 ij) const {
    auto j = ij / size_.width;
    auto i = ij % size_.width;
    return (T &) (*((char *) data_ + j * pitch_ + i * sizeof(T)));
  }
  /// \verbatim embed:rst:leading-slashes
  ///    .. warning::
  ///       This method does **not** check if ``i`` or ``j`` are out of bounds.
  /// \endverbatim
  /// \param i **[in]** column index
  /// \param j **[in]** row index
  /// \return T& reference to element in row ``i`` and column ``j``
  T &operator()(u32 i, u32 j) {
    return (T &) (*((char *) data_ + j * pitch_ + i * sizeof(T)));
  }
  /// \verbatim embed:rst:leading-slashes
  ///    .. warning::
  ///       This method does **not** check if ``i`` or ``j`` are out of bounds.
  /// \endverbatim
  /// \param i **[in]** column index
  /// \param j **[in]** row index
  /// \return const reference to element at ``ij`` position
  const T &operator()(u32 i, u32 j) const {
    return (T &) (*((char *) data_ + j * pitch_ + i * sizeof(T)));
  }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  //                                                                                                             size
  /// Changes the dimensions
  /// \verbatim embed:rst:leading-slashes
  ///    .. note::
  ///       All previous data is erased.
  /// \endverbatim
  /// \param new_size new row and column counts
  void resize(const size2 &new_size) {
    delete[](char *) data_;
    pitch_ = std::max(pitch_, sizeof(T) * new_size.width);
    size_ = new_size;
    data_ = new char[pitch_ * new_size.height];
  }
  /// Computes actual memory usage
  /// \return memory usage (in bytes)
  [[nodiscard]] u64 memorySize() const { return size_.height * pitch_; }
  /// \return dimensions (in elements_ count)
  [[nodiscard]] size2 size() const { return size_; }
  /// \return pitch size (in bytes)
  [[nodiscard]] u64 pitch() const { return pitch_; }
  //                                                                                                           access
  /// \return const pointer to raw data (**row major**)
  const T *data() const { return (const T *) data_; }
  /// \return pointer to raw data (**row major**)
  T *data() { return (T *) data_; }
  /// Copies data from another Array2
  ///
  /// - This gets resized if necessary.
  /// \param other **[in]**
  void copy(const Array2 &other) {
    pitch_ = other.pitch_;
    size_ = other.size_;
    resize(size_);
    memcpy(data_, other.data_, memorySize());
  }
  /// Checks if ``ij`` is not out of bounds
  /// \param ij position index
  /// \return ``true`` if position can be accessed
  [[nodiscard]] bool stores(const index2 &ij) const {
    return ij.i >= 0 &&
        static_cast<i64>(ij.i) < static_cast<i64>(size_.width) &&
        ij.j >= 0 && static_cast<i64>(ij.j) < static_cast<i64>(size_.height);
  }
  //                                                                                                        iterators
  Array2Iterator<T> begin() {
    return Array2Iterator<T>((T *) data_, size_, pitch_, index2(0, 0));
  }
  Array2Iterator<T> end() {
    return Array2Iterator<T>((T *) data_, size_, pitch_, index2(-1, -1));
  }
  ConstArray2Iterator<T> begin() const {
    return ConstArray2Iterator<T>((T *) data_, size_, pitch_, index2(0, 0));
  }
  ConstArray2Iterator<T> end() const {
    return ConstArray2Iterator<T>((T *) data_, size_, pitch_, index2(-1, -1));
  }

private:
  size_t pitch_{0};
  size2 size_{};
  void *data_ = nullptr;
};

// *********************************************************************************************************************
//                                                                                                                 IO
// *********************************************************************************************************************
template<typename T>
std::ostream &operator<<(std::ostream &os, const Array<T, MemoryLocation::HOST> &array) {
  // print name
  os << "Array";
  for (auto i = 0; i < array.dimensions(); ++i)
    os << "[" << array.size()[i] << "]";
  os << "\n";

  // exit if empty
  if (array.empty())
    return os;

  // compute text width
  int w = 12;
  if (std::is_same_v<T, u8> || std::is_same_v<T, i8>)
    w = 4;

  // header
  if (array.dimensions() < 3)
    for (u32 i = 0; i < array.size().width; ++i)
      os << std::setw(w) << std::right << (Str() << "[" << i << "]");

  // data
  auto formated_str = [&](T data) {
    os << std::setw(w) << std::right;
    if (std::is_same<T, u8>())
      os << (int) data;
    else if (std::is_same_v<T, f32> || std::is_same_v<T, f64>)
      os << std::setprecision(8) << data;
    else
      os << data;
  };
  if (array.dimensions() == 1) {
    os << "\n";
    for (u32 i = 0; i < array.size().width; ++i)
      formated_str(array[i]);
    os << "\n";
  } else if (array.dimensions() == 2) {
    os << "\n";
    for (i32 j = 0; j < array.size().height; ++j) {
      os << "[," << j << "]";
      for (i32 i = 0; i < array.size().width; ++i)
        formated_str(array[{i, j}]);
      os << " [," << j << "]\n";
    }
  }

  // footer
  if (array.dimensions() == 2)
    for (u32 i = 0; i < array.size().width; ++i)
      os << std::setw(w) << std::right << (Str() << "[" << i << "]");
  os << "\n";

  return os;
}
template<typename T>
std::ostream &operator<<(std::ostream &os, const Array1<T> &array) {
  os << "Array1[" << array.size() << "]\n\t";
  // compute text width
  int w = 12;
  if (std::is_same_v<T, u8> || std::is_same_v<T, i8>)
    w = 4;
  for (u32 i = 0; i < array.size(); ++i)
    os << std::setw(w) << std::right << (Str() << "[" << i << "]");
  os << std::endl << '\t';
  for (u32 i = 0; i < array.size(); ++i) {
    os << std::setw(w) << std::right;
    if (std::is_same<T, u8>())
      os << (int) array[i];
    else if (std::is_same_v<T, f32> || std::is_same_v<T, f64>)
      os << std::setprecision(8) << array[i];
    else
      os << array[i];
  }
  os << std::endl;
  return os;
}
template<typename T>
std::ostream &operator<<(std::ostream &os, const Array2<T> &array) {
  os << "Array2[" << array.size() << "]\n\t\t";
  int w = 12;
  if (std::is_same_v<T, u8> || std::is_same_v<T, i8>)
    w = 4;
  for (u32 i = 0; i < array.size().width; ++i)
    os << std::setw(w) << std::right << (Str() << "[" << i << "]");
  os << std::endl;
  for (u32 j = 0; j < array.size().height; ++j) {
    os << "\t[," << j << "]";
    for (u32 i = 0; i < array.size().width; ++i) {
      os << std::setw(w) << std::right;
      if (std::is_same<T, u8>() || std::is_same_v<T, i8>)
        os << (int) array[index2(i, j)];
      else if (std::is_same_v<T, f32> || std::is_same_v<T, f64>)
        os << std::setprecision(8) << array[index2(i, j)];
      else
        os << array[index2(i, j)];
    }
    os << " [," << j << "]";
    os << std::endl;
  }
  os << "\t\t";
  for (u32 i = 0; i < array.size().width; ++i)
    os << std::setw(w) << std::right << (Str() << "[" << i << "]");
  os << std::endl;
  return os;
}

// *********************************************************************************************************************
//                                                                                                           TYPEDEFS
// *********************************************************************************************************************
template<typename T>
using HostArray = Array<T, MemoryLocation::HOST>;
using array1d = Array1<f64>;
using array1f = Array1<f32>;
using array1i = Array1<i32>;
using array1u = Array1<u32>;
using array2d = Array2<f64>;
using array2f = Array2<f32>;
using array2i = Array2<i32>;
using array2u = Array2<u32>;

} // namespace hermes

#endif // HERMES_STORAGE_ARRAY_H