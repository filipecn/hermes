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
///\file span.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-10-15
///\note This code based on absl::Span from Google's Abseil library.
///\brief

#ifndef HERMES_HERMES_STORAGE_ARRAY_SLICE_H
#define HERMES_HERMES_STORAGE_ARRAY_SLICE_H

#include <hermes/common/defs.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                         ArraySlice
// *********************************************************************************************************************
template<typename T>
class ArraySlice {
public:
  using value_type = typename std::remove_cv_t<T>;
  using iterator = T *;
  using const_iterator = const T *;
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  // *******************************************************************************************************************
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE ArraySlice() : data_(nullptr), size_(0) {}
  HERMES_DEVICE_CALLABLE ArraySlice(T *data, size_t size) : data_(data), size_(size) {}
  HERMES_DEVICE_CALLABLE ArraySlice(std::initializer_list<value_type> a) : ArraySlice(a.begin(), a.size()) {}
  //                                                                                                       assignment
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE T &operator[](size_t i) { return data_[i]; }
  HERMES_DEVICE_CALLABLE const T &operator[](size_t i) const { return data_[i]; }
  //                                                                                                       assignment
  //                                                                                                       arithmetic
  //                                                                                                          boolean
  //                                                                                                        iteration
  HERMES_DEVICE_CALLABLE iterator begin() { return data_; }
  HERMES_DEVICE_CALLABLE iterator end() { return data_ + size_; }
  HERMES_DEVICE_CALLABLE const_iterator begin() const { return data_; }
  HERMES_DEVICE_CALLABLE const_iterator end() const { return data_ + size_; }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  [[nodiscard]] HERMES_DEVICE_CALLABLE size_t size() const { return size_; };
  [[nodiscard]] HERMES_DEVICE_CALLABLE bool empty() const { return size() == 0; }
  HERMES_DEVICE_CALLABLE T *data() { return data_; }
  HERMES_DEVICE_CALLABLE const T *data() const { return data_; }
  HERMES_DEVICE_CALLABLE T front() const { return data_[0]; }
  HERMES_DEVICE_CALLABLE T back() const { return data_[size_ - 1]; }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
private:
  T *data_{nullptr};
  std::size_t size_{0};
};

}

#endif //HERMES_HERMES_STORAGE_ARRAY_SLICE_H
