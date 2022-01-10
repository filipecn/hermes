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
///\file optional.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-10-09
///
///\note This code is based in pbrt's optional class, that carries the license:
/// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
/// The pbrt source code is licensed under the Apache License, Version 2.0.
/// SPDX: Apache-2.0
///
///\brief

#ifndef HERMES_HERMES_COMMON_OPTIONAL_H
#define HERMES_HERMES_COMMON_OPTIONAL_H

#include <hermes/common/defs.h>
#include <iostream>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                              Optional
// *********************************************************************************************************************
/// Works just as std::optional, but supports GPU code
/// It may contain a value or not.
/// \tparam T
template<typename T>
class Optional {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  //                                                                                                              new
  HERMES_DEVICE_CALLABLE Optional() {}
  HERMES_DEVICE_CALLABLE Optional(const T &v) : has_value_(true) {
    new(reinterpret_cast<T *>(&value_)) T(v);
  }
  HERMES_DEVICE_CALLABLE Optional(T &&v) : has_value_(true) {
    new(reinterpret_cast<T *>(&value_)) T(std::move(v));
  }
  HERMES_DEVICE_CALLABLE ~Optional() { reset(); }
  //                                                                                                       assignment
  HERMES_DEVICE_CALLABLE Optional(const Optional &other) {
    *this = other;
  }
  HERMES_DEVICE_CALLABLE Optional(Optional &&other) noexcept {
    *this = std::move(other);
  }
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE explicit operator bool() const { return has_value_; }
  //                                                                                                       assignment
  HERMES_DEVICE_CALLABLE Optional &operator=(const Optional &other) {
    reset();
    if (other.has_value_) {
      has_value_ = true;
      new(reinterpret_cast<T *>(&value_)) T(other.value());
    }
    return *this;
  }
  HERMES_DEVICE_CALLABLE Optional &operator=(Optional &&other) noexcept {
    reset();
    if (other.has_value_) {
      has_value_ = true;
      new(reinterpret_cast<T *>(&value_)) T(std::move(other.value()));
    }
    return *this;
  }
  HERMES_DEVICE_CALLABLE Optional &operator=(const T &v) {
    reset();
    new(reinterpret_cast<T *>(&value_)) T(v);
    has_value_ = true;
    return *this;
  }
  HERMES_DEVICE_CALLABLE Optional &operator=(T &&v) {
    reset();
    new(reinterpret_cast<T *>(&value_)) T(std::move(v));
    has_value_ = true;
    return *this;
  }
  //                                                                                                           access
  HERMES_DEVICE_CALLABLE T *operator->() { return &value(); }
  HERMES_DEVICE_CALLABLE const T *operator->() const { return &value(); }
  HERMES_DEVICE_CALLABLE T &operator*() { return value(); }
  HERMES_DEVICE_CALLABLE const T &operator*() const { return value(); }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  ///
  HERMES_DEVICE_CALLABLE void reset() {
    if (has_value_) {
      value().~T();
      has_value_ = false;
    }
  }
  ///
  /// \return
  [[nodiscard]] HERMES_DEVICE_CALLABLE bool hasValue() const { return has_value_; }
  //                                                                                                           access
  HERMES_DEVICE_CALLABLE T valueOr(const T &v) const { return has_value_ ? value() : v; }
  /// \return
  HERMES_DEVICE_CALLABLE T &value() {
    return *reinterpret_cast<T *>(&value_);
  }
  /// \return
  HERMES_DEVICE_CALLABLE const T &value() const {
    return *reinterpret_cast<const T *>(&value_);
  }
private:
  bool has_value_{false};
  typename std::aligned_storage<sizeof(T), alignof(T)>::type value_;
};

// *********************************************************************************************************************
//                                                                                                                 IO
// *********************************************************************************************************************
template<typename T>
inline std::ostream &operator<<(std::ostream &o, const Optional<T> &optional) {
  if (optional.hasValue())
    return o << "Optional<" << typeid(T).name() << "> = " << optional.hasValue();
  return o << "Optional<" << typeid(T).name() << "> = [no value]";
}

}

#endif //HERMES_HERMES_COMMON_OPTIONAL_H
