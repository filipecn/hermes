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
///\brief Optional value holder
///
///\ingroup common
///\addtogroup common
/// @{

#ifndef HERMES_HERMES_COMMON_OPTIONAL_H
#define HERMES_HERMES_COMMON_OPTIONAL_H

#include <hermes/common/defs.h>
#include <iostream>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                              Optional
// *********************************************************************************************************************
/// \brief Works just as std::optional, but supports GPU code. It may contain a value or not.
///
/// - Example:
/// \code{.cpp}
///     Optional<int> a;
///     // check if a is currently holding an int
///     if(a.hasValue()) {}
///     // assign a value to a
///     a = 1;
///     // access like this
///     a.value();
///     // if you are not sure if value is there, you can access this way
///     a.valueOr(-1); // you will get -1 if there is no value in a
/// \endcode
///
/// \tparam T
template<typename T>
class Optional {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  //                                                                                                              new
  /// \brief Default constructor
  HERMES_DEVICE_CALLABLE Optional() {}
  /// \brief Value constructor
  /// \param v
  HERMES_DEVICE_CALLABLE Optional(const T &v) : has_value_(true) {
    new(reinterpret_cast<T *>(&value_)) T(v);
  }
  /// \brief Move value constructor
  /// \param v
  HERMES_DEVICE_CALLABLE Optional(T &&v) : has_value_(true) {
    new(reinterpret_cast<T *>(&value_)) T(std::move(v));
  }
  ///
  HERMES_DEVICE_CALLABLE ~Optional() { reset(); }
  //                                                                                                       assignment
  /// \brief Copy constructor
  /// \param other
  HERMES_DEVICE_CALLABLE Optional(const Optional &other) {
    *this = other;
  }
  /// \brief Move constructor
  /// \param other
  HERMES_DEVICE_CALLABLE Optional(Optional &&other) noexcept {
    *this = std::move(other);
  }
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  /// \brief Casts to bool (indicates whether this contains value)
  /// \return
  HERMES_DEVICE_CALLABLE explicit operator bool() const { return has_value_; }
  //                                                                                                       assignment
  /// \brief Copy assignment
  /// \param other
  /// \return
  HERMES_DEVICE_CALLABLE Optional &operator=(const Optional &other) {
    reset();
    if (other.has_value_) {
      has_value_ = true;
      new(reinterpret_cast<T *>(&value_)) T(other.value());
    }
    return *this;
  }
  /// \brief Move assinment
  /// \param other
  /// \return
  HERMES_DEVICE_CALLABLE Optional &operator=(Optional &&other) noexcept {
    reset();
    if (other.has_value_) {
      has_value_ = true;
      new(reinterpret_cast<T *>(&value_)) T(std::move(other.value()));
    }
    return *this;
  }
  /// \brief Value assignment
  /// \param v
  /// \return
  HERMES_DEVICE_CALLABLE Optional &operator=(const T &v) {
    reset();
    new(reinterpret_cast<T *>(&value_)) T(v);
    has_value_ = true;
    return *this;
  }
  /// \brief Move value assignment
  /// \param v
  /// \return
  HERMES_DEVICE_CALLABLE Optional &operator=(T &&v) {
    reset();
    new(reinterpret_cast<T *>(&value_)) T(std::move(v));
    has_value_ = true;
    return *this;
  }
  //                                                                                                           access
  /// \brief Gets value pointer
  /// \return
  HERMES_DEVICE_CALLABLE T *operator->() { return &value(); }
  /// \brief Gets const value pointer
  /// \return
  HERMES_DEVICE_CALLABLE const T *operator->() const { return &value(); }
  /// \brief Gets value reference
  /// \return
  HERMES_DEVICE_CALLABLE T &operator*() { return value(); }
  /// \brief Gets value const reference
  /// \return
  HERMES_DEVICE_CALLABLE const T &operator*() const { return value(); }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  /// \brief Destroys stored value (if present)
  HERMES_DEVICE_CALLABLE void reset() {
    if (has_value_) {
      value().~T();
      has_value_ = false;
    }
  }
  /// \brief Checks if this holds a value
  /// \return
  [[nodiscard]] HERMES_DEVICE_CALLABLE bool hasValue() const { return has_value_; }
  //                                                                                                           access
  /// \brief Gets value copy (if present)
  /// \param v value returned in case of empty
  /// \return
  HERMES_DEVICE_CALLABLE T valueOr(const T &v) const { return has_value_ ? value() : v; }
  /// \brief Gets value's reference
  /// \return
  HERMES_DEVICE_CALLABLE T &value() {
    return *reinterpret_cast<T *>(&value_);
  }
  /// \brief Gets value's const reference
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
/// \brief Support of hermes::Optional to `std::ostream`'s << operator
/// \tparam T
/// \param o
/// \param optional
/// \return
template<typename T>
inline std::ostream &operator<<(std::ostream &o, const Optional<T> &optional) {
  if (optional.hasValue())
    return o << "Optional<" << typeid(T).name() << "> = " << optional.hasValue();
  return o << "Optional<" << typeid(T).name() << "> = [no value]";
}

}

#endif //HERMES_HERMES_COMMON_OPTIONAL_H

/// @}
