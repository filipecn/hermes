/// Copyright (c) 2022, FilipeCN.
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
///\file result_or.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2022-05-20
///
///\brief

#ifndef HERMES_HERMES_COMMON_RESULT_H
#define HERMES_HERMES_COMMON_RESULT_H

#include <hermes/common/defs.h>
#include <hermes/common/debug.h>

namespace hermes {

template<class T>
struct UnexpectedResultType {
  T value{};
};

// *********************************************************************************************************************
//                                                                                                              Result
// *********************************************************************************************************************
/// \brief Holds a valid object or an error
template<class T, class E = HeResult>
class Result {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE explicit Result(const UnexpectedResultType<E> &err = {}) : ok_(false) {
    new(reinterpret_cast<E *>(&err_)) E(err.value);
  }
  /// \brief Value constructor
  /// \param v
  HERMES_DEVICE_CALLABLE explicit Result(const T &v) : ok_(true) {
    new(reinterpret_cast<T *>(&value_)) T(v);
  }
  /// \brief Move value constructor
  /// \param v
  HERMES_DEVICE_CALLABLE explicit Result(T &&v) : ok_(true) {
    new(reinterpret_cast<T *>(&value_)) T(std::move(v));
  }
  //                                                                                                       assignment
  /// \brief Copy constructor
  /// \param other
  HERMES_DEVICE_CALLABLE Result(const Result &other) {
    *this = other;
  }
  /// \brief Move constructor
  /// \param other
  HERMES_DEVICE_CALLABLE Result(Result &&other) noexcept {
    *this = std::move(other);
  }
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  /// \brief Casts to bool (indicates whether this contains value)
  /// \return
  HERMES_DEVICE_CALLABLE explicit operator bool() const noexcept { return ok_; }
  //                                                                                                       assignment
  /// \brief Copy assignment
  /// \param other
  /// \return
  HERMES_DEVICE_CALLABLE Result &operator=(const Result &other) {
    reset();
    ok_ = other.ok_;
    if (other.ok_)
      new(reinterpret_cast<T *>(&value_)) T(other.value());
    else
      new(reinterpret_cast<E *>(&err_)) E(other.status());
    return *this;
  }
  /// \brief Move assignment
  /// \param other
  /// \return
  HERMES_DEVICE_CALLABLE Result &operator=(Result &&other) noexcept {
    reset();
    ok_ = other.ok_;
    if (other.ok_)
      new(reinterpret_cast<T *>(&value_)) T(std::move(other.value()));
    else
      new(reinterpret_cast<E *>(&err_)) E(std::move(other.status()));
    return *this;
  }
  /// \brief Value assignment
  /// \param v
  /// \return
  HERMES_DEVICE_CALLABLE Result &operator=(const T &v) {
    reset();
    ok_ = true;
    new(reinterpret_cast<T *>(&value_)) T(v);
    return *this;
  }
  /// \brief Move value assignment
  /// \param v
  /// \return
  HERMES_DEVICE_CALLABLE Result &operator=(T &&v) {
    reset();
    ok_ = true;
    new(reinterpret_cast<T *>(&value_)) T(std::move(v));
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
  [[nodiscard]] HERMES_DEVICE_CALLABLE bool good() const { return ok_; }
  [[nodiscard]] HERMES_DEVICE_CALLABLE E status() const { return err_; }
  /// \brief Destroys stored value (if present)
  HERMES_DEVICE_CALLABLE void reset() {
    if (good()) {
      value().~T();
      ok_ = false;
    }
  }
  //                                                                                                           access
  /// \brief Gets value copy (if present)
  /// \param v value returned in case of empty
  /// \return
  HERMES_DEVICE_CALLABLE T valueOr(const T &v) const { return good() ? value() : v; }
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
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
private:
  union {
    E err_{};
    typename std::aligned_storage<sizeof(T), alignof(T)>::type value_;
  };
  bool ok_{false};
};

}

#endif //HERMES_HERMES_COMMON_RESULT_H
