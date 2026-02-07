/* Copyright (c) 2022, FilipeCN.
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
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR rhs
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR rhsWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR rhs DEALINGS
 * IN THE SOFTWARE.
 */

/// \file   result_or.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2022-05-20
///  Expected/Error results returned by functions.

#pragma once

#include <hermes/core/types.h>

#include <utility> // std::move

/// Enumeration of errors handled by hermes.
enum class HeError
{
  None = 0,           //!< no errors occurred
  BadAllocation = 1,  //!< memory related errors
  OutOfBounds = 2,    //!< invalid index access attempt
  InvalidInput = 3,   //!< function received invalid parameters
  BadOperation = 4,   //!< function pre-conditions were not fulfilled
  NotImplemented = 5, //!< function not implemented
  Custom = 6,         //!< custom error
  Unknown = 7,        //!< unknown error
};

namespace hermes
{

  namespace detail
  {
    template <class T>
    struct UnexpectedResultType
    {
      T value{};
    };
  } // namespace detail

  // *****************************************************************************
  //                                                                      Result
  // *****************************************************************************

  /// Holds the expected value on success, or an error otherwise.
  template <class T, class E = HeError>
  class Result
  {
  public:
    // ***************************************************************************
    //                                                         STATIC FUNCTIONS
    // ***************************************************************************

    /// \param e
    /// \return
    HERMES_CPU_GPU static Result<T, E> error(E e)
    {
      return Result<T, E>(detail::UnexpectedResultType<E>{e});
    }

    // ***************************************************************************
    //                                                             CONSTRUCTORS
    // ***************************************************************************

    /// Error constructor.
    /// \param err
    HERMES_CPU_GPU Result(const E &err) : ok_(false)
    {
      new (reinterpret_cast<E *>(&err_)) E(err);
    }
    /// Error constructor.
    /// \param err
    HERMES_CPU_GPU Result(const detail::UnexpectedResultType<E> &err = {})
        : ok_(false)
    {
      new (reinterpret_cast<E *>(&err_)) E(err.value);
    }
    /// Value constructor
    /// \param v
    HERMES_CPU_GPU Result(const T &v) : ok_(true)
    {
      new (reinterpret_cast<T *>(&value_)) T(v);
    }
    /// Move value constructor
    /// \param v
    HERMES_CPU_GPU Result(T &&v) : ok_(true)
    {
      new (reinterpret_cast<T *>(&value_)) T(std::move(v));
    }
    /// Copy constructor
    /// \param rhs
    HERMES_CPU_GPU Result(const Result &rhs) { *this = rhs; }
    /// Move constructor
    /// \param rhs
    HERMES_CPU_GPU Result(Result &&rhs) HERMES_NOEXCEPT
    {
      *this = std::move(rhs);
    }
    HERMES_CPU_GPU ~Result() noexcept { reset(); }

    // ***************************************************************************
    //                                                                OPERATORS
    // ***************************************************************************

    /// Casts to bool (indicates whether this contains a value).
    HERMES_CPU_GPU explicit operator bool() const HERMES_NOEXCEPT { return ok_; }

    //                                                               assignment

    /// Copy assignment.
    HERMES_CPU_GPU Result &operator=(const Result &rhs)
    {
      reset();
      ok_ = rhs.ok_;
      if (rhs.ok_)
        new (reinterpret_cast<T *>(&value_)) T(rhs.value());
      else
        new (reinterpret_cast<E *>(&err_)) E(rhs.status());
      return *this;
    }
    /// Move assignment.
    HERMES_CPU_GPU Result &operator=(Result &&rhs) HERMES_NOEXCEPT
    {
      reset();
      ok_ = rhs.ok_;
      if (rhs.ok_)
        new (reinterpret_cast<T *>(&value_)) T(std::move(rhs.value()));
      else
        new (reinterpret_cast<E *>(&err_)) E(std::move(rhs.status()));
      return *this;
    }
    /// Value assignment.
    HERMES_CPU_GPU Result &operator=(const T &v)
    {
      reset();
      ok_ = true;
      new (reinterpret_cast<T *>(&value_)) T(v);
      return *this;
    }
    /// Move value assignment.
    HERMES_CPU_GPU Result &operator=(T &&v)
    {
      reset();
      ok_ = true;
      new (reinterpret_cast<T *>(&value_)) T(std::move(v));
      return *this;
    }

    //                                                                    access

    /// \return Pointer to the stored value.
    HERMES_CPU_GPU T *operator->() { return &value(); }
    /// \return Const pointer to the stored value.
    HERMES_CPU_GPU const T *operator->() const { return &value(); }
    /// \return Reference to value.
    HERMES_CPU_GPU T &operator*() { return value(); }
    /// \return Const reference to value.
    HERMES_CPU_GPU const T &operator*() const { return value(); }

    // ***************************************************************************
    //                                                                  METHODS
    // ***************************************************************************

    /// \return True if this holds a valid value or false if it holds an error.
    HERMES_NODISCARD HERMES_CPU_GPU bool good() const { return ok_; }
    /// \return Error status.
    HERMES_NODISCARD HERMES_CPU_GPU E status() const { return err_; }
    /// Destroys stored value (if present) by calling its destructor.
    HERMES_CPU_GPU void reset()
    {
      if (good())
      {
        value().~T();
        ok_ = false;
      }
    }

    //                                                                      access

    /// \param fallback_value value returned on error.
    /// \return A copy to the stored value, or 'fallback_value' otherwise.
    HERMES_NODISCARD HERMES_CPU_GPU T valueOr(const T &fallback_value) const
    {
      return good() ? value() : fallback_value;
    }
    /// \return Reference to the stored value.
    HERMES_NODISCARD HERMES_CPU_GPU T &value() &
    {
      return *reinterpret_cast<T *>(&value_);
    }
    /// \return Const reference to value.
    HERMES_NODISCARD HERMES_CPU_GPU const T &value() const &
    {
      return *reinterpret_cast<const T *>(&value_);
    }
    /// \return Moved stored value.
    HERMES_NODISCARD HERMES_CPU_GPU T value() &&
    {
      return std::move(*reinterpret_cast<T *>(&value_));
    }

  private:
    union
    {
      E err_{};
      // typename std::aligned_storage<sizeof(T), alignof(T)>::type value_;
      alignas(T) std::byte value_[sizeof(T)];
    };
    bool ok_{false};
  };

#ifdef HERMES_INCLUDE_TO_STRING
  inline std::string_view to_string(HeError error)
  {
#define ENUM_NAME(E)       \
  if (HeError::E == error) \
    return #E;
    ENUM_NAME(None)
    ENUM_NAME(BadAllocation)
    ENUM_NAME(OutOfBounds)
    ENUM_NAME(InvalidInput)
    ENUM_NAME(BadOperation)
    ENUM_NAME(NotImplemented)
    ENUM_NAME(Custom)
    ENUM_NAME(Unknown)
    return "";
#undef ENUM_NAME
  }
#endif

} // namespace hermes
