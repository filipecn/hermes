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
///\file debug.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2017-01-04
///
///\brief Debug, logging and assertion macros
///
///\ingroup common
///\addtogroup common
/// @{

#pragma once

#include <hermes/core/types.h>
#include <hermes/log/log.h>

#include <cmath>

#ifndef HERMES_DEBUG
#define HERMES_DEBUG
#endif

#ifndef CHECKS_ENABLED
#define CHECKS_ENABLED
#endif

#ifndef ASSERTIONS_ENABLED
#define ASSERTIONS_ENABLED
#endif

// *****************************************************************************
//                                                                      UTILS
// *****************************************************************************
// *****************************************************************************
//                                                       COMPILATION WARNINGS
// *****************************************************************************
#ifndef HERMES_UNUSED_VARIABLE
/// \brief Specifies that variable is not used in this scope
/// \param x variable
#define HERMES_UNUSED_VARIABLE(x) ((void)x);
#endif

#ifndef HERMES_NOT_IMPLEMENTED
/// \brief Logs "calling code not implemented" warning
#define HERMES_NOT_IMPLEMENTED                                                 \
  printf("[%s][%d][%s] calling not implemented function.", __FILE__, __LINE__, \
         __FUNCTION__);
#endif
// *****************************************************************************
//                                                                 DEBUG MODE
// *****************************************************************************
#ifdef HERMES_DEBUG
#define HERMES_DEBUG_CODE(CODE_CONTENT) {CODE_CONTENT}
#else
#define HERMES_DEBUG_CODE(CODE_CONTENT)
#endif

// *****************************************************************************
//                                                                     CHECKS
// *****************************************************************************
#ifdef CHECKS_ENABLED

/// \brief Warns if values are different
/// \param A first value
/// \param B second value
#define HERMES_CHECK_EQUAL(A, B, ...)                                          \
  if (A == B) {                                                                \
  } else {                                                                     \
    hermes::Log::message(                                                      \
        hermes::logging_options::none, hermes::Log::Level::warn,               \
        "[CHECK_EQUAL FAIL {} == {}] {}",                                      \
        hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__}, (#A), (#B),   \
        A, B, Str::format(__VA_ARGS__));                                       \
  }

/// \brief Warns if expression is false
/// \param expr expression
#define HERMES_CHECK(expr, ...)                                                \
  if (expr) {                                                                  \
  } else {                                                                     \
    hermes::Log::message(                                                      \
        hermes::logging_options::none, hermes::Log::Level::warn,               \
        "[CHECK_EXP FAIL {}] {}",                                              \
        hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__}, (#expr),      \
        Str::format(__VA_ARGS__));                                             \
  }

#else

#define HERMES_CHECK_EXP(expr, ...)

#endif // CHECKS_ENABLED
// *****************************************************************************
//                                                                  ASSERTION
// *****************************************************************************
#ifdef ASSERTIONS_ENABLED

// #define debugBreak() asm("int 3")
#define debugBreak() exit(-1)

/// \brief Errors if expression is false
/// \param expr expression
#define HERMES_ASSERT(expr, ...)                                               \
  if (expr) {                                                                  \
  } else {                                                                     \
    hermes::Log::message(                                                      \
        hermes::logging_options::none, hermes::Log::Level::error,              \
        "[ASSERT FAIL {}] {}",                                                 \
        hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__}, #expr,        \
        Str::format(__VA_ARGS__));                                             \
    debugBreak();                                                              \
  }
#else

#define HERMES_ASSERT(expr)
#define HERMES_ASSERT_WITH_LOG(expr, M)

#endif // ASSERTIONS_ENABLED
// *****************************************************************************
//                                                                  CODE FLOW
// *****************************************************************************
/// \brief Calls return if condition is true
/// \param A condition
#define HERMES_RETURN_IF(A)                                                    \
  if (A) {                                                                     \
    return;                                                                    \
  }
/// \brief Calls return if condition is false
/// \param A condition
#define HERMES_RETURN_IF_NOT(A)                                                \
  if (!(A)) {                                                                  \
    return;                                                                    \
  }
/// \brief Return value if condition is true
/// \param A condition
/// \param R value
#define HERMES_RETURN_VALUE_IF(A, R)                                           \
  if (A) {                                                                     \
    return R;                                                                  \
  }
/// \brief Return value if condition is false
/// \param A condition
/// \param R value
#define HERMES_RETURN_VALUE_IF_NOT(A, R)                                       \
  if (!(A)) {                                                                  \
    return R;                                                                  \
  }
/// \brief Logs and return value if condition is false
/// \param A condition
/// \param R value
/// \param M log message
#define HERMES_LOG_AND_RETURN_VALUE_IF_NOT(A, R, M)                            \
  if (!(A)) {                                                                  \
    HERMES_INFO(M);                                                            \
    return R;                                                                  \
  }
/// \brief Logs and return if condition is false
/// \param A condition
/// \param M log message
#define HERMES_LOG_AND_RETURN_IF_NOT(A, M)                                     \
  if (!(A)) {                                                                  \
    HERMES_INFO(M);                                                            \
    return;                                                                    \
  }

/// @}
