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

#ifndef HERMES_LOG_DEBUG_H
#define HERMES_LOG_DEBUG_H

#include <hermes/common/defs.h>
#include <hermes/logging/logging.h>
#include <cmath>
#include <iostream>
#include <sstream>

#ifndef HERMES_DEBUG
#define HERMES_DEBUG
#endif

#ifndef CHECKS_ENABLED
#define CHECKS_ENABLED
#endif

#ifndef ASSERTIONS_ENABLED
#define ASSERTIONS_ENABLED
#endif

#ifndef INFO_ENABLED
#define INFO_ENABLED
#endif

/// \brief Enum returned by functions
enum class HeResult {
  SUCCESS = 0,           //!< no errors occurred
  BAD_ALLOCATION = 1,    //!< memory related errors
  OUT_OF_BOUNDS = 2,     //!< invalid index access attempt
  INVALID_INPUT = 3,     //!< function received invalid parameters
  BAD_OPERATION = 4      //!< function pre-conditions were not fulfilled
};

// *********************************************************************************************************************
//                                                                                                              UTILS
// *********************************************************************************************************************
// *********************************************************************************************************************
//                                                                                               COMPILATION WARNINGS
// *********************************************************************************************************************
#ifndef HERMES_UNUSED_VARIABLE
/// \brief Specifies that variable is not used in this scope
/// \param x variable
#define HERMES_UNUSED_VARIABLE(x) ((void)x);
#endif

#ifndef HERMES_NOT_IMPLEMENTED
/// \brief Logs "calling code not implemented" warning
#define HERMES_NOT_IMPLEMENTED \
  printf("[%s][%d][%s] calling not implemented function.", __FILE__, __LINE__, __FUNCTION__);
#endif
// *********************************************************************************************************************
//                                                                                                         DEBUG MODE
// *********************************************************************************************************************
#ifdef HERMES_DEBUG
#define HERMES_DEBUG_CODE(CODE_CONTENT) {CODE_CONTENT}
#else
#define HERMES_DEBUG_CODE(CODE_CONTENT)
#endif
// *********************************************************************************************************************
//                                                                                                            LOGGING
// *********************************************************************************************************************
#ifdef INFO_ENABLED

#ifndef HERMES_PING
/// \brief Logs into info stream code location
#define HERMES_PING hermes::Log::info("[{}][{}][{}]", __FILE__, __LINE__, __FUNCTION__);
#endif

#ifndef HERMES_LOG
/// \brief Logs into info log stream
/// \code{cpp}
///     HERMES_LOG("my log with {} as value", 3) // produces "my log with 3 as value"
///     HERMES_LOG("simple log")
/// \endcode
/// \param FMT a const char* following hermes format (use "{}" to place each value in the string)
/// \param ... format values
#define HERMES_LOG(FMT, ...) hermes::Log::logMessage(hermes::logging_options::info, FMT, \
  hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__} __VA_OPT__(,) __VA_ARGS__)
#endif
/// \brief Logs into warning log stream
/// \code{cpp}
///     HERMES_LOG_WARNING("my log with {} as value", 3) // produces "my log with 3 as value"
///     HERMES_LOG_WARNING("simple log")
/// \endcode
/// \param FMT a const char* following hermes format (use "{}" to place each value in the string)
/// \param ... format values
#ifndef HERMES_LOG_WARNING
#define HERMES_LOG_WARNING(FMT, ...) hermes::Log::logMessage(hermes::logging_options::warn, FMT, \
  hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__} __VA_OPT__(,) __VA_ARGS__)
#endif
/// \brief Logs into error log stream
/// \code{cpp}
///     HERMES_LOG_ERROR("my log with {} as value", 3) // produces "my log with 3 as value"
///     HERMES_LOG_ERROR("simple log")
/// \endcode
/// \param FMT a const char* following hermes format (use "{}" to place each value in the string)
/// \param ... format values
#ifndef HERMES_LOG_ERROR
#define HERMES_LOG_ERROR(FMT, ...) hermes::Log::logMessage(hermes::logging_options::error, FMT, \
  hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__} __VA_OPT__(,) __VA_ARGS__)
#endif
/// \brief Logs into critical log stream
/// \code{cpp}
///     HERMES_LOG_CRITICAL("my log with {} as value", 3) // produces "my log with 3 as value"
///     HERMES_LOG_CRITICAL("simple log")
/// \endcode
/// \param FMT a const char* following hermes format (use "{}" to place each value in the string)
/// \param ... format values
#ifndef HERMES_LOG_CRITICAL
#define HERMES_LOG_CRITICAL(FMT, ...) hermes::Log::logMessage(hermes::logging_options::critical, FMT, \
  hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__} __VA_OPT__(,) __VA_ARGS__)
#endif

#ifndef HERMES_LOG_VARIABLE
/// \brief Logs variable name and value into info log stream
/// \pre All variables must support `std::stringstream` << operator
/// \param A variable or literal
#define HERMES_LOG_VARIABLE(A) hermes::Log::logMessage(hermes::logging_options::info, "{} = {}", \
  hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__}, #A, A)
#endif

/// \brief Auxiliary support to log multiple variables
/// \tparam T
/// \param s
/// \param first
template<typename T>
static inline void hermes_log_variables_r(std::stringstream &s, const T &first) {
  s << first << "\n";
}
/// \brief Auxiliary support to log multiple variables
/// \tparam T
/// \tparam Args
/// \param s
/// \param first
/// \param rest
template<typename T, typename ...Args>
static inline void hermes_log_variables_r(std::stringstream &s, const T &first, Args &&...rest) {
  s << first << " | ";
  if constexpr(sizeof ...(rest) > 0)
    hermes_log_variables_r(s, std::forward<Args>(rest) ...);
}
/// \brief Auxiliary support to log multiple variables
/// \tparam Args
/// \param args
/// \return
template<class... Args>
static inline std::string hermes_log_variables(Args &&... args) {
  std::stringstream s;
  if constexpr(sizeof...(args) > 0) {
    hermes_log_variables_r(s, std::forward<Args>(args) ...);
    return s.str();
  }
  return "";
}

#ifndef HERMES_LOG_VARIABLES
/// \brief Logs multiple variables into info log stream
/// \pre All variables must support `std::stringstream` << operator
/// \param ... variables
#define HERMES_LOG_VARIABLES(...) \
  hermes::Log::info("[{}][{}][{}]: {}", __FILE__, __LINE__, __FUNCTION__, hermes_log_variables(__VA_ARGS__))
#endif

#ifndef HERMES_C_LOG
/// \brief Logs into stdout in printf style
/// \code{cpp}
///     HERMES_C_LOG("my log with %d as value", 3) // produces "my log with 3 as value"
///     HERMES_C_LOG("simple log")
/// \endcode
/// \param FMT string format following printf format
/// \param ... format values
#define HERMES_C_LOG(FMT, ...)                                                                                      \
fprintf(stdout, "[%s][%d][%s]: ", __FILE__, __LINE__, __FUNCTION__);                                                \
fprintf(stdout, FMT __VA_OPT__(,) __VA_ARGS__);                                                                     \
fprintf(stdout, "\n")
#endif
#ifndef HERMES_C_LOG_ERROR
/// \brief Logs into stderr in printf style
/// \code{cpp}
///     HERMES_C_LOG_ERROR("my log with %d as value", 3) // produces "my log with 3 as value"
///     HERMES_C_LOG_ERROR("simple log")
/// \endcode
/// \param FMT string format following printf format
/// \param ... format values
#define HERMES_C_LOG_ERROR(FMT, ...)                                                                                    \
fprintf(stderr, "[%s][%d][%s]: ", __FILE__, __LINE__, __FUNCTION__);                                                \
fprintf(stderr, FMT __VA_OPT__(,) __VA_ARGS__);                                                                     \
fprintf(stderr, "\n")
#endif
#ifndef HERMES_C_DEVICE_LOG
/// \brief Logs into info stdout from device code
/// \code{cpp}
///     HERMES_C_LOG("my log with %d as value", 3) // produces "my log with 3 as value"
///     HERMES_C_LOG("simple log")
/// \endcode
/// \param FMT string format following printf format
/// \param ... format values
#define HERMES_C_DEVICE_LOG(FMT, ...)                                                                               \
printf("[%s][%d][%s]: ", __FILE__, __LINE__, __FUNCTION__);                                                         \
printf(FMT __VA_OPT__(,) __VA_ARGS__);                                                                              \
printf("\n")
#endif
#ifndef HERMES_C_DEVICE_ERROR
/// \brief Logs into stderr from device code
/// \code{cpp}
///     HERMES_C_LOG("my log with %d as value", 3) // produces "my log with 3 as value"
///     HERMES_C_LOG("simple log")
/// \endcode
/// \param FMT string format following printf format
/// \param ... format values
#define HERMES_C_DEVICE_ERROR(FMT, ...)                                                                             \
printf("[%s][%d][%s]: ", __FILE__, __LINE__, __FUNCTION__);                                                         \
printf(FMT __VA_OPT__(,) __VA_ARGS__);                                                                              \
printf("\n")
#endif

#else

#define HERMES_PING
#define HERMES_LOG
#define HERMES_LOG_VARIABLE

#endif
// *********************************************************************************************************************
//                                                                                                             CHECKS
// *********************************************************************************************************************
#ifdef CHECKS_ENABLED

/// \brief Warns if expression is false
/// \param expr expression
#define HERMES_CHECK_EXP(expr)                                                                                      \
  if(expr) {}                                                                                                       \
  else {                                                                                                            \
    hermes::Log::warn("[{}][{}][CHECK_EXP FAIL {}]", __FILE__, __LINE__, (#expr));                                  \
  }

/// \brief Warns if expression is false with message
/// \param expr expression
/// \param M custom warn message
#define HERMES_CHECK_EXP_WITH_LOG(expr, M)                                                                          \
  if(expr) {}                                                                                                       \
  else {                                                                                                            \
    hermes::Log::warn("[{}][{}][CHECK_EXP FAIL {}]: {}", __FILE__, __LINE__, (#expr), M);                           \
  }
#else

#define HERMES_CHECK_EXP(expr)
#define HERMES_CHECK_EXP_WITH_LOG(expr, M)

#endif // CHECKS_ENABLED
// *********************************************************************************************************************
//                                                                                                          ASSERTION
// *********************************************************************************************************************
#ifdef ASSERTIONS_ENABLED

//#define debugBreak() asm ("int 3")
#define debugBreak()

/// \brief Errors if expression is false
/// \param expr expression
#define HERMES_ASSERT(expr)                                                                                         \
  if(expr) {}                                                                                                       \
  else {                                                                                                            \
    hermes::Log::error("[{}][{}][ASSERT FAIL {}]", __FILE__, __LINE__, #expr);                                      \
    debugBreak();                                                                                                   \
  }
/// \brief Errors if expression is false with message
/// \param expr expression
/// \param M custom error message
#define HERMES_ASSERT_WITH_LOG(expr, M)                                                                             \
  if(expr) {}                                                                                                       \
  else {                                                                                                            \
    hermes::Log::error("[{}][{}][ASSERT FAIL {}]: {}", __FILE__, __LINE__, #expr, M);                               \
    debugBreak();                                                                                                   \
  }
#else

#define HERMES_ASSERT(expr)
#define HERMES_ASSERT_WITH_LOG(expr, M)

#endif // ASSERTIONS_ENABLED
// *********************************************************************************************************************
//                                                                                                          CODE FLOW
// *********************************************************************************************************************
/// \brief Calls return if condition is true
/// \param A condition
#define HERMES_RETURN_IF(A)                                                                                         \
  if (A) {                                                                                                          \
    return;                                                                                                         \
  }
/// \brief Calls return if condition is false
/// \param A condition
#define HERMES_RETURN_IF_NOT(A)                                                                                     \
  if (!(A)) {                                                                                                       \
    return;                                                                                                         \
  }
/// \brief Return value if condition is true
/// \param A condition
/// \param R value
#define HERMES_RETURN_VALUE_IF(A, R)                                                                                \
  if (A) {                                                                                                          \
    return R;                                                                                                       \
  }
/// \brief Return value if condition is false
/// \param A condition
/// \param R value
#define HERMES_RETURN_VALUE_IF_NOT(A, R)                                                                            \
  if (!(A)) {                                                                                                       \
    return R;                                                                                                       \
  }
/// \brief Logs and return value if condition is false
/// \param A condition
/// \param R value
/// \param M log message
#define HERMES_LOG_AND_RETURN_VALUE_IF_NOT(A, R, M)                                                                 \
  if (!(A)) {                                                                                                       \
    HERMES_LOG(M);                                                                                                  \
    return R;                                                                                                       \
  }

#endif

/// @}
