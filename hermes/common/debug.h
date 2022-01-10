/*
 * Copyright (c) 2017 FilipeCN
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
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

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

/// Enum returned by functions
enum class HeResult {
  SUCCESS = 0,
  BAD_ALLOCATION = 1,
  OUT_OF_BOUNDS = 2,
  INVALID_INPUT = 3,
  BAD_OPERATION = 4
};

// *********************************************************************************************************************
//                                                                                                              UTILS
// *********************************************************************************************************************
#ifndef LOG_LOCATION
#define LOG_LOCATION "[" << __FILE__ << "][" << __LINE__ << "]"
#endif
// *********************************************************************************************************************
//                                                                                               COMPILATION WARNINGS
// *********************************************************************************************************************
#ifndef HERMES_UNUSED_VARIABLE
#define HERMES_UNUSED_VARIABLE(x) ((void)x);
#endif

#ifndef HERMES_NOT_IMPLEMENTED
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
#define HERMES_PING hermes::Log::info("[{}][{}][{}]", __FILE__, __LINE__, __FUNCTION__);
#endif

#ifndef HERMES_LOG
//#define HERMES_LOG(A) hermes::Log::info("[{}][{}][{}]: {}", __FILE__, __LINE__, __FUNCTION__, A);
#define HERMES_LOG(A) hermes::Log::logMessage(hermes::logging_options::info, "{}", \
  hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__}, A);
#endif

#ifndef HERMES_LOG_WARNING
//#define HERMES_LOG_WARNING(A) hermes::Log::warn("[{}][{}][{}]: {}", __FILE__, __LINE__, __FUNCTION__, A);
#define HERMES_LOG_WARNING(A) hermes::Log::logMessage(hermes::logging_options::warn, "{}", \
  hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__}, A);
#endif

#ifndef HERMES_LOG_ERROR
//#define HERMES_LOG_ERROR(A) hermes::Log::error("[{}][{}][{}]: {}", __FILE__, __LINE__, __FUNCTION__, A);
#define HERMES_LOG_ERROR(A) hermes::Log::logMessage(hermes::logging_options::error, "{}", \
  hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__}, A);
#endif

#ifndef HERMES_LOG_CRITICAL
//#define HERMES_LOG_CRITICAL(A) hermes::Log::critical("[{}][{}][{}]: {}", __FILE__, __LINE__, __FUNCTION__, A);
#define HERMES_LOG_CRITICAL(A) hermes::Log::logMessage(hermes::logging_options::critical, "{}", \
  hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__}, A);
#endif

#ifndef HERMES_LOG_VARIABLE
//#define HERMES_LOG_VARIABLE(A) hermes::Log::info("[{}][{}][{}]: {} = {}", __FILE__, __LINE__, __FUNCTION__, #A, A);
#define HERMES_LOG_VARIABLE(A) hermes::Log::logMessage(hermes::logging_options::info, "{} = {}", \
  hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__}, #A, A);
#endif

template<typename T>
static inline void hermes_log_variables_r(std::stringstream &s, const T &first) {
  s << first << "\n";
}

template<typename T, typename ...Args>
static inline void hermes_log_variables_r(std::stringstream &s, const T &first, Args &&...rest) {
  s << first << " | ";
  if constexpr(sizeof ...(rest) > 0)
    hermes_log_variables_r(s, std::forward<Args>(rest) ...);
}

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
#define HERMES_LOG_VARIABLES(...) \
  hermes::Log::info("[{}][{}][{}]: {}", __FILE__, __LINE__, __FUNCTION__, hermes_log_variables(__VA_ARGS__));
#endif

#ifndef HERMES_C_LOG
#define HERMES_C_LOG(FMT, ...)                                                                                      \
fprintf(stdout, "[%s][%d][%s]: ", __FILE__, __LINE__, __FUNCTION__);                                                \
fprintf(stdout, FMT __VA_OPT__(,) __VA_ARGS__);                                                                     \
fprintf(stdout, "\n");
#endif

#ifndef HERMES_C_ERROR
#define HERMES_C_ERROR(FMT, ...)                                                                                    \
fprintf(stderr, "[%s][%d][%s]: ", __FILE__, __LINE__, __FUNCTION__);                                                \
fprintf(stderr, FMT __VA_OPT__(,) __VA_ARGS__);                                                                     \
fprintf(stderr, "\n");
#endif

#ifndef HERMES_C_DEVICE_LOG
#define HERMES_C_DEVICE_LOG(FMT, ...)                                                                               \
printf("[%s][%d][%s]: ", __FILE__, __LINE__, __FUNCTION__);                                                         \
printf(FMT __VA_OPT__(,) __VA_ARGS__);                                                                              \
printf("\n");
#endif

#ifndef HERMES_C_DEVICE_ERROR
#define HERMES_C_DEVICE_ERROR(FMT, ...)                                                                             \
printf("[%s][%d][%s]: ", __FILE__, __LINE__, __FUNCTION__);                                                         \
printf(FMT __VA_OPT__(,) __VA_ARGS__);                                                                              \
printf("\n");
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

#define HERMES_CHECK_EXP(expr)                                                                                      \
  if(expr) {}                                                                                                       \
  else {                                                                                                            \
    hermes::Log::warn("[{}][{}][CHECK_EXP FAIL {}]", __FILE__, __LINE__, (#expr));                                  \
  }

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

#define HERMES_ASSERT(expr)                                                                                         \
  if(expr) {}                                                                                                       \
  else {                                                                                                            \
    hermes::Log::error("[{}][{}][ASSERT FAIL {}]", __FILE__, __LINE__, #expr);                                      \
    debugBreak();                                                                                                   \
  }
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
#define HERMES_RETURN_IF(A)                                                                                         \
  if (A) {                                                                                                          \
    return;                                                                                                         \
  }
#define HERMES_RETURN_IF_NOT(A)                                                                                     \
  if (!(A)) {                                                                                                       \
    return;                                                                                                         \
  }
#define HERMES_RETURN_VALUE_IF(A, R)                                                                                \
  if (A) {                                                                                                          \
    return R;                                                                                                       \
  }
#define HERMES_RETURN_VALUE_IF_NOT(A, R)                                                                            \
  if (!(A)) {                                                                                                       \
    return R;                                                                                                       \
  }
#define HERMES_LOG_AND_RETURN_VALUE_IF_NOT(A, R, M)                                                                 \
  if (!(A)) {                                                                                                       \
    HERMES_LOG(M)                                                                                                   \
    return R;                                                                                                       \
  }

#endif
