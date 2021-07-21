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
#define HERMES_UNUSED_VARIABLE(x) ((void)x)
#endif

#ifndef HERMES_NOT_IMPLEMENTED
#define HERMES_NOT_IMPLEMENTED \
  hermes::Log::warn("[{}][{}][{}] calling not implemented function.", __FILE__, __LINE__, __FUNCTION__);
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
#define HERMES_LOG(A) hermes::Log::info("[{}][{}][{}]: {}", __FILE__, __LINE__, __FUNCTION__, A);
#endif

#ifndef HERMES_LOG_WARNING
#define HERMES_LOG_WARNING(A) hermes::Log::warn("[{}][{}][{}]: {}", __FILE__, __LINE__, __FUNCTION__, A);
#endif

#ifndef HERMES_LOG_ERROR
#define HERMES_LOG_ERROR(A) hermes::Log::error("[{}][{}][{}]: {}", __FILE__, __LINE__, __FUNCTION__, A);
#endif

#ifndef HERMES_LOG_CRITICAL
#define HERMES_LOG_CRITICAL(A) hermes::Log::critical("[{}][{}][{}]: {}", __FILE__, __LINE__, __FUNCTION__, A);
#endif

#ifndef HERMES_LOG_VARIABLE
#define HERMES_LOG_VARIABLE(A) hermes::Log::info("[{}][{}][{}]: {} = {}", __FILE__, __LINE__, __FUNCTION__, #A, A);
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
#define HERMES_RETURN_IF(A, R)                                                                                      \
  if (A) {                                                                                                          \
    return R;                                                                                                       \
  }
#define HERMES_RETURN_IF_NOT(A, R)                                                                                  \
  if (!(A)) {                                                                                                       \
    return R;                                                                                                       \
  }
#define HERMES_LOG_AND_RETURN_IF_NOT(A, R, M)                                                                       \
  if (!(A)) {                                                                                                       \
    HERMES_LOG(M)                                                                                                   \
    return R;                                                                                                       \
  }

#endif
