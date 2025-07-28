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
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/// \file   debug.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2017-01-04
/// \brief  Debug and assertion macros

#pragma once

#include <hermes/core/types.h>
#include <hermes/io/logger.h>

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

#ifndef HERMES_TO_STRING_DEBUG_METHOD
#ifdef HERMES_DEBUG
#define HERMES_TO_STRING_DEBUG_METHOD                                          \
  std::string to_string(u32 tab_size = 0) const;

/// Auxiliary struct for implementing the to_string classes method.
struct DebugFields {
  enum class Type { Inline, NextLine, Separator };
  DebugFields(const std::string &name) : name(name) {}
  std::string name;
  std::vector<std::tuple<Type, std::string, std::string>> fields;
  void add(Type type, const std::string &name, const std::string &value) {
    fields.emplace_back(std::make_tuple(type, name, value));
  }
  std::string to_string(u32 tab_size = 0) {
    std::string tab(tab_size, ' ');
    std::stringstream ss;
    ss << tab << "++++++ " << name << " +++++++\n";
    for (const auto &field : fields) {
      std::string field_name, value;
      Type type;
      std::tie(type, field_name, value) = field;
      switch (type) {
      case Type::Inline:
        ss << tab << "  " << field_name << ": " << value << "\n";
        break;
      case Type::NextLine:
        ss << tab << "  " << field_name << ":\n";
        ss << tab << " " << value << "\n";
        break;
      case Type::Separator:
        ss << tab << "--------------------------\n";
      }
    }
    return ss.str();
  }
};

#ifndef HERMES_TO_STRING_DEBUG_METHOD_BEGIN
#define HERMES_TO_STRING_DEBUG_METHOD_BEGIN(NAME)                              \
  std::string NAME::to_string(u32 tab_size) const {                            \
    DebugFields debug_fields(#NAME);
#endif

#ifndef HERMES_TO_STRING_DEBUG_METHOD_END
#define HERMES_TO_STRING_DEBUG_METHOD_END                                      \
  return debug_fields.to_string(tab_size);                                     \
  }
#endif

#ifndef HERMES_PUSH_DEBUG_HERMES_FIELD
#define HERMES_PUSH_DEBUG_HERMES_FIELD(F)                                      \
  debug_fields.add(DebugFields::Type::NextLine, #F, F.to_string(tab_size + 2));
#endif

#ifndef HERMES_PUSH_DEBUG_HERMES_PTR_FIELD
#define HERMES_PUSH_DEBUG_HERMES_PTR_FIELD(F)                                  \
  debug_fields.add(DebugFields::Type::NextLine, #F,                            \
                   F ? F->to_string(tab_size + 2) : "nullptr");
#endif

#ifndef HERMES_PUSH_DEBUG_CUSTOM_FIELD
#define HERMES_PUSH_DEBUG_CUSTOM_FIELD(F, V)                                   \
  debug_fields.add(DebugFields::Type::Inline, #F, V);
#endif

#ifndef HERMES_PUSH_DEBUG_RAW_PTR_FIELD
#define HERMES_PUSH_DEBUG_RAW_PTR_FIELD(F)                                     \
  debug_fields.add(                                                            \
      DebugFields::Type::Inline, #F,                                           \
      F ? venus::Str<char>::addressOf(reinterpret_cast<std::uintptr_t>(F))     \
        : "nullptr");
#endif

#ifndef HERMES_PUSH_DEBUG_FIELD
#define HERMES_PUSH_DEBUG_FIELD(F)                                             \
  debug_fields.add(DebugFields::Type::Inline, #F, std::to_string(F));
#endif

#ifndef HERMES_PUSH_DEBUG_SEPARATOR_LINE
#define HERMES_PUSH_DEBUG_SEPARATOR_LINE                                       \
  debug_fields.add(DebugFields::Type::Separator, "", "");
#endif

#ifndef HERMES_PUSH_DEBUG_VK_FIELD
#define HERMES_PUSH_DEBUG_VK_FIELD(F)                                          \
  debug_fields.add(DebugFields::Type::Inline, #F, vk::to_string(F));
#endif

#ifndef HERMES_PUSH_DEBUG_VK_RAII_FIELD
#define HERMES_PUSH_DEBUG_VK_RAII_FIELD(F)                                     \
  debug_fields.add(DebugFields::Type::Inline, #F,                              \
                   (*F == nullptr) ? "nullptr" : "good");
#endif

#ifndef HERMES_PUSH_DEBUG_GLM_FIELD
#define HERMES_PUSH_DEBUG_GLM_FIELD(F)                                         \
  debug_fields.add(DebugFields::Type::NextLine, #F, glm::to_string(F));
#endif

#ifndef HERMES_PUSH_DEBUG_ARRAY_FIELD_BEGIN
#define HERMES_PUSH_DEBUG_ARRAY_FIELD_BEGIN(F, I)                              \
  tab_size += 2;                                                               \
  debug_fields.add(DebugFields::Type::Inline, #F, std::to_string(F.size()));   \
  for (u32 i = 0; i < F.size(); ++i) {                                         \
    const auto &I = F[i];                                                      \
    debug_fields.add(DebugFields::Type::Inline, #I, std::to_string(i));
#endif

#ifndef HERMES_PUSH_DEBUG_ARRAY_FIELD_END
#define HERMES_PUSH_DEBUG_ARRAY_FIELD_END                                      \
  }                                                                            \
  tab_size -= 2;
#endif

#ifndef HERMES_PUSH_DEBUG_MAP_FIELD_BEGIN
#define HERMES_PUSH_DEBUG_MAP_FIELD_BEGIN(F, K, V)                             \
  tab_size += 2;                                                               \
  debug_fields.add(DebugFields::Type::Inline, #F, std::to_string(F.size()));   \
  for (const auto &item : F) {                                                 \
    const auto &K = item.first;                                                \
    const auto &V = item.second;                                               \
    debug_fields.add(DebugFields::Type::Inline, #K, K);
#endif

#ifndef HERMES_PUSH_DEBUG_MAP_FIELD_END
#define HERMES_PUSH_DEBUG_MAP_FIELD_END                                        \
  }                                                                            \
  tab_size -= 2;
#endif

#else
#define HERMES_TO_STRING_METHOD
#endif
#endif

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
    hermes::Logger::message(                                                   \
        hermes::logging_option_bits::none, hermes::Logger::Level::warn,        \
        "[CHECK_EQUAL FAIL {} == {}] {}",                                      \
        hermes::Logger::Location{__FILE__, __LINE__, __FUNCTION__}, (#A),      \
        (#B), A, B, cstr::format(__VA_ARGS__));                                \
  }

/// \brief Warns if expression is false
/// \param expr expressicssn
#define HERMES_CHECK(expr, ...)                                                \
  if (expr) {                                                                  \
  } else {                                                                     \
    hermes::Logger::message(                                                   \
        hermes::logging_option_bits::none, hermes::Logger::Level::warn,        \
        "[CHECK_EXP FAIL {}] {}",                                              \
        hermes::Logger::Location{__FILE__, __LINE__, __FUNCTION__}, (#expr),   \
        cstr::format(__VA_ARGS__));                                            \
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
    hermes::Logger::message(                                                   \
        hermes::logging_option_bits::none, hermes::Logger::Level::error,       \
        "[ASSERT FAIL {}] {}",                                                 \
        hermes::Logger::Location{__FILE__, __LINE__, __FUNCTION__}, #expr,     \
        cstr::format(__VA_ARGS__));                                            \
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
