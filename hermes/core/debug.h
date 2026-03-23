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

#ifdef HERMES_INCLUDE_DEBUG_TRAITS
#include <algorithm> //  std::count
#endif

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
//                                                                      UTILS
// *****************************************************************************

#ifndef HERMES_COMMA
#define HERMES_COMMA ,
#endif

#ifndef HERMES_NAME_OF
#define HERMES_NAME_OF(A) #A
#endif

#ifdef HERMES_INCLUDE_DEBUG_TRAITS

namespace hermes {

template <typename T, typename = void>
struct is_stream_writable : std::false_type {};

template <typename T>
struct is_stream_writable<T,
                          std::void_t<decltype(std::declval<std::ostream &>()
                                               << std::declval<const T &>())>>
    : std::true_type {};

/// @brief Trait for objects that can be converted to debug strings
template <typename T> struct DebugTraits {
  static HERMES_CONST_OR_CONSTEXPR bool is_string_serializable = false;
};

struct DebugMessage {
  DebugMessage() = default;
  DebugMessage(const DebugMessage &rhs) noexcept
      : offset_{rhs.offset_}, tab_level_{rhs.tab_level_},
        tab_size_{rhs.tab_size_} {
    ss_ << rhs.ss_.str();
  }
  DebugMessage &setOffset(h_size offset) {
    offset_ = offset;
    return *this;
  }
  DebugMessage &pushTab() {
    tab_level_++;
    return *this;
  }
  DebugMessage &popTab() {
    tab_level_ = tab_level_ > 0 ? tab_level_ - 1 : 0;
    return *this;
  }
  template <typename... Ts>
  DebugMessage &addRawFmt(const char *fmt, Ts &&...args) {
    ss_ << std::vformat(fmt, std::make_format_args(args...));
    return *this;
  }
  template <typename... Ts>
  DebugMessage &addFmt(const char *fmt, Ts &&...args) {
    std::stringstream ss;
    ss << std::vformat(fmt, std::make_format_args(args...));
    auto lines = Str<char>::split(ss.str(), "\n");
    for (const auto &line : lines) {
      ss_ << std::string(tab_level_ * tab_size_, ' ') << line << std::endl;
    }
    return *this;
  }
  DebugMessage &addSeparator(h_size size) {
    addFmt("{}\n", std::string(size, '-'));
    return *this;
  }
  template <typename... Ts>
  DebugMessage &addTitle(const char *fmt, Ts &&...args) {
    std::stringstream ss;
    ss << std::vformat(fmt, std::make_format_args(args...));
    addFmt("{}\n", ss.str());
    addSeparator(ss.str().size());
    pushTab();
    return *this;
  }
  template <typename... Ts> DebugMessage(const char *fmt, Ts &&...args) {
    ss_ << std::vformat(fmt, std::make_format_args(args...));
  }
  template <typename T>
  DebugMessage &addAddress(const std::string &name, const T *p) {
    addFmt("{} = {:p}\n", name, static_cast<const void *>(p));
    return *this;
  }
  template <typename T>
  typename std::enable_if<DebugTraits<T>::is_string_serializable,
                          DebugMessage &>::type
  addArray(const std::string &name, const std::vector<T> &arr) {
    for (h_index i = 0; i < arr.size(); ++i) {
      auto m = DebugTraits<T>::message(arr[i]);
      if (m.isMultiline()) {
        std::stringstream ss;
        ss << std::format("{}[{}]:", name, i);
        addFmt("{}\n{}\n", ss.str(), m.setOffset(ss.str().size()).str());
      } else
        addFmt("{}[{}] = {}\n", name, i, m.str());
    }
    return *this;
  }
  template <typename T>
  typename std::enable_if<!DebugTraits<T>::is_string_serializable,
                          DebugMessage &>::type
  addArray(const std::string &name, const std::vector<T> &arr) {
    for (h_index i = 0; i < arr.size(); ++i)
      addFmt("{}[{}] = {}\n", name, i, arr[i]);
    return *this;
  }
  template <typename T>
  DebugMessage &
  addArray(const std::string &name, const std::vector<T> &arr,
           const std::function<DebugMessage(h_index, const T &)> &f) {
    for (h_index i = 0; i < arr.size(); ++i) {
      auto m = f(i, arr[i]);
      if (m.isMultiline()) {
        std::stringstream ss;
        ss << std::format("{}[{}]:", name, i);
        addFmt("{}\n{}\n", ss.str(), m.setOffset(ss.str().size()).str());
      } else
        addFmt("{}[{}] = {}\n", name, i, f(i, arr[i]).str());
    }
    return *this;
  }
  template <typename K, typename V>
  typename std::enable_if<DebugTraits<V>::is_string_serializable,
                          DebugMessage &>::type
  addMap(const std::string &name, const std::unordered_map<K, V> &map) {
    for (const auto &item : map) {
      auto m = DebugTraits<V>::message(item.second);
      if (m.isMultiline()) {
        std::stringstream ss;
        if constexpr (DebugTraits<K>::is_string_serializable)
          ss << std::format("{}[{}]:", name,
                            DebugTraits<K>::message(item.first));
        else
          ss << std::format("{}[{}]:", name, item.first);
        addFmt("{}\n{}\n", ss.str(), m.setOffset(ss.str().size()).str());
      } else
        addFmt("{}[{}] = {}\n", name, item.first, m.str());
    }
    return *this;
  }
  template <typename K, typename V>
  typename std::enable_if<!DebugTraits<V>::is_string_serializable,
                          DebugMessage &>::type
  addMap(const std::string &name, const std::unordered_map<K, V> &map) {
    for (const auto &item : map)
      addFmt("{}[{}] = {}\n", name, item.first, item.second);
    return *this;
  }
  bool isMultiline() const {
    auto text = str();
    return std::count(text.begin(), text.end(), '\n') > 1;
  }
  template <typename T>
  typename std::enable_if<DebugTraits<T>::is_string_serializable,
                          DebugMessage &>::type
  add(const std::string &name, const T &t) {
    auto m = DebugTraits<T>::message(t);
    if (m.isMultiline()) {
      addFmt("{}:\n", name);
      pushTab();
      addFmt("{}\n", m.str());
      popTab();
    } else
      addFmt("{} = {}\n", name, m.str());
    return *this;
  }
  template <typename T>
  typename std::enable_if<!DebugTraits<T>::is_string_serializable &&
                              is_stream_writable<T>::value,
                          DebugMessage &>::type
  add(const std::string &name, const T &t) {
    std::stringstream ss;
    ss << t;
    addFmt("{} = {}\n", name, ss.str());
    return *this;
  }
  DebugMessage &add(const std::string &txt) {
    addFmt("{}\n", txt);
    return *this;
  }
  std::string str() const {
    auto lines = Str<char>::split(ss_.str(), "\n");
    std::stringstream ss;
    if (lines.size() == 1) {
      ss << std::string(offset_, ' ') << lines.front();
      return ss.str();
    }
    for (const auto &line : lines) {
      ss << std::string(offset_, ' ') << line;
      if (line.back() != '\n')
        ss << std::endl;
    }
    return ss.str();
  }

private:
  h_size offset_{0};
  u32 tab_level_{0};
  u32 tab_size_{2};
  std::stringstream ss_;
};

template <typename T>
typename std::enable_if<DebugTraits<T>::is_string_serializable,
                        std::string>::type
to_string(const T &t) {
  return DebugTraits<T>::message(t).str();
}

template <typename T>
typename std::enable_if<!DebugTraits<T>::is_string_serializable,
                        std::string>::type
to_string(const T &t) {
  return DebugMessage("{}", t).str();
}

} // namespace hermes

template <typename T>
typename std::enable_if<hermes::DebugTraits<T>::is_string_serializable,
                        std::ostream &>::type
operator<<(std::ostream &os, const T &t) {
  os << hermes::to_string(t);
  return os;
}

#endif

// *****************************************************************************
//                                                                 DEBUG MODE
// *****************************************************************************
#ifdef HERMES_DEBUG
#define HERMES_DEBUG_CODE(CODE_CONTENT) {CODE_CONTENT}
#else
#define HERMES_DEBUG_CODE(CODE_CONTENT)
#endif

#ifndef HERMES_CSTR_FORMAT
#ifdef HERMES_DEVICE_ENABLED
#define HERMES_CSTR_FORMAT(...) ""
#else
#define HERMES_CSTR_FORMAT(...) hermes::cstr::format(__VA_ARGS__)
#endif
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
    hermes::io::Logger::message(                                               \
        hermes::io::logger_option_bits::none, hermes::io::Logger::Level::warn, \
        "[CHECK_EQUAL FAIL {} == {}] {}",                                      \
        hermes::io::Logger::Location{__FILE__, __LINE__, __FUNCTION__}, (#A),  \
        (#B), A, B, HERMES_CSTR_FORMAT(__VA_ARGS__));                          \
  }

/// \brief Warns if expression is false
/// \param expr expressicssn
#define HERMES_CHECK(expr, ...)                                                \
  if (expr) {                                                                  \
  } else {                                                                     \
    hermes::io::Logger::message(                                               \
        hermes::io::logger_option_bits::none, hermes::io::Logger::Level::warn, \
        "[CHECK_EXP FAIL {}] {}",                                              \
        hermes::io::Logger::Location{__FILE__, __LINE__, __FUNCTION__},        \
        (#expr), HERMES_CSTR_FORMAT(__VA_ARGS__));                             \
  }

#else

#define HERMES_CHECK(expr, ...)

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
    hermes::io::Logger::message(                                               \
        hermes::io::logger_option_bits::none,                                  \
        hermes::io::Logger::Level::error, "[ASSERT FAIL {}] {}",               \
        hermes::io::Logger::Location{__FILE__, __LINE__, __FUNCTION__}, #expr, \
        HERMES_CSTR_FORMAT(__VA_ARGS__));                                      \
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

// *****************************************************************************
//                                                             RESULT HANDLING
// *****************************************************************************

#ifndef HERMES_CHECK_HE_RESULT
#define HERMES_CHECK_HE_RESULT(A)                                              \
  {                                                                            \
    HeError _hermes_check_ve_error_ = (A);                                     \
    if ((int)_hermes_check_ve_error_) {                                        \
      HERMES_ERROR("Error at: {}", #A);                                        \
      HERMES_ERROR("  w/ err: {}",                                             \
                   hermes::to_string(_hermes_check_ve_error_));                \
    }                                                                          \
  }
#endif
#ifndef HERMES_CHECK_OR_RESULT
#define HERMES_CHECK_OR_RESULT(A)                                              \
  {                                                                            \
    if (!(A)) {                                                                \
      HERMES_ERROR("Check error: {}", #A);                                     \
      return VeResult::checkError();                                           \
    }                                                                          \
  }
#endif
#ifndef HERMES_RETURN_HE_ERROR
#define HERMES_RETURN_HE_ERROR(A)                                              \
  {                                                                            \
    HeError _hermes_return_he_error_ = (A);                                    \
    if ((int)_hermes_return_he_error_) {                                       \
      HERMES_ERROR("Error at: {}", #A);                                        \
      HERMES_ERROR("  w/ err: {}",                                             \
                   hermes::to_string(_hermes_return_he_error_));               \
      return _hermes_return_he_error_;                                         \
    }                                                                          \
  }
#endif
#ifndef HERMES_RETURN_BAD_RESULT
#define HERMES_RETURN_BAD_RESULT(A)                                            \
  {                                                                            \
    HeError _hermes_return_he_error_ = (A);                                    \
    if ((int)_hermes_return_he_error_) {                                       \
      HERMES_ERROR("Error at: {}", #A);                                        \
      HERMES_ERROR("  w/ err: {}",                                             \
                   hermes::to_string(_hermes_return_he_error_));               \
      return {hermes::detail::UnexpectedResultType<HeError>(                   \
          _hermes_return_he_error_)};                                          \
    }                                                                          \
  }
#endif

#ifndef HERMES_ASSIGN_RESULT
#define HERMES_ASSIGN_RESULT(R, V)                                             \
  if (auto _hermes_result_ = V)                                                \
    R = std::move(*_hermes_result_);                                           \
  else {                                                                       \
    HERMES_ERROR("Error at: {} = {}", #R, #V);                                 \
    HERMES_ERROR("  w/ err: {}", hermes::to_string(_hermes_result_.status())); \
  }
#endif

#ifndef HERMES_ASSIGN_OR
#define HERMES_ASSIGN_OR(R, V, O)                                              \
  if (auto _hermes_result_ = V)                                                \
    R = std::move(*_hermes_result_);                                           \
  else {                                                                       \
    HERMES_ERROR("Error at: {} = {}", #R, #V);                                 \
    HERMES_ERROR("  w/ err: {}", hermes::to_string(_hermes_result_.status())); \
    O;                                                                         \
  }
#endif

#ifndef HERMES_ASSIGN_OR_RETURN_BAD_RESULT
#define HERMES_ASSIGN_OR_RETURN_BAD_RESULT(R, V)                               \
  if (auto _hermes_result_ = V)                                                \
    R = std::move(*_hermes_result_);                                           \
  else {                                                                       \
    HERMES_ERROR("Error at: {} = {}", #R, #V);                                 \
    HERMES_ERROR("  w/ err: {}", hermes::to_string(_hermes_result_.status())); \
    return _hermes_result_.status();                                           \
  }

#endif

#ifndef HERMES_ASSIGN_OR_RETURN_HE_ERROR
#define HERMES_ASSIGN_OR_RETURN_HE_ERROR(R, V)                                 \
  if (auto _hermes_result_ = V)                                                \
    R = std::move(*_hermes_result_);                                           \
  else {                                                                       \
    HERMES_ERROR("Error at: {} = {}", #R, #V);                                 \
    HERMES_ERROR("  w/ err: {}", hermes::to_string(_hermes_result_.status())); \
    return _hermes_result_.status();                                           \
  }

#endif

#ifndef HERMES_ASSIGN_OR_RETURN
#define HERMES_ASSIGN_OR_RETURN(R, V, B)                                       \
  if (auto _hermes_result_ = V)                                                \
    R = std::move(*_hermes_result_);                                           \
  else {                                                                       \
    HERMES_ERROR("Error at: {} = {}", #R, #V);                                 \
    HERMES_ERROR("  w/ err: {}", hermes::to_string(_hermes_result_.status())); \
    return B;                                                                  \
  }
#endif

#ifndef HERMES_ASSIGN_OR_RETURN_VOID
#define HERMES_ASSIGN_OR_RETURN_VOID(R, V)                                     \
  if (auto _hermes_result_ = V)                                                \
    R = std::move(*_hermes_result_);                                           \
  else {                                                                       \
    HERMES_ERROR("Error at: {} = {}", #R, #V);                                 \
    HERMES_ERROR("  w/ err: {}", hermes::to_string(_hermes_result_.status())); \
    return;                                                                    \
  }
#endif
