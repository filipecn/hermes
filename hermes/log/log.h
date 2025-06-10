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
///\file logging.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-06-20
///
///\brief Logging functions
///
///\ingroup logging
///\addtogroup logging
/// @{

#pragma once

#include <hermes/common/bitmask_operators.h>
#include <hermes/common/str.h>
#include <hermes/log/console_colors.h>
#include <hermes/system/file_system.h>

#include <chrono>
#include <cstdarg>
#include <cstring>

namespace hermes {

/// \brief Options for logging output
/// \note You can use bitwise operators to combine these options
enum class logging_options {
  none = 0x00,                 //!< default behaviour
  location = 1 << 0,           //!< logs code location
  time = 1 << 1,               //!< logs message time point
  abbreviate = 1 << 2,         //!< abbreviate long paths
  use_colors = 1 << 3,         //!< output colored messages
  full_path_location = 1 << 4, //!< output full path locations
  callback_only = 1 << 5       //!< redirect output to callback only
};

HERMES_ENABLE_BITMASK_OPERATORS(logging_options);

/// \brief Static class that manages logging messages
class Log {
public:
  /// \brief Holds information about log code location
  struct Location {
    const char *file_name;     //!< file path
    int line;                  //!< file line number
    const char *function_name; //!< scope name
  };
  /// \brief Represents the log level
  enum class Level {
    debug = 0,
    trace = 1,
    info = 2,
    warn = 3,
    error = 4,
    critical = 5,
    COUNT = 6
  };

  /// \brief Logs a formatted message with code location information
  /// \tparam Ts
  /// \param message_options
  /// \param fmt
  /// \param location
  /// \param args
  template <typename... Ts>
  HERMES_DEVICE_CALLABLE static inline void
  message(logging_options message_options, Level level, const char *fmt,
          Location location, Ts &&...args) {
    if (static_cast<u8>(level) < static_cast<u8>(filter_level_))
      return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
    printf(fmt, std::forward<Ts>(args)...);
#else
    // merge options_
    message_options = options_ | message_options;
    bool use_colors =
        HERMES_MASK_BIT(message_options, logging_options::use_colors);
    // label
    Str s;
    s += label(message_options, level, location);
    // message
    if (use_colors)
      s += ConsoleColors::color(
          message_colors_[static_cast<std::size_t>(level)]);
    s += Str::format(fmt, std::forward<Ts>(args)...);
    if (use_colors)
      s += ConsoleColors::reset;
    if (log_callback)
      log_callback(s, message_options);
    if (HERMES_MASK_BIT(message_options, logging_options::callback_only))
      return;
    *os_ << s << "\n";
#endif
  }
  /// \brief Enables logging options
  /// \param options_to_add
  static inline void addOptions(logging_options options_to_add) {
    options_ = options_ | options_to_add;
  }
  /// \brief Disables logging options
  /// \param options_to_remove
  static inline void removeOptions(logging_options options_to_remove) {
    options_ = options_ & ~options_to_remove;
  }

private:
  static Str label(const logging_options &message_options, Level level,
                   const Location &location);
  static Str abbreviate(logging_options message_options, const char *str);
  static Str processPath(logging_options options,
                         const std::filesystem::path &path);

  static Level filter_level_;
  static std::ostream *os_; //!< output stream
  static logging_options options_;
  static u8 message_colors_[static_cast<u8>(Level::COUNT)];
  static u8 label_colors_[static_cast<u8>(Level::COUNT)];
  static u32 abbreviation_size_; //!< size after abbreviation (in characters)

  static std::function<void(const Str &, logging_options)>
      log_callback; //!< redirection callback
  static std::function<void(const Str &)>
      callbacks[static_cast<u8>(Level::COUNT)];
};

} // namespace hermes

// *********************************************************************************************************************
//                                                                                                            LOGGING
// *********************************************************************************************************************
#ifndef INFO_ENABLED
#define INFO_ENABLED
#endif

#ifdef INFO_ENABLED

#ifndef HERMES_PING
/// \brief Logs into info stream code location
#define HERMES_PING                                                            \
  hermes::Log::message(                                                        \
      hermes::logging_options::none, hermes::Log::Level::debug, "",            \
      hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__});
#endif

#ifndef HERMES_DEBUG
/// \brief Logs into info log stream
/// \code{cpp}
///     HERMES_DEBUG("my log with {} as value", 3) // produces "my log with 3 as
///     value" HERMES_DEBUG("simple log")
/// \endcode
/// \param FMT a const char* following hermes format (use "{}" to place each
/// value in the string)
/// \param ... format values
#define HERMES_DEBUG(FMT, ...)                                                 \
  hermes::Log::message(                                                        \
      hermes::logging_options::none, hermes::Log::Level::debug, FMT,           \
      hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__} __VA_OPT__(, )   \
          __VA_ARGS__)
#endif
/// \brief Logs into warning log stream
/// \code{cpp}
///     HERMES_LOG_TRACE("my log with {} as value", 3) // produces "my log
///     with 3 as value" HERMES_TRACE("simple log")
/// \endcode
/// \param FMT a const char* following hermes format (use "{}" to place each
/// value in the string)
/// \param ... format values
#ifndef HERMES_TRACE
#define HERMES_TRACE(FMT, ...)                                                 \
  hermes::Log::message(                                                        \
      hermes::logging_options::none, hermes::Log::Level::trace, FMT,           \
      hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__} __VA_OPT__(, )   \
          __VA_ARGS__)
#endif
/// \brief Logs into warning log stream
/// \code{cpp}
///     HERMES_LOG_INFO("my log with {} as value", 3) // produces "my log
///     with 3 as value" HERMES_INFO("simple log")
/// \endcode
/// \param FMT a const char* following hermes format (use "{}" to place each
/// value in the string)
/// \param ... format values
#ifndef HERMES_INFO
#define HERMES_INFO(FMT, ...)                                                  \
  hermes::Log::message(                                                        \
      hermes::logging_options::none, hermes::Log::Level::info, FMT,            \
      hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__} __VA_OPT__(, )   \
          __VA_ARGS__)
#endif
/// \brief Logs into warning log stream
/// \code{cpp}
///     HERMES_LOG_WARNING("my log with {} as value", 3) // produces "my log
///     with 3 as value" HERMES_WARN("simple log")
/// \endcode
/// \param FMT a const char* following hermes format (use "{}" to place each
/// value in the string)
/// \param ... format values
#ifndef HERMES_WARN
#define HERMES_WARN(FMT, ...)                                                  \
  hermes::Log::message(                                                        \
      hermes::logging_options::none, hermes::Log::Level::warn, FMT,            \
      hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__} __VA_OPT__(, )   \
          __VA_ARGS__)
#endif
/// \brief Logs into error log stream
/// \code{cpp}
///     HERMES_ERROR("my log with {} as value", 3) // produces "my log with
///     3 as value" HERMES_ERROR("simple log")
/// \endcode
/// \param FMT a const char* following hermes format (use "{}" to place each
/// value in the string)
/// \param ... format values
#ifndef HERMES_ERROR
#define HERMES_ERROR(FMT, ...)                                                 \
  hermes::Log::message(                                                        \
      hermes::logging_options::none, hermes::Log::Level::error, FMT,           \
      hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__} __VA_OPT__(, )   \
          __VA_ARGS__)
#endif
/// \brief Logs into critical log stream
/// \code{cpp}
///     HERMES_CRITICAL("my log with {} as value", 3) // produces "my log
///     with 3 as value" HERMES_CRITICAL("simple log")
/// \endcode
/// \param FMT a const char* following hermes format (use "{}" to place each
/// value in the string)
/// \param ... format values
#ifndef HERMES_CRITICAL
#define HERMES_CRITICAL(FMT, ...)                                              \
  hermes::Log::message(                                                        \
      hermes::logging_options::none, hermes::Log::Level::critical, FMT,        \
      hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__} __VA_OPT__(, )   \
          __VA_ARGS__)
#endif

#ifndef HERMES_LOG_VARIABLE
/// \brief Logs variable name and value into info log stream
/// \pre All variables must support `std::stringstream` << operator
/// \param A variable or literal
#define HERMES_LOG_VARIABLE(A)                                                 \
  hermes::Log::message(                                                        \
      hermes::logging_options::none, hermes::Log::Level::info, "{} = {}",      \
      hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__}, #A, A)
#endif

#ifndef HERMES_LOG_ARRAY
/// \brief Logs array elements into info log stream
/// \pre All elements must support `std::stringstream` << operator
/// \param array/vector object
#define HERMES_LOG_ARRAY(A)                                                    \
  HERMES_LOG("values of \"{}\":", #A);                                         \
  for (const auto &hermes_log_array_element : A)                               \
  HERMES_LOG("  {}", hermes_log_array_element)
#endif
/// \brief Auxiliary support to log multiple variables
/// \tparam T
/// \param s
/// \param first
template <typename T>
static inline void hermes_log_variables_r(std::stringstream &s,
                                          const T &first) {
  s << first << "\n";
}
/// \brief Auxiliary support to log multiple variables
/// \tparam T
/// \tparam Args
/// \param s
/// \param first
/// \param rest
template <typename T, typename... Args>
static inline void hermes_log_variables_r(std::stringstream &s, const T &first,
                                          Args &&...rest) {
  s << first << " | ";
  if constexpr (sizeof...(rest) > 0)
    hermes_log_variables_r(s, std::forward<Args>(rest)...);
}
/// \brief Auxiliary support to log multiple variables
/// \tparam Args
/// \param args
/// \return
template <class... Args>
static inline std::string hermes_log_variables(Args &&...args) {
  std::stringstream s;
  if constexpr (sizeof...(args) > 0) {
    hermes_log_variables_r(s, std::forward<Args>(args)...);
    return s.str();
  }
  return "";
}

#ifndef HERMES_LOG_VARIABLES
/// \brief Logs multiple variables into info log stream
/// \pre All variables must support `std::stringstream` << operator
/// \param ... variables
#define HERMES_LOG_VARIABLES(...)                                              \
  hermes::Log::message(                                                        \
      hermes::logging_options::none, hermes::Log::Level::info, "{}",           \
      hermes::Log::Location{__FILE__, __LINE__, __FUNCTION__},                 \
      hermes_log_variables(__VA_ARGS__))
#endif

#ifndef HERMES_C_LOG
/// \brief Logs into stdout in printf style
/// \code{cpp}
///     HERMES_C_LOG("my log with %d as value", 3) // produces "my log with 3 as
///     value" HERMES_C_LOG("simple log")
/// \endcode
/// \param FMT string format following printf format
/// \param ... format values
#define HERMES_C_LOG(FMT, ...)                                                 \
  fprintf(stdout, "[%s][%d][%s]: ", __FILE__, __LINE__, __FUNCTION__);         \
  fprintf(stdout, FMT __VA_OPT__(, ) __VA_ARGS__);                             \
  fprintf(stdout, "\n")
#endif
#ifndef HERMES_C_ERROR
/// \brief Logs into stderr in printf style
/// \code{cpp}
///     HERMES_C_ERROR("my log with %d as value", 3) // produces "my log
///     with 3 as value" HERMES_C_ERROR("simple log")
/// \endcode
/// \param FMT string format following printf format
/// \param ... format values
#define HERMES_C_ERROR(FMT, ...)                                               \
  fprintf(stderr, "[%s][%d][%s]: ", __FILE__, __LINE__, __FUNCTION__);         \
  fprintf(stderr, FMT __VA_OPT__(, ) __VA_ARGS__);                             \
  fprintf(stderr, "\n")
#endif
#ifndef HERMES_C_DEVICE_LOG
/// \brief Logs into info stdout from device code
/// \code{cpp}
///     HERMES_C_LOG("my log with %d as value", 3) // produces "my log with 3 as
///     value" HERMES_C_LOG("simple log")
/// \endcode
/// \param FMT string format following printf format
/// \param ... format values
#define HERMES_C_DEVICE_LOG(FMT, ...)                                          \
  printf("[%s][%d][%s]: ", __FILE__, __LINE__, __FUNCTION__);                  \
  printf(FMT __VA_OPT__(, ) __VA_ARGS__);                                      \
  printf("\n")
#endif
#ifndef HERMES_C_DEVICE_ERROR
/// \brief Logs into stderr from device code
/// \code{cpp}
///     HERMES_C_LOG("my log with %d as value", 3) // produces "my log with 3 as
///     value" HERMES_C_LOG("simple log")
/// \endcode
/// \param FMT string format following printf format
/// \param ... format values
#define HERMES_C_DEVICE_ERROR(FMT, ...)                                        \
  printf("[%s][%d][%s]: ", __FILE__, __LINE__, __FUNCTION__);                  \
  printf(FMT __VA_OPT__(, ) __VA_ARGS__);                                      \
  printf("\n")
#endif

#else

#define HERMES_PING
#define HERMES_LOG
#define HERMES_LOG_VARIABLE

#endif

/// @}
