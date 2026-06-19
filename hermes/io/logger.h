/* Copyright (c) 2021, FilipeCN.
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

/// \file   logger.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2021-06-20
/// \brief  Logging functions

#pragma once

#include <hermes/base/flags.h>
#include <hermes/base/str.h>
#include <hermes/io/console_colors.h>

#include <cstring>

namespace hermes::io {

/// \brief Options for logging output.
/// \note You can use bitwise operators to combine these options.
enum class logger_option_bits : u32 {
  none = 0x00,                 //!< default behavior
  location = 1 << 0,           //!< logs code location
  time = 1 << 1,               //!< logs message time point
  abbreviate = 1 << 2,         //!< abbreviate long paths
  use_colors = 1 << 3,         //!< output colored messages
  full_path_location = 1 << 4, //!< output full path locations
  callback_only = 1 << 5       //!< redirect output to callback only
};

using logger_options = hermes::Flags<hermes::io::logger_option_bits>;

} // namespace hermes::io

namespace hermes {

template <> struct FlagTraits<io::logger_option_bits> {
  static HERMES_CONST_OR_CONSTEXPR bool is_bitmask = true;
  static HERMES_CONST_OR_CONSTEXPR io::logger_options all_flags =
      io::logger_option_bits::location | io::logger_option_bits::time |
      io::logger_option_bits::abbreviate | io::logger_option_bits::use_colors |
      io::logger_option_bits::full_path_location |
      io::logger_option_bits::callback_only;
};

} // namespace hermes

namespace hermes::io {

// *****************************************************************************
//                                                                      LOGGER
// *****************************************************************************

/// Static class that manages logging messages
class Logger {
public:
  /// Holds information about log call location
  struct Location {
    const char *file_name;     //!< file path
    int line;                  //!< file line number
    const char *function_name; //!< scope name
  };
  /// Represents the log level.
  /// \note Log messages can be filtered by level.
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
  HERMES_CPU_GPU static inline void message(logger_options message_options,
                                            Level level, const char *fmt,
                                            Location location, Ts &&...args) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
    // printf(fmt, std::forward<Ts>(args)...);
    printf("CUDA LOG not supported");
    return;
#else
    if (static_cast<u8>(level) < static_cast<u8>(filter_level_))
      return;
    // merge options_
    message_options = options_ | message_options;
    bool use_colors = message_options.contain(logger_option_bits::use_colors);
    cstr s;
    // message
    if (use_colors)
      s += colors::console::color(
          message_colors_[static_cast<std::size_t>(level)]);
    s += std::vformat(fmt, std::make_format_args(args...));
    if (use_colors)
      s += colors::console::reset;
    if (log_callback_)
      log_callback_(s, level, message_options);
    if (callbacks_[static_cast<int>(level)])
      callbacks_[static_cast<int>(level)](s);

    // insert a label in every line break
    auto label_txt = label(message_options, level, location);
    cstr final;
    auto lines = Str<char>::split(s.str(), "\n");
    for (const auto &line : lines) {
      if (!line.empty()) {
        final += label_txt;
        final += line;
        if (line.back() != '\n')
          final += '\n';
      }
    }
    if (message_options.contain(logger_option_bits::callback_only))
      return;
    *os_ << final;
#endif
  }
  /// \brief Enables logging options
  /// \param options_to_add
  static void addOptions(logger_options options_to_add);
  /// \brief Disables logging options
  /// \param options_to_remove
  static void removeOptions(logger_options options_to_remove);
  /// \brief
  /// \param s ostream pointer
  static void setStream(std::ostream *s);
  /// \brief Sets minimal log level
  /// \param level filter level
  static void setLevel(Level level);
  /// Sets a callback function that is called for every log level
  /// \param callback
  static void setLogCallback(
      const std::function<void(const cstr &, Level, logger_options)> &callback);
  /// Sets a callback function that is called for the given log level calls
  /// \parma level
  /// \param callback
  static void setLogCallback(Level level,
                             const std::function<void(const cstr &)> &callback);

private:
  static cstr label(const logger_options &message_options, Level level,
                    const Location &location);
  static cstr abbreviate(logger_options message_options, const char *str);
  static cstr processPath(logger_options options,
                          const std::filesystem::path &path);

  static Level filter_level_; //!< filter all messages at least at filter level
  static std::ostream *os_;   //!< output stream
  static logger_options options_;
  static u8 message_colors_[static_cast<u8>(Level::COUNT)];
  static u8 label_colors_[static_cast<u8>(Level::COUNT)];
  static u32 abbreviation_size_; //!< size after abbreviation (in characters)

  static std::function<void(const cstr &, Level, logger_options)>
      log_callback_; //!< redirection callback
  static std::function<void(const cstr &)>
      callbacks_[static_cast<u8>(Level::COUNT)];
};

} // namespace hermes::io

#ifndef INFO_ENABLED
#define INFO_ENABLED
#endif

#ifdef INFO_ENABLED

#ifndef HERMES_PING
/// \brief Logs into info stream code location
#define HERMES_PING                                                            \
  hermes::io::Logger::message(                                                 \
      hermes::io::logger_option_bits::none, hermes::io::Logger::Level::debug,  \
      "", hermes::io::Logger::Location{__FILE__, __LINE__, __FUNCTION__});
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
  hermes::io::Logger::message(                                                 \
      hermes::io::logger_option_bits::none, hermes::io::Logger::Level::debug,  \
      FMT,                                                                     \
      hermes::io::Logger::Location{__FILE__, __LINE__,                         \
                                   __FUNCTION__} __VA_OPT__(, ) __VA_ARGS__)
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
  hermes::io::Logger::message(                                                 \
      hermes::io::logger_option_bits::none, hermes::io::Logger::Level::trace,  \
      FMT,                                                                     \
      hermes::io::Logger::Location{__FILE__, __LINE__,                         \
                                   __FUNCTION__} __VA_OPT__(, ) __VA_ARGS__)
#endif
/// \brief Logs into warning log stream
/// \code{cpp}
///     HERMES_INFO("my log with {} as value", 3) // produces "my log
///     with 3 as value" HERMES_INFO("simple log")
/// \endcode
/// \param FMT a const char* following hermes format (use "{}" to place each
/// value in the string)
/// \param ... format values
#ifndef HERMES_INFO
#define HERMES_INFO(FMT, ...)                                                  \
  hermes::io::Logger::message(                                                 \
      hermes::io::logger_option_bits::none, hermes::io::Logger::Level::info,   \
      FMT,                                                                     \
      hermes::io::Logger::Location{__FILE__, __LINE__,                         \
                                   __FUNCTION__} __VA_OPT__(, ) __VA_ARGS__)
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
  hermes::io::Logger::message(                                                 \
      hermes::io::logger_option_bits::none, hermes::io::Logger::Level::warn,   \
      FMT,                                                                     \
      hermes::io::Logger::Location{__FILE__, __LINE__,                         \
                                   __FUNCTION__} __VA_OPT__(, ) __VA_ARGS__)
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
  hermes::io::Logger::message(                                                 \
      hermes::io::logger_option_bits::none, hermes::io::Logger::Level::error,  \
      FMT,                                                                     \
      hermes::io::Logger::Location{__FILE__, __LINE__,                         \
                                   __FUNCTION__} __VA_OPT__(, ) __VA_ARGS__)
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
  hermes::io::Logger::message(                                                 \
      hermes::io::logger_option_bits::none,                                    \
      hermes::io::Logger::Level::critical, FMT,                                \
      hermes::io::Logger::Location{__FILE__, __LINE__,                         \
                                   __FUNCTION__} __VA_OPT__(, ) __VA_ARGS__)
#endif

#ifndef HERMES_LOG_VARIABLE
/// \brief Logs variable name and value into info log stream
/// \pre All variables must support `std::stringstream` << operator
/// \param A variable or literal
#define HERMES_LOG_VARIABLE(A)                                                 \
  hermes::io::Logger::message(                                                 \
      hermes::io::logger_option_bits::none, hermes::io::Logger::Level::info,   \
      "{} = {}",                                                               \
      hermes::io::Logger::Location{__FILE__, __LINE__, __FUNCTION__}, #A,      \
      hermes::to_string(A))
#endif

#ifndef HERMES_LOG_ARRAY
/// \brief Logs array elements into info log stream
/// \pre All elements must support `std::stringstream` << operator
/// \param array/vector object
#define HERMES_LOG_ARRAY(A)                                                    \
  hermes::io::Logger::message(                                                 \
      hermes::io::logger_option_bits::none, hermes::io::Logger::Level::info,   \
      "values of \"{}\":",                                                     \
      hermes::io::Logger::Location{__FILE__, __LINE__, __FUNCTION__}, #A);     \
  for (const auto &hermes_log_array_element : A)                               \
    hermes::io::Logger::message(                                               \
        hermes::io::logger_option_bits::none, hermes::io::Logger::Level::info, \
        "  {}",                                                                \
        hermes::io::Logger::Location{__FILE__, __LINE__, __FUNCTION__},        \
        hermes::to_string(hermes_log_array_element));
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
  hermes::io::Logger::message(                                                 \
      hermes::io::logger_option_bits::none, hermes::io::Logger::Level::info,   \
      "{}", hermes::io::Logger::Location{__FILE__, __LINE__, __FUNCTION__},    \
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
