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

#ifndef HERMES_LOG_LOGGING_H
#define HERMES_LOG_LOGGING_H

#include <hermes/common/str.h>
#include <hermes/logging/console_colors.h>
#include <hermes/common/bitmask_operators.h>
#include <hermes/common/file_system.h>
#include <cstdarg>
#include <chrono>

namespace hermes {

/// \brief Options for logging output
/// \note You can use bitwise operators to combine these options
enum class logging_options {
  none = 0x00,                  //!< default behaviour
  info = 0x01,                  //!< logs into info stream
  warn = 0x02,                  //!< logs into warning stream
  error = 0x04,                 //!< logs into error stream
  critical = 0x08,              //!< logs into critical stream
  location = 0x10,              //!< logs code location
  time = 0x10,                  //!< logs message time point
  abbreviate = 0x20,            //!< abbreviate long paths
  use_colors = 0x40,            //!< output colored messages
  full_path_location = 0x80,    //!< output full path locations
  callback_only = 0x100         //!< redirect output to callback only
};

HERMES_ENABLE_BITMASK_OPERATORS(logging_options);

/// \brief Static class that manages logging messages
class Log {
public:
  /// \brief Holds information about log code location
  struct Location {
    const char *file_name;      //!< file path
    int line;                   //!< file line number
    const char *function_name;  //!< scope name
  };

  /// \brief Logs a formatted message with code location information
  /// \tparam Ts
  /// \param message_options
  /// \param fmt
  /// \param location
  /// \param args
  template<typename ...Ts>
  HERMES_DEVICE_CALLABLE static inline void logMessage(logging_options message_options, const char *fmt,
                                                       Location location,
                                                       Ts &&...args) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
    printf(fmt);
#else
    // merge options_
    message_options = options_ | message_options;
    bool use_colors = HERMES_MASK_BIT(message_options, logging_options::use_colors);
    u8 color, label_color;
    configFrom(message_options, color, label_color);
    // label
    Str s;
    if (use_colors)
      s += ConsoleColors::color(color);
    s += label(message_options);
    // location
    if (HERMES_MASK_BIT(message_options, logging_options::location))
      s += Str::format("[{}][{}][{}] ",
                       processPath(message_options, abbreviate(message_options, location.file_name)), location.line,
                       abbreviate(message_options, location.function_name));
    // message
    if (use_colors)
      s += ConsoleColors::color(label_color);
    s += Str::format(fmt, std::forward<Ts>(args)...);
    if (use_colors)
      s += ConsoleColors::reset;
    if (log_callback)
      log_callback(s, message_options);
    if (HERMES_MASK_BIT(message_options, logging_options::callback_only))
      return;
    printf("%s\n", s.c_str());
#endif
  }
  /// \brief Logs a formatted message
  /// \tparam Ts
  /// \param message_options
  /// \param fmt
  /// \param args
  template<typename ...Ts>
  HERMES_DEVICE_CALLABLE static inline void logMessage(logging_options message_options, const char *fmt, Ts &&...args) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
    printf(fmt);
#else
    // merge options_
    message_options = options_ | message_options;
    bool use_colors = HERMES_MASK_BIT(message_options, logging_options::use_colors);
    u8 color, label_color;
    configFrom(message_options, color, label_color);
    // label
    Str s;
    if (use_colors)
      s += ConsoleColors::color(color);
    s += label(message_options);
    // message
    if (use_colors)
      s += ConsoleColors::color(label_color);
    s += Str::format(fmt, std::forward<Ts>(args)...);
    if (use_colors)
      s += ConsoleColors::reset;
    if (log_callback)
      log_callback(s, message_options);
    if (HERMES_MASK_BIT(message_options, logging_options::callback_only))
      return;
    printf("%s\n", s.c_str());
#endif
  }
  /// \brief Logs into info stream
  /// \tparam Ts
  /// \param fmt
  /// \param args
  template<typename ...Ts>
  HERMES_DEVICE_CALLABLE static inline void info(const char *fmt, Ts &&...args) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    printf(fmt, std::forward<Ts>(args)...);
#else
    auto message_options = options_ | logging_options::info;
    bool use_colors = HERMES_MASK_BIT(message_options, logging_options::use_colors);
    Str s;
    if (use_colors)
      s += ConsoleColors::color(info_label_color);
    s += label(message_options);
    if (use_colors)
      s += ConsoleColors::color(info_color);
    s += Str::format(fmt, std::forward<Ts>(args)...);
    if (use_colors)
      s += ConsoleColors::reset;
    if (info_callback)
      info_callback(s);
    else
      printf("%s\n", s.c_str());
#endif
  }
  /// \brief Logs into warn stream
  /// \tparam Ts
  /// \param fmt
  /// \param args
  template<typename ...Ts>
  HERMES_DEVICE_CALLABLE static inline void warn(const char *fmt, Ts &&...args) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    printf(fmt, std::forward<Ts>(args)...);
#else
    auto message_options = options_ | logging_options::warn;
    bool use_colors = HERMES_MASK_BIT(message_options, logging_options::use_colors);
    Str s;
    if (use_colors)
      s += ConsoleColors::color(warn_label_color);
    s += label(message_options);
    if (use_colors)
      s += ConsoleColors::color(warn_color);
    s += Str::format(fmt, std::forward<Ts>(args)...);
    if (use_colors)
      s += ConsoleColors::reset;
    if (warn_callback)
      warn_callback(s);
    else
      printf("%s\n", s.c_str());
#endif
  }
  /// \brief Logs into error stream
  /// \tparam Ts
  /// \param fmt
  /// \param args
  template<typename ...Ts>
  static HERMES_DEVICE_CALLABLE inline void error(const char *fmt, Ts &&...args) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    printf(fmt, std::forward<Ts>(args)...);
#else
    auto message_options = options_ | logging_options::error;
    bool use_colors = HERMES_MASK_BIT(message_options, logging_options::use_colors);
    Str s;
    if (use_colors)
      s += ConsoleColors::color(error_label_color);
    s += label(message_options);
    if (use_colors)
      s += ConsoleColors::color(error_color);
    s += Str::format(fmt, std::forward<Ts>(args)...);
    if (use_colors)
      s += ConsoleColors::reset;
    if (error_callback)
      error_callback(s);
    else
      printf("%s\n", s.c_str());
#endif
  }
  /// \brief Logs into critical stream
  /// \tparam Ts
  /// \param fmt
  /// \param args
  template<typename ...Ts>
  HERMES_DEVICE_CALLABLE static inline void critical(const std::string &fmt, Ts &&...args) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    printf("[%s][%d][%s] CRITICAL\n", std::forward<Ts>(args)...);
#else
    auto message_options = options_ | logging_options::critical;
    bool use_colors = HERMES_MASK_BIT(message_options, logging_options::use_colors);
    Str s;
    if (use_colors)
      s += ConsoleColors::color(critical_label_color);
    s += label(message_options);
    if (use_colors)
      s += ConsoleColors::color(critical_color);
    s += Str::format(fmt, std::forward<Ts>(args)...);
    if (use_colors)
      s += ConsoleColors::reset;
    if (critical_callback)
      critical_callback(s);
    else
      printf("%s\n", s.c_str());
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

  static u8 info_color;                     //!< info stream messages color
  static u8 warn_color;                     //!< warn stream messages color
  static u8 error_color;                    //!< error stream messages color
  static u8 critical_color;                 //!< critical stream messages color

  static u8 info_label_color;               //!< info stream label color
  static u8 warn_label_color;               //!< warn stream label color
  static u8 error_label_color;              //!< error stream label color
  static u8 critical_label_color;           //!< critical stream label color

  static size_t abbreviation_size;          //!< size after abbreviation (in characters)

  static std::function<void(const Str &, logging_options)> log_callback;   //!< redirection callback
  static std::function<void(const Str &)> info_callback;                   //!< info stream redirection callback
  static std::function<void(const Str &)> warn_callback;                   //!< warn stream redirection callback
  static std::function<void(const Str &)> error_callback;                  //!< error stream redirection callback
  static std::function<void(const Str &)> critical_callback;               //!< critical stream redirection callback

private:

  static inline Str label(const logging_options &message_options) {
    const std::chrono::time_point<std::chrono::system_clock> now =
        std::chrono::system_clock::now();
    const std::time_t t_c = std::chrono::system_clock::to_time_t(now);
    Str s;
    s += std::put_time(std::localtime(&t_c), "[%F %T]");
    if (HERMES_MASK_BIT(message_options, logging_options::info))
      s += " [info] ";
    else if (HERMES_MASK_BIT(message_options, logging_options::warn))
      s += " [warn] ";
    else if (HERMES_MASK_BIT(message_options, logging_options::error))
      s += " [error] ";
    else if (HERMES_MASK_BIT(message_options, logging_options::critical))
      s += " [critical] ";
    return s;
  }

  static inline void configFrom(const logging_options &message_options,
                                u8 &label_color,
                                u8 &color) {
    if (HERMES_MASK_BIT(message_options, logging_options::info)) {
      color = info_color;
      label_color = info_label_color;
    } else if (HERMES_MASK_BIT(message_options, logging_options::warn)) {
      color = warn_color;
      label_color = warn_label_color;
    } else if (HERMES_MASK_BIT(message_options, logging_options::error)) {
      color = error_color;
      label_color = error_label_color;
    } else if (HERMES_MASK_BIT(message_options, logging_options::critical)) {
      color = critical_color;
      label_color = critical_label_color;
    }
  }

  static inline Str abbreviate(logging_options message_options, const char *str) {
    Str s;
    if (HERMES_MASK_BIT(message_options, logging_options::abbreviate)) {
      size_t l = strlen(str);
      if (l > abbreviation_size + 3) {
        s += "...";
        s += &str[l - abbreviation_size];
        return s;
      }
    }
    return s + str;
  }

  static inline Str processPath(logging_options options, const hermes::Path &path) {
    if (!HERMES_MASK_BIT(options, logging_options::full_path_location))
      return path.name();
    return path.fullName();
  }

  static logging_options options_;
};

}

// *********************************************************************************************************************
//                                                                                                            LOGGING
// *********************************************************************************************************************
#ifndef INFO_ENABLED
#define INFO_ENABLED
#endif

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

#ifndef HERMES_LOG_ARRAY
/// \brief Logs array elements into info log stream
/// \pre All elements must support `std::stringstream` << operator
/// \param array/vector object
#define HERMES_LOG_ARRAY(A)                                            \
     HERMES_LOG("values of \"{}\":", #A);               \
     for(const auto& hermes_log_array_element : A)                   \
     HERMES_LOG("  {}", hermes_log_array_element)
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

#endif //HERMES_LOG_LOGGING_H

/// @}
