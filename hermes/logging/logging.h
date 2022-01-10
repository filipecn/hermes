//
// Created by filipecn on 20/06/2021.
//


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
///\brief

#ifndef HERMES_LOG_LOGGING_H
#define HERMES_LOG_LOGGING_H

#include <hermes/common/str.h>
#include <hermes/logging/console_colors.h>
#include <hermes/common/bitmask_operators.h>
#include <hermes/common/file_system.h>
#include <cstdarg>
#include <chrono>

namespace hermes {

enum class logging_options {
  none = 0x00,
  info = 0x01,
  warn = 0x02,
  error = 0x04,
  critical = 0x08,
  location = 0x10,
  time = 0x10,
  abbreviate = 0x20,
  use_colors = 0x40,
  full_path_location = 0x80
};

HERMES_ENABLE_BITMASK_OPERATORS(logging_options);

class Log {
public:
  struct Location {
    const char *file_name;
    int line;
    const char *function_name;
  };

  /// \return
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

  template<typename ...Ts>
  HERMES_DEVICE_CALLABLE static inline void logMessage(logging_options message_options, const char *fmt,
                                                       Location location,
                                                       Ts &&...args) {
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
    else
      printf("%s\n", s.c_str());
  }

  template<typename ...Ts>
  HERMES_DEVICE_CALLABLE static inline void logMessage(logging_options message_options, const char *fmt, Ts &&...args) {
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
    else
      printf("%s\n", s.c_str());
  }

  ///
  /// \tparam Ts
  /// \param fmt
  /// \param args
  template<typename ...Ts>
  HERMES_DEVICE_CALLABLE static inline void info(const char *fmt, Ts &&...args) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    printf("[%s][%d][%s] INFO\n", std::forward<Ts>(args)...);
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
  ///
  /// \tparam Ts
  /// \param fmt
  /// \param args
  template<typename ...Ts>
  HERMES_DEVICE_CALLABLE static inline void warn(const char *fmt, Ts &&...args) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    printf("[%s][%d][%s] WARN\n", std::forward<Ts>(args)...);
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
  ///
  /// \tparam Ts
  /// \param fmt
  /// \param args
  template<typename ...Ts>
  static HERMES_DEVICE_CALLABLE inline void error(const char *fmt, Ts &&...args) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    printf("[%s][%d][%s] ERROR\n", std::forward<Ts>(args)...);
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
  ///
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

  HERMES_DEVICE_CALLABLE static inline void addOptions(logging_options options_to_add) {
    options_ = options_ | options_to_add;
  }

  HERMES_DEVICE_CALLABLE static inline void removeOptions(logging_options options_to_remove) {
    options_ = options_ & ~options_to_remove;
  }

  static u8 info_color;
  static u8 warn_color;
  static u8 error_color;
  static u8 critical_color;

  static u8 info_label_color;
  static u8 warn_label_color;
  static u8 error_label_color;
  static u8 critical_label_color;

  static size_t abbreviation_size;

  static std::function<void(const Str &, logging_options)> log_callback;
  static std::function<void(const Str &)> info_callback;
  static std::function<void(const Str &)> warn_callback;
  static std::function<void(const Str &)> error_callback;
  static std::function<void(const Str &)> critical_callback;

private:
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

#endif //HERMES_LOG_LOGGING_H
