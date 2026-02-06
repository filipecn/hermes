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

/// \file   logging.cpp
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2021-06-20

#include <hermes/io/logger.h>

#include <hermes/system/time.h>

#include <format>

namespace hermes::io {

Logger::Level Logger::filter_level_{Logger::Level::debug};

logger_options Logger::options_ =
    logger_option_bits::use_colors | logger_option_bits::location |
    logger_option_bits::full_path_location | logger_option_bits::abbreviate;

std::ostream *Logger::os_{&std::cout};

u8 Logger::message_colors_[] = {
    247, //  debug
    247, //  trace
    247, //  info
    191, //  warn
    9,   //  error
    197, //  critical
};

u8 Logger::label_colors_[] = {
    247, //  debug
    247, //  trace
    247, //  info
    191, //  warn
    9,   //  error
    197, //  critical
};

u32 Logger::abbreviation_size_ = 10;

std::function<void(const cstr &, Logger::Level, logger_options)>
    Logger::log_callback_;
std::function<void(const cstr &)>
    Logger::callbacks_[static_cast<u8>(Logger::Level::COUNT)];

void Logger::addOptions(logger_options options_to_add) {
  options_ = options_ | options_to_add;
}
void Logger::removeOptions(logger_options options_to_remove) {
  options_ = options_ & ~options_to_remove;
}
void Logger::setStream(std::ostream *s) { os_ = s; }

void Logger::setLevel(Level level) { filter_level_ = level; }

void Logger::setLogCallback(
    const std::function<void(const cstr &, Level, logger_options)> &callback) {
  log_callback_ = callback;
}

void Logger::setLogCallback(Level level,
                            const std::function<void(const cstr &)> &callback) {
  callbacks_[static_cast<int>(level)] = callback;
}

cstr Logger::label(const logger_options &message_options, Logger::Level level,
                   const Logger::Location &location) {
  cstr s;

  if (message_options.contain(logger_option_bits::use_colors))
    s += colors::console::threadColor(std::this_thread::get_id());
  s += std::format("[{} | {}] ", std::this_thread::get_id(),
                   timeLabel(SystemTime::wallTime()));

  static const char *level_names[6] = {"DEBUG", "TRACE", "INFO",
                                       "WARN",  "ERROR", "CRITICAL"};
  if (message_options.contain(logger_option_bits::use_colors))
    s += colors::console::color(label_colors_[static_cast<std::size_t>(level)]);

  if (message_options.contain(logger_option_bits::location) ||
      message_options.contain(logger_option_bits::full_path_location))
    s +=
        std::format("[{}][{}][{}][{}] ",
                    processPath(message_options,
                                abbreviate(message_options, location.file_name))
                        .c_str(),
                    location.line,
                    abbreviate(message_options, location.function_name).c_str(),
                    level_names[(u8)level]);

  return s;
}

cstr Logger::abbreviate(logger_options message_options, const char *str) {
  cstr s;
  if (message_options.contain(logger_option_bits::abbreviate)) {
    size_t l = std::strlen(str);
    if (l > abbreviation_size_ + 3) {
      s += "...";
      s += &str[l - abbreviation_size_];
      return s;
    }
  }
  return s + str;
}

cstr Logger::processPath(logger_options options,
                         const std::filesystem::path &path) {
  if (!options.contain(logger_option_bits::full_path_location))
#ifdef HERMES_WINDOWS
    return path.stem().string().c_str();
  return path.string().c_str();
#else
    return path.stem().c_str();
  return path.c_str();
#endif
}

} // namespace hermes::io
