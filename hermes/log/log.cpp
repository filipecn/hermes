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
///\file logging.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-06-20
///
///\brief

#include <hermes/common/bitmask_operators.h>
#include <hermes/log/console_colors.h>
#include <hermes/log/log.h>
#include <hermes/system/time.h>

#include <format>

namespace hermes {

Log::Level Log::filter_level_{Log::Level::debug};

logging_options Log::options_ =
    logging_options::use_colors | logging_options::location |
    logging_options::full_path_location | logging_options::abbreviate;

std::ostream *Log::os_{&std::cout};

u8 Log::message_colors_[] = {
    247, //  debug
    247, //  trace
    247, //  info
    191, //  warn
    9,   //  error
    197, //  critical
};

u8 Log::label_colors_[] = {
    247, //  debug
    247, //  trace
    247, //  info
    191, //  warn
    9,   //  error
    197, //  critical
};

u32 Log::abbreviation_size_ = 10;

std::function<void(const Str &, logging_options)> Log::log_callback;
std::function<void(const Str &)>
    Log::callbacks[static_cast<u8>(Log::Level::COUNT)];

Str Log::label(const logging_options &message_options, Log::Level level,
               const Log::Location &location) {
  Str s;

  if (HERMES_MASK_BIT(message_options, logging_options::use_colors))
    s += ConsoleColors::threadColor(std::this_thread::get_id());
  s += std::format("[{} | {}] ", std::this_thread::get_id(),
                   timeLabel(SystemTime::wallTime()));

  static const char *level_names[6] = {"DEBUG", "TRACE", "INFO",
                                       "WARN",  "ERROR", "CRITICAL"};
  if (HERMES_MASK_BIT(message_options, logging_options::use_colors))
    s += ConsoleColors::color(label_colors_[static_cast<std::size_t>(level)]);

  if (HERMES_MASK_BIT(message_options, logging_options::location) ||
      HERMES_MASK_BIT(message_options, logging_options::full_path_location))
    s += std::format(
        "[{}][{}][{}] ",
        processPath(message_options,
                    abbreviate(message_options, location.file_name))
            .c_str(),
        location.line,
        abbreviate(message_options, location.function_name).c_str());

  return s;
}

Str Log::abbreviate(logging_options message_options, const char *str) {
  Str s;
  if (HERMES_MASK_BIT(message_options, logging_options::abbreviate)) {
    size_t l = std::strlen(str);
    if (l > abbreviation_size_ + 3) {
      s += "...";
      s += &str[l - abbreviation_size_];
      return s;
    }
  }
  return s + str;
}

Str Log::processPath(logging_options options,
                     const std::filesystem::path &path) {
  if (!HERMES_MASK_BIT(options, logging_options::full_path_location))
    return path.stem().c_str();
  return path.c_str();
}

} // namespace hermes
