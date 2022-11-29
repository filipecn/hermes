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

#include <hermes/logging/logging.h>

namespace hermes {

logging_options
    Log::options_ = logging_options::use_colors | logging_options::location | logging_options::full_path_location
    | logging_options::abbreviate;

u8 Log::info_color = 247;//253;
u8 Log::warn_color = 191;//215;
u8 Log::error_color = 9;
u8 Log::critical_color = 197;//196;

u8 Log::info_label_color = 247;
u8 Log::warn_label_color = 191;
u8 Log::error_label_color = 9;
u8 Log::critical_label_color = 197;

size_t Log::abbreviation_size = 10;

std::function<void(const Str &, logging_options)> Log::log_callback;
std::function<void(const Str &)> Log::info_callback;
std::function<void(const Str &)> Log::warn_callback;
std::function<void(const Str &)> Log::error_callback;
std::function<void(const Str &)> Log::critical_callback;

}

