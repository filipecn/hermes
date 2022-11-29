/// Copyright (c) 2022, FilipeCN.
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
///\file str_view.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2022-11-12
///
///\brief

#include <hermes/common/str_view.h>
#include <hermes/common/debug.h>

namespace hermes {

Result<ConstStrView> ConstStrView::from(const std::string &str, size_t pos, i64 len) {
  // check str
  if (str.empty())
    return Result<ConstStrView>(ConstStrView("", 0, 0));
  // check pos
  if (pos == str.size())
    return Result<ConstStrView>(ConstStrView("", 0, 0));
  if (pos > str.size())
    return Result<ConstStrView>::error(HeResult::OUT_OF_BOUNDS);
  size_t end = 0;
  // check len
  if (len < 0)
    end = str.size();
  else
    end = std::min(str.size(), pos + static_cast<size_t>(len));
  return Result<ConstStrView>(ConstStrView(str.c_str(), pos, end));
}

Result<ConstStrView> ConstStrView::from(const char *str, size_t len, size_t pos) {
  return Result<ConstStrView>(ConstStrView(str, pos, pos + len));
}

ConstStrView::ConstStrView(const char *str, size_t start, size_t end) : str_(str), start_(start), end_(end) {
  size_ = end - start;
}

size_t ConstStrView::size() const {
  return size_;
}

bool ConstStrView::operator==(const std::string &s) const {
  if (s.size() != size_)
    return false;
  for (size_t i = 0; i < size_; ++i)
    if (s[i] != str_[start_ + i])
      return false;
  return true;
}

}
