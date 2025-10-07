/* Copyright (c) 2025, FilipeCN.
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

/// \file   ref.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2025-10-07
/// \brief  Auxiliary object for holding references.

#pragma once

#include <hermes/core/result.h>
#include <memory>
#include <variant>

namespace hermes {

/// Holds a reference for an (owned or not) object.
template <typename T> class Ref {
public:
  template <class... Args> static Ref shared(Args &&...args) {
    Ref r;
    r.data_ = std::make_shared<T>(std::forward<Args>(args)...);
    return r;
  }
  static Ref ptr(T *ptr) {
    Ref r;
    r.data_ = ptr;
    return r;
  }
  Ref() = default;
  Ref(T *ptr) : data_(ptr) {}
  Ref(std::shared_ptr<T> ptr) : data_(ptr) {}

  T *operator*() {
    T *d = nullptr;
    std::visit(
        [&d](auto &&arg) {
          using V = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<V, T *>) {
            d = arg;
          } else if constexpr (std::is_same_v<V, std::shared_ptr<T>>) {
            d = arg.get();
          } else
            static_assert(false, "hermes internal error (Ref)");
        },
        data_);
    return d;
  }

  const T *operator*() const {
    const T *d = nullptr;
    std::visit(
        [&d](auto &&arg) {
          using V = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<V, T *>) {
            d = arg;
          } else if constexpr (std::is_same_v<V, std::shared_ptr<T>>) {
            d = arg.get();
          } else
            static_assert(false, "hermes internal error (Ref)");
        },
        data_);
    return d;
  }

private:
  std::variant<T *, std::shared_ptr<T>> data_;
};

} // namespace hermes
