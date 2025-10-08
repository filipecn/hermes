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
#include <type_traits>
#include <variant>

namespace hermes {

/// Holds a reference for an (owned or not) object.
template <typename T> class Ref {
public:
  using StorageType =
      std::variant<std::monostate, T *, std::shared_ptr<T>, std::weak_ptr<T>>;

  template <class... Args> static Ref shared(Args &&...args) {
    return Ref(std::make_shared<T>(std::forward<Args>(args)...));
  }
  template <typename D>
    requires std::is_base_of_v<T, D> || std::is_same_v<T, D>
  static Ref weak(const std::shared_ptr<D> &ptr) {
    std::weak_ptr<T> p = ptr;
    return Ref(p);
  }
  template <typename D>
    requires std::is_base_of_v<T, D> || std::is_same_v<T, D>
  static Ref weak(const Ref<D> &ptr) {
    std::weak_ptr<T> p = (std::shared_ptr<D>)ptr;
    return Ref(p);
  }
  static Ref ptr(T *ptr) {
    Ref r;
    r.data_ = ptr;
    return r;
  }
  Ref() = default;
  Ref(const std::weak_ptr<T> &ptr) : data_(ptr) {}

  template <typename D>
    requires std::is_base_of_v<T, D> || std::is_same_v<T, D>
  Ref(D *ptr) : data_(reinterpret_cast<T *>(ptr)) {}
  template <typename D>
    requires std::is_base_of_v<T, D> || std::is_same_v<T, D>
  Ref(const std::shared_ptr<D> &ptr) : data_(ptr) {}
  template <typename D>
    requires std::is_base_of_v<T, D> || std::is_same_v<T, D>
  Ref(const Ref<D> &ref) {
    *this = ref;
  }
  template <typename D>
    requires std::is_base_of_v<T, D> || std::is_same_v<T, D>
  Ref(Ref<D> &&ref) {
    *this = std::move(ref);
  }

  template <typename D>
    requires std::is_base_of_v<T, D> || std::is_same_v<T, D>
  Ref &operator=(const Ref<D> &rhs) {
    destroy();
    std::visit(
        [&](auto &&arg) {
          using V = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<V, D *>) {
            data_ = reinterpret_cast<T *>(arg);
          } else if constexpr (std::is_same_v<V, std::shared_ptr<D>>) {
            std::shared_ptr<T> p = arg;
            data_ = p;
          } else if constexpr (std::is_same_v<V, std::weak_ptr<D>>) {
            data_ = arg;
          } else if constexpr (std::is_same_v<V, std::monostate>) {
            data_ = {};
          } else
            static_assert(false, "hermes internal error (Ref)");
        },
        rhs.data());
    return *this;
  }

  template <typename D>
    requires std::is_base_of_v<T, D> || std::is_same_v<T, D>
  Ref &operator=(Ref<D> &&rhs) {
    destroy();
    std::visit(
        [&](auto &&arg) {
          using V = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<V, D *>) {
            data_ = reinterpret_cast<T *>(arg);
          } else if constexpr (std::is_same_v<V, std::shared_ptr<D>>) {
            std::shared_ptr<T> p = arg;
            data_ = p;
          } else if constexpr (std::is_same_v<V, std::weak_ptr<D>>) {
            data_ = arg;
          } else if constexpr (std::is_same_v<V, std::monostate>) {
            data_ = {};
          } else
            static_assert(false, "hermes internal error (Ref)");
        },
        rhs.data());
    rhs.destroy();
    return *this;
  }

  operator bool() const {
    return std::visit(
        [](auto &&arg) -> bool {
          using V = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<V, T *>) {
            return arg != nullptr;
          } else if constexpr (std::is_same_v<V, std::shared_ptr<T>>) {
            return arg.get() != nullptr;
          } else if constexpr (std::is_same_v<V, std::weak_ptr<T>>) {
            return !arg.expired();
          } else if constexpr (std::is_same_v<V, std::monostate>) {
            return false;
          } else
            static_assert(false, "hermes internal error (Ref)");
        },
        data_);
  }

  operator std::shared_ptr<T>() const { return getShared(); }

  operator std::weak_ptr<T>() const { return getWeak(); }

  bool isWeak() const {
    return std::visit(
        [](auto &&arg) -> bool {
          using V = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<V, T *>) {
            return false;
          } else if constexpr (std::is_same_v<V, std::shared_ptr<T>>) {
            return false;
          } else if constexpr (std::is_same_v<V, std::weak_ptr<T>>) {
            return true;
          } else if constexpr (std::is_same_v<V, std::monostate>) {
            return false;
          } else
            static_assert(false, "hermes internal error (Ref)");
        },
        data_);
  }

  bool isShared() const {
    return std::visit(
        [](auto &&arg) -> bool {
          using V = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<V, T *>) {
            return false;
          } else if constexpr (std::is_same_v<V, std::shared_ptr<T>>) {
            return true;
          } else if constexpr (std::is_same_v<V, std::weak_ptr<T>>) {
            return false;
          } else if constexpr (std::is_same_v<V, std::monostate>) {
            return false;
          } else
            static_assert(false, "hermes internal error (Ref)");
        },
        data_);
  }

  bool isPtr() const {
    return std::visit(
        [](auto &&arg) -> bool {
          using V = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<V, T *>) {
            return true;
          } else if constexpr (std::is_same_v<V, std::shared_ptr<T>>) {
            return false;
          } else if constexpr (std::is_same_v<V, std::weak_ptr<T>>) {
            return false;
          } else if constexpr (std::is_same_v<V, std::monostate>) {
            return false;
          } else
            static_assert(false, "hermes internal error (Ref)");
        },
        data_);
  }

  T *get() {
    return std::visit(
        [](auto &&arg) -> T * {
          using V = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<V, T *>) {
            return arg;
          } else if constexpr (std::is_same_v<V, std::shared_ptr<T>>) {
            return arg.get();
          } else if constexpr (std::is_same_v<V, std::weak_ptr<T>>) {
            return arg.lock().get();
          } else if constexpr (std::is_same_v<V, std::monostate>) {
            return nullptr;
          } else
            static_assert(false, "hermes internal error (Ref)");
        },
        data_);
  }

  const T *get() const {
    return std::visit(
        [](auto &&arg) -> const T * {
          using V = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<V, T *>) {
            return arg;
          } else if constexpr (std::is_same_v<V, std::shared_ptr<T>>) {
            return arg.get();
          } else if constexpr (std::is_same_v<V, std::weak_ptr<T>>) {
            return arg.lock().get();
          } else if constexpr (std::is_same_v<V, std::monostate>) {
            return nullptr;
          } else
            static_assert(false, "hermes internal error (Ref)");
        },
        data_);
  }

  T &operator*() {
    return std::visit(
        [](auto &&arg) -> T & {
          using V = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<V, T *>) {
            return *arg;
          } else if constexpr (std::is_same_v<V, std::shared_ptr<T>>) {
            return *arg;
          } else if constexpr (std::is_same_v<V, std::weak_ptr<T>>) {
            return *(arg.lock());
          } else if constexpr (std::is_same_v<V, std::monostate>) {
            return dummy_;
          } else
            static_assert(false, "hermes internal error (Ref)");
        },
        data_);
  }

  const T &operator*() const {
    return std::visit(
        [](auto &&arg) -> const T & {
          using V = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<V, T *>) {
            return *arg;
          } else if constexpr (std::is_same_v<V, std::shared_ptr<T>>) {
            return *arg;
          } else if constexpr (std::is_same_v<V, std::weak_ptr<T>>) {
            return *(arg.lock());
          } else if constexpr (std::is_same_v<V, std::monostate>) {
            return dummy_;
          } else
            static_assert(false, "hermes internal error (Ref)");
        },
        data_);
  }

  T *operator->() { return get(); }

  const T *operator->() const { return get(); }

  void destroy() { data_ = {}; }

  StorageType &data() { return data_; }

  const StorageType &data() const { return data_; }

  std::shared_ptr<T> getShared() const {
    return std::visit(
        [](auto &&arg) -> std::shared_ptr<T> {
          using V = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<V, T *>) {
            return std::shared_ptr<T>();
          } else if constexpr (std::is_same_v<V, std::shared_ptr<T>>) {
            return arg;
          } else if constexpr (std::is_same_v<V, std::weak_ptr<T>>) {
            return arg.lock();
          } else if constexpr (std::is_same_v<V, std::monostate>) {
            return std::shared_ptr<T>();
          } else
            static_assert(false, "hermes internal error (Ref)");
        },
        data_);
  }

  std::weak_ptr<T> getWeak() const {
    return std::visit(
        [](auto &&arg) -> std::weak_ptr<T> {
          using V = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<V, T *>) {
            return std::weak_ptr<T>();
          } else if constexpr (std::is_same_v<V, std::shared_ptr<T>>) {
            return arg;
          } else if constexpr (std::is_same_v<V, std::weak_ptr<T>>) {
            return arg;
          } else if constexpr (std::is_same_v<V, std::monostate>) {
            return std::shared_ptr<T>();
          } else
            static_assert(false, "hermes internal error (Ref)");
        },
        data_);
  }

private:
  static T dummy_;

  StorageType data_;
};

template <typename T> T Ref<T>::dummy_{};

} // namespace hermes
