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

/// \file   threads.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2022-01-05
/// \brief  CPU threading classes.

#pragma once

#include <hermes/core/debug.h>

#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>

namespace hermes {

// *****************************************************************************
//                                                                        Task
// *****************************************************************************

struct Task {
  enum class Priority { LOW = 0, NORMAL = 1, HIGH = 2, CRITICAL = 3 };

  std::function<void()> f; //< task code
  Priority priority;       //< task priority level

  friend bool operator<(const Task &a, const Task &b) {
    return static_cast<int>(a.priority) < static_cast<int>(b.priority);
  }
};

// *****************************************************************************
//                                                                  ThreadPool
// *****************************************************************************

// based on
// https://github.com/DeveloperPaul123/thread-pool/blob/master/include/thread_pool/thread_pool.h
//

/// Manages a thread pool
///
/// \core{.cpp}
///    int f(int a, int b) {...}
///
///    ThreadPool pool(5);
///    pool.enqueue(Task::Priority::NORMAL, f, 1, 2);
///
/// \endcode
class ThreadPool {
public:
  using Ptr = std::shared_ptr<ThreadPool>;

  /// \param thread_count max number of concurrent threads.
  ThreadPool(std::size_t thread_count);
  ~ThreadPool();

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

  /// \brief Wait until all current tasks complete.
  void wait();

  /// \brief Push a new task to the pool.
  /// \param priority the new task's priority level.
  /// \param F task function name
  /// \param args task function parameters
  /// \return future object for tasks return value
  template <typename F, typename... Args,
            typename R = std::invoke_result_t<F &&, Args &&...>>
  std::future<R> enqueue(Task::Priority priority, F &&f, Args &&...args) {
    auto shared_promise = std::make_shared<std::promise<R>>();
    auto task = [func = std::move(f), ... largs = std::move(args),
                 promise = shared_promise]() {
      try {
        if constexpr (std::is_same_v<R, void>) {
          func(largs...);
          promise->set_value();
        } else {
          promise->set_value(func(largs...));
        }

      } catch (...) {
        promise->set_exception(std::current_exception());
      }
    };

    // get the future before enqueuing the task
    auto future = shared_promise->get_future();
    // enqueue the task
    enqueueTask(priority, std::move(task));
    return future;
  }

private:
  template <typename F> void enqueueTask(Task::Priority priority, F &&f) {
    {
      const auto q_size =
          enqueued_tasks_count_.fetch_add(1, std::memory_order_release);
      if (q_size == 0)
        threads_complete_.store(false, std::memory_order_release);

      // assign task
      std::unique_lock<std::mutex> lock(tasks_mutex_);
      tasks_.push({.f = std::move(f), .priority = priority});
    }
  }

  std::vector<std::jthread> threads_;
  std::priority_queue<Task> tasks_;
  std::mutex tasks_mutex_;
  std::atomic_bool threads_complete_{false};
  std::atomic_int_fast64_t enqueued_tasks_count_{0}, running_tasks_count_{0};
};

} // namespace hermes
