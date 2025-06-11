/// Copyright (c) 2025, FilipeCN.
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
///\file parallel.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2022-01-05
///
///\brief

#include <hermes/system/threads.h>

#include <atomic>

namespace hermes {

ThreadPool::ThreadPool(std::size_t thread_count) {
  for (std::size_t i = 0; i < thread_count; ++i) {
    threads_.emplace_back([&, id = i](const std::stop_token &stop) {
      Task task;
      do {
        // acquire task
        bool task_acquired = false;
        {
          std::unique_lock<std::mutex> lock(tasks_mutex_);

          if (!tasks_.empty()) {
            task = std::move(tasks_.top());
            tasks_.pop();
            task_acquired = true;
          }
        }

        if (task_acquired) {
          enqueued_tasks_count_.fetch_sub(1, std::memory_order_release);
          running_tasks_count_.fetch_add(1, std::memory_order_release);
          std::invoke(std::move(task.f));
          running_tasks_count_.fetch_sub(1, std::memory_order_release);
          if (running_tasks_count_.load(std::memory_order_acquire) == 0 &&
              enqueued_tasks_count_.load(std::memory_order_acquire) == 0) {
            threads_complete_.store(true, std::memory_order_release);
            threads_complete_.notify_one();
          }
        }

      } while (!stop.stop_requested());
    });
  }
}

ThreadPool::~ThreadPool() {
  wait();

  for (auto &thread : threads_) {
    thread.request_stop();
    thread.join();
  }
}

void ThreadPool::wait() {
  if (enqueued_tasks_count_.load(std::memory_order_acquire) > 0 ||
      running_tasks_count_.load(std::memory_order_acquire) > 0)
    threads_complete_.wait(false);
}

} // namespace hermes
