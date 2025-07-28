/* Copyright (c) 2022, FilipeCN.
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

/// \file   profile.cpp
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2022-01-05
///
///\brief

#include <hermes/system/profile.h>

#include <hermes/base/str.h>
#include <hermes/core/debug.h>
#include <hermes/io/console_colors.h>

#include <coroutine>
#include <mutex>

namespace hermes::profile {

Profiler::BlockDescriptor::BlockDescriptor(u32 id, u32 color, u32 line,
                                           const char *name)
    : id(id), color(color), line(line), name(name) {}

Profiler::BlockDescriptor::BlockDescriptor(const char *name, u32 color)
    : color(color), name(name) {}

u32 Profiler::pushBlockDescriptor(const char *name, u32 color) {
  auto &p = instance();
  std::unique_lock<std::shared_mutex> lock(p.block_descriptors_mutex_);
  auto desc = new BlockDescriptor(name, color);
  p.block_descriptors_.emplace_back(desc);
  return p.block_descriptors_.size() - 1;
}

void Profiler::startBlock(Profiler::Block &block) {
  auto &p = instance();
  Profile::Ptr prof;
  {
    std::shared_lock<std::shared_mutex> lock(p.thread_profiles_mutex_);
    auto it = p.thread_profiles_.find(std::this_thread::get_id());
    if (it != p.thread_profiles_.end())
      prof = it->second;
  }
  if (!prof) {
    std::unique_lock<std::shared_mutex> lock(p.thread_profiles_mutex_);
    prof = p.thread_profiles_[std::this_thread::get_id()] =
        std::make_shared<Profile>();
  }

  block.start();
  block.level = prof->block_stack.size();

  if (p.max_block_count_ && prof->block_list.size() >= p.max_block_count_) {
    prof->block_stack.push(prof->block_list_start);
    prof->block_list[prof->block_list_start] = block;
    prof->block_list_start = (prof->block_list_start + 1) % p.max_block_count_;
  } else {
    prof->block_stack.push(prof->block_list.size());
    prof->block_list.emplace_back(block);
  }
}

void Profiler::endBlock() {
  auto &p = instance();
  Profile::Ptr prof;
  {
    std::shared_lock<std::shared_mutex> lock(p.thread_profiles_mutex_);
    auto it = p.thread_profiles_.find(std::this_thread::get_id());
    if (it != p.thread_profiles_.end())
      prof = it->second;
  }
  if (!prof)
    return;
  if (!prof->block_stack.empty()) {
    auto top = prof->block_stack.top();
    prof->block_stack.pop();
    prof->block_list[top].end();
  }
}

std::generator<std::tuple<std::thread::id, const Profiler::Block &>>
Profiler::iterateBlocks() {
  auto &p = instance();

  std::shared_lock<std::shared_mutex> lock(p.thread_profiles_mutex_);

  for (const auto &item : p.thread_profiles_) {
    auto thread_id = item.first;
    auto prof = item.second;

    u64 m = p.max_block_count_
                ? std::min(p.max_block_count_, prof->block_list.size())
                : prof->block_list.size();
    u64 i = prof->block_list_start;
    u64 k = 0;
    do {
      if (k++ > m)
        break;
      if (i < prof->block_list.size())
        co_yield std::make_tuple(thread_id, prof->block_list[i]);
      i = (i + 1) % m;
    } while (i != prof->block_list_start);
  }
}

std::string Profiler::report() {
  auto &p = instance();
  std::string s = "Profiler Report:\n";
  s += std::format("Block Type List [{}]\n", p.block_descriptors_.size());
  for (const auto *desc : p.block_descriptors_)
    s += std::format("  {}\n", desc->name);

  std::thread::id current_thread;

  for (const auto &[thread_id, block] : Profiler::iterateBlocks()) {
    if (thread_id != current_thread) {
      s += std::format("THREAD {}:\n", thread_id);
      current_thread = thread_id;
    }
    s += std::format(
        "{}{} - {} : {} - {}\n", std::string((block.level + 1) * 2, ' '),
        timeLabel(std::chrono::duration(
            block.wall_start_ - SystemTime::initTime(thread_id).wall_time)),
        p.block_descriptors_[block.descriptor_id]->name,
        timeLabel(block.wallDuration()), timeLabel(block.cpuDuration()));
  }

  return s;
}

std::string Profiler::trace() {
  auto &p = instance();
  std::string s = "Profiler Trace:\n";

  std::thread::id current_thread;

  for (const auto &[thread_id, block] : Profiler::iterateBlocks()) {
    s += colors::console::threadColor(thread_id);
    if (thread_id != current_thread) {
      s += std::format("THREAD {}:\n", thread_id);
      current_thread = thread_id;
    }
    s += std::format("  {}", timeLabel(std::chrono::duration(
                                 block.wall_start_ -
                                 SystemTime::initTime(thread_id).wall_time)));
    s += colors::console::reset;
    s += std::format("{}{: <20}{} : {} - {}\n",
                     // function color
                     colors::console::color(block.descriptor_id * 10),
                     std::string((block.level + 1) * 2, ' ') +
                         p.block_descriptors_[block.descriptor_id]->name,
                     colors::console::reset,
                     timeDurationLabel(block.wallDuration()),
                     timeDurationLabel(block.cpuDuration()));
  }

  return s;
}

Profiler::Profiler() noexcept = default;

const Profiler::BlockDescriptor &
Profiler::blockDescriptor(const Profiler::Block &block) {
  return *instance().block_descriptors_[block.descriptor_id];
}

bool Profiler::isEnabled() { return instance().enabled; }

void Profiler::enable() { instance().enabled = true; }

void Profiler::disable() {
  auto &p = instance();
  p.enabled = false;
  HERMES_NOT_IMPLEMENTED;
  // while (!p.block_stack_.empty())
  //   endBlock();
}

void Profiler::setMaxBlockCount(size_t max_block_count) {
  instance().max_block_count_ = max_block_count;
}

void Profiler::reset() {
  auto &p = instance();
  HERMES_NOT_IMPLEMENTED;
  // p.block_list_start_ = 0;
  // p.block_stack_ = std::stack<u32>();
}

Profiler::Block::Block(u32 desc_id) : descriptor_id(desc_id) {}

void Profiler::Block::start() {
  wall_start_ = SystemTime::WallClock::now();
  cpu_start_ = SystemTime::CPUClock::now();
}

void Profiler::Block::end() {
  wall_end_ = SystemTime::WallClock::now();
  cpu_end_ = SystemTime::CPUClock::now();
}

Profiler::ScopedBlock::ScopedBlock(u32 desc_id) {
  block_.descriptor_id = desc_id;
  startBlock(block_);
}

Profiler::ScopedBlock::~ScopedBlock() { endBlock(); }

} // namespace hermes::profile
