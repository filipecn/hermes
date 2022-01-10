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
///\file profiler.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2022-01-05
///
///\brief

#include <hermes/common/profiler.h>
#include <hermes/common/debug.h>
#include <hermes/common/str.h>
#include <hermes/common/profiler.h>

namespace hermes::profiler {

Profiler::BlockDescriptor::BlockDescriptor(u32 id, u32 color, u32 line, const char *name) :
    id(id),
    color(color),
    line(line),
    name(name) {}

Profiler::BlockDescriptor::BlockDescriptor(const char *name,
                                           u32 color) :
    color(color), name(name) {}

u32 Profiler::pushBlockDescriptor(const char *name, u32 color) {
  auto desc = new BlockDescriptor(name, color);
  auto &p = instance();
  p.block_descriptors_.emplace_back(desc);
  return p.block_descriptors_.size() - 1;
}

void Profiler::startBlock(Profiler::Block &block) {
  auto &p = instance();
  if (!p.enabled)
    return;
  block.start();
  block.level = p.block_stack_.size();
  // here we allocate a new block index
  if (p.max_block_count_ && p.block_list_.size() >= p.max_block_count_) {
    p.block_stack_.push(p.block_list_start_);
    // in the case the block being destroyed is not finished yet
    if (!p.block_list_[p.block_list_start_].end()) {
      HERMES_NOT_IMPLEMENTED
      exit(0);
    }
    p.block_list_[p.block_list_start_] = block;
    p.block_list_start_ = (p.block_list_start_ + 1) % p.max_block_count_;
  } else {
    p.block_stack_.push(p.block_list_.size());
    p.block_list_.emplace_back(block);
  }
}

void Profiler::endBlock() {
  auto &p = instance();
  if (!p.enabled)
    return;
  // TODO due to possible profiler pause the stack might be empty
  // TODO which is not exactly right?!
  if (!p.block_stack_.empty()) {
    HERMES_ASSERT(!p.block_stack_.empty())
    auto top = p.block_stack_.top();
    p.block_stack_.pop();
    HERMES_ASSERT(top >= 0 && top < p.block_list_.size())
    // if this is an invalid block, just ignore
    if (!p.block_list_[top].end())
      p.block_list_[top].finish();
  }
}

std::string Profiler::dump() {
  auto &p = instance();
  Str s;
#define PRINT_TIMES(TICKS) \
TICKS << " (" << ticks2us(TICKS) << " us " << ticks2ns(TICKS) << " ns)"
  s = s << "Block Type List [" << p.block_descriptors_.size() << "]\n";
  for (const auto *desc : p.block_descriptors_)
    s = s << "  " << desc->name << "\n";
  s = s << "Active blocks[" << p.block_stack_.size() << "]\n";
  s = s << "Finished blocks[" << p.block_list_.size() << "]\n";

  iterateBlocks([&](const Profiler::Block &block) {
    s = s << block.level << " - ";
    s = s << PRINT_TIMES(block.begin()) << " ~ ";
    s = s << PRINT_TIMES(block.end()) << " ";
    s = s << PRINT_TIMES(block.duration()) << " ";
    s = s << p.block_descriptors_[block.descriptor_id]->name << "\n";
  });

  return s.str();
#undef PRINT_TIMES
}

Profiler::Profiler() {
  cpu_frequency_ = computeCPUFrequency();
  profiler_start_time_ = hermes::profiler::now();
}

const std::vector<Profiler::Block> &Profiler::blockList() {
  return instance().block_list_;
}

const Profiler::BlockDescriptor &Profiler::blockDescriptor(const Profiler::Block &block) {
  return *instance().block_descriptors_[block.descriptor_id];
}

bool Profiler::isEnabled() {
  return instance().enabled;
}

void Profiler::enable() {
  instance().enabled = true;
}

void Profiler::disable() {
  auto &p = instance();
  p.enabled = false;
  while (!p.block_stack_.empty())
    endBlock();
}

u64 Profiler::initTime() {
  return instance().profiler_start_time_;
}

i64 Profiler::cpuFrequency() {
  return instance().cpu_frequency_;
}

void Profiler::setMaxBlockCount(size_t max_block_count) {
  instance().max_block_count_ = max_block_count;
}

void Profiler::reset() {
  auto &p = instance();
  p.block_list_start_ = 0;
  p.block_stack_ = std::stack<u32>();
}

void Profiler::iterateBlocks(const std::function<void(const Profiler::Block &)> &f) {
  auto &p = instance();
  u64 m = p.max_block_count_ ? std::min(p.max_block_count_, p.block_list_.size()) : p.block_list_.size();
  u64 i = p.block_list_start_;
  HERMES_ASSERT(i <= p.block_list_.size())
  HERMES_ASSERT(m <= p.block_list_.size())
  u64 k = 0;
  do {
    if (k++ > m)
      break;
    if (i >= 0 && i < p.block_list_.size())
      f(p.block_list_[i]);
    i = (i + 1) % m;
  } while (i != p.block_list_start_);
}

void Profiler::Block::start() {
  start_ = ::hermes::profiler::now();
}

void Profiler::Block::finish() {
  end_ = ::hermes::profiler::now();
}

Profiler::Block::Block(u32 desc_id) : descriptor_id(desc_id) {}

Profiler::ScopedBlock::ScopedBlock(u32 desc_id) {
  block_.descriptor_id = desc_id;
  startBlock(block_);
}

Profiler::ScopedBlock::~ScopedBlock() {
  endBlock();
}

}