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
///\file profiler.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2022-01-04
///
///\brief
///
///\note hermes::profiler was based on Sergey Yagovtsev's Easy Profiler source code
///      https://github.com/yse/easy_profiler

#ifndef HERMES_HERMES_COMMON_PROFILER_H
#define HERMES_HERMES_COMMON_PROFILER_H

#include <hermes/common/defs.h>
#include <hermes/logging/argb_colors.h>
#include <functional>
#include <chrono>
#include <vector>
#include <stack>

namespace hermes::profiler {

// *********************************************************************************************************************
//                                                                                                           Profiler
// *********************************************************************************************************************
class Profiler {
public:
  struct BlockDescriptor {
    BlockDescriptor(u32 id, u32 color, u32 line, const char *name);
    explicit BlockDescriptor(const char *name,
                             u32 color);
    u32 color{};
    u32 id{};
    u32 line{};
    const char *name{}; //!< block name
  };

  class Block {
    friend Profiler;
  public:
    Block() = default;
    explicit Block(u32 desc_id);
    [[nodiscard]] inline u64 begin() const noexcept { return start_; }
    [[nodiscard]] inline u64 end() const noexcept { return end_; }
    [[nodiscard]] inline u64 duration() const noexcept { return end_ - start_; }

    u32 descriptor_id{0};       //< block descriptor identifier
    u32 level{0};               //< profile stack level
  private:
    void start();
    void finish();

    u64 start_{0};              //< start time / tick
    u64 end_{0};                //< end time / tick
  };

  class ScopedBlock {
  public:
    explicit ScopedBlock(u32 block_descriptor_id);
    ~ScopedBlock();

  private:
    Block block_;
  };
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  //                                                                                                           access
  static const std::vector<Block> &blockList();
  static const BlockDescriptor &blockDescriptor(const Block &block);
  //
  static u64 initTime();
  static i64 cpuFrequency();
  static bool isEnabled();
  static void enable();
  static void disable();
  static u32 pushBlockDescriptor(const char *name, u32 color = argb_colors::Default);
  static void startBlock(Block &block);
  static void endBlock();
  static void setMaxBlockCount(size_t max_block_count);
  static void reset();
  static void iterateBlocks(const std::function<void(const Block &)> &f);
  //                                                                                                           output
  static std::string dump();
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  ~Profiler() = default;
  //                                                                                                       assignment
  Profiler(Profiler &&other) = delete;
  Profiler(const Profiler &other) = delete;
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  Profiler &operator=(const Profiler &other) = delete;
  Profiler &operator=(Profiler &&other) = delete;

private:
  Profiler();
  ///
  /// \return
  static Profiler &instance() {
    static Profiler _instance;
    return _instance;
  }

  u64 profiler_start_time_{};
  i64 cpu_frequency_{};
  bool enabled{true};
  size_t max_block_count_{0};
  u64 block_list_start_{0};
  ///
  std::vector<BlockDescriptor *> block_descriptors_{};
  std::vector<Block> block_list_;
  std::stack<u32> block_stack_;
};

static inline u64 now() {
  // high res option
  return static_cast<u64>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  // steady option
  return static_cast<u64>(std::chrono::steady_clock::now().time_since_epoch().count());
  //
#if (defined(__GNUC__) || defined(__ICC))
  // part of code from google/benchmark library (Licensed under the Apache License, Version 2.0)
  // see https://github.com/google/benchmark/blob/master/src/cycleclock.h#L111
#if defined(__i386__)
  int64_t ret;
      __asm__ volatile("rdtsc" : "=A"(ret));
      return ret;
#elif defined(__x86_64__) || defined(__amd64__)
  uint64_t low, high;
  __asm__ volatile("rdtsc" : "=a"(low), "=d"(high));
  return (high << 32) | low;
#endif
#endif
}

static inline u64 ns2ticks(u64 ns) {
  return static_cast<u64>(ns * Profiler::cpuFrequency() / 1000000000LL);
}

static inline u64 us2ticks(u64 us) {
  return static_cast<u64>(us * Profiler::cpuFrequency() / 1000000LL);
}

static inline u64 ms2ticks(u64 ms) {
  return static_cast<u64>(ms * Profiler::cpuFrequency() / 1000LL);
}

static inline u64 s2ticks(u64 s) {
  return static_cast<u64>(s * Profiler::cpuFrequency());
}

static inline u64 ticks2ns(u64 ticks) {
  return static_cast<u64>(ticks * 1000000000LL / Profiler::cpuFrequency());
  // no chrono support TODO
  return static_cast<u64>(ticks / Profiler::cpuFrequency());
}

static inline u64 ticks2us(u64 ticks) {
  return static_cast<u64>(ticks * 1000000LL / Profiler::cpuFrequency());
  // no chrono support TODO
  return static_cast<u64>(ticks * 1000 / Profiler::cpuFrequency());
}

static inline u64 ticks2ms(u64 ticks) {
  return static_cast<u64>(ticks * 1000LL / Profiler::cpuFrequency());
  // no chrono support TODO
  return static_cast<u64>(ticks * 1000000LL / Profiler::cpuFrequency());
}

static inline u64 ticks2s(u64 ticks) {
  return static_cast<u64>(ticks / Profiler::cpuFrequency());
  // no chrono support TODO
  return static_cast<u64>(ticks * 1000000000LL / Profiler::cpuFrequency());
}

static inline i64 computeCPUFrequency() {
  return std::chrono::high_resolution_clock::period::den / std::chrono::high_resolution_clock::period::num;
}
#define HERMES_PROFILE_ENABLED

#ifdef HERMES_PROFILE_ENABLED

template<class ... TArgs>
inline constexpr u32 extract_color(TArgs...);

template<>
inline constexpr u32 extract_color<>() {
  return hermes::argb_colors::Default;
}

template<class T>
inline constexpr u32 extract_color(T) {
  return hermes::argb_colors::Default;
}

template<>
inline constexpr u32 extract_color<u32>(u32 _color) {
  return _color;
}

template<class ... TArgs>
inline constexpr u32 extract_color(u32 _color, TArgs...) {
  return _color;
}

template<class T, class ... TArgs>
inline constexpr u32 extract_color(T, TArgs... _args) {
  return extract_color(_args...);
}

}

#define HERMES_TOKEN_JOIN(x, y) x ## y
#define HERMES_TOKEN_CONCATENATE(x, y) HERMES_TOKEN_JOIN(x, y)

#define HERMES_PROFILE_START_BLOCK(name, ...) \
 static u32 HERMES_TOKEN_CONCATENATE(hermes_block_desc_id_, __LINE__) = \
 hermes::profiler::Profiler::pushBlockDescriptor(name, hermes::profiler::extract_color(__VA_ARGS__)); \
 hermes::profiler::Profiler::Block HERMES_TOKEN_CONCATENATE(block, __LINE__)\
  (HERMES_TOKEN_CONCATENATE(hermes_block_desc_id_, __LINE__));          \
  hermes::profiler::Profiler::startBlock(HERMES_TOKEN_CONCATENATE(block, __LINE__));

#define HERMES_PROFILE_END_BLOCK hermes::profiler::Profiler::endBlock();

#define HERMES_PROFILE_SCOPE(name, ...) \
static u32 HERMES_TOKEN_CONCATENATE(hermes_block_desc_id_, __LINE__) = \
 hermes::profiler::Profiler::pushBlockDescriptor(name, hermes::profiler::extract_color(__VA_ARGS__)); \
 hermes::profiler::Profiler::ScopedBlock HERMES_TOKEN_CONCATENATE(block, __LINE__)\
  (HERMES_TOKEN_CONCATENATE(hermes_block_desc_id_, __LINE__));

#define HERMES_PROFILE_FUNCTION(...) HERMES_PROFILE_SCOPE(__func__, ## __VA_ARGS__)

#define HERMES_ENABLE_PROFILER hermes::profiler::Profiler::enable();

#define HERMES_DISABLE_PROFILER hermes::profiler::Profiler::disable();

#define HERMES_RESET_PROFILER hermes::profiler::Profiler::reset();

#endif

#endif //HERMES_HERMES_COMMON_PROFILER_H
