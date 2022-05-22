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
///\brief Code profiler
///
///\note hermes::profiler was based on Sergey Yagovtsev's Easy Profiler source code
///      https://github.com/yse/easy_profiler
///
///\ingroup common
///\addtogroup common
/// @{

#ifndef HERMES_HERMES_COMMON_PROFILER_H
#define HERMES_HERMES_COMMON_PROFILER_H

#include <hermes/common/defs.h>
#include <hermes/colors/argb_colors.h>
#include <functional>
#include <chrono>
#include <vector>
#include <stack>

namespace hermes::profiler {

// *********************************************************************************************************************
//                                                                                                           Profiler
// *********************************************************************************************************************
/// \brief Singleton code profiler
///
/// This profiler works by registering a sequence labeled blocks (time intervals). The labeled blocks can represent
/// blocks of code lines, functions bodies and sections. Blocks can also reside inside other blocks constituting
/// hierarchies - useful to find out which section of a function is slower for example.
///
/// Ideally, the class should be used indirectly by the auxiliary MACROS. Here is an example of different types of
/// blocks being used:
/// \code{.cpp}
///     void profiled_function() {
///         // register a block taking the function's name as the label
///         // the block is automatically finished after leaving this function
///         HERMES_PROFILE_FUNCTION()
///         // some code
///         {
///             // register a block with the label "code scope"
///             // the block is automatically finished after leaving this function
///             HERMES_PROFILE_SCOPE("code scope")
///         }
///         // you can also initiate and finish a block manually
///         HERMES_PROFILE_START_BLOCK("my block")
///         // some code
///         HERMES_PROFILE_END_BLOCK
///         // remember to finish blocks consistently, as the profiler uses a simple stack
///         // to manage block creation/completion
///     }
/// \endcode
/// - In case memory is a limitation or for any other reason, you may also limit the maximum number of blocks being
///   stored at any time
/// \code{.cpp}
///     // only keep the last 100 blocks
///     hermes::profiler::Profiler::setMaxBlockCount(100);
/// \endcode
class Profiler {
public:
  /// \brief Describes a block label
  struct BlockDescriptor {
    /// \param id
    /// \param color
    /// \param line
    /// \param name
    BlockDescriptor(u32 id, u32 color, u32 line, const char *name);
    /// \param name
    /// \param color
    explicit BlockDescriptor(const char *name,
                             u32 color);
    u32 color{};        //!< block color
    u32 id{};           //!< block unique id
    u32 line{};         //!< code line
    const char *name{}; //!< block name
  };

  /// \brief Holds a labeled profiler block with start/end time points
  class Block {
    friend Profiler;
  public:
    ///
    Block() = default;
    /// \brief Value constructor
    /// \param desc_id block descriptor id (label)
    explicit Block(u32 desc_id);
    /// \brief block start time point (in ticks)
    /// \return
    [[nodiscard]] inline u64 begin() const noexcept { return start_; }
    /// \brief block end time point (in ticks)
    /// \return
    [[nodiscard]] inline u64 end() const noexcept { return end_; }
    /// \brief block duration (in ticks)
    /// \return
    [[nodiscard]] inline u64 duration() const noexcept { return end_ - start_; }

    u32 descriptor_id{0};       //!< block descriptor identifier
    u32 level{0};               //!< profile stack level
  private:
    void start();
    void finish();

    u64 start_{0};              //!< start time / tick
    u64 end_{0};                //!< end time / tick
  };

  /// \brief RAII Profiler Block
  class ScopedBlock {
  public:
    /// \param block_descriptor_id
    explicit ScopedBlock(u32 block_descriptor_id);
    ~ScopedBlock();

  private:
    Block block_;
  };
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  //                                                                                                           access
  /// \brief Raw block data
  /// \return
  static const std::vector<Block> &blockList();
  /// \brief Get block descriptor from block
  /// \param block
  /// \return
  static const BlockDescriptor &blockDescriptor(const Block &block);
  //
  /// \brief Get time point when Profiler was instantiated
  /// \return time in CPU ticks
  static u64 initTime();
  /// \brief Computed CPU frequency
  /// \return frequency in Hz
  static i64 cpuFrequency();
  /// \brief Checks if Profiler is currently enabled
  /// \return
  static bool isEnabled();
  /// \brief Enables Profiler
  static void enable();
  /// \brief Disables Profiler
  static void disable();
  /// \brief Registers a new block description
  /// \param name
  /// \param color
  /// \return
  static u32 pushBlockDescriptor(const char *name, u32 color = argb_colors::Default);
  /// \brief Starts a new block by taking this call time point
  /// \note The block is put on top of the stack
  /// \param block
  static void startBlock(Block &block);
  /// \brief Finishes the block at the top of the stack
  /// \note The top of the stack is popped
  static void endBlock();
  /// \brief Sets a limit into the maximum number of stored blocks
  /// \param max_block_count
  static void setMaxBlockCount(size_t max_block_count);
  /// \brief Clears all blocks
  static void reset();
  /// \brief Iterates over stored blocks sequentially
  /// \param f callback function for each block
  static void iterateBlocks(const std::function<void(const Block &)> &f);
  //                                                                                                           output
  /// \brief Dumps profiling into a string
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

/// \brief Gets current time point
/// \return time point (in ticks)
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
/// \brief Converts nanoseconds into ticks
/// \param ns
/// \return
static inline u64 ns2ticks(u64 ns) {
  return static_cast<u64>(ns * Profiler::cpuFrequency() / 1000000000LL);
}
/// \brief Converts microseconds into ticks
/// \param us
/// \return
static inline u64 us2ticks(u64 us) {
  return static_cast<u64>(us * Profiler::cpuFrequency() / 1000000LL);
}
/// \brief Converts milliseconds into ticks
/// \param ms
/// \return
static inline u64 ms2ticks(u64 ms) {
  return static_cast<u64>(ms * Profiler::cpuFrequency() / 1000LL);
}
/// \brief Converts seconds into ticks
/// \param s
/// \return
static inline u64 s2ticks(u64 s) {
  return static_cast<u64>(s * Profiler::cpuFrequency());
}
/// Converts ticks into nanoseconds
/// \param ticks
/// \return
static inline u64 ticks2ns(u64 ticks) {
  return static_cast<u64>(ticks * 1000000000LL / Profiler::cpuFrequency());
  // no chrono support TODO
  return static_cast<u64>(ticks / Profiler::cpuFrequency());
}
/// Converts ticks into microseconds
/// \param ticks
/// \return
static inline u64 ticks2us(u64 ticks) {
  return static_cast<u64>(ticks * 1000000LL / Profiler::cpuFrequency());
  // no chrono support TODO
  return static_cast<u64>(ticks * 1000 / Profiler::cpuFrequency());
}
/// Converts ticks into milliseconds
/// \param ticks
/// \return
static inline u64 ticks2ms(u64 ticks) {
  return static_cast<u64>(ticks * 1000LL / Profiler::cpuFrequency());
  // no chrono support TODO
  return static_cast<u64>(ticks * 1000000LL / Profiler::cpuFrequency());
}
/// Converts ticks into seconds
/// \param ticks
/// \return
static inline u64 ticks2s(u64 ticks) {
  return static_cast<u64>(ticks / Profiler::cpuFrequency());
  // no chrono support TODO
  return static_cast<u64>(ticks * 1000000000LL / Profiler::cpuFrequency());
}
/// \brief Computes CPU clock frequency
/// \return
static inline i64 computeCPUFrequency() {
  return std::chrono::high_resolution_clock::period::den / std::chrono::high_resolution_clock::period::num;
}
#define HERMES_PROFILE_ENABLED

#ifdef HERMES_PROFILE_ENABLED

/// \brief Auxiliary function to pick variadic color argument
/// \tparam TArgs
/// \param ...
/// \return
template<class ... TArgs>
inline constexpr u32 extract_color(TArgs...);
/// \brief Auxiliary function to pick variadic color argument
/// \return
template<>
inline constexpr u32 extract_color<>() {
  return hermes::argb_colors::Default;
}
/// \brief Auxiliary function to pick variadic color argument
/// \tparam T
/// \return
template<class T>
inline constexpr u32 extract_color(T) {
  return hermes::argb_colors::Default;
}
/// \brief Auxiliary function to pick variadic color argument
/// \param _color
/// \return
template<>
inline constexpr u32 extract_color<u32>(u32 _color) {
  return _color;
}
/// \brief Auxiliary function to pick variadic color argument
/// \tparam TArgs
/// \param _color
/// \param ...
/// \return
template<class ... TArgs>
inline constexpr u32 extract_color(u32 _color, TArgs...) {
  return _color;
}
/// \brief Auxiliary function to pick variadic color argument
/// \tparam T
/// \tparam TArgs
/// \param _args
/// \return
template<class T, class ... TArgs>
inline constexpr u32 extract_color(T, TArgs... _args) {
  return extract_color(_args...);
}

}

/// \brief Joins two tokens
/// \param x
/// \param y
#define HERMES_TOKEN_JOIN(x, y) x ## y

/// \brief Concatenates two tokens
/// \param x
/// \param y
#define HERMES_TOKEN_CONCATENATE(x, y) HERMES_TOKEN_JOIN(x, y)

/// \brief Starts a new non-scoped block with a given label
/// \param name - block label name
/// \param ... - block descriptor options
#define HERMES_PROFILE_START_BLOCK(name, ...) \
 static u32 HERMES_TOKEN_CONCATENATE(hermes_block_desc_id_, __LINE__) = \
 hermes::profiler::Profiler::pushBlockDescriptor(name, hermes::profiler::extract_color(__VA_ARGS__)); \
 hermes::profiler::Profiler::Block HERMES_TOKEN_CONCATENATE(block, __LINE__)\
  (HERMES_TOKEN_CONCATENATE(hermes_block_desc_id_, __LINE__));          \
  hermes::profiler::Profiler::startBlock(HERMES_TOKEN_CONCATENATE(block, __LINE__))

/// \brief Finishes the current top block
#define HERMES_PROFILE_END_BLOCK hermes::profiler::Profiler::endBlock();

/// \brief Starts a scoped block with a given label
/// \param name - block label name
/// \param ... - block descriptor options
#define HERMES_PROFILE_SCOPE(name, ...) \
static u32 HERMES_TOKEN_CONCATENATE(hermes_block_desc_id_, __LINE__) = \
 hermes::profiler::Profiler::pushBlockDescriptor(name, hermes::profiler::extract_color(__VA_ARGS__)); \
 hermes::profiler::Profiler::ScopedBlock HERMES_TOKEN_CONCATENATE(block, __LINE__)\
  (HERMES_TOKEN_CONCATENATE(hermes_block_desc_id_, __LINE__))

/// \brief Starts a scoped block using the enclosing function as label
/// \param ... - block descriptor options
#define HERMES_PROFILE_FUNCTION(...) HERMES_PROFILE_SCOPE(__func__, ## __VA_ARGS__)

/// \brief Enables profiler
#define HERMES_ENABLE_PROFILER hermes::profiler::Profiler::enable();

/// \brief Disables profiler
#define HERMES_DISABLE_PROFILER hermes::profiler::Profiler::disable();

/// \brief Clears profiler history and current stack
#define HERMES_RESET_PROFILER hermes::profiler::Profiler::reset();

#endif

#endif //HERMES_HERMES_COMMON_PROFILER_H

/// @}
