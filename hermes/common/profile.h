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
///\note hermes::profiler was based on Sergey Yagovtsev's Easy Profiler source
///      code:
///                   https://github.com/yse/easy_profiler
///
///\ingroup common
///\addtogroup common
/// @{

#pragma once

#include <hermes/colors/argb_colors.h>
#include <hermes/core/types.h>
#include <hermes/system/time.h>

#include <chrono>
#include <generator>
#include <shared_mutex>
#include <stack>
#include <tuple>
#include <vector>

namespace hermes::profile {

// *****************************************************************************
//                                                                   Profiler
// *****************************************************************************
/// \brief Singleton code profiler
///
/// This profiler works by registering a sequence labeled blocks (time
/// intervals). The labeled blocks can represent blocks of code lines, functions
/// bodies and sections. Blocks can also reside inside other blocks constituting
/// hierarchies - useful to find out which section of a function is slower for
/// example.
///
/// Ideally, the class should be used indirectly by the auxiliary MACROS. Here
/// is an example of different types of blocks being used:
/// \code{.cpp}
///     void profiled_function() {
///         // register a block taking the function's name as the label
///         // the block is automatically finished after leaving this function
///         HERMES_PROFILE_FUNCTION()
///         // some code
///         {
///             // register a block with the label "code scope"
///             // the block is automatically finished after leaving this
///             function HERMES_PROFILE_SCOPE("code scope")
///         }
///         // you can also initiate and finish a block manually
///         HERMES_PROFILE_START_BLOCK("my block")
///         // some code
///         HERMES_PROFILE_END_BLOCK
///         // remember to finish blocks consistently, as the profiler uses a
///         simple stack
///         // to manage block creation/completion
///     }
/// \endcode
/// - In case memory is a limitation or for any other reason, you may also limit
/// the maximum number of blocks being
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
    explicit BlockDescriptor(const char *name, u32 color);
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
    [[nodiscard]] auto wallDuration() const {
      return std::chrono::duration(wall_end_ - wall_start_);
    }
    [[nodiscard]] auto cpuDuration() const {
      return std::chrono::duration(cpu_end_ - cpu_start_);
    }

    u32 descriptor_id{0}; //!< block descriptor identifier
    u32 level{0};         //!< profile stack level
  private:
    void start();
    void end();

    SystemTime::WallSample wall_start_, wall_end_;
    SystemTime::CPUSample cpu_start_, cpu_end_;
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
  // ***************************************************************************
  //                                                           STATIC METHODS
  // ***************************************************************************
  //                                                                      access
  /// \brief Get block descriptor from block
  /// \param block
  /// \return
  static const BlockDescriptor &blockDescriptor(const Block &block);
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
  static u32 pushBlockDescriptor(const char *name,
                                 u32 color = argb_colors::Default);
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
  static std::generator<std::tuple<std::thread::id, const Block &>>
  iterateBlocks();
  //                                                                      output
  /// \brief Dumps profiling into a string
  static std::string trace();
  static std::string report();
  // ***************************************************************************
  //                                                             CONSTRUCTORS
  // ***************************************************************************
  ~Profiler() = default;
  //                                                                  assignment
  Profiler(Profiler &&other) = delete;
  Profiler(const Profiler &other) = delete;
  // ***************************************************************************
  //                                                                OPERATORS
  // ***************************************************************************
  Profiler &operator=(const Profiler &other) = delete;
  Profiler &operator=(Profiler &&other) = delete;

private:
  Profiler() noexcept;
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
  std::vector<BlockDescriptor *> block_descriptors_{};
  std::shared_mutex block_descriptors_mutex_;

  struct Profile {
    using Ptr = std::shared_ptr<Profile>;
    u64 block_list_start{0};
    std::vector<Block> block_list;
    std::stack<u32> block_stack;
  };
  std::unordered_map<std::thread::id, Profile::Ptr> thread_profiles_;
  std::shared_mutex thread_profiles_mutex_;
};

#define HERMES_PROFILE_ENABLED

#ifdef HERMES_PROFILE_ENABLED

/// \brief Auxiliary function to pick variadic color argument
/// \tparam TArgs
/// \param ...
/// \return
template <class... TArgs> inline constexpr u32 extract_color(TArgs...);
/// \brief Auxiliary function to pick variadic color argument
/// \return
template <> inline constexpr u32 extract_color<>() {
  return hermes::argb_colors::Default;
}
/// \brief Auxiliary function to pick variadic color argument
/// \tparam T
/// \return
template <class T> inline constexpr u32 extract_color(T) {
  return hermes::argb_colors::Default;
}
/// \brief Auxiliary function to pick variadic color argument
/// \param _color
/// \return
template <> inline constexpr u32 extract_color<u32>(u32 _color) {
  return _color;
}
/// \brief Auxiliary function to pick variadic color argument
/// \tparam TArgs
/// \param _color
/// \param ...
/// \return
template <class... TArgs>
inline constexpr u32 extract_color(u32 _color, TArgs...) {
  return _color;
}
/// \brief Auxiliary function to pick variadic color argument
/// \tparam T
/// \tparam TArgs
/// \param _args
/// \return
template <class T, class... TArgs>
inline constexpr u32 extract_color(T, TArgs... _args) {
  return extract_color(_args...);
}

} // namespace hermes::profiler

/// \brief Joins two tokens
/// \param x
/// \param y
#define HERMES_TOKEN_JOIN(x, y) x##y

/// \brief Concatenates two tokens
/// \param x
/// \param y
#define HERMES_TOKEN_CONCATENATE(x, y) HERMES_TOKEN_JOIN(x, y)

/// \brief Starts a new non-scoped block with a given label
/// \param name - block label name
/// \param ... - block descriptor options
#define HERMES_PROFILE_START_BLOCK(name, ...)                                  \
  static u32 HERMES_TOKEN_CONCATENATE(hermes_block_desc_id_, __LINE__) =       \
      hermes::profile::Profiler::pushBlockDescriptor(                          \
          name, hermes::profile::extract_color(__VA_ARGS__));                  \
  hermes::profile::Profiler::Block HERMES_TOKEN_CONCATENATE(block, __LINE__)(  \
      HERMES_TOKEN_CONCATENATE(hermes_block_desc_id_, __LINE__));              \
  hermes::profile::Profiler::startBlock(                                       \
      HERMES_TOKEN_CONCATENATE(block, __LINE__))

/// \brief Finishes the current top block
#define HERMES_PROFILE_END_BLOCK hermes::profile::Profiler::endBlock();

/// \brief Starts a scoped block with a given label
/// \param name - block label name
/// \param ... - block descriptor options
#define HERMES_PROFILE_SCOPE(name, ...)                                        \
  static u32 HERMES_TOKEN_CONCATENATE(hermes_block_desc_id_, __LINE__) =       \
      hermes::profile::Profiler::pushBlockDescriptor(                          \
          name, hermes::profile::extract_color(__VA_ARGS__));                  \
  hermes::profile::Profiler::ScopedBlock HERMES_TOKEN_CONCATENATE(block,       \
                                                                  __LINE__)(   \
      HERMES_TOKEN_CONCATENATE(hermes_block_desc_id_, __LINE__))

/// \brief Starts a scoped block using the enclosing function as label
/// \param ... - block descriptor options
#define HERMES_PROFILE_FUNCTION(...)                                           \
  HERMES_PROFILE_SCOPE(__func__, ##__VA_ARGS__)

/// \brief Enables profiler
#define HERMES_ENABLE_PROFILER hermes::profile::Profiler::enable();

/// \brief Disables profiler
#define HERMES_DISABLE_PROFILER hermes::profile::Profiler::disable();

/// \brief Clears profiler history and current stack
#define HERMES_RESET_PROFILER hermes::profile::Profiler::reset();

#endif

/// @}
