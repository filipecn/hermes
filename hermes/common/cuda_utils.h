/// Copyright (c) 2021, FilipeCN.
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
///\file cuda_utils.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-06-30
///
///\brief

#ifndef HERMES_COMMON_CUDA_UTILS_H
#define HERMES_COMMON_CUDA_UTILS_H

#include <hermes/common/size.h>
#include <hermes/common/index.h>

namespace hermes::cuda_utils {

#ifdef ENABLE_CUDA

#define GPU_BLOCK_SIZE 1024

struct LaunchInfo {
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \param s grid size (blocks)
  /// \param b block size (threads per block)
  /// \param shared_memory_size_in_bytes (per block)
  /// \param stream stream id
  LaunchInfo(u32 n, size_t shared_memory_size_in_bytes = 0, cudaStream_t stream = {}) :
      shared_memory_size{shared_memory_size_in_bytes},
      stream_id{stream} {
    block_size.x = std::min((u32) GPU_BLOCK_SIZE, n);
    grid_size.x = n / block_size.x + 1;
  }
  /// \param s grid size (blocks)
  /// \param b block size (threads per block)
  /// \param shared_memory_size_in_bytes per (per block)
  /// \param stream stream id
  LaunchInfo(size2 s, size2 b = {16, 16},
             size_t shared_memory_size_in_bytes = 0,
             cudaStream_t stream = {}) :
      shared_memory_size{shared_memory_size_in_bytes},
      stream_id{stream} {
    block_size = dim3(b.width, b.height);
    grid_size = dim3((s.width + block_size.x - 1) / block_size.x,
                     (s.height + block_size.y - 1) / block_size.y);
  }
  /// \param s grid size (blocks)
  /// \param b block size (threads per block)
  /// \param shared_memory_size_in_bytes (per block)
  /// \param stream stream id
  LaunchInfo(size3 s, size3 b = {16, 16, 16},
             size_t shared_memory_size_in_bytes = 0,
             cudaStream_t stream = {}) :
      shared_memory_size{shared_memory_size_in_bytes},
      stream_id{stream} {
    block_size = dim3(b.width, b.height, b.depth);
    grid_size = dim3((s.width + block_size.x - 1) / block_size.x,
                     (s.height + block_size.y - 1) / block_size.y,
                     (s.depth + block_size.z - 1) / block_size.z);
  }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  dim3 grid_size;
  dim3 block_size;
  size_t shared_memory_size{0};
  cudaStream_t stream_id{};
};

// *********************************************************************************************************************
//                                                                                                             MEMORY
// *********************************************************************************************************************
/// \param src
/// \param dst
/// \return
inline cudaMemcpyKind copyDirection(MemoryLocation src, MemoryLocation dst) {
  if (src == MemoryLocation::DEVICE && dst == MemoryLocation::DEVICE)
    return cudaMemcpyDeviceToDevice;
  if (src == MemoryLocation::DEVICE && dst == MemoryLocation::HOST)
    return cudaMemcpyDeviceToHost;
  if (src == MemoryLocation::HOST && dst == MemoryLocation::HOST)
    return cudaMemcpyHostToHost;
  return cudaMemcpyHostToDevice;
}

#endif

} // namespace hermes::cuda_utils

#define HERMES_CUDA_DEVICE_SYNCHRONIZE HERMES_CHECK_CUDA(cudaDeviceSynchronize())

#define HERMES_CUDA_LAUNCH(LAUNCH_INFO, NAME, ...)                                                                  \
{                                                                                                                   \
  auto _hli_ = hermes::cuda_utils::LaunchInfo LAUNCH_INFO;                                                          \
  NAME<<< _hli_.grid_size, _hli_.block_size, _hli_.shared_memory_size, _hli_.stream_id >>> (__VA_ARGS__);           \
  HERMES_CHECK_LAST_CUDA                                                                                            \
}

#define HERMES_CUDA_LAUNCH_AND_SYNC(LAUNCH_INFO, NAME, ...)                                                         \
{                                                                                                                   \
  auto _hli_ = hermes::cuda_utils::LaunchInfo LAUNCH_INFO;                                                          \
  NAME<<< _hli_.grid_size, _hli_.block_size, _hli_.shared_memory_size, _hli_.stream_id >>> (__VA_ARGS__);           \
  HERMES_CHECK_LAST_CUDA                                                                                            \
  HERMES_CUDA_DEVICE_SYNCHRONIZE                                                                                    \
}

#define HERMES_CUDA_THREAD_INDEX_I                                                                                  \
  u32 i = threadIdx.x + blockIdx.x * blockDim.x;

#define HERMES_CUDA_THREAD_INDEX2_IJ                                                                                \
  hermes::index2 ij(threadIdx.x + blockIdx.x * blockDim.x,                                                          \
                    threadIdx.y + blockIdx.y * blockDim.y);

#define HERMES_CUDA_THREAD_INDEX3_IJK                                                                               \
  hermes::index3 ijk(threadIdx.x + blockIdx.x * blockDim.x,                                                         \
                     threadIdx.y + blockIdx.y * blockDim.y,                                                         \
                     threadIdx.z + blockIdx.z * blockDim.z);

#define HERMES_CUDA_RETURN_IF_NOT_THREAD_0                                                                          \
{ HERMES_CUDA_THREAD_INDEX3_IJK                                                                                     \
  if(ijk != hermes::index3(0,0,0))                                                                                  \
    return;                                                                                                         \
}

#endif // HERMES_COMMON_CUDA_UTILS_H
