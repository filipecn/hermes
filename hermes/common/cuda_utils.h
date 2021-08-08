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
#include <hermes/common/debug.h>
#include <iostream>

#ifdef HERMES_DEVICE_ENABLED

namespace hermes::cuda_utils {

#define GPU_BLOCK_SIZE 1024
#define GPU_BLOCK_SIZE_X 1024
#define GPU_BLOCK_SIZE_Y 1024
#define GPU_BLOCK_SIZE_Z 64
#define GPU_WARP_SIZE 32

/// \note - Each block cannot have more than 512/1024 threads in total
/// (Compute Capability 1.x or 2.x and later respectively)
/// \note - The maximum dimensions of each block are limited to [512,512,64]/[1024,1024,64]
/// (Compute 1.x/2.x or later)
/// \note - Each block cannot consume more than 8k/16k/32k/64k/32k/64k/32k/64k/32k/64k registers total
/// (Compute 1.0,1.1/1.2,1.3/2.x-/3.0/3.2/3.5-5.2/5.3/6-6.1/6.2/7.0)
/// \note - Each block cannot consume more than 16kb/48kb/96kb of shared memory
/// (Compute 1.x/2.x-6.2/7.0)
struct LaunchInfo {
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  /// Tries to achieve good occupancy by recomputing block and grid sizes
  /// \param max_b maximum number of threads per block
  /// \param n total number of threads
  /// \param b output block size
  /// \param g output grid size
  static void distribute(u32 max_b, u32 n, u32 &b, u32 &g) {
    if (n <= max_b) {
      b = n;
      g = 1;
    } else {
      // round n to a multiple of warp size
      auto m = (n % GPU_WARP_SIZE) ? ((n + GPU_WARP_SIZE) / GPU_WARP_SIZE) * GPU_WARP_SIZE : n;
      auto b_candidate = max_b;
      auto min_candidate = b_candidate;
      auto min_r = m % b_candidate;
      while (b_candidate > 128) {
        auto r = m % b_candidate;
        if (r < min_r) {
          min_candidate = b_candidate;
          min_r = r;
        }
        b_candidate >>= 1;
      }
      b = min_candidate;
      g = (m % b) ? (m + b) / b : m / b;
    }
  }
  /// Redistribute threads to fit the gpu block size limits
  /// \param b
  /// \param g
  /// \param new_b
  /// \param new_g
  static void redistribute(dim3 b, dim3 g, dim3 &new_b, dim3 &new_g) {
    dim3 m(b.x * g.x, b.y * g.y, b.z * g.z);
    dim3 b_candidate = b;
    while (b_candidate.x * b_candidate.y * b_candidate.z > GPU_BLOCK_SIZE) {
      // split max dimension
      if (b_candidate.x > b_candidate.y && b_candidate.x > b_candidate.z)
        b_candidate.x >>= 1;
      else if (b_candidate.y >= b_candidate.x && b_candidate.y >= b_candidate.z)
        b_candidate.y >>= 1;
      else
        b_candidate.z >>= 1;
    }
    new_b = b_candidate;
    new_g = dim3((m.x % new_b.x) ? (m.x + new_b.x) / new_b.x : m.x / new_b.x,
                 (m.y % new_b.y) ? (m.y + new_b.y) / new_b.y : m.y / new_b.y,
                 (m.z % new_b.z) ? (m.z + new_b.z) / new_b.z : m.z / new_b.z);
  }
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \param n thread count
  /// \param shared_memory_size_in_bytes (per block)
  /// \param stream stream id
  LaunchInfo(u32 n, size_t shared_memory_size_in_bytes = 0, cudaStream_t stream = {}) :
      shared_memory_size{shared_memory_size_in_bytes},
      stream_id{stream} {
    distribute(GPU_BLOCK_SIZE_X, n, block_size.x, grid_size.x);
    HERMES_CHECK_EXP(blockThreadCount() <= GPU_BLOCK_SIZE)
  }
  /// \param b block size (threads per block)
  /// \param s grid size (blocks)
  /// \param shared_memory_size_in_bytes per (per block)
  /// \param stream stream id
  LaunchInfo(size2 b, size2 s = {0, 0},
             size_t shared_memory_size_in_bytes = 0,
             cudaStream_t stream = {}) :
      shared_memory_size{shared_memory_size_in_bytes},
      stream_id{stream} {
    block_size = dim3(b.width, b.height, 1);
    grid_size = dim3(s.width, s.height, 1);
    if (s.total() == 0) {
      distribute(GPU_BLOCK_SIZE_X, b.width, block_size.x, grid_size.x);
      distribute(GPU_BLOCK_SIZE_Y, b.height, block_size.y, grid_size.y);
      redistribute(block_size, grid_size, block_size, grid_size);
    }
    HERMES_CHECK_EXP(blockThreadCount() <= GPU_BLOCK_SIZE)
  }
  /// \param b block size (threads per block)
  /// \param s grid size (blocks)
  /// \param shared_memory_size_in_bytes (per block)
  /// \param stream stream id
  LaunchInfo(size3 b, size3 s = {0, 0, 0},
             size_t shared_memory_size_in_bytes = 0,
             cudaStream_t stream = {}) :
      shared_memory_size{shared_memory_size_in_bytes},
      stream_id{stream} {
    block_size = dim3(b.width, b.height, 1);
    grid_size = dim3(s.width, s.height, 1);
    if (s.total() == 0) {
      distribute(GPU_BLOCK_SIZE_X, b.width, block_size.x, grid_size.x);
      distribute(GPU_BLOCK_SIZE_Y, b.height, block_size.y, grid_size.y);
      distribute(GPU_BLOCK_SIZE_Z, b.depth, block_size.z, grid_size.z);
      redistribute(block_size, grid_size, block_size, grid_size);
    }
    HERMES_CHECK_EXP(blockThreadCount() <= GPU_BLOCK_SIZE)
  }
  LaunchInfo(LaunchInfo &other) = delete;
  LaunchInfo(LaunchInfo &&other) = delete;
  LaunchInfo(const LaunchInfo &other) = delete;
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  [[nodiscard]] u32 threadCount() const {
    return grid_size.x * grid_size.y * grid_size.z * block_size.x * block_size.y * block_size.z;
  }
  [[nodiscard]] u32 blockThreadCount() const {
    return block_size.x * block_size.y * block_size.z;
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
//                                                                                                    SYNCHRONIZATION
// *********************************************************************************************************************

class Lock {
public:
  Lock();
  ~Lock();
  HERMES_DEVICE_FUNCTION void lock();
  HERMES_DEVICE_FUNCTION void unlock();
private:
  int *mutex{nullptr};
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

// *********************************************************************************************************************
//                                                                                                                 IO
// *********************************************************************************************************************
inline std::ostream &operator<<(std::ostream &o, const LaunchInfo &info) {
  o << "[block size (" << info.block_size.x << " " << info.block_size.y << " " << info.block_size.z << ") ";
  o << "grid size (" << info.grid_size.x << " " << info.grid_size.y << " " << info.grid_size.z << ")]";
  return o;
}

} // namespace hermes::cuda_utils

#define HERMES_CUDA_TIME(LAUNCH, ELAPSED_TIME_IN_MS)                                                                      \
{ cudaEvent_t cuda_event_start_t, cuda_event_stop_t;                                                                \
  cudaEventCreate(&cuda_event_start_t);                                                                             \
  cudaEventCreate(&cuda_event_stop_t);                                                                              \
  cudaEventRecord(cuda_event_start_t, 0);                                                                           \
  LAUNCH                                                                                                            \
  cudaEventRecord(cuda_event_stop_t, 0);                                                                            \
  cudaEventSynchronize(cuda_event_stop_t);                                                                          \
  cudaEventElapsedTime(&ELAPSED_TIME_IN_MS, cuda_event_start_t, cuda_event_stop_t); }

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

#define HERMES_CUDA_THREAD_INDEX_IJ                                                                                \
  hermes::index2 ij(threadIdx.x + blockIdx.x * blockDim.x,                                                          \
                    threadIdx.y + blockIdx.y * blockDim.y);

#define HERMES_CUDA_THREAD_INDEX_IJK                                                                               \
  hermes::index3 ijk(threadIdx.x + blockIdx.x * blockDim.x,                                                         \
                     threadIdx.y + blockIdx.y * blockDim.y,                                                         \
                     threadIdx.z + blockIdx.z * blockDim.z);

#define HERMES_CUDA_RETURN_IF_NOT_THREAD_0                                                                          \
{ HERMES_CUDA_THREAD_INDEX_IJK                                                                                     \
  if(ijk != hermes::index3(0,0,0))                                                                                  \
    return;                                                                                                         \
}

#define HERMES_CUDA_THREAD_INDEX_LT(I, BOUNDS)                                                                      \
  u32 I = threadIdx.x + blockIdx.x * blockDim.x;                                                                    \
  if(I >= (BOUNDS)) return;

#define HERMES_CUDA_THREAD_INDEX2_LT(IJ, BOUNDS)                                                                    \
  hermes::index2 IJ(threadIdx.x + blockIdx.x * blockDim.x,                                                          \
                    threadIdx.y + blockIdx.y * blockDim.y);                                                         \
  if(IJ >= (BOUNDS)) return;

#define HERMES_CUDA_THREAD_INDEX3_LT(IJK, BOUNDS)                                                                   \
  hermes::index3 IJK(threadIdx.x + blockIdx.x * blockDim.x,                                                         \
                     threadIdx.y + blockIdx.y * blockDim.y,                                                         \
                     threadIdx.z + blockIdx.z * blockDim.z);                                                        \
  if(IJK >= (BOUNDS)) return;

#define HERMES_CUDA_THREAD_INDEX_I_LT(BOUNDS) HERMES_CUDA_THREAD_INDEX_LT(i, BOUNDS)
#define HERMES_CUDA_THREAD_INDEX_IJ_LT(BOUNDS) HERMES_CUDA_THREAD_INDEX2_LT(ij, BOUNDS)
#define HERMES_CUDA_THREAD_INDEX_IJK_LT(BOUNDS) HERMES_CUDA_THREAD_INDEX3_LT(ijk, BOUNDS)

// *********************************************************************************************************************
//                                                                                                              ERROR
// *********************************************************************************************************************
#define HERMES_CHECK_CUDA(err)                                                                                      \
  {                                                                                                                 \
      auto hermes_cuda_result = (err);                                                                              \
      if(hermes_cuda_result != cudaSuccess) {                                                                       \
        HERMES_LOG_CRITICAL(cudaGetErrorString(hermes_cuda_result))                                                 \
        cudaDeviceReset();                                                                                          \
        exit(99);                                                                                                   \
        }                                                                                                           \
  }

#define HERMES_CHECK_LAST_CUDA HERMES_CHECK_CUDA(cudaGetLastError())

inline void hermes_print_cuda_devices() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Major compute capability: %d\n", prop.major);
    printf("  Minor compute capability: %d\n", prop.minor);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("  Device can map host memory with "
           "cudaHostAlloc/cudaHostGetDevicePointer: %d\n",
           prop.canMapHostMemory);
    printf("  Clock frequency in kilohertz: %d\n", prop.clockRate);
    printf("  Compute mode (See cudaComputeMode): %d\n", prop.computeMode);
    printf("  Device can concurrently copy memory and execute a kernel: %d\n",
           prop.deviceOverlap);
    printf("  Device is integrated as opposed to discrete: %d\n",
           prop.integrated);
    printf("  Specified whether there is a run time limit on kernels: %d\n",
           prop.kernelExecTimeoutEnabled);
    printf("  Maximum size of each dimension of a grid: %d %d %d\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Maximum size of each dimension of a block: %d %d %d\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Maximum number of threads per block: %d\n",
           prop.maxThreadsPerBlock);
    printf("  Maximum pitch in bytes allowed by memory copies: %zu\n",
           prop.memPitch);
    printf("  Number of multiprocessors on device: %d\n",
           prop.multiProcessorCount);
    printf("  32-bit registers available per block: %d\n", prop.regsPerBlock);
    printf("  Shared memory available per block in bytes: %zu\n",
           prop.sharedMemPerBlock);
    printf("  Alignment requirement for textures: %zu\n", prop.textureAlignment);
    printf("  Constant memory available on device in bytes: %zu\n",
           prop.totalConstMem);
    printf("  Global memory available on device in bytes: %zu\n",
           prop.totalGlobalMem);
    printf("  Warp size in threads: %d\n", prop.warpSize);
  }
}

inline void hermes_print_cuda_memory_usage() {
  size_t free_byte;
  size_t total_byte;
  HERMES_CHECK_CUDA(cudaMemGetInfo(&free_byte, &total_byte));
  auto free_db = (double) free_byte;
  auto total_db = (double) total_byte;
  double used_db = total_db - free_db;
  printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
         used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0,
         total_db / 1024.0 / 1024.0);
}

#define CUDA_MEMORY_USAGE                                                                                           \
  {                                                                                                                 \
    std::cerr << "[INFO][" << __FILE__ << "][" << __LINE__ << "]";                                                  \
    hermes_print_cuda_memory_usage();                                                                                      \
  }
#else

#define HERMES_CHECK_CUDA(err)
#define CUDA_MEMORY_USAGE

#endif

#endif // HERMES_COMMON_CUDA_UTILS_H
