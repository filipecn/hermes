/* Copyright (c) 2021, FilipeCN.
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

/// \file   memory.cpp
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2021-09-22

#include <hermes/storage/memory.h>
#include <hermes/system/gpu.h>

#include <tuple>

namespace hermes::mem
{

  u32 sizes::cache_l1_size = 64;

  h_size alignment::alignedSize(h_size size, h_size alignment)
  {
    return (size + alignment - 1) & ~(alignment - 1);
  }

  h_size alignment::alignTo(std::size_t number_of_bytes, std::size_t align)
  {
    return number_of_bytes > 0 ? (1u + (number_of_bytes - 1u) / align) * align
                               : 0;
  }

  std::size_t alignment::leftAlignShift(uintptr_t address, std::size_t align)
  {
    const std::size_t mask = align - 1;
    HERMES_ASSERT((align & mask) == 0);
    return address - (address & ~mask);
  }

  std::size_t alignment::rightAlignShift(uintptr_t address, std::size_t align)
  {
    const std::size_t mask = align - 1;
    HERMES_ASSERT((align & mask) == 0);
    return ((address + mask) & ~mask) - address;
  }

  uintptr_t alignment::alignAddress(uintptr_t address, std::size_t align)
  {
    const std::size_t mask = align - 1;
    HERMES_ASSERT((align & mask) == 0);
    return (address + mask) & ~mask;
  }

  void *allocation::allocAligned(size_t size, size_t align)
  {
    // allocate align more bytes to store shift value
    size_t actual_bytes = size + align;
    // allocate unaligned block
    u8 *p_raw_mem = new u8[actual_bytes];
    // align block
    u8 *p_aligned_mem = alignment::alignPointer(p_raw_mem, align);
    // if no alignment occurred, shift it up the full 'align' bytes to make room
    // to store the shift
    if (p_aligned_mem == p_raw_mem)
      p_aligned_mem += align;
    // determine the shift and store it
    ptrdiff_t shift = p_aligned_mem - p_raw_mem;
    // alignment can't be greater than 256
    HERMES_ASSERT(shift > 0 && shift <= 256)
    p_aligned_mem[-1] = static_cast<u8>(shift & 0xFF);
    return p_aligned_mem;
  }

  void allocation::freeAligned(void *p_mem)
  {
    if (p_mem)
    {
      u8 *p_aligned_mem = reinterpret_cast<u8 *>(p_mem);
      // extract the shift
      ptrdiff_t shift = p_aligned_mem[-1];
      if (shift == 0)
        shift = 256;
      // back up to the actual allocated address and array-delete it
      u8 *p_raw_mem = p_aligned_mem - shift;
      delete[] p_raw_mem;
    }
  }

  HeError allocation::freeMemory(void *data, MemoryLocation location)
  {
    if (!data)
      return HeError::None;
    switch (location)
    {
    case MemoryLocation::HOST:
      std::free(data);
      break;
    case MemoryLocation::DEVICE:
#ifdef HERMES_DEVICE_ENABLED
      HERMES_CHECK_CUDA_CALL(cudaFree(data))
#endif
      break;
    case MemoryLocation::UNIFIED:
#ifdef HERMES_DEVICE_ENABLED
      HERMES_CHECK_CUDA_CALL(cudaFree(data))
#endif
      break;
    }
    return HeError::None;
  }

  Result<void *> allocation::allocate(h_size byte_count,
                                      MemoryLocation location)
  {
    void *data = nullptr;
    switch (location)
    {
    case MemoryLocation::HOST:
      data = std::malloc(byte_count);
      break;
    case MemoryLocation::DEVICE:
#ifdef HERMES_DEVICE_ENABLED
      HERMES_CHECK_CUDA_CALL(cudaMalloc(&data, byte_count));
#endif
      break;
    case MemoryLocation::UNIFIED:
#ifdef HERMES_DEVICE_ENABLED
      HERMES_CHECK_CUDA_CALL(cudaMallocManaged(&data, byte_count))
#endif
      break;
    }
    if (!data)
      return HeError::BadAllocation;
    return Result<void *>(data);
  }

  Result<std::tuple<void *, h_size>>
  allocation::allocate(const size2 &size, MemoryLocation location)
  {
    void *data = nullptr;
    h_size pitch = 0;
    switch (location)
    {
    case MemoryLocation::HOST:
      data = std::malloc(size.total());
      pitch = size.width;
      break;
    case MemoryLocation::DEVICE:
#ifdef HERMES_DEVICE_ENABLED
      HERMES_CHECK_CUDA_CALL(
          cudaMallocPitch(&data, &pitch, size.width, size.height));
#endif
      break;
    case MemoryLocation::UNIFIED:
#ifdef HERMES_DEVICE_ENABLED
      HERMES_CHECK_CUDA_CALL(cudaMallocManaged(&data, size.total()))
      pitch = size.width;
#endif
      break;
    }
    return std::tuple<void *, h_size>(data, pitch);
  }

  Result<std::tuple<void *, h_size>>
  allocation::allocate(const size3 &size, MemoryLocation location)
  {
    void *data = nullptr;
    h_size pitch = 0;
    switch (location)
    {
    case MemoryLocation::HOST:
      data = std::malloc(size.total());
      pitch = size.width;
      break;
    case MemoryLocation::DEVICE:
#ifdef HERMES_DEVICE_ENABLED
    {
      cudaPitchedPtr pdata{};
      cudaExtent extent = make_cudaExtent(size.width, size.height, size.depth);
      HERMES_CHECK_CUDA_CALL(cudaMalloc3D(&pdata, extent));
      pitch = pdata.pitch;
    }
#endif
    break;
    case MemoryLocation::UNIFIED:
#ifdef HERMES_DEVICE_ENABLED
    {
      HERMES_CHECK_CUDA_CALL(cudaMallocManaged(&data, size.total()))
      pitch = size.width;
    }
#endif
    break;
    }
    return std::tuple<void *, h_size>(data, pitch);
  }

  HeError writes::copyDevice2Device(void *dst, void *src, h_size byte_count)
  {
    HERMES_UNUSED_VARIABLE(dst);
    HERMES_UNUSED_VARIABLE(src);
    HERMES_UNUSED_VARIABLE(byte_count);
#ifdef HERMES_DEVICE_ENABLED
    HERMES_CHECK_CUDA_CALL(
        cudaMemcpy(dst, src, byte_count, cudaMemcpyDeviceToDevice));
#endif
    return HeError::None;
  }

  HeError writes::copyDevice2Host(void *dst, void *src, h_size byte_count)
  {
    HERMES_UNUSED_VARIABLE(dst);
    HERMES_UNUSED_VARIABLE(src);
    HERMES_UNUSED_VARIABLE(byte_count);
#ifdef HERMES_DEVICE_ENABLED
    HERMES_CHECK_CUDA_CALL(
        cudaMemcpy(dst, src, byte_count, cudaMemcpyDeviceToHost));
#endif
    return HeError::None;
  }

  HeError writes::copyHost2Device(void *dst, void *src, h_size byte_count)
  {
#ifdef HERMES_DEVICE_ENABLED
    HERMES_CHECK_CUDA_CALL(
        cudaMemcpy(dst, src, byte_count, cudaMemcpyHostToDevice));
#else
    HERMES_UNUSED_VARIABLE(dst);
    HERMES_UNUSED_VARIABLE(src);
    HERMES_UNUSED_VARIABLE(byte_count);
#endif
    return HeError::None;
  }

  HeError writes::copyHost2Host(void *dst, void *src, h_size byte_count)
  {
    std::memcpy(dst, src, byte_count);
    return HeError::None;
  }

  HeError writes::copy(MemoryLocation dst_location, void *dst,
                       MemoryLocation src_location, void *src,
                       h_size byte_count)
  {

    switch (src_location)
    {
    case MemoryLocation::HOST:
      switch (dst_location)
      {
      case MemoryLocation::DEVICE:
        return copyHost2Device(dst, src, byte_count);
        break;
      default:
        return copyHost2Host(dst, src, byte_count);
      }
      break;
    case MemoryLocation::DEVICE:
      switch (dst_location)
      {
      case MemoryLocation::HOST:
        return copyDevice2Host(dst, src, byte_count);
        break;
      case MemoryLocation::DEVICE:
        return copyDevice2Device(dst, src, byte_count);
        break;
      case MemoryLocation::UNIFIED:
        return HeError::NotImplemented;
      }
      break;
    case MemoryLocation::UNIFIED:
      switch (dst_location)
      {
      case MemoryLocation::DEVICE:
        return copyHost2Device(dst, src, byte_count);
        break;
      default:
        return copyHost2Host(dst, src, byte_count);
      }
    }
    return HeError::None;
  }

  HeError writes::copyHost2Device(void *dst, h_size dst_pitch, void *src,
                                  h_size src_pitch, const size2 &src_size)
  {
    HERMES_UNUSED_VARIABLE(dst);
    HERMES_UNUSED_VARIABLE(dst_pitch);
    HERMES_UNUSED_VARIABLE(src);
    HERMES_UNUSED_VARIABLE(src_pitch);
    HERMES_UNUSED_VARIABLE(src_size);
#ifdef HERMES_DEVICE_ENABLED
    HERMES_CHECK_CUDA_CALL(cudaMemcpy2D(dst, dst_pitch, src, src_pitch,
                                        src_size.width, src_size.height,
                                        cudaMemcpyHostToDevice));
#endif
    return HeError::None;
  }

  HeError writes::copyDevice2Host(void *dst, h_size dst_pitch, void *src,
                                  h_size src_pitch, const size2 &src_size)
  {
    HERMES_UNUSED_VARIABLE(dst);
    HERMES_UNUSED_VARIABLE(dst_pitch);
    HERMES_UNUSED_VARIABLE(src);
    HERMES_UNUSED_VARIABLE(src_pitch);
    HERMES_UNUSED_VARIABLE(src_size);
#ifdef HERMES_DEVICE_ENABLED
    HERMES_CHECK_CUDA_CALL(cudaMemcpy2D(dst, dst_pitch, src, src_pitch,
                                        src_size.width, src_size.height,
                                        cudaMemcpyHostToDevice));
#endif
    return HeError::None;
  }

  HeError writes::copyDevice2Device(void *dst, h_size dst_pitch, void *src,
                                    h_size src_pitch, const size2 &src_size)
  {
    HERMES_UNUSED_VARIABLE(dst);
    HERMES_UNUSED_VARIABLE(dst_pitch);
    HERMES_UNUSED_VARIABLE(src);
    HERMES_UNUSED_VARIABLE(src_pitch);
    HERMES_UNUSED_VARIABLE(src_size);
#ifdef HERMES_DEVICE_ENABLED
    HERMES_CHECK_CUDA_CALL(cudaMemcpy2D(dst, dst_pitch, src, src_pitch,
                                        src_size.width, src_size.height,
                                        cudaMemcpyDeviceToDevice));
#endif
    return HeError::None;
  }

  HeError writes::copy(MemoryLocation dst_location, void *dst, h_size dst_pitch,
                       MemoryLocation src_location, void *src, h_size src_pitch,
                       const size2 &src_size)
  {
    if (src_size.height == 1)
      return copy(dst_location, dst, src_location, src, src_pitch);
    switch (src_location)
    {
    case MemoryLocation::HOST:
      switch (dst_location)
      {
      case MemoryLocation::DEVICE:
        return copyHost2Device(dst, dst_pitch, src, src_pitch, src_size);
      default:
        return copyHost2Host(dst, src, src_pitch * src_size.height);
      }
      break;
    case MemoryLocation::DEVICE:
      switch (dst_location)
      {
      case MemoryLocation::HOST:
        return copyDevice2Host(dst, dst_pitch, src, src_pitch, src_size);
      case MemoryLocation::DEVICE:
        return copyDevice2Device(dst, dst_pitch, src, src_pitch, src_size);
      case MemoryLocation::UNIFIED:
        return HeError::NotImplemented;
      }
      break;
    case MemoryLocation::UNIFIED:
      switch (dst_location)
      {
      case MemoryLocation::DEVICE:
        return copyHost2Device(dst, dst_pitch, src, src_pitch, src_size);
      default:
        return copyHost2Host(dst, src, src_pitch * src_size.height);
      }
    }
    return HeError::None;
  }

  HeError writes::copyHost2Device(void *dst, h_size dst_pitch,
                                  const size3 &dst_size, void *src,
                                  h_size src_pitch, const size3 &src_size)
  {
    HERMES_UNUSED_VARIABLE(dst);
    HERMES_UNUSED_VARIABLE(dst_pitch);
    HERMES_UNUSED_VARIABLE(dst_size);
    HERMES_UNUSED_VARIABLE(src);
    HERMES_UNUSED_VARIABLE(src_pitch);
    HERMES_UNUSED_VARIABLE(src_size);
#ifdef HERMES_DEVICE_ENABLED
    // 3d pitched memory
    cudaMemcpy3DParms p = {};
    p.srcPtr.ptr = src;
    p.srcPtr.pitch = src_pitch;
    p.srcPtr.xsize = src_size.width;
    p.srcPtr.ysize = src_size.height;
    p.dstPtr.ptr = dst;
    p.dstPtr.pitch = dst_pitch;
    p.dstPtr.xsize = dst_size.width;
    p.dstPtr.ysize = dst_size.height;
    p.extent.width = src_size.width;
    p.extent.height = src_size.height;
    p.extent.depth = src_size.depth;
    p.kind = cudaMemcpyHostToDevice;
    HERMES_CHECK_CUDA_CALL(cudaMemcpy3D(&p));
#endif
    return HeError::None;
  }

  HeError writes::copyDevice2Host(void *dst, h_size dst_pitch,
                                  const size3 &dst_size, void *src,
                                  h_size src_pitch, const size3 &src_size)
  {
    HERMES_UNUSED_VARIABLE(dst);
    HERMES_UNUSED_VARIABLE(dst_pitch);
    HERMES_UNUSED_VARIABLE(dst_size);
    HERMES_UNUSED_VARIABLE(src);
    HERMES_UNUSED_VARIABLE(src_pitch);
    HERMES_UNUSED_VARIABLE(src_size);
#ifdef HERMES_DEVICE_ENABLED
    // 3d pitched memory
    cudaMemcpy3DParms p = {};
    p.srcPtr.ptr = src;
    p.srcPtr.pitch = src_pitch;
    p.srcPtr.xsize = src_size.width;
    p.srcPtr.ysize = src_size.height;
    p.dstPtr.ptr = dst;
    p.dstPtr.pitch = dst_pitch;
    p.dstPtr.xsize = dst_size.width;
    p.dstPtr.ysize = dst_size.height;
    p.extent.width = src_size.width;
    p.extent.height = src_size.height;
    p.extent.depth = src_size.depth;
    p.kind = cudaMemcpyDeviceToHost;
    HERMES_CHECK_CUDA_CALL(cudaMemcpy3D(&p));
#endif
    return HeError::None;
  }

  HeError writes::copyDevice2Device(void *dst, h_size dst_pitch,
                                    const size3 &dst_size, void *src,
                                    h_size src_pitch, const size3 &src_size)
  {
    HERMES_UNUSED_VARIABLE(dst);
    HERMES_UNUSED_VARIABLE(dst_pitch);
    HERMES_UNUSED_VARIABLE(dst_size);
    HERMES_UNUSED_VARIABLE(src);
    HERMES_UNUSED_VARIABLE(src_pitch);
    HERMES_UNUSED_VARIABLE(src_size);
#ifdef HERMES_DEVICE_ENABLED
    // 3d pitched memory
    cudaMemcpy3DParms p = {};
    p.srcPtr.ptr = src;
    p.srcPtr.pitch = src_pitch;
    p.srcPtr.xsize = src_size.width;
    p.srcPtr.ysize = src_size.height;
    p.dstPtr.ptr = dst;
    p.dstPtr.pitch = dst_pitch;
    p.dstPtr.xsize = dst_size.width;
    p.dstPtr.ysize = dst_size.height;
    p.extent.width = src_size.width;
    p.extent.height = src_size.height;
    p.extent.depth = src_size.depth;
    p.kind = cudaMemcpyDeviceToDevice;
    HERMES_CHECK_CUDA_CALL(cudaMemcpy3D(&p));
#endif
    return HeError::None;
  }

  HeError writes::copy(MemoryLocation dst_location, void *dst, h_size dst_pitch,
                       const size3 &dst_size, MemoryLocation src_location,
                       void *src, h_size src_pitch, const size3 &src_size)
  {
    if (src_size.depth == 1 && src_size.height == 1)
      return copy(dst_location, dst, src_location, src, src_pitch);
    if (src_size.depth == 1)
      return copy(dst_location, dst, dst_pitch, src_location, src, src_pitch,
                  src_size.slice(0, 1));
    switch (src_location)
    {
    case MemoryLocation::HOST:
      switch (dst_location)
      {
      case MemoryLocation::DEVICE:
        return copyHost2Device(dst, dst_pitch, dst_size, src, src_pitch,
                               src_size);
      default:
        return copyHost2Host(dst, src,
                             src_pitch * src_size.height * src_size.depth);
      }
      break;
    case MemoryLocation::DEVICE:
      switch (dst_location)
      {
      case MemoryLocation::HOST:
        return copyDevice2Host(dst, dst_pitch, dst_size, src, src_pitch,
                               src_size);
      case MemoryLocation::DEVICE:
        return copyDevice2Device(dst, dst_pitch, dst_size, src, src_pitch,
                                 src_size);
      case MemoryLocation::UNIFIED:
        return HeError::NotImplemented;
      }
      break;
    case MemoryLocation::UNIFIED:
      switch (dst_location)
      {
      case MemoryLocation::DEVICE:
        return copyHost2Device(dst, dst_pitch, dst_size, src, src_pitch,
                               src_size);
      default:
        return copyHost2Host(dst, src, src_pitch * src_size.height);
      }
    }
    return HeError::None;
  }

} // namespace hermes::mem
