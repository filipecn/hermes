/// Copyright (c) 2019, FilipeCN.
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
///\file defs.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2019-01-04
///
///\brief Data type definitions
///
///\ingroup common
///\addtogroup common
/// @{

#ifndef HERMES_COMMON_DEFS_H
#define HERMES_COMMON_DEFS_H

// *********************************************************************************************************************
//                                                                                                       CUDA SUPPORT
// *********************************************************************************************************************
#if defined(ENABLE_CUDA)

#include <cuda_runtime.h>

/// \brief Specifies that the function can only be called from host side
#define HERMES_HOST_FUNCTION __host__
/// \brief Specifies that the function can be called from both host and device sides
#define HERMES_DEVICE_CALLABLE __device__ __host__
/// \brief Specifies that the function can only be called from device side
#define HERMES_DEVICE_FUNCTION __device__
/// \brief Specifies that hermes is compiled with CUDA support
#define HERMES_DEVICE_ENABLED __CUDA_ARCH__
/// \brief Defines a CUDA kernel function
/// \param NAME kernel name
/// \note the kernel's name receives a suffix _k
#define HERMES_CUDA_KERNEL(NAME) __global__ void NAME ## _k
/// \brief Wraps a block of code intended to be compiled only when using CUDA
#define HERMES_CUDA_CODE(CODE) {CODE}

#else

#define HERMES_HOST_FUNCTION
#define HERMES_DEVICE_CALLABLE
#define HERMES_DEVICE_FUNCTION
#define HERMES_CUDA_CODE(CODE)

#endif

#include <cstdint>
#include <type_traits>
#include <string>
// *********************************************************************************************************************
//                                                                                                         DATA TYPES
// *********************************************************************************************************************
#ifdef HERMES_USE_DOUBLE_AS_DEFAULT
using real_t = double;                       //!< default floating point type
#else
using real_t = float;                        //!< default floating point type
#endif

using f32 = float;                           //!< 32 bit size floating point type
using f64 = double;                          //!< 64 bit size floating point type

using i8 = int8_t;                           //!<  8 bit size integer type
using i16 = int16_t;                         //!< 16 bit size integer type
using i32 = int32_t;                         //!< 32 bit size integer type
using i64 = int64_t;                         //!< 64 bit size integer type

using u8 = uint8_t;                          //!<  8 bit size unsigned integer type
using u16 = uint16_t;                        //!< 16 bit size unsigned integer type
using u32 = uint32_t;                        //!< 32 bit size unsigned integer type
using u64 = uint64_t;                        //!< 64 bit size unsigned integer type

using ulong = unsigned long;                 //!< unsigned long type
using uint = unsigned int;                   //!< unsigned int type
using ushort = unsigned short;               //!< unsigned short type
using uchar = unsigned char;                 //!< unsigned char type

using byte = uint8_t;                        //!< unsigned byte

namespace hermes {

/// \brief Enum class for integral types
enum class DataType : u8 {
  I8 = 0,         //!< i8 type identifier
  I16 = 1,        //!< i16 type identifier
  I32 = 2,        //!< i32 type identifier
  I64 = 3,        //!< i64 type identifier
  U8 = 4,         //!< u8 type identifier
  U16 = 5,        //!< u16 type identifier
  U32 = 6,        //!< u32 type identifier
  U64 = 7,        //!< u64 type identifier
  F16 = 8,        //!< f16 type identifier
  F32 = 9,        //!< f32 type identifier
  F64 = 10,       //!< f64 type identifier
  CUSTOM = 11     //!< unidentified type
};

/// \brief DataType set of auxiliary functions
class DataTypes {
public:
  /// \brief Translates DataType from identifier number
  /// \param index
  /// \return
  HERMES_DEVICE_CALLABLE static DataType typeFrom(u8 index) {
#define MATCH_TYPE(Type) \
  if((u8)DataType::Type == index) \
    return DataType::Type;
    MATCH_TYPE(I8)
    MATCH_TYPE(I16)
    MATCH_TYPE(I32)
    MATCH_TYPE(I64)
    MATCH_TYPE(U8)
    MATCH_TYPE(U16)
    MATCH_TYPE(U32)
    MATCH_TYPE(U64)
    MATCH_TYPE(F32)
    MATCH_TYPE(F64)
    return DataType::CUSTOM;
#undef MATCH_TYPE
  }
  /// \brief Translates template type T to DataType
  /// \tparam T
  /// \return
  template<typename T>
  HERMES_DEVICE_CALLABLE static DataType typeFrom() {
#define MATCH_TYPE(Type, R) \
  if(std::is_same_v<T, Type>) \
    return DataType::R;
    MATCH_TYPE(i8, I8)
    MATCH_TYPE(i16, I16)
    MATCH_TYPE(i32, I32)
    MATCH_TYPE(i64, I64)
    MATCH_TYPE(u8, U8)
    MATCH_TYPE(u16, U16)
    MATCH_TYPE(u32, U32)
    MATCH_TYPE(u64, U64)
    MATCH_TYPE(f32, F32)
    MATCH_TYPE(f64, F64)
    return DataType::CUSTOM;
#undef MATCH_TYPE
  }
  /// \brief Computes number of bytes from DataType
  /// \param type
  /// \return
  static u32 typeSize(DataType type) {
#define TYPE_SIZE(Size, Type) \
        if(DataType::Type == type) \
        return Size;
    TYPE_SIZE(sizeof(i8), I8)
    TYPE_SIZE(sizeof(i16), I16)
    TYPE_SIZE(sizeof(i32), I32)
    TYPE_SIZE(sizeof(i64), I64)
    TYPE_SIZE(sizeof(u8), U8)
    TYPE_SIZE(sizeof(u16), U16)
    TYPE_SIZE(sizeof(u32), U32)
    TYPE_SIZE(sizeof(u64), U64)
    TYPE_SIZE(sizeof(f32), F32)
    TYPE_SIZE(sizeof(f64), F64)
    return 0;
#undef TYPE_SIZE
  }
  /// \brief Gets DataType string name
  /// \param type
  /// \return
  static std::string typeName(DataType type) {
#define DATA_TYPE_NAME(Type) \
      if(DataType::Type == type) \
    return #Type;
    DATA_TYPE_NAME(I8)
    DATA_TYPE_NAME(I16)
    DATA_TYPE_NAME(I32)
    DATA_TYPE_NAME(I64)
    DATA_TYPE_NAME(U8)
    DATA_TYPE_NAME(U16)
    DATA_TYPE_NAME(U32)
    DATA_TYPE_NAME(U64)
    DATA_TYPE_NAME(F16)
    DATA_TYPE_NAME(F32)
    DATA_TYPE_NAME(F64)
    DATA_TYPE_NAME(CUSTOM)
    return "CUSTOM";
#undef DATA_TYPE_NAME
  }
};
/// \brief Specifies where memory is stored
enum class MemoryLocation {
  DEVICE,     //!< GPU side
  HOST,       //!< CPU side
  UNIFIED     //!< unified memory
};
/// \brief Gets MemoryLocation value string name
/// \param location
/// \return
inline std::string memoryLocationName(MemoryLocation location) {
#define ENUM_NAME(E)                  \
    if(MemoryLocation::E == location) \
      return #E;
  ENUM_NAME(DEVICE)
  ENUM_NAME(HOST)
  ENUM_NAME(UNIFIED)
  return "CUSTOM";
#undef ENUM_NAME
}

// *********************************************************************************************************************
//                                                                                                                 IO
// *********************************************************************************************************************
/// \brief MemoryLocation support for `std::ostream` << operator
/// \param o
/// \param location
/// \return
inline std::ostream &operator<<(std::ostream &o, MemoryLocation location) {
  o << memoryLocationName(location);
  return o;
}

} // namespace hermes

#endif

/// @}
