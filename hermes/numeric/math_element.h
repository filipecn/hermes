/// Copyright (c) 2020, FilipeCN.
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
///\file math_element.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-17-09
///
///\brief Base class for all geometric objects
///
///\ingroup numeric
///\addtogroup numeric
/// @{

#ifndef HERMES_GEOMETRY_MATH_ELEMENT_H
#define HERMES_GEOMETRY_MATH_ELEMENT_H

#include <hermes/common/defs.h>
#include <hermes/logging/memory_dump.h>

namespace hermes {

/// \brief Interface used by all basic geometric entities
/// \tparam NUMERIC_TYPE
/// \tparam COMPONENT_COUNT
template<typename NUMERIC_TYPE, u64 COMPONENT_COUNT>
class MathElement {
public:
  /// \brief Underlying data type
  static NUMERIC_TYPE numeric_data;
  /// \brief Gets the number of dimensional components
  /// \return
  static inline constexpr u64 componentCount() { return COMPONENT_COUNT; };
  /// \brief Gets the size in bytes of underlying data type
  /// \return
  static inline constexpr u64 numericTypeSizeInBytes() { return sizeof(NUMERIC_TYPE); };
  /// \brief Gets memory layout
  /// \return
  [[nodiscard]] static MemoryDumper::RegionLayout memoryDumpLayout() {
    return {
        .offset = 0,
        .field_size_in_bytes = sizeof(NUMERIC_TYPE),
        .count = COMPONENT_COUNT,
        .color = ConsoleColors::default_color,
        .sub_regions = {},
        .type = DataTypes::typeFrom<NUMERIC_TYPE>()
    };
  }
};

}

#endif //HERMES_GEOMETRY_MATH_ELEMENT_H

/// @}
