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

///\file   memory_dump.h
///\author FilipeCN (filipedecn@gmail.com)
///\date   2021-03-14
///\brief  Classes to output memory footprints.

#pragma once

#include <hermes/base/str.h>
#include <hermes/core/debug.h>
#include <hermes/io/console_colors.h>
#include <hermes/math/math.h>

#include <cstdlib> // system
#include <iomanip>
#include <iostream>

namespace hermes {

/// \brief MemoryDumper output options
/// \note You may use bitwise operators to combine these options
enum class memory_dumper_option_bits {
  none = 0x0,               //!< default usage
  binary = 0x1,             //!< output memory contents in binary
  decimal = 0x2,            //!< output memory contents in decimal
  hexadecimal = 0x4,        //!< output memory contents in hexadecimal
  hexii = 0x8,              //!< output memory contents in hexii
  hide_header = 0x10,       //!< hide memory column names
  cache_align = 0x20,       //!< aligns starting address to cache size
  hide_zeros = 0x40,        //!< do not output bytes with 0 as value
  show_ascii = 0x80,        //!< show ascii characters for each memory byte
  save_to_string = 0x100,   //!< redirect output to string
  write_to_console = 0x200, //!< directly dump into stdout
  colored_output = 0x400,   //!< use terminal colors
  type_values = 0x800       //!< cast values and output their values properly
};

using memory_dumper_options = Flags<memory_dumper_option_bits>;

template <> struct FlagTraits<memory_dumper_option_bits> {
  static HERMES_CONST_OR_CONSTEXPR bool is_bitmask = true;
  static HERMES_CONST_OR_CONSTEXPR memory_dumper_options all_flags =
      memory_dumper_option_bits::binary | memory_dumper_option_bits::decimal |
      memory_dumper_option_bits::hexadecimal |
      memory_dumper_option_bits::hexii |
      memory_dumper_option_bits::hide_header |
      memory_dumper_option_bits::cache_align |
      memory_dumper_option_bits::hide_zeros |
      memory_dumper_option_bits::show_ascii |
      memory_dumper_option_bits::save_to_string |
      memory_dumper_option_bits::write_to_console |
      memory_dumper_option_bits::colored_output |
      memory_dumper_option_bits::type_values;
};

// *****************************************************************************
//                                                               MemoryDumper
// *****************************************************************************

/// \brief Auxiliary logging class for printing blocks of memory
class MemoryDumper {
public:
  /// \brief Memory region description
  ///
  /// A Layout describes how data is laid out in a specific portion of
  /// memory. The memory region can be composed of multiple sub-regions
  /// recursively. Each final sub-region will contain one or more elements of
  /// the same type. You can pick a different color for each sub-region and also
  /// repeat them over the layout (when you have arrays).
  ///
  /// - Example
  /// \code{cpp}
  /// // suppose you have an array of structs S
  /// struct S {
  ///     hermes::vec3 v;
  ///     hermes::point2 p;
  /// };
  /// S v[3];
  /// // a layout describing this array can be created like this:
  /// // start by setting the entire memory size (3 S structs)
  /// auto layout = MemoryDumper::Layout().withSizeOf<S>(3)
  ///    // the first subregion is the field v, which we paint blue
  ///    .withSubRegion(vec3::memoryDumpLayout().withColor(colors::console::blue))
  ///    // the second subregion is the field p, which we paint blue
  ///    .withSubRegion(point2::memoryDumpLayout().withColor(colors::console::yellow));
  /// \endcode
  /// \note Most types in hermes, such as `point2` and `vec3`, provide their
  /// Layout for you
  struct Layout {
    /// \brief Default constructor
    Layout() {}
    /// \brief Modifies layout offset
    /// \param offset_in_bytes
    /// \return
    Layout &withOffset(std::size_t offset_in_bytes) {
      offset = offset_in_bytes;
      return *this;
    }
    /// \brief Modifies layout color
    /// \param console_color
    /// \return
    Layout &withColor(const std::string &console_color) {
      color = console_color;
      return *this;
    }
    /// \brief Modifies layout count
    /// \param region_count
    /// \return
    Layout &withCount(std::size_t region_count) {
      count = region_count;
      return *this;
    }
    /// \brief Modifies layout base data type
    /// \param t
    /// \return
    Layout &withType(DataType t) {
      type = t;
      return *this;
    }
    /// \brief Appends a layout representing a sub-region of this layout
    /// \param sub_region
    /// \param increment_to_parent_size
    /// \return
    Layout &withSubRegion(const Layout &sub_region,
                          bool increment_to_parent_size = false) {
      std::size_t new_offset = 0;
      if (!sub_regions.empty())
        new_offset =
            sub_regions.back().offset +
            sub_regions.back().field_size_in_bytes * sub_regions.back().count;
      sub_regions.push_back(sub_region);
      sub_regions.back().offset = new_offset;
      if (increment_to_parent_size)
        field_size_in_bytes +=
            sub_region.field_size_in_bytes * sub_region.count;
      return *this;
    }
    /// \brief Appends a layout representing a sub-region of this layout
    /// \param sub_region
    /// \param increment_to_parent_size
    void pushSubRegion(const Layout &sub_region,
                       bool increment_to_parent_size = false) {
      std::size_t new_offset = 0;
      if (!sub_regions.empty())
        new_offset =
            sub_regions.back().offset +
            sub_regions.back().field_size_in_bytes * sub_regions.back().count;
      sub_regions.push_back(sub_region);
      sub_regions.back().offset = new_offset;
      if (increment_to_parent_size)
        field_size_in_bytes +=
            sub_region.field_size_in_bytes * sub_region.count;
    }
    /// \brief Modifies layout base data type based on a given type
    /// \tparam T
    /// \return
    template <typename T> Layout &withTypeFrom() {
      type = DataTypes::typeFrom<T>();
      return *this;
    }
    /// \brief Modifies layout size based on given type and count
    /// \tparam T
    /// \param element_count
    /// \return
    template <typename T> Layout &withSizeOf(std::size_t element_count = 1) {
      count = element_count;
      field_size_in_bytes = sizeof(T);
      return *this;
    }
    /// \brief Modifies layout size based on given quantities
    /// \param size_in_bytes
    /// \param element_count
    /// \return
    Layout &withSize(std::size_t size_in_bytes, std::size_t element_count = 1) {
      count = element_count;
      field_size_in_bytes = size_in_bytes;
      return *this;
    }
    /// \brief Gets layout size in bytes
    /// \return
    [[nodiscard]] std::size_t sizeInBytes() const {
      return field_size_in_bytes * count;
    }
    /// \brief Resizes number of sub-regions of this layout
    /// \param sub_regions_count
    void resizeSubRegions(size_t sub_regions_count) {
      sub_regions.resize(sub_regions_count);
      field_size_in_bytes = 0;
      for (auto &s : sub_regions)
        field_size_in_bytes += s.field_size_in_bytes * s.count;
    }
    /// \brief Removes all description
    void clear() { *this = Layout(); }

    std::size_t offset{0};              //!< Layout offset in bytes
    std::size_t field_size_in_bytes{0}; //!< Region size
    std::size_t count{1};               //!< Region count
    std::string color = colors::console::default_color; //!< Region color
    std::vector<Layout> sub_regions{}; //!< Sub-region descriptions
    DataType type{DataType::CUSTOM};   //!< Base data type
  };

  // ***************************************************************************
  //                                                           STATIC METHODS
  // ***************************************************************************

  /// \brief Dumps memory info about a given memory region
  /// \tparam T
  /// \param data
  /// \param size
  /// \return
  template <typename T>
  static std::string dumpInfo(const T *data, std::size_t size) {
    auto alignment = alignof(T);
    auto ptr = reinterpret_cast<const u8 *>(data);
    ptrdiff_t down_shift = reinterpret_cast<uintptr_t>(ptr) & (64 - 1);
    uintptr_t aligned_base_address =
        reinterpret_cast<uintptr_t>(ptr) - down_shift;
    auto size_in_bytes = sizeof(T) * size + down_shift;
    cstr s = "Memory Block Information\n";
    s.appendLine("    Address:\t",
                 cstr::addressOf(reinterpret_cast<uintptr_t>(data)));
    s.appendLine("    Block Size:\t", sizeof(T) * size, " bytes");
    s.appendLine("  Left Alignment");
    s.appendLine("    Type Alignment:\t", alignment);
    s.appendLine("    Shift:\t", down_shift);
    s.appendLine("    Address:\t", cstr::addressOf(aligned_base_address));
    s.appendLine("    Total Block Size:\t", size_in_bytes, " bytes");
    return s.str();
  }
  /// \brief Dumps memory region
  /// \tparam T
  /// \param data
  /// \param size
  /// \param bytes_per_row
  /// \param region
  /// \param options
  /// \return
  template <typename T>
  static std::string
  dump(const T *data, std::size_t size, u32 bytes_per_row = 8,
       const Layout &region = Layout(),
       memory_dumper_options options = memory_dumper_option_bits::none) {
    // check options_
    auto hide_zeros = options.contain(memory_dumper_option_bits::hide_zeros);
    auto include_header =
        !options.contain(memory_dumper_option_bits::hide_header);
    auto align_data = options.contain(memory_dumper_option_bits::cache_align);
    auto show_ascii = options.contain(memory_dumper_option_bits::show_ascii);
    auto write_to_console =
        options.contain(memory_dumper_option_bits::write_to_console);
    auto save_string =
        options.contain(memory_dumper_option_bits::save_to_string);
    auto colored_output =
        options.contain(memory_dumper_option_bits::colored_output);
    auto show_type_values =
        options.contain(memory_dumper_option_bits::type_values);
    if (!write_to_console && !save_string)
      write_to_console = true;
    // output string
    cstr output_string;
    // address size
    u32 address_digit_count = 8;
    // compute column size for text alignment
    u8 data_digit_count = 2;
    if (options.contain(memory_dumper_option_bits::decimal))
      data_digit_count = 3;
    else if (options.contain(memory_dumper_option_bits::binary))
      data_digit_count = 8;
    u8 header_digit_count = numbers::countHexDigits(bytes_per_row);
    u8 column_size = (std::max)(header_digit_count, data_digit_count);
    u8 address_column_size = address_digit_count + 2 + 2; // 0x + \t
    if (include_header) {
      cstr s = std::string(address_column_size, ' ');
      for (u32 i = 0; i < bytes_per_row; ++i) {
        auto bs = cstr::binary2Hex(i, true, true);
        if (i % 8 == 0)
          s.append(" ");
        s.append(std::setw(column_size), !bs.empty() ? bs : "0", " ");
      }
      if (save_string)
        output_string += s;
      if (write_to_console)
        std::cout << s;
    }
    auto alignment = (align_data) ? 64 : 1;
    auto ptr = reinterpret_cast<const u8 *>(data);
    ptrdiff_t shift = reinterpret_cast<uintptr_t>(ptr) & (alignment - 1);
    uintptr_t aligned_base_address = reinterpret_cast<uintptr_t>(ptr) - shift;
    ptrdiff_t byte_offset = 0;
    ptrdiff_t size_in_bytes = sizeof(T) * size + shift;
    auto line_count = 0;
    while (byte_offset < size_in_bytes) {
      { // ADDRESS
        cstr s;
        s.appendLine();
        s.append(
            cstr::addressOf(reinterpret_cast<uintptr_t>(
                                (void *)(aligned_base_address + byte_offset)))
                .c_str(),
            "  ");
        if (save_string)
          output_string += s;
        if (write_to_console) {
          if (colored_output && line_count % 2)
            std::cout << colors::console::dim << s
                      << colors::console::reset_dim;
          else
            std::cout << s;
        }
        line_count++;
      }
      std::string ascii_data;
      std::string type_values;
      for (ptrdiff_t i = 0; i < bytes_per_row; i++, byte_offset++) {
        if (i % 8 == 0) {
          if (write_to_console)
            std::cout << " ";
          if (save_string)
            output_string.append(" ");
        }
        if (aligned_base_address + byte_offset <
                reinterpret_cast<uintptr_t>(ptr) ||
            byte_offset >= size_in_bytes) {
          if (write_to_console)
            std::cout << std::string(column_size, ' ') + " ";
          if (save_string)
            output_string += std::string(column_size, ' ') + " ";
          ascii_data += '.';
          continue;
        }
        u8 byte = *(reinterpret_cast<u8 *>(aligned_base_address + byte_offset));
        cstr s;
        if (!hide_zeros || byte) {
          if (options.contain(memory_dumper_option_bits::hexadecimal))
            s.append(cstr::binary2Hex(byte), " ");
          else if (options.contain(memory_dumper_option_bits::decimal))
            s.append(std::setfill('0'), std::setw(column_size),
                     static_cast<u32>(byte), ' ');
          else if (options.contain(memory_dumper_option_bits::binary))
            s.append(cstr::byte2Binary((h_byte)byte), " ");
          else if (options.contain(memory_dumper_option_bits::hexii))
            s.append(std::string(column_size, ' '), " ");
          else
            s.append(cstr::binary2Hex(byte), " ");
        } else
          s.append(std::string(column_size, ' '), " ");

        if (save_string)
          output_string += s;
        std::string current_byte_color = byteColor(byte_offset - shift, region);
        if (write_to_console) {
          if (colored_output)
            std::cout << current_byte_color << s.str()
                      << colors::console::default_color
                      << colors::console::reset;
          else
            std::cout << s.str();
        }
        if (colored_output)
          ascii_data += current_byte_color;
        if (std::isalnum(byte))
          ascii_data += byte;
        else
          ascii_data += '.';
        if (colored_output) {
          ascii_data += colors::console::default_color;
          ascii_data += colors::console::reset;
        }
        // compute type value (if any)
        if (show_type_values) {
          if (colored_output)
            type_values += current_byte_color;
          type_values += typeValue(
              byte_offset - shift,
              reinterpret_cast<u8 *>(aligned_base_address + byte_offset),
              region);
          if (colored_output) {
            type_values += colors::console::default_color;
            type_values += colors::console::reset;
          }
          type_values += " ";
        }
      }
      if (show_ascii) {
        if (write_to_console)
          std::cout << "\t|" << ascii_data << "|";
        if (save_string)
          output_string.append("\t|", ascii_data, "|");
      }
      if (show_type_values) {
        if (write_to_console)
          std::cout << "\t<" << type_values << ">";
        if (save_string)
          output_string.append("\t<", type_values, ">");
      }
    }
    if (save_string)
      output_string += '\n';
    if (write_to_console)
      std::cout << "\n";
    return output_string.str();
  }

private:
  static std::string byteColor(std::size_t byte_index, const Layout &region) {
    std::function<std::string(const std::vector<Layout> &, std::size_t,
                              const std::string &)>
        f;
    f = [&](const std::vector<Layout> &subregions, std::size_t byte_offset,
            const std::string &parent_color) -> std::string {
      for (const auto &sub_region : subregions) {
        auto region_start = sub_region.offset;
        auto region_end =
            region_start + sub_region.field_size_in_bytes * sub_region.count;
        if (byte_offset >= region_start && byte_offset < region_end) {
          if (sub_region.sub_regions.empty()) {
            // in the case of an array of elements, lets alternate between
            // dimmed colors to make it easy to visually identify elements
            if (((byte_offset - region_start) /
                 sub_region.field_size_in_bytes) %
                2)
              return colors::console::combine(colors::console::bold,
                                              sub_region.color);
            return sub_region.color;
          }
          return f(sub_region.sub_regions,
                   (byte_offset - region_start) %
                       sub_region.field_size_in_bytes,
                   sub_region.color);
        }
      }
      return parent_color;
    };
    return f({region}, byte_index, colors::console::default_color);
  }

  static std::string typeValue(std::size_t byte_index, u8 *data,
                               const Layout &region) {
    std::function<std::string(const std::vector<Layout> &, std::size_t,
                              const std::string &)>
        f;
    f = [&](const std::vector<Layout> &subregions, std::size_t byte_offset,
            const std::string &parent_color) -> std::string {
      HERMES_UNUSED_VARIABLE(parent_color);
      for (const auto &sub_region : subregions) {
        auto region_start = sub_region.offset;
        auto region_end =
            region_start + sub_region.field_size_in_bytes * sub_region.count;
        if (byte_offset >= region_start && byte_offset < region_end) {
          if (sub_region.sub_regions.empty() &&
              sub_region.type != DataType::CUSTOM) {
            if ((byte_offset - region_start) %
                    DataTypes::typeSize(sub_region.type) ==
                0) {
              std::stringstream ss;
#define RETURN_TYPE(T)                                                         \
  if (sub_region.type == DataTypes::typeFrom<T>()) {                           \
    ss << std::setw(10) << std::right << std::setprecision(3)                  \
       << *reinterpret_cast<T *>(data);                                        \
    return ss.str();                                                           \
  }
              RETURN_TYPE(i8)
              RETURN_TYPE(i16)
              RETURN_TYPE(i32)
              RETURN_TYPE(i64)
              RETURN_TYPE(u8)
              RETURN_TYPE(u16)
              RETURN_TYPE(u32)
              RETURN_TYPE(u64)
              RETURN_TYPE(f32)
              RETURN_TYPE(f64)
#undef RETURN_TYPE
              return "ERROR";
            }
            return "";
          }
          return f(sub_region.sub_regions,
                   (byte_offset - region_start) %
                       sub_region.field_size_in_bytes,
                   sub_region.color);
        }
      }
      return "";
    };
    return f({region}, byte_index, colors::console::default_color);
  }
};

#ifdef HERMES_INCLUDE_DEBUG_TRAITS
inline std::string_view to_string(const MemoryDumper::Layout &layout) {
  cstr s;
  s << layout.color << "MemoryLayout [offset = " << layout.offset;
  s << " field size (bytes) = " << layout.field_size_in_bytes;
  s << " count = " << layout.count;
  s << " type = " << to_string(layout.type) << "]\n";
  s << "\tsub regions [" << layout.sub_regions.size() << "]\n";
  if (!layout.sub_regions.empty())
    for (const auto &sr : layout.sub_regions) {
      s += to_string(sr);
      s += "\n";
    }
  s += "\n";
  return s.str();
}

/// Adds MemoryDumper::Layout support for `std::ostream` `<<` operator
inline std::ostream &operator<<(std::ostream &os,
                                const MemoryDumper::Layout &layout) {
  os << to_string(layout);
  return os;
}
#endif

} // namespace hermes
