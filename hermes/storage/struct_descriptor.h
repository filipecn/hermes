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
///\file struct_descriptor_.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-07-14
///
///\brief

#ifndef HERMES_STORAGE_STRUCT_DESCRIPTOR_H
#define HERMES_STORAGE_STRUCT_DESCRIPTOR_H

#include <hermes/common/defs.h>
#include <hermes/numeric/math_element.h>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <fstream>

#define HERMES_STRUCT_DESCRIPTOR_VIEW_MAX_FIELD_COUNT 10

namespace hermes {

// *********************************************************************************************************************
//                                                                                               FORWARD DECLARATIONS
// *********************************************************************************************************************
template<MemoryLocation L> class ArrayOfStructs;
class StructDescriptor;

// *********************************************************************************************************************
//                                                                                               StructDescriptorView
// *********************************************************************************************************************
/// Describes the layout of a structure without field names
class StructDescriptorView {
public:
  // *******************************************************************************************************************
  //                                                                                                   FRIEND STRUCTS
  // *******************************************************************************************************************
  friend class StructDescriptor;
  friend class AoSView;
  friend class ConstAoSView;
  // *******************************************************************************************************************
  //                                                                                                     FIELD STRUCT
  // *******************************************************************************************************************
  struct Field {
    u64 size{0};                            //!< field size in bytes
    u64 offset{0};                          //!< field offset in bytes inside structure
    u32 component_count{1};                 //!< component count of data type
    DataType type{DataType::CUSTOM};        //!< data type id
  };
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  ///
  /// \tparam T
  /// \param data
  /// \param field_id
  /// \param i
  /// \return
  template<typename T>
  HERMES_DEVICE_CALLABLE T &valueAt(u8 *data, u64 field_id, u64 i) const {
    return *reinterpret_cast<T *>( reinterpret_cast<u8 *>(data) + i * struct_size + fields_[field_id].offset);
  }
  ///
  /// \tparam T
  /// \param data
  /// \param field_id
  /// \param i
  /// \return
  template<typename T>
  HERMES_DEVICE_CALLABLE const T &valueAt(const u8 *data, u64 field_id, u64 i) const {
    return *reinterpret_cast<const T *>( data + i * struct_size + fields_[field_id].offset);
  }
  //                                                                                                       assignment
  //                                                                                                       arithmetic
  //                                                                                                          boolean
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************

  // *******************************************************************************************************************
  //                                                                                                           LAYOUT
  // *******************************************************************************************************************
  /// Compute buffer offset of field at array index
  /// \param field_id
  /// \param i array index
  /// \return
  HERMES_DEVICE_CALLABLE [[nodiscard]] ptrdiff_t addressOffsetOf(u64 field_id, u64 i) const;
  HERMES_DEVICE_CALLABLE [[nodiscard]] u64 offsetOf(u64 field_id) const;
  HERMES_DEVICE_CALLABLE [[nodiscard]] u64 sizeOf(u64 field_id) const;
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  const u64 struct_size{0}; //!< in bytes
private:
  explicit StructDescriptorView(const StructDescriptor &descriptor);

  Field fields_[HERMES_STRUCT_DESCRIPTOR_VIEW_MAX_FIELD_COUNT];
};

// *********************************************************************************************************************
//                                                                                                   StructDescriptor
// *********************************************************************************************************************
/// Describes the structure that is stored in an array of structures
class StructDescriptor {
public:
  friend class StructDescriptorView;
  friend class ArrayOfStructs<MemoryLocation::DEVICE>;
  friend class ArrayOfStructs<MemoryLocation::HOST>;
  friend class ArrayOfStructs<MemoryLocation::UNIFIED>;
  // *******************************************************************************************************************
  //                                                                                                     FIELD STRUCT
  // *******************************************************************************************************************
  /// Field description
  /// \verbatim embed:rst:leading-slashes
  ///    **Example**::
  ///       Suppose a single field named "field_a" and defined as float[3].
  ///       Then name is "field_a", size is 3 * sizeof(float), offset is 0,
  ///       component_count is 3 and type is DataType::F32
  /// \endverbatim
  struct Field {
    std::string name;                       //!< field name
    u64 size{0};                            //!< field size in bytes
    u64 offset{0};                          //!< field offset in bytes inside structure
    u32 component_count{1};                 //!< component count of data type
    DataType type{DataType::CUSTOM};        //!< data type id
  };
  // *******************************************************************************************************************
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  //                                                                                                               io
  friend std::ostream &operator<<(std::ostream &o, const StructDescriptor &sd) {
    o << "Struct (size in bytes: " << sd.sizeInBytes() << ")\n";
    o << "fields: ";
    int i = 0;
    for (const auto &f : sd.fields_) {
      o << "field #" << i++ << " (" << f.name << " ):\n";
      o << "\tbase data type: " << DataTypes::typeName(f.type) << "\n";
      o << "\tbase data size in bytes: " << f.size << "\n";
      o << "\tcomponent count: " << f.component_count << "\n";
    }
    return o;
  }
  friend std::ofstream &operator<<(std::ofstream &o, const StructDescriptor &sd) {
    o.write(reinterpret_cast<const char *>(&sd.struct_size_), sizeof(u64));
    u64 field_count = sd.fields_.size();
    o.write(reinterpret_cast<const char *>(&field_count), sizeof(u64));
    for (const auto &f : sd.fields_) {
      o.write(reinterpret_cast<const char *>(&f.size), sizeof(StructDescriptor::Field::size));
      o.write(reinterpret_cast<const char *>(&f.component_count), sizeof(StructDescriptor::Field::component_count));
      o.write(reinterpret_cast<const char *>(&f.offset), sizeof(StructDescriptor::Field::offset));
      o.write(reinterpret_cast<const char *>(&f.type), sizeof(StructDescriptor::Field::type));
      u64 name_size = f.name.size();
      o.write(reinterpret_cast<const char *>(&name_size), sizeof(u64));
      auto name = f.name.c_str();
      for (u64 i = 0; i < name_size; ++i)
        o.write(reinterpret_cast<const char *>(&name[i]), sizeof(char));
    }
    return o;
  }
  friend std::ifstream &operator>>(std::ifstream &i, StructDescriptor &sd) {
    i.read(reinterpret_cast<char *>(&sd.struct_size_), sizeof(u64));
    u64 field_count = 0;
    i.read(reinterpret_cast<char *>(&field_count), sizeof(u64));
    for (u64 f = 0; f < field_count; ++f) {
      StructDescriptor::Field field;
      i.read(reinterpret_cast<char *>(&field.size), sizeof(StructDescriptor::Field::size));
      i.read(reinterpret_cast<char *>(&field.component_count), sizeof(StructDescriptor::Field::component_count));
      i.read(reinterpret_cast<char *>(&field.offset), sizeof(StructDescriptor::Field::offset));
      i.read(reinterpret_cast<char *>(&field.type), sizeof(StructDescriptor::Field::type));
      u64 name_size = 0;
      i.read(reinterpret_cast<char *>(&name_size), sizeof(u64));
      field.name.resize(name_size);
      i.read(&field.name[0], name_size);
      sd.field_id_map_[field.name] = sd.fields_.size();
      sd.fields_.emplace_back(field);
    }
    return i;
  }
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  StructDescriptor() = default;
  ~StructDescriptor() = default;
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  //                                                                                                           access
  StructDescriptorView view() const { return StructDescriptorView(*this); }
  //                                                                                                     field access
  /// Register field of structure.
  /// Note: fields are assumed to be stored in the same order they are pushed
  /// \tparam T base data type
  /// \param name field name
  /// \return field push id
  template<typename T>
  u64 pushField(const std::string &name) {
    field_id_map_[name] = fields_.size();
    Field d = {
        name,
        sizeof(T),
        struct_size_,
        1,
        DataTypes::typeFrom<T>()
    };
#define MATCH_HERMES_TYPES(Type, DT, C) \
    if (std::is_base_of_v<MathElement<Type, C>, T>) { \
      d.component_count = C; \
      d.type = DataType::DT;\
    }
    MATCH_HERMES_TYPES(f32, F32, 2u)
    MATCH_HERMES_TYPES(f32, F32, 3u)
    MATCH_HERMES_TYPES(f32, F32, 4u)
    MATCH_HERMES_TYPES(f32, F32, 9u)
    MATCH_HERMES_TYPES(f32, F32, 16u)
#undef MATCH_HERMES_TYPES
    fields_.emplace_back(d);
    struct_size_ += d.size;
    return fields_.size() - 1;
  }
  ///
  /// \tparam T
  /// \param data
  /// \param field_id
  /// \param i
  /// \return
  template<typename T>
  T &valueAt(void *data, u64 field_id, u64 i) const {
    return *reinterpret_cast<T *>( reinterpret_cast<u8 *>(data) + i * struct_size_ + fields_[field_id].offset);
  }
  ///
  /// \tparam T
  /// \param data
  /// \param field_id
  /// \param i
  /// \return
  template<typename T>
  const T &valueAt(const void *data, u64 field_id, u64 i) const {
    return *reinterpret_cast<const T *>( reinterpret_cast<const u8 *>(data) + i * struct_size_
        + fields_[field_id].offset);
  }
  //                                                                                                       field list
  /// \param field_id field push id
  /// \return field name
  inline const std::string &fieldName(u64 field_id) const { return fields_[field_id].name; }
  bool contains(const std::string &field_name) const { return field_id_map_.count(field_name); }
  inline const std::vector<Field> &fields() const { return fields_; }
  u64 fieldId(const std::string &field_name) const {
    auto it = field_id_map_.find(field_name);
    if (it != field_id_map_.end())
      return it->second;
    return 0;
  }
  // *******************************************************************************************************************
  //                                                                                                           LAYOUT
  // *******************************************************************************************************************
  ptrdiff_t addressOffsetOf(u64 field_id, u64 i) const;
  u64 offsetOf(const std::string &field_name) const;
  u64 offsetOf(u64 field_id) const;
  u64 sizeOf(const std::string &field_name) const;
  u64 sizeOf(u64 field_id) const;
  inline u64 sizeInBytes() const { return struct_size_; }

private:
  u64 struct_size_{0}; //!< in bytes
  std::vector<Field> fields_;
  std::unordered_map<std::string, u64> field_id_map_;
};

}

#endif // HERMES_STORAGE_STRUCT_DESCRIPTOR_H
