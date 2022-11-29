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
///\file multi_hash_map.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2022-11-14
///
///\brief

#ifndef HERMES_HERMES_DATA_STRUCTURES_MULTI_HASH_MAP_H
#define HERMES_HERMES_DATA_STRUCTURES_MULTI_HASH_MAP_H

#include <algorithm>
#include <hermes/common/result.h>
#include <unordered_map>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                       MultiHashMap
// *********************************************************************************************************************
/// \brief Holds a hierarchy of std::unordered_map to map arrays of values.
/// \tparam KeyDataType
/// \tparam MappedValueType
template<class KeyElementDataType, class MappedValueType>
class MultiHashMap {
public:
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  // *******************************************************************************************************************
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  MultiHashMap() {
    hash_nodes_.template emplace_back();
  }
  virtual ~MultiHashMap() = default;
  //                                                                                                       assignment
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  MappedValueType &operator[](const std::vector<KeyElementDataType> &key) {
    size_t current_node_id = 0;
    for (const auto &key_element : key) {
      HERMES_CHECK_EXP(current_node_id < hash_nodes_.size());
      hash_nodes_[current_node_id].has_key = true;
      if (!hash_nodes_[current_node_id].hash_map.count(key_element)) {
        hash_nodes_[current_node_id].hash_map[key_element] = hash_nodes_.size();
        hash_nodes_.push_back({
                                  .has_value = false,
                                  .has_key = false,
                                  .value = {},
                                  .hash_map = {}
                              });
      }
      current_node_id = hash_nodes_[current_node_id].hash_map[key_element];
    }
    if (hash_nodes_[current_node_id].has_value)
      return hash_nodes_[current_node_id].value;
    hash_nodes_[current_node_id].has_value = true;
    value_count_++;
    hash_nodes_[current_node_id].value = {};
    return hash_nodes_[current_node_id].value;
  }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  /// \brief Gets the number of values in the map.
  /// \return
  [[nodiscard]] size_t size() const { return value_count_; }
  ///
  /// \param key
  /// \return
  bool contains(const std::vector<KeyElementDataType> &key) const {
    size_t current_node_id = 0;
    for (const auto &key_element : key) {
      HERMES_CHECK_EXP(current_node_id < hash_nodes_.size());
      const auto &node = hash_nodes_[current_node_id];
      if (node.hash_map.count(key_element)) {
        auto it = node.hash_map.find(key_element);
        current_node_id = it->second;
      } else
        return false;
    }
    return hash_nodes_[current_node_id].has_value;
  }
  ///
  /// \param key
  /// \return
  Result <MappedValueType> get(const std::vector<KeyElementDataType> &key) {
    size_t current_node_id = 0;
    for (const auto &key_element : key) {
      HERMES_CHECK_EXP(current_node_id < hash_nodes_.size());
      const auto &node = hash_nodes_[current_node_id];
      auto it = node.hash_map.find(key_element);
      if (it != node.hash_map.end()) {
        current_node_id = it->second;
      } else
        return Result<MappedValueType>::error(HeResult::OUT_OF_BOUNDS);
    }
    if (hash_nodes_[current_node_id].has_value)
      return Result<MappedValueType>(hash_nodes_[current_node_id].value);
    return Result<MappedValueType>::error(HeResult::OUT_OF_BOUNDS);
  }
  /// \brief Inserts a (key, value) pair into the map.
  /// \note The new value overwrites any previous value.
  /// \param key
  /// \param value
  void insert(const std::vector<KeyElementDataType> &key, const MappedValueType &value) {
    size_t current_node_id = 0;
    for (const auto &key_element : key) {
      HERMES_CHECK_EXP(current_node_id < hash_nodes_.size());
      hash_nodes_[current_node_id].has_key = true;
      if (!hash_nodes_[current_node_id].hash_map.count(key_element)) {
        hash_nodes_[current_node_id].hash_map[key_element] = hash_nodes_.size();
        hash_nodes_.push_back({
                                  .has_value = false,
                                  .has_key = false,
                                  .value = {},
                                  .hash_map = {}
                              });
      }
      current_node_id = hash_nodes_[current_node_id].hash_map[key_element];
    }
    hash_nodes_[current_node_id].has_value = true;
    hash_nodes_[current_node_id].value = value;
    value_count_++;
  }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
private:
  struct MapNode {
    MapNode() = default;
    bool has_value{false};
    bool has_key{false};
    MappedValueType value{};
    std::unordered_map<KeyElementDataType, size_t> hash_map{};
  };
  size_t value_count_{0};
  std::vector<MapNode> hash_nodes_;
};

}

#endif //HERMES_HERMES_DATA_STRUCTURES_MULTI_HASH_MAP_H
