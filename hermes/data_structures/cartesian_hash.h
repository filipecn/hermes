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
///\file cartesian_hash.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2022-11-24
///
///\brief

#ifndef HERMES_HERMES_DATA_STRUCTURES_CARTESIAN_HASH_H
#define HERMES_HERMES_DATA_STRUCTURES_CARTESIAN_HASH_H

#include <hermes/geometry/bbox.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                    CartesianHashMap
// *********************************************************************************************************************
/// \brief Point set structure that allows spatial searches.
class CartesianHashMap2 {
public:
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  ///
  /// \param point_set
  /// \param cell_size search grid cell size
  /// \return
  static CartesianHashMap2 from(const std::vector<point2> &point_set, real_t cell_size);
  // *******************************************************************************************************************
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  friend std::ostream &operator<<(std::ostream &os, const CartesianHashMap2 &chm);
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \brief Construct from parameters
  /// \param cell_size
  explicit CartesianHashMap2(real_t cell_size = 1);
  virtual ~CartesianHashMap2();
  //                                                                                                       assignment
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       assignment
  //                                                                                                       arithmetic
  //                                                                                                          boolean
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  /// \brief Search points within a region in space
  /// \param region
  /// \param callback
  /// \return number of points found
  size_t search(const bbox2 &region, const std::function<bool(size_t)> &callback) const;
  /// \brief Search points within a radius distance from the query position
  /// \param center
  /// \param radius
  /// \param callback
  /// \return number of points found
  size_t search(const hermes::point2 &center, real_t radius, const std::function<bool(size_t)> &callback) const;
  /// \brief Computes the hash key (cell id) for a given position
  /// \param p
  /// \return
  [[nodiscard]] index2_64 hashValue(const point2 &p) const;
  /// \brief Sets search grid cell size
  /// \note This method takes extra processing as it updates the internal search structure.
  /// \param cell_size
  void setSearchCellSize(real_t cell_size);
  /// \brief Updates the internal search structure
  void update();
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
private:
  /// \brief Iterate over points in a cell
  /// \param cell_id
  /// \param callback
  /// \return number of points found
  size_t search(const index2_64 &cell_id, const std::function<bool(size_t)> &callback) const;
  struct Element {
    index2_64 hash_value;
    point2 position;
    size_t index{0};
  };
  std::vector<Element> point_set_;
  real_t cell_size_{1};
};

}

#endif //HERMES_HERMES_DATA_STRUCTURES_CARTESIAN_HASH_H
