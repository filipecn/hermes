#ifndef HERMES_GEOMETRY_SEGMENT_H
#define HERMES_GEOMETRY_SEGMENT_H

#include <hermes/geometry/point.h>

namespace hermes {

/* Line segment **ab**
 */
template<typename T> class Segment {
public:
  Segment() {}
  Segment(T _a, T _b) : a(_a), b(_b) {}
  virtual ~Segment() {}

  float length() const { return (b - a).length(); }

  float length2() const { return (b - a).length2(); }
  T &operator[](size_t i) { return (i == 0) ? a : b; }
  T a, b;
};

typedef Segment<point2> Segment2;
typedef Segment<point3> Segment3;

} // hermes namespace

#endif // HERMES_GEOMETRY_SEGMENT_H
