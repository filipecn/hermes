/*
 * Copyright (c) 2019 FilipeCN
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#ifndef HERMES_GEOMETRY_CUDA_POINT_H
#define HERMES_GEOMETRY_CUDA_POINT_H

#include <hermes/geometry/vector.h>
#include <ponos/geometry/point.h>

namespace hermes {

template<typename T> class Point2 {
  static_assert(std::is_same<T, f32>::value
                    || std::is_same<T, f64>::value ||
                    std::is_same<T, float>::value || std::is_same<T, double>::value,
                "Point2 must hold an float type!");
public:
  // ***********************************************************************
  //                           STATIC METHODS
  // ***********************************************************************
  typedef T ScalarType;
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  __host__ __device__ explicit Point2(T f = T(0)) { x = y = f; }
  __host__ __device__ explicit Point2(const real_t *v) : x(v[0]), y(v[1]) {}
  __host__ __device__ Point2(real_t _x, real_t _y) : x(_x), y(_y) {}
  __host__ explicit Point2(const ponos::Point2<T> &ponos_point)
      : x(ponos_point.x), y(ponos_point.y) {}
  // access
  // arithmetic
  // ***********************************************************************
  //                           ARITHMETIC
  // ***********************************************************************
  __host__ __device__ Point2 &operator+=(const Vector2<T> &v) {
    x += v.x;
    y += v.y;
    return *this;
  }
  __host__ __device__ Point2 &operator-=(const Vector2<T> &v) {
    x -= v.x;
    y -= v.y;
    return *this;
  }
  __host__ __device__ Point2 &operator/=(real_t d) {
    x /= d;
    y /= d;
    return *this;
  }
  // ***********************************************************************
  //                           METHODS
  // ***********************************************************************
  bool HasNaNs() const;
  ponos::Point2<T> ponos() const { return ponos::Point2<T>(x, y); }
  // ***********************************************************************
  //                           DEBUG
  // ***********************************************************************
  template<typename TT>
  friend std::ostream &operator<<(std::ostream &os, const Point2<TT> &p) {
    os << "[Point2] " << p.x << " " << p.y << std::endl;
    return os;
  }
  // ***********************************************************************
  //                           FIELDS
  // ***********************************************************************
  __host__ __device__ T operator[](int i) const { return (&x)[i]; }
  __host__ __device__ T &operator[](int i) { return (&x)[i]; }

  T x = T(0.0);
  T y = T(0.0);
};

// ***********************************************************************
//                           ARITHMETIC
// ***********************************************************************
template<typename T>
Point2<T> operator+(const Point2<T> a, const Vector2<T> &v) {
  return Point2<T>(a.x + v.x, a.y + v.y);
}
template<typename T>
Point2<T> operator-(const Point2<T> a, const Vector2<T> &v) {
  return Point2<T>(a.x - v.x, a.y - v.y);
}
template<typename T> Point2<T> operator+(const Point2<T> a, const T &f) {
  return Point2<T>(a.x + f, a.y + f);
}
template<typename T> Point2<T> operator-(const Point2<T> a, const T &f) {
  return Point2<T>(a.x - f, a.y - f);
}
template<typename T>
Vector2<T> operator-(const Point2<T> &a, const Point2<T> &b) {
  return Vector2<T>(a.x - b.x, a.y - b.y);
}
template<typename T> Point2<T> operator/(const Point2<T> a, real_t f) {
  return Point2<T>(a.x / f, a.y / f);
}
template<typename T> Point2<T> operator*(const Point2<T> a, real_t f) {
  return Point2<T>(a.x * f, a.y * f);
}
template<typename T> Point2<T> operator*(real_t f, const Point2<T> &a) {
  return Point2<T>(a.x * f, a.y * f);
}
// ***********************************************************************
//                           BOOLEAN
// ***********************************************************************
template<typename T> bool operator==(const Point2<T> &a, const Point2<T> &b) {
  return Check::is_equal(a.x, b.x) && Check::is_equal(a.y, b.y);
}
template<typename T> bool operator<(const Point2<T> &a, const Point2<T> &b) {
  return !(a.x >= b.x || a.y >= b.y);
}
template<typename T> bool operator>=(const Point2<T> &a, const Point2<T> &b) {
  return a.x >= b.x && a.y >= b.y;
}
template<typename T> bool operator<=(const Point2<T> &a, const Point2<T> &b) {
  return a.x <= b.x && a.y <= b.y;
}
// ***********************************************************************
//                           GEOMETRY
// ***********************************************************************
template<typename T> __host__ __device__ real_t distance(const Point2<T> &a, const Point2<T> &b) {
  return (a - b).length();
}
template<typename T> __host__ __device__ real_t distance2(const Point2<T> &a, const Point2<T> &b) {
  return (a - b).length2();
}
template<typename T> __host__ __device__ Point2<T> floor(const Point2<T> &a) {
  return {static_cast<int>(a.x), static_cast<int>(a.y)};
}

typedef Point2<float> point2;
typedef Point2<float> point2f;
typedef Point2<double> point2d;

template<typename T> class Point3 {
  static_assert(std::is_same<T, f32>::value
                    || std::is_same<T, f64>::value ||
                    std::is_same<T, float>::value || std::is_same<T, double>::value,
                "Point3 must hold an float type!");
public:
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  __host__ __device__ Point3() : x{0}, y{0}, z{0} {};
  __host__ __device__ Point3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
  __host__ __device__ explicit Point3(const Vector3<T> &v) : x(v.x), y(v.y), z(v.z) {}
  __host__ __device__ explicit Point3(const Point2<T> &p) : x(p.x), y(p.y), z(0) {}
  __host__ __device__ explicit Point3(const T *v) : x(v[0]), y(v[1]), z(v[2]) {}
  __host__ __device__ explicit Point3(T v) : x(v), y(v), z(v) {}
  // ***********************************************************************
  //                           ACCESS
  // ***********************************************************************
  __host__ __device__ T operator[](int i) const { return (&x)[i]; }
  __host__ __device__ T &operator[](int i) { return (&x)[i]; }
  __host__ __device__ Point2<T> xy() const { return Point2<T>(x, y); }
  __host__ __device__ Point2<T> yz() const { return Point2<T>(y, z); }
  __host__ __device__ Point2<T> xz() const { return Point2<T>(x, z); }
  __host__ __device__ T u() const { return x; }
  __host__ __device__ T v() const { return y; }
  __host__ __device__ T s() const { return z; }
  // ***********************************************************************
  //                           ARITHMETIC
  // ***********************************************************************
  __host__ __device__ Point3<T> operator+() const { return *this; }
  __host__ __device__ Point3<T> operator-() const {
    return Point3<T>(-x, -y, -z);
  }
  __host__ __device__ Point3<T> &operator+=(const Vector3<T> &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }
  __host__ __device__ Point3<T> &operator*=(T d) {
    x *= d;
    y *= d;
    z *= d;
    return *this;
  }
  __host__ __device__ Point3<T> &operator-=(const Vector3<T> &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
  }
  __host__ __device__ Point3<T> &operator/=(T d) {
    x /= d;
    y /= d;
    z /= d;
    return *this;
  }
  // ***********************************************************************
  //                           DEBUG
  // ***********************************************************************
  template<typename TT>
  __host__ friend std::ostream &operator<<(std::ostream &os, const Point3<TT> &p) {
    os << "[Point3] (" << p.x << " " << p.y << " " << p.z << ")";
    return os;
  }
  // ***********************************************************************
  //                           FIELDS
  // ***********************************************************************
  T x = 0, y = 0, z = 0;
};

typedef Point3<float> point3;
typedef Point3<unsigned int> point3u;
typedef Point3<int> point3i;
typedef Point3<float> point3f;
typedef Point3<float> point3d;

// ***********************************************************************
//                           ARITHMETIC
// ***********************************************************************
template<typename T>
__host__ __device__ Point3<T> operator+(T f, const Point3<T> &p) {
  return Point3<T>(p.x + f, p.y + f, p.z + f);
}
template<typename T>
__host__ __device__ Point3<T> operator-(T f, const Point3<T> &p) {
  return Point3<T>(p.x - f, p.y - f, p.z - f);
}
template<typename T>
__host__ __device__ Point3<T> operator*(T d, const Point3<T> &p) {
  return Point3<T>(p.x * d, p.y * d, p.z * d);
}
template<typename T>
__host__ __device__ Point3<T> operator+(const Point3<T> &p, T f) {
  return Point3<T>(p.x + f, p.y + f, p.z + f);
}
template<typename T>
__host__ __device__ Point3<T> operator-(const Point3<T> &p, T f) {
  return Point3<T>(p.x - f, p.y - f, p.z - f);
}
template<typename T>
__host__ __device__ Point3<T> operator*(const Point3<T> &p, T d) {
  return Point3<T>(p.x * d, p.y * d, p.z * d);
}
template<typename T>
__host__ __device__ Point3<T> operator/(const Point3<T> &p, T d) {
  return Point3<T>(p.x / d, p.y / d, p.z / d);
}
template<typename T>
__host__ __device__ Point3<T> operator+(const Point3<T> &p,
                                        const Vector3<T> &v) {
  return Point3<T>(p.x + v.x, p.y + v.y, p.z + v.z);
}
template<typename T>
__host__ __device__ Point3<T> operator-(const Point3<T> &p,
                                        const Vector3<T> &v) {
  return Point3<T>(p.x - v.x, p.y - v.y, p.z - v.z);
}
template<typename T>
__host__ __device__ Point3<T> operator+(const Point3<T> &p,
                                        const Point3<T> &v) {
  return Point3<T>(p.x + v.x, p.y + v.y, p.z + v.z);
}
template<typename T>
__host__ __device__ Vector3<T> operator-(const Point3<T> &q,
                                         const Point3<T> &p) {
  return Vector3<T>(q.x - p.x, q.y - p.y, q.z - p.z);
}

// ***********************************************************************
//                           BOOLEAN
// ***********************************************************************
template<typename T>
__host__ __device__ bool operator==(const Point3<T> &p, const Point3<T> &q) {
  return Check::is_equal(p.x, q.x) && Check::is_equal(p.y, q.y) &&
      Check::is_equal(p.z, q.z);
}
template<typename T>
__host__ __device__ bool operator>=(const Point3<T> &q, const Point3<T> &p) {
  return q.x >= p.x && q.y >= p.y && q.z >= p.z;
}
template<typename T>
__host__ __device__ bool operator<=(const Point3<T> &q, const Point3<T> &p) {
  return q.x <= p.x && q.y <= p.y && q.z <= p.z;
}
template<typename T>
__host__ __device__ bool operator<(const Point3<T> &left,
                                   const Point3<T> &right) {
  if (left.x < right.x)
    return true;
  else if (left.x > right.x)
    return false;
  if (left.y < right.y)
    return true;
  else if (left.y > right.y)
    return false;
  if (left.z < right.z)
    return true;
  else if (left.z > right.z)
    return false;
  return false;
}

// ***********************************************************************
//                           GEOMETRY
// ***********************************************************************
template<typename T>
T __host__ __device__ distance(const Point3<T> &a, const Point3<T> &b) {
  return (a - b).length();
}
template<typename T>
T __host__ __device__ distance2(const Point3<T> &a, const Point3<T> &b) {
  return (a - b).length2();
}
template<typename T> __host__ __device__ Point3<T> floor(const Point3<T> &a) {
  return {static_cast<int>(a.x), static_cast<int>(a.y), static_cast<int>(a.z)};
}

} // namespace hermes

#endif
