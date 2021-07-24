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
///\file data_structures_tests.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-07-24
///
///\brief
#include <catch2/catch.hpp>

#include <hermes/data_structures/n_mesh.h>

using namespace hermes;

TEST_CASE("NMesh", "[mesh]") {
  SECTION("Sanity") {
    auto mesh = NMesh::buildFrom(
        {
            {0.f, 1.f, 0.f}, // v0
            {0.f, 0.f, 0.f}, // v1
            {1.f, 0.f, 0.f}, // v2
            {1.f, 1.f, 0.f}, // v3
            {2.f, 1.f, 0.f}, // v4
        },
        {
            0, 1, 2, 3, // cell 0
            3, 2, 4 // cell 1
        },
        {4, 3}
    );
// METRICS
    REQUIRE(mesh.vertexCount() == 5);
    REQUIRE(mesh.faceCount() == 6);
    REQUIRE(mesh.cellCount() == 2);
    REQUIRE(mesh.cellFaceCount(0) == 4);
    REQUIRE(mesh.cellFaceCount(1) == 3);
// TOPOLOGY
    REQUIRE(mesh.nextHalfEdge(0) == 1);
    REQUIRE(mesh.nextHalfEdge(1) == 2);
    REQUIRE(mesh.nextHalfEdge(2) == 3);
    REQUIRE(mesh.nextHalfEdge(3) == 0);

    REQUIRE(mesh.nextHalfEdge(4) == 5);
    REQUIRE(mesh.nextHalfEdge(5) == 6);
    REQUIRE(mesh.nextHalfEdge(6) == 4);

    REQUIRE(mesh.nextHalfEdge(7) == 8);
    REQUIRE(mesh.nextHalfEdge(8) == 9);
    REQUIRE(mesh.nextHalfEdge(9) == 10);
    REQUIRE(mesh.nextHalfEdge(10) == 11);
    REQUIRE(mesh.nextHalfEdge(11) == 7);

    REQUIRE(mesh.previousHalfEdge(0) == 3);
    REQUIRE(mesh.previousHalfEdge(1) == 0);
    REQUIRE(mesh.previousHalfEdge(2) == 1);
    REQUIRE(mesh.previousHalfEdge(3) == 2);

    REQUIRE(mesh.previousHalfEdge(4) == 6);
    REQUIRE(mesh.previousHalfEdge(5) == 4);
    REQUIRE(mesh.previousHalfEdge(6) == 5);

    REQUIRE(mesh.previousHalfEdge(7) == 11);
    REQUIRE(mesh.previousHalfEdge(8) == 7);
    REQUIRE(mesh.previousHalfEdge(9) == 8);
    REQUIRE(mesh.previousHalfEdge(10) == 9);
    REQUIRE(mesh.previousHalfEdge(11) == 10);

    REQUIRE(mesh.halfEdgeFace(0) == 0);
    REQUIRE(mesh.halfEdgeFace(1) == 1);
    REQUIRE(mesh.halfEdgeFace(2) == 2);
    REQUIRE(mesh.halfEdgeFace(3) == 3);
    REQUIRE(mesh.halfEdgeFace(4) == 2);
    REQUIRE(mesh.halfEdgeFace(5) == 4);
    REQUIRE(mesh.halfEdgeFace(6) == 5);
    REQUIRE(mesh.halfEdgeFace(7) == 0);
    REQUIRE(mesh.halfEdgeFace(8) == 3);
    REQUIRE(mesh.halfEdgeFace(9) == 5);
    REQUIRE(mesh.halfEdgeFace(10) == 4);
    REQUIRE(mesh.halfEdgeFace(11) == 1);

    u64 a, b;
    mesh.faceVertices(0, a, b);
    REQUIRE(((a == 0 && b == 1) || (a == 1 && b == 0)));
    mesh.faceVertices(1, a, b);
    REQUIRE(((a == 2 && b == 1) || (a == 1 && b == 2)));
    mesh.faceVertices(2, a, b);
    REQUIRE(((a == 2 && b == 3) || (a == 3 && b == 2)));
    mesh.faceVertices(3, a, b);
    REQUIRE(((a == 0 && b == 3) || (a == 3 && b == 0)));
    mesh.faceVertices(4, a, b);
    REQUIRE(((a == 2 && b == 4) || (a == 4 && b == 2)));
    mesh.faceVertices(5, a, b);
    REQUIRE(((a == 3 && b == 4) || (a == 4 && b == 3)));

    std::cerr << mesh;
  }//
  SECTION("ITERATORS") {
    auto mesh = NMesh::buildFrom(
        {
            {0.f, 1.f, 0.f}, // v0
            {0.f, 0.f, 0.f}, // v1
            {1.f, 0.f, 0.f}, // v2
            {1.f, 1.f, 0.f}, // v3
            {2.f, 1.f, 0.f}, // v4
        },
        {
            0, 1, 2, 3, // cell 0
            3, 2, 4 // cell 1
        },
        {4, 3}
    );
    SECTION("VERTEX STAR") {
      for (auto s : mesh.vertexStar(0))
        std::cerr << s.faceIndex() << " " << s.cellIndex() << std::endl;
    }//
  }//
  SECTION("DUAL") {
//     v6 -------- v7 -------- v8
//      |    c2     |    c3    |
//     v3 -------- v4 -------- v5
//      |    c0     |    c1    |
//     v0 -------- v1 -------- v2
    PMesh<4> t_mesh;
    t_mesh.buildFrom({
                         {0.f, 0.f, 0.f},
                         {1.f, 0.f, 0.f},
                         {2.f, 0.f, 0.f},
                         {0.f, 1.f, 0.f},
                         {1.f, 1.f, 0.f},
                         {2.f, 1.f, 0.f},
                         {0.f, 2.f, 0.f},
                         {1.f, 2.f, 0.f},
                         {2.f, 2.f, 0.f},
                     }, {
                         0, 1, 4, 3, //
                         1, 2, 5, 4, //
                         3, 4, 7, 6, //
                         4, 5, 8, 7 //
                     });
    auto dual_mesh = NMesh::buildDualFrom(t_mesh);
    std::cerr << dual_mesh;
  } //
  SECTION("CellStar") {
//             v8  f11   v9
//            f12   c3   f10
//   v4   f2   v5   f6   v6   f9   v7
//   f3   c0   f1   c1   f5   c2   f8
//   v0   f0   v1   f4   v2   f7   v3
    NMesh mesh = NMesh::buildFrom(
        {
            {0.f, 0.f, 0.f}, // v0
            {1.f, 0.f, 0.f}, // v1
            {2.f, 0.f, 0.f}, // v2
            {3.f, 0.f, 0.f}, // v3
            {0.f, 1.f, 0.f}, // v4
            {1.f, 1.f, 0.f}, // v5
            {2.f, 1.f, 0.f}, // v6
            {3.f, 1.f, 0.f}, // v7
            {1.f, 2.f, 0.f}, // v8
            {2.f, 2.f, 0.f}, // v9
        },
        {
            0, 1, 5, 4, // c0
            1, 2, 6, 5, // c1
            2, 3, 7, 6, // c2
            5, 6, 9, 8, // c3
        }, {4, 4, 4, 4});
    auto I = Numbers::greatest<u64>();
    u64 expected_cell_id[4][4] = {
        {I, 1, I, I}, // c0
        {I, 2, 3, 0}, // c1
        {I, I, I, 1}, // c2
        {1, I, I, I}, // c3
    };
    u64 expected_face_id[4][4] = {
        {0, 1, 2, 3}, // c0
        {4, 5, 6, 1}, // c1
        {7, 8, 9, 5}, // c2
        {6, 10, 11, 12}, // c3
    };
    u64 expected_vertex_id[4][4] = {
        {0, 1, 5, 4}, // c0
        {1, 2, 6, 5}, // c1
        {2, 3, 7, 6}, // c2
        {5, 6, 9, 8}, // c3
    };
    for (auto C : mesh.cells()) {
      int i = 0;
      for (auto c : C.star()) {
        REQUIRE(c.vertexIndex() == expected_vertex_id[C.index][i]);
        REQUIRE(c.faceIndex() == expected_face_id[C.index][i]);
        REQUIRE(c.cellIndex() == expected_cell_id[C.index][i++]);
      }
    }
  }SECTION("Copy Assignment Operators") {
    auto source = NMesh::buildFrom(
        {
            {0.f, 1.f, 0.f}, // v0
            {0.f, 0.f, 0.f}, // v1
            {1.f, 0.f, 0.f}, // v2
            {1.f, 1.f, 0.f}, // v3
            {2.f, 1.f, 0.f}, // v4
        },
        {
            0, 1, 2, 3, // cell 0
            3, 2, 4 // cell 1
        },
        {4, 3}
    );
    NMesh moved = std::move(source);
    NMesh copied;
    copied = moved;
// METRICS
    REQUIRE(copied.vertexCount() == 5);
    REQUIRE(copied.faceCount() == 6);
    REQUIRE(copied.cellCount() == 2);
    REQUIRE(copied.cellFaceCount(0) == 4);
    REQUIRE(copied.cellFaceCount(1) == 3);
    REQUIRE(moved.vertexCount() == 5);
    REQUIRE(moved.faceCount() == 6);
    REQUIRE(moved.cellCount() == 2);
    REQUIRE(moved.cellFaceCount(0) == 4);
    REQUIRE(moved.cellFaceCount(1) == 3);
  }//
}

TEST_CASE("PMesh", "[mesh]") {
  SECTION("TMESH") {
    PMesh<3> mesh;
    mesh.buildFrom(
        {
            {0.0f, 0.f, 0.f}, // v0
            {1.0f, 0.f, 0.f}, // v1
            {0.5f, 1.f, 0.f}, // v2
            {1.5f, 1.f, 0.f} // v3
        },
        {
            0, 1, 2, // t0
            1, 3, 2 // t1
        }
    );

    std::cerr << mesh << std::endl;

    REQUIRE(mesh.vertexCount() == 4);
    REQUIRE(mesh.faceCount() == 5);
    REQUIRE(mesh.cellCount() == 2);

    REQUIRE(mesh.nextHalfEdge(0) == 1);
    REQUIRE(mesh.nextHalfEdge(1) == 2);
    REQUIRE(mesh.nextHalfEdge(2) == 0);
    REQUIRE(mesh.nextHalfEdge(3) == 4);
    REQUIRE(mesh.nextHalfEdge(4) == 5);
    REQUIRE(mesh.nextHalfEdge(5) == 3);

    REQUIRE(mesh.nextHalfEdge(6) == 7);
    REQUIRE(mesh.nextHalfEdge(7) == 8);
    REQUIRE(mesh.nextHalfEdge(8) == 9);
    REQUIRE(mesh.nextHalfEdge(9) == 6);

    REQUIRE(mesh.previousHalfEdge(1) == 0);
    REQUIRE(mesh.previousHalfEdge(2) == 1);
    REQUIRE(mesh.previousHalfEdge(0) == 2);
    REQUIRE(mesh.previousHalfEdge(4) == 3);
    REQUIRE(mesh.previousHalfEdge(5) == 4);
    REQUIRE(mesh.previousHalfEdge(3) == 5);

    REQUIRE(mesh.previousHalfEdge(7) == 6);
    REQUIRE(mesh.previousHalfEdge(8) == 7);
    REQUIRE(mesh.previousHalfEdge(9) == 8);
    REQUIRE(mesh.previousHalfEdge(6) == 9);

    REQUIRE(mesh.halfEdgeFace(0) == 0);
    REQUIRE(mesh.halfEdgeFace(1) == 1);
    REQUIRE(mesh.halfEdgeFace(2) == 2);
    REQUIRE(mesh.halfEdgeFace(3) == 3);
    REQUIRE(mesh.halfEdgeFace(4) == 4);
    REQUIRE(mesh.halfEdgeFace(5) == 1);

    u64 a, b;
    mesh.faceVertices(0, a, b);
    REQUIRE(((a == 0 && b == 1) || (b == 0 && a == 1)) == true);
    mesh.faceVertices(1, a, b);
    REQUIRE(((a == 2 && b == 1) || (b == 2 && a == 1)) == true);
    mesh.faceVertices(2, a, b);
    REQUIRE(((a == 2 && b == 0) || (b == 2 && a == 0)) == true);
    mesh.faceVertices(3, a, b);
    REQUIRE(((a == 3 && b == 1) || (b == 3 && a == 1)) == true);
    mesh.faceVertices(4, a, b);
    REQUIRE(((a == 2 && b == 3) || (b == 2 && a == 3)) == true);

    auto centroid = mesh.cellCentroid(0);
    REQUIRE(centroid.x == Approx(0.5));
    REQUIRE(centroid.y == Approx(1. / 3.));
    REQUIRE(centroid.z == Approx(0.0));
    centroid = mesh.cellCentroid(1);
    REQUIRE(centroid.x == Approx(1.0));
    REQUIRE(centroid.y == Approx(2. / 3.));
    REQUIRE(centroid.z == Approx(0.0));

    auto normal = mesh.cellNormal(0);
    REQUIRE(normal.x == Approx(0.0));
    REQUIRE(normal.y == Approx(0.0));
    REQUIRE(normal.z == Approx(1.0));
    normal = mesh.cellNormal(1);
    REQUIRE(normal.x == Approx(0.0));
    REQUIRE(normal.y == Approx(0.0));
    REQUIRE(normal.z == Approx(1.0));
  } //
}

TEST_CASE("PMesh File IO", "[mesh]") {
  PMesh<3> mesh;
  mesh.cellInterleavedData().pushField<hermes::point3>("centroid");
  mesh.cellInterleavedData().pushField<i32>("count");
  mesh.cellInterleavedData().pushField<f32>("temperature");
  mesh.vertexInterleavedData().pushField<f32>("temperature");
// distance between cell centroids
  mesh.faceInterleavedData().pushField<f32>("delta");
// face area
  mesh.faceInterleavedData().pushField<f32>("area");
// dot product of face tangent and cell centroids distance vector
  mesh.faceInterleavedData().pushField<f32>("tl");
// face normal
  mesh.faceInterleavedData().pushField<hermes::vec3>("normal");
  mesh.faceInterleavedData().pushField<f32>("temperature");
  std::vector<hermes::point3> positions;
  std::vector<u64> indices;
  const u64 N = 5;
  const f64 dx = 1. / N;
// generates a mesh representing a quad with a shear angle of 60 degrees
//    /______/
//   /      /
//  /______/ ) 60 degrees
  auto angle = 60 * hermes::Constants::pi / 180.0;
  hermes::vec3 u(dx, 0, 0);
  hermes::vec3 v(dx * std::cos(angle), dx * std::sin(angle), 0);
  hermes::point3 o;
  for (u64 yi = 0; yi <= N; ++yi)
    for (u64 xi = 0; xi <= N; ++xi) {
      positions.emplace_back(o + v * static_cast<f32>(yi) + u * static_cast<f32>(xi));
      if (yi < N && xi < N) {
        indices.emplace_back((yi + 0) * (N + 1) + xi + 0);
        indices.emplace_back((yi + 1) * (N + 1) + xi + 1);
        indices.emplace_back((yi + 1) * (N + 1) + xi + 0);
        indices.emplace_back((yi + 0) * (N + 1) + xi + 0);
        indices.emplace_back((yi + 0) * (N + 1) + xi + 1);
        indices.emplace_back((yi + 1) * (N + 1) + xi + 1);
      }
    }
  mesh.buildFrom(positions, indices);
// temperature field
  auto temperatureAt = [](const hermes::point3 &p) -> f32 {
    return p.x * p.x;
  };
// init cell data
  auto cell_centroids = mesh.cellInterleavedData().field<hermes::point3>("centroid");
  auto cell_temperature = mesh.cellInterleavedData().field<f32>("temperature");
  for (const auto c : mesh.cells()) {
    cell_centroids[c.index] = c.centroid();
// generate temperature field in cell centroids
    cell_temperature[c.index] = temperatureAt(cell_centroids[c.index]);
  }
// init face data
  auto face_delta = mesh.faceInterleavedData().field<f32>("delta");
  auto face_area = mesh.faceInterleavedData().field<f32>("area");
  auto face_tl = mesh.faceInterleavedData().field<f32>("tl");
  for (auto face : mesh.faces()) {
    face_area[face.index] = face.area();
    u64 a, b;
    face.cells(a, b);
    hermes::point3 ca = (a < mesh.cellCount()) ? cell_centroids[a] : face.centroid();
    hermes::point3 cb = (b < mesh.cellCount()) ? cell_centroids[b] : face.centroid();
    face_delta[face.index] = (cb - ca).length();
    face_tl[face.index] = hermes::dot(face.tangent(), cb - ca);
  }

  std::ofstream file_out("tmesh_data", std::ios::binary);
  file_out << mesh;
  file_out.close();
  PMesh<3> mesh2;
  std::ifstream file_in("tmesh_data", std::ios::binary | std::ios::in);
  file_in >> mesh2;
  file_in.close();
  REQUIRE(mesh.cellCount() == mesh2.cellCount());
//  std::cerr << mesh << std::endl;
//  std::cerr << mesh2 << std::endl;
}

TEST_CASE("PMesh Iterators", "[mesh]") {
  SECTION("CellStar") {
    PMesh<4> mesh;
//             v8  f11   v9
//            f12   c3   f10
//   v4   f2   v5   f6   v6   f9   v7
//   f3   c0   f1   c1   f5   c2   f8
//   v0   f0   v1   f4   v2   f7   v3
    mesh.buildFrom(
        {
            {0.f, 0.f, 0.f}, // v0
            {1.f, 0.f, 0.f}, // v1
            {2.f, 0.f, 0.f}, // v2
            {3.f, 0.f, 0.f}, // v3
            {0.f, 1.f, 0.f}, // v4
            {1.f, 1.f, 0.f}, // v5
            {2.f, 1.f, 0.f}, // v6
            {3.f, 1.f, 0.f}, // v7
            {1.f, 2.f, 0.f}, // v8
            {2.f, 2.f, 0.f}, // v9
        },
        {
            0, 1, 5, 4, // c0
            1, 2, 6, 5, // c1
            2, 3, 7, 6, // c2
            5, 6, 9, 8, // c3
        });
    u64 expected_cell_id[4][4] = {
        {4, 1, 4, 4}, // c0
        {6, 2, 3, 0}, // c1
        {6, 5, 5, 1}, // c2
        {1, 5, 5, 4}, // c3
    };
    u64 expected_face_id[4][4] = {
        {0, 1, 2, 3}, // c0
        {4, 5, 6, 1}, // c1
        {7, 8, 9, 5}, // c2
        {6, 10, 11, 12}, // c3
    };
    u64 expected_vertex_id[4][4] = {
        {0, 1, 5, 4}, // c0
        {1, 2, 6, 5}, // c1
        {2, 3, 7, 6}, // c2
        {5, 6, 9, 8}, // c3
    };
    for (auto C : mesh.cells()) {
      int i = 0;
      for (auto c : C.star()) {
        REQUIRE(c.vertexIndex() == expected_vertex_id[C.index][i]);
        REQUIRE(c.faceIndex() == expected_face_id[C.index][i]);
        REQUIRE(c.cellIndex() == expected_cell_id[C.index][i++]);
      }
    }
  }//
  SECTION("VertexStar") {
    PMesh<3> mesh;
    mesh.buildFrom(
        {
            {0.0f, 0.f, 0.f}, // v0
            {1.0f, 0.f, 0.f}, // v1
            {0.5f, 1.f, 0.f}, // v2
            {1.5f, 1.f, 0.f} // v3
        },
        {
            0, 1, 2, // t0
            1, 3, 2 // t1
        }
    );
    std::cerr << mesh << std::endl;
    for (auto s : mesh.vertexStar(2)) {
      std::cerr << s.cellIndex() << std::endl;
    }
  }//
  SECTION("VertexStar Loop") {
    PMesh<3> mesh;
    mesh.buildFrom(
        {
            {-1.0f, 0.f, 0.f}, // v0
            {1.0f, 0.f, 0.f}, // v1
            {2.0f, 1.f, 0.f}, // v2
            {0.0f, 2.f, 0.f}, // v3
            {-2.0f, 1.f, 0.f}, // v4
            {0.0f, 0.5f, 0.f} // v5
        },
        {
            0, 1, 5, // t0
            1, 2, 5, // t1
            2, 3, 5, // t2
            3, 4, 5, // t3
            4, 0, 5, // t4
        }
    );
    std::cerr << mesh << std::endl;
    for (auto s : mesh.vertexStar(5)) {
      std::cerr << s.cellIndex() << std::endl;
    }
  }//
  SECTION("CellIterator") {
    PMesh<3> mesh;
    mesh.buildFrom(
        {
            {-1.0f, 0.f, 0.f}, // v0
            {1.0f, 0.f, 0.f}, // v1
            {2.0f, 1.f, 0.f}, // v2
            {0.0f, 2.f, 0.f}, // v3
            {-2.0f, 1.f, 0.f}, // v4
            {0.0f, 0.5f, 0.f} // v5
        },
        {
            0, 1, 5, // t0
            1, 2, 5, // t1
            2, 3, 5, // t2
            3, 4, 5, // t3
            4, 0, 5, // t4
        }
    );
    std::cerr << mesh << std::endl;
    for (auto c : mesh.cells()) {
      std::cerr << c.index << c.centroid() << std::endl;
    }
  }//
  SECTION("FaceIterator") {

/*                2(0,1)
 *           f2             f4
 *    0(-1,0)       f1          3(1,0)
 *           f0             f3
 *                1(0,-1)
*/
    PMesh<3> mesh;
    mesh.buildFrom(
        {
            {-1.0f, 0.f, 0.f}, // v0
            {0.0f, -1.f, 0.f}, // v1
            {0.0f, 1.f, 0.f}, // v2
            {1.0f, 0.f, 0.f}, // v3
        },
        {
            0, 1, 2, // t0
            1, 3, 2, // t1
        }
    );
    for (auto f : mesh.faces()) {
      u64 a, b;
      f.vertices(a, b);
      std::cerr << f.index << ": " << a << ", " << b << " " << f.centroid() << std::endl;
      std::cerr << f.area() << std::endl;
      std::cerr << f.tangent() << std::endl;
      std::cerr << f.normal() << std::endl;
      if (f.isBoundary())
        std::cerr << "is boundary\n";
    }
  }//
}


/*
struct Primitive {
  [[nodiscard]] const bbox3 &bounds() const { return b; };
  bbox3 b;
  u32 id{};
};

TEST_CASE("BVH", "[structures]") {
  SECTION("Empty") {
    std::vector<Primitive> primitives;
    BVH <Primitive> bvh(primitives);
  }//
  SECTION("Single Primitive") {
    std::vector<Primitive> primitives = {{bbox3::unitBox(), 0u}};
    BVH <Primitive> bvh(primitives);
// traverse and count leaves
    u64 leaf_count = 0;
    bvh.traverse([](const bbox3 &bounds) -> bool { return true; }, [&](const BVH<Primitive>::LinearNode &node) {
      leaf_count++;
    });
    REQUIRE(leaf_count == 1);
  }//
  SECTION("Grid") {
    std::vector<Primitive> primitives;
    u32 n = 2;
    for (u32 x = 0; x < n; ++x)
      for (u32 y = 0; y < n; ++y)
        for (u32 z = 0; z < n; ++z)
          primitives.push_back({{{x - .1f, y - .1f, z - .1f},
                                 {x + .1f, y + .1f, z + .1f}},
                                x * n * n + y * n + z});
    BVH <Primitive> bvh(primitives, 4);
// traverse and count leaves
    u64 leaf_count = 0;
    bvh.traverse([](const bbox3 &bounds) -> bool { return true; }, [&](const BVH<Primitive>::LinearNode &node) {
      leaf_count++;
    });
    REQUIRE(leaf_count == n * n * n);
    ray3 r{{0.f, -1.f, 0.f}, {0.f, 1.f, 0.f}};
    auto primitive = bvh.intersect(r, [&](const Primitive &p) -> auto {
      return GeometricPredicates::intersect(p.bounds(), r);
    });
    REQUIRE(primitive.has_value());
    REQUIRE(primitive.value().id == 0u);
  }//
}
*/

