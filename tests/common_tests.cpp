//
// Created by filipecn on 26/06/2021.
//


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
///\file common_tests.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-06-26
///
///\brief

#include <catch2/catch.hpp>

#include <hermes/common/arg_parser.h>
#include <hermes/common/size.h>
#include <hermes/common/index.h>
#include <hermes/common/bitmask_operators.h>
#include <hermes/common/file_system.h>
#include <hermes/common/cuda_utils.h>
#include <hermes/storage/array.h>

using namespace hermes;

TEST_CASE("ArgParser") {
  SECTION("simple") {
    const char *argv[3] = {"bin", "int_argument", "3"};
    ArgParser parser("test_bin", "test bin description");
    parser.addArgument("int_argument", "int argument description");
    parser.addArgument("arg2");
    REQUIRE(parser.parse(3, argv, true));
    REQUIRE(parser.get<int>("int_argument", 0) == 3);
    REQUIRE(!parser.check("arg2"));
  }//
  SECTION("simple 2") {
    const char *argv[6] = {"bin", "-v", "-f", "100", "-fps", "200"};
    ArgParser parser;
    parser.addArgument("-f");
    parser.addArgument("-fps");
    parser.addArgument("-v");
    parser.parse(6, argv, true);
    REQUIRE(parser.get<int>("-f", 0) == 100);
    REQUIRE(parser.get<int>("-fps", 0) == 200);
    REQUIRE(parser.check("-v"));

  }//
  SECTION("simple 2") {
    const char *argv[4] = {"bin", "100", "200", "-v"};
    ArgParser parser;
    parser.addArgument("-f");
    parser.addArgument("-fps");
    parser.addArgument("-v");
    parser.parse(4, argv);
    REQUIRE(parser.get<int>("-f", 0) == 100);
    REQUIRE(parser.get<int>("-fps", 0) == 200);
    REQUIRE(parser.check("-v"));
  }//
  SECTION("simple 3") {
    const char *argv[5] = {"bin", "-f", "100", "-fps", "200"};
    ArgParser parser;
    parser.addArgument("-f");
    parser.addArgument("-fps");
    parser.addArgument("-v");
    parser.parse(5, argv);
    REQUIRE(parser.get<int>("-f", 0) == 100);
    REQUIRE(parser.get<int>("-fps", 0) == 200);
    REQUIRE(!parser.check("-v"));
    REQUIRE(!parser.check("-v"));

  }//
  SECTION("required") {
    const char *argv[3] = {"bin", "int_argument", "3"};
    ArgParser parser;
    parser.addArgument("req", "", true);
    parser.addArgument("int_argument");
    REQUIRE(!parser.parse(3, argv));

  }//
  SECTION("positional arguments") {
    const char *argv[5] = {"bin", "4", "arg", "1", "2"};
    ArgParser parser;
    parser.addArgument("a0");
    parser.addArgument("a1");
    parser.addArgument("a2");
    parser.addArgument("a3");
    REQUIRE(parser.parse(5, argv));
    REQUIRE(parser.get<int>("a0") == 4);
    REQUIRE(parser.get<std::string>("a1") == "arg");
    REQUIRE(parser.get<int>("a2") == 1);
    REQUIRE(parser.get<int>("a3") == 2);
    REQUIRE(parser.check("a0"));
    REQUIRE(parser.check("a1"));
    REQUIRE(parser.check("a2"));
    REQUIRE(parser.check("a3"));
  }//
  SECTION("positional arguments mixed") {
    const char *argv[5] = {"bin", "4", "a1", "1", "2"};
    ArgParser parser;
    parser.addArgument("a0");
    parser.addArgument("a1");
    parser.addArgument("a2");
    REQUIRE(parser.parse(5, argv));
    REQUIRE(parser.get<int>("a0") == 4);
    REQUIRE(parser.get<int>("a1") == 1);
    REQUIRE(parser.get<int>("a2") == 2);
    REQUIRE(parser.check("a0"));
    REQUIRE(parser.check("a1"));
    REQUIRE(parser.check("a2"));
  }//
  SECTION("print help") {
    ArgParser parser("test bin", "test bin description.");
    parser.addArgument("a0", "a0 description", false);
    parser.addArgument("a1", "a1 description", false);
    parser.addArgument("a2", "a2 description", true);
    parser.addArgument("a3", "a3 description", true);
    parser.printHelp();
  }//
}

namespace hermes {
enum class Test : unsigned {
  a1 = 0x1,
  a2 = 0x2,
};
HERMES_ENABLE_BITMASK_OPERATORS(Test);
}

TEST_CASE("bitmask operators") {
  REQUIRE(static_cast<unsigned>(Test::a1 | Test::a2) == 0x3);
  REQUIRE(static_cast<unsigned>(Test::a1 ^ Test::a2) == 0x3);
  REQUIRE(static_cast<unsigned>(Test::a1 & Test::a2) == 0x0);
  REQUIRE(static_cast<unsigned>(~Test::a1) == 0xfffffffe);
  REQUIRE(HERMES_MASK_BIT(Test::a1 | Test::a2, Test::a1));
}

TEST_CASE("Str", "[common]") {
  SECTION("format") {
    REQUIRE(Str::format("word") == "word");
    REQUIRE(Str::format("word", 3) == "word");
    REQUIRE(Str::format("word {}") == "word {}");
    REQUIRE(Str::format("word {}", 3) == "word 3");
    REQUIRE(Str::format("word {} word {}", 3) == "word 3 word {}");
    REQUIRE(Str::format("word {} word {}", 3, 4) == "word 3 word 4");
    REQUIRE(Str::format("word {} word {}", 3, 4, 5) == "word 3 word 4");
  }//
  SECTION("split") {
    std::string a = "1 22 3 44 5";
    auto s = Str::split(a);
    REQUIRE(s.size() == 5);
    REQUIRE(s[0] == "1");
    REQUIRE(s[1] == "22");
    REQUIRE(s[2] == "3");
    REQUIRE(s[3] == "44");
    REQUIRE(s[4] == "5");
  }//
  SECTION("join") {
    std::vector<std::string> s = {"a", "b", "c"};
    auto ss = Str::join(s, ",");
    REQUIRE(ss == "a,b,c");
    std::vector<int> ints = {1, 2, 3};
    ss = Str::join(ints, " ");
    REQUIRE(ss == "1 2 3");
  }//
  SECTION("split with delimiter") {
    std::string a = "1 2, 3,4, 5";
    auto s = Str::split(a, ",");
    REQUIRE(s.size() == 4);
    REQUIRE(s[0] == "1 2");
    REQUIRE(s[1] == " 3");
    REQUIRE(s[2] == "4");
    REQUIRE(s[3] == " 5");
  }//
  SECTION("concat") {
    std::string a = Str::concat("a", " ", 2, "b");
    REQUIRE(a == "a 2b");
  }//
  SECTION("regex match") {
    REQUIRE(Str::match_r("subsequence123", "\\b(sub)([^ ]*)"));
    REQUIRE(!Str::match_r("susequence123", "\\b(sub)([^ ]*)"));
    REQUIRE(Str::match_r("sub-sequence123", "\\b(sub)([^ ]*)"));
  }//
  SECTION("regex contains") {
    REQUIRE(Str::contains_r("subsequence123", "\\b(sub)"));
    REQUIRE(!Str::contains_r("subsequence123", "\\b(qen)"));
    REQUIRE(Str::contains_r("/usr/local/lib.a", ".*\\.a"));
  }//
  SECTION("regex search") {
    std::string s("this subject has a submarine as a subsequence");
    auto result = Str::search_r(s, "\\b(sub)([^ ]*)");
    REQUIRE(result.size() == 3);
    REQUIRE(result[0] == "subject");
    REQUIRE(result[1] == "sub");
    REQUIRE(result[2] == "ject");
    int index = 0;
    std::string expected[3] = {"subject", "submarine", "subsequence"};
    Str::search_r(s, "\\b(sub)([^ ]*)", [&](const std::smatch &m) {
      REQUIRE(m[0] == expected[index++]);
    });
  }//
  SECTION("regex replace") {
    std::string s("there is a subsequence in the string");
    REQUIRE(Str::replace_r(s, "\\b(sub)([^ ]*)", "sub-$2") == "there is a sub-sequence in the string");
    REQUIRE(Str::replace_r(s, "\\b(sub)([^ ]*)", "$2") == "there is a sequence in the string");
    std::string s2("/home//usr/local");
    REQUIRE(Str::replace_r(s2, "\\b//", "/") == "/home/usr/local");
  }//
  SECTION("string class") {
    Str s;
    REQUIRE((s += "abc") == "abc");
    REQUIRE(s << 2 << 3 == "abc23");
    REQUIRE(s + 2 == "abc2");
    s = "3";
    REQUIRE(s == 3);
  }//
}

TEST_CASE("Path", "[common]") {
  Path folder("path_test_folder/folder");
  REQUIRE(folder.mkdir());
  REQUIRE(FileSystem::touch(folder + "file.txt"));
  REQUIRE(FileSystem::touch("path_test_folder/file.txt"));
  SECTION("dir") {
    Path path("folder_path");
    REQUIRE(static_cast<std::string>(path) == "folder_path");
    REQUIRE(!path.exists());
    path = std::string("path_test_folder");
    path.cd("folder");
    REQUIRE(path == "path_test_folder/folder");
    path.cd("../folder");
    REQUIRE(path == "path_test_folder/folder");
    path.cd("../folder/..").cd("folder");
    REQUIRE(path == "path_test_folder/folder");
    REQUIRE(path.isDirectory());
    REQUIRE(!path.isFile());
    path.join("folder2").join("/folder3");
    REQUIRE(path == "path_test_folder/folder/folder2/folder3");
    path.cd("../../");
    REQUIRE(path == "path_test_folder/folder");
  }//
  SECTION("file") {
    Path path("folder_path/file.txt");
    REQUIRE(!path.exists());
    path = folder + "file.txt";
    REQUIRE(path.exists());
    REQUIRE(path.isFile());
    REQUIRE(path.cwd() == folder);
    REQUIRE(path.extension() == "txt");
  }
}

TEST_CASE("FileSystem", "[common]") {
  SECTION("basename") {
    REQUIRE(FileSystem::basename("/usr/local/file.ext") == "file.ext");
    REQUIRE(FileSystem::basename("/usr/local/file.ext", ".ext") == "file");
    REQUIRE(FileSystem::basename("file.ext", ".ext") == "file");
    REQUIRE(FileSystem::basename("/usr/file.ex", ".ext") == "file.ex");
    REQUIRE(FileSystem::basename("/usr/").empty());
    REQUIRE(FileSystem::basename("/usr/", ".ext").empty());
  }//
  SECTION("basenames") {
    std::vector<std::string> paths = {
        "/usr/local/file.ext",
        "file.ext",
        "/usr/file.ex",
        "/usr/.ext",
        "/usr/"};
    std::vector<std::string> expected = {
        "file", "file", "file.ex", "", ""
    };
    auto basenames = FileSystem::basename(paths, ".ext");
    for (u64 i = 0; i < basenames.size(); ++i)
      REQUIRE(basenames[i] == expected[i]);
  }//
  SECTION("file extension") {
    REQUIRE(FileSystem::fileExtension("path/to/file.ext4") == "ext4");
    REQUIRE(FileSystem::fileExtension("path/to/file").empty());
  }//
  SECTION("read invalid file") {
    REQUIRE(!FileSystem::fileExists("invalid__file"));
    REQUIRE(FileSystem::readFile("invalid___file").empty());
    REQUIRE(FileSystem::readBinaryFile("invalid___file").empty());
  }//
  SECTION("isFile and isDirectory") {
    REQUIRE(FileSystem::writeFile("filesystem_test_file.txt", "test") == 4);
    REQUIRE(FileSystem::isFile("filesystem_test_file.txt"));
    REQUIRE(FileSystem::mkdir("path/to/dir"));
    REQUIRE(FileSystem::isDirectory("path/to/dir"));
  }//
  SECTION("copy file") {
    REQUIRE(FileSystem::writeFile("source", "source_content") > 0);
    REQUIRE(FileSystem::copyFile("source", "destination"));
    REQUIRE(FileSystem::fileExists("destination"));
    REQUIRE(FileSystem::readFile("destination") == "source_content");
  }//
  SECTION("append") {
    REQUIRE(FileSystem::writeFile("append_test", "") == 0);
    REQUIRE(FileSystem::fileExists("append_test"));
    REQUIRE(FileSystem::readFile("append_test").empty());
    REQUIRE(FileSystem::appendToFile("append_test", "append_content"));
    REQUIRE(FileSystem::readFile("append_test") == "append_content");
    REQUIRE(FileSystem::appendToFile("append_test", "123"));
    REQUIRE(FileSystem::readFile("append_test") == "append_content123");
  }//
  SECTION("ls") {
    REQUIRE(FileSystem::mkdir("ls_folder/folder"));
    REQUIRE(FileSystem::touch("ls_folder/file4"));
    REQUIRE(FileSystem::touch("ls_folder/folder/file1"));
    REQUIRE(FileSystem::touch("ls_folder/folder/file2"));
    REQUIRE(FileSystem::touch("ls_folder/folder/file3"));
    REQUIRE(FileSystem::mkdir("ls_folder/folder2"));
    REQUIRE(FileSystem::touch("ls_folder/folder2/file2"));
    REQUIRE(FileSystem::touch("ls_folder/folder2/file3"));
    REQUIRE(FileSystem::mkdir("ls_folder/folder2/folder"));
    REQUIRE(FileSystem::touch("ls_folder/folder2/folder/file1"));
    // ls_folder
    //  | folder
    //     | file1
    //     | file2
    //     | file3
    //  | folder2
    //     | folder
    //        | file1
    //     | file2
    //     | file3
    //  | file4
    { // simple ls
      auto ls = FileSystem::ls("ls_folder/folder");
      std::sort(ls.begin(), ls.end(), [](const Path &a, const Path &b) { return a.name() < b.name(); });
      std::vector<std::string> expected = {"file1", "file2", "file3"};
      REQUIRE(ls.size() == expected.size());
      for (u64 i = 0; i < ls.size(); ++i)
        REQUIRE(ls[i].name() == expected[i]);
    }
    {
      auto ls = FileSystem::ls("ls_folder", ls_options::recursive | ls_options::files | ls_options::sort);
      std::vector<std::string> expected = {"file4", "file1", "file2", "file3", "file2", "file3", "file1"};
      REQUIRE(ls.size() == expected.size());
      for (u64 i = 0; i < ls.size(); ++i)
        REQUIRE(ls[i].name() == expected[i]);
    }
    {
      auto ls = FileSystem::ls("ls_folder", ls_options::recursive | ls_options::files | ls_options::reverse_sort);
      std::vector<std::string> expected = {"file1", "file3", "file2", "file3", "file2", "file1", "file4"};
      REQUIRE(ls.size() == expected.size());
      for (u64 i = 0; i < ls.size(); ++i)
        REQUIRE(ls[i].name() == expected[i]);
    }
    {
      auto ls = FileSystem::ls("ls_folder", ls_options::sort | ls_options::group_directories_first);
      std::vector<std::string> expected = {"folder", "folder2", "file4"};
      REQUIRE(ls.size() == expected.size());
      for (u64 i = 0; i < ls.size(); ++i)
        REQUIRE(ls[i].name() == expected[i]);
    }
  }//
  SECTION("filter") {
    REQUIRE(FileSystem::mkdir("find_dir"));
    Path find_dir("find_dir");
    for (int i = 0; i < 5; i++)
      REQUIRE((find_dir + (Str() << "file" << i << ".ext1")).touch());
    for (int i = 0; i < 5; i++)
      REQUIRE((find_dir + (Str() << "file" << i << ".ext2")).touch());
    REQUIRE(FileSystem::mkdir("find_dir/folder"));
    REQUIRE(find_dir.cd("folder").join("file5.ext1").touch());
    { // search ext2
      auto f = FileSystem::find("find_dir", ".*\.ext2", find_options::sort);
      REQUIRE(f.size() == 5);
      for (int i = 0; i < 5; ++i)
        REQUIRE(f[i].name() == (Str() << "file" << i << ".ext2"));
    }
    { // search ext1 rec
      auto f = FileSystem::find("find_dir", ".*\.ext1", find_options::sort | find_options::recursive);
      REQUIRE(f.size() == 6);
      for (int i = 0; i < 6; ++i)
        REQUIRE(f[i].name() == (Str() << "file" << i << ".ext1"));
    }
  }//
  SECTION("lines") {
    FileSystem::writeLine("lines_file", "line1");
    auto lines = FileSystem::readLines("lines_file");
    REQUIRE(lines.size() == 1);
    REQUIRE(lines[0] == "line1");
    for (int i = 2; i <= 10; ++i)
      FileSystem::appendLine("lines_file", "line" + std::to_string(i));
    lines = FileSystem::readLines("lines_file");
    REQUIRE(lines.size() == 10);
    for (int i = 2; i < 10; ++i)
      REQUIRE(lines[i] == "line" + std::to_string(i + 1));
  }
}

TEST_CASE("size", "[common]") {}

TEST_CASE("index", "[common]") {
  SECTION("Index2 arithmetic") {
    index2 ij(1, -3);
    REQUIRE(ij + index2(-7, 10) == index2(-6, 7));
    REQUIRE(ij - index2(-7, 10) == index2(8, -13));
    REQUIRE(ij + size2(7, 10) == index2(8, 7));
    REQUIRE(ij - size2(7, 10) == index2(-6, -13));
    REQUIRE(size2(7, 10) + ij == index2(8, 7));
    REQUIRE(size2(7, 10) - ij == index2(6, 13));
  }//
  SECTION("Index2") {
    index2 a;
    index2 b;
    REQUIRE(a == b);
    b.j = 1;
    REQUIRE(a != b);
  }//
  SECTION("index2 friends") {
    index2 a(-1, 2);
    index2 b(4, 1);
    REQUIRE(max(a, b) == index2(4, 2));
    REQUIRE(min(a, b) == index2(-1, 1));
  }//
  SECTION("Index2Range") {
    range2 r({1, 1}, {3, 3});
    REQUIRE(r.contains({1, 1}));
    REQUIRE(r.contains({1, 2}));
    REQUIRE(!r.contains({0, 0}));
    REQUIRE(!r.contains({3, 3}));
    REQUIRE(r.area() == 4);

    int cur = 0;
    range2 range(10, 10);
    for (auto index : range) {
      REQUIRE(cur % 10 == index.i);
      REQUIRE(cur / 10 == index.j);
      REQUIRE(range.contains(index));
      cur++;
    }
    REQUIRE(cur == 10 * 10);
    SECTION("intersection") {
      range2 a({0, 0}, {10, 10});
      range2 b({5, -5}, {7, 70});
      REQUIRE(intersect(a, b) == range2({5, 0}, {7, 10}));
    }//
  }//
  SECTION("Index3 arithmetic") {
    index3 ij(1, -3, 0);
    REQUIRE(ij + index3(-7, 10, 1) == index3(-6, 7, 1));
    REQUIRE(ij - index3(-7, 10, 1) == index3(8, -13, -1));
    REQUIRE(ij + size3(7, 10, 3) == index3(8, 7, 3));
    REQUIRE(ij - size3(7, 10, 3) == index3(-6, -13, -3));
    REQUIRE(size3(7, 10, 5) + ij == index3(8, 7, 5));
    REQUIRE(size3(7, 10, 5) - ij == index3(6, 13, 5));
  }//
  SECTION("Index3") {
    index3 a;
    index3 b;
    REQUIRE(a == b);
    b.j = 1;
    REQUIRE(a != b);
  }//
  SECTION("Index3Range") {
    int cur = 0;
    for (auto index : Index3Range<i32>(10, 10, 10)) {
      REQUIRE((cur % 100) % 10 == index.i);
      REQUIRE((cur % 100) / 10 == index.j);
      REQUIRE(cur / 100 == index.k);
      cur++;
    }
    REQUIRE(cur == 10 * 10 * 10);
    Index3Range<i32> range(index3(-5, -5, -5), index3(5, 5, 5));
    REQUIRE(range.size().total() == 10 * 10 * 10);
    cur = 0;
    for (auto index : range) {
      REQUIRE((cur % 100) % 10 - 5 == index.i);
      REQUIRE((cur % 100) / 10 - 5 == index.j);
      REQUIRE(cur / 100 - 5 == index.k);
      cur++;
    }
    REQUIRE(cur == 10 * 10 * 10);
  }//
}

#ifdef HERMES_DEVICE_ENABLED
HERMES_CUDA_KERNEL(check_thread_index)(int bounds, int *result) {
  HERMES_CUDA_THREAD_INDEX_I_LT(bounds)
  if (i >= bounds)
    *result = -1;
}
HERMES_CUDA_KERNEL(check_thread_index2)(size2 bounds, int *result) {
  HERMES_CUDA_THREAD_INDEX_IJ_LT(bounds)
  if (ij >= bounds)
    *result = -1;
}
HERMES_CUDA_KERNEL(check_thread_index3)(size3 bounds, int *result) {
  HERMES_CUDA_THREAD_INDEX_IJK_LT(bounds)
  if (ijk >= bounds)
    *result = -1;
}
TEST_CASE("cuda utils", "[cuda]") {
  SECTION("LaunchInfo") {
    cuda_utils::LaunchInfo info(1024);
    REQUIRE(info.threadCount() == 1024);
    REQUIRE(info.grid_size.x == 1);
    REQUIRE(info.grid_size.y == 1);
    REQUIRE(info.grid_size.z == 1);
    cuda_utils::LaunchInfo info2(2000);
    REQUIRE(info2.grid_size.x == 8);
    cuda_utils::LaunchInfo info3(2048);
    REQUIRE(info3.grid_size.x == 2);
  }//
  SECTION("LaunchInfo2") {
    cuda_utils::LaunchInfo info(size2(1024, 128));
    HERMES_LOG_VARIABLE(info)
    cuda_utils::LaunchInfo info2(size2(32, 32));
    HERMES_LOG_VARIABLE(info2)
  }//
  SECTION("LaunchInfo3") {
    cuda_utils::LaunchInfo info(size3(1024, 128, 4));
    HERMES_LOG_VARIABLE(info)
    cuda_utils::LaunchInfo info2(size3(32, 32,32));
    HERMES_LOG_VARIABLE(info2)
  }//
  UnifiedArray<int> results(1);
  results[0] = 0;
  HERMES_CUDA_LAUNCH_AND_SYNC((128), check_thread_index_k, 100, results.data())
  REQUIRE(results[0] == 0);
  results[0] = 0;
  HERMES_CUDA_LAUNCH_AND_SYNC((size2(128, 128)), check_thread_index2_k, { 100, 100 }, results.data())
  REQUIRE(results[0] == 0);
  results[0] = 0;
  HERMES_CUDA_LAUNCH_AND_SYNC((size3(128, 128, 128)), check_thread_index3_k, { 100, 100, 100 }, results.data())
  REQUIRE(results[0] == 0);
}

#endif
