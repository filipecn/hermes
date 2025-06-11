/// Copyright (c) 2025, FilipeCN.
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
///\file system_tests.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2025-06-09
///
///\brief

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <hermes/system/file_system.h>
#include <hermes/system/threads.h>

#include <filesystem>

using namespace hermes;

TEST_CASE("FileSystem", "[system]") {
  SECTION("basename") {
    REQUIRE(FileSystem::basename("/usr/local/file.ext") == "file.ext");
    REQUIRE(FileSystem::basename("/usr/local/file.ext", ".ext") == "file");
    REQUIRE(FileSystem::basename("file.ext", ".ext") == "file");
    REQUIRE(FileSystem::basename("/usr/file.ex", ".ext") == "file.ex");
    REQUIRE(FileSystem::basename("/usr/").empty());
    REQUIRE(FileSystem::basename("/usr/", ".ext").empty());
  } //
  SECTION("basenames") {
    std::vector<std::string> paths = {"/usr/local/file.ext", "file.ext",
                                      "/usr/file.ex", "/usr/.ext", "/usr/"};
    std::vector<std::string> expected = {"file", "file", "file.ex", "", ""};
    auto basenames = FileSystem::basename(paths, ".ext");
    for (u64 i = 0; i < basenames.size(); ++i)
      REQUIRE(basenames[i] == expected[i]);
  } //
  SECTION("file extension") {
    REQUIRE(std::filesystem::path("path/to/file.ext4").extension() == ".ext4");
    REQUIRE(std::filesystem::path("path/to/file").extension().empty());
  } //
  SECTION("read invalid file") {
    REQUIRE(!std::filesystem::exists("invalid__file"));
    REQUIRE(FileSystem::readFile("invalid___file").empty());
    REQUIRE(FileSystem::readBinaryFile("invalid___file").empty());
  } //
  SECTION("isFile and isDirectory") {
    REQUIRE(FileSystem::writeFile("filesystem_test_file.txt", "test") == 4);
    REQUIRE(std::filesystem::is_regular_file("filesystem_test_file.txt"));
    REQUIRE(FileSystem::mkdir("path/to/dir"));
    REQUIRE(std::filesystem::is_directory("path/to/dir"));
    std::filesystem::remove_all("filesystem_test_file.txt");
    std::filesystem::remove_all("path");
  } //
  SECTION("copy file") {
    REQUIRE(FileSystem::writeFile("source", "source_content") > 0);
    std::filesystem::copy("source", "destination");
    REQUIRE(std::filesystem::exists("destination"));
    REQUIRE(FileSystem::readFile("destination") == "source_content");
    std::filesystem::remove_all("source");
    std::filesystem::remove_all("destination");
  } //
  SECTION("append") {
    REQUIRE(FileSystem::writeFile("append_test", "") == 0);
    REQUIRE(std::filesystem::exists("append_test"));
    REQUIRE(FileSystem::readFile("append_test").empty());
    REQUIRE(FileSystem::appendToFile("append_test", "append_content"));
    REQUIRE(FileSystem::readFile("append_test") == "append_content");
    REQUIRE(FileSystem::appendToFile("append_test", "123"));
    REQUIRE(FileSystem::readFile("append_test") == "append_content123");
    std::filesystem::remove_all("append_test");
  } //
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
      std::sort(ls.begin(), ls.end());
      //,
      //[](const std::filesystem::path &a, const std::filesystem::path &b) {
      //  return a.name() < b.name();
      //});
      std::vector<std::string> expected = {"file1", "file2", "file3"};
      REQUIRE(ls.size() == expected.size());
      for (u64 i = 0; i < ls.size(); ++i)
        REQUIRE(ls[i].filename() == expected[i]);
    }
    {
      auto ls =
          FileSystem::ls("ls_folder", ls_options::recursive |
                                          ls_options::files | ls_options::sort);
      std::vector<std::string> expected = {"file4", "file1", "file2", "file3",
                                           "file2", "file3", "file1"};
      REQUIRE(ls.size() == expected.size());
      for (u64 i = 0; i < ls.size(); ++i)
        REQUIRE(ls[i].filename() == expected[i]);
    }
    {
      auto ls = FileSystem::ls("ls_folder", ls_options::recursive |
                                                ls_options::files |
                                                ls_options::reverse_sort);
      std::vector<std::string> expected = {"file1", "file3", "file2", "file3",
                                           "file2", "file1", "file4"};
      REQUIRE(ls.size() == expected.size());
      for (u64 i = 0; i < ls.size(); ++i)
        REQUIRE(ls[i].filename() == expected[i]);
    }
    {
      auto ls = FileSystem::ls(
          "ls_folder", ls_options::sort | ls_options::group_directories_first);
      std::vector<std::string> expected = {"folder", "folder2", "file4"};
      REQUIRE(ls.size() == expected.size());
      for (u64 i = 0; i < ls.size(); ++i)
        REQUIRE(ls[i].filename() == expected[i]);
    }
    std::filesystem::remove_all("ls_folder");
  } //
  SECTION("filter") {
    REQUIRE(FileSystem::mkdir("find_dir"));
    std::filesystem::path find_dir("find_dir");
    for (int i = 0; i < 5; i++)
      REQUIRE(FileSystem::touch(find_dir / (Str() << "file" << i << ".ext1")));
    for (int i = 0; i < 5; i++)
      REQUIRE(FileSystem::touch(find_dir / (Str() << "file" << i << ".ext2")));
    REQUIRE(FileSystem::mkdir("find_dir/folder"));
    REQUIRE(
        FileSystem::touch(FileSystem::cd(find_dir, "folder") / "file5.ext1"));
    { // search ext2
      auto f = FileSystem::find("find_dir", ".*\.ext2", find_options::sort);
      REQUIRE(f.size() == 5);
      for (int i = 0; i < 5; ++i)
        REQUIRE(f[i].filename().c_str() == (Str() << "file" << i << ".ext2"));
    }
    { // search ext1 rec
      auto f = FileSystem::find("find_dir", ".*\.ext1",
                                find_options::sort | find_options::recursive);
      REQUIRE(f.size() == 6);
      for (int i = 0; i < 6; ++i)
        REQUIRE(f[i].filename().c_str() == (Str() << "file" << i << ".ext1"));
    }
    std::filesystem::remove_all("find_dir");
  } //
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
    std::filesystem::remove_all("lines_file");
  } //
}

int task(int a, int b) {
  using namespace std::chrono_literals;
  std::this_thread::sleep_for(100ms);
  return a + b;
}

TEST_CASE("ThreadPool", "[system]") {
  hermes::ThreadPool pool(5);
  std::vector<std::future<int>> r;
  for (int i = 0; i < 10; ++i) {
    r.emplace_back(pool.enqueue(hermes::Task::Priority::NORMAL, task, i, i));
  }
  pool.wait();
  for (int i = 0; i < 10; ++i) {
    REQUIRE(r[i].get() == i + i);
  }
}
