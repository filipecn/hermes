#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <hermes/system/os.h>
#include <hermes/system/profile.h>
#include <hermes/system/threads.h>

#include <filesystem>

using namespace hermes;

TEST_CASE("os", "[system]") {
  SECTION("basename") {
    REQUIRE(os::basename("/usr/local/file.ext") == "file.ext");
    REQUIRE(os::basename("/usr/local/file.ext", ".ext") == "file");
    REQUIRE(os::basename("file.ext", ".ext") == "file");
    REQUIRE(os::basename("/usr/file.ex", ".ext") == "file.ex");
    REQUIRE(os::basename("/usr/").empty());
    REQUIRE(os::basename("/usr/", ".ext").empty());
  } //
  SECTION("basenames") {
    std::vector<std::string> paths = {"/usr/local/file.ext", "file.ext",
                                      "/usr/file.ex", "/usr/.ext", "/usr/"};
    std::vector<std::string> expected = {"file", "file", "file.ex", "", ""};
    auto basenames = os::basename(paths, ".ext");
    for (u64 i = 0; i < basenames.size(); ++i)
      REQUIRE(basenames[i] == expected[i]);
  } //
  SECTION("file extension") {
    REQUIRE(std::filesystem::path("path/to/file.ext4").extension() == ".ext4");
    REQUIRE(std::filesystem::path("path/to/file").extension().empty());
  } //
  SECTION("read invalid file") {
    REQUIRE(!std::filesystem::exists("invalid__file"));
    REQUIRE(os::readFile("invalid___file").empty());
    REQUIRE(os::readBinaryFile("invalid___file").empty());
  } //
  SECTION("isFile and isDirectory") {
    REQUIRE(os::writeFile("filesystem_test_file.txt", "test") == 4);
    REQUIRE(std::filesystem::is_regular_file("filesystem_test_file.txt"));
    REQUIRE(os::mkdir("path/to/dir"));
    REQUIRE(std::filesystem::is_directory("path/to/dir"));
    std::filesystem::remove_all("filesystem_test_file.txt");
    std::filesystem::remove_all("path");
  } //
  SECTION("copy file") {
    REQUIRE(os::writeFile("source", "source_content") > 0);
    std::filesystem::copy("source", "destination");
    REQUIRE(std::filesystem::exists("destination"));
    REQUIRE(os::readFile("destination") == "source_content");
    std::filesystem::remove_all("source");
    std::filesystem::remove_all("destination");
  } //
  SECTION("append") {
    REQUIRE(os::writeFile("append_test", "") == 0);
    REQUIRE(std::filesystem::exists("append_test"));
    REQUIRE(os::readFile("append_test").empty());
    REQUIRE(os::appendToFile("append_test", "append_content"));
    REQUIRE(os::readFile("append_test") == "append_content");
    REQUIRE(os::appendToFile("append_test", "123"));
    REQUIRE(os::readFile("append_test") == "append_content123");
    std::filesystem::remove_all("append_test");
  } //
  SECTION("ls") {
    REQUIRE(os::mkdir("os::ls_folder/folder"));
    REQUIRE(os::touch("os::ls_folder/file4"));
    REQUIRE(os::touch("os::ls_folder/folder/file1"));
    REQUIRE(os::touch("os::ls_folder/folder/file2"));
    REQUIRE(os::touch("os::ls_folder/folder/file3"));
    REQUIRE(os::mkdir("os::ls_folder/folder2"));
    REQUIRE(os::touch("os::ls_folder/folder2/file2"));
    REQUIRE(os::touch("os::ls_folder/folder2/file3"));
    REQUIRE(os::mkdir("os::ls_folder/folder2/folder"));
    REQUIRE(os::touch("os::ls_folder/folder2/folder/file1"));
    // os::ls_folder
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
      auto ls = os::ls("os::ls_folder/folder");
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
      auto ls = os::ls("os::ls_folder", os::ls_option_bits::recursive |
                                            os::ls_option_bits::files |
                                            os::ls_option_bits::sort);
      std::vector<std::string> expected = {"file4", "file1", "file2", "file3",
                                           "file2", "file3", "file1"};
      REQUIRE(ls.size() == expected.size());
      for (u64 i = 0; i < ls.size(); ++i)
        REQUIRE(ls[i].filename() == expected[i]);
    }
    {
      auto ls = os::ls("os::ls_folder", os::ls_option_bits::recursive |
                                            os::ls_option_bits::files |
                                            os::ls_option_bits::reverse_sort);
      std::vector<std::string> expected = {"file1", "file3", "file2", "file3",
                                           "file2", "file1", "file4"};
      REQUIRE(ls.size() == expected.size());
      for (u64 i = 0; i < ls.size(); ++i)
        REQUIRE(ls[i].filename() == expected[i]);
    }
    {
      auto ls = os::ls("os::ls_folder",
                       os::ls_option_bits::sort |
                           os::ls_option_bits::group_directories_first);
      std::vector<std::string> expected = {"folder", "folder2", "file4"};
      REQUIRE(ls.size() == expected.size());
      for (u64 i = 0; i < ls.size(); ++i)
        REQUIRE(ls[i].filename() == expected[i]);
    }
    std::filesystem::remove_all("os::ls_folder");
  } //
  SECTION("filter") {
    REQUIRE(os::mkdir("os::find_dir"));
    std::filesystem::path find_dir("os::find_dir");
    for (int i = 0; i < 5; i++)
      REQUIRE(os::touch(find_dir / (cstr() << "file" << i << ".ext1")));
    for (int i = 0; i < 5; i++)
      REQUIRE(os::touch(find_dir / (cstr() << "file" << i << ".ext2")));
    REQUIRE(os::mkdir("os::find_dir/folder"));
    REQUIRE(os::touch(os::cd(find_dir, "folder") / "file5.ext1"));
    { // search ext2
      auto f =
          os::find("os::find_dir", ".*\\.ext2", os::find_option_bits::sort);
      REQUIRE(f.size() == 5);
      for (int i = 0; i < 5; ++i)
        REQUIRE(f[i].filename().c_str() == (cstr() << "file" << i << ".ext2"));
    }
    { // search ext1 rec
      auto f = os::find("os::find_dir", ".*\\.ext1",
                        os::find_option_bits::sort |
                            os::find_option_bits::recursive);
      REQUIRE(f.size() == 5);
      for (int i = 0; i < 5; ++i)
        REQUIRE(f[i].filename().c_str() == (cstr() << "file" << i << ".ext1"));
    }
    std::filesystem::remove_all("os::find_dir");
  } //
  SECTION("lines") {
    os::writeLine("lines_file", "line1");
    auto lines = os::readLines("lines_file");
    REQUIRE(lines.size() == 1);
    REQUIRE(lines[0] == "line1");
    for (int i = 2; i <= 10; ++i)
      os::appendLine("lines_file", "line" + std::to_string(i));
    lines = os::readLines("lines_file");
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

void foo() {
  using namespace std::chrono_literals;
  HERMES_PROFILE_FUNCTION();
  hermes::SystemTime::init();
  std::this_thread::sleep_for(100ms);
  for (int j = 0; j < 2; ++j) {
    HERMES_PROFILE_SCOPE("for loop");
    std::this_thread::sleep_for(200us);
  }
}

int f(int a, int b) {
  using namespace std::chrono_literals;
  HERMES_PROFILE_FUNCTION();
  hermes::SystemTime::init();
  std::this_thread::sleep_for(100ms);
  foo();
  HERMES_WARN("a + b = {}", a + b);
  return a + b;
}

TEST_CASE("Profile", "[core]") {
  hermes::Logger::setLevel(hermes::Logger::Level::debug);
  hermes::ThreadPool pool(5);
  for (int i = 0; i < 10; ++i) {
    pool.enqueue(hermes::Task::Priority::NORMAL, f, i, i);
  }
  pool.wait();

  HERMES_INFO("{}", hermes::profile::Profiler::trace());
  HERMES_INFO("{}", hermes::profile::Profiler::report());
}
