/* Copyright (c) 2020, FilipeCN.
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

///\file   os.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date   2020-10-07

#include <hermes/system/os.h>

#include <hermes/core/debug.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stack>
#include <sys/stat.h>
#include <utility>

#ifdef _WIN32
#include <direct.h>
#include <fstream>
#include <string>
#include <utility>
#include <windows.h>
// Allow use of freopen() without compilation warnings/errors.
// For use with custom allocated console window.
#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#else
#include <dirent.h>
#include <unistd.h>
#endif

namespace hermes::os {

std::string basename(const std::string &path, const std::string &suffix) {
  std::size_t found = path.find_last_of("/\\");
  std::string base_name =
      (found != std::string::npos) ? path.substr(found + 1) : path;
  if (!suffix.empty() && suffix.size() <= base_name.size() &&
      base_name.substr(base_name.size() - suffix.size()) == suffix)
    return base_name.substr(0, base_name.size() - suffix.size());
  return base_name;
}

std::vector<std::string> basename(const std::vector<std::string> &paths,
                                  const std::string &suffix) {
  std::vector<std::string> base_names;
  for (const auto &p : paths)
    base_names.emplace_back(basename(p, suffix));
  return base_names;
}

#ifdef WIN32
// TODO handle errors
int readFile(const char *filename, char **text) {
  std::ifstream file(filename);
  std::string str;
  std::string contents;
  while (std::getline(file, str)) {
    contents += str;
    contents.push_back('\n');
  }
  if (!contents.size())
    return 0;
  *text = new char[contents.size() + 1];
  std::strcpy(*text, contents.c_str());
  (*text)[contents.size()] = '\0';
  return contents.size();
}
#endif

#ifndef WIN32

u64 readFile(const char *filename, char **text) {
  u64 count_;

  int fd = open(filename, O_RDONLY);
  if (fd == -1)
    return 0;

  u64 size = (u64)(lseek(fd, 0, SEEK_END) + 1);
  close(fd);
  *text = new char[size];

  FILE *f = fopen(filename, "r");
  if (!f)
    return 0;

  fseek(f, 0, SEEK_SET);
  count_ = (int)fread(*text, 1, size, f);
  (*text)[count_] = '\0';

  if (ferror(f))
    count_ = 0;

  fclose(f);
  return count_;
}

#endif

std::vector<unsigned char> readBinaryFile(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open())
    return std::vector<unsigned char>();
  const auto size = file.tellg();
  file.seekg(0, std::ios::beg);
  auto bytes = std::vector<unsigned char>(size);
  file.read(reinterpret_cast<char *>(&bytes[0]), size);
  file.close();
  return bytes;
}

std::string readFile(const std::filesystem::path &path) {
  std::string content;
  std::ifstream file(path, std::ios::in);
  if (file.good()) {
    std::ostringstream ss;
    ss << file.rdbuf();
    content = ss.str();
    file.close();
  }
  return content;
}

std::vector<std::string> readLines(const std::filesystem::path &path) {
  std::vector<std::string> lines;
  std::ifstream file(path, std::ios::in);
  std::string line;
  while (std::getline(file, line))
    lines.emplace_back(std::move(line));
  return lines;
}

bool touch(const std::filesystem::path &path_to_file) {
  std::ofstream file(path_to_file);
  if (file.good()) {
    file.close();
    return true;
  }
  return false;
}

u64 writeFile(const std::filesystem::path &path,
              const std::vector<char> &content, bool is_binary) {
  std::ios_base::openmode flags = std::ofstream::out;
  if (is_binary)
    flags |= std::ofstream::binary;
  std::ofstream file(path, flags);
  if (file.good()) {
    file.write(content.data(), content.size());
    file.close();
    return content.size();
  }
  return 0;
}

u64 writeFile(const std::filesystem::path &path, const std::string &content,
              bool is_binary) {
  std::ios_base::openmode flags = std::ios::out;
  if (is_binary)
    flags |= std::ios::binary;
  std::ofstream file(path, flags);
  if (file.good()) {
    file << content;
    file.close();
    return content.size();
  }
  return 0;
}

u64 writeLine(const std::filesystem::path &path, const std::string &line,
              bool is_binary) {
  std::ios_base::openmode flags = std::ios::out;
  if (is_binary)
    flags |= std::ios::binary;
  std::ofstream file(path, flags);
  if (file.good()) {
    file << line << std::endl;
    file.close();
    return line.size();
  }
  return 0;
}

u64 appendToFile(const std::filesystem::path &path,
                 const std::vector<char> &content, bool is_binary) {
  auto flags = std::ios::out | std::ios::app;
  if (is_binary)
    flags |= std::ios::binary;
  std::ofstream file(path, flags);
  if (file.good()) {
    file.write(content.data(), content.size());
    file.close();
    return content.size();
  }
  return 0;
}

u64 appendToFile(const std::filesystem::path &path, const std::string &content,
                 bool is_binary) {
  auto flags = std::ios::out | std::ios::app;
  if (is_binary)
    flags |= std::ios::binary;
  std::ofstream file(path, flags);
  if (file.good()) {
    file << content;
    file.close();
    return content.size();
  }
  return 0;
}

u64 appendLine(const std::filesystem::path &path, const std::string &line,
               bool is_binary) {
  auto flags = std::ios::out | std::ios::app;
  if (is_binary)
    flags |= std::ios::binary;
  std::ofstream file(path, flags);
  if (file.good()) {
    file << line << std::endl;
    file.close();
    return line.size();
  }
  return 0;
}

std::vector<std::filesystem::path> ls(const std::filesystem::path &path,
                                      ls_options options) {
  std::vector<std::filesystem::path> l;

  if (options.contain(ls_option_bits::recursive)) {
    for (auto const &dir_entry :
         std::filesystem::recursive_directory_iterator{path}) {
      bool is_directory = std::filesystem::is_directory(dir_entry);
      if (options.contain(ls_option_bits::directories) && !is_directory)
        continue;
      if (options.contain(ls_option_bits::files) && is_directory)
        continue;
      l.emplace_back(dir_entry);
    }
  } else {
    for (auto const &dir_entry : std::filesystem::directory_iterator{path}) {
      bool is_directory = std::filesystem::is_directory(dir_entry);
      if (options.contain(ls_option_bits::directories) && !is_directory)
        continue;
      if (options.contain(ls_option_bits::files) && is_directory)
        continue;
      l.emplace_back(dir_entry);
    }
  }

  if ((options & (ls_option_bits::sort | ls_option_bits::reverse_sort |
                  ls_option_bits::group_directories_first)) !=
      ls_option_bits::none) {
    bool reverse_order = (options & ls_option_bits::reverse_sort) ==
                         ls_option_bits::reverse_sort;
    bool group_directories =
        (options & ls_option_bits::group_directories_first) ==
        ls_option_bits::group_directories_first;
    auto cmp = [&](const std::filesystem::path &a,
                   const std::filesystem::path &b) -> bool {
      if (group_directories) {
        bool a_is_directory = std::filesystem::is_directory(a);
        bool b_is_directory = std::filesystem::is_directory(b);
        if (a_is_directory && b_is_directory)
          return reverse_order ? a > b : a < b;
        return a_is_directory;
      }
      return reverse_order ? a > b : a < b;
    };
    std::sort(l.begin(), l.end(), cmp);
  }
  return l;
}

bool mkdir(const std::filesystem::path &path) {
  return std::filesystem::create_directories(path);
}

std::filesystem::path cd(const std::filesystem::path &path,
                         const std::filesystem::path &step) {
  auto separator = "/";
  auto current_path = cstr::split(path.c_str(), separator);
  std::stack<std::string> stack;
  for (const auto &s : current_path)
    stack.push(s);
  auto subpaths = cstr::split(step, separator);
  for (const auto &p : subpaths) {
    if (p == ".")
      continue;
    if (p == ".." && !stack.empty())
      stack.pop();
    else if (p != "..")
      stack.push(p);
  }
  cstr r;
  bool first = true;
  while (!stack.empty()) {
    if (first)
      r = stack.top();
    else
      r = cstr() << stack.top() << separator << r;
    first = false;
    stack.pop();
  }
  return r;
}

std::string normalizePath(const std::string &path, bool with_backslash) {
  HERMES_UNUSED_VARIABLE(with_backslash);
  if (path.empty())
    return path;
  std::string result = path;
  std::replace(result.begin(), result.end(), '\\', '/');
  std::vector<std::string> tokens = cstr::split(result, "/");
  unsigned int index = 0;
  while (index < tokens.size()) {
    if ((tokens[index] == "..") && (index > 0)) {
      tokens.erase(tokens.begin() + index - 1, tokens.begin() + index + 1);
      index -= 2;
    }
    index++;
  }
  result = "";
  if (path[0] == '/')
    result = "/";
  result = result + tokens[0];
  for (unsigned int i = 1; i < tokens.size(); i++)
    result += "/" + tokens[i];
  return result;
}

std::vector<std::filesystem::path> find(const std::filesystem::path &path,
                                        const std::string &pattern,
                                        find_options options) {
  std::vector<std::filesystem::path> found;
  ls_options lso = ls_option_bits::files;
  if ((options & find_option_bits::recursive) == find_option_bits::recursive)
    lso = lso | ls_option_bits::recursive;
  if ((options & find_option_bits::sort) == find_option_bits::sort)
    lso = lso | ls_option_bits::sort;
  const auto &l = ls(path, lso);
  for (const auto &p : l)
    if (cstr::regex::contains(p.c_str(), pattern))
      found.emplace_back(p);
  return found;
}

} // namespace hermes::os
