/// Copyright (c) 2020, FilipeCN.
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
///\file file_system.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-10-07
///
///\brief Filesystem utils
///
///\ingroup common
///\addtogroup common
/// @{

#pragma once

#include <hermes/common/bitmask_operators.h>
#include <hermes/common/str.h>
#include <hermes/core/types.h>

#include <filesystem>
#include <string>
#include <vector>

namespace hermes {

// *****************************************************************************
//                                                                 ls_options
// *****************************************************************************
/// \brief List of options for ls the method
enum class ls_options {
  none = 0x0,                     //!< default behaviour
  sort = 0x1,                     //!< sorts results in lexicographical order
  reverse_sort = 0x2,             //!< sorts in reverse order
  directories = 0x4,              //!< list only directories
  files = 0x8,                    //!< list only files
  group_directories_first = 0x10, //!< list directories first
  recursive = 0x20,               //!< list recursively
};
HERMES_ENABLE_BITMASK_OPERATORS(ls_options);
// *****************************************************************************
//                                                               find_options
// *****************************************************************************
/// \brief list of options for find the method
enum class find_options {
  none = 0x0,      //!< default behaviour
  recursive = 0x1, //!< searches recursively in directories
  sort = 0x2,      //!< sort results in lexicographical order
};
HERMES_ENABLE_BITMASK_OPERATORS(find_options);

// *****************************************************************************
//                                                                 FileSystem
// *****************************************************************************
/// \brief Set of useful functions to manipulate files and directories
class FileSystem {
public:
  // ***************************************************************************
  //                                                           STATIC METHODS
  // ***************************************************************************
  //                                                              path structure
  /// \brief Strips directory and suffix from filenames
  /// \param paths **[in]** {/path/to/filename1suffix,...}
  /// \param suffix **[in | optional]**
  /// \return {filename1, filename2, ...}
  static std::vector<std::string>
  basename(const std::vector<std::string> &paths,
           const std::string &suffix = "");
  /// \brief Strips directory and suffix from filename
  /// \param path **[in]** /path/to/filename<suffix>
  /// \param suffix **[in | optional]**
  /// \return filename
  static std::string basename(const std::string &path,
                              const std::string &suffix = "");
  /// \brief Fixes path separators and ".." parts
  /// \param path
  /// \param with_backslash
  /// \return
  static std::string normalizePath(const std::string &path,
                                   bool with_backslash = false);
  //                                                                   file read
  /// \brief loads contents from file
  /// \param filename **[in]** path/to/file.
  /// \param text     **[out]** receives file content.
  /// \return number of bytes successfully read.
  static u64 readFile(const char *filename, char **text);
  /// \brief loads binary content from file
  /// \param filename **[in]** path/to/file.ext
  /// \return vector of bytes read
  static std::vector<unsigned char>
  readBinaryFile(const std::filesystem::path &filename);
  /// \brief Read file's contents separated by line breaks
  /// \param path
  /// \return
  static std::vector<std::string> readLines(const std::filesystem::path &path);
  /// \brief Read ascii contents from file
  /// \param filename path/to/file.ext
  /// \return file's content
  static std::string readFile(const std::filesystem::path &filename);
  //                                                                  file write
  /// \brief Creates an empty file or access it.
  /// \param path_to_file valid file path
  /// \return **true** if success
  static bool touch(const std::filesystem::path &path_to_file);
  /// \brief Writes content to file.
  /// \param path **[in]** path/to/file.ext
  /// \param content
  /// \param is_binary **[in | default = false]** write in binary mode
  /// \return number of bytes successfully written.
  static u64 writeFile(const std::filesystem::path &path,
                       const std::vector<char> &content,
                       bool is_binary = false);
  /// \brief Writes content to file.
  /// \param path **[in]** path/to/file.ext
  /// \param content
  /// \param is_binary **[in | default = false]** write in binary mode
  /// \return number of bytes successfully written.
  static u64 writeFile(const std::filesystem::path &path,
                       const std::string &content, bool is_binary = false);
  /// \brief Writes line to path
  /// \param path
  /// \param line
  /// \param is_binary
  /// \return
  static u64 writeLine(const std::filesystem::path &path,
                       const std::string &line, bool is_binary = false);
  //                                                                 file append
  /// \brief Appends content to file.
  /// \param path **[in]** path/to/file.ext
  /// \param content
  /// \param is_binary **[in | default = false]** write in binary mode
  /// \return number of bytes successfully written.
  static u64 appendToFile(const std::filesystem::path &path,
                          const std::vector<char> &content,
                          bool is_binary = false);
  /// \brief Appends content to file.
  /// \param path **[in]** path/to/file.ext
  /// \param content
  /// \param is_binary **[in | default = false]** write in binary mode
  /// \return number of bytes successfully written.
  static u64 appendToFile(const std::filesystem::path &path,
                          const std::string &content, bool is_binary = false);
  /// \brief Appends line to file
  /// \param path
  /// \param line
  /// \param is_binary
  /// \return
  static u64 appendLine(const std::filesystem::path &path,
                        const std::string &line, bool is_binary = false);
  //                                                                 directories
  /// \brief Lists files inside a directory
  /// \param path **[in]** path/to/directory
  /// \param options **[in | ls_options::none]** options_ based on ls command:
  ///     none = the default behaviour;
  ///     sort = sort paths following lexicographical order;
  ///     reverse_sort = sort in reverse order;
  ///     directories = list only directories;
  ///     files = list only files;
  ///     group_directories_first = directories come first in sorting;
  ///     recursive = list directories contents;
  /// \return list of paths
  static std::vector<std::filesystem::path>
  ls(const std::filesystem::path &path, ls_options options = ls_options::none);
  /// \brief Recursively creates the path of directories
  /// \param path path/to/directory
  /// \return true on success success
  static bool mkdir(const std::filesystem::path &path);

  static std::filesystem::path cd(const std::filesystem::path &path,
                                  const std::filesystem::path &step);
  //                                                                      search
  /// \brief Search for files in a directory hierarchy
  /// \param path root directory
  /// \param pattern **[in | ""]** regular expression
  /// \param options **[in | find_options::none]**
  ///     none = default behaviour;
  ///     recursive = recursively search on directories bellow **path**
  /// \return
  static std::vector<std::filesystem::path>
  find(const std::filesystem::path &path, const std::string &pattern,
       find_options options = find_options::none);
};

} // namespace hermes

/// @}
