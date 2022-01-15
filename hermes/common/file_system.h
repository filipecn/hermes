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

#ifndef HERMES_FILE_SYSTEM_H
#define HERMES_FILE_SYSTEM_H

#include <hermes/common/defs.h>
#include <hermes/common/str.h>
#include <hermes/common/bitmask_operators.h>
#include <vector>
#include <string>
#include <ostream>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                               Path
// *********************************************************************************************************************
/// \brief Representation of a directory/file in the filesystem
class Path {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \brief Default constructor
  Path() = default;
  /// \brief Constructor from `hermes::Str`
  /// \param path
  Path(const Str &path);
  /// \brief Constructor from c string
  /// \param path
  Path(const char *const &&path);
  /// \brief Constructor from `std::string`
  /// \param path
  Path(std::string path);
  /// \brief Copy constructor
  /// \param other
  Path(const Path &other);
  /// \brief Move constructor
  Path(Path &&other) noexcept;
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                          casting
  /// \brief Casts to `std::string`
  /// \return
  explicit operator std::string() const { return path_.str(); }
  //                                                                                                       assignment
  /// \brief Copy assignment
  /// \param path
  /// \return
  Path &operator=(const Path &path);
  //                                                                                                          boolean
  /// \brief Full comparison
  /// \param b
  /// \return
  bool operator==(const Path &b) const;
  //                                                                                                       arithmetic
  /// \brief Generates copy concatenated with separator
  /// \param b
  /// \return
  Path operator+(const Str &b) const;
  /// \brief Generates copy concatenated with separator
  /// \param other
  /// \return
  Path operator/(const Path &other) const;
  /// \brief Joins with separator
  /// \param other
  /// \return
  Path &operator/=(const Path &other);
  /// \brief Joins with separator
  /// \param other
  /// \return
  Path &operator+=(const std::string &other);
  /// \brief Joins with separator
  /// \param other
  /// \return
  Path &operator+=(const Path &other);
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  /// \brief Joins with separator
  /// \param path
  /// \return
  Path &join(const Path &path);
  /// \brief Jump to path
  /// \param path
  Path &cd(const std::string &path);
  /// \brief Check if this path exists in filesystem
  /// \return
  [[nodiscard]] bool exists() const;
  /// \brief Check if this path represents a folder
  /// \return
  [[nodiscard]] bool isDirectory() const;
  /// \brief Check if this path represents a file
  /// \return
  [[nodiscard]] bool isFile() const;
  /// \brief Splits this path into a list
  /// \return
  [[nodiscard]] std::vector<std::string> parts() const;
  /// \brief Gets last folder/file name
  /// \return
  [[nodiscard]] std::string name() const;
  /// \brief Gets this full path string
  /// \return
  [[nodiscard]] const std::string &fullName() const;
  /// \brief Gets file extension
  /// \return
  [[nodiscard]] std::string extension() const;
  /// \brief Checks if this path matches pattern
  /// \param regular_expression
  /// \return
  [[nodiscard]] bool match_r(const std::string &regular_expression) const;
  /// \brief Gets this path's folder location
  /// \return
  [[nodiscard]] Path cwd() const;
  /// \brief Creates folder from this path
  /// \return
  [[nodiscard]] bool mkdir() const;
  /// \brief Creates empty file from this path
  /// \return
  [[nodiscard]] bool touch() const;
  /// \brief Writes content into this file
  /// \param content
  /// \return
  [[nodiscard]] u64 writeTo(const std::string &content) const;
  /// \brief Appends to this file
  /// \param content
  /// \return
  [[nodiscard]] u64 appendTo(const std::string &content) const;
  /// \brief Reads ascii content from this file
  /// \return
  [[nodiscard]] std::string read() const;
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  std::string separator{"/"}; //!< OS path separator
private:
  Str path_;
};

// *********************************************************************************************************************
//                                                                                                         ls_options
// *********************************************************************************************************************
/// \brief List of options for ls the method
enum class ls_options {
  none = 0x0,                                  //!< default behaviour
  sort = 0x1,                                  //!< sorts results in lexicographical order
  reverse_sort = 0x2,                          //!< sorts in reverse order
  directories = 0x4,                           //!< list only directories
  files = 0x8,                                 //!< list only files
  group_directories_first = 0x10,              //!< list directories first
  recursive = 0x20,                            //!< list recursively
};
HERMES_ENABLE_BITMASK_OPERATORS(ls_options);
// *********************************************************************************************************************
//                                                                                                       find_options
// *********************************************************************************************************************
/// \brief list of options for find the method
enum class find_options {
  none = 0x0,                //!< default behaviour
  recursive = 0x1,           //!< searches recursively in directories
  sort = 0x2,                //!< sort results in lexicographical order
};
HERMES_ENABLE_BITMASK_OPERATORS(find_options);

// *********************************************************************************************************************
//                                                                                                         FileSystem
// *********************************************************************************************************************
/// \brief Set of useful functions to manipulate files and directories
class FileSystem {
public:
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  //                                                                                                   path structure
  /// \brief Strips directory and suffix from filenames
  /// \param paths **[in]** {/path/to/filename1suffix,...}
  /// \param suffix **[in | optional]**
  /// \return {filename1, filename2, ...}
  static std::vector<std::string> basename(const std::vector<std::string> &paths, const std::string &suffix = "");
  /// \brief Strips directory and suffix from filename
  /// \param path **[in]** /path/to/filenamesuffix
  /// \param suffix **[in | optional]**
  /// \return filename
  static std::string basename(const std::string &path, const std::string &suffix = "");
  /// \brief Retrieves file's extension
  /// \param filename path/to/filename.extension
  /// \return file's extension, if any (after last '.')
  static std::string fileExtension(const std::string &filename);
  /// \brief Fixes path separators and ".." parts
  /// \param path
  /// \param with_backslash
  /// \return
  static std::string normalizePath(const std::string &path, bool with_backslash = false);
  //                                                                                                        file read
  /// \brief loads contents from file
  /// \param filename **[in]** path/to/file.
  /// \param text     **[out]** receives file content.
  /// \return number of bytes successfully read.
  static u64 readFile(const char *filename, char **text);
  /// \brief loads binary content from file
  /// \param filename **[in]** path/to/file.ext
  /// \return vector of bytes read
  static std::vector<unsigned char> readBinaryFile(const char *filename);
  /// \brief Read file's contents separated by line breaks
  /// \param path
  /// \return
  static std::vector<std::string> readLines(const Path &path);
  /// \brief Read ascii contents from file
  /// \param filename path/to/file.ext
  /// \return file's content
  static std::string readFile(const Path &filename);
  //                                                                                                       file write
  /// \brief Creates an empty file or access it.
  /// \param path_to_file valid file path
  /// \return **true** if success
  static bool touch(const Path &path_to_file);
  /// \brief Writes content to file.
  /// \param path **[in]** path/to/file.ext
  /// \param content
  /// \param is_binary **[in | default = false]** write in binary mode
  /// \return number of bytes successfully written.
  static u64 writeFile(const Path &path,
                       const std::vector<char> &content, bool is_binary = false);
  /// \brief Writes content to file.
  /// \param path **[in]** path/to/file.ext
  /// \param content
  /// \param is_binary **[in | default = false]** write in binary mode
  /// \return number of bytes successfully written.
  static u64 writeFile(const Path &path,
                       const std::string &content, bool is_binary = false);
  /// \brief Writes line to path
  /// \param path
  /// \param line
  /// \param is_binary
  /// \return
  static u64 writeLine(const Path &path, const std::string &line, bool is_binary = false);
  //                                                                                                      file append
  /// \brief Appends content to file.
  /// \param path **[in]** path/to/file.ext
  /// \param content
  /// \param is_binary **[in | default = false]** write in binary mode
  /// \return number of bytes successfully written.
  static u64 appendToFile(const Path &path,
                          const std::vector<char> &content, bool is_binary = false);
  /// \brief Appends content to file.
  /// \param path **[in]** path/to/file.ext
  /// \param content
  /// \param is_binary **[in | default = false]** write in binary mode
  /// \return number of bytes successfully written.
  static u64 appendToFile(const Path &path,
                          const std::string &content, bool is_binary = false);
  /// \brief Appends line to file
  /// \param path
  /// \param line
  /// \param is_binary
  /// \return
  static u64 appendLine(const Path &path, const std::string &line, bool is_binary = false);
  //                                                                                                          queries
  /// \brief Checks if file exists
  /// \param path /path/to/file.ext
  /// \return **true** if file exists
  static bool fileExists(const Path &path);
  /// \brief Checks if filename corresponds to a file.
  /// \param path /path/to/file.ext
  /// \return **true** if filename points to a file.
  static bool isFile(const Path &path);
  /// \brief Checks if dir_name corresponds to a directory.
  /// \param dir_name **[in]** /path/to/directory
  /// \return **true** if dir_name points to a directory.
  static bool isDirectory(const Path &dir_name);
  //                                                                                                      directories
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
  static std::vector<Path> ls(const Path &path, ls_options options = ls_options::none);
  /// \brief Recursively creates the path of directories
  /// \param path path/to/directory
  /// \return true on success success
  static bool mkdir(const Path &path);
  /// \brief Copy file's contents to destination file
  /// \param source
  /// \param destination
  /// \return
  static bool copyFile(const Path &source, const Path &destination);
  //                                                                                                           search
  /// \brief Search for files in a directory hierarchy
  /// \param path root directory
  /// \param pattern **[in | ""]** regular expression
  /// \param options **[in | find_options::none]**
  ///     none = default behaviour;
  ///     recursive = recursively search on directories bellow **path**
  /// \return
  static std::vector<Path> find(const Path &path,
                                const std::string &pattern,
                                find_options options = find_options::none);
};

// *********************************************************************************************************************
//                                                                                                                 IO
// *********************************************************************************************************************
/// \brief Path's << operator support for `std::ostream`
/// \param o
/// \param path
/// \return
std::ostream &operator<<(std::ostream &o, const Path &path);

}

#endif //HERMES_FILE_SYSTEM_H

/// @}
