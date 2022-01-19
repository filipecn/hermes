# Common Utilities

\tableofcontents

Welcome, here I will present the very basic tools that `hermes`
provides to help you during development. The classes and
data types listed here are used throughout the entire library and 
I hope you could use them in your code too.

Before anything, I would like to list here the very primitive
types that you will find in `hermes/common/defs.h`:

| name | description |
|--------------|-------|
| `i8`,`i16`,`i32`, and`i64`  | signed integers, respectively `int8_t`, `int16_t`, `int32_t`, and `int64_t`|
| `u8`,`u16`,`u32`, and`u64`  | unsigned integers, respectively `uint8_t`, `uint16_t`, `uint32_t`, and `uint64_t`|
| `byte`  | 8-bit unsigned (`uint8_t`)|
| `f32` and `f64`  | floating-point types, respectively `float` and `double`|

Sometimes you want to somehow, work with the types in
unusual ways. The `hermes::DataTypes` is an auxiliary 
namespace that provides 
some functions that convert between the declared types, and
the enum class `hermes::DataType`, that holds labels for
each of the types above: 
```cpp
#include <hermes/common/defs.h>

int main() {
  // extract class from type
  auto u8_type = hermes::DataTypes::typeFrom<u8>();
  // check type size
  hermes::DataTypes::typeSize(u8_type);
  // get type name (string)
  hermes::DataTypes::typeName(u8_type);
  return 0;
}
```

## Logging
Hermes provides 4 streams of log messages: `info`, `warn`, `error` and
`critical`. In practice, they just have different console colors
(although in the future I may actually add some features here). The
class taking care of logging is `hermes::Log`, but the easiest and
recommended way to perform logging is though the macros.

```cpp
HERMES_LOG(FMT, ...);
HERMES_LOG_WARNING(FMT, ...);
HERMES_LOG_ERROR(FMT, ...);
HERMES_LOG_CRITICAL(FMT, ...);
```

`FMT` is your logging message that can be formatted to include variable
values - just like `printf`. However, it does not follow `printf`'s
directives. `hermes::Log` functions format strings in a more simple way
assuming that every argument will accept `std::stringstream` << operator
(i.e., your variables must work with `std::cout` for example). To put
an argument inside your message, you use `{}`, like this:
```cpp
HERMES_LOG_ERROR("{} errors in {}", 3, "foo");
// this will log the message
// "3 errors in foo"
```

By using the macros above, Hermes will prefix your messages with a label
containig file location, function name, line number, time and log stream.
So the message above would probably appear like this:
```text
[2022-01-18 14:23:26] [error] [../my_code.cpp][35][foo] 3 errors in foo
```
You can customize the label and the colors by configuring `hermes::Log`
variables and options:
```cpp
#include <hermes/common/debug.h>
int main() {
  // tell hermes to abbreviate file path locations and
  // output colored messages
  hermes::Log::addOptions(hermes::log_options::abbreviate |
                          hermes::log_options::use_colors);
  // choose warn message colors
  hermes::Log::warn_color = 123; // value in [0,255]
  HERMES_LOG_WARNING("warning message!");
  return 0;
}
```
> You can choose among 256 colors, commonly used in terminals. You can consult them [here](https://misc.flogisoft.com/bash/tip_colors_and_formatting) in the _88/256 Colors_ section.

Sometimes you just want to log a code location while debugging to check if the
coding is getting there, or simply log variables. Here is what you can do in
those situations:
```cpp
// log just the code location
HERMES_PING
// log a variable like: variable_name = value
HERMES_LOG_VARIABLE(variable_name);
// log multiple variable values in the same line
HERMES_LOG_VARIABLES(...);
```

In case of logging in a `CUDA` code, you will not be able to use any of the
macros above, you will have to use `printf`. The following macros do that
for you:
```cpp
// outputs to stdout
HERMES_C_LOG(FMT,...);
// outputs to stderr
HERMES_C_LOG_ERROR(FMT,...);
// for convenience, the following macros do the same
// outputs to stdout
HERMES_C_DEVICE_LOG(FMT,...);
// outputs to stderr
HERMES_C_DEVICE_ERROR(FMT,...);
```
> Note that in `FMT` now follows `printf` format options!

Finally, you may also intercept the log output as well. `hermes::Log`
allows you to register callbacks to intercept log messages:
```cpp
#include <hermes/common/debug.h>
int main() {
  // register a warn stream callback
  hermes::Log::warn_callback = [](const hermes::Str& message) {
    // handle message
  };
  // you can also fully redirect messages to your callbacks this way
  hermes::log::addOptions(hermes::log_options::callback_only);
  HERMES_LOG_WARNING("this message will not appear in console!");
  return 0;
}
```

## Debugging
Assertions and checks can be done by using the following macros:
```cpp
#include <hermes/common/debug.h>

// warns if expr is false
HERMES_CHECK_EXP(expr);
// warns with message M if expr is false
HERMES_CHECK_EXP_WITH_LOG(expr, M);
// errors if expr is false
HERMES_ASSERT(expr);
// errors with message M if expr is false
HERMES_ASSERT_WITH_LOG(expr, M);
```
Sometimes you want some piece of code to be compiled only in debug mode,
this macro can be convenient in this situation:
```cpp
HERMES_DEBUG_CODE(CODE_CONTENT)
```
Then use this way:
```cpp
#include <hermes/common/debug.h>

int main() {
  HERMES_DEBUG_CODE(
      int a = 3;
      printf("%d", a);
      )
  return 0;
}
```
both lines will not be included in release.

## Profiling
Hermes provides a profiling tool to track your code performance,
the hermes::profiler::Profiler singleton class. It works by
registering blocks that represent execution time of code sections,
scopes and functions. Each block receives a name (and a color if you want),
so you can analyze your data later.

You can profile your code by using a set of macros
like this:

```cpp
#include <hermes/common/profiler.h>
// suppose you want to profile the following function
void profiled_function() {
    // register a block taking the function's name as the label
    // the block is automatically finished after leaving this function
    HERMES_PROFILE_FUNCTION()
    // some code
    {
        // register a block with the label "code scope"
        // the block is automatically finished after leaving this function
        HERMES_PROFILE_SCOPE("code scope")
    }
    // you can also initiate and finish a block manually
    HERMES_PROFILE_START_BLOCK("my block")
    // some code
    // finish "my block" (always remember do finish your custom blocks!)
    HERMES_PROFILE_END_BLOCK
}

int main() {
  profiled_function();
  return 0;
}
```
> The profiler uses with a simple stack to manage block creation and completion.
> So remember to finish blocks consistently.

You can access the history of blocks as well:
```cpp
using hermes::profiler;
Profiler::iterateBlocks([](const Profiler::Block &block {
      auto block_desc = Profiler::blockDescriptor(block);
      // block name
      block_desc.name;
      // block start time
      block.begin();
      // block duration
      block.duration();
    }));
```
Sometimes you don't want to store all blocks created since the start of your
program, maybe to save memory. You can limit the profiler to keep only the
last `n` blocks by calling:
```cpp
hermes::profiler::Profiler::setMaxBlockCount(n);
```
You can also enable or disable the profiler in runtime with the following
macros, respectively:
```cpp
HERMES_PROFILE_ENABLE
HERMES_PROFILE_DISABLE
```
Sometimes, it is also useful to set colors for your blocks. The block label
struct holds a field `u32 color;` for that purpose. You can encode your color
in this unsigned integer the way you prefer, but `hermes` provide a namespace
containing `u32` colors for you called [hermes::argb_colors](). You can
set the block's color with the same profiling macros:

```cpp
HERMES_PROFILE_FUNCTION(hermes::argb_colors::GreenA200);
HERMES_PROFILE_SCOPE("my scoped block", hermes::argb_colors::BlueA200);
HERMES_PROFILE_START_BLOCK("my custom block", hermes::argb_colors::Coral);
```

## Parsing Arguments
Reading command line arguments or parsing command strings can be done
with hermes::ArgParser:
```cpp
#include <hermes/common/arg_parser.h>

int main(int argc, char** argv) {
    hermes::ArgParser parser("my program", "description");
    // define a simple float argument
    parser.addArgument("--float_argument", "description");
    // an required argument
    parser.addArgument("--int_argument", "argument description", true);
    // parse arguments
    parser.parse(argc, argv);
    // access argument value with default value
    parser.get<int>("--int_argument", 0);
    // check if argument was given
    if(parser.check("--float_argument"))
      HERMES_LOG_VARIABLE(parser.get<float>("--float_argument"));
    return 0;
}
```
For the code above, you code pass arguments like this:
```shell
./a.out --int_argument 3 --float_argument 2.0
```
It works by parsing all arguments, in order, pairing tokens separated
by spaces. So you don't need to explicitly put the names of the arguments
if you don't want to:
```shell
./a.out 3 --float_argument 2.0
```
In that case, `3` will be parsed and considered to be the value of your `--int_argument`,
because the parser will **follow the addArgument order**.

## Indices and Sizes
A common task is the iteration over multi-dimensional arrays and index ranges.
Hermes offers 2-dimensional and 3-dimensional index and size operations that
may facilitate these tasks with the following types:
```cpp
// integer based indices
hermes::index2; // (i,j)
hermes::index3; // (i,j,k)
// unsigned integer based sizes
hermes::size2; // (width, height)
hermes::size3; // (width, height, depth)
```
You can do all sort of arithmetic operations between them. But you can also
work with range of indices as well:
```cpp
// ranges represent a half-open interval [lower, upper)
hermes::range2; 
hermes::range3; 
```
which can be useful when you want to iterate over such type of indices:
```cpp
#include <hermes/common/index.h>
int main() {
  // let's iterate over all indices in [(0,0), (9,9)]
  hermes::size2 size(10, 10);
  for(auto index : hermes::range2(size)) {
    // access index coordinates as
    index.i;
    index.j;
  }
  return 0;
}
```

## Strings
Some functions of strings can be found in `hermes::Str`. Here are some examples
of what you can do:
```cpp
#include <hermes/common/str.h>

int main() {
  // strip string sides
  hermes::Str::strip(" asd \n", " \n"); // "asd"
  // split into substrings
  hermes::Str::split("1,2,3", ","); // {"1","2","3"}
  // join strings
  hermes::Str::join({"a","b","c"}, ","); // "a,b,c"
  // concatenate
  hermes::Str::concat("a", " ", 2); // "a 2"
  // format
  hermes::Str::format("{} has {} letters", "word", 4); // "word has 4 letters
  // numbers
  hermes::Str::isInteger("-234235"); // true
  hermes::Str::isNumber("-2e5"); // true
  hermes::Str::binaryToHex(10); // "0x1010"
  hermes::Str::addressOf(ptr); // ptr address value "0xFF..."
  // regex match
  hermes::Str::match_r("subsequence123". "\\b(sub)([^ ]*)"); // true
  // regex contains
  hermes::Str::contains_r("subsequence123", "\\b(sub)"); // true
  // and more ...
  return 0;
}
```

An instance of `hermes::Str` is just a `std::string` wrapper with an `<<` operator. 
So you can do stuff like:
```cpp
hermes::Str s;
s = s << "bla";
```

## Files
The `hermes::Filesystem` provides some _shell_-like functions that can help you
with files and directories. A auxiliary class is the `hermes::Path`,
which holds a filesystem path for a file or directory.

With `hermes::Path` in hands, you can ask all sorts of things like file extension,
directory name, absolute path, etc. You can also test if your file or directory
exists and create it if necessary. You can also construct your path like this:
```cpp
#include <hermes/common/filesystem.h>

int main() {
  hermes::Path parent("parent_folder");
  auto child = parent / "child";
  // now child is the path "parent/child"
  // you can create this path if necessary
  if(!child.exists())
    child.mkdir();
  // the same goes with a file
  hermes::Path file("parent_folder/file.txt");
  if(!file.exists())
    file.touch();
  // you could, for example check the file extension
  HERMES_LOG(file.extension());
  // the file name
  HERMES_LOG(file.name());
  // you can also check what type of path you have
  child.isDirectory();
  file.isFile();
  // in the case of the file, you can read 
  auto content = file.read();
  // and write
  file.writeTo("files content");
  return 0;
}
```

Sometimes you want to iterate over directories and files, find or copy. Here are
examples of how you can do:
```cpp
#include <hermes/common/filesystem.h>

int main() {
  // copy files
  hermes::Filesystem::copyFile("source/path", "destination/path");
  // find files recursively using regular expressions
  // also sort the results
  hermes::Filesystem::find("root/path", 
                           "*.cpp", // look for cpp files
                           hermes::find_options::recursive |
                           hermes::find_options::sort);
  // list only files in a directory, recursively
  hermes::FileSystem::ls("root/path", 
                         hermes::ls_options::files |
                         hermes::ls_options::recursive);
  // list a directory, putting directories first
  hermes::FileSystem::ls("root/path", 
                         hermes::ls_options::group_directories_first);
  return 0;
}
```

## CUDA
Hermes defines some macros to help you distinguish device code from
host code. 
```cpp
#define HERMES_HOST_FUNCTION __host__
#define HERMES_DEVICE_CALLABLE __device__ __host__
#define HERMES_DEVICE_FUNCTION __device__
// you can check if there is CUDA support with:
#define HERMES_DEVICE_ENABLED
// Wraps a block of code that gets compiled only when using CUDA
#define HERMES_CUDA_CODE(CODE) {CODE}
```
> When you build `hermes` without `CUDA` support, all macros listed in this section get empty or are simply not defined.

Here is how you can declare a `CUDA` kernel:
```c++
// this macro creates a kernel called my_kernel_k
HERMES_CUDA_KERNEL(my_kernel)(int int_argument) {
  // kernel code   
}
```
> Note that the macro appends `_k` to your kernel's name

Usually your kernel will use thread indices that can be 1-dimensional, 2-dimensional or 3-dimensional
depending on you launch configurations. Here are some macros that create those indices for you:
```c++
// creates a u32 i containing the thread index
HERMES_CUDA_THREAD_INDEX_I;
// creates a hermes::index2 ij containing the thread index
HERMES_CUDA_THREAD_INDEX_IJ;
// creates a hermes::index3 ijk containing the thread index
HERMES_CUDA_THREAD_INDEX_IK;  
// sometimes you may want the thread to execute only if its index is less then a size:
HERMES_CUDA_THREAD_INDEX_I_LT(BOUNDS);
HERMES_CUDA_THREAD_INDEX_IJ_LT(BOUNDS);
HERMES_CUDA_THREAD_INDEX_IJK_LT(BOUNDS);
// if you want to define the index variable name, use:
HERMES_CUDA_THREAD_INDEX_LT(I, BOUNDS);
HERMES_CUDA_THREAD_INDEX2_LT(IJ, BOUNDS);
HERMES_CUDA_THREAD_INDEX3_LT(IJK, BOUNDS);
```
For debugging purposes, you may want to quickly make the first thread the only thread to execute,
then you can use:
```c++
HERMES_CUDA_RETURN_IF_NOT_THREAD_0
```
The most important thing you want to do is to check for errors, `hermes` lets you use:
```c++
// check CUDA function returns
HERMES_CHECK_CUDA_CALL(err);
// for functions that do not return error codes or for kernel launches
// call it right after:
HERMES_CHECK_LAST_CUDA_CALL 
```
When launching kernels, `hermes::cuda_utils::LaunchInfo` holds launch information such
as number of threads, blocks, shared memory size and stream. It also 
redistributes threads for you trying to optimize occupancy. The following macro
can be used to launch kernels:
```c++
HERMES_CUDA_LAUNCH_AND_SYNC(LAUNCH_INFO, NAME, ...);
```
In this case, LAUNCH_INFO is the constructor parameters, surrounded by `()`, of `hermes::cuda_utils::LaunchInfo`.
Here is a complete example:
```c++
#include <vector>
#include <hermes/common/cuda_utils.h>

// A kernel that stores in c, the sum of a and b
// All vectors have n elements
HERMES_CUDA_KERNEL(sum)(size_t n, float* a, float* b, float* c) {
  HERMES_CUDA_THREAD_INDEX_I_LT(n);
  c[i] = a[i] + b[i];
}

int main() {
  // lets create 3 arrays of size n
  size_t n = 1000;
  std::vector<float> host_a(n), host_b(n), host_c(n, 0);
  // fill a and b with numbers
  // ...
  // lets now allocate memory in device
  float *device_a, *device_b, *device_c;
  HERMES_CHECK_CUDA_CALL(cudaMalloc(&device_a, n * sizeof(float)));
  HERMES_CHECK_CUDA_CALL(cudaMalloc(&device_b, n * sizeof(float)));
  HERMES_CHECK_CUDA_CALL(cudaMalloc(&device_c, n * sizeof(float)));
  // now lets send the data to device
  HERMES_CHECK_CUDA_CALL(cudaMemcpy(device_a, &host_a[0], n * sizeof(float), cudaMemcpyHostToDevice));
  HERMES_CHECK_CUDA_CALL(cudaMemcpy(device_b, &host_b[0], n * sizeof(float), cudaMemcpyHostToDevice));
  HERMES_CHECK_CUDA_CALL(cudaMemcpy(device_c, &host_c[0], n * sizeof(float), cudaMemcpyHostToDevice));
  // and finally call the kernel
  HERMES_CUDA_LAUNCH_AND_SYNC((n), sum_k, n, device_a, device_b, device_c);
  return 0;
}
```
> Please check [storage classes](4_storage.md) for device arrays.

