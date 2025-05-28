#coding=utf-8

common_built_in_functions_of_python = [
    'print',
    'len',
    'type',
    'int',
    'float',
    'list',
    'tuple',
    'set',
    'dict',
    'range',
    'input',
    'sorted',
    'max',
    'min',
    'sum',
    'abs',
    'round',
    'zip',
    'enumerate',
    'open',
    "findall",
    "append",
    "re.sub",
    "re.findall",
    "re.match",
    'str.capitalize', 'str.casefold', 'str.center', 'str.count', 'str.encode',
    'str.endswith', 'str.expandtabs', 'str.find', 'str.format', 'str.format_map',
    'str.index', 'str.isalnum', 'str.isalpha', 'str.isascii', 'str.isdecimal',
    'str.isdigit', 'str.isidentifier', 'str.islower', 'str.isnumeric', 'str.isprintable',
    'str.isspace', 'str.istitle', 'str.isupper', 'str.join', 'str.ljust',
    'str.lower', 'str.lstrip', 'str.maketrans', 'str.partition', 'str.replace',
    'str.rfind', 'str.rindex', 'str.rjust', 'str.rpartition', 'str.rsplit',
    'str.rstrip', 'str.split', 'str.splitlines', 'str.startswith', 'str.strip',
    'str.swapcase', 'str.title', 'str.translate', 'str.upper', 'str.zfill',
    're.compile', 're.search', 're.match', 're.fullmatch', 're.split',
    're.findall', 're.finditer', 're.sub', 're.subn', 're.escape',
    're.error',
    'list.append', 'list.clear', 'list.copy', 'list.count', 'list.extend',
    'list.index', 'list.insert', 'list.pop', 'list.remove', 'list.reverse',
    'list.sort',
    'set.add', 'set.clear', 'set.copy', 'set.difference', 'set.difference_update',
    'set.discard', 'set.intersection', 'set.intersection_update', 'set.isdisjoint',
    'set.issubset', 'set.issuperset', 'set.pop', 'set.remove', 'set.symmetric_difference',
    'set.symmetric_difference_update', 'set.union', 'set.update',
    'dict.clear', 'dict.copy', 'dict.fromkeys', 'dict.get', 'dict.items',
    'dict.keys', 'dict.pop', 'dict.popitem', 'dict.setdefault', 'dict.update',
    'dict.values'

]

java_keywords = ["abstract", "assert", "boolean", "break", "byte", "case", "catch", "do", "double", "else", "enum",
                 "extends", "final", "finally", "float", "for", "goto", "if", "implements", "import", "instanceof",
                 "int", "interface", "long", "native", "new", "package", "private", "protected", "public", "return",
                 "short", "static", "strictfp", "super", "switch", "throws", "transient", "try", "void", "volatile",
                 "while"]

java_special_ids = ["main", "args", "Math", "System", "Random", "Byte", "Short", "Integer", "Long", "Float", "Double", "Character",
                    "Boolean", "Data", "ParseException", "SimpleDateFormat", "Calendar", "Object", "String", "StringBuffer",
                    "StringBuilder", "DateFormat", "Collection", "List", "Map", "Set", "Queue", "ArrayList", "HashSet", "HashMap"]

c_keywords = ["auto", "break", "case", "char", "const", "continue",
                 "default", "do", "double", "else", "enum", "extern",
                 "float", "for", "goto", "if", "inline", "int", "long",
                 "register", "restrict", "return", "short", "signed",
                 "sizeof", "static", "struct", "switch", "typedef",
                 "union", "unsigned", "void", "volatile", "while",
                 "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
                 "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
                 "_Thread_local", "__func__"]

c_macros = ["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX",  # <stdio.h> macro
              "FILENAME_MAX", "L_tmpnam", "SEEK_CUR", "SEEK_END", "SEEK_SET",
              "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]     # <stdlib.h> macro

c_special_ids = ["main",  # main function
                   "stdio", "cstdio", "stdio.h",                                # <stdio.h> & <cstdio>
                   "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",     # <stdio.h> types & streams
                   "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush", # <stdio.h> functions
                   "fopen", "freopen", "setbuf", "setvbuf", "fprintf", "fscanf",
                   "printf", "scanf", "snprintf", "sprintf", "sscanf", "vprintf",
                   "vscanf", "vsnprintf", "vsprintf", "vsscanf", "fgetc", "fgets",
                   "fputc", "getc", "getchar", "putc", "putchar", "puts", "ungetc",
                   "fread", "fwrite", "fgetpos", "fseek", "fsetpos", "ftell",
                   "rewind", "clearerr", "feof", "ferror", "perror", "getline"
                   "stdlib", "cstdlib", "stdlib.h",                             # <stdlib.h> & <cstdlib>
                   "size_t", "div_t", "ldiv_t", "lldiv_t",                      # <stdlib.h> types
                   "atof", "atoi", "atol", "atoll", "strtod", "strtof", "strtold",  # <stdlib.h> functions
                   "strtol", "strtoll", "strtoul", "strtoull", "rand", "srand",
                   "aligned_alloc", "calloc", "malloc", "realloc", "free", "abort",
                   "atexit", "exit", "at_quick_exit", "_Exit", "getenv",
                   "quick_exit", "system", "bsearch", "qsort", "abs", "labs",
                   "llabs", "div", "ldiv", "lldiv", "mblen", "mbtowc", "wctomb",
                   "mbstowcs", "wcstombs",
                   "string", "cstring", "string.h",                                 # <string.h> & <cstring>
                   "memcpy", "memmove", "memchr", "memcmp", "memset", "strcat",     # <string.h> functions
                   "strncat", "strchr", "strrchr", "strcmp", "strncmp", "strcoll",
                   "strcpy", "strncpy", "strerror", "strlen", "strspn", "strcspn",
                   "strpbrk" ,"strstr", "strtok", "strxfrm",
                   "memccpy", "mempcpy", "strcat_s", "strcpy_s", "strdup",      # <string.h> extension functions
                   "strerror_r", "strlcat", "strlcpy", "strsignal", "strtok_r",
                   "iostream", "istream", "ostream", "fstream", "sstream",      # <iostream> family
                   "iomanip", "iosfwd",
                   "ios", "wios", "streamoff", "streampos", "wstreampos",       # <iostream> types
                   "streamsize", "cout", "cerr", "clog", "cin",
                   "boolalpha", "noboolalpha", "skipws", "noskipws", "showbase",    # <iostream> manipulators
                   "noshowbase", "showpoint", "noshowpoint", "showpos",
                   "noshowpos", "unitbuf", "nounitbuf", "uppercase", "nouppercase",
                   "left", "right", "internal", "dec", "oct", "hex", "fixed",
                   "scientific", "hexfloat", "defaultfloat", "width", "fill",
                   "precision", "endl", "ends", "flush", "ws", "showpoint",
                   "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",    # <math.h> functions
                   "cosh", "tanh", "exp", "sqrt", "log", "log10", "pow", "powf",
                   "ceil", "floor", "abs", "fabs", "cabs", "frexp", "ldexp",
                   "modf", "fmod", "hypot", "ldexp", "poly", "matherr","sort","length","size","std","using","vector","begin","to_string","to_lower",
                   "first", "empty", "M_PI","push_back","substr","end","second","front","stoi","find","rbegin","rend","type","set"
                   ,"INT_MAX","bitset","tie","accumulate","istringstream","static_assert","top","pair","set_intersection",
                   "toupper","insert","map","reverse","make_pair","stack","get","tuple","gcvt","isspace","greater",
                   "min","max","erase","INT_MIN","pop","getline","min_element","partial_sort_copy","unordered_map",
                   "push","isalnum","max_element","emplace_back"]

js_keywords = [
    "break", "case", "catch", "class", "const", "continue", "debugger", "default", "delete", "do",
    "else", "export", "extends", "finally", "for", "function", "if", "import", "in", "instanceof",
    "new", "return", "super", "switch", "this", "throw", "try", "typeof", "var", "void",
    "while", "with", "yield","let","const","console","log"
]

# JavaScript 未来保留字
js_future_reserved_words = [
    "enum", "await", 
    # 严格模式下的保留字
    "implements", "package", "protected", "interface", "private", "public"
]

# JavaScript 全局对象、函数和类
js_global_objects = [
    "Array", "Boolean", "Date", "Error", "Function", "JSON", "Math", "Number", "Object", "RegExp",
    "String", "Map", "Set", "WeakMap", "WeakSet", "Symbol", "Promise", "Proxy", "Reflect",
    "GlobalThis", "Infinity", "NaN", "undefined", "null", "eval", "isFinite", "isNaN",
    "parseFloat", "parseInt", "decodeURI", "decodeURIComponent", "encodeURI", "encodeURIComponent"
]

base_module = {
    'python': [
        'os',
        'sys',
        'math',
        'datetime',
        'json',
        're',  # Regular expressions
        'collections',
        'itertools',
        'random',
        'subprocess',
        'io',  # Input/Output
        'csv',
        'unittest',  # Testing
        'threading',  # Multithreading
        'multiprocessing',  # Multiprocessing
        'socket',  # Networking
        'http',  # HTTP client/server
        'urllib',  # URL handling
        'argparse',  # Command-line parsing
        'logging',  # Logging
        'pickle',  # Object serialization
    ],
    'cpp': [
        # C++标准库模块通常通过头文件引入，以下是一些常见的头文件
        '<iostream>',
        '<vector>',
        '<map>',
        '<set>',
        '<queue>',
        '<stack>',
        '<algorithm>',
        '<string>',
        '<fstream>',
        '<sstream>',
        '<iomanip>',
        '<cmath>',
        '<cstdlib>',
        '<ctime>',
        '<cstdio>',
        '<memory>',
        '<thread>',
        '<mutex>',
        '<condition_variable>',
        '<atomic>',
    ],
    'java': [
        'java.lang',
        'java.util',
        'java.io',
        'java.nio',
        'java.math',
        'java.time',
        'java.sql',
        'java.net',
        'java.awt',
        'javax.swing',
        'java.security',
        'java.text',
        'java.rmi',
        'java.beans',
        'java.applet',
        'java.desktop',
        'java.management',
        'java.xml',
        'java.scripting',
        'java.compiler',
    ],
    'javascript': [
    'fs',  # Node.js文件系统模块
    'path',  # Node.js路径模块
    'http',  # Node.js HTTP模块
    'https',  # Node.js HTTPS模块
    'os',  # Node.js操作系统模块
    'url',  # Node.js URL模块
    'querystring',  # Node.js查询字符串模块
    'events',  # Node.js事件模块
    'stream',  # Node.js流模块
    'zlib',  # Node.js压缩模块
    'crypto',  # Node.js加密模块
    'util',  # Node.js实用工具模块
    'assert',  # Node.js断言模块
    'buffer',  # Node.js缓冲区模块
    'child_process',  # Node.js子进程模块
    'cluster',  # Node.js集群模块
    'dgram',  # Node.js数据报模块
    'dns',  # Node.jsDNS模块
    'net',  # Node.js网络模块
    'readline',  # Node.js读取行模块
]
}