import torch
import torch.nn as nn
import copy
import random
import sys
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import csv
python_keywords = ['import', '', '[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">",
                   '+', '-', '*', '/', 'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break',
                   'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global',
                   'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
                   'while', 'with', 'yield']
java_keywords = ["abstract", "assert", "boolean", "break", "byte", "case", "catch", "do", "double", "else", "enum",
                 "extends", "final", "finally", "float", "for", "goto", "if", "implements", "import", "instanceof",
                 "int", "interface", "long", "native", "new", "package", "private", "protected", "public", "return",
                 "short", "static", "strictfp", "super", "switch", "throws", "transient", "try", "void", "volatile",
                 "while"]
java_special_ids = ["main", "args", "Math", "System", "Random", "Byte", "Short", "Integer", "Long", "Float", "Double", "Character",
                    "Boolean", "Data", "ParseException", "SimpleDateFormat", "Calendar", "Object", "String", "StringBuffer",
                    "StringBuilder", "DateFormat", "Collection", "List", "Map", "Set", "Queue", "ArrayList", "HashSet", "HashMap"]

java_all_names = java_keywords + java_special_ids

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
                   "modf", "fmod", "hypot", "ldexp", "poly", "matherr"]

special_char = ['[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">", '+', '-', '*', '/',
                '|']

c_all_names = c_keywords + c_macros + c_special_ids + special_char

go_keywords = ['break','default','func','interface','select',
'case', 'defer','go','map','struct',
'chan','else','goto','package','switch',
'const','fallthrough','if','range','type',
'continue','for','import','return','var',"main","fmt","Println"]

go_predefined_identifiers = [
    "bool", "byte", "complex64", "complex128", "error",
    "float32", "float64", "int", "int8", "int16",
    "int32", "int64", "rune", "string", "uint",
    "uint8", "uint16", "uint32", "uint64", "uintptr",
    "true", "false", "iota",
    "nil",
    "append", "cap", "close", "complex", "copy",
    "delete", "imag", "len", "make", "new",
    "panic", "print", "println", "real", "recover"
]

go_all_names = go_keywords + go_predefined_identifiers

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

# 合并所有列表
js_all_names = js_keywords + js_future_reserved_words + js_global_objects

php_keywords = [
    "__halt_compiler", "abstract", "and", "array", "as", "break", "callable", "case", "catch", "class",
    "clone", "const", "continue", "declare", "default", "die", "do", "echo", "else", "elseif",
    "empty", "enddeclare", "endfor", "endforeach", "endif", "endswitch", "endwhile", "eval", "exit", "extends",
    "final", "finally", "fn", "for", "foreach", "function", "global", "goto", "if", "implements",
    "include", "include_once", "instanceof", "insteadof", "interface", "isset", "list", "match", "namespace", "new",
    "or", "print", "private", "protected", "public", "readonly", "require", "require_once", "return", "static",
    "switch", "throw", "trait", "try", "unset", "use", "var", "while", "xor", "yield",
    "yield from"
]

# PHP 魔术常量
php_magic_constants = [
    "__LINE__", "__FILE__", "__DIR__", "__FUNCTION__", "__CLASS__", "__TRAIT__", "__METHOD__", "__NAMESPACE__"
]

# 常用的预定义常量和类名（示例）
php_common_predefined = [
    "TRUE", "FALSE", "NULL", "PHP_VERSION", "PHP_INT_MAX", "PHP_INT_MIN", "PHP_FLOAT_MAX", "PHP_FLOAT_MIN",
    "DIRECTORY_SEPARATOR", "PATH_SEPARATOR", "STDIN", "STDOUT", "STDERR"
]

# 合并所有列表
php_all_names = php_keywords + php_magic_constants + php_common_predefined

ruby_keywords = [
    "BEGIN", "END", "__ENCODING__", "__END__", "__FILE__", "__LINE__",
    "alias", "and", "begin", "break", "case", "class", "def", "defined?",
    "do", "else", "elsif", "end", "ensure", "false", "for", "if", "in",
    "module", "next", "nil", "not", "or", "redo", "rescue", "retry", "return",
    "self", "super", "then", "true", "undef", "unless", "until", "when",
    "while", "yield"
]


rust_reserved_keywords = [
    "as", "break", "const", "continue", "crate", "else", "enum", "extern",
    "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod",
    "move", "mut", "pub", "ref", "return", "self", "Self", "static", "struct",
    "super", "trait", "true", "type", "unsafe", "use", "where", "while",
    "async", "await", "dyn"
]

rust_weak_keywords = [
    "abstract", "become", "box", "do", "final", "macro", "override", "priv",
    "typeof", "unsized", "virtual", "yield", "try"
]

rust_keywords = rust_reserved_keywords + rust_weak_keywords

csharp_keywords = [
    "abstract", "as", "base", "bool", "break", "byte", "case", "catch", "char", "checked",
    "class", "const", "continue", "decimal", "default", "delegate", "do", "double", "else",
    "enum", "event", "explicit", "extern", "false", "finally", "fixed", "float", "for",
    "foreach", "goto", "if", "implicit", "in", "int", "interface", "internal", "is", "lock",
    "long", "namespace", "new", "null", "object", "operator", "out", "override", "params",
    "private", "protected", "public", "readonly", "ref", "return", "sbyte", "sealed", "short",
    "sizeof", "stackalloc", "static", "string", "struct", "switch", "this", "throw", "true",
    "try", "typeof", "uint", "ulong", "unchecked", "unsafe", "ushort", "using", "virtual",
    "void", "volatile", "while","System","Program","Main","Console","WriteLine"
]

# C# 上下文关键字
csharp_contextual_keywords = [
    "add", "alias", "ascending", "async", "await", "by", "descending", "dynamic", "equals",
    "from", "get", "global", "group", "into", "join", "let", "nameof", "on", "orderby",
    "partial", "remove", "select", "set", "value", "var", "when", "where", "yield"
]

# 常见的预定义类型名称
csharp_predefined_types = [
    "Object", "String", "Boolean", "Byte", "SByte", "Char", "Decimal", "Double", "Single",
    "Int32", "UInt32", "Int64", "UInt64", "Int16", "UInt16", "Void"
]

# 合并列表
csharp_avoid = csharp_keywords + csharp_contextual_keywords + csharp_predefined_types


all_keywords = {'c':c_all_names, 'cpp':c_all_names,'c_sharp':csharp_avoid, 'go':go_all_names, 'java':java_all_names, 'javascript':js_all_names, 'php':php_all_names, 'python':python_keywords, 'ruby':ruby_keywords, 'rust':rust_keywords}

stmt_separators = {
    "c": ";",
    "c_sharp": ";",
    "go": "\n", # 在正式代码中通常是可选的，由编译器自动插入
    "java": ";",
    "javascript": ";", # 虽然在某些情况下可以省略，但通常用于分隔语句
    "php": ";",
    "python": "\n", # 或使用分号 (;) 来在同一行中分隔多个语句
    "ruby": "\n", # 或使用分号 (;)
    "rust": ";", # 表达式语句以分号结尾，但定义和其他一些语句不需要
}


block_separators = {
    "c": ["{", "}"],  # C语言使用大括号来分隔代码块
    "c_sharp": ["{", "}"],  # C#也使用大括号
    "go": ["{", "}"],  # Go语言使用大括号
    "java": ["{", "}"],  # Java使用大括号
    "javascript": ["{", "}"],  # JavaScript使用大括号
    "php": ["{", "}"],  # PHP使用大括号
    "python": ["\n"],  # Python使用缩进来分隔代码块，因此只有一个“符号”
    "rust": ["{", "}"],  # Rust使用大括号
    "ruby":["\t"]
}



literal_comparsion = {
  "c": {
    "normal": ['char_literal', 'compound_literal_expression', 'number_literal', 'string_literal'],
    "container": []
  },
  "c_sharp": {
    "normal": ['integer_literal', 'preproc_integer_literal', \
        'string_literal_encoding', 'verbatim_string_literal', 'real_literal', 'character_literal_unescaped', \
            'character_literal', 'string_literal', 'preproc_string_literal', 'raw_string_literal', 'null_literal', 'boolean_literal',"false","true"],
    "container": ["array_creation_expression"]
  },
  "cpp": {
    "normal": ['char_literal', 'raw_string_literal', 'compound_literal_expression', 'string_literal', 'number_literal', 'literal_suffix', 'user_defined_literal'],
    "container": ["initializer_list"]
  },
  "go": {
    "normal": ['interpreted_string_literal', 'rune_literal', 'literal_value', 'int_literal', 'raw_string_literal', 'imaginary_literal', 'func_literal', 'literal_element', 'composite_literal', 'float_literal',"true", "false", "nil"],
    "container": ["composite_literal"]
  },
  "java": {
    "normal": ['decimal_integer_literal', 'hex_integer_literal', 'character_literal', 'octal_integer_literal', 'decimal_floating_point_literal',\
        'binary_integer_literal', 'null_literal', 'string_literal', 'hex_floating_point_literal', 'class_literal', '_literal',"false","true"],
    "container": ["array_initializer"]
  },
  "javascript": {
    "normal": ["numeric_literal", "string_literal", "boolean_literal", "null","false","true"],
    "container": ["array", "object"]
  },
  "php": {
    "normal": ["integer", "float", "string", "boolean", "null"],
    "container": ["array"]
  },
  "python": {
    "normal": ["integer", "float", "imaginary", "string", "True", "False", "None"],
    "container": ["list", "tuple", "dict", "set"]
  },
  "ruby": {
    "normal": ["int", "float", "string", "symbol", "true", "false", "nil"],
    "container": ["array", "hash"]
  },
  "rust": {
    "normal": ['float_literal', 'boolean_literal', 'negative_literal', 'literal', '_literal_pattern', 'raw_string_literal', '_literal', 'string_literal', 'char_literal', 'integer_literal'],
    "container": ["array_expression", "struct_expression", "tuple_expression"]
  }
}
#这些node_type在tree_sitter中为常量类型,其中normal一定为常量,container还需判断其中的所有元素是否都为常量


all_languages_constant_variable_types = {
    "c": {
        "char_literal": ["char"],
        "compound_literal_expression": [""],  # C语言中，复合字面量的类型依赖于具体复合类型的定义，不直接书写类型
        "number_literal": ["double", "int"],  # 数字字面量默认为int，但double为更通用类型
        "string_literal": ["char[]"],
    },
    "c_sharp": {
        "integer_literal": ["int"],
        "string_literal_fragment": [""],  # 动态类型，不直接书写类型
        "preproc_integer_literal": [""],  # 预处理器中使用，不直接书写类型
        "string_literal_encoding": ["string"],
        "verbatim_string_literal": ["string"],
        "real_literal": ["double", "float"],
        "character_literal_unescaped": ["char"],
        "character_literal": ["char"],
        "string_literal": ["string"],
        "preproc_string_literal": [""],  # 预处理器中使用，不直接书写类型
        "raw_string_literal": ["string"],
        "null_literal": ["null"],
        "boolean_literal": ["bool"],
        "false": ["bool"],
        "true": ["bool"],
    },
    "go": {
        "interpreted_string_literal": ["string"],
        "rune_literal": ["rune"],
        "literal_value": [""],  # 根据上下文推断，不直接书写类型
        "int_literal": ["int"],
        "raw_string_literal": ["string"],
        "imaginary_literal": ["complex128", "complex64"],
        "func_literal": ["func"],  # 函数字面量类型由其签名决定
        "literal_element": [""],  # 根据上下文推断，不直接书写类型
        "composite_literal": [""],  # 根据上下文推断，不直接书写类型
        "float_literal": ["float64", "float32"],
        "true": ["bool"],
        "false": ["bool"],
        "nil": ["nil"],
    },
    "java": {
        "decimal_integer_literal": ["int", "long"],
        "hex_integer_literal": ["int", "long"],
        "character_literal": ["char"],
        "octal_integer_literal": ["int", "long"],
        "decimal_floating_point_literal": ["double", "float"],
        "binary_integer_literal": ["int", "long"],
        "null_literal": ["null"],
        "string_literal": ["String"],
        "hex_floating_point_literal": ["double", "float"],
        "class_literal": [""],  # 类字面量，用于反射，类型为Class
        "_literal": [""],  # 特殊情况，不直接书写类型
        "false": ["boolean"],
        "true": ["boolean"],
    },
    "javascript": {
        "numeric_literal": [""],  # JavaScript为动态类型语言，不直接书写类型
        "string_literal": [""],
        "boolean_literal": [""],
        "null": [""],
        "false": [""],
        "true": [""],
    },
    "php": {
        "integer": [""],  # PHP为动态类型语言，不直接书写类型
        "float": [""],
        "string": [""],
        "boolean": [""],
        "null": [""],
    },
    "python": {
        "integer": [""],  # Python为动态类型语言，不直接书写类型
        "float": [""],
        "imaginary": [""],
        "string": [""],
        "True": [""],
        "False": [""],
        "None": [""],
    },
    "ruby": {
        "int": [""],  # Ruby为动态类型语言，不直接书写类型
        "float": [""],
        "string": [""],
        "symbol": [""],
        "true": [""],
        "false": [""],
        "nil": [""],
    },
    "rust": {
        "float_literal": ["f64", "f32"],  # 浮点字面量，默认为f64，可通过后缀指定为f32
        "boolean_literal": ["bool"],
        "negative_literal": ["i32", "i64", "f32", "f64"],  # 负数字面量，默认为i32，可为其他整数或浮点类型
        "literal": [""],  # 根据上下文推断，不直接书写类型
        "_literal_pattern": [""],  # 用于模式匹配，不直接书写类型
        "raw_string_literal": ["&str"],  # 原生字符串字面量
        "_literal": [""],  # 特殊占位符，可能表示任何字面量，根据上下文推断
        "string_literal": ["&str"],
        "char_literal": ["char"],
        "integer_literal": ["i32", "u32", "i64", "u64"],  # 整数字面量，默认为i32，可通过后缀指定其他整数类型
    },
}

condition_stmt_name_convert_dict = {"c":{"if_statement":"if_statement","while_statement":"while_statement"}, "java":{"if_statement":"if_statement","while_statement":"while_statement"}, \
    "python":{"if_statement":"if_statement","while_statement":"while_statement"},\
        "go":{"if_statement":"if_statement","while_statement":"for_statement"}, \
            "javascript":{"if_statement":"if_statement","while_statement":"while_statement"},\
                "php":{"if_statement":"if_statement","while_statement":"while_statement"}, \
                    "ruby":{"if_statement":"if","while_statement":"while"}, \
                        "rust":{"if_statement":"if_expression","while_statement":"while_expression"}, \
                            "c_sharp":{"if_statement":"if_statement","while_statement":"while_statement"}}


IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import statistics",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
    ],
    "cpp": [
        "using namespace std;",
        "#include<stdlib.h>",
        "#include<algorithm>",
        "#include<cmath>",
        "#include<math.h>",
        "#include<numeric>",
        "#include<stdio.h>",
        "#include<vector>",
        "#include<set>",
        "#include<map>",
        "#include<queue>",
        "#include<stack>",
        "#include<list>",
        "#include<deque>",
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
        "#include<sstream>",
        "#include<fstream>",
    ],
    "java": [
        "import java.util.*;",
        "import java.util.OptionalInt;",
        "import java.util.stream.IntStream;",
        "import java.util.stream.Collectors;",
        "import java.util.regex.Matcher;",
        "import java.util.regex.Pattern;",
        "import java.util.Arrays;",
        "import java.util.ArrayList;"
    ],
    "javascript":[]
}

LANGUAGE_TO_TIMEOUT = {
    "python": 10,
    "cpp": 60,
    "javascript": 10,
    "java": 10,
    "go": 20,
    "rust": 300, # Necessary for first-time compilation of cargo
}

# Java sometimes fails with more workers; For javascript it's twice as fast with 4 workers
LANGUAGE_TO_NUM_WORKERS = {
    "python": 4,
    "cpp": 4,
    "javascript": 4,
    "java": 1,
    "go": 4,
    "rust": 1,
}


LANGUAGES = ["python", "cpp", "javascript", "java"]

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



class Recorder():
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.f = open(file_path, 'w')
        self.writer = csv.writer(self.f)
        self.writer.writerow(["Index",
                        "Original Code", 
                        "Program Length", 
                        "Ground truth", 
                        "Original Prediction", 
                        "Adversarial Code",
                        "Adversarial truth", 
                        "Adv Prediction", 
                        "Is Success", 
                        "Query Times",
                        "Time Cost",
                        "Perturbation Type",
                        "orig prob",
                        "current prob",
                        "Attack path"])
    
    def write(self, index, original_code,prog_length,ground_truth,orig_prediction,adv_code,adv_truth,adv_prediction,is_sccuess,query_times,time_cost,perturbation_type,orig_prob,current_prob,attack_path):
        self.writer.writerow([index,
                              original_code,
                              prog_length,
                              ground_truth,
                              orig_prediction,
                              adv_code,
                              adv_truth,
                              adv_prediction,
                              is_sccuess,
                              query_times,
                              time_cost,
                              perturbation_type,
                              orig_prob,
                              current_prob,
                              attack_path
                        ])


class Defend_recoder:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.f = open(file_path, 'w')
        self.writer = csv.writer(self.f)
        self.writer.writerow(["source code","orig prob","orig label","adversarial code","adversarial prob","adversarial label","constant propogation code","constant propogation prob","constant propogation label","delete dead code","delete dead prob","delete dead label"\
            ,"outlier code","outlier prob","outlier label","normalized code","normalized prob","normalized label","llm helpd code","llm helped prob","llm helpd label","open llm helped code","open llm helped prob","open llm helped label"])
    
    def write(self, source_code,orig_prob,orig_label,adversarial_code,adversarial_prob,adversarial_label,\
        constant_propogation_code,constant_propogation_prob,constant_propogation_label,delete_dead_code,\
            delete_dead_prob,delete_dead_label,outlier_code,outlier_prob,outlier_label,normalized_code,normalized_prob,normalized_label,llm_helpd_code,llm_helped_prob,llm_helpd_label,open_llm_helped_code,open_llm_helped_prob,open_llm_helped_label):
        self.writer.writerow([source_code,orig_prob,orig_label,adversarial_code,adversarial_prob,adversarial_label,\
            constant_propogation_code,constant_propogation_prob,constant_propogation_label,delete_dead_code,\
                delete_dead_prob,delete_dead_label,outlier_code,outlier_prob,outlier_label,normalized_code,normalized_prob,normalized_label,llm_helpd_code,llm_helped_prob,llm_helpd_label,\
                    open_llm_helped_code,open_llm_helped_prob,open_llm_helped_label])

#