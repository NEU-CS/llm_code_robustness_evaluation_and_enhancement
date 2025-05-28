# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

from tree_sitter import Language, Parser

Language.build_library(
  # Store the library in the `build` directory
  'my-languages.so',

  # Include one or more languages
  [
    'tree-sitter-python',
    'tree-sitter-java',
    'tree-sitter-cpp',
    'tree-sitter-c',
    'tree-sitter-go',
    'tree-sitter-php',
    'tree-sitter-javascript',
    'tree-sitter-ruby',
    'tree-sitter-rust',
    'tree-sitter-c-sharp'

  ]
)

