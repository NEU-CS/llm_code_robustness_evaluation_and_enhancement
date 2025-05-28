# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

from tree_sitter import Language, Parser
from .utils import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
from .traverse_tree import traverse_tree

def DFG_rust(root_node, index_to_code, states):
    #模仿上述DFG算法,写一个对于go语言的DFG算法
    #不需要写DFG算法，我们只想要得到代码的identifier信息，直接遍历树提取其中的identifier即可
    DFG = []
    for node in traverse_tree(root_node):
        if node.type == 'identifier':
            DFG.append([node.text.decode('utf-8'),node.start_point,node.end_point])
    return DFG,None
    
    
    
    
