# coding=utf-8
import argparse
from os import replace
import sys
import os
from queue import Queue
import copy
import pandas as pd
import re
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,AutoConfig
import openai
import re

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上一级目录的路径
parent_dir = os.path.dirname(current_dir)

# 将上一级目录添加到 sys.path 中
sys.path.insert(0, parent_dir)


from parser_folder.DFG_python import DFG_python
from parser_folder.DFG_c import DFG_c
from parser_folder.DFG_java import DFG_java
from parser_folder.DFG_go import DFG_go
from parser_folder.DFG_javascript import DFG_js
from parser_folder.DFG_php import DFG_php
from parser_folder.DFG_ruby import DFG_ruby
from parser_folder.DFG_rust import DFG_rust
from parser_folder.DFG_csharp import DFG_csharp
from parser_folder import (remove_comments_and_docstrings,
                           tree_to_token_index,
                           index_to_code_token,)
from tree_sitter import Language, Parser
import os
import random
from typing import Generator
from tree_sitter import Language, Parser, Tree, Node
from utils_adv import java_keywords , java_special_ids, c_keywords, c_macros, c_special_ids, go_all_names, js_all_names,\
    ruby_keywords, php_all_names, rust_keywords,stmt_separators,csharp_avoid,all_keywords,block_separators,\
        literal_comparsion,all_languages_constant_variable_types,condition_stmt_name_convert_dict,python_keywords


from keyword import iskeyword


path = 'parser_folder/my-languages.so'
path = os.path.join(os.path.dirname(__file__),path)

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'c': DFG_c,
    'go':DFG_go,
    'javascript':DFG_js,
    'php':DFG_php,
    'ruby':DFG_ruby,
    'rust':DFG_rust,
    'c_sharp':DFG_csharp
}


LANG_LIB_MAP = {
    'python': 'tree_sitter_assets/python.so',
    'c': 'tree_sitter_assets/c.so',
    'cpp': 'tree_sitter_assets/cpp.so',
    'java': 'tree_sitter_assets/java.so',
    'go': 'tree_sitter_assets/go.so',
    'javascript': 'tree_sitter_assets/javascript.so',
    'php': 'tree_sitter_assets/php.so',
    'ruby': 'tree_sitter_assets/ruby.so',
    'rust': 'tree_sitter_assets/rust.so',
    'c_sharp': 'tree_sitter_assets/c-sharp.so',
    
}

LANG_REPO_MAP = {
    'python': 'tree-sitter-python',
    'c': 'tree-sitter-c',
    'cpp': 'tree-sitter-cpp',
    'java': 'tree-sitter-java',
    'go': 'tree-sitter-go',
    'javascript': 'tree-sitter-javascript',
    'php': 'tree-sitter-php',
    'ruby': 'tree-sitter-ruby',
    'rust': 'tree-sitter-rust',
    'c_sharp': 'tree-sitter-c-sharp',
    
}



if not os.path.exists(path):
    for lang in LANG_LIB_MAP:
        print(f'Installing {lang} language library...')
        if not os.path.exists(LANG_REPO_MAP[lang]):
            os.popen(
                f'git clone https://github.com/tree-sitter/{LANG_REPO_MAP[lang]}.git'
            ).read()
    Language.build_library(path, list(LANG_REPO_MAP.values()))

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language(path, lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


def is_valid_variable_name(name: str, lang: str) -> bool:
    # check if matches language keywords and special ids
    def is_valid_variable_python(name: str) -> bool:
        return name.isidentifier() and not iskeyword(name) and (name not  in python_keywords)

    def is_valid_variable_java(name: str) -> bool:
        if not name.isidentifier():
            return False
        elif name in java_keywords:
            return False
        elif name in java_special_ids:
            return False
        return True

    def is_valid_variable_c(name: str) -> bool:

        if not name.isidentifier():
            return False
        elif name in c_keywords:
            return False
        elif name in c_macros:
            return False
        elif name in c_special_ids:
            return False
        return True

    def is_valid_variable_go(name:str) -> bool:
        
        if not name.isidentifier():
            return False
        elif name in go_all_names:
            return False
        return True

    def is_valid_variable_js(name: str) -> bool:
        if not name.isidentifier():
            return False
        elif name in js_all_names:
            return False
        return True

    def is_valid_variable_ruby(name: str) -> bool:
        if not name.isidentifier():
            return False
        elif name in ruby_keywords:
            return False
        return True

    def is_valid_variable_php(name: str) -> bool:
        if not name.isidentifier():
            return False
        elif name in php_all_names:
            return False
        return True

    def is_valid_variable_cpp(name: str) -> bool:
        return is_valid_variable_c(name)

    def is_valid_variable_rust(name: str) -> bool:
        if not name.isidentifier():
            return False
        elif name in rust_keywords:
            return False
        return True

    def is_valid_variable_csharp(name: str) -> bool:
        if not name.isidentifier():
            return False
        elif name in csharp_avoid:
            return False
        return True
    functions = {"c":is_valid_variable_c,"java":is_valid_variable_java,"python":is_valid_variable_python,"go":\
        is_valid_variable_go,"javascript":is_valid_variable_js,"php":is_valid_variable_php,\
            "ruby":is_valid_variable_ruby,"rust":is_valid_variable_rust,\
                "c_sharp":is_valid_variable_csharp}
    return functions[lang](name)

def get_code_tokens(code, lang):
    code = code.split('\n')
    code_tokens = [x + '\\n' for x in code if x ]
    return code_tokens

def extract_dataflow(code, lang):
    parser = parsers[lang]
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    parser = parsers[lang]
    tree = parser[0].parse(bytes(code, 'utf-8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    index_to_code = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_to_code[index] = (idx, code)

    index_table = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_table[idx] = index

    DFG, _ = parser[1](root_node, index_to_code, {})

    DFG = sorted(DFG, key=lambda x: x[1])
    return DFG, index_table, code_tokens

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def get_identifiers(code, lang):
    dfg, index_table, code_tokens = extract_dataflow(code, lang)
    ret = []
    for d in dfg:
        if is_valid_variable_name(d[0], lang):
            ret.append(d[0])
    ret = unique(ret)
    ret = [ [i] for i in ret]
    
    return ret, code_tokens
        
def get_program_stmt_nums(code,lang):
    #得到源代码中statement的位置和数量
    ret = code.split(stmt_separators[lang])
    return ret

def replace_node_of_code(code:str,node:Node,replaced:str,diff:int,lang:str):
        code = code[:node.start_byte + diff] + replaced + code[node.end_byte + diff:]
        diff += -(node.end_byte - node.start_byte) + len(replaced)
        return code[:],diff

def insert_code_of_a_node(code,node,inserted,diff,lang):
        # 在node之前insert一段代码
        real_inserted = inserted + stmt_separators[lang] 
        acc = len(real_inserted)
        code = code[:node.start_byte + diff] +  real_inserted + code[node.start_byte + diff:]
        diff += acc
        return code[:],diff   
    
def build_cfg(code,lang):
    #构建一段代码的控制流图，以便于后续操作
    parser = parsers[lang]
    tree = parser[0].parse(bytes(code, 'utf-8'))
    #1. 构建基本块；2. 根据跳转关系连接基本块；
    #把所有的语言都统一转换为一种中间（三地址码）表示，然后再进行构建
    #一个if或while语句的node就是一个基本块，它的所有node就是要跳转到的语句
    #对于死代码检测，很简单，我们只需要考察if或while node的第二个child的值是不是恒定为false，如果是，那就是死代码
    #对于常量False，很好判断；但是对于变量或者是变量之间的比较，就需要进行常量传播后分析了
    #那么对于我们这里判断死代码，只判断常量False。我们理想的情况是在判断死代码前先进行常量分析，将所有的变量转换为常量，然后再进行判断


        



class mytree_node:
    def __init__(self,node,is_source=True):
        self.node = node
        self.is_source = is_source
        self.children = []
        self.parent = None
    
    def set_parent(self,attack_code):
        flag = False if self.node.parent and self.node.parent.text.decode("utf-8") in attack_code else True
        self.parent = mytree_node(self.node.parent if self.node.parent else None,flag)
    
    def set_children(self,attack_code):
        for child in self.node.children:
            flag = False if child and child.text.decode("utf-8") in attack_code else True
            self.children.append(mytree_node(child,flag))
    
class mytree:
    def __init__(self,code,lang,attack_code = []):
        self.__parser = parsers[lang]
        self.tree = self.__parser[0].parse(bytes(code,"utf-8"))
        self.my_root_node = mytree_node(self.tree.root_node,True)
        def dfg(node:mytree_node):
            node.set_parent(attack_code)
            node.set_children(attack_code)
            for child in node.children:
                dfg(child)
        dfg(self.my_root_node)

           
        
        
def traverse_tree(tree: Tree) -> Generator[Node, None, None]:
    cursor = tree.walk()
    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent():
            break


def trasverse_my_tree(node:mytree_node):
    myqueue = [node]
    while myqueue:
        onenode = myqueue.pop()
        yield onenode
        myqueue.extend(onenode.children)

class Defense_technique:
    def __init__(self,attack_code = []) -> None:
        self.attack_code = attack_code[:]
    
    def return_root_node(self,code,lang):
        tree = mytree(code,lang,self.attack_code)
        return tree.my_root_node
    
    def valid_code(self,code,lang):
        parser = parsers[lang]
        tree = parser[0].parse(bytes(code, 'utf-8'))
        for node in traverse_tree(tree):
            if node.is_error:
                return False
        return True
    
    def judge_print_statement(self,node,lang):
            def c(node):
                if node.type == 'expression_statement':
                    expression = node.children[0]
                    if expression.type == 'call_expression':
                        function = expression.child_by_field_name('function')
                        if function.type == 'identifier' and function.text.decode("utf-8") == 'printf':
                            return True
                return False
            def c_sharp(node):
                if node.type == 'expression_statement':
                    expression = node.children[0]
                    if expression.type == 'invocation_expression':
                        function = expression.child_by_field_name('function')
                        if function.type == 'member_access_expression' and function.text.decode("utf-8") == 'Console.WriteLine':
                            return True
                return False
            
            def go(node):
                if node.type == "expression_statement":
                    expression = node.children[0]
                    if expression.type == "call_expression":
                        function = expression.child_by_field_name("function")
                        if function.type == "selector_expression" and function.text.decode("utf-8") == "fmt.Println":
                            return True
                return False
            #解析java语言，判断node是否为print语句
            def java(node: Node) -> bool:
                if node.type == 'expression_statement':
                    expression = node.children[0]
                    if expression.type == 'method_invocation':
                        method = expression.child_by_field_name('name')
                        if method.type == 'identifier' and method.text.decode("utf-8") == 'println':
                            return True
                return False
            
            def javascript(node):
                if node.type == 'expression_statement':
                    expression = node.children[0]
                    if expression.type == 'call_expression':
                        callee = expression.child_by_field_name('function')
                        if callee.type == 'member_expression' and callee.text.decode("utf-8") == 'console.log':
                            return True
                return False
            def php(node):
                if node.type == 'echo_statement':
                    return True
                return False
            def python(node: Node) -> bool:
                if node.type == 'expression_statement':
                    expression = node.children[0]
                    if expression.type == 'call':
                        callee = expression.child_by_field_name('function')
                        if callee.type == 'identifier' and callee.text.decode("utf-8") == 'print':
                            return True
                return False
            def ruby(node: Node) -> bool:
                if node.type == "call":
                    child = node.children[0]
                    if child.type == "identifier" and child.text.decode("utf-8") == "puts":
                        return True
                return False
            
            def rust(node: Node) -> bool:
                if node.type == 'expression_statement':
                    expression = node.children[0]
                    if expression.type == 'macro_invocation':
                        callee = expression.child_by_field_name('macro')
                        if callee.type == 'identifier' and callee.text.decode("utf-8") == 'println':
                            return True
                return False
        
            functions = {"c":c,"java":java,"python":python,"go":go,"javascript":javascript,"php":php,"ruby":ruby,"c_sharp":c_sharp,"rust":rust}
        
            return functions[lang](node)
    

    def judge_false_condition(self,node,lang):
        #检查条件是否为永假
        if node.type == condition_stmt_name_convert_dict[lang]["if_statement"] or node.type == condition_stmt_name_convert_dict[lang]["while_statement"]:
            condition_node = node.child_by_field_name('condition')
            if condition_node:
                condition_code = re.findall("[0-9a-z]+",condition_node.text.decode("utf-8").lower())[0]
                if condition_code == '0' or condition_code == 'false':
                    return True
            else:
                condition_node = node.children[1]
                if condition_node:
                    condition_code = re.findall("[0-9a-z]+",condition_node.text.decode("utf-8").lower())[0]
                    if condition_code == '0' or condition_code == 'false':
                        return True




    def judge_difinition_or_assign_for_a_var(self,node,var,lang):
        #检查此node是否为var的定义或赋值语句
        def c(node:Node,var):
            var_name = ""
            f = False
            right = None
            if node.type.count("declaration") > 0 and node.type.count("parameter") <= 0:
                child = node.child_by_field_name("declarator")
                var_name = child.child_by_field_name("declarator")
                if var_name:
                    var_name = var_name.text.decode("utf-8")
                right = child.child_by_field_name("value")
                f = True
            elif node.type == "assignment_expression":
                child = node.child_by_field_name("left")
                var_name = child.text.decode("utf-8")
                right = node.child_by_field_name("right")
                f = True
            if var_name == var:
                    return True,var_name,f,right
            return False,var_name,f,right
        def c_sharp(node:Node,var):
            var_name = ""
            f = False
            right = None
            if node.type == "variable_declaration":
                child = node.children[1]
                var_name = child.child_by_field_name("name").text.decode("utf-8")
                right = child.children[1]
                f = True
            elif node.type == "assignment_expression":
                child = node.child_by_field_name("left")
                right = node.child_by_field_name("right")
                var_name = child.text.decode("utf-8")
                f = True
            if var_name == var:
                return True,var_name,f,right
            return False,var_name,f,right
        
        #解析go语言，检查node是否为var的定义或赋值语句
        def go(node: Node, var: str) -> bool:
            var_name = ""
            f = False
            right = None
            if node.type == 'var_declaration':
                var_spec = node.children[1]
                var_name = var_spec.child_by_field_name('name').text.decode("utf-8")
                right = var_spec.child_by_field_name('value')
                f = True
            elif node.type == 'assignment_statement':
                var_name = node.child_by_field_name('left').text.decode("utf-8")
                right = node.child_by_field_name('right')
                f = True
            if var_name == var:
                return True,var_name,f,right
            return False,var_name,f,right
        #解析java语言，检查node是否为var的定义或赋值语句
        def java(node: Node, var: str) -> bool:
            var_name = ""
            f = False
            right = None
            if node.type == 'local_variable_declaration':
                child = node.child_by_field_name('declarator')
                var_name = child.child_by_field_name("name").text.decode("utf-8")
                right = child.child_by_field_name('value')
                f = True
            elif node.type == 'assignment_expression':
                var_name = node.child_by_field_name('left').text.decode("utf-8")
                right = node.child_by_field_name('right')
                f = True
            if var_name == var:
                return True,var_name,f,right
            return False,var_name,f,right
        def javascript(node: Node, var: str) -> bool:
            var_name = ""
            f = False
            right = None
            if node.type == 'variable_declarator':
                
                var_name = node.child_by_field_name('name').text.decode("utf-8")
                right = node.child_by_field_name('value')
                f = True
            elif node.type == 'assignment_expression':
                var_name = node.child_by_field_name('left').text.decode("utf-8")
                right = node.child_by_field_name('right')
                f = True
            if var_name == var:
                return True,var_name,f,right
            return False ,var_name,f,right
        def php(node: Node, var: str) -> bool:
            var_name = ""
            f = False
            right = None
            if node.type == 'static_variable_declaration':
                var_name = node.child_by_field_name("name").text.decode("utf-8")
                right = node.child_by_field_name('value')
                f = True
            elif node.type == 'assignment_expression':
                child = node.child_by_field_name("left").children[1]
                right = node.child_by_field_name('right')
                var_name = child.text.decode("utf-8")
                f = True
            if var_name == var:
                return True,var_name,f,right
            return False,var_name,f,right
        
        #解析python语言，检查node是否为var的定义或赋值语句
        def python(node: Node, var: str) -> bool:
            var_name = ""
            f = False
            right = None
            if node.type == 'assignment':
                left_child = node.child_by_field_name('left')
                right = node.child_by_field_name('right')
                var_name = left_child.text.decode("utf-8")
                f = True
                if left_child and  var_name== var:
                    return True,var_name,f,right
            return False,var_name,f,right
        
        #解析ruby语言，检查node是否为var的定义或赋值语句
        def ruby(node: Node, var: str) -> bool:
            var_name = ""
            f = False
            right = None
            if node.type == "assignment":
                left_child = node.child_by_field_name("left")
                right = node.child_by_field_name("right")
                var_name = left_child.text.decode("utf-8")
                f = True
                if left_child.type == "identifier" and  var_name == var:
                    return True,var_name,f,right
            return False,var_name,f,right
        
        #解析rust语言，检查node是否为var的定义或赋值语句
        def rust(node: Node, var: str) -> bool:
            var_name = ""
            f = False
            right = None
            if node.type == "let_declaration":
                left_child = node.child_by_field_name('pattern')
                right = node.child_by_field_name('value')
                var_name = left_child.text.decode("utf-8")
                f = True
                if left_child.type == "identifier" and var_name == var:
                    return True,var_name,f,right
            elif node.type == "assignment_expression":
                left_child = node.child_by_field_name("left")
                right = node.child_by_field_name('right')
                var_name = left_child.text.decode("utf-8")
                f = True
                if left_child.type == "identifier" and  var_name == var:
                    return True,var_name,f,right
            return False,var_name,f,right
        
        functions = {"c":c,"java":java,"python":python,"go":go,"javascript":javascript,"php":php,"ruby":ruby,"c_sharp":c_sharp,"rust":rust}
        
        #返回值 1.代表是不是var的定义或赋值语句;2.代表从这个node中得到的var的名字（如果是定义或赋值语句);3.代表这个node是否是定义或赋值语句;4.代表从这个node中得到的right的名字(如果是定义或赋值语句)
        return functions[lang](node,var)
      

    def judge_no_use_difinition_or_assign(self,node,lang,vars_use_pos):
        #检查左侧变量永远没有使用到的定义或赋值语句
        #首先要判断这个node是否为一个定义或赋值语句
        _,left,f,_ = self.judge_difinition_or_assign_for_a_var(node,"",lang)
        if f:
            if len(vars_use_pos.get(left,set())) == 0:
                return True
        return False
        
    def find_use_var_pos(self,code,lang):
        #找到一段代码中所有将var作为右值使用的位置
        vars, _ = get_identifiers(code,lang)
        ret = {}
        for var in vars:
            ret[var[0]] = self.find_use_stmts_for_a_var(code,var[0],lang)   

        return ret

    def find_use_stmts_for_a_var(self,code,var,lang):
        #找到code中对于一个变量var的所有作为右值使用的语句
        parser = parsers[lang]
        tree = parser[0].parse(bytes(code, 'utf-8'))
        ret = set()
        #问题在这里，这里应该dfg
        root_node = tree.root_node
        
        def dfg(node):
            f,_,_,_ = self.judge_difinition_or_assign_for_a_var(node,var,lang)
            if f:
                return 
            elif node.text.decode("utf-8") == var:
                ret.add(node)
                return
            for child in node.children:
                dfg(child) 
            
        dfg(root_node)
        ret = list(ret)
        ret.sort(key = lambda x:x.end_byte)
        return ret


    def dead_code_judge(self,code,lang):
        #判断一段代码中所有存在的死代码，并将死代码的位置返回
        ret = set()
        vars_use_pos = self.find_use_var_pos(code,lang)
        root_node = self.return_root_node(code,lang)
        def dfg(node:mytree_node):
            
            if self.judge_false_condition(node.node,lang) and not node.is_source: #永远不会执行的死代码
                ret.add(node.node)
                return
            elif self.judge_no_use_difinition_or_assign(node.node,lang,vars_use_pos) and not node.is_source: #永远不会使用的定义或赋值语句
                if node.node.prev_sibling and node.node.prev_sibling.text.decode("utf-8") in all_keywords[lang] :
                    ret.add(node.node.prev_sibling)
                if node.node.next_sibling and node.node.next_sibling.text.decode("utf-8") == stmt_separators[lang]:
                    ret.add(node.node.next_sibling)
                ret.add(node.node)
                return
            elif self.judge_print_statement(node.node,lang) and not node.is_source: #print语句也是死代码
                ret.add(node.node)
                return
            for child in node.children:
                dfg(child)
        
        dfg(root_node)
        ret = list(ret)
        ret.sort(key = lambda x:x.end_byte)
        return ret
                    
    def delete_dead_code(self,code,lang):
        #删除一段代码中所有的死代码
        ret = self.dead_code_judge(code,lang)
        acc = 0
        for v in ret:
            code,acc = replace_node_of_code(code[:],v,"",acc,lang)
        return code


    #下面是简化版常量传播
    def find_var_assignments(self,code,var,lang):
        #根据code找到某个变量的所有定义和赋值语句
        parser = parsers[lang]
        tree = parser[0].parse(bytes(code, 'utf-8'))
        ret = set()
         
        
        for node in traverse_tree(tree):
            f,left,_,right = self.judge_difinition_or_assign_for_a_var(node,var,lang)
            if f and right:
                ret.add(right)
        
        ret = list(ret)
        ret.sort(key = lambda x:x.end_byte)
        return ret

    def find_var_relations(self,code,var,lang):
        #如果一个var是右侧变量，那么它对应的所有左侧变量进行关联
        parser = parsers[lang]
        tree = parser[0].parse(bytes(code, 'utf-8'))
        ret = set()
        
           
        for node in traverse_tree(tree):   
            _,left,f,right = self.judge_difinition_or_assign_for_a_var(node,"",lang)
            if f:#是一个定义或赋值语句
                if right :
                    if right.text.decode("utf-8") == var:#右侧是var
                        ret.add(left) #左侧就是关联的变量
                    children = right.children
                    for onechild in children:
                        if onechild.text.decode("utf-8") == var:
                            ret.add(left)
        
        return ret

    #对一段代码中所有变量创建一个dict，表示这个变量的取值



    def is_constant_or_identifier(self,node,num,lang):
            # 判断node是否是一个常量或者是一个identifier，如果是，返回node.text（忽略node中的符号)
            # 返回值1：是否为常量；返回值2：是否为变量；返回值3：node.text (如果1和2有一个为true，否则返回空字符串)

            if node.type in literal_comparsion[lang]["normal"] + literal_comparsion[lang]["container"]:
                num += 1
                return num == 1,False,node.text.decode("utf-8")
            elif is_valid_variable_name(node.text.decode("utf-8"),lang):
                num += 1
                return False,num == 1,node.text.decode("utf-8")
            f1 = False
            f2 = False
            for child in node.children:
                f1_1,f2_2,name = self.is_constant_or_identifier(child,num,lang)
                f1 = f1 | f1_1
                f2 = f2 | f2_2
                if f1 or f2:
                    return f1,f2,name
            return False,False,""

    def is_function_call(self,node,lang):
        #判断是否为function call
        if node.type.count("call") > 0:
            return True
        return False

    def constant_propogation(self,code,lang):
        #常量传播
        vars,_ = get_identifiers(code,lang)
        var_values = {var[0]:"UDF" for var in vars} #值统一用python的str来表示
        # "UDF"表示undefined,"NAC"表示not a constant
        work_list = [var[0] for var in vars]
        var_values_copy = copy.deepcopy(var_values)
        var_assigns = {var[0]:self.find_var_assignments(code,var[0],lang) for var in vars}
        var_relations = {var[0]:self.find_var_relations(code,var[0],lang) for var in vars}
        while work_list:
            var_values_copy = copy.deepcopy(var_values)
            work_list_temp = []
            for var in work_list:
                value = copy.deepcopy(var_values.get(var,"UDF"))
                all_assigns = var_assigns.get(var,[])
                for right_var in all_assigns:
                    if self.is_function_call(right_var,lang):
                        var_values[var] = "NAC"
                        continue
                    is_constant,is_identifier,ret_str = self.is_constant_or_identifier(right_var,0,lang)
                    if is_constant:
                        if var_values[var] == "UDF":
                            var_values[var] = ret_str
                        elif var_values[var] != ret_str:
                            var_values[var] = "NAC"
                        
                        #如果右侧是常量，那么就是一个常量，如果右侧是变量，还需要进一步判断
                    else:
                        #如果右侧是变量,首先考虑右侧只有一个变量的情况
                        #这个判断有问题，应该判断right_var中是否只有一个identifier，如果只有一个identifier，返回它即可
                        if is_identifier:
                            #右侧只有一个变量    
                            if var_values[var] == "UDF":
                                var_values[var] = var_values.get(ret_str,"UDF")
                            elif var_values[var] != var_values.get(ret_str,"UDF") and var_values.get(ret_str,"UDF") != "UDF":
                                var_values[var] = "NAC"
                            
                        else:
                            #右侧超过一个变量,说明右侧是一个表达式，这时候就需要计算右侧表达式的值到底是不是一个常量
                            #这下面一定是要多于一个identifier或者常量的表达式才行，否则就有问题
                            if True:
                                children = right_var.children
                                f = True
                                for onechild in children:
                                    #如果右边有一个"NAC",那么就一定是NAC
                                    if var_values.get(onechild.text.decode("utf-8"),"UDF") == "NAC":
                                        var_values[var] = "NAC"
                                        f = False
                                        break
                                    #如果右边有一个identifier是"UDF",那么就是"UDF"
                                    elif is_valid_variable_name(onechild.text.decode("utf-8"),lang) and var_values.get(onechild.text.decode("utf-8"),"UDF") == "UDF":
                                        var_values[var] = "UDF"
                                        f = False
                                        break
                                if f:
                                    if var_values[var] == "UDF":
                                        var_values[var] = right_var.text.decode("utf-8") #把表达式统一视为常量
                                    elif right_var.text.decode("utf-8") != var_values.get(var,"UDF"):
                                        var_values[var] = "NAC"
                if value != var_values.get(var,"UDF"):
                    work_list_temp.extend(var_relations[var])
            work_list = copy.deepcopy(work_list_temp)
            if var_values_copy == var_values:
                break
        
        return var_values


    def edit_code_by_constant_information(self,code,lang,attack_codes):
        #根据常量传播的结果将code的所有constant转换
        constant_res = self.constant_propogation(code,lang)
        #找到constrant_res中是常量的部分
        code_res = copy.deepcopy(code)
        vars_use_pos = self.find_use_var_pos(code,lang)
        replace_order = []
        for k,v in constant_res.items():
            if v!="UDF" and v!="NAC":
                #这里说明k变量是一个常量，接下来将k变量的所有位置替换掉
                
                if k.strip() in attack_codes:
                    for node in vars_use_pos[k]:
                        #变量k的所有位置
                        replace_order.append([k,v,node])
        
        replace_order.sort(key=lambda x:x[2].end_byte)
        acc = 0
        for value in replace_order:
            k = value[0]
            v = value[1]
            node = value[2]
            code_res,acc = replace_node_of_code(code_res,node,v,acc,lang)
        return code_res
    
    def get_example(self,code, tgt_word, substitute, lang):
        #为什么不直接用code.replace?因为我们只replace identifier,不会将code中所有与其相符的字符串都replace掉，这样有可能会有问题
        #如：一个identifier为f，如果直接用code.replace("f","g")，那么所有的f都会被替换掉，这样就会有问题。if语句里面的f就错了
        parser = parsers[lang]
        tree = parser[0].parse(bytes(code, 'utf-8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        replace_pos = {}
        for index, code_token in enumerate(code_tokens):
            if code_token == tgt_word:
                try:
                    replace_pos[tokens_index[index][0][0]].append((tokens_index[index][0][1], tokens_index[index][1][1]))
                except:
                    replace_pos[tokens_index[index][0][0]] = [(tokens_index[index][0][1], tokens_index[index][1][1])]
        diff = len(substitute) - len(tgt_word)
        for line in replace_pos.keys():
            for index, pos in enumerate(replace_pos[line]):
                code[line] = code[line][:pos[0]+index*diff] + substitute + code[line][pos[1]+index*diff:]

        return "\n".join(code)
    
    def outlier_detection(self,code:str,lang,word_embeddings,tokenizer,threshold):
        #复现DAMP论文中的outlier_detection
        identifiers,_ = get_identifiers(code,lang)
        identifiers = [x[0] for x in identifiers]
        identifier_embedding_dict = {x:None for x in identifiers}
        for x in identifiers:
            tensors = tokenizer(x,return_tensors="pt",add_special_tokens = False)
            id_embedding = word_embeddings(tensors.input_ids.cuda())
            identifier_embedding_dict[x] = torch.sum(id_embedding,dim=1)[0]/id_embedding.shape[1]
            assert id_embedding.shape[-1] == len(identifier_embedding_dict[x])
        #计算所有的identifier之间的相似度
        all_similarities = {x:0 for x in identifiers}
        for i in range(len(identifiers)):
            embeddings = torch.zeros(size=(len(identifier_embedding_dict[identifiers[0]]),)).cuda()
            for j in range(len(identifiers)):
                if i == j:
                    continue
                embeddings += identifier_embedding_dict[identifiers[j]]
            sub_res = embeddings/(len(identifiers)-1) - identifier_embedding_dict[identifiers[i]]
            all_similarities[identifiers[i]] = torch.dot(sub_res.T,sub_res)
        simi_sort = []
        for k,v in all_similarities.items():
            simi_sort.append((k,v))
        simi_sort.sort(key=lambda x:x[1],reverse=True)
        #print(simi_sort)
        for id,simi in simi_sort:
            if simi > threshold:
                code = self.get_example(code[:],id,str(tokenizer.unk_token),lang)
            else:
                break
        return code
                
        
    
    def open_source_llm_recoginze_unrelated_identifier(self,code:str,lang,model,tokenizer):
        #询问大模型
        """
        我将给你一段代码，它是使用某种编程语言编写的。在这段代码中，有一些变量名被恶意修改，打破了原代码的可读性。修改后的变量名通常有如下特点：1. 和整段代码的执行功能不统一，如一段求和的代码，变量名却为div；2. 和代码中其他变量名格格不入，不适配。
请你帮我找出这些被修改后的变量名，并尝试猜测修改前的变量名是什么。
在代码中：带有<|identifier|>后缀的是一个变量，不需要你去识别哪些是一个变量
输出格式：原变量名 -> 修改后的变量名
        """
        prompt = """I will give you a piece of code written in the """ + lang + """ language. In this code, some variable names have been maliciously modified, breaking the original code's readability. The modified variable names usually have the following characteristics: 1. They are inconsistent with the overall functionality of the code, such as having a code snippet for summation but with a variable name like "div"; 2. They do not match other variable names in the code, appearing out of place.

Please help me identify these modified variable names and try to guess what the original variable names were.

In the code: Variables with suffix "<|identifier|>" are variables. You don't need to identify which ones are variables.

Output format: Original variable name -> Modified variable name """

        def preprocess_code(code:str):
            identifiers,_ = get_identifiers(code,lang)
            identifiers = [x[0] for x in identifiers]
            for x in identifiers:
                code = self.get_example(code[:],x,x + "<|identifier|>",lang)
            return code
            
        codepr = preprocess_code(code)      
        prompt = prompt + codepr
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs,max_new_tokens = 100,do_sample = True)
        output_str = tokenizer.decode(outputs[0])
        res = output_str.replace(prompt,"")
        print(res)
        re_res = re.findall("([\"a-zA-Z_0-9<|>]+) -> ([\"a-zA-Z_0-9 <|>]+)",res)
        print(re_res)
        for s,t in re_res:
            s = s.replace("<|identifier|>","")
            s = s.replace("\"","")
            t = re.findall("([a-zA-Z_0-9]+)",t)
            t = t[0]
            #print(s,t)
            code = self.get_example(code[:],s,t,lang)
        return code

    
    def gpt_3_llm_recoginze_unrelated_identifier(self,code:str,lang,api_key,model="gpt-3.5-turbo",seed = 42):
        client = OpenAI(api_key=api_key)
        
        prompt = """I will give you a piece of code written in the """ + lang + """ language. In this code, some of the variable names have been maliciously modified, breaking the original code's readability. The modified variable names usually have the following characteristics: 1. They are inconsistent with the overall functionality of the code, such as having a code snippet for summation but with a variable name like "div"; 2. They do not match other variable names in the code, appearing out of place.

Please help me identify these modified variable names and try to guess what the original variable names were. 

In the code: Variables with suffix "<|identifier|>" are variables. You don't need to identify which ones are variables. In the result, you need to remove the "<|identifier|>" suffix. And only one modified variable is good enough.

Output format: Original variable name -> Modified variable name"""

        def preprocess_code(code:str):
            identifiers,_ = get_identifiers(code,lang)
            identifiers = [x[0] for x in identifiers]
            for x in identifiers:
                code = self.get_example(code[:],x,x + "<|identifier|>",lang)
            return code
            
        codepr = preprocess_code(code)      
        
        chat_completion = client.chat.completions.create(
            model = model,
            messages=[
                {
                "role": "system",
                "content": prompt
                },
                {
                "role": "user",
                "content": codepr
                }
            ],
            temperature = 0.3,
            top_p = 0.1,
            max_tokens=100,
            seed = seed
        )
        res = chat_completion.choices[0].message.content
        #print(res)
        re_res = re.findall("([\"a-zA-Z_0-9<|>]+) -> ([\"a-zA-Z_0-9 <|>]+)",res)
        #print(re_res)
        for s,t in re_res:
            s = s.replace("<|identifier|>","")
            s = s.replace("\"","")
            t = re.findall("([a-zA-Z_0-9]+)",t)
            t = t[0]
            #print(s,t)
            code = self.get_example(code[:],s,t,lang)
        return code
        
    
    def normalize_identifier_names(self,code,lang):
        #统一变量命名方式
        #这是一种re-training的防御方法，也可以不重训练（但可能影响performance）
        #缺点就是会影响代码的可读性，以及可能影响模型的performance，以及重训练代价比较大
        identifiers,_ = get_identifiers(code,lang)
        identifiers = [x[0] for x in identifiers]
        replace_name_lst = ["a"]
        replace_number = 0
        replace_name = ''.join(replace_name_lst)
        replace_recod = {} #记录某个变量名被替换成了什么
        for x in identifiers:
            if x in replace_recod:
                code = self.get_example(code[:],x,replace_recod[x],lang)
                continue
            replace_recod[x] = replace_name+str(replace_number)
            code = self.get_example(code[:],x,replace_recod[x],lang)
            replace_number += 1
            if replace_number == 10:
                replace_number = 0
                if replace_name_lst[-1] == 'z':
                    replace_name_lst.append("a")
                else:
                    replace_name_lst[-1] = chr(ord(replace_name_lst[-1])+1)
                replace_name = ''.join(replace_name_lst)
        return code

    
class Attack_technique:
    def __init__(self) -> None:
        self.attack_code = []


    def definition_stmt(self,var_name, var_type, var_value, lang):
        # 根据语言映射到对应的类型
        try:
            if lang.lower() in ["python", "javascript", "php", "ruby"]:
                # 这些语言定义变量时不需要类型
                if lang.lower() == "python":
                    return f"{var_name} = {var_value}"
                elif lang.lower() == "javascript":
                    return f"let {var_name} = {var_value}"
                elif lang.lower() == "php":
                    return f"${var_name} = {var_value}"
                elif lang.lower() == "ruby":
                    return f"{var_name} = {var_value}"
            else:
                # 对于其他语言，需要转换常量类型到变量类型
                var_types = all_languages_constant_variable_types[lang.lower()][var_type]
                if not var_types:
                    raise ValueError("Unsupported constant type for the language.")
                var_type_str = var_types[0]  # 选择最通用的类型
                if lang.lower() == "c" or lang.lower() == "c_sharp":
                    return f"{var_type_str} {var_name} = {var_value}"
                elif lang.lower() == "go":
                    return f"var {var_name} {var_type_str} = {var_value}"
                elif lang.lower() == "java":
                    return f"{var_type_str} {var_name} = {var_value}"
                elif lang.lower() == "rust":
                    return f"let {var_name}: {var_type_str} = {var_value}"
        except KeyError:
            return "Unsupported language or constant type."

    def pre_process_code(self,code):
        #预处理code
        #每一行只能存在一个语句，一行存在多个语句的，直接移动到下一行，并保持相同的缩进
        code = code.replace("\\n", "\n")
        code_lst = code.split("\n")
        all_tab_count = []
        for line in code_lst:
            now_tab = ""
            for i,v in enumerate(line):
                if v == "\t" or v == " ":
                    now_tab += v
                else:
                    break
            all_tab_count.append(now_tab)

        return all_tab_count


    def const_transfer_identifier(self,code:str,var_names:list,lang:str,tab_count:list):
        # 将代码中的常量转换为变量，如false转换为变量a, a = false ,use(a)
        # 找到一个常量的位置，然后将这个常量转换为一个变量，然后在这个位置插入一个变量定义语句
        # 常量只能用作右值
        code = code.replace("\\n", "\n")
        parser = parsers[lang]
        tree = parser[0].parse(bytes(code, 'utf-8'))
        acc = 0
        index = 0
        #如何解决缩进问题，保持和node相同的缩进
        visited = {i:False for i in range(len(code))}
        identifiers,_ = get_identifiers(code,lang)
        identifiers = [v[0] for v in identifiers]
        transfer_res = []
        line_diff = 0
        for node in traverse_tree(tree):
            if  node.type in literal_comparsion[lang]["normal"] + literal_comparsion[lang]["container"] :
                #找到同一个block内的上一个分隔符的位置，在那个分隔符后面插入一个变量定义语句
                if not visited.get(node.end_byte + acc,True): #如果找不到就不进入
                    visited[node.end_byte + acc] = True
                    this_line_tab_count = tab_count[node.start_point[0] + line_diff]
                    pos = node.start_byte + acc
                    while pos > 0 and code[pos] != stmt_separators[lang]:
                        if code[pos] in block_separators[lang]:
                            break
                        pos -= 1
                    if pos > 0:
                        pos += 1
                    if index >= len(var_names):
                        break
                    replace_var_name = var_names[index]
                    while replace_var_name in identifiers:
                        index += 1
                        if index >= len(var_names):
                            break
                        replace_var_name = var_names[index]
                    if index >= len(var_names):
                        break
                    identifiers.append(replace_var_name)
                    inserted =  self.definition_stmt(replace_var_name,node.type,node.text.decode("utf-8"),lang)
                    mid_var = "\n" if stmt_separators[lang] != "\n" else ""
                    real_inserted = mid_var + this_line_tab_count + inserted + stmt_separators[lang]
                    tab_count = tab_count[:node.start_point[0]] + [this_line_tab_count] + tab_count[node.start_point[0]:]
                    transfer_res.append((inserted + stmt_separators[lang]).strip())
                    transfer_res.append(replace_var_name.strip())
                    code = code[:pos] + real_inserted + code[pos:]
                    acc += len(real_inserted)
                    code,acc = replace_node_of_code(code,node,replace_var_name,acc,lang)
                    index += 1
                    line_diff += 1
                    
                    
        return code,transfer_res
    
    def generate_deadcode(self,var_name:str, var_list:list, language:str, flag:bool):
        # 首先，需要确保 random 模块已经导入
        random.seed(0)  # 确保示例具有可预测性，实际使用时可能不需要
        
        # 对于flag为False的情况，生成一个空的打印语句
        if flag == False:
            dead_code_snippets = {
                'python': ['print()'],
                'java': ['System.out.println();'],
                'c': ['printf("");'],
                'c_sharp': ['Console.WriteLine();'],
                'go': ['fmt.Println("")'],
                'javascript': ['console.log("");'],
                'php': ['echo "";'],
                'ruby': ['puts ""'],
                'rust': ['println!("");']
            }
            try:
                return dead_code_snippets[language]
            except:
                raise ValueError(f"No dead code pattern available for {language}")
            

        # 当flag为True时，生成具体的死代码
        dead_code_patterns = {
            'python': [
                f"print({var_name})",
                f"if False:\n    {var_name}={random.choice(var_list)[0]}",
                f"while False:\n    {var_name}={random.choice(var_list)[0]}"
            ],
            'java': [
                f"System.out.println({var_name});",
                f"if(false){{\n    {var_name}={random.choice(var_list)[0]};\n}}",
                f"while(false){{\n    {var_name}={random.choice(var_list)[0]};\n}}"
            ],
            'c': [
                f'printf("%s", {var_name});',
                f'if(0){{\n    {var_name} = "{random.choice(var_list)[0]}";\n}}',
                f'while(0){{\n    {var_name} = "{random.choice(var_list)[0]}";\n}}'
            ],
            'c_sharp': [
                f'Console.WriteLine({var_name});',
                f'if(false){{\n    {var_name} = "{random.choice(var_list)[0]}";\n}}',
                f'while(false){{\n    {var_name} = "{random.choice(var_list)[0]}";\n}}'
            ],
            'go': [
                f'fmt.Println({var_name})',
                f'if false {{\n    {var_name} = "{random.choice(var_list)[0]}"\n}}',
                f'for false {{\n    {var_name} = "{random.choice(var_list)[0]}"\n}}'
            ],
            'javascript': [
                f'console.log({var_name});',
                f'if(false){{\n    {var_name} = "{random.choice(var_list)[0]}";\n}}',
                f'while(false){{\n    {var_name} = "{random.choice(var_list)[0]}";\n}}'
            ],
            'php': [
                f'echo {var_name};',
                f'if(false){{\n    ${var_name} = "{random.choice(var_list)[0]}";\n}}',
                f'while(false){{\n    ${var_name} = "{random.choice(var_list)[0]}";\n}}'
            ],
            'ruby': [
                f'puts {var_name}',
                f'if false\n    {var_name} = "{random.choice(var_list)[0]}"\nend',
                f'while false\n    {var_name} = "{random.choice(var_list)[0]}"\nend'
            ],
            'rust': [
                f'println!("{{}}", {var_name});',
                f'if false {{\n    let {var_name} = "{random.choice(var_list)[0]}";\n}}',
                f'while false {{\n    let {var_name} = "{random.choice(var_list)[0]}";\n}}'
            ]
        }

        # 返回指定语言的死代码列表，如果没有找到匹配的语言，则抛出异常
        if language in dead_code_patterns:
            return dead_code_patterns[language]
        else:
            raise ValueError(f"No dead code pattern available for {language}")
    
    def insert_deadcode(self,pos,source_code,insert_deadcode:str,lang):
        #在源代码中插入死代码
        all_insert_pos = get_program_stmt_nums(source_code,lang)
        tab_count = 0
        tab_pos = 0
        space_count = 0
        exmine = all_insert_pos[pos - 1].lstrip("\n")
        while tab_pos < len(exmine):
            if exmine[tab_pos] == "\t":
                tab_count += 1
            elif exmine[tab_pos] == " ":
                space_count += 1
                if space_count == 4:
                    tab_count += 1
                    space_count = 0
            else:
                break
            tab_pos += 1
        insert_deadcode = "\n" + " " * tab_count * 4 + insert_deadcode
        try:
            all_insert_pos[pos] = insert_deadcode + all_insert_pos[pos]
        except:
            all_insert_pos.append(insert_deadcode)
        code = stmt_separators[lang].join(all_insert_pos)
        return code

    
    def get_example(self,code, tgt_word, substitute, lang):
        parser = parsers[lang]
        tree = parser[0].parse(bytes(code, 'utf-8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        replace_pos = {}
        for index, code_token in enumerate(code_tokens):
            if code_token == tgt_word:
                try:
                    replace_pos[tokens_index[index][0][0]].append((tokens_index[index][0][1], tokens_index[index][1][1]))
                except:
                    replace_pos[tokens_index[index][0][0]] = [(tokens_index[index][0][1], tokens_index[index][1][1])]
        diff = len(substitute) - len(tgt_word)
        for line in replace_pos.keys():
            for index, pos in enumerate(replace_pos[line]):
                code[line] = code[line][:pos[0]+index*diff] + substitute + code[line][pos[1]+index*diff:]

        return "\n".join(code)
    
    def get_example_batch(self,code, chromesome, lang):
        parser = parsers[lang]
        tree = parser[0].parse(bytes(code, 'utf-8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        replace_pos = {}
        for tgt_word in chromesome.keys():
            diff = len(chromesome[tgt_word]) - len(tgt_word)
            for index, code_token in enumerate(code_tokens):
                if code_token == tgt_word:
                    try:
                        replace_pos[tokens_index[index][0][0]].append((tgt_word, chromesome[tgt_word], diff, tokens_index[index][0][1], tokens_index[index][1][1]))
                    except:
                        replace_pos[tokens_index[index][0][0]] = [(tgt_word, chromesome[tgt_word], diff, tokens_index[index][0][1], tokens_index[index][1][1])]
        for line in replace_pos.keys():
            diff = 0
            for index, pos in enumerate(replace_pos[line]):
                code[line] = code[line][:pos[3]+diff] + pos[1] + code[line][pos[4]+diff:]
                diff += pos[2]

        return "\n".join(code)

    def println(code):
        print(code,end="\n---------------------------------------\n")
    
    
    
if __name__ == '__main__':
    # 1. 对注释不要动
    # 2. 原来的代码不要动，只修改我自己攻击的代码
    
    code = '''
    #include <stdio.h>

int main() {
    int flag = 0;
    int h = flag;
    int l = h;
    printf("C Language.\n");
    
    if (flag) {
        printf("This will never be executed due to flag.\n");
    }
    
    flag = h;
    
    while (h) {
        printf("This loop will never run due to h.\n");
    }
    
    return 0;
}

    '''
    lang = "c"

    model_name = "../Models/deepseek-ai/deepseek-coder-6.7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,config = config)
    dt = Defense_technique()
    print(dt.open_source_llm_recoginze_unrelated_identifier(code,lang,model,tokenizer))
    
