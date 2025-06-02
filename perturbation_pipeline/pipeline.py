#coding=utf-8
import os
from typing import Generator,List,Set,Tuple
import tree_sitter
import re
import random
from keyword import iskeyword,kwlist

from utils import common_built_in_functions_of_python,java_keywords,java_special_ids,c_keywords,c_macros,c_special_ids,js_future_reserved_words,\
js_global_objects,js_keywords,base_module,all_languages_constant_variable_types,python_literal,cpp_literal,java_literal,javascript_literal,literals\
,python_tree_sitter_simple_statement,python_complex_treesitter_statement
from transformers import AutoModelForMaskedLM,AutoTokenizer,pipeline
import torch
import copy
from collections import Counter

class PerturbationPipeline:
    def __init__(self):
        self.__parser = None

    def set_seed(self,seed):
        torch.manual_seed(seed)
        random.seed(seed)
        # 如果使用 GPU，还需要设置 CUDA 的随机种子
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
    def __init_tree_sitter_parser(self, lang: str):
        LANGUAGE = tree_sitter.Language('./tree_sitter/my-languages.so', lang)
        self.__parser = tree_sitter.Parser()
        self.__parser.set_language(LANGUAGE)
    
    def init_pretrained_model(self,path="../llm_robustness_eval_and_enhance/model/codebert/codebert-mlm"):
        '''
        如果需要使用codebert-mlm进行变量或函数名替换,首先使用此函数初始化预训练模型
        '''
        self.pretrained_model = AutoModelForMaskedLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pretrained_model.to(self.device)
        self.pipeline = pipeline(model=self.pretrained_model,tokenizer=self.tokenizer,task="fill-mask",device = self.device)
    



    def __is_valid_identifier_name(self, var_name: str, lang: str) -> bool:
        # check if matches language keywords and special ids
        def is_valid_identifier_python(name: str) -> bool:
            return name.isidentifier() and not iskeyword(name) and (name not in kwlist + common_built_in_functions_of_python)
        
        def is_valid_identifier_java(name: str) -> bool:
            if not name.isidentifier():
                return False
            elif name in java_keywords:
                return False
            elif name in java_special_ids:
                return False
            return True

        def is_valid_identifier_cpp(name: str) -> bool:

            if not name.isidentifier():
                return False
            elif name in c_keywords:
                return False
            elif name in c_macros:
                return False
            elif name in c_special_ids:
                return False
            return True
        
        def is_valid_identifier_js(name: str) -> bool:
            if not name.isidentifier():
                return False
            elif name in js_future_reserved_words + js_global_objects + js_keywords:
                return False
            return True
            
        functions = {"python":is_valid_identifier_python,'java':is_valid_identifier_java,'cpp':is_valid_identifier_cpp,'javascript':is_valid_identifier_js}
        return functions[lang](var_name) and bool(re.match(r'[a-zA-Z0-9_]',var_name))
        
    
    def preprocess_code(self, code: str,lang:str) -> str:
        """
        预处理code,主要有以下两步:
        1. 避免不同平台换行符、回车符不同，统一替换
        2. 对代码段中所有中文字符进行暂存
        """
        self.__init_tree_sitter_parser(lang)
        code = code.replace("\r\n", "\n").replace("\\n", "\n")
        self.lang = lang
        return code

    def __post_preprocess_code(self, code: str) -> str:
        """
        后处理代码
        """
        return code

    def __traverse_tree(self, tree) -> Generator[tree_sitter.Node, None, None]:
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

    def __replace_node_of_code(self, code: str, node: tree_sitter.Node, replaced: str, diff: int):
        '''
        将代码段code中node节点替换为replaced
        '''
        start_pos = node.start_byte
        end_pos = node.end_byte
        code_bytes = code.encode("utf-8")
        replaced_bytes = replaced.encode("utf-8")
        code_bytes = code_bytes[:start_pos + diff] + replaced_bytes + code_bytes[end_pos + diff:]
        diff += -(end_pos - start_pos) + len(replaced_bytes)
        return code_bytes.decode("utf-8"), diff

    def __get_class_names(self,code:str) -> List[str]:
        '''
        获得一段代码中所有的函数名,及其对应于tree的节点node
        目前主要针对于cpp语言,因为cpp语言的构造函数和类名一致,我们需要判断一个函数是不是构造函数,就必须要获得类名
        '''
        tree = self.__parser.parse(bytes(code, 'utf-8'))

        def cpp(node:tree_sitter.Node):
            if node.type == "type_identifier":
                if node.parent.type == 'class_specifier':
                    return True
            return False
        
        def python(node):
            pass

        def java(node):
            pass

        def javascript(node):
            pass

        functions = {'cpp':cpp,'java':java,'python':python,'javascript':javascript}
        ret = set()
        names = set()
        for node in self.__traverse_tree(tree):
            if functions[self.lang](node):
                name = node.text.decode("utf-8")
                if self.__is_valid_identifier_name(name,self.lang):
                    ret.add(node)
                    names.add(name)
        ret = list(ret)
        ret.sort(key=lambda x:x.start_byte)
        names = list(names)
        names.sort()
        return ret,names

    def get_invoke_func_names(self,code:str) -> List[str]:
        '''
        获得一段代码中所有invoke的函数名
        '''
        tree = self.__parser.parse(bytes(code, 'utf-8'))
        
        #这里定义的函数是为了判断一个结点是否为“函数”的定义语句里面的函数名，只要找到定义语句里面的所有函数名，就找到了所有的函数名
        #不需要去寻找函数调用语句里面的函数名
        _,class_names = self.__get_class_names(code)
        def python(node:tree_sitter.Node):
            #print(node)
            if node.type == "identifier":
                if node.parent.type == "call":
                    return True
            return False

        def cpp(node:tree_sitter.Node):
            #print(node)
            if node.type == "identifier" and node.text.decode("utf-8") != "assert":
                if "call" in node.parent.type:
                    return True
            return False

        def java(node:tree_sitter.Node):
            #print(node)
            if node.type == "identifier":
                if "invo" in node.parent.type:
                    return True
            return False

        def javascript(node:tree_sitter.Node):
            #print(node)
            if node.type == "identifier":
                if "call" in node.parent.type:
                    return True
            return False
        
        ret = set()
        names = set()
        functions = {"cpp":cpp,'python':python,'java':java,'javascript':javascript}
        for node in self.__traverse_tree(tree):
            if functions[self.lang](node):
                name = node.text.decode("utf-8")
                ret.add(node)
                names.add(name)
        ret = list(ret)
        ret.sort(key=lambda x:x.start_byte)
        names = list(names)
        names.sort()
        return ret,names

    def get_function_names(self,code:str) -> List[str]:
        '''
        获得一段代码中所有的函数名(定义的函数名),及其对应于tree的节点node
        '''
        tree = self.__parser.parse(bytes(code, 'utf-8'))
        
        #这里定义的函数是为了判断一个结点是否为“函数”的定义语句里面的函数名，只要找到定义语句里面的所有函数名，就找到了所有的函数名
        #不需要去寻找函数调用语句里面的函数名
        _,class_names = self.__get_class_names(code)
        def python(node:tree_sitter.Node):
            if node.type == "identifier":
                if node.parent.type == "function_definition" and node.prev_sibling.text.decode("utf-8") == "def":
                    return True
            return False

        def cpp(node:tree_sitter.Node):
            if node.type == "identifier":
                if node.parent.type == "function_declarator":
                    if node.parent.parent and node.parent.parent.type in ["function_definition"]:#函数定义和函数声明
                        if node.text.decode("utf-8") in class_names:
                            return False
                        return True
            return False

        def java(node:tree_sitter.Node):
            if node.type == "identifier" and node.parent.type == "method_declaration":
                return True
            return False

        def javascript(node:tree_sitter.Node):
            if node.type == "identifier":
                if node.parent.type == "function_declaration":
                    return True
                if node.next_sibling:
                    if node.next_sibling.text.decode("utf-8") == "=": #等号表示一个function的定义 arrow_function
                        if node.next_sibling.next_sibling and node.next_sibling.next_sibling.type == "arrow_function":
                            return True
            return False
        
        ret = set()
        names = set()
        functions = {"cpp":cpp,'python':python,'java':java,'javascript':javascript}
        for node in self.__traverse_tree(tree):
            if functions[self.lang](node):
                name = node.text.decode("utf-8")
                if self.__is_valid_identifier_name(name,self.lang):
                    ret.add(node)
                    names.add(name)
        ret = list(ret)
        ret.sort(key=lambda x:x.start_byte)
        names = list(names)
        names.sort()
        return ret,names
    
    def rename_function_name(self,code:str,func_name:str,substitute:str) -> str:
        '''
        将function name替换为substitute
        '''
        diff = 0
        ret_code = code
        tree = self.__parser.parse(bytes(code, 'utf-8'))
        #这里的函数是为了判断一个结点是否为函数定义、函数调用语句的函数名
        #一个函数只可能在两个地方出现，一种是函数定义语句，一种是函数调用语句
        #我们需要修改两个地方的函数名
        def python(node:tree_sitter.Node):
            if node.type == "identifier":
                if node.parent.type == "function_definition" and node.prev_sibling.text.decode("utf-8") == "def":
                    return True
                elif node.parent.type == "call":
                    if node.next_sibling.text.decode("utf-8") == ".":#x.y()也是一种call，但x是一个变量，不是一个function,y才是一个function
                        return False
                    return True
                elif node.parent.parent.type == "call" and node.parent.parent.children[0].text.decode("utf-8") in ["map","filter","reduce"]:
                    #可能是map,filter,reduce的第一个参数，这个参数也是函数
                    return True
            return False

        def cpp(node:tree_sitter.Node):
            def recur(node:tree_sitter.Node):
                    if node.parent.type == "function_declarator": 
                        if node.parent.parent and node.parent.parent.type in ["function_definition"]:#函数定义和函数声明
                            return True
                    elif node.parent.type == "call_expression": #普通函数调用
                        if node.next_sibling.text.decode("utf-8") == ".":
                            return False
                        return True
                    elif node.parent.type == "template_function":
                        return recur(node.parent)
                    elif node.parent.type == "argument_list":
                        if node.parent.parent and node.parent.parent.type == "call_expression":
                            node = node.parent.parent
                            if node.children[0].type in ["qualified_identifier"]:
                                node = node.children[0]
                            if node.child_by_field_name('name') and node.child_by_field_name('name').text.decode("utf-8") in ['sort',"equal"]:
                                return True
                            
                    return False
            if node.type == "identifier":
                return recur(node)
            return False

        def java(node:tree_sitter.Node):
            if node.type == "identifier" :
                if node.parent.type == "method_declaration":
                    return True
                elif node.parent.type == "method_invocation":
                    if node.next_sibling.text.decode("utf-8") == ".":
                        return False
                    return True
                elif node.parent.type == "method_reference":
                    return True
            return False

        def javascript(node:tree_sitter.Node):
            if node.type == "identifier":
                if node.parent.type == "function_declaration":
                    return True
                if node.next_sibling:
                    if node.next_sibling.next_sibling and node.next_sibling.next_sibling.type == "arrow_function":
                        return True
                if node.parent.type == "call_expression":
                    if node.next_sibling.text.decode("utf-8") == ".":
                        return False
                    return True
                if node.parent.type == "arguments" :
                    if node.parent.parent and node.parent.parent.children[0]:
                        if node.parent.parent.children[0].children and node.parent.parent.children[0].children[-1].text.decode("utf-8") in ['map']:
                            return True
            return False

        functions = {'cpp':cpp,'python':python,'java':java,'javascript':javascript}
        for node in self.__traverse_tree(tree):
            function_name = node.text.decode("utf-8")
            if function_name == func_name:
                if functions[self.lang](node):
                    ret_code, diff = self.__replace_node_of_code(ret_code, node, substitute, diff)
        return ret_code
        
    

    def __generate_random_name(self,name:str,exist_words:set) -> str:
        
        def flip_char(c):
            if 'a' <= c <= 'z':
                return chr(122 - (ord(c) - 97))
            elif 'A' <= c <= 'Z':
                return chr(90 - (ord(c) - 65))
            elif '0' <= c <= '9':
                return str(9 - int(c))
            return c
        length = len(name)
        flip_nums = int(random.random() * length) + 1
        flip_indexs = list(set(random.randint(0,length - 1) for _ in range(flip_nums)))
        substitue = ""
        for i,v in enumerate(name):
            if i in flip_indexs:
                v = flip_char(v)
            substitue += v
        while substitue in exist_words:
            substitue = substitue + random.choice([chr(ord('a') + i) for i in range(26)])
        
        return substitue

    def random_filp_function_name(self,code:str) -> str:
        _,function_names = self.get_function_names(code)
        if not function_names:
            return code
        func_name = random.choice(function_names)
        exist_words = set(function_names)
        substitue = self.__generate_random_name(func_name,exist_words)
        code = self.rename_function_name(code,func_name,substitue)
        return self.__post_preprocess_code(code)

    def normalize_function_names(self,code:str) -> str:
        '''
        使用统一func_0,func_1的形式
        '''
        _,function_names = self.get_function_names(code)
        ret_code = code[:]
        for i,fcn in enumerate(function_names):
            ret_code = self.rename_function_name(ret_code,fcn,"func_" + str(i))[:]
        return self.__post_preprocess_code(ret_code)


    def syn_replace_functions(self,code:str) -> str:
        '''
        
        '''
        from nltk.corpus import wordnet
        _,function_names = self.get_function_names(code)
        ret_code = code[:]
        pattern = re.compile(r'[a-zA-Z_]\w*')
        exist_words = set(function_names)
        for i,fcn in enumerate(function_names):
            words = wordnet.synsets(fcn)
            if words:
                for oneword in words:
                    now = pattern.search(oneword.lemmas()[0].name()).group()
                    if  now in exist_words:
                        continue
                    ret_code = self.rename_function_name(ret_code,fcn,now)[:]
                    break
        return self.__post_preprocess_code(ret_code)

    def codebert_rename_func_name(self,code:str) -> str:
        '''
        使用codebert重命名所有func_name。选择prob最高的单词
        '''
        _,function_names = self.get_function_names(code)
        ret_code = code
        exist_words = set(function_names)
        pattern = re.compile(r'[a-zA-Z_]\w*')
        def rename_once(renamed_name,ret_code):
            nonlocal exist_words
            input_pred = self.rename_function_name(ret_code,renamed_name,"<mask>")
            input_pred = input_pred[:514]
            if "<mask>" in input_pred:
                for token_info in self.pipeline(input_pred):
                    if type(token_info) == list:
                        for x in token_info:
                            name = x['token_str'].strip()
                            name = pattern.search(name)
                            if name:
                                name = name.group()
                                if name not in exist_words:
                                    break
                    elif type(token_info) == dict:
                        name = token_info['token_str'].strip()
                        name = pattern.search(name)
                        if name:
                            name = name.group()
                    if name and self.__is_valid_identifier_name(name,self.lang) and name not in exist_words:
                        ret_code = self.rename_function_name(ret_code,renamed_name,name)
                        exist_words.add(name)
                        exist_words.remove(renamed_name)
                        break
            return ret_code
        
        for name in function_names:
            ret_code = rename_once(name,ret_code[:])[:]
        return ret_code

    def get_identifiers(self, code: str) -> Set[Tuple[str, tree_sitter.Node]]:
        '''
        获得一段代码中的所有标识符,及其对应于tree的节点node
        '''
        tree = self.__parser.parse(bytes(code, 'utf-8'))
        ret = set()
        names = set()
        def cpp(node:tree_sitter.Node):
            if node.type == "identifier":
                def recur(node):
                    if node.parent.type in ["parameter_declaration","init_declarator","declaration","reference_declarator"]:
                        return True
                    elif node.parent.type == 'for_range_loop':
                        return True
                    elif node.parent.type == "function_declarator":
                        return recur(node.parent)
                    return False

                return recur(node)

            return False

        def python(node:tree_sitter.Node):
            if node.type == "identifier":
                def recur(node:tree_sitter.Node):
                    if node.parent.type == "parameters": #函数形式参数
                        return True
                    if node.parent.type == "for_statement" and node.prev_sibling.text.decode("utf-8") == "for": #for循环变量
                        return True
                    if node.parent.type == "assignment": #赋值语句左值(提取到右值也没关系，因为右值只有function call可能是错误的，但function call节点的父节点为call)
                        if len(node.parent.children_by_field_name('left')) == 1:                            
                            if node.parent.child_by_field_name('right') and node.parent.child_by_field_name('right').type == "lambda":
                                return False
                        else:
                            index = -1
                            for i,child in enumerate(node.parent.children_by_field_name('left')):
                                if child.text.decode("utf-8") == node.text.decode("utf-8"):
                                    index = i
                                    break
                            for i,child in enumerate(node.parent.children_by_field_name('right')):
                                if child.type == "lambda" and i == index:
                                    return False
                        return True
                    if node.parent.type == "for_in_clause":
                        return True
                    if node.parent.type == "lambda_parameters":
                        return True
                    if node.parent.type == "pattern_list":
                        return recur(node.parent)
                    return False
                return recur(node)
            return False

        def java(node):
            if node.type == "identifier":
                if node.parent.type == "formal_parameter": #函数形式参数
                    return True
                if node.parent.type == "variable_declarator": #赋值语句左值
                    return True
                if node.parent.type == "enhanced_for_statement":#for循环变量
                    return True
            return False
                

        def javascript(node:tree_sitter.Node):
            if node.type == "identifier":
                if node.parent.type == "formal_parameters": #函数形式参数
                    return True
                if node.parent.type == "for_in_statement":
                    return True
                if node.parent.type == "arrow_function":
                    return True
                if node.parent.type == "variable_declarator": #变量定义语句左值
                    if node.next_sibling:
                        if node.next_sibling.next_sibling and node.next_sibling.next_sibling.type == "arrow_function":
                            return False
                    return True
            return False
        
        functions = {"cpp":cpp,'python':python,'java':java,'javascript':javascript}
        for node in self.__traverse_tree(tree):
            if functions[self.lang](node):
                name = node.text.decode("utf-8")
                if self.__is_valid_identifier_name(name,self.lang):
                    ret.add(node)
                    names.add(name)

        ret = list(ret)
        ret.sort(key=lambda x:x.start_byte)
        names = list(names)
        names.sort()
        return ret,names

    def __get_type_names(self,code:str) -> str:
        '''
        获得一段代码中的所有类型标识符,及其对应于tree的节点node
        '''
        tree = self.__parser.parse(bytes(code, 'utf-8'))
        ret = set()
        names = set()
        def cpp(node:tree_sitter.Node):
            if node.type in ["type_identifier","field_identifier"]:
                return True
            return False

        def python(node:tree_sitter.Node):
            if node.type == "identifier":
                return False

        def java(node):
            if node.type.endswith("type"):
                return True
            return False
                

        def javascript(node:tree_sitter.Node):
            if node.type == "identifier":
                return False
        
        functions = {"cpp":cpp,'python':python,'java':java,'javascript':javascript}
        for node in self.__traverse_tree(tree):
            if functions[self.lang](node):
                name = node.text.decode("utf-8")
                if self.__is_valid_identifier_name(name,self.lang):
                    ret.add(node)
                    names.add(name)

        ret = list(ret)
        ret.sort(key=lambda x:x.start_byte)
        names = list(names)
        names.sort()
        return ret,names

    def __get_import_names(self,code:str)->str:
        tree = self.__parser.parse(bytes(code, 'utf-8'))
        ret = set()
        names = set()
        def python(node:tree_sitter.Node):
            if node.type == "identifier" and node.parent.type in ["dotted_name","aliased_import"] and node.parent.parent.type in ["import_statement","import_from_statement"]:
                return True
            return False
        
        def cpp(node:tree_sitter.Node):
            return False

        
        def java(node:tree_sitter.Node):
            return False

        
        def javascript(node:tree_sitter.Node):
            return False
        
        functions = {"cpp":cpp,'python':python,'java':java,'javascript':javascript}
        for node in self.__traverse_tree(tree):
            if functions[self.lang](node):
                name = node.text.decode("utf-8")
                if self.__is_valid_identifier_name(name,self.lang):
                    ret.add(node)
                    names.add(name)

        ret = list(ret)
        ret.sort(key=lambda x:x.start_byte)
        names = list(names)
        names.sort()
        return ret,names



    def __rename_identifier(self, code: str, tgt_word: str, substitute: str) -> str:
        '''
        将一段代码的目标标识符tgt_word全部替换为substitute
        '''
        diff = 0
        ret_code = code
        tree = self.__parser.parse(bytes(code, 'utf-8'))
        def cpp(node:tree_sitter.Node):
            if node.type in ["identifier","type_identifier"]:
                if node.parent.type in ["qualified_identifier"]:
                    return False
                elif node.parent.type == "call_expression":
                    if node.next_sibling and node.next_sibling.text.decode("utf-8") == ".":
                        return True
                    return False
                elif node.parent.type == "function_declarator" and node.parent.parent.type == "function_definition":
                    return False
                return True
            elif node.type == "field_identifier":
                if node.parent.type == "field_expression" and node.parent.parent.type == "call_expression":
                    return False
                return True
            return False
        
        def python(node:tree_sitter.Node):
            if node.type == "identifier":
                if node.parent.type in ["function_definition"]:#不能是函数名
                    return False
                elif node.parent.type == "attribute" and node.parent.parent.type == "call" and node.prev_sibling:#不能是a.x()中的x (a1.x1.x2()中的x1,x2也不能是)
                    return False
                elif node.parent.type == "call":
                    if node.next_sibling and node.next_sibling.text.decode("utf-8") == ".":#x.y()也是一种call，但x是一个变量，不是一个function,y才是一个function
                        return True
                    return False
                return True
            return False

        def java(node:tree_sitter.Node):
            if node.type == "identifier": #没有改任何函数定义的名称，那么函数调用时，也不能改
                if node.parent.type in ["method_declaration"]: #不能是函数名
                    return False
                elif node.parent.type in ["method_invocation"]: # :  #可以是x.y()中的x,但不能是y，也不能是x()中的x
                    if node.prev_sibling and node.prev_sibling.text.decode("utf-8") == ".": #有前一个兄弟，说明是x.y()中的y
                        return False
                    elif node.next_sibling and node.next_sibling.type == "argument_list": #下一个兄弟就是arguments，说明是x()中的x
                       return False
                    return True
                return True
            return False
        
        def javascript(node:tree_sitter.Node):
            if node.type == "identifier":
                if node.parent.type in ["function_declaration"]:#不能是函数名
                    return False
                if node.parent.type == "arguments":#是一个实际参数
                    return True
                if node.next_sibling:
                    if node.next_sibling.next_sibling:
                        if node.next_sibling.next_sibling.type == "arrow_function":
                            return False
                if node.parent.type == "call_expression":
                    if node.next_sibling and node.next_sibling.text.decode("utf-8") == ".":
                        return True
                    return False
                return True
            return False

        functions = {'cpp':cpp,'python':python,'java':java,'javascript':javascript}
        for node in self.__traverse_tree(tree):
            identifier = node.text.decode("utf-8")
            if identifier == tgt_word:
                if functions[self.lang](node):
                    ret_code, diff = self.__replace_node_of_code(ret_code, node, substitute, diff)
        return ret_code
    
    def random_flip_identifier(self,code:str) -> str:
        """
        字符串随机翻转
        IT1
        """
        #对其中的某个identifier进行随机翻转
        identifier_and_node,identifier_names = self.get_identifiers(code)
        exist_words = set(identifier_names)
        ret_code = code
        if len(identifier_names) == 0:
            return ret_code
        get_flip_identifier = random.choice(identifier_names)
        substitue = self.__generate_random_name(get_flip_identifier,exist_words)
        if self.__is_valid_identifier_name(substitue,self.lang):
            ret_code = self.__rename_identifier(ret_code,get_flip_identifier,substitue)
        return self.__post_preprocess_code(ret_code)

    def normalize_identifer(self,code:str) -> str:
        '''
        使用统一var_0,var_1的形式
        '''
        _,identifier_names = self.get_identifiers(code)
        ret_code = code[:]
        for i,idtfs in enumerate(identifier_names):
            if self.__is_valid_identifier_name("var_" + str(i),self.lang):
                ret_code = self.__rename_identifier(ret_code,idtfs,"var_" + str(i))[:]
        return self.__post_preprocess_code(ret_code)

    


    def codebert_rename_identifier(self,code:str) -> str:
        '''
        使用codebert重命名所有identifier。选择prob最高的单词
        '''
        _,identifier_names = self.get_identifiers(code)
        _,function_names = self.get_function_names(code)
        _,import_names = self.__get_import_names(code)
        _,type_names = self.__get_type_names(code)
        ret_code = code
        exist_words = set(identifier_names+function_names+import_names+type_names)
        pattern = re.compile(r'[a-zA-Z_]\w*')
        def rename_once(renamed_name,ret_code):
            nonlocal exist_words
            input_pred = self.__rename_identifier(ret_code,renamed_name,"<mask>")
            input_pred = input_pred[:514]
            if "<mask>" in input_pred:
                for token_info in self.pipeline(input_pred):
                    if type(token_info) == list:
                        for x in token_info:
                            name = x['token_str'].strip()
                            name = pattern.search(name)
                            if name:
                                name = name.group()
                                if name not in exist_words:
                                    break
                    elif type(token_info) == dict:
                        name = token_info['token_str'].strip()
                        name = pattern.search(name)
                        if name:
                            name = name.group()
                    if name and self.__is_valid_identifier_name(name,self.lang) and name not in exist_words:
                        ret_code = self.__rename_identifier(ret_code,renamed_name,name)
                        exist_words.add(name)
                        exist_words.remove(renamed_name)
                        break
            return ret_code
        
        for name in identifier_names:
            ret_code = rename_once(name,ret_code[:])[:]
        return ret_code
    
    def syn_replace_identifier(self,code:str) -> str:
        from nltk.corpus import wordnet
        _,identifier_names = self.get_identifiers(code)
        ret_code = code[:]
        pattern = re.compile(r'[a-zA-Z_]\w*')
        exist_words = set(identifier_names)
        for i,idtn in enumerate(identifier_names):
            words = wordnet.synsets(idtn)
            if words:
                for oneword in words:
                    now = pattern.search(oneword.lemmas()[0].name()).group()
                    if now in exist_words:
                        continue
                    ret_code = self.__rename_identifier(ret_code,idtn,now)[:]
                    break
        return self.__post_preprocess_code(ret_code)
    
    def bool2int(self,code):
        '''
        将一段代码中的所有true,false替换为1,0
        '''
        diff = 0
        ret_code = code
        tree = self.__parser.parse(bytes(code, 'utf-8'))
        for node in self.__traverse_tree(tree):
            if node.type.lower() == 'true':
                ret_code,diff = self.__replace_node_of_code(ret_code,node,'1',diff)
            elif node.type.lower() == 'false':
                ret_code,diff = self.__replace_node_of_code(ret_code,node,'0',diff)
        return self.__post_preprocess_code(ret_code)

    def __find_all_base_var_type(self,code):
        '''
        找到一段代码中的所有int,char*,float定义的位置
        '''
        int_pattern = r'\bint\b(?! *main\(\))'
        char_star_pattern = r'\bchar\*\b'
        float_pattern = r'\bfloat\b'
        int_matches = [(match.start(), match.end()) for match in re.finditer(int_pattern, code)]
        char_matches = [(match.start(), match.end()) for match in re.finditer(char_star_pattern, code)]
        float_matches = [(match.start(), match.end()) for match in re.finditer(float_pattern, code)]
        return {'int':int_matches,'char*':char_matches,'float':float_matches}


    def more_universe_var_type(self,code):
        '''
        将一段代码的变量类型转换为更通用的类型,随机替换一个
        '''
        candidate = self.__find_all_base_var_type(code)
        verse_map = {'int':'long','char*':'string','float':'double'}
        types = [key for key,value in candidate.items() if value]
        if len(types) == 0:
            return code
        substitute_type = random.choice(types)
        now_candidate = candidate[substitute_type]
        length = len(now_candidate)
        substitute_pos = now_candidate[random.randint(0,length-1)]
        ret_code = code[:substitute_pos[0]] + verse_map[substitute_type] + code[substitute_pos[1]:]
        return self.__post_preprocess_code(ret_code)
    
    def tab_indent(self,code):
        '''
        将制表符替换为4个空格
        '''
        return re.sub(r'\t',' '*4,code)
    

    def line_split(self,code):
        '''
        将一段代码最长的一行拆分为两行,拆分的位置在最中间单词
        '''
        code_lst = code.split('\n')
        max_length = 0
        diff = 0
        ret_code = code
        tree = self.__parser.parse(bytes(code, 'utf-8'))
        max_length = -1
        max_length_index = -1
        for i,v in enumerate(code_lst):
            if len(v) > max_length:
                max_length = len(v)
                max_length_index = i
        token_map = {'java':"",'cpp':"\\",'javascript':"",'python':'\\'}
        for node in self.__traverse_tree(tree):
            if node.start_point[0] <= max_length_index <= node.end_point[0]:
                if not node.children and node.type.count("return") <= 0: #如果没有孩子，说明是标识符或者常量。并且不能是return语句
                    ret_code,diff = self.__replace_node_of_code(ret_code,node,node.text.decode("utf-8") + token_map[self.lang] + "\n",diff)
                    break
        return self.__post_preprocess_code(ret_code)

    def doc2comments(self,code):
        '''
        将文档字符串转换为注释行
        '''
        def python_handler(code):
            pattern = '[\'\"]{3}(?P<docstring>.*?)[\'\"]{3}'
            matches = re.finditer(pattern, code, re.DOTALL)
            new_code = code
            for match in reversed(list(matches)):
                docstring = match.group('docstring').strip()
                docstring_lst = ["#" + oneline.strip() for oneline in docstring.split('\n')]
                new_code = new_code[:match.start()] + f"\n".join(docstring_lst) + new_code[match.end():]
            return new_code

        def java_cpp_javascript_handler(code):
            pattern = r'\/\*(?P<docstring>.*?)\*/'
            matches = re.finditer(pattern, code, re.DOTALL)
            new_code = code
            for match in reversed(list(matches)):
                docstring = match.group('docstring').strip()
                docstring_lst = ["//" + oneline.strip() for oneline in docstring.split('\n')]
                new_code = new_code[:match.start()] + f"\n".join(docstring_lst) + new_code[match.end():]
            return new_code

        if self.lang == 'python':
            return self.__post_preprocess_code(python_handler(code))
        elif self.lang in ['java', 'cpp', 'javascript']:
            return self.__post_preprocess_code(java_cpp_javascript_handler(code))
        return self.__post_preprocess_code(code)
    
    def newline_afterdoc(self,code):
        '''
        文档字符串后插入新行
        '''
        if self.lang == 'python':
            pattern = r'[\'\"]{3}(?P<docstring>.*?)[\'\"]{3}'
        elif self.lang in ['java', 'cpp', 'javascript']:
            pattern = r'\/\*(?P<docstring>.*?)\*/'
        else:
            return code[:]
        matches = re.finditer(pattern, code, re.DOTALL)
        new_code = code
        for match in reversed(list(matches)):
            end_pos = match.end()
            new_code = new_code[:end_pos] + '\n' + new_code[end_pos:]
        return self.__post_preprocess_code(new_code)


    def newline_random(self,code):
        '''
        随机在代码中插入空行
        '''
        code_lst = code.split('\n')
        index = random.randint(0,len(code_lst) - 1)
        code_lst = code_lst[:index] + ['\n'] + code_lst[index:]
        return self.__post_preprocess_code('\n'.join(code_lst))

    def newline_aftercode(self,code):
        '''
        这个变换在代码的末尾插入一个空行。
        '''
        return self.__post_preprocess_code(code + "\n")
    
    def __generate_hard_dead_code(self):
        '''
        生成较为复杂的死代码
        '''


    def __generate_simple_dead_code(self, existed_words,ident_level):
        '''
        生成死代码
        '''
        prompt = "This function is used to remove a item of a list"
        count = len(existed_words)
        var_name = f"temp_var_{count + 1}"
        dead_code_templates = {
            "python": [
                f"{var_name} = 9.9\n{ident_level}if {var_name} < 9.12: print('{prompt}'); {var_name} += 1",
                f"{var_name} = 9.9\n{ident_level}while {var_name} < 9.12: print('{prompt}'); {var_name} += 1",
                f"lambda: print('{prompt}')",
                f"{var_name} = 9.9\n{ident_level}print('{prompt}')",
            ],
            "java": [
                f"double {var_name} = 9.9;if ({var_name} < 9.12) {{System.out.println(\"{prompt}\");{var_name} = {var_name} + 1;}}",
                f"double {var_name} = 9.9;while ({var_name} < 9.12) {{System.out.println(\"{prompt}\");{var_name} = {var_name} + 1;}}",
                f"new Runnable() {{@Override public void run() {{System.out.println(\"{prompt}\");}}}};",
                f"double {var_name} = 9.9;System.out.println(\"{prompt}\");",
            ],
            "cpp": [
                f"double {var_name} = 9.9;if ({var_name} < 9.12) {{std::cout << \"{prompt}\" << std::endl;{var_name} = {var_name} + 1;}}",
                f"double {var_name} = 9.9;while ({var_name} < 9.12) {{std::cout << \"{prompt}\" << std::endl;{var_name} = {var_name} + 1;}}",
                f"[]() {{std::cout << \"{prompt}\" << std::endl;}};",
                f"double {var_name} = 9.9;std::cout << \"{prompt}\" << std::endl;",
            ],
            "javascript": [
                f"let {var_name} = 9.9;if ({var_name} < 9.12) {{{var_name} = {var_name} + 1;}}",
                f"let {var_name} = 9.9;while ({var_name} < 9.12) {{{var_name} = {var_name} + 1;}}",
                f"let {var_name} = 9.9;{var_name} = {var_name} + 1;",
                f"(function() {{let {var_name} = 9.9; {var_name} = {var_name} + 1;}})();"
            ],
        }

        if self.lang not in dead_code_templates:
            raise Exception(f"Unsupported language: {self.lang}")

        template = random.choice(dead_code_templates[self.lang])
        return template
    
    def insert_dead_code(self,code):
    
        ret_code = code
        tree = self.__parser.parse(bytes(ret_code, 'utf-8'))
        insert_pos = Counter()
        types = {'python':"block",'cpp':'compound_statement','java':'block','javascript':'statement_block'}
        for node in self.__traverse_tree(tree):
            if self.lang in ['cpp','java','javascript']:
                if node.type == types[self.lang]:
                    for num in range(node.start_point[0],node.end_point[0]):
                        insert_pos[num] += 1
                elif node.type.count("statement"):
                    for num in range(node.start_point[0],node.end_point[0]):
                        insert_pos[num] -= 1
            elif self.lang == 'python':
                if node.type == "block":
                    for num in range(node.start_point[0],node.end_point[0] + 1):
                        insert_pos[num] += 1
                elif node.type.count("statement"):
                    for num in range(node.start_point[0],node.end_point[0]):
                        insert_pos[num] -= 1
        if not insert_pos:
            pass
            #print(ret_code)
        insert_pos = [i for i,v in insert_pos.items() if v > 0]
        lines = ret_code.split("\n")
        insert_pos.append(len(lines) - 2)
        pos = random.choice(insert_pos)
        _,existed_words = self.get_identifiers(ret_code)
        ident_level = self.__get_indent_level(lines[pos])
        dead_code = self.__generate_simple_dead_code(set(existed_words),ident_level)
        #print(pos)
        lines = lines[:pos + 1] + [ident_level + dead_code] + lines[pos + 1:]
        return self.__post_preprocess_code('\n'.join(lines))
    
    def insert_comment(self,code):
        '''
        在代码中任意一行插入注释
        注释内容: this is a comment line.
        '''
        lines = code.split('\n')
        index = random.randint(0,len(lines) - 1)
        comment_token = {'python':'#','java':r'//','javascript':r'//','cpp':r'//'}
        comments = 'This is a comment line.'
        lines = lines[:index] + [comment_token[self.lang] + comments] + lines[index:]
        return self.__post_preprocess_code('\n'.join(lines))

    def remove_comments(self, code):
        '''
        删除所有注释
        '''
        diff = 0
        ret_code = code
        tree = self.__parser.parse(bytes(code, 'utf-8'))
        for node in self.__traverse_tree(tree):
            if node.type == "comment":
                ret_code,diff = self.__replace_node_of_code(ret_code,node,"",diff)
        return self.__post_preprocess_code(ret_code)
    

    def for_var_inner(self,code):
        '''
        for循环变量内部化
        '''
        try:
            if self.lang in ['python']:
                return code[:]
            lines = code.split('\n')
            pattern = re.compile(r"for\s*\((?P<id_definition>[^;]*);[^;]+;(?P<recur_var>[^\)]+)\)")
            indexs = []
            for i,line in enumerate(lines):
                matched = pattern.search(line)
                if matched:
                    id_definition = matched.group('id_definition')
                    if len(id_definition.strip()) == 0:
                        indexs.append(i)
            if len(indexs) == 0:
                return code
            index = random.choice(indexs)
            line = lines[index]
            matched = pattern.search(line)
            recur_var = matched.group('recur_var')
            #提取出recur_var中的变量名
            var_name = re.search("[a-zA-Z0-9_]+",recur_var).group()
            def is_valid_variable_definition(s1, s2):
                pattern = r'^\s*(int|float|double|char|long|short|boolean|byte|String|var|let|const|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(=.*)?\s*;?\s*$'
                match = re.match(pattern, s2)
                if match:
                    variable_name = match.group(2)
                    return s1 == variable_name,' '.join(match.groups())
                else:
                    return False,None
            difinition_stmt = ""
            difinition_index = -1
            for i,line in enumerate(lines):
                f,stmt = is_valid_variable_definition(var_name,line.strip())
                if f:
                    difinition_stmt = stmt
                    difinition_index = i
            
            if difinition_index == -1:
                #没有找到变量定义语句
                return code
            st = ""
            for i,v in enumerate(lines[index]):
                if v == ";":
                    lines[index] = st + difinition_stmt + lines[index][i:]
                    break
                st = st + v
            del lines[difinition_index]

            return self.__post_preprocess_code('\n'.join(lines))
        except:
            return code
        

    
    def __get_indent_level(self,code):
        '''
        获取某行的缩进级别
        '''
        ret = ""
        for i,v in enumerate(code):
            if v == ' ':
                ret += v
            elif v == '\t':
                ret += v
            else:
                break
        return ret
    
    def for_var_outer_once(self,code):
        '''
        对代码中的一个for循环变量外部化
        '''
        try:
            if self.lang in ['python']:
                return code[:]
            
            diff = 0
            ret_code = code
            name_map = {'cpp':{'i':'initializer','c':'condition','b':'body','u':'update'},\
                        'java':{'i':'init','c':'condition','b':'body','u':'update'},\
                            'javascript':{'i':'initializer','c':'condition','b':'body','u':'increment'}}
            for_nodes = []
            tree = self.__parser.parse(bytes(code, 'utf-8'))
            for node in self.__traverse_tree(tree):
                if node.type == "for_statement":
                    initializer = node.child_by_field_name(name_map[self.lang]['i'])
                    if initializer and len(initializer.text.decode("utf-8").strip()) > 1:
                        for_nodes.append(node)
            for_nodes.sort(key=lambda x:x.end_byte)

            if len(for_nodes) > 0:
                node = for_nodes[0]
                initializer = node.child_by_field_name(name_map[self.lang]['i']).text.decode("utf-8").strip(";")
                for_node_text = node.text.decode("utf-8")
                for_node_text = for_node_text.replace(initializer,"")
                lines = ret_code.split("\n")
                indent_level = self.__get_indent_level(lines[node.start_point[0]])
                replaced = indent_level + " " * 4 + initializer + ";\n"
                for_node_text = for_node_text.split("\n")
                for i,v in enumerate(for_node_text):
                    for_node_text[i] = indent_level + v
                for_node_text[0] = " " * 4 + for_node_text[0]
                for_node_text = '\n'.join(for_node_text)
                replaced += for_node_text + "\n"
                text_end = "{\n" + replaced + indent_level + "}"
                ret_code,diff = self.__replace_node_of_code(ret_code,node,text_end,diff)

            return self.__post_preprocess_code(ret_code),len(for_nodes) - 1
        except:
            return code
    

    def for_var_outer(self,code:str) -> str:
        try:
            if self.lang in ['python']:
                return code[:]
            ret_code,nums = self.for_var_outer_once(code)
            while nums > 0:
                ret_code,nums= self.for_var_outer_once(ret_code)
            
            return ret_code
        except:
            return code
    
    def un_relate_package_import_insert(self,code:str) -> str:
        '''
        插入无关的或重复的包导入
        cpp:#include
        python:import
        java:import
        javascript:import
        '''
        import_stmt = {
            'python':'import',
            'cpp':'#include',
            'java':'import',
            'javascript':'const',
        }
        modules = base_module[self.lang]
        random_module = random.choice(modules)
        end_token = {
            'python':'',
            'cpp':'',
            'javascript':';',
            'java':';'
        }
        if self.lang == "javascript":
            code = import_stmt[self.lang] + f" {random_module} = require('{random_module}'){end_token[self.lang]}\n{code}"
        else:
            code = import_stmt[self.lang] + f" {random_module}{end_token[self.lang]}\n{code}"
        return self.__post_preprocess_code(code)
    
    def __for2whileonce(self,code,exist_words,count):
        '''
        将代码中一个for循环转换为while循环
        '''
        if self.lang == 'python':
            return code,0
        '''
        for循环结构
        for({循环变量定义};{循环条件};循环迭代)
        {
            循环执行语句
        }

        while循环结构
        循环变量定义
        while({循环条件})
        {
            循环执行语句
            循环迭代
        }
        '''
        diff = 0
        ret_code = code
        tree = self.__parser.parse(bytes(code, 'utf-8'))
        lines = code.split('\n')
        name_map = {'cpp':{'i':'initializer','c':'condition','b':'body','u':'update'},\
                    'java':{'i':'init','c':'condition','b':'body','u':'update'},\
                        'javascript':{'i':'initializer','c':'condition','b':'body','u':'increment'}}
        for_nodes = []
        for node in self.__traverse_tree(tree):
            if node.type == "for_statement":
                for_nodes.append(node)
        for_nodes.sort(key=lambda x:x.end_byte)
        

        if len(for_nodes) > 0:
            node = for_nodes[0]
            initializer = node.child_by_field_name(name_map[self.lang]['i']).text.decode("utf-8").strip(";")
            pattern = re.compile(r'(?P<type>.+)?(\s+)?(?P<var>\w+)\s*\=\s*(?P<right>.+)')
            condition = node.child_by_field_name(name_map[self.lang]['c']).text.decode("utf-8").rstrip(';')
            body = node.child_by_field_name(name_map[self.lang]['b']).text.decode("utf-8")
            body = body.lstrip(r"\{")
            body = body.rstrip(r"\}")
            update = node.child_by_field_name(name_map[self.lang]['u']).text.decode("utf-8")
            body = body.split('\n')
            for i,line in enumerate(body):
                if line.count("continue") > 0:
                    pos = line.find("continue")
                    body[i] = line[:pos] + update + ";" + line[pos:]
            body = '\n'.join(body)
            replaced = ""
            count = 0
            try:
                if pattern.search(initializer).group('type'):
                    replaced += '{\n'
                    count = 4
            except:
                return code
            replaced += self.__get_indent_level(lines[node.start_point[0]]) + " " * 4 + initializer + ';\n'
            replaced += self.__get_indent_level(lines[node.start_point[0]]) + " " * 4 + f"while({condition})" + '{' + body
            replaced += self.__get_indent_level(lines[node.start_point[0]]) + " " * 4 + update + ";\n"
            replaced += self.__get_indent_level(lines[node.start_point[0]]) + " " * 4 + "}"
            if pattern.search(initializer).group('type'):
                replaced += '\n' + self.__get_indent_level(lines[node.start_point[0]]) + '}'
            ret_code,diff = self.__replace_node_of_code(ret_code,node,replaced,diff)
            return ret_code,len(for_nodes) - 1
        return code,0
    
    def for2while(self,code):
        try:
            if self.lang == 'python':
                return code
            exist_words = set()
            count = 0
            ret_code,for_node_num = self.__for2whileonce(code,exist_words,count)
            while for_node_num > 0:
                ret_code,for_node_num = self.__for2whileonce(ret_code,exist_words,count)
            return ret_code
        except:
            return code
                
                
    
    def while2for(self,code):
        '''
        将while循环转换为for循环
        '''
        try:
            if self.lang == 'python':
                return code
            diff = 0
            ret_code = code
            tree = self.__parser.parse(bytes(code, 'utf-8'))
            for node in self.__traverse_tree(tree):
                if node.type == "while_statement":
                    body = node.child_by_field_name('body').text.decode("utf-8").lstrip("{").rstrip('}')
                    condition = node.child_by_field_name('condition').text.decode("utf-8").rstrip(';')
                    replaced = f"for(;{condition};)"
                    replaced += "{" + body + "}"
                    ret_code,diff = self.__replace_node_of_code(ret_code,node,replaced,diff)
            return ret_code
        except:
            return code

    
    def exchange_operators(self,code:str) -> str:
        '''
        判断具有交换律的操作符,并转换前后操作数
        '''
        diff = 0
        ret_code = code
        tree = self.__parser.parse(bytes(code, 'utf-8'))
        '''
        +,*在数学运算中具有交换律
        ==,!=,&&,||在逻辑运算中具有交换律 &&和||在python中是and 和 or
        &,|,^在整数之间的按位运算中具有交换律
        '''
        math_ops = ['+','*']
        com_ops = ['==','!=']
        logic_ops = ['&&','||']
        bit_ops = ['&','|','^']

        def node_is_type(node:tree_sitter.Node,type:list):
            if node.type in type:
                return True
            ret = False
            for child in node.children:
                ret = ret | node_is_type(child,type)
            return ret

        def python(node:tree_sitter.Node):
            if node.type in ["binary_operator","boolean_operator"]:
                ops = node.child_by_field_name('operator')
                left = node.child_by_field_name('left')
                right = node.child_by_field_name('right')
                ops_text = ops.text.decode("utf-8")
                if node.type == "binary_operator":
                    if ops_text in math_ops:
                        if not node_is_type(left,['string','identifier']) or not node_is_type(right,['string','identifier']):#只要有一个是整数或浮点数就行，因为python二元操作数运算只能和相同类型运算
                            return right,ops,left
                    elif ops_text in bit_ops:
                        if node_is_type(left,['integer']) or node_is_type(right,['integer']):#我们不考虑identifier，因为python的identifier类型我们并不知道
                            return right,ops,left
                elif node.type == "boolean_operator":
                    #python中的and和or不完全具备交换律，因为python在判断and时是从前往后进行判断。有时候交换了会引起index out of range等错误
                    if ops.type in ['and','or']:
                        return None,None,None
                    return right,ops,left
            elif node.type == "comparison_operator" and len(node.children) == 3:
                ops,left,right = node.children[1],node.children[0],node.children[2]
                if ops.text.decode("utf-8") in com_ops:
                    return right,ops,left
            return None,None,None
        
        def cpp(node:tree_sitter.Node):
            if node.type == "binary_expression":
                ops = node.child_by_field_name('operator')
                left = node.child_by_field_name('left')
                right = node.child_by_field_name('right')
                ops_text = ops.text.decode("utf-8")
                if ops_text == "+":
                    if not node_is_type(left,['string_literal','identifier']) or not node_is_type(right,['string_literal','identifier']):
                        return right,ops,left
                elif ops_text in math_ops + com_ops  + bit_ops:
                    return right,ops,left
            return None,None,None



        def java(node:tree_sitter.Node):
            if node.type == "binary_expression":
                ops = node.child_by_field_name('operator')
                left = node.child_by_field_name('left')
                right = node.child_by_field_name('right')
                ops_text = ops.text.decode("utf-8")
                if ops_text == "+":
                    if not node_is_type(left,['string_literal','identifier']) or not node_is_type(right,['string_literal','identifier']):
                        return right,ops,left
                elif ops_text in math_ops + com_ops  + bit_ops:
                    return right,ops,left
            return None,None,None

        def javascript(node:tree_sitter.Node):
            if node.type == "binary_expression":
                ops = node.child_by_field_name('operator')
                left = node.child_by_field_name('left')
                right = node.child_by_field_name('right')
                ops_text = ops.text.decode("utf-8")
                if ops_text == "+":
                    if not node_is_type(left,['string','identifier']) or not node_is_type(right,['string','identifier']):
                        return right,ops,left
                elif ops_text in math_ops + com_ops  + bit_ops + ['===']:
                    return right,ops,left
            return None,None,None
        
        functions = {'python':python,'cpp':cpp,'java':java,'javascript':javascript}
        for node in self.__traverse_tree(tree):
            left,ops,right = functions[self.lang](node)
            if left and ops and right:
                if not left.children and not right.children:
                    ret_code,diff = self.__replace_node_of_code(ret_code,node,left.text.decode("utf-8")\
                                                                + " " + ops.text.decode("utf-8")\
                                                                    + " " + right.text.decode("utf-8"),diff)
        tree = self.__parser.parse(bytes(ret_code, 'utf-8'))
        existed_nodes = [len(ret_code.encode("utf-8")) + 1,-1] #已经交换过的节点的最小起始点和最大终止点
        diff = 0
        for node in self.__traverse_tree(tree):
            left,ops,right = functions[self.lang](node)
            if left and ops and right:
                if  left.children or right.children:
                    if (node.start_byte <= existed_nodes[1] and node.start_byte >= existed_nodes[0]) or (node.end_byte >= existed_nodes[0] and node.end_byte <= existed_nodes[1]):
                        continue
                    ret_code,diff = self.__replace_node_of_code(ret_code,node,left.text.decode("utf-8")\
                                                                + " " + ops.text.decode("utf-8")\
                                                                    + " " + right.text.decode("utf-8"),diff)
                    if existed_nodes[0] > node.start_byte:
                        existed_nodes[0] = node.start_byte
                    if existed_nodes[1] < node.end_byte:
                        existed_nodes[1] = node.end_byte
            
        return ret_code

    def __definition_stmt(self,var_name, var_type, var_value):
        # 根据语言映射到对应的类型
        lang = self.lang
        try:
            if lang.lower() in ["python", "javascript"]:
                # 这些语言定义变量时不需要类型
                if lang.lower() == "python":
                    return f"{var_name} = {var_value}"
                elif lang.lower() == "javascript":
                    return f"const {var_name} = {var_value};"
            else:
                # 对于其他语言，需要转换常量类型到变量类型
                var_types = all_languages_constant_variable_types[lang.lower()][var_type]
                if not var_types:
                    raise ValueError("Unsupported constant type for the language.")
                var_type_str = var_types[0]  # 选择最通用的类型
                if lang.lower() == "cpp":
                    return f"{var_type_str} {var_name} = {var_value};"
                elif lang.lower() == "java":
                    return f"{var_type_str} {var_name} = {var_value};"
        except KeyError:
            return "Unsupported language or constant type."
    
    def fold_constant(self,code:str) -> str:
        '''
        检测代码中的常量,并将常量定义为一个变量
        变量的名称为var_0,var_1...如有变量名冲突,以此类推
        变量定义语句的插入位置:常量所在statement的前一行
        '''
        def python(node:tree_sitter.Node):
            '''
            python中的常量:
            false,true,float,integer,none,string
            '''
            if node.type in python_literal:
                return True
            return False
        
        def cpp(node:tree_sitter.Node):
            '''
            cpp中的常量:
            string_literal,char_literal,number_literal

            string_literal:string_content
            char_literal:character
            number_literal
            '''
            if node.type in cpp_literal and node.parent.type != "throw_statement":
                return True
            return False

        def java(node:tree_sitter.Node):
            '''
            java中的常量:
            _literal
            下面再细分
            '''
            if node.type in java_literal:
                return True
            return False
        
        def javascript(node:tree_sitter.Node):
            '''
            javascript中的常量:

            '''
            if node.type in javascript_literal and (node.parent!='pair' or node.parent.children[0] != node):
                return True
            return False
        
        

        ret_code = code
        diff = 0
        tree = self.__parser.parse(bytes(ret_code, 'utf-8'))
        functions = {'python':python,'cpp':cpp,'java':java,'javascript':javascript}
        var_count = 0
        _,existed_var_names = self.get_identifiers(code)
        existed_var_names = set(existed_var_names)
        line_diff = 0

        change_nodes = []
        vistied = set()

        statament_flag = {'python':"statement","cpp":'statement',"java":"","javascript":"statement"}
        module_flag = {'python':["module"],"cpp":['compound_statement'],"java":["block","class_body"],"javascript":['statement_block']}

        def dfs(node:tree_sitter.Node):
            if not node or node in vistied: 
                return
            vistied.add(node)
            if functions[self.lang](node):
                statement_parent = node
                while statement_parent.type.count(statament_flag[self.lang]) <= 0 or self.lang in ["cpp","java","javascript"]:
                    statement_parent = statement_parent.parent
                    if statement_parent.type in module_flag[self.lang] or  not statement_parent.parent:
                        break
                stmt_line = statement_parent.start_point[0]
                if statement_parent.type in module_flag[self.lang]:
                    stmt_line += 1
                f = False
                if self.lang == 'java' and statement_parent.type == "class_body": #针对于java的静态类型声明
                    f = True

                change_nodes.append((node,stmt_line,f))
            children = node.children
            for child in children:
                dfs(child)

        dfs(tree.root_node)
        change_nodes.sort(key = lambda x:x[0].end_byte)
        insert_stmts = []
        for node,stmt_line,f in change_nodes:
            if functions[self.lang](node):
                var_name = "var_" + str(var_count)
                while var_name in existed_var_names:
                    var_count += 1
                    var_name = "var_" + str(var_count)
                existed_var_names.add(var_name)
                ret_code,diff = self.__replace_node_of_code(ret_code,node,f"{var_name} ",diff) #先替换，再插入
                lines = ret_code.split("\n")
                try:
                    indent = self.__get_indent_level(lines[stmt_line + line_diff])
                except:
                    indent = self.__get_indent_level(lines[-1])
                definition_stmt = indent + str("static " if f else "") + self.__definition_stmt(var_name,node.type,node.text.decode("utf-8")) #static是针对于Java的静态类型声明
                insert_stmts.append((definition_stmt,stmt_line))
        
        insert_stmts.sort(key=lambda x:x[1])
        for definition_stmt,stmt_line in insert_stmts:
            lines = ret_code.split("\n")
            lines = lines[:stmt_line + line_diff] + [definition_stmt] + lines[stmt_line + line_diff:]
            ret_code = '\n'.join(lines)
            line_diff += 1
        
        return ret_code
    
    def __get_var_type_value(self,code,var_name):
        '''
        获得一个变量的类型信息
        '''
        
        def cpp(node:tree_sitter.Node):
            if node.type in ['declaration']:
                pattern = re.compile(r'\b\w+\b')
                type = node.child_by_field_name('type')
                declarator = node.child_by_field_name('declarator')
                if declarator.type == "init_declarator":
                    value = declarator.child_by_field_name('value')
                    if var_name in pattern.findall(declarator.text.decode("utf-8"))[:2]:
                        return value,type
            elif node.type == "for_range_loop":
                identifier = node.child_by_field_name('declarator')
                type = node.child_by_field_name('type')
                declarator = node.child_by_field_name('right')
                if identifier.text.decode("utf-8") == var_name:
                    return "NAC",type
            elif node.type == "parameter_list":
                for parameter_declaration in node.children:
                    type = parameter_declaration.child_by_field_name('type')
                    declarator = parameter_declaration.child_by_field_name('declarator')
                    if type and declarator:
                        pattern = re.compile(r'\b\w+\b')
                        if var_name in pattern.findall(declarator.text.decode("utf-8"))[:2]:
                            return "NAC",type #先写一个单函数的常量传播,COT表示函数形式参数是一个常量,但具体的值是多少未知
            return None,None


        def java(node:tree_sitter.Node):
            if node.type in ['variable_declarator']:
                name = node.child_by_field_name('name').text.decode("utf-8")
                type = node.parent.child_by_field_name('type')
                if var_name == name:
                    return None,type
            return None,None
        
        def python(node:tree_sitter.Node):
            pass

        def javascript(node:tree_sitter.Node):
            pass

        functions = {'cpp':cpp,'java':java,'python':python,'javascript':javascript}
        
        tree = self.__parser.parse(bytes(code, 'utf-8'))
        for node in self.__traverse_tree(tree):
            ret = functions[self.lang](node)
            if ret and ret[0] and ret[1]:
                return ret
        return None,None
                

    def equal_expression_transformation(self,code):
        '''
        等价表达式转换
        '''

        def cpp(code,diff,node:tree_sitter.Node):
            ret_code = code
            if node.type in ['assignment_expression']: 
                op = node.child_by_field_name('operator')
                op_text = op.text.decode("utf-8").replace("=","")
                if len(op_text.strip()) <= 0:
                    return ret_code,diff
                left = node.child_by_field_name('left').text.decode("utf-8")
                right = node.child_by_field_name('right').text.decode("utf-8")
                replaced = left + " = " + left + " " + op_text + " (" + right + ")"
                ret_code,diff = self.__replace_node_of_code(code,node,replaced,diff)
            elif node.type in ['update_expression']:
                if not node.parent.type.endswith("statement"):
                    return ret_code,diff
                op = node.child_by_field_name('operator').text.decode("utf-8")[0]
                left = node.child_by_field_name('argument').text.decode("utf-8")
                _,type = self.__get_var_type_value(ret_code,left)
                if type and type.text.decode("utf-8") in ["int","long","double","float"]:
                    replaced = left + " = " + left + " " + op + " " + "1"
                    ret_code,diff = self.__replace_node_of_code(code,node,replaced,diff)
            return ret_code,diff
        
        def python(code,diff,node:tree_sitter.Node):
            ret_code = code
            if node.type in ['augmented_assignment']:
                op = node.child_by_field_name('operator')
                op_text = op.text.decode("utf-8").replace("=","")
                if len(op_text.strip()) <= 0:
                    return ret_code,diff
                left = node.child_by_field_name('left').text.decode("utf-8")
                right = node.child_by_field_name('right').text.decode("utf-8")
                replaced = left + " = " + left + " " + op_text + " (" + right + ")"
                ret_code,diff = self.__replace_node_of_code(code,node,replaced,diff)
            return ret_code,diff

        def java(code,diff,node:tree_sitter.Node):
            ret_code = code
            if node.type in ['assignment_expression']:
                op = node.child_by_field_name('operator')
                op_text = op.text.decode("utf-8").replace("=","")
                if len(op_text.strip()) <= 0:
                    return ret_code,diff
                left = node.child_by_field_name('left').text.decode("utf-8")
                right = node.child_by_field_name('right').text.decode("utf-8")
                _,type = self.__get_var_type_value(ret_code,left)
                if type:
                    type_text = type.text.decode("utf-8")
                    if type_text.lower() != "string":
                        op_text = op_text + f" ({type_text})"
                replaced = left + " = " + left + " " + op_text + " (" + right + ")"
                ret_code,diff = self.__replace_node_of_code(code,node,replaced,diff)
            elif node.type in ['update_expression']:
                if len(node.children) > 2 or not node.parent.type.endswith("statement"):
                    return ret_code,diff
                op = node.children[1].text.decode("utf-8")[0]
                left = node.children[0].text.decode("utf-8")
                pattern = re.compile(r'\w+')
                if not pattern.search(left):
                    op = node.children[0].text.decode("utf-8")[0]
                    left = node.children[1].text.decode("utf-8")
                replaced = left + " = " + left + " " + op + " " + "1"
                ret_code,diff = self.__replace_node_of_code(code,node,replaced,diff)
            return ret_code,diff

        def javascript(code,diff,node:tree_sitter.Node):
            ret_code = code
            if node.type in ['augmented_assignment_expression']: 
                op = node.child_by_field_name('operator')
                op_text = op.text.decode("utf-8").replace("=","")
                if len(op_text.strip()) <= 0:
                    return ret_code,diff
                left = node.child_by_field_name('left').text.decode("utf-8")
                right = node.child_by_field_name('right').text.decode("utf-8")
                replaced = left + " = " + left + " " + op_text + " (" + right + ")"
                ret_code,diff = self.__replace_node_of_code(code,node,replaced,diff)
            elif node.type in ['update_expression']:
                if not node.parent.type.endswith("statement"):
                    return ret_code,diff
                op = node.child_by_field_name('operator').text.decode("utf-8")[0]
                left = node.child_by_field_name('argument').text.decode("utf-8")
                replaced = left + " = " + left + " " + op + " " + "1"
                ret_code,diff = self.__replace_node_of_code(code,node,replaced,diff)
            return ret_code,diff

        ret_code = code
        diff = 0
        tree = self.__parser.parse(bytes(ret_code, 'utf-8'))
        functions = {'python':python,'cpp':cpp,'java':java,'javascript':javascript}
        for node in self.__traverse_tree(tree):
            ret_code,diff = functions[self.lang](ret_code,diff,node)
        return ret_code
    
    
        
    def constant_propogation(self,code):
        '''
        常量传播,检查代码中的变量是否运行过程中始终为常量,并将其替换为常量
        '''
        '''
        我们需要纪录什么信息:
        找到一个变量最开始的定义语句,纪录它最开始的类型和值
        '''
        def cal_expression_value(expression:tree_sitter.Node):
            '''
            计算一个表达的值
            '''
            if expression.type == "identifier":
                return record[expression.text.decode("utf-8")][1]
            elif expression.type in literals[self.lang]:
                return expression.text.decode("utf-8")
            elif expression.type == "binary_expression":
                op = expression.child_by_field_name('operator').text.decode("utf-8")
                left = expression.child_by_field_name('left')
                right = expression.child_by_field_name('right')
                left_value = cal_expression_value(left)
                right_value = cal_expression_value(right)
                if op:
                    opt = op.text.decode("utf-8")
                    if opt == "+":
                        return left_value + right_value
                    elif opt == "-":
                        return left_value - right_value
                    elif opt == "*":
                        return left_value * right_value
                    elif opt == "/": #暂时只做简单的加减乘除
                        if right_value != 0:
                            return left_value / right_value
                        else:
                            return "UDF"
            
            return "NAC"

        _,identifier_names = self.get_identifiers(code)
        record = dict(zip(identifier_names,(None,None)))
        for i,v in enumerate(identifier_names):
            value,types = self.__get_var_type_value(code,v)
            if type(value) == tree_sitter.Node:
                if value.type.endswith("literal"): #这里后面要换成每种语言各自的判断，目前是cpp
                    value = value.text.decode('utf-8')
                else:
                    value = "NAC"
            record[v] = [types.text.decode("utf-8"),value]
        record_re = None
        tree = self.__parser.parse(bytes(code, 'utf-8'))
        while record_re != record:
            record_re = copy.deepcopy(record)
            for node in self.__traverse_tree(tree):
                if node.type in ["assignment_expression"]:
                    left = node.child_by_field_name('left')
                    op = node.child_by_field_name('operator')
                    right = node.child_by_field_name('right')
                    value = cal_expression_value(right)
                    if value == "NAC":
                        record[left.text.decode("utf-8")][1] = value
                    elif value == "UDF":
                        pass
                    else:
                        if record[left.text.decode("utf-8")][1] == "UDF":
                            record[left.text.decode("utf-8")][1] = value
                        elif record[left.text.decode("utf-8")][1] != value:
                            record[left.text.decode("utf-8")][1] = "NAC"
                elif node.type == "update_expression":
                    left = node.child_by_field_name('argument')
                    record[left.text.decode("utf-8")][1] = "NAC"

        
        print(record)
        

    def __elseif2else_if_once(self, code):
        '''
        将else if{}转换为else{if{}}
        '''
        if self.lang == 'python':
            return code
        diff = 0
        ret_code = code
        tree = self.__parser.parse(bytes(code, 'utf-8'))
        flag = False
        for node in self.__traverse_tree(tree):
            if node.type == "else_clause":
                replaced = node.text.decode('utf-8')
                if replaced.replace(' ','')[4:6] != 'if':
                    continue
                indent = node.start_point[1]
                print("indent:", indent)
                replaced = replaced[4:].lstrip()
                replaced = 'else{\n' + ' '*indent + replaced + '\n' + ' '*(indent-4) + '}' 
                lines = replaced.split('\n')  # 以换行符分割字符串
                modified_lines = ['    ' + line for line in lines]  # 在每行加上4个空格
                replaced = '\n'.join(modified_lines)  # 拼接字符串
                replaced = replaced[4:]
                ret_code,diff = self.__replace_node_of_code(ret_code,node,replaced,diff)
                flag = True
                break
        return ret_code, flag
                
    def elseif2else_if(self, code):
        '''
        将else if{}转换为else{if{}}
        '''
        try:
            if self.lang == 'python':
                return code
            ret_code = code
            flag = True
            while flag == True:
                ret_code, flag = self.__elseif2else_if_once(ret_code)
            return ret_code
        except:
            return code
    
    def __else_if2elseif_once(self, code):
        '''
        将else{if{}}转换为else if{}
        '''
        def add_indent(text, initial_indent=0):
            lines = text.split('\n')  # 分割字符串为多行
            current_indent = initial_indent  # 初始化缩进值
            indented_lines = []  # 用于保存处理过的每一行

            for line in lines:
                # 在每行开头添加当前缩进的空格
                line = line.lstrip()
                indented_line = ' ' * current_indent + line

                # 更新缩进量
                if '{' in line:
                    current_indent += 4
                if '}' in line:
                    current_indent -= 4
                    indented_line = indented_line[4:]
                indented_lines.append(indented_line)

            # 将处理过的每一行拼接回一个字符串
            return '\n'.join(indented_lines)
        if self.lang == 'python':
            return code,False
        diff = 0
        ret_code = code
        tree = self.__parser.parse(bytes(code, 'utf-8'))
        flag = False
        for node in self.__traverse_tree(tree):
            if node.type == "else_clause":
                replaced = node.text.decode('utf-8')
                if replaced.replace(' ','').replace('\n','')[4:7] != '{if' or len(node.children[1].children)!=2:
                    continue
                indent = node.start_point[1]
                print("indent:", indent)
                replaced = replaced[replaced.find('(') : replaced.rfind('}')].rstrip()
                replaced = add_indent(replaced, indent)
                replaced = replaced.lstrip()
                replaced = 'else if' + replaced
                print(replaced)
                ret_code,diff = self.__replace_node_of_code(ret_code,node,replaced,diff)
                flag = True
                break
        return ret_code, flag
    
    def else_if2elseif(self, code):
        '''
        将else{if{}}转换为else if{}
        '''
        try:
            if self.lang == 'python':
                return code[:]
            ret_code = code
            flag = True
            while flag == True:
                ret_code, flag = self.__else_if2elseif_once(ret_code)
            return ret_code
        except:
            return code


    def Swap_Determinator(self, code):
        '''
        交换判断符号的两边
        '''
        diff = 0
        ret_code = code
        tree = self.__parser.parse(bytes(code, 'utf-8'))
        if self.lang == 'python':
            for node in self.__traverse_tree(tree):
                if node.type == "comparison_operator":
                    flag = True
                    left = node.children[0].text.decode('utf-8')
                    operator = node.children[1].text.decode('utf-8')
                    right = node.children[2].text.decode('utf-8')
                    if operator == '<':
                        new_operator = '>'
                    elif operator == '>':
                        new_operator = '<'
                    elif operator == '<=':
                        new_operator = '>='
                    elif operator == '>=':
                        new_operator = '<='
                    elif operator == '!=' or operator == '==':
                        new_operator = operator
                    else:
                        flag = False
                    if flag == True:
                        replaced = right + ' ' + new_operator + ' ' + left
                        print(replaced)
                        ret_code,diff = self.__replace_node_of_code(ret_code,node,replaced,diff)
        else:
            operators = {
                '==': '==',
                '!=': '!=',
                '<=': '>=',
                '>=': '<=',
                '<': '>',
                '>': '<'
            }
            for node in self.__traverse_tree(tree):
                if node.type == "binary_expression":
                    child = node.children
                    if len(child) != 3:
                        continue
                    left = node.children[0].text.decode('utf-8')
                    operator = node.children[1].text.decode('utf-8')
                    right = node.children[2].text.decode('utf-8')
                    if operator not in operators.keys():
                        continue
                    replaced = right + ' ' + operators[operator] + ' ' + left
                    print(replaced)
                    ret_code,diff = self.__replace_node_of_code(ret_code,node,replaced,diff)
        return ret_code
    
    def Not_Determinator_once(self, code):
        '''
        if 与 else 内容交换
        '''
        try:
            diff = 0
            ret_code = code
            tree = self.__parser.parse(bytes(code, 'utf-8'))
            if self.lang == 'python':
                flag = False
                for node in self.__traverse_tree(tree):
                    if node.type == "if_statement":
                        text = node.text.decode('utf-8')
                        indent = node.start_point[1]
                        if text.replace(' ', '')[2:5] == 'not':
                            continue
                        if node.children[-1].type != 'else_clause':
                            continue
                        if len(node.children) > 5:
                            continue
                        mode = 0
                        if '\t' in text:
                            mode = 1
                        if mode == 0:
                            replaced = 'if not(' + node.children[1].text.decode('utf-8') + '):\n' + ' '*(indent+4)
                            if_block = node.children[3].text.decode('utf-8')
                            else_block = node.children[4].children[2].text.decode('utf-8')
                            replaced = replaced + else_block + '\n' + ' '*(indent) + 'else:\n' + ' '*(indent+4) + if_block
                        if mode == 1:
                            replaced = 'if not(' + node.children[1].text.decode('utf-8') + '):\n' + '\t'*(indent+1)
                            if_block = node.children[3].text.decode('utf-8')
                            else_block = node.children[4].children[2].text.decode('utf-8')
                            replaced = replaced + else_block + '\n' + '\t'*(indent) + 'else:\n' + '\t'*(indent+1) + if_block
                        flag = True
                        ret_code,diff = self.__replace_node_of_code(ret_code,node,replaced,diff)
                        break
                return ret_code
            else:
                flag = False
                for node in self.__traverse_tree(tree):
                    if node.type == "if_statement":
                        text = node.text.decode('utf-8')
                        indent = node.start_point[1]
                        if text.replace(' ', '')[3] == '!':
                            continue
                        if node.children[-1].type != 'else_clause':
                            continue
                        if len(node.children) > 4:
                            continue
                        replaced = 'if (!' + node.children[1].text.decode('utf-8') + ') '
                        if_block = node.children[2].text.decode('utf-8')
                        else_block = node.children[3].children[1].text.decode('utf-8')
                        replaced = replaced + else_block + '\n' + ' '*(indent) + 'else ' + if_block
                        flag = True
                        ret_code,diff = self.__replace_node_of_code(ret_code,node,replaced,diff)
                        break
                return ret_code
        except:
            return code
    
    def no_change(self,code):
        return code
    
    def code_style_perturbtion(self):
        #reformatting
        return [(self.tab_indent,"tab_indent"),\
                                    (self.line_split,"line_split"),\
                                        (self.doc2comments,"doc2comments"),\
                                            (self.newline_afterdoc,"newline_afterdoc"),\
                                    (self.newline_random,"newline_random"),\
                                        (self.newline_aftercode,"newline_aftercode")] 

    def rename_perturbation(self):
        #rename
        return [(self.random_filp_function_name,"random_filp_function_name"),\
                                   (self.normalize_function_names,"normalize_function_names"),\
                                   # (self.syn_replace_functions,"syn_replace_functions"),\
                                            (self.codebert_rename_func_name,"codebert_rename_func_name"),\
                                   (self.random_flip_identifier,"random_flip_identifier"),\
                (self.normalize_identifer,"normalize_identifer"),(self.codebert_rename_identifier,"codebert_rename_identifier")]#
                                     # (self.syn_replace_identifier,"syn_replace_identifier")]
    
    def code_expression_perturbtion(self):
        #expression substituion
        return [(self.bool2int,"bool2int"),\
                                    (self.more_universe_var_type,"more_universe_var_type"),\
                                        (self.exchange_operators,"exchange_operators"),\
                                            (self.equal_expression_transformation,"equal_expression_transformation")] 
    

    def code_stmt_perturbtion(self):
        #statement substituion
        return [(self.for_var_inner,"for_var_inner"),\
                                        (self.for_var_outer,"for_var_outer"),\
                                            (self.for2while,"for2while"),\
                                    (self.while2for,"while2for"),(self.elseif2else_if,"elseif2else_if"),\
                                        (self.else_if2elseif,'else_if2elseif'),\
                                        (self.Not_Determinator_once,'Not_Determinator_once')]
    
    def insert_perturbation(self):
        return [(self.insert_dead_code,"insert_dead_code"),\
                                (self.insert_comment,"insert_comment"),\
                                (self.remove_comments,"remove_comments"),\
                                (self.un_relate_package_import_insert,"un_relate_package_import_insert"),\
                                    (self.fold_constant,"fold_constant")] 

    
    
    def no_change_perturbation(self):
        return [(self.no_change,"no_change")]

        

        

    


                
        

        
        
    

if __name__ == '__main__':
        

    a = PerturbationPipeline()
    code = """public class ForLoopExample {
        public static void main(String[] args) {
            // 使用for循环打印数字1到10
            for (int i = 1; i <= 10; i++) {
                System.out.println("当前数字是: " + i);
            }
        }
    }
    """

    cpp_code = """#include <iostream>

    int main() {
        // 使用for循环打印数字1到10
        int i = 1;
        for (; i <= 10; i++) {
            std::cout << "当前数字是: " << i << std::endl;
        }
        int j = 1;
        for (; j <= 10; j++) {

        }
        return 0;
    }
    """

    js_code = """// 导入第三方库 lodash
    const _ = require('lodash');

    // 导入Node.js内置模块 fs
    const fs = require('fs');

    // 自定义模块函数
    function add(a, b) {
        return a + b;
    }

    function subtract(a, b) {
        return a - b;
    }

    // 使用 lodash 库
    console.log(_.capitalize('hello')); // 输出: Hello

    // 使用自定义模块函数
    console.log(add(2, 3)); // 输出: 5
    console.log(subtract(5, 2)); // 输出: 3

    // 使用 fs 模块读取文件
    fs.readFile('example.txt', 'utf8', (err, data) => {
        if (err) {
            console.error(err);
            return;
        }
        console.log(data);
    });
    """

    code = a.preprocess_code(js_code,"javascript")
    print(a.un_relate_package_import_insert(code))

    