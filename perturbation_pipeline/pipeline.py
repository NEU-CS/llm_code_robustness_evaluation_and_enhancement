#coding=utf-8
from typing import Generator,List,Set,Tuple
from tree_sitter import Language, Parser, Tree, Node
import re
import random
from keyword import iskeyword,kwlist
import os
from utils import common_built_in_functions_of_python,java_keywords,java_special_ids,c_keywords,c_macros,c_special_ids,js_future_reserved_words,\
js_global_objects,js_keywords,base_module

class PerturbationPipeline:
    def __init__(self):
        self.__parser = None

    def set_seed(self,seed):
        random.seed(seed)

    def __init_tree_sitter_parser(self, lang: str):
        LANGUAGE = Language('tree_sitter/my-languages.so', lang)
        self.__parser = Parser()
        self.__parser.set_language(LANGUAGE)

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

    def __traverse_tree(self, tree: Tree) -> Generator[Node, None, None]:
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

    def __replace_node_of_code(self, code: str, node: Node, replaced: str, diff: int):
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

    def get_function_name(self,code:str) -> list[str]:
        tree = self.__parser.parse(bytes(code, 'utf-8'))
        
        def cpp(node):
            if not node.children:
                return ([],[])
            if "function_declarator" == node.type:
                if "identifier" == node.children[0].type:
                    return [node.children[0].text.decode("utf-8")]
            ret = []
            for child in node.children:
                ret.extend(cpp(child))
            return ret
        
        def python(node):
            if not node.children:
                return []
            if "function_definition" == node.type:
                if "identifier" == node.children[1].type: # "def" == node.children[0].type
                    return [node.children[1].text.decode("utf-8")]
            ret = []
            for child in node.children:
                ret.extend(python(child))
            return ret

        def java(node):
            if not node.children:
                return []
            if "method_declaration" == node.type:
                if "identifier" == node.children[2].type: # "def" == node.children[0].type
                    return [node.children[2].text.decode("utf-8")]
            ret = []
            for child in node.children:
                ret.extend(python(child))
            return ret

        def javascript(node:Node):
            if not node.children:
                return []
            if "function_declaration" == node.type:
                if "identifier" == node.children[1].type: # "def" == node.children[0].type
                    return [node.children[1].text.decode("utf-8")]
            if "arrow_function" == node.type:
                if node.prev_sibling:
                    if node.prev_sibling.prev_sibling == node.prev_sibling.prev_sibling.type == "identifier":
                        return [node.prev_sibling.prev_sibling.text.decode("utf-8")]
            ret = []
            for child in node.children:
                ret.extend(javascript(child))
            return ret

        functions = {"cpp":cpp,'python':python,'java':java,'javascript':javascript}
        return list(set(functions[self.lang](tree.root_node)))
    
    def __rename_function_name(self,code:str,func_name:str,substitute:str) -> str:
        '''
        将function name替换为substitute
        '''
        diff = 0
        ret_code = code
        tree = self.__parser.parse(bytes(code, 'utf-8'))
        def python(node:Node):
            if node.type == "identifier" and node.parent.type == "function_definition" and node.prev_sibling.text.decode("utf-8") == "def":
                return True
            return False

        def cpp(node:Node):
            if node.type == "identifier" and node.parent.type == "function_declarator":
                return True
            return False

        def java(node:Node):
            if node.type == "identifier" and node.parent.type == "method_declaration":
                return True
            return False

        def javascript(node:Node):
            if node.type == "identifier" and node.parent.type == "function_declaration":
                return True
            if node.next_sibling:
                if node.next_slibing.next_sibling and node.next_sibling.next_sibling.type == "arrow_function":
                    return True
            return False

        functions = {'cpp':cpp,'python':python,'java':java,'javascript':javascript}
        for node in self.__traverse_tree(tree):
            function_name = node.text.decode("utf-8")
            if function_name == func_name:
                if functions[self.lang](node):
                    ret_code, diff = self.__replace_node_of_code(ret_code, node, substitute, diff)
        return self.__post_preprocess_code(ret_code)
    

    def __generate_random_name(self,name:str) -> str:
        pass


    def random_filp_function_name(self,code:str) -> str:
        func_names = self.get_function_name(code)
        if not func_names:
            return code
        func_name = random.choice(func_names)
        
    



    def get_identifiers(self, code: str) -> Set[Tuple[str, Node]]:
        '''
        获得一段代码中的所有标识符,及其对应于tree的节点node
        '''
        tree = self.__parser.parse(bytes(code, 'utf-8'))
        ret = set()
        names = set()
        def cpp(node:Node):
            if node.type == "identifier" and node.parent.type in ["parameter_declaration","init_declarator","declaration","reference_declarator"]:
                return True
            elif node.type == 'identifier' and node.parent.type == 'for_range_loop' and node.prev_sibling.type == 'primitive_type':
                return True
            return False

        def python(node:Node):
            if node.type == "identifier":
                if node.parent.type == "parameters": #函数形式参数
                    return True
                if node.parent.type == "for_statement" and node.prev_sibling.text.decode("utf-8") == "for": #for循环变量
                    return True
                if node.parent.type == "assignment": #赋值语句左值(提取到右值也没关系，因为右值只有function call可能是错误的，但function call节点的父节点为call)
                    return True
            return False

        def java(node):
            if node.type == "identifier":
                if node.parent.type == "formal_parameter": #函数形式参数
                    return True
                if node.parent.type == "variable_declarator": #赋值语句左值
                    return True
            return False
                

        def javascript(node:Node):
            if node.type == "identifier":
                if node.parent.type == "formal_parameters": #函数形式参数
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

    def __rename_identifier(self, code: str, tgt_word: str, substitute: str) -> str:
        '''
        将一段代码的目标标识符tgt_word全部替换为substitute
        '''
        diff = 0
        ret_code = code
        tree = self.__parser.parse(bytes(code, 'utf-8'))
        def cpp(node:Node):
            if node.type in ["identifier","type_identifier"] and node.parent.type not in ["function_declarator","qualified_identifier"]:
                return True
            elif node.type == "field_identifier":
                if node.parent.type == "field_expression" and node.parent.parent.type == "call_expression":
                    return False
                return True
            return False
        
        def python(node:Node):
            if node.type == "identifier":
                if node.parent.type in ["function_definition"]:#不能是函数名
                    return False
                elif node.parent.type == "attribute" and node.parent.parent.type == "call" and node.prev_sibling:#不能是a.x()中的x (a1.x1.x2()中的x1,x2也不能是)
                    return False
                return True
            return False

        def java(node:Node):
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
        
        def javascript(node:Node):
            if node.type == "identifier":
                if node.parent.type in ["function_declaration"]:#不能是函数名
                    return False
                if node.parent.type == "arguments":#是一个实际参数
                    return True
                elif node.next_sibling:
                    if node.next_sibling.next_sibling:
                        if node.next_sibling.next_sibling.type == "arrow_function":
                            return False
                return True
            return False

        functions = {'cpp':cpp,'python':python,'java':java,'javascript':javascript}
        for node in self.__traverse_tree(tree):
            identifier = node.text.decode("utf-8")
            if identifier == tgt_word:
                if functions[self.lang](node):
                    ret_code, diff = self.__replace_node_of_code(ret_code, node, substitute, diff)
        return self.__post_preprocess_code(ret_code)
    
    def random_flip_identifier(self,code:str) -> str:
        """
        字符串随机翻转
        IT1
        """
        #对其中的某个identifier进行随机翻转
        identifier_and_node,identifier_names = self.get_identifiers(code)
        exist_words = set(identifier_names)
        if len(identifier_names) == 0:
            return code
        get_flip_identifier = random.choice(identifier_names)
        length = len(get_flip_identifier)
        flip_nums = int(random.random() * length) + 1
        flip_indexs = list(set(random.randint(0,length - 1) for _ in range(flip_nums)))
        substitue = ""
        def flip_char(c):
            if 'a' <= c <= 'z':
                return chr(122 - (ord(c) - 97))
            elif 'A' <= c <= 'Z':
                return chr(90 - (ord(c) - 65))
            elif '0' <= c <= '9':
                return str(9 - int(c))
            return c
        for i,v in enumerate(get_flip_identifier):
            if i in flip_indexs:
                v = flip_char(v)
            substitue += v
        while substitue in exist_words:
            substitue = substitue + random.choice([chr(ord('a') + i) for i in range(26)])
        return self.__rename_identifier(code,get_flip_identifier,substitue)

    def normalize_identifer(self,code:str) -> str:
        '''
        使用统一var_0,var_1的形式
        '''
        _,identifier_names = self.get_identifiers(code)
        ret_code = code[:]
        for i,idtfs in enumerate(identifier_names):
            ret_code = self.__rename_identifier(ret_code,idtfs,"var_" + str(i))[:]
        return ret_code

    def codebert_rename_identifiers(self,code:str) -> List[str]:
        '''
        为了运行速度不太慢,每次随机选择一个identifier,并使用codebert重命名这个identifier。选择prob最高的单词
        '''
        _,identifier_names = self.get_identifiers(code)
        renamed_name = random.choice(identifier_names)
        if renamed_name == "":
            return code
        
        return 
    
    def bool2int(self,code):
        '''
        将一段代码中的所有true,false替换为1,0
        '''
        code = re.sub(r'\bTrue\b', '1', code)
        code = re.sub(r'\bFalse\b', '0', code)

        code = re.sub(r'\btrue\b', '1', code)
        code = re.sub(r'\bfalse\b', '0', code)
        return code

    def __find_all_base_var_type(self,code):
        '''
        找到一段代码中的所有int,char*,float定义的位置
        '''
        int_pattern = 'int'
        char_star_pattern = 'char\*'
        float_pattern = 'float'
        int_matches = [(match.start(), match.end()) for match in re.finditer(int_pattern, code)]
        char_matches = [(match.start(), match.end()) for match in re.finditer(char_star_pattern, code)]
        float_matches = [(match.start(), match.end()) for match in re.finditer(float_pattern, code)]
        return {'int':int_matches,'char*':char_matches,'float':float_matches}


    def more_universe_var_type(self,code):
        '''
        将一段代码的变量类型转换为更通用的类型,随机替换一个
        '''
        candidate = self.__find_all_base_var_type(code)
        types = ['int','char*','float']
        verse_map = {'int':'long','char*':'string','float':'double'}
        substitute_type = types[random.randint(0,2)]
        now_candidate = candidate[substitute_type]
        length = len(now_candidate)
        substitute_pos = now_candidate[random.randint(0,length-1)]
        ret_code = code[:substitute_pos[0]] + verse_map[substitute_type] + code[substitute_pos[1]:]
        return ret_code
    
    def tab_indent(self,code):
        '''
        将制表符替换为4个空格
        '''
        return re.sub(r'\t',' '*4,code)
    

    def line_split(self,code):
        '''
        将一段代码最长的一行拆分为两行,拆分的位置在最中间
        '''
        code_lst = code.split('\n')
        max_length = 0
        index = -1
        for i,v in enumerate(code_lst):
            if len(v) > max_length:
                max_length = len(v)
                index = i
        code_lst[index] = code_lst[index][:max_length//2] + '\\' + '\n' + code_lst[index][max_length//2:]
        return '\n'.join(code_lst)

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
            pattern = '\/\*(?P<docstring>.*?)\*/'
            matches = re.finditer(pattern, code, re.DOTALL)
            new_code = code
            for match in reversed(list(matches)):
                docstring = match.group('docstring').strip()
                docstring_lst = ["//" + oneline.strip() for oneline in docstring.split('\n')]
                new_code = new_code[:match.start()] + f"\n".join(docstring_lst) + new_code[match.end():]
            return new_code

        if self.lang == 'python':
            return python_handler(code)
        elif self.lang in ['java', 'cpp', 'javascript']:
            return java_cpp_javascript_handler(code)
        else:
            raise ValueError("Unsupported language")
    
    def newline_afterdoc(self,code):
        '''
        文档字符串后插入新行
        '''
        if self.lang == 'python':
            pattern = '[\'\"]{3}(?P<docstring>.*?)[\'\"]{3}'
        elif self.lang in ['java', 'cpp', 'javascript']:
            pattern = '\/\*(?P<docstring>.*?)\*/'
        else:
            raise ValueError("Unsupported language")
        matches = re.finditer(pattern, code, re.DOTALL)
        new_code = code
        for match in reversed(list(matches)):
            end_pos = match.end()
            new_code = new_code[:end_pos] + '\n' + new_code[end_pos:]
        return new_code


    def newline_random(self,code):
        '''
        随机在代码中插入空行
        '''
        code_lst = code.split('\n')
        index = random.randint(0,len(code_lst) - 1)
        code_lst = code_lst[:index] + ['\n'] + code_lst[index:]
        return '\n'.join(code_lst)

    def newline_aftercode(self,code):
        '''
        这个变换在代码的末尾插入一个空行。
        '''
        return code + "\n"
    
    def __generate_dead_code(self):
        '''
        生成死代码
        '''
        dead_code_templates = {
            "python": [
                "if False:\n    print('This is dead code in Python - if statement')\n",
                "while False:\n    print('This is dead code in Python - while loop')\n",
                "lambda: print('This function will never be called in Python')\n",
                "print('This line will never be executed in Python')\n",
            ],
            "java": [
                "if (false) {\n    System.out.println(\"This is dead code in Java - if statement\");\n}",
                "while (false) {\n    System.out.println(\"This is dead code in Java - while loop\");\n}",
                "new Runnable() {\n            @Override\n            public void run() {\n                System.out.println(\"This method will never be called in Java\");\n            }\n        }\n"
                "System.out.println(\"This line will never be executed in Java\");\n",
            ],
            "cpp": [
                "if (false) {\n    std::cout << \"This is dead code in C++ - if statement\" << std::endl;\n}",
                "while (false) {\n    std::cout << \"This is dead code in C++ - while loop\" << std::endl;\n}",
                "[]() {\n        std::cout << \"This function will never be called in C++\" << std::endl;\n    }\n",
                "std::cout << \"This line will never be executed in C++\" << std::endl;\n",
            ],
            "javascript": [
                "if (false) {\n    console.log('This is dead code in JavaScript - if statement');\n}",
                "while (false) {\n    console.log('This is dead code in JavaScript - while loop');\n}",
                "function() {\n    console.log('This function will never be called in JavaScript');\n};",
                "console.log('This line will never be executed in JavaScript');\n",
            ],
        }

        if self.lang not in dead_code_templates:
            raise Exception(f"Unsupported language: {self.lang}")

        template = random.choice(dead_code_templates[self.lang])
        return template
    
    def insert_dead_code(self,code):
        language = self.lang
        dead_code = self.__generate_dead_code()
        lines = code.split('\n')
        
        if language == "python":
            insertion_points = [i for i, line in enumerate(lines) if not line.strip().startswith('#')]
            
            if not insertion_points:
                return code
            insert_index = random.choice(insertion_points)
            
            lines.insert(insert_index, dead_code)
        
        elif language in ["java", "cpp", "javascript"]:
            # 找到所有花括号块的位置
            block_starts = [i for i, line in enumerate(lines) if re.findall(r'{', line) and not re.findall(r'class\s+\w+\s+{',line)]
            block_ends = [i for i, line in enumerate(lines) if re.findall(r'}', line)]

            if not block_starts or not block_ends:
                return code
            # 随机选择一个花括号块
            block_start = random.choice(block_starts)
            block_end = None
            for end in block_ends:
                if end > block_start:
                    block_end = end
                    break
            
            if block_end is None:
                return code
            
            # 在花括号块内随机插入死代码
            insert_index = random.randint(block_start + 1, block_end)
            lines.insert(insert_index, '    ' + dead_code)
        
        return '\n'.join(lines)
    
    def insert_comment(self,code):
        '''
        在代码中任意一行插入注释
        注释内容: this is a comment line.
        '''
        lines = code.split('\n')
        index = random.randint(0,len(lines) - 1)
        comment_token = {'python':'#','java':r'//','javascript':r'//','cpp':r'//'}
        lines = lines[:index] + [comment_token[self.lang] + 'This is a comment line.'] + lines[index:]
        return '\n'.join(lines)


    def remove_comments(self, code):
        # 定义不同语言的注释模式
        language = self.lang
        patterns = {
            "python": r'(#.*?$|""".*?""")',
            "java": r'(//.*?$|/\*.*?\*/)',
            "cpp": r'(//.*?$|/\*.*?\*/)',
            "javascript": r'(//.*?$|/\*.*?\*/)'
        }
        
        # 获取对应语言的注释模式
        pattern = patterns.get(language)
        
        if not pattern:
            raise ValueError(f"Unsupported language: {language}")
        
        # 使用正则表达式删除注释
        cleaned_code = re.sub(pattern, '', code, flags=re.MULTILINE | re.DOTALL)
        
        return cleaned_code
    

    def for_var_inner(self,code):
        '''
        for循环变量内部化
        '''
        lines = code.split('\n')
        pattern = re.compile("for\s*\((?P<id_definition>[^;]*);[^;]+;(?P<recur_var>[^\)]+)\)")
        indexs = []
        for i,line in enumerate(lines):
            matched = pattern.search(line)
            if matched:
                id_definition = matched.group('id_definition')
                if len(id_definition.strip()) == 0:
                    indexs.append(i)
        
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

        return '\n'.join(lines)
        

    
    def __get_indent_level(self,code):
        '''
        获取某行的缩进级别
        '''
        return len(code) - len(code.lstrip())
    
    def for_var_outer(self,code):
        '''
        for循环变量外部化
        '''
        lines = code.split('\n')
        pattern = re.compile("for\s*\((?P<id_definition>[^;]+);[^;]+;[^\)]+\)")
        indexs = []
        for i,line in enumerate(lines):
            matched = pattern.search(line)
            if matched:
                indexs.append(i)
                
        index = random.choice(indexs)
        line = lines[index]
        matched = pattern.search(line)
        id_definition = matched.group('id_definition')
        line = re.sub(id_definition,"",line)
        indent_level = self.__get_indent_level(line)
        lines[index] = indent_level * ' ' + id_definition + ';\n' + line

        return '\n'.join(lines)
    
    
    def un_relate_package_import_insert(self,code):
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
        if self.lang == "javascript":
            code = import_stmt[self.lang] + f" {random_module} = require('{random_module}')\n{code}"
        else:
            code = import_stmt[self.lang] + f" {random_module}\n{code}"
        return code



    
    
    

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