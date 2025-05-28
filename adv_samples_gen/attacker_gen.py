import sys
sys.path.append("../peturbation_pipeline")
import copy
import torch
import random
from tqdm import tqdm
import re
from pipeline import PerturbationPipeline

class Attacker():
    def __init__(self, classifier,language,iter_nums=10,transfrom_iters=1,perturbation_type="rename",use_sa=True,p=0.8,accptance=0.005,beam_size = 1,model_type = "base") -> None:
        self.classifier = classifier
        self.language = language
        self.replace_index = None #记录当前代码段替换到第几个identifier了
        self.codes = [] #维持beam size大小个final code
        self.final_code = None #记录当前正在替换的代码段
        self.subs = None #记录当前正在替换代码段的substitute candidates
        self.true_label = None #记录当前正在攻击代码段的True label
        self.doc = None #记录当前正在攻击代码段的example所有信息
        self.iter_nums = iter_nums #hyper parameter,攻击轮次数
        self.transfrom_iters = transfrom_iters #hyper parameter，每段代码进行扰动的次数
        self.perturbation_type = perturbation_type #当前使用哪种transformation
        self.use_sa = use_sa #是否使用模拟退火
        self.p = p # hyper parameter，模拟退火接受概率
        self.accptance = accptance # hyper parameter，模拟退火接受度
        self.beam_size = beam_size # beam size
        self.perturbationPipeline = PerturbationPipeline()
        self.perturbationPipeline.init_pretrained_model()
        self.perturbationPipeline.set_seed(42)
        self.perturbate = {'rename':self.perturbationPipeline.rename_perturbation,'code_stmt_exchange':self.perturbationPipeline.code_stmt_perturbtion,\
                           'code_expression_exchange':self.perturbationPipeline.code_expression_perturbtion,'insert':self.perturbationPipeline.insert_perturbation,\
                            'code_style':self.perturbationPipeline.code_style_perturbtion,'no_change':self.perturbationPipeline.no_change_perturbation}
        self.patterns = {'python':re.compile(r'assert.+',re.DOTALL),\
                'java':re.compile(r'public\s+class\s+Main\s*\{.*\}',re.DOTALL),\
                    'javascript':re.compile(r'const\s+\w+\s*\=\s*\(\s*\)\s*=>\s*.*',re.DOTALL),\
                        'cpp':re.compile(r'int\s+main.*',re.DOTALL)}
        self.model_type = model_type
    
    """
    def get_importance_score(self,code: str,identifier_names,identifier_positions: list,orig_prob:float):
        '''Compute the importance score of each variable'''
        
        if not identifier_names:
            return None
        
        positions = get_identifier_posistions_from_code(code, variable_names)
        
        # 需要注意大小写.
        if len(identifier_positions) == 0:
            ## 没有提取出可以mutate的position
            return None

        new_example = {}

        # 2. 得到Masked_tokens
        masked_token_list, _ = get_masked_code_by_position(code, positions)
        # replace_token_positions 表示着，哪一个位置的token被替换了.


        for variable, masked_one_variable in masked_token_list.items():
            new_code = ' '.join(masked_one_variable)
            new_example[variable] = new_code

        
        results = {}
        for variable,code_masked in new_example.items():
            results[variable] = self.classifier.predict({'code':code_masked,'doc':self.doc})[0]['score']

        #orig_prob = results[0]['score'] #这个为什么不直接参数传进来，非要重新计算一次

        importance_score = {}

        for variable,prob in results.items():
            importance_score[variable] = orig_prob - prob
        return importance_score
    """

    def apply_semantic_preserved_transformation(self,source_code:list[str],perturbation_type:str):
        """_summary_
        对单个source_code进行preserved_transformation(包含完全不变的source_code)
        Args:
            source_code (_type_): _description_
        Return:
            一个list,包含进行了transoformed后的所有代码
        """
        if source_code == None:
            return None,None
        ret = []
        perturbations_one_time = self.perturbate[perturbation_type]()
        changes = []
        for i,s_c in enumerate(source_code):#对每一个source_code
            nums = self.transfrom_iters
            res = []
            change_res = []
            while nums:#进行transfrom_iters次扰动
                nums -= 1
                real_pertubertion = random.choice(perturbations_one_time) #随机选择扰动
                res.append(real_pertubertion[0](s_c))
                change_res.append(real_pertubertion[1])

            ret.append(list(res))
            changes.append(change_res)

        return ret,changes
    

    def apply_attack(self,code_list):
        """_summary_
        beam search:从code_list中选择k个score最低的,k为beam size
        Args:
            code_list (_type_): _description_
        """
        temp_examples = {'code':code_list ,'doc':self.doc}
        #doc只有一个，code可以是一个list
        temp_results = self.classifier.predict(temp_examples)
        return temp_results
    

    def remove_half(self,code,lang):
        
        def cpp(st):
            st_line = st.strip().split("\n")
            st_line_real = []
            import_line = []
            using_namespace_line = []
            for i,v in enumerate(st_line):
                if v.strip():
                    if v.startswith("#include"):
                        import_line.append(v)
                    elif v.startswith("using"):
                        using_namespace_line.append(v)
                    else:
                        st_line_real.append(v)
            length = len(st_line_real)
            st_line_real = st_line_real[:length//2] + [st_line_real[-1]]
            st_line_real[length//2] = "//begin to write code\n"
            return '\n'.join(import_line + using_namespace_line + st_line_real)

        def java(st):
            st_line = st.strip().split("\n")
            st_line_real = []
            import_line = []
            using_namespace_line = []
            for i,v in enumerate(st_line):
                if v.strip():
                    if v.startswith("import"):
                        import_line.append(v)
                    elif v.startswith("class"):
                        using_namespace_line.append(v)
                    else:
                        st_line_real.append(v)
            length = len(st_line_real)
            st_line_real = st_line_real[:length//2]
            indent = ""
            if st_line_real:
                for v in st_line_real[-1]:
                    if v == ' ' or v == '\t':
                        indent += v
                    else:
                        break
            st_line_real.append(indent + "//begin to write code\n")
            return '\n'.join(import_line + using_namespace_line + st_line_real)

        def javascript(st):
            st_line = st.strip().split("\n")
            st_line_real = []
            import_line = []
            using_namespace_line = []
            for i,v in enumerate(st_line):
                if v.strip():
                    if v.startswith("import"):
                        import_line.append(v)
                    elif v.startswith("class"):
                        using_namespace_line.append(v)
                    else:
                        st_line_real.append(v)
            length = len(st_line_real)
            st_line_real = st_line_real[:length//2]
            indent = ""
            if st_line_real:
                for v in st_line_real[-1]:
                    if v == ' ' or v == '\t':
                        indent += v
                    else:
                        break
            st_line_real.append(indent + "//begin to write code\n")
            return '\n'.join(import_line + using_namespace_line + st_line_real)

        def python(st):
            st_line = st.strip().split("\n")
            st_line_real = []
            import_line = []
            func_head_line = []
            pattern = re.compile(r"from.+import")
            for i,v in enumerate(st_line):
                if v.strip():
                    if v.startswith("import") or pattern.search(v):
                        import_line.append(v)
                    elif v.strip().startswith("def") and len(func_head_line) == 0:
                        func_head_line.append(v.strip())
                    else:
                        if v.strip():
                            st_line_real.append(v)
            length = len(st_line_real)
            st_line_real = st_line_real[:length//2]
            indent = ""
            if st_line_real:
                for v in st_line_real[-1]:
                    if v == ' ' or v == '\t':
                        indent += v
                    else:
                        break
            if not indent:
                indent = "    "
            st_line_real.append(indent + "#begin to write code\n")
            return '\n'.join(import_line + func_head_line + st_line_real)

        functions = {'cpp':cpp,'java':java,'javascript':javascript,'python':python}
        return functions[lang](code)
    
    def greedy_attack(self,example):
        '''
        return
            original program: code
            program length: prog_length
            adversar program: adv_program
            true label: true_label
            original prediction: orig_label
            adversarial prediction: temp_label
            is_attack_success: is_success
            extracted variables: variable_names
            importance score of variables: names_to_importance_score
            number of changed variables: nb_changed_var
            number of changed positions: nb_changed_pos
            substitues for variables: replaced_words
        '''
            # 先得到tgt_model针对原始Example的预测信息.
        #理想状态：把它写成一个框架，可以在任意一点随机调用其他各种attack方式
        self.doc = example
        code_col_name = "code_str_deleted" if self.model_type == "instruct" else "code_str_generate"
        code = example[code_col_name] #被删除后的代码
        code = self.perturbationPipeline.preprocess_code(code,self.language)
        self.replace_index = 0

        
        self.true_label = example['code_str']
    
        orig_code = example[code_col_name]
        self.doc['flag'] = True
        results = self.classifier.predict({"code":[code],"doc":self.doc})
        self.doc['flag'] = False
        if type(results) == dict:
            results = [results]
        orig_prediction = results[0]['code']
        current_prob = results[0]['score']
        origin_prob = current_prob
        adv_code = ''
        prog_length = len(orig_code)


        print("Ground truth: \n",self.true_label)
        print("Orig prediction: \n",orig_prediction)
        print("Orig prob: \n",origin_prob)
        
        if current_prob == 0: #如果原本的score为0，直接return
            is_success = -2
            return orig_code,prog_length,self.true_label,orig_prediction,None,None,None,is_success,None,origin_prob,current_prob,None
        
        '''
        importance_score = self.get_importance_score(
                                                code,
                                                variable_names,
                                                self.classifier,
                                                current_prob,
                                                self.doc)
        '''
        
        self.codes = [self.true_label] + [None for _ in range(self.beam_size-1)]
        is_success = -1
        adv_prediction = orig_prediction
        adv_perturbation_type = None
        attack_paths = {code:[]} #在第一次iter进行更新，一个code对应一个path        
        for _ in tqdm(range(self.iter_nums)):
            all_code_samples = []
            refer_to = []
            all_changes = []
            all_complete_perturbed_codes = []
            code_samples,changes = self.apply_semantic_preserved_transformation(self.codes,self.perturbation_type)
            if code_samples:
                for i,codes in enumerate(code_samples):
                    for v in codes:
                        all_code_samples.append(self.remove_half(v,self.language))
                    refer_to.extend([self.codes[i] for _ in range(len(codes))]) #表示改变前的源代码是什么
                    all_changes.extend(changes[i])
                    all_complete_perturbed_codes.extend(codes)

            temp_results = self.apply_attack(all_code_samples) #这里是并行化的,所有code_samples会预测出来同等数量的predictions。所有predictions和test cases共同测评出来一个pass@1

            def dict_operation(your_dict,your_type):
                temp = copy.deepcopy(your_dict)
                ret = {}
                for onecode in self.codes:
                    ret[onecode] = temp.get(onecode,copy.deepcopy(your_type))    
                return ret
            
            ready2sort = []
            assert len(all_code_samples) == len(refer_to) == len(temp_results) == len(all_changes) == len(all_complete_perturbed_codes)

            for i in range(len(all_code_samples)):
                ready2sort.append((all_code_samples[i],refer_to[i],temp_results[i],all_changes[i],all_complete_perturbed_codes[i]))
            ready2sort.sort(key=lambda x:x[2]['score'])
            #加入模拟退火因素，有可能选择效果beam_size以外的一些数据（即效果更差的数据）
            if self.use_sa:
                if len(ready2sort) > self.beam_size:
                    randindex = random.randint(self.beam_size,len(ready2sort)-1)
                    replaceindex = random.randint(0,self.beam_size-1)
                    pp = random.random()
                    if pp > self.p:
                        if abs(ready2sort[randindex][2]['score'] - ready2sort[replaceindex][2]['score']) / ready2sort[replaceindex][2]['score'] < self.accptance:
                            print("SAN:accept lower score",ready2sort[replaceindex][2]['score'],"to",ready2sort[randindex][2]['score'],"replace",replaceindex,"to",randindex)
                            ready2sort[replaceindex] = ready2sort[randindex]

            ready2sort = ready2sort[:self.beam_size] #取前beam_size个
            for i in range(self.beam_size):
                
                self.codes[i] = ready2sort[i][4]
                refer = ready2sort[i][1]          
                attack_paths[self.codes[i]] = copy.deepcopy(attack_paths.get(refer,[]))
                attack_paths[self.codes[i]].append((ready2sort[i][0],ready2sort[i][2]['code'],ready2sort[i][3]))
            
            attack_paths = copy.deepcopy(dict_operation(attack_paths,[]))
            assert len(attack_paths.keys()) == self.beam_size

            adv_code = ready2sort[0][0] #没有测试用例的代码
            adv_truth = ready2sort[0][4]
            adv_prediction = ready2sort[0][2]['code']
            current_prob = ready2sort[0][2]['score']
            adv_perturbation_type = ready2sort[0][3]
            if current_prob == 0:
                #如果当前结果已经为0，可以直接结束
                is_success = 1
                break 

        for i,onecode in enumerate(self.codes):
            print(f"Path{i}:")
            for adv_code_p,adv_prediction_p,perturbation_typei in attack_paths[onecode]: 
                print(f"perturbation_type_{i}:")
                print(f"{self.perturbation_type}:{perturbation_typei}")
                print(f"adv_code_{i}:")
                print(adv_code_p)
                print(f"adv_prediction_{i}:")
                print(adv_prediction_p)
                print(f"original_prob:")
                print(origin_prob)
                print(f"adv_prob:")
                print(current_prob)
                print("----------next_iteration-------")

        
        return orig_code,prog_length,self.true_label,orig_prediction,adv_code,adv_truth,adv_prediction,is_success,adv_perturbation_type,origin_prob,current_prob,attack_paths
    
    
            
