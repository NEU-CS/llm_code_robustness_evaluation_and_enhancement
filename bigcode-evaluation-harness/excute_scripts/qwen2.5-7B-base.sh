conda activate adv
export CUDA_VISIBLE_DEVICES=11,12,13,14
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_series=qwen2.5 --model_type=causal_base --model_name=Qwen2.5-7b-base --model_path=/data1/model/qwen/Qwen/Qwen2.5-7B\
  --tasks=mbpp_generate_javascript_robust_no_change,mbpp_generate_javascript_robust_rename,mbpp_generate_javascript_robust_code_stmt_exchange,\
mbpp_generate_javascript_robust_code_expression_exchange,mbpp_generate_javascript_robust_insert,\
mbpp_generate_javascript_robust_code_style,\
mbpp_generate_java_robust_no_change,mbpp_generate_java_robust_rename,\
mbpp_generate_java_robust_code_stmt_exchange,\
mbpp_generate_java_robust_insert,mbpp_generate_java_robust_code_style

python calculate_pass_drop.py --language=python --model_name=Qwen2.5-7b-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=Qwen2.5-7b-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=Qwen2.5-7b-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=Qwen2.5-7b-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=Qwen2.5-7b-base --perturbation=code_style --model_type=causal_base


python calculate_pass_drop.py --language=cpp --model_name=Qwen2.5-7b-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=Qwen2.5-7b-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=Qwen2.5-7b-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=Qwen2.5-7b-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=Qwen2.5-7b-base --perturbation=code_style --model_type=causal_base




python calculate_pass_drop.py --language=java --model_name=Qwen2.5-7b-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=Qwen2.5-7b-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=Qwen2.5-7b-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=Qwen2.5-7b-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=Qwen2.5-7b-base --perturbation=code_style --model_type=causal_base



python calculate_pass_drop.py --language=javascript --model_name=Qwen2.5-7b-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=Qwen2.5-7b-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=Qwen2.5-7b-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=Qwen2.5-7b-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=Qwen2.5-7b-base --perturbation=code_style --model_type=causal_base