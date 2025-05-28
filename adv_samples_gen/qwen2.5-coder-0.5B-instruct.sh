conda activate adv
export CUDA_VISIBLE_DEVICES=11

python attack.py \
    --language=cpp \
    --model_name_or_path=/data1/model/qwen/Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --model_type=instruct \
    --eval_batch_size 4 \
    --use_sa \
    --beam_size 1 \
    --transfrom_iters 2 \
    --perturbation_type=rename \
    --seed 42| tee results/Qwen2.5-Coder-0.5B-Instruct/cpp/rename.log

python attack.py \
    --language=cpp \
    --model_name_or_path=/data1/model/qwen/Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --model_type=instruct \
    --eval_batch_size 4 \
    --use_sa \
    --beam_size 1 \
    --transfrom_iters 2 \
    --perturbation_type=code_stmt_exchange \
    --seed 42| tee results/Qwen2.5-Coder-0.5B-Instruct/cpp/code_stmt_exchange.log


python attack.py \
    --language=cpp \
    --model_name_or_path=/data1/model/qwen/Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --model_type=instruct \
    --eval_batch_size 4 \
    --use_sa \
    --beam_size 1 \
    --transfrom_iters 2 \
    --perturbation_type=code_expression_exchange \
    --seed 42| tee results/Qwen2.5-Coder-0.5B-Instruct/cpp/code_expression_exchange.log


python attack.py \
    --language=cpp \
    --model_name_or_path=/data1/model/qwen/Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --model_type=instruct \
    --eval_batch_size 4 \
    --use_sa \
    --beam_size 1 \
    --transfrom_iters 2 \
    --perturbation_type=insert \
    --seed 42| tee results/Qwen2.5-Coder-0.5B-Instruct/cpp/insert.log


python attack.py \
    --language=cpp \
    --model_name_or_path=/data1/model/qwen/Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --model_type=instruct \
    --eval_batch_size 4 \
    --use_sa \
    --beam_size 1 \
    --transfrom_iters 2 \
    --perturbation_type=code_style \
    --seed 42| tee results/Qwen2.5-Coder-0.5B-Instruct/cpp/code_style.log


