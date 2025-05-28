conda activate adv
python attack.py \
    --language=python \
    --model_name_or_path=/data1/model/qwen/Qwen/Qwen2.5-Coder-0.5B \
    --model_type=base \
    --eval_batch_size 4 \
    --use_sa \
    --beam_size 1 \
    --transfrom_iters 2 \
    --perturbation_type=code_style \
    --seed 42| tee results/Qwen2.5-Coder-0.5B/python/code_style2.log





