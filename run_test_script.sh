

for llm in "NTQAI/Nxcode-CQ-7B-orpo" "deepseek-ai/deepseek-coder-33b-base"
do
    for mtask in "sw-functional"  "sw-modular"
    do
        lm_eval --model hf --model_args pretrained="${llm},trust_remote_code=True,dtype=float16" --tasks $mtask --num_fewshot 0  --output_path cma_results --log_samples --batch_size auto --device cuda:0
    done
don