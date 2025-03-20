for task in rag; do
  python eval.py --config configs/${task}_short_sample.yaml \
    --model_name_or_path "meta-llama/Llama-3.1-8B" \
    --output_dir output/"meta-llama_Llama-3.1-8B"/$task_sample_short  \
    --use_chat_template False # only if you are using non-instruction-tuned models, otherwise use the default.
done

for task in rag; do
  python eval.py --config configs/${task}_short_sample_zero_shot.yaml \
    --model_name_or_path "meta-llama/Llama-3.1-8B" \
    --output_dir output/"meta-llama_Llama-3.1-8B"/$task_sample_short_zero_shot  \
    --use_chat_template False # only if you are using non-instruction-tuned models, otherwise use the default.
done


# recall rag rerank cite longqa summ icl