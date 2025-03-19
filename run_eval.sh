for task in longqa; do
  python eval.py --config configs/${task}_short.yaml \
    --model_name_or_path "meta-llama/Llama-3.1-8B" \
    --output_dir outputs/"meta-llama_Llama-3.1-8B"/$task_short  \
    --use_chat_template False # only if you are using non-instruction-tuned models, otherwise use the default.
done


# recall rag rerank cite longqa summ icl