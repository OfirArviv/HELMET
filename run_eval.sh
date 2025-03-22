for task in rag; do
  python eval.py --config configs/${task}_short_example.yaml \
    --model_name_or_path "meta-llama/Llama-3.1-8B" \
    --output_dir output/"meta-llama_Llama-3.1-8B"/$task_short_example  \
    --use_chat_template False # only if you are using non-instruction-tuned models, otherwise use the default.
done

for task in rag; do
  python eval.py --config configs/${task}_short_example_zero_shot.yaml \
    --model_name_or_path "meta-llama/Llama-3.1-8B" \
    --output_dir output/"meta-llama_Llama-3.1-8B"/$task_short_example_zero_shot  \
    --use_chat_template False # only if you are using non-instruction-tuned models, otherwise use the default.
done

for task in rag; do
  python eval.py --config configs/${task}_short_example.yaml \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --output_dir output/"meta-llama_Llama-3.1-8B-Instruct"/$task_short_example  \
    --use_chat_template False # only if you are using non-instruction-tuned models, otherwise use the default.
done

for task in rag; do
  python eval.py --config configs/${task}_short_example_zero_shot.yaml \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --output_dir output/"meta-llama_Llama-3.1-8B-Instruct"/$task_short_sample_zero_shot  \
    --use_chat_template False # only if you are using non-instruction-tuned models, otherwise use the default.
done


# recall rag rerank cite longqa summ icl