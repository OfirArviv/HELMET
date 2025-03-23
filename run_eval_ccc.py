import glob

from tqdm import tqdm

from cluster_strategies import SingleRunArgs, CCCClusterStrategy

instruct_model_names = [
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "Qwen/Qwen2.5-7B-Instruct",
    # "Qwen/Qwen2.5-7B-Instruct-1M",
    # "google/gemma-3-4b-it",
    # "google/gemma-3-12b-it",
    # "google/gemma-2-9b-it",
    # "tiiuae/Falcon3-7B-Instruct",
    "tiiuae/Falcon3-Mamba-7B-Instruct",
    # "Zyphra/Zamba2-7B-Instruct-v2",
]

base_model_names = [
    # "meta-llama/Llama-3.1-8B",
    # "Qwen/Qwen2.5-7B",
    # "google/gemma-3-4b-pt",
    # "google/gemma-3-12b-pt",
    # "google/gemma-2-9b",
    # "tiiuae/Falcon3-7B-Base",
    # "tiiuae/Falcon3-Mamba-7B-Base",
    # "Zyphra/Zamba2-7B",
]


configs = glob.glob("configs/*_short*.yaml")
configs = glob.glob("configs/recall_short.yaml")

args_list = []
for config in configs:
    for model_name in base_model_names + instruct_model_names:
        output_dir = f'output/{model_name.replace("/", "_").replace(".", "_").replace("-", "_")}'
        use_chat_template = "True" if model_name in instruct_model_names else "False"
        args = SingleRunArgs(model_name_or_path=model_name,
                             use_chat_template=use_chat_template,
                             config=config,
                             output_dir=output_dir)
        args_list.append(args)

cluster_strategy = CCCClusterStrategy(
    tasks_args_list=args_list,
    num_of_parallel_jobs=20,
    _debug_disable_multi_processing=False,
    queue="nonstandard",
    mem="128g",
    gpu_type="a100_80gb",
    project_name="FM-Evaluation",
    job_name="Bamna",
    num_gpus=1
)

for res in tqdm(
            cluster_strategy.run_task_list(),
            total=len(args_list),
            desc="Running benchmark",
        ):
    print("done")











