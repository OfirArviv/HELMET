import glob
import os

from tqdm import tqdm

from arguments import parse_arguments
from cluster_strategies import SingleRunArgs, CCCClusterStrategy


def does_exists(cmd_str: str):
    args = parse_arguments(cmd_str)

    assert args.model_name_or_path is not None
    os.makedirs(args.output_dir, exist_ok=True)

    datasets = args.datasets.split(",")
    test_files = args.test_files.split(",")
    demo_files = args.demo_files.split(",")
    max_lengths = ([int(args.input_max_length)] * len(datasets)) if isinstance(args.input_max_length, int) or len(
        args.input_max_length.split(",")) == 1 else [int(l) for l in args.input_max_length.split(",")]
    gen_lengths = ([int(args.generation_max_length)] * len(datasets)) if isinstance(args.generation_max_length,
                                                                                    int) or len(
        args.generation_max_length.split(",")) == 1 else [int(l) for l in args.generation_max_length.split(",")]
    assert len(test_files) == len(demo_files)

    args.input_max_length = max(max_lengths)

    for dataset, test_file, demo_file, max_length, gen_length in zip(datasets, test_files, demo_files, max_lengths,
                                                                     gen_lengths):
        args.datasets = dataset
        args.test_files = test_file
        args.demo_files = demo_file
        args.input_max_length = max_length
        args.generation_max_length = gen_length

        tag = args.tag
        if dataset == "popqa":
            tag += f"_pop{args.popularity_threshold}"

        test_name = os.path.splitext(os.path.basename(test_file))[0]
        output_path = os.path.join(args.output_dir,
                                   f"{dataset}_{tag}_{test_name}_in{args.input_max_length}_size{args.max_test_samples}_shots{args.shots}_samp{args.do_sample}max{args.generation_max_length}min{args.generation_min_length}t{args.temperature}p{args.top_p}_chat{args.use_chat_template}_{args.seed}.json")
        exists = os.path.exists(output_path)
        if exists is False:
            print(output_path)
            return False
    return True


instruct_model_names = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct-1M",
    # "google/gemma-3-4b-it",
    # "google/gemma-3-12b-it",
    "google/gemma-2-9b-it",
    "tiiuae/Falcon3-7B-Instruct",
    "tiiuae/Falcon3-Mamba-7B-Instruct",
    "Zyphra/Zamba2-7B-Instruct-v2",
]

base_model_names = [
    "meta-llama/Llama-3.1-8B",
    "Qwen/Qwen2.5-7B",
    # "google/gemma-3-4b-pt",
    # "google/gemma-3-12b-pt",
    "google/gemma-2-9b",
    "tiiuae/Falcon3-7B-Base",
    "tiiuae/Falcon3-Mamba-7B-Base",
    "Zyphra/Zamba2-7B",
]


configs = glob.glob("configs/*_short*.yaml")
# configs = configs[3:4]

args_list = []
for config in configs:
    for model_name in base_model_names + instruct_model_names:
        output_dir = f'output/{model_name.replace("/", "_").replace(".", "_").replace("-", "_")}'
        use_chat_template = "True" if model_name in instruct_model_names else "False"
        args = SingleRunArgs(model_name_or_path=model_name,
                             use_chat_template=use_chat_template,
                             config=config,
                             output_dir=output_dir)
        args_strings = args.get_args_dict()
        arg_names_and_values = [
            f"--{arg_name} {arg_str}" for arg_name, arg_str in args_strings.items()
        ]
        args_string = " ".join(arg_names_and_values)
        if not does_exists(args_string):
            args_list.append(args)

cluster_strategy = CCCClusterStrategy(
    tasks_args_list=args_list,
    num_of_parallel_jobs=20,
    _debug_disable_multi_processing=False,
    queue="nonstandard",
    mem="200g",
    gpu_type="a100_80gb",
    project_name="FM-Evaluation",
    job_name="Bamba",
    num_gpus=1
)

for res in tqdm(
            cluster_strategy.run_task_list(),
            total=len(args_list),
            desc="Running benchmark",
        ):
    print("done")


