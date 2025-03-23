import json
import multiprocessing
import os
import random
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from queue import Queue
from typing import Dict, Generator, List, NamedTuple, Optional, Union


@dataclass
class SingleRunArgs:
    model_name_or_path: str
    output_dir: str
    use_chat_template: str
    config: str

    def get_args_dict(self) -> Dict[str, str]:
        return self.__dict__

@dataclass
class SingleRunResult:
    done = True



class TaskQueueData(NamedTuple):
    args: SingleRunArgs
    num_of_retries_left: int
    """ task_id uniquely idendifies each process across tasks
    """
    task_id: int


class WorkerInput(NamedTuple):
    """An input for the worker function of the Cluster Strategy with the following properties:
    tasks_args_queue (Queue[TaskQueueData]): The task_args_queue contains the run_text2text arguments to be
        run, in the form of SingleRunArgs type (see definition above), and the number of retries left. It's the
        responsibility on the _work_function to manage the retry, as well as the error handling mechanism.
    output_queue (Queue[SingleRunResult]): A multiprocessor queue, in which to put the results (or the error),
        (see type SingleRunResult).
    job_id (int): The job_id is used to differentiate the different tasks that run in
        parallel. A common use of this arg is to set the api_key (if needed) for the run_text2text function. See
        existing implementations for example.
    """

    tasks_args_queue: Queue[TaskQueueData]
    output_queue: Queue[SingleRunResult]
    job_id: int


class ClusterStrategy(ABC):
    """ClusterStrategy is the class type that allows the benchmark runner to run on different machines.

    This is an abstract class for the implementation of specific-machine executors for the benchmark runner.
     Currently, (may change in the future, as we improve the code), the class only need to implement the
     _worker_function method for any subclasses (see method documentation for more details).
     The method is run in a multiprocess manner based on the 'num_of_parallel_jobs' input param for the constructor.
     Te method is given input and output queues and facilitated the communication of the processes.

    Args:
        tasks_args_list (List[SingleRunArgs]): The arguments to the run_text2text function, which is (currently)
         the backbone of fmeval.
        num_of_parallel_jobs (int): The amount of parallel jobs to run.
         ***If running using BAM api*** -  as the number parallel job should not be larger than
         the number of available api keys (if it is you will most likely run into an RATE LIMIT error from BAM.
        num_of_retries (int): The amount of retries for each job in cases of an error.
        _debug_disable_multi_processing(bool):  When set to TRUE, the code will run the _worker_function method
        in a serial way, using the main process only. Make it easier when debugging (exception might
        not show in multiprocess, etc.).


    Returns:
        bool: Description of the return value.
    """

    def __init__(
        self,
        tasks_args_list: List[SingleRunArgs],
        num_of_parallel_jobs: int,
        num_of_retries: int = 1,
        _debug_disable_multi_processing: bool = False,
    ):
        self._num_of_parallel_jobs = num_of_parallel_jobs
        self._debug_disable_multi_processing = _debug_disable_multi_processing

        m = multiprocessing.Manager()
        self._tasks_args_queue: Queue[TaskQueueData] = m.Queue()
        self._tasks_output_queue: Queue[SingleRunResult] = m.Queue()

        for task_id, task_args in enumerate(tasks_args_list):
            self._tasks_args_queue.put(
                TaskQueueData(task_args, num_of_retries, task_id + 1)
            )

    @abstractmethod
    def _worker_function(self, worker_input: WorkerInput) -> None:
        """The main function that runs the job on the designated machine and needs to be implemented in each subclass.
        Instances of this function will run in parallel using a multiprocess pool, based on the constructor
        input args of '_num_of_parallel_jobs'.

            Args:
                inpt (WorkerInput): an input NamedTuple with the following properties:
                 tasks_args_queue (Queue[TaskQueueData]): The task_args_queue contains the run_text2text arguments to be
                   run, in the form of SingleRunArgs type (see definition above), and the number of retries left. It's the
                   responsibility on the _work_function to manage the retry, as well as the error handling mechanism.
                 output_queue (Queue[SingleRunResult]): A multiprocessor queue, in which to put the results (or the error),
                   (see type SingleRunResult).
                 job_id (int): The job_id is used to differentiate the different tasks the run in
                   parallel. A common use of this args is to set the api_key (if needed) for the run_text2text function. See
                   existing implementations for example.

            Returns:
                None: Due to the function running in a multiprocess manner, the outputs are put in the process-safe
                 queue, "output_queue", that is received as an input.
        """
        raise NotImplementedError()

    def run_task_list(self) -> Generator[SingleRunResult, None, None]:
        if self._debug_disable_multi_processing:
            workers_input = [
                WorkerInput(self._tasks_args_queue, self._tasks_output_queue, i)
                for i in [0]
            ]
            list(map(self._worker_function, workers_input))
            while not self._tasks_output_queue.empty():
                yield self._tasks_output_queue.get_nowait()
        else:
            with multiprocessing.Pool(processes=self._num_of_parallel_jobs) as pool:
                workers_input = [
                    WorkerInput(self._tasks_args_queue, self._tasks_output_queue, i)
                    for i in range(self._num_of_parallel_jobs)
                ]
                # Each "worker" is a "producer" that can produce many outputs. We want to consume each output when it's
                #  ready, that is why we use the map_async function that does not block and wait for all the workers
                #  to finish.
                results = pool.map_async(self._worker_function, workers_input)
                # The read function checks if all the workers finished their job. While they operate, we check the
                # output queue for new outputs and yield them.
                while not results.ready():
                    if not self._tasks_output_queue.empty():
                        yield self._tasks_output_queue.get_nowait()

        # TODO: For some reason the queue is not always empty when results.ready == True. Need to investigate this
        while not self._tasks_output_queue.empty():
            yield self._tasks_output_queue.get_nowait()
        assert self._tasks_output_queue.empty()


class CCCClusterStrategy(ClusterStrategy):
    def __init__(
        self,
        tasks_args_list: List[SingleRunArgs],
        num_of_parallel_jobs: int,
        queue: str,
        mem: str,
        gpu_type: str,
        project_name: str,
        job_name: str,
        num_gpus: int = 1,
        num_of_retries: int = 1,
        avoid_nodes: Optional[List[str]] = None,
        _debug_disable_multi_processing: bool = False,
    ):
        super().__init__(
            tasks_args_list,
            num_of_parallel_jobs,
            num_of_retries,
            _debug_disable_multi_processing,
        )

        self.queue = queue
        self.mem = mem
        self.gpu_type = gpu_type
        self.project_name = project_name
        self.job_name = job_name
        self.num_gpus = num_gpus
        self.avoid_nodes = avoid_nodes

    @staticmethod
    def _get_job_status(jobid) -> Optional[str]:
        r = subprocess.run(["jbinfo", str(jobid)], stdout=subprocess.PIPE)
        jbsub_output = r.stdout.decode("utf-8")
        lines = jbsub_output.split("\n")
        try:
            line = next(x for x in lines if x.strip().startswith(str(jobid)))
            return line.split()[2]
        except StopIteration:
            return None

    @staticmethod
    def _is_job_completed_successfully(
        job_id: str, jbsub_output: str, output_dir: str
    ) -> Union[Dict, Exception]:
        status = CCCClusterStrategy._get_job_status(job_id)
        # The "wait" param should take of it. But sometimes it doesn't work properly, so just in case
        while status in ["AVAIL", "RUN"]:
            # We are using the wait flag when submitting a job. But sometimes it takes a couple of secs for
            # the job status to get updated, and it's still in 'RUN' status, even thou the function returned.
            print("\nwaiting...")
            time.sleep(10)
            status = CCCClusterStrategy._get_job_status(job_id)
        if status is None:
            return Exception(
                f"Could not get job status of job {job_id}."
                f"\nJsubj output: {jbsub_output}"
            )
        elif status == "EXIT":
            return Exception(
                f"Job {job_id} failed (EXIT status)." f"\nJsubj output: {jbsub_output}"
            )
        elif status == "DONE":
            print(f"ccc job {job_id} completed with DONE status.")
        else:
            raise ValueError(
                f"Unexpected CCC job status {status}!" f"\nJsubj output: {jbsub_output}"
            )


        print(
            f"ccc job {job_id} completed successfully and result file is present."
        )
        return {}

    def _launch_cmd(self, args_strings):
        is_accelerate = self.num_gpus > 1
        if is_accelerate:
            port = random.randint(15000, 60000)  # port 0 is not working on CCC;
            # error torch.distributed.DistNetworkError: The client socket has timed out after 1800s
            # while trying to connect to (127.0.0.1, 0)
            return (
                f"accelerate launch "
                f"--main_process_port {port} "
                f"--config_file fm_eval/runnables/config/accelerate_fsdp_defaults.yaml "
                f"--num_processes {self.num_gpus}"
            )
        else:
            return "python"

    def _worker_function(self, inpt: WorkerInput) -> None:
        from cvar_pyutils.ccc import submit_job

        while not inpt.tasks_args_queue.empty():
            task_data = inpt.tasks_args_queue.get()
            args_strings = task_data.args.get_args_dict()
            args_strings["api_key_n"] = str(inpt.job_id)
            arg_names_and_values = [
                f"--{arg_name} {arg_str}" for arg_name, arg_str in args_strings.items()
            ]
            args_string = " ".join(arg_names_and_values)
            print(f"args string: {args_string}")
            python_file_to_run = os.path.join(
                "fm_eval", "runnables", "run_text2text.py"
            )
            command_to_run = (
                f"{self._launch_cmd(args_strings)} {python_file_to_run} {args_string}"
            )

            print(f"\nDispatching ccc job with command: {command_to_run}")
            # Because we are using the 'wait' flag, exiting the app using KeyboardInterrupt, also cancel the job.
            # This is a desired behavior.
            require_str = (
                " && ".join(["hname!=" + node for node in self.avoid_nodes])
                if self.avoid_nodes
                else None
            )
            job_id, jbsub_output = submit_job(
                command_to_run=command_to_run,
                queue=self.queue,
                num_cores=min(4, self.num_gpus * 4),
                num_gpus=self.num_gpus,
                mem=self.mem,
                gpu_type=self.gpu_type,
                require=require_str,
                project_name=self.project_name,
                name=self.job_name,
                wait=True,
                out_file=f"{task_data.args.output_dir}/output.log",
                err_file=f"{task_data.args.output_dir}/error.log",
                x11=False,
            )
            if job_id is None:
                raise Exception(f"Failed to send job {jbsub_output}, Stopping run...")

            result = CCCClusterStrategy._is_job_completed_successfully(
                job_id, jbsub_output, task_data.args.output_dir
            )
            if isinstance(result, Exception):
                if task_data.num_of_retries_left > 1:
                    # TODO: Find a way to do it, in a LIFO way, so to give the service some time to recover.
                    #  Right now its FIFO
                    inpt.tasks_args_queue.put(
                        TaskQueueData(
                            task_data.args,
                            task_data.num_of_retries_left - 1,
                            task_data.task_id,
                        )
                    )
                else:
                    inpt.output_queue.put(SingleRunResult())

            else:
                inpt.output_queue.put(SingleRunResult())

