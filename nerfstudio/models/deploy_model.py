# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Base Model implementation which takes in RayBundles
"""
import atexit
import bisect
from pathlib import Path

import torch
import torch.multiprocessing as mp
from loguru import logger
from torch import Tensor
from torch.multiprocessing import Barrier

from ..cameras.rays import RayBundle
from ..utils.eval_utils import eval_setup


class _StopToken:
    pass


class _PredictWorker(mp.Process):
    def __init__(self, *, load_config: Path, gpu_num,
                 task_queue, result_queue):
        self.load_config = load_config
        self.gpu_num = gpu_num
        self.task_queue = task_queue
        self.result_queue = result_queue
        super().__init__()
        self.barrier = Barrier(gpu_num)

    def run(self):
        import torch
        torch.cuda.set_device(self.gpu_num)
        torch.set_num_threads(1)
        logger.debug(f"initializing {self.__class__.__name__} with gpu_num={self.gpu_num}")
        # self.barrier.wait()
        _, self.pipeline, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=None,
            test_mode="inference",
        )
        logger.debug(f"initialized {self.__class__.__name__} with gpu_num={self.gpu_num}")
        # self.barrier.wait()

        while True:
            task = self.task_queue.get()
            if isinstance(task, _StopToken):
                logger.debug("subprocess breaks")
                break
            idx, data = task
            logger.debug("received")
            result = self.pipeline.model.get_outputs_for_camera_ray_bundle(data)
            logger.debug("processed")
            self.result_queue.put((idx, self.to_cpu(result)))
            del result

    def to_cuda(self):
        pass

    def to_cpu(self, x):
        if isinstance(x, Tensor):
            return x.cpu()
        if isinstance(x, dict):
            return {k: self.to_cpu(v) for k, v in x.items()}
        raise TypeError(x.__class__.__name__)


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    def __init__(self, load_config: Path, num_gpus: int = 8, ):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        self.load_config = load_config
        num_workers = max(num_gpus, 1)
        self.num_workers = num_workers
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            self.procs.append(
                _PredictWorker(
                    load_config=load_config, gpu_num=gpuid, task_queue=self.task_queue, result_queue=self.result_queue
                )
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(_StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5

    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle):
        size = camera_ray_bundle.shape[0]
        chunk = self.num_workers

        def split(a):
            k, m = divmod(len(a), chunk)
            return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(chunk)]

        cur_split_list = split(list(range(size)))
        for cur_split in cur_split_list:
            cur_data = camera_ray_bundle[(cur_split,)]
            self.put(cur_data)
        results = []
        for i in range(len(cur_split_list)):
            results.append(self.get())
        return self.merge_raybundle(*results)

    def merge_raybundle(self, *raybundle: RayBundle):
        return dict(rgb=torch.cat([x["rgb"] for x in raybundle], dim=0),
                    accumulation=torch.cat([x["accumulation"] for x in raybundle], dim=0),
                    depth=torch.cat([x["depth"] for x in raybundle], dim=0),
                    prop_depth_0=torch.cat([x["prop_depth_0"] for x in raybundle], dim=0),
                    prop_depth_1=torch.cat([x["prop_depth_1"] for x in raybundle], dim=0)
                    )
