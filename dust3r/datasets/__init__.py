from .utils.transforms import *
from .base.batched_sampler import BatchedRandomSampler  # noqa

from .kubric_motion import Kubric_Motion
from .dynamic_replica_motion import DynamicReplica
from .pointodyssey_motion import PointOdyssey
from .stereo4d import Stereo4D
from .vkitti2 import VKITTI2

from .depth_arkitscenes import ARKitScenesDepth
from .depth_dynamicreplica import DynamicReplicaDepth
from .depth_hypersim import HyperSIMDepth
from .depth_kitti import KittiDepth
from .depth_pointodyssey import PointOdysseyDepth
from .depth_scannetpp import ScanNetPPDepth
from .depth_tartanair import TartanAirDepth
from .depth_vkitti import VKitti2Depth
from .depth_waymo import WaymoDepth

from accelerate import Accelerator

def get_data_loader(
    dataset,
    batch_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    accelerator: Accelerator = None,
    fixed_length=False,
):
    import torch

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    try:
        sampler = dataset.make_sampler(
            batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            world_size=accelerator.num_processes,
            fixed_length=fixed_length,
        )
        shuffle = False

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            # pin_memory=pin_mem,
            pin_memory=True,
            prefetch_factor=4,          # PyTorch>=1.8
            persistent_workers=True,
        )

    except (AttributeError, NotImplementedError):
        sampler = None

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            # pin_memory=pin_mem,
            pin_memory=True,
            prefetch_factor=4,          # PyTorch>=1.8
            persistent_workers=True,
            drop_last=drop_last,
        )

    return data_loader
