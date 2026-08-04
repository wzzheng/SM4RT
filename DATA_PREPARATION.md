# Dataset Preparation

We do not require further preprocessing. Just download and organize the directory properly, then off you go!

Notice: Before downloading, make sure you obtain all the required licence.

## Motion Datasets

### [Kubric](https://github.com/google-research/kubric/)

Please follow steps by [DELTAv2](https://github.com/snap-research/DenseTrack3Dv2) to retrieve the dataset. 

We provided our preprocessed data with world tracking annotation:

`https://www.modelscope.cn/datasets/shjlin/kubric_world_processed`

### [Stereo4D](https://stereo4d.github.io/)

Preprocessed by other users: 

`https://huggingface.co/datasets/ZhengGuangze/Stereo4D_vlbm`


## [Dynamic Replica](https://github.com/facebookresearch/dynamic_stereo)

`https://huggingface.co/datasets/geyongtao/dynamic_replica/`

## [PointOdyssey](https://pointodyssey.com/)

`https://huggingface.co/datasets/aharley/pointodyssey/`

Note: Samples that does not contain 3D trajectories should be filtered: 

samples begin with `character`; `gso_in_big`; `gso_out_big`

## Depth Datasets

### [ARKitScenes](https://github.com/apple/ARKitScenes), [MVS-Synth](https://phuang17.github.io/DeepMVS/mvs-synth.html), [TartanAir](https://theairlab.org/tartanair-dataset/), [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)

Preprocessed by other users, please replace `{DATASET}` with `ARKitScenes_preprocessed, MVS_Synth_preprocessed, TartanAir_preprocessed, 
VKITTI2_vlbm`:

`https://huggingface.co/datasets/ZhengGuangze/{DATASET}/`

### [HyperSim](https://github.com/apple/ml-hypersim)

Preprocessed by other users: 

`https://huggingface.co/datasets/KevinConnorLee/preprocessed_Hypersim/`

## [Scannet++](https://kaldir.vc.in.tum.de/scannetpp/)

Preprocessed by other users: 

`https://huggingface.co/datasets/HarrisonPENG/scannetpp/`

## [Waymo](https://github.com/waymo-research/waymo-open-dataset)

Preprocessed by other users: 

`https://huggingface.co/datasets/Brainkite/waymo_processed/`
