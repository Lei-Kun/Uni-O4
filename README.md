<p align="center">

  <h1 align="center"><a href="https://lei-kun.github.io/uni-o4/">Uni-O4</a>:
<a href="https://lei-kun.github.io/uni-o4/">Uni</a>fying <a href="https://lei-kun.github.io/uni-o4/">O</a>nline and <a href="https://lei-kun.github.io/uni-o4/">O</a>ffline Deep Reinforcement Learning with Multi-Step <a href="https://lei-kun.github.io/uni-o4/">O</a>n-Policy <a href="https://lei-kun.github.io/uni-o4/">O</a>ptimization
<h2 align="center">Preprints</h2>
  <p align="center">
    <a><strong>Kun Lei</strong></a>
    ·
    <a><strong>Zhengmao He*</strong></a>
    ·
    <a><strong>Chenhao Lu*</strong></a>
    ·
    <a><strong>Kaizhe Hu</strong></a>
    ·
    <a><strong>Yang Gao</strong></a>
    ·
    <a><strong>Huazhe Xu</strong></a>
  </p>

</p>
<h3 align="center">
  <a href="https://lei-kun.github.io/uni-o4/"><strong>Project Page</strong></a>
  |
  <a href="https://arxiv.org/abs/2311.03351"><strong>arXiv</strong></a>
  |
  <a href="https://twitter.com/kunlei15/status/1721885793369964703"><strong>Twitter</strong></a>
</h3>
<div align="center">
  <img src="pipeline_gif.gif" alt="Logo" width="100%">
</div>

## Code Overview
We evaluate Uni-O4 on standard D4RL benchmarks during offline and online fine-tuning phases. In addition, we utilize Uni-O4 to enable rapid adaptation of our quadrupedal robot dog to new and challenging environments. This repo contains five branches:
- `master (default) ->  Uni-O4`
- `go1-sdk -> sdk set-up for go1 robot`
- `data-collecting-deployment -> Deploying go1 in real-world for data collecting`
- `unio4-offline-dog -> Run Uni-O4 on dataset collected dy real-world robot dog`
- `go1-online-finetuning -> Fine-tuning the robot in real-world online`

## For D4RL benchmarks
### Requirements
- `torch                         1.12.0`
- `mujoco                        2.2.1`
- `mujoco-py                     2.1.2.14`
- `d4rl                          1.1`
To install all the required dependencies:
1. Install MuJoCo from [here](https://mujoco.org/download).
2. Install Python packages listed in `requirements.txt` using `pip install -r requirements.txt`. You should specify the version of `mujoco-py` in `requirements.txt` depending on the version of MuJoCo engine you have installed.
3. Manually download and install `d4rl` package from [here](https://github.com/rail-berkeley/d4rl).
### Running the code 
- `python main.py`: trains the network, storing checkpoints along the way. Other domain set-up comming soon.
- `Example`: 
```bash
./scripts/mujoco_loco/hm.sh
```
## Real-world tasks set-up (comming within one day)
## Citation 
If you use Uni-O4, please cite our paper as follows:
```
@inproceedings{
lei2024unio,
title={Uni-O4: Unifying Online and Offline Deep Reinforcement Learning with Multi-Step On-Policy Optimization},
author={Kun LEI and Zhengmao He and Chenhao Lu and Kaizhe Hu and Yang Gao and Huazhe Xu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=tbFBh3LMKi}
}
```