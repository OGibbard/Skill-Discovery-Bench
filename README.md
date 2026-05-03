# Skill-Discovery-Bench

This repository is an unofficial research implementation for comparing unsupervised skill-discovery objectives under degraded observations.

The code provides a shared Soft Actor-Critic stack for DIAYN-style discriminator rewards, DADS-style transition-prediction rewards, and LSD-style displacement rewards. All objectives use the same policy architecture, replay buffer, optimizer, MuJoCo/Gymnasium wrapper, observation-corruption wrapper, and logging pipeline. The goal is to compare the skill objective rather than differences between standalone research codebases.

This repository is not the official DIAYN, DADS, LSD, SAC, rllab, Google, Berkeley, University of Michigan, Seoul National University, or University of Bath release.

## Table of Contents

  * Setup
  * Usage
  * Observation Degradation
  * Analysis
  * Citation
  * Disclaimer

## Setup

#### (1) Setup MuJoCo

Install MuJoCo and confirm that Gymnasium can create the target environment:

    python -c "import gymnasium as gym; gym.make('HalfCheetah-v4')"

The reported experiments used Python 3.10, Gymnasium, MuJoCo 3, and TensorFlow 2 in TF1-compatibility mode.

#### (2) Setup environment

For Apple Silicon:

    conda env create -f environment.yml
    conda activate diayn

For Linux:

    conda env create -f environment-linux.yml
    conda activate diayn

The `requirements.txt` file is provided for users who prefer to manage the environment manually.

#### (3) Check the source tree

    python -m compileall -q \
      run_full_diayn.py \
      run_scaled_diayn.py \
      mujoco_diayn_sim.py \
      visualize_diayn.py \
      analysis \
      sac \
      rllab

## Usage

Run a short DIAYN smoke test:

    python run_scaled_diayn.py \
      --env half-cheetah \
      --skill_objective diayn \
      --exp_name smoke \
      --seeds 0 \
      --num_skills 8 \
      --n_epochs_override 2 \
      --max_path_length_override 100 \
      --epoch_length_override 100

Switch objectives with:

    --skill_objective diayn
    --skill_objective dads
    --skill_objective lsd_delta

The main training scripts write logs and generated results under local output directories. Generated outputs, checkpoints, videos, TensorFlow event logs, copied papers, and course materials are intentionally not committed.

## Observation Degradation

The main training entry point supports Gaussian sensor noise, fixed observation delay, and learned or compressed observation encodings:

    --obs_noise_std 0.1
    --obs_delay_steps 3
    --obs_encoder_type linear
    --obs_encoder_dim 8
    --obs_encoder_hidden_dim 64
    --obs_corruption_seed 0

These settings are recorded in each run's `robustness_manifest.json`, so analysis scripts can recover the experimental condition from run metadata.

## Analysis

Analysis utilities are in `analysis/`. They aggregate robustness runs, summarize degradation effects, evaluate per-skill repertoires, and produce plots.

Example commands after producing or restoring run logs:

    python analysis/build_main_plus_noise_delay_degradation.py \
      --main path/to/main_degradation.csv \
      --extension path/to/noise_delay_degradation.csv \
      --output outputs/tables/main_plus_noise_delay_degradation.csv

    python analysis/plot_robustness_curves.py \
      --root path/to/half-cheetah/logs \
      --output-dir outputs/figures

    python analysis/plot_skill_heatmaps.py \
      --input path/to/skill_rollouts.csv \
      --output-dir outputs/figures/skill_heatmaps

To rerun the full post-hoc rollout pipeline, first generate or restore the checkpoint logs, then run:

    bash analysis/run_posthoc_n5seeds_full.sh

## Repository Layout

    analysis/                 Aggregation, plotting, and post-hoc evaluation utilities
    rllab/                    Vendored rllab compatibility layer
    sac/                      SAC and skill-discovery objective implementation
    environment.yml           Apple Silicon conda environment
    environment-linux.yml     Linux conda environment
    THIRD_PARTY_NOTICES.md    Upstream attribution and license notes

## Citation

If you use this repository, please cite this codebase and the original skill-discovery papers. The repository citation metadata is provided in `CITATION.cff`.

DIAYN:

    @article{eysenbach2018diversity,
      title={Diversity is All You Need: Learning Skills without a Reward Function},
      author={Eysenbach, Benjamin and Gupta, Abhishek and Ibarz, Julian and Levine, Sergey},
      journal={arXiv preprint arXiv:1802.06070},
      year={2018}
    }

DADS:

    @article{sharma2019dynamics,
      title={Dynamics-Aware Unsupervised Discovery of Skills},
      author={Sharma, Archit and Gu, Shixiang and Levine, Sergey and Kumar, Vikash and Hausman, Karol},
      journal={arXiv preprint arXiv:1907.01657},
      year={2019}
    }

LSD:

    @inproceedings{park2022lipschitz,
      title={Lipschitz-constrained Unsupervised Skill Discovery},
      author={Park, Seohong and Choi, Jongwook and Kim, Jaekyeom and Lee, Honglak and Kim, Gunhee},
      booktitle={International Conference on Learning Representations},
      year={2022}
    }

## Disclaimer

This is an unofficial research codebase. It is provided as-is for inspection, reproduction, and extension. It is not an officially supported product or an official release from the authors of the original DIAYN, DADS, LSD, SAC, or rllab projects.
