# Localizing and Editing the Goal Representation of a Vision-Language-Action Model

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23871372.svg)](https://doi.org/10.5281/zenodo.23871372)

Code and analysis outputs for the paper *Localizing and Editing the Goal Representation of a Vision-Language-Action Model*, presented at the SPAIS Workshop at the 10th Conference on Robot Learning (CoRL 2026), Austin, Texas, USA.

The code studies OpenVLA-7B fine-tuned on LIBERO-Goal. The 10 LIBERO-Goal tasks share one simulated scene, so the goal that each closed-loop rollout executes can be labeled from the goal predicates of all 10 tasks. The code runs negated and redirected instructions, localizes the goal with activation patching, edits it with patches and steering vectors, and evaluates goal read-outs as runtime monitors.

## Main results

| Result | Value |
|---|---|
| Forbidden task executed under negated instructions | 95 of 120 rollouts (79%) |
| Success on the plain instruction | 78% of 352 rollouts |
| Transition of the goal between token positions (activation patching) | layers 12 to 16 |
| Instruction-span steering vector at layer 8 | 12 of 18 rollouts redirected |
| Fixed vectors and prefill patches at layers 12 to 20 | 0 of 158 rollouts redirected |
| Decode-stream patching at layers 16 and 24 | 7 of 18 and 15 of 18 rollouts redirected |
| Within-pair AUROC for redirect non-compliance, layer-32 goal read-out and vision+text probe | 0.88 and 0.87 |

All results come from one model in one simulated scene.

## Repository layout

```
code/            rollouts, interventions, analyses, numbers and figures
plans/           episode plans (task, initial state, instruction, condition)
data/episodes/   per-episode summaries with the executed goal of each rollout
data/results/    analysis outputs used for the figures and Table 1
data/steer/      steering vectors used in the closed-loop tests
paper/           generated numbers (numbers.tex), Table 1 (tab_monitor.tex) and Figures 1 and 2
```

## Reproduce the numbers, Table 1 and the figures (CPU only)

```bash
pip install numpy==1.26.4 scipy scikit-learn matplotlib
python code/make_numbers4.py
python code/figs4.py
```

`make_numbers4.py` writes `paper/numbers.tex` and `paper/tab_monitor.tex`. `figs4.py` writes `paper/figs/fig_read.pdf` and `paper/figs/fig_write.pdf`. Both scripts read the episode summaries in `data/episodes` and the analysis outputs in `data/results`. The raw rollouts (about 29 GB of activations and frames) are not included.

## Run the experiments (GPU)

The experiments ran on one NVIDIA RTX A6000 with Python 3.11 under Windows 11. Each rollout worker needs about 16 GB of GPU memory.

1. Install PyTorch 2.5.1 with CUDA 12.1, then run `pip install -r requirements.txt`.
2. Install LIBERO from https://github.com/Lifelong-Robot-Learning/LIBERO. Copy `data/libero_cfg/config.yaml.example` to `data/libero_cfg/config.yaml` and set the paths of your LIBERO installation.
3. Download the checkpoint `openvla/openvla-7b-finetuned-libero-goal` from Hugging Face into `data/ckpt/openvla-7b-finetuned-libero-goal`.
4. Run the rollouts of a plan with `bash code/run.sh <plan> <run> <workers> [arguments]`. For example, `bash code/run.sh c0 c0 2` runs the nominal condition with 2 workers. The closed-loop interventions use `--stop_on_any 1`.

| Step | Script | Output |
|---|---|---|
| Episode plans | `make_plan.py` | `plans/` |
| Rollouts | `rollout.py` through `run.sh` | `data/runs/<run>/` |
| Activation patching (Figure 1a) | `patching3.py` | `data/results/patching3.json` |
| Per-dimension transfer (Figure 2a) | `patching4.py`, `patching5.py`, `patching6.py` | `data/results/patching4.json` to `patching6.json` |
| Negation control (Figure 1b) | `negctrl.py` | `data/results/negctrl.json` |
| Steering vectors and their plans | `make_steer.py`, `make_steer2.py`, `make_steer3.py`, `make_cfvec.py` | `data/steer/`, `plans/` |
| Vision baseline features | `dino_feats.py` | `data/feats/` |
| Monitors (Table 1, Figure 2b) | `analyze_v3.py`, `paired_within.py` | `data/results/v3.json`, `data/results/paired_within.json` |
| Episode summaries | `export_episodes.py` | `data/episodes/` |
| Numbers, Table 1 and figures | `make_numbers4.py`, `figs4.py` | `paper/` |

Each script documents its arguments in its docstring. On Windows, robosuite needs `MUJOCO_GPU_RENDERING = False` in `robosuite/macros_private.py`.

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{srivastava2026localizing,
  title     = {Localizing and Editing the Goal Representation of a Vision-Language-Action Model},
  author    = {Srivastava, Sudhanshu and Aqlan, Faisal},
  booktitle = {SPAIS Workshop at the Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```

The code is archived on Zenodo with DOI [10.5281/zenodo.23871372](https://doi.org/10.5281/zenodo.23871372) (version 1.0.0). `CITATION.cff` gives the citation for the code.

## License

This code is released under the MIT License. OpenVLA and LIBERO are separate projects with their own licenses.
