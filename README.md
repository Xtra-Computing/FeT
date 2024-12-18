# [NeruIPS 2024] Federated Transformer (FeT)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![NeurIPS 2024](https://img.shields.io/badge/NeurIPS-2024-red.svg)](https://neurips.cc/Conferences/2024)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.1.2](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)




This paper _"[Federated Transformer: Multi-Party Vertical Federated Learning on Practical Fuzzily Linked Data](https://arxiv.org/pdf/2410.17986)"_ has been accepted by _**NeurIPS 2024**_.

## Project Overview

The **Federated Transformer (FeT)** is a novel framework designed to handle **multi-party Vertical Federated Learning (VFL)** scenarios involving **fuzzy identifiers**, where distinct features of shared data instances are provided by different parties without directly sharing raw data. FeT addresses the challenges of performance degradation and privacy overhead commonly faced in existing multi-party VFL models. It innovatively **encodes fuzzy identifiers into data representations** and **distributes a transformer architecture across parties** to enhance collaborative learning. FeT integrates **differential privacy** and **secure multi-party computation** to ensure strong privacy protection while minimizing utility loss. Experimental results show that FeT significantly improves performance - boosting accuracy by up to 46% over [FedSim](https://github.com/Xtra-Computing/FedSim) when scaled to 50 parties - and achieves superior performance even in **two-party fuzzy VFL** scenarios compared to [FedSim](https://github.com/Xtra-Computing/FedSim).

## Features
- Multi-party vertical federated learning
- Promising Performance on Fuzzy identifiers
- SplitAvg Framework: Differential Privacy and Secure Multi-party Computation

## Prerequisites
### Hardware Requirements

The Federated Transformer (FeT) framework is designed to operate efficiently without necessitating high-memory GPUs. For small-scale implementations, such as two-party fuzzy Vertical Federated Learning (VFL) on modest datasets, a single GPU with 4GB of memory is sufficient. However, for more extensive applications, particularly 50-party fuzzy VFL on large-scale datasets, we recommend utilizing A100 GPUs with a minimum of 40GB memory capacity. It is important to note that the current implementation does not support multi-GPU training configurations.

### Software Dependencies

The codebase has been developed and tested using Python version `3.10` with CUDA version `12.1`. While these specific versions are recommended, the framework is expected to be compatible with subsequent versions of both Python and CUDA. 

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/JerryLife/FeT.git
   cd FeT
   ```
2. Set up a virtual environment (recommended):
   ```bash
   python -m venv fet
   source fet/bin/activate  # On Windows, use `fet\Scripts\activate`
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Dataset

The real-world datasets used in FeT are the same as [FedSim](https://github.com/Xtra-Computing/FedSim). The synthetic datasets generated by splitting `gisette` and `mnist` dataset. Those synthetic datasets can be obtained by
```bash
bash ./src/script/download_dataset.sh     # download gisette and mnist dataset
bash ./src/script/split_scale.sh          # split them into multiple parties
```

## Usage

To train the Federated Transformer model, run the `train_fet.py` script located in the `src/script` directory. Below is the API documentation for `train_fet.py` along with example usage:

### API Documentation

#### Arguments
- `-g`, `--gpu` (int): GPU ID. Use `-1` for CPU. (default: `0`)
- `-d`, `--dataset` (str): Dataset to use.
- `-p`, `--n_parties` (int): Number of parties. Should be `>=2`. (default: `4`)
- `-pp`, `--primary_party` (int): Primary party ID. Should be in `[0, n_parties-1]`. (default: `0`)
- `-sp`, `--splitter` (str): Splitter method to use. (default: `'imp'`)
- `-w`, `--weights` (float): Weights for the ImportanceSplitter. (default: `1`)
- `-b`, `--beta` (float): Beta for the CorrelationSplitter. (default: `1`)
- `-e`, `--epochs` (int): Number of training epochs. (default: `100`)
- `-lr`, `--lr` (float): Learning rate. (default: `1e-3`)
- `-wd`, `--weight_decay` (float): Weight decay for regularization. (default: `1e-5`)
- `-bs`, `--batch_size` (int): Batch size. (default: `128`)
- `-c`, `--n_classes` (int): Number of classes. `1` for regression, `2` for binary classification, `>=3` for multi-class classification. (default: `1`)
- `-m`, `--metric` (str): Metric to evaluate the model. Supported metrics: `['accuracy', 'rmse']`. (default: `'acc'`)
- `-rp`, `--result-path` (str): Path to save the result. (default: `None`)
- `-s`, `--seed` (int): Random seed. (default: `0`)
- `-ld`, `--log-dir` (str): Log directory. (default: `'log'`)
- `-ded`, `--data-embed-dim` (int): Data embedding dimension. (default: `200`)
- `-ked`, `--key-embed-dim` (int): Key embedding dimension. (default: `200`)
- `-nh`, `--num-heads` (int): Number of heads in multi-head attention. (default: `4`)
- `--dropout` (float): Dropout rate. (default: `0.0`)
- `--party-dropout` (float): Dropout rate for entire party. (default: `0.0`)
- `-nlb`, `--n-local-blocks` (int): Number of local blocks. (default: `6`)
- `-nab`, `--n-agg-blocks` (int): Number of aggregation blocks. (default: `6`)
- `--knn-k` (int): k for KNN. (default: `100`)
- `--disable-pe` (bool): Disable positional encoding if set. (default: `False`)
- `--disable-dm` (bool): Disable dynamic masking if set. (default: `False`)
- `-paf`, `--pe-average-freq` (int): Average frequency for positional encoding on each party. (default: `0`)

### Example Usage

To train the FeT model on the `house` dataset, run the following command:

```bash
python src/script/train_fet.py -d house -m rmse -c 1 -p 2 -s 0 --knn-k 100 -nh 4 -ded 100 -ked 100 -nlb 3 -nab 3 -paf 1 --dropout 0.3 -g 0
```

### Experimentation Scripts

For conducting various experiments included in the paper, you can find the relevant scripts in the `src/script` directory. These scripts are designed to facilitate different experimental setups and can be customized as needed for your specific research requirements. The detailed usage of these scripts is as follows:

- `download_datasets.sh`: This script is used to download `gisette` and `mnist` datasets used in the experiments.
- `split_scale.sh`: This script is used to split `gisette` and `mnist` datasets into multiple parties with different hyperparameters.
- `run_real_fet.sh`: This script is used to run the FeT model on three real-world datasets, including `house`, `taxi`, and `hdb`.
- `run_scale.sh`: This script is used to run the FeT model on synthetic multi-party VFL datasets generated by splitting `gisette` and `mnist`.
- `ablation*.sh`: This series of scripts are used to run the ablation studies on different components of the FeT model.
  - `ablation_dm_or_not.sh`: This script runs experiments to compare the performance of FeT with and without dynamic masking.
  - `ablation_keynoise_baseline.sh`: This script conducts experiments to evaluate the impact of key noise on baseline models.
  - `ablation_keynoise.sh`: This script tests the robustness of FeT against different levels of key noise.
  - `ablation_knnk_real.sh`: This script performs ablation studies on the effect of different k values in KNN for real-world datasets.
  - `ablation_knnk.sh`: This script examines the impact of varying k values in KNN on synthetic datasets.
  - `ablation_party_dropout.sh`: This script evaluates the model's performance under different party dropout rates.
  - `ablation_pe_average_freq.sh`: This script investigates the effect of different average frequencies in positional encoding.
  - `ablation_pe_or_not.sh`: This script compares the performance of FeT with and without positional encoding.
  - `ablation_real_dm_or_not.sh`: This script runs dynamic masking ablation studies specifically on real-world datasets.


## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{wu2024fet,
  title={Federated Transformer: Multi-Party Vertical Federated Learning on Practical Fuzzily Linked Data},
  author={Wu, Zhaomin and Hou, Junyi and Diao, Yiqun and He, Bingsheng},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```
