# Training Setup

This guide outlines the steps to set up the environment, prepare the dataset, and run training for OLMoE models.

## 1. Prerequisites

- **CUDA Version**: `12.4`
- **Python Version**: `3.9+` (It is best to use 3.10 or above. Although 3.9 can run, there are many incompatible uses.)
- **Operating System**: `Linux`

---

## 2. Environment Setup

It's recommended to use a virtual environment:

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 2.1 Additional Package Installation

After installing `requirements.txt`, please ensure the following versions are installed.

First, clone and install `OLMo`:
```bash
git clone https://github.com/allenai/OLMo.git
cd OLMo
pip install -e .[all]
```

Next, install specific package versions:
```bash
pip install git+https://github.com/Muennighoff/megablocks.git@olmoe
pip install -e .

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install ai2-olmo==0.6.0
pip install megablocks==0.7.0
pip install stanford-stk
pip install importlib_metadata
```
**Important:** Please ensure the above versions are correct. We are using CUDA version 12.4.

### 2.2 Grouped GEMM Installation
To ensure you can use `grouped_gemm`, we recommend installing it directly from the source repository:

```bash
pip install --verbose git+https://github.com/fanshiqing/grouped_gemm@main
```

### 2.3 OLMo Configuration Setup

#### Replace configuration files:

First, find the `olmo` library installation location:
```bash
pip show olmo
```
Navigate to the `olmo` library directory and replace all configuration files with the files provided in the `OLMo/newconfig` directory.

#### Configure attention mechanisms:

**For default SPAttention (no modification needed):**
- No changes required when using default `SPAttention`.

**For ablation study variants:**
- Modify line 488 in both `model.py` files (in the `olmo` library directory and `OLMo/olmo/model.py`) according to your needs:

```python
# Default SPAttention
from olmo.SPAttention import create_sp_attention_cache, sparse_attention

# LocalOnly variant
# from olmo.SPAttention_LocalOnly import create_sp_attention_cache, sparse_attention

# GappedBands variant
# from olmo.SPAttention_GappedBands import create_sp_attention_cache, sparse_attention

# ExclusiveBands variant
# from olmo.SPAttention_ExclusiveBands import create_sp_attention_cache, sparse_attention

# For standard causal mask configuration:
# from olmo.CausalAttention import create_sp_attention_cache, sparse_attention

# For other sparse attention configurations:
# from olmo.attention_variants import create_sp_attention_cache, sparse_attention
```

Then set the environment variable for your chosen configuration:

```bash
# Choose one of the following:
export ATTENTION_TYPE=bigbird
export ATTENTION_TYPE=reformer
export ATTENTION_TYPE=longformer
```

**Important:** Any modifications to `model.py` must be synchronized to the corresponding `model.py` file in the olmo library.


## 3. Dataset Download and Preprocessing

1.  **Download the dataset:**
    Go to `https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0` and download the dataset files from the `dclm` directory. Place them in a directory, e.g., `./dclm_raw_data`.

2.  **Convert to trainable format:**
    Use the `dolma` tool to tokenize and prepare the data.

    ```bash
    # Run tokenization
    dolma tokens \
        --documents "${PATH_TO_DOWNLOADED_DATA}/*.jsonl.gz" \ # Adjust glob pattern if needed
        --destination ${PATH_WHERE_TO_SAVE_TOKENIZED_DATA} \
        --tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
        --max_size '2_147_483_648' \
        --seed 0 \
        --tokenizer.eos_token_id 50279 \
        --tokenizer.pad_token_id 1 \
        --processes ${NUMBER_OF_CPU_CORES_TO_USE}
    ```


## 4. Running Training

You have two primary options to start training:

#### Option 1: Using `olmoe-gantry.sh`
This script likely handles more complex configurations or distributed training setups.

```bash
bash scripts/olmoe-gantry.sh
```
**Note:** You will need to select an appropriate configuration file (often YAML or similar) and adapt it to your specific setup (e.g., paths to tokenized data, model hyperparameters, hardware resources). Refer to the script or accompanying documentation for details on configuration.

#### Option 2: Using `train.py`
This script might be a more direct way to launch training, potentially for single-node or simpler setups.

```bash
python scripts/train.py
```
