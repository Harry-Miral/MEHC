# MEHC

Training Setup
This guide outlines the steps to set up the environment, prepare the dataset, and run training for OLMoE models.

1. Prerequisites
CUDA Version: 12.4
Python Version: 3.9+ (It is best to use 3.10 or above. Although 3.9 can run, there are many incompatible uses.)
Operating System: Linux
2. Environment Setup
It's recommended to use a virtual environment:

python3.10 -m venv venv
source venv/bin/activate
Install the required Python packages:

pip install -r requirements.txt
2.1 Additional Package Installation
After installing requirements.txt, please ensure the following versions are installed:
git clone https://github.com/allenai/OLMo.git
cd OLMo
pip install -e .[all]

pip install git+https://github.com/Muennighoff/megablocks.git@olmoe
pip install -e .

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install ai2-olmo==0.6.0
pip install megablocks==0.7.0
pip install stanford-stk
pip install importlib_metadata
Important: Please ensure the above versions are correct. We are using CUDA version 12.4.

2.2 Grouped GEMM Installation
To ensure you can use grouped_gemm, we recommend manual local compilation and installation.

Note: Due to file size limitations, the grouped_gemm.zip file cannot be uploaded through anonymous channels. You will need to download it separately. If you can install a pre-compiled version via pip, you do not need to perform the subsequent grouped_gemm modifications. If you use the pre-compiled version, please ensure you follow the modification process described below.

Extract and compile grouped_gemm:

# Extract the provided grouped_gemm.zip
unzip grouped_gemm.zip
cd grouped_gemm

# Install dependencies and compile
sudo apt update
sudo apt install ninja-build
python setup.py build_ext --inplace
python setup.py egg_info
Note: Ensure the grouped_gemm folder is located under the OLMo directory.

Configure megablocks integration:

# Find megablocks installation location
pip show megablocks
Navigate to the megablocks installation directory, find megablocks/grouped_gemm_util.py, and add the following code at the beginning of the file:

import sys
import os
sys.path.insert(0, '/data/OLMo/grouped_gemm')
import torch
torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
os.environ['LD_LIBRARY_PATH'] = f"{torch_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
Test the installation:

python -c "import grouped_gemm_backend; print('✓ Backend imported successfully')"
python -c "import grouped_gemm; print('✓ Main package imported successfully')"
2.3 OLMo Configuration Setup
Replace configuration files:

# Find olmo library installation location
pip show olmo
Navigate to the olmo library directory and replace all configuration files with the files provided in the OLMo/newconfig directory.

Configure attention mechanisms:

For default SPAttention (no modification needed):

No changes required when using default SPAttention.
For ablation study variants: Modify line 488 in both model.py files (in olmo library directory and OLMo/olmo/model.py) according to your needs:

# Default SPAttention
from olmo.SPAttention import create_sp_attention_cache, sparse_attention

# LocalOnly variant
from olmo.SPAttention_LocalOnly import create_sp_attention_cache, sparse_attention

# GappedBands variant
from olmo.SPAttention_GappedBands import create_sp_attention_cache, sparse_attention

# ExclusiveBands variant
from olmo.SPAttention_ExclusiveBands import create_sp_attention_cache, sparse_attention
For standard causal mask configuration:

from olmo.CausalAttention import create_sp_attention_cache, sparse_attention
For other sparse attention configurations:

from olmo.attention_variants import create_sp_attention_cache, sparse_attention
Then set the environment variable for your chosen configuration:

# Choose one of the following:
export ATTENTION_TYPE=bigbird
export ATTENTION_TYPE=reformer
export ATTENTION_TYPE=longformer
Important: Any modifications to model.py must be synchronized to the corresponding model.py file in the olmo library.

3. Dataset Download and Preprocessing
Download the dataset: Go to https://huggingface.co/allenai/OLMoE-1B-7B-0924/tree/main/dclm and download the dataset files from the dclm directory. Place them in a directory, e.g., ./dclm_raw_data.

Convert to trainable format: Use the dolma tool to tokenize and prepare the data.

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
4. Running Training
You have two primary options to start training:

Option 1: Using olmoe-gantry.sh This script likely handles more complex configurations or distributed training setups.

bash scripts/olmoe-gantry.sh
Note: You will need to select an appropriate configuration file (often YAML or similar) and adapt it to your specific setup (e.g., paths to tokenized data, model hyperparameters, hardware resources). Refer to the script or accompanying documentation for details on configuration.
Option 2: Using train.py This script might be a more direct way to launch training, potentially for single-node or simpler setups.

python scripts/train.py
Note: This script might also require configuration, typically through command-line arguments or a configuration file. Check python scripts/train.py --help or the script's contents for how to specify dataset paths, model parameters, etc.
Ensure that your training scripts/configurations point to the ${PATH_WHERE_TO_SAVE_TOKENIZED_DATA} created in the preprocessing step.
