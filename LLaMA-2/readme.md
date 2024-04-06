## Requirements
Our implementation is based on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). Please refer to this project for setting up the environment.

## Data
Please follow this [link](https://github.com/NVIDIA/Megatron-LM?tab=readme-ov-file#data-preprocessing) to prepare the training data

## Implementation
The implementation of TempNet and loss function can be found in `megatron/model/tempnet.py` and `megatron/core/tensor_parallel/cross_entropy.py`, respectively.

## Training
Configure your training settings within `pretrain_llama_distributed.sh`, then launch the training process by running 
```bash
bash pretrain_llama_distributed.sh
```


## Extract Tempnet
One can run `weight_convert.sh` to export the trained TempNet model.
