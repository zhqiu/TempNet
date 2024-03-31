The implementation of TempNet can be found in "megatron/model/tempnet.py".

The loss function is located in "megatron/core/tensor_parallel/cross_entropy.py".

Configure training parameters in "pretrain_llama_distributed.sh", and run "bash pretrain_llama_distributed.sh" to train. 

Requirements need to be installed from https://github.com/NVIDIA/Megatron-LM.
