## Requirements
Our implementation is based on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). Please refer to this project for setting up the environment.

## Data
To download and tokenize the openwebtext2 dataset with the GPT2 Tokenizer, saving them to ./data you can run:
```bash
python prepare_data.py -d ./data openwebtext2
```

## Implementation
The implementation of TempNet and loss function can be found in `megatron/model/tempnet.py` and `megatron/core/tensor_parallel/cross_entropy.py`, respectively.

## Training
Configure training parameters in "pretrain_llama_distributed.sh", and run "bash pretrain_llama_distributed.sh" to train. 


## Evaluation
Run the following command for evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python ./deepy.py eval.py -d configs llama/7B.yml llama/train_config.yml \
	--eval_tasks piqa hellaswag winogrande lambada_openai wikitext lambada_standard arc_easy arc_challenge openbookqa boolq sciq siqa mathqa logiqa swag \
	--eval_results_prefix testtest
```
