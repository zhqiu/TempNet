## Requirements
Our implementation is based on [GPT-NeoX](https://github.com/EleutherAI/gpt-neox). Please refer to this project for setting up the environment.  
When evaluating the model, we use the [lm-evaluation-harness library](https://github.com/EleutherAI/lm-evaluation-harness).

## Data
To download and tokenize the openwebtext2 dataset with the GPT2 Tokenizer, saving them to ./data you can run:
```bash
python prepare_data.py -d ./data openwebtext2
```

## Implementation
The implementation of TempNet and loss function can be found in `megatron/model/tempnet.py`.

## Training
Due to the large size of the LLaMA 7B model parameters, we use 4 Nvidia A6000 GPUs. The command to start the experiment is as follows:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./deepy.py train.py -d configs llama/7B.yml llama/train_config.yml
```
In `configs/llama/7B.yml`, you can set parameters such as learning rate, batch size, and training iterations; in `configs/llama/train_config.yml`, you can specify the locations for log and checkpoint files, and some configure parameters related to TempNet, such as $\rho$ and TempNet's learning rate.



## Evaluation
Run the following command for evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python ./deepy.py eval.py -d configs llama/7B.yml llama/train_config.yml \
	--eval_tasks piqa hellaswag winogrande lambada_openai wikitext lambada_standard arc_easy arc_challenge openbookqa boolq sciq siqa mathqa logiqa swag \
	--eval_results_prefix testtest
```
