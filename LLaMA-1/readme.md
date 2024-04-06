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


## Evaluation
For the model trained by the baseline method:
```bash
python ./deepy.py evaluate.py configs/125M.yml owt2_setup.yml --eval_results_prefix tau1.0 \
	--eval_tasks logiqa mathqa hellaswag swag lambada_openai lambada_standard piqa sciq wikitext
```

For the model trained by TempNet:
```bash
python ./deepy.py evaluate.py configs/125M.yml owt2_setup_tempnet.yml --eval_results_prefix tempnet \
       	--eval_tasks logiqa mathqa hellaswag swag lambada_openai lambada_standard piqa sciq wikitext
```
