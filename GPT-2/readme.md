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
Configuration: In `125M.yml`, you can set parameters such as learning rate, batch size, and training iterations; in `owt2_setup.yml`, you can specify the locations for log and checkpoint files. Furthermore, in `owt2_setup_tempnet.yml`, you can also configure parameters related to TempNet, such as $\rho$ and TempNet's learning rate.

Run the baseline method:
```bash
python ./deepy.py train.py configs/125M.yml owt2_setup.yml
```

Training GPT-2 with TempNet:
```bash
python ./deepy.py train.py configs/125M.yml owt2_setup_tempnet.yml
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
