The implementation of TempNet and loss function can be found in "megatron/model/tempnet.py".

Run "bash run_gpt2.sh" to train and run "bash eval_gpt2.sh" to evaluate the model.




## Requirements
Our implementation is based on [GPT-NeoX](https://github.com/EleutherAI/gpt-neox). Please refer to this project for setting up the environment.

## Data
To download and tokenize the openwebtext2 dataset with the GPT2 Tokenizer, saving them to ./data you can run:
```bash
python prepare_data.py -d ./data openwebtext2
```

## Training
Run the baseline method:
```bash

```

Training GPT-2 with TempNet:
```bash

```

## Evaluation
