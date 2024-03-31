CUDA_VISIBLE_DEVICES=0 python ./deepy.py eval.py -d configs llama/7B.yml llama/train_config.yml \
	--eval_tasks piqa hellaswag winogrande lambada_openai wikitext lambada_standard arc_easy arc_challenge openbookqa boolq sciq siqa mathqa logiqa swag \
	--eval_results_prefix testtest




