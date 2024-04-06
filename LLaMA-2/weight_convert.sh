python weights_conversion/megatron_to_hf.py \
    --input_dir /data/siqi/Megatron-LM/models/llama2-13b-chat \
    --num_output_shards 4 \
    --model llama2 \
    --output_dir tempnet_models/chat/llama2-13b-chat/rho8.0-tau2-re-lr2e-4 \
    --vocab_file /home/grads/s/siqi/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93/tokenizer.model \
    --tokenizer_model /home/grads/s/siqi/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93/tokenizer.model
