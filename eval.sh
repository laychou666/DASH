# export HF_HOME="/path/to/your/hf_cache"
# Set WRAPPER=DASH to enable DASH compression; set to None to disable.
export WRAPPER=DASH

DASH_RHO_AUDIO=0.45
DASH_RHO_VIDEO=0.76
DASH_G=3
DASH_CONTEXTUAL_RATIO=0.05

# Multi-GPU Configuration
NUM_GPUS=8

if [ "$NUM_GPUS" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ "$NUM_GPUS" -eq 8 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

accelerate launch \
    --num_processes=$NUM_GPUS \
    --num_machines=1 \
    --mixed_precision=no \
    --main_process_port=12347 \
    -m lmms_eval \
    --model qwen2_5_omni \
    --model_args "pretrained=Qwen/Qwen2.5-Omni-7B,attn_implementation=flash_attention_2,max_num_frames=768,DASH_RHO_AUDIO=${DASH_RHO_AUDIO},DASH_RHO_VIDEO=${DASH_RHO_VIDEO},DASH_G=${DASH_G},DASH_CONTEXTUAL_RATIO=${DASH_CONTEXTUAL_RATIO}" \
    --tasks videomme \
    --batch_size 1 \
    --output_path ./logs/dash/
