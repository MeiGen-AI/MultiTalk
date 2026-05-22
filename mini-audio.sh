
export WAN_DISABLE_FLASH_ATTN="1"



# 1. Define your variables
IMAGE_PATH="/mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/local_data/avatars/sales_executive/executive.png"
PROMPT="A professional speaks confidently directly to the camera."
AUDIO_PATH="/mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/local_data/audio-tests/google-german.wav"


OUTPUT_PATH="/mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/video_generators/multitalk/local_data/temp/audio.json"

cat <<EOF > "$OUTPUT_PATH"
{
    "prompt": "$PROMPT",
    "cond_image": "$IMAGE_PATH",
    "cond_audio": {
        "person1": "$AUDIO_PATH"
    }
}
EOF

echo "JSON saved to $OUTPUT_PATH"

python generate_multitalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --input_json "$OUTPUT_PATH" \
    --sample_steps 8 \
    --mode streaming \
    --num_persistent_param_in_dit 30 \
    --audio_mode localfile \
    --audio_save_dir local_data/sales_test
