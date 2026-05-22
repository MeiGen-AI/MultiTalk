
export WAN_DISABLE_FLASH_ATTN="1"



# 1. Define your variables
IMAGE_PATH="/mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/local_data/avatars/sales_executive/executive.png"
VOICE_PATH="weights/Kokoro-82M/voices/af_heart.pt"
PROMPT="A professional speaks confidently directly to the camera."
SCRIPT_TEXT="Hello, welcome to Meity, a world of avatars for outreach and CRM. I'll be your virtual assistant."

# af_alloy.pt    af_nova.pt   am_fenrir.pt   bf_emma.pt      ef_dora.pt   hm_psi.pt         jm_kumo.pt      zf_xiaoyi.pt
# af_aoede.pt    af_river.pt  am_liam.pt     bf_isabella.pt  em_alex.pt   if_sara.pt        pf_dora.pt      zm_yunjian.pt
# af_bella.pt    af_sarah.pt  am_michael.pt  bf_lily.pt      em_santa.pt  im_nicola.pt      pm_alex.pt      zm_yunxi.pt
# af_heart.pt    af_sky.pt    am_onyx.pt     bm_daniel.pt    ff_siwis.pt  jf_alpha.pt       pm_santa.pt     zm_yunxia.pt
# af_jessica.pt  am_adam.pt   am_puck.pt     bm_fable.pt     hf_alpha.pt  jf_gongitsune.pt  zf_xiaobei.pt   zm_yunyang.pt
# af_kore.pt     am_echo.pt   am_santa.pt    bm_george.pt    hf_beta.pt   jf_nezumi.pt      zf_xiaoni.pt
# af_nicole.pt   am_eric.pt   bf_alice.pt    bm_lewis.pt     hm_omega.pt  jf_tebukuro.pt    zf_xiaoxiao.pt

IMAGE_PATH="/mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/local_data/avatars/greenscreen/green_woman4.png"
VOICE_PATH="weights/Kokoro-82M/voices/bf_isabella.pt"
PROMPT="A professional speaks confidently directly to the camera."
SCRIPT_TEXT="Hello, welcome to Meity, a world of avatars for outreach and CRM. I'll be your virtual assistant."


# 2. Use the variables inside the Heredoc

OUTPUT_PATH="/mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/video_generators/multitalk/local_data/temp/mini.json"

cat <<EOF > "$OUTPUT_PATH"
{
    "prompt": "$PROMPT",
    "cond_image": "$IMAGE_PATH",
    "tts_audio": {
        "text": "$SCRIPT_TEXT",
        "human1_voice": "$VOICE_PATH"
    },
    "cond_audio": {}
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
    --audio_mode tts \
    --audio_save_dir local_data/sales_test
