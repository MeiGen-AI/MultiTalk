AVATAR_DIR = '/home/web/partha/vidLink/video_generators/multitalk/Input_outputs/input_files/sales_executive'
CKPT_DIR = './weights/Wan2.1-I2V-14B-480P'
WAV2VEC_DIR = './weights/chinese-wav2vec2-base'
KOKORO_DIR = './weights/Kokoro-82M'
TTS_VOICE = './weights/Kokoro-82M/voices/af_heart.pt'
SAMPLE_STEPS = 30

# Time required to generate 1 second of video (in seconds).
# For multitalk: 4 minutes per video second = 240 seconds
TIME_PER_VIDEO_SECOND_SECONDS = 300.0
