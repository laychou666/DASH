import sys
sys.path = [p for p in sys.path if '.local' not in p] + [p for p in sys.path if '.local' in p]

import soundfile as sf
from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import argparse

parser = argparse.ArgumentParser(description="DASH Demo with Qwen2.5-Omni")
parser.add_argument('--model_path', type=str, default="Qwen/Qwen2.5-Omni-7B", help='Path to model (local or HuggingFace)')
parser.add_argument('--video', type=str, default="assets/example.mp4", help='Path to input video file')
parser.add_argument('--describe', type=str, default="Describe this video.", help='Instruction text for the model')
parser.add_argument('--dash', action='store_true', help='Whether to enable DASH compression')
parser.add_argument('--rho_audio', type=float, default=0.45, help='Merging ratio for audio')
parser.add_argument('--rho_video', type=float, default=0.76, help='Merging ratio for video')
parser.add_argument('--g', type=int, default=3)
parser.add_argument('--contextual_ratio', type=float, default=0.05, help='Contextual ratio')
parser.add_argument('--audio_guidance_strength', type=float, default=0.3, help='Audio guidance strength for DASH')
args = parser.parse_args()

if args.dash:
    print("Using DASH for Inference")
    dash_config = {
        "rho_audio": args.rho_audio,
        "rho_video": args.rho_video,
        "g": args.g,
        "contextual_ratio": args.contextual_ratio,
        "use_adaptive_grouping": True,
        "audio_guidance_strength": args.audio_guidance_strength,
    }
    from dash.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
else:
    from transformers import Qwen2_5OmniForConditionalGeneration

model_path = args.model_path
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)

model.dash_config = None
if args.dash:
    model.thinker.dash_config = dash_config
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": args.video},
            {"type": "text", "text": args.describe},
        ],
    },
]

USE_AUDIO_IN_VIDEO = True

text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
num_input_frames = videos[0].shape[0]
if hasattr(model, 'thinker'):
    model.thinker.nframes = num_input_frames
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)

print('Generating...')

from transformers import TextStreamer
streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False, temperature=1, streamer=streamer)
