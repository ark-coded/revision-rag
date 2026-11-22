import json
from pathlib import Path
from faster_whisper import WhisperModel, BatchedInferencePipeline

#"large-v3" for heavy Hinglish/technical terms" &  large-v3-turbo" for faster runs
model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
batched_model = BatchedInferencePipeline(model=model)