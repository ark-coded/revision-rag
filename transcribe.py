import json
from pathlib import Path
from faster_whisper import WhisperModel, BatchedInferencePipeline

#"large-v3" for heavy Hinglish/technical terms" &  large-v3-turbo" for faster runs
model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
batched_model = BatchedInferencePipeline(model=model)

#Paths
audio_dir = Path("audio")
out_dir = Path("transcripts")
out_dir.mkdir(exist_ok=True)

for audio_file in sorted(audio_dir.glob("*.m4a")):
    out_file = out_dir / f"{audio_file.stem}.json"
    if out_file.exists():
        print(f"Skipping (already done): {audio_file.name}")
        continue

    info_file = audio_dir / f"{audio_file.stem}.info.json"
    meta = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}
    