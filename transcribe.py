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
    
    #Lower batch_size from 8 to 4 if CUDA error
    segments, _ = batched_model.transcribe(str(audio_file), batch_size=8, vad_filter=True)
    records = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments
    ]

    payload = {
        "video_id": meta.get("id", audio_file.stem.split("_", 1)[-1]),
        "title": meta.get("title", audio_file.stem),
        "url": meta.get("webpage_url", ""),
        "segments": records,
    }