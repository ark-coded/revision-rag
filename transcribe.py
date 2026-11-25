import json
import subprocess
import tempfile
import numpy as np
import soundfile as sf
import os
from pathlib import Path

nvidia_root = Path(__file__).resolve().parent / ".venv" / "Lib" / "site-packages" / "nvidia"
if nvidia_root.exists():
    dll_dirs = [str(p) for p in nvidia_root.glob("*/bin") if p.is_dir()]
    if dll_dirs:
        os.environ["PATH"] = os.pathsep.join(dll_dirs) + os.pathsep + os.environ.get("PATH", "")
        for d in dll_dirs:
            try:
                os.add_dll_directory(d)
            except (OSError, AttributeError):
                pass
from faster_whisper import WhisperModel, BatchedInferencePipeline

model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
batched_model = BatchedInferencePipeline(model=model)

audio_dir = Path("audio")
out_dir = Path("transcripts")
out_dir.mkdir(exist_ok=True)

def load_audio_16k_mono(file_path: Path) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(file_path),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                str(wav_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        audio, sr = sf.read(str(wav_path), dtype="float32")
        return audio
    finally:
        if wav_path.exists():
            wav_path.unlink()

for audio_file in sorted(audio_dir.glob("*.m4a")):
    out_file = out_dir / f"{audio_file.stem}.json"
    if out_file.exists():
        print(f"Skipping (already done): {audio_file.name}")
        continue

    info_file = audio_dir / f"{audio_file.stem}.info.json"
    meta = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}

    audio_data = load_audio_16k_mono(audio_file)
    segments, _ = batched_model.transcribe(audio_data, batch_size=8, vad_filter=True)
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
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Transcribed: {payload['title']} ({len(records)} segments)")