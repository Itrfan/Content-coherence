import whisper
model = whisper.load_model("base")
result = model.transcribe("small_audio.wav")
print(result["text"])
