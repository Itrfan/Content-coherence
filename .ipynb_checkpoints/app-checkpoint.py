import streamlit as st
import numpy as np
import re
import joblib
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from tempfile import NamedTemporaryFile
from pydub import AudioSegment
AudioSegment.converter = r"C:\ffmpeg\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg\ffprobe.exe"
import whisper

# ---------------------
# Load models
# ---------------------
rf = joblib.load("rf_model.pkl")
tfidf = joblib.load("tfidf.pkl")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
whisper_model = whisper.load_model("base")

# ---------------------
# Session state
# ---------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------
# Helpers
# ---------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,!? ]", "", text)
    return text.strip()

def filler_ratio(text):
    fillers = ["um", "uh", "er", "like", "you know"]
    words = text.split()
    return sum(words.count(f) for f in fillers) / len(words) if words else 0

def sentence_count(text):
    return len(sent_tokenize(text))

def sentence_length_variance(text):
    sents = sent_tokenize(text)
    lengths = [len(s.split()) for s in sents]
    return np.var(lengths) if len(lengths) > 1 else 0

def short_sentence_ratio(text):
    sents = sent_tokenize(text)
    short = [s for s in sents if len(s.split()) <= 3]
    return len(short) / len(sents) if sents else 0

def sentence_coherence(text):
    sents = sent_tokenize(text)
    if len(sents) < 2:
        return 0.0
    embeddings = embedder.encode(sents, show_progress_bar=False)
    sims = cosine_similarity(embeddings[:-1], embeddings[1:])
    return float(np.mean(sims))

# ---------------------
# Convert any audio to WAV
# ---------------------
def convert_to_wav(input_path):
    audio = AudioSegment.from_file(input_path)
    tmp_wav = NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(tmp_wav.name, format="wav")
    return tmp_wav.name

# ---------------------
# Analyze pauses
# ---------------------
def analyze_pauses(wav_path, silence_threshold=500, min_pause_sec=0.7):
    import wave
    with wave.open(wav_path, "rb") as wf:
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)

    energy = np.abs(audio)
    silent = energy < silence_threshold

    pauses = []
    current = 0
    for s in silent:
        if s:
            current += 1
        else:
            if current > 0:
                pauses.append(current)
                current = 0

    pauses_sec = [p / rate for p in pauses if p / rate >= min_pause_sec]

    return {
        "num_pauses": len(pauses_sec),
        "longest_pause": max(pauses_sec) if pauses_sec else 0.0,
        "avg_pause": np.mean(pauses_sec) if pauses_sec else 0.0,
    }

# ---------------------
# Score transcript
# ---------------------
def score_transcript(text):
    cleaned = clean_text(text)

    coherence = sentence_coherence(cleaned)
    filler = filler_ratio(cleaned)
    sent_cnt = sentence_count(cleaned)
    length_var = sentence_length_variance(cleaned)
    short_ratio = short_sentence_ratio(cleaned)

    sent_score = min(sent_cnt / 6, 1.0)

    raw_improvement = (
        coherence * 5
        + sent_score * 2
        - filler * 5
        - short_ratio * 3
    )

    extra_features = np.array([[
        coherence * 3.0,
        length_var * 2.0,
        filler * -5.0,
        sent_cnt * 1.0,
        short_ratio * -4.0
    ]])

    X_tfidf = tfidf.transform([cleaned])
    X = hstack([X_tfidf, extra_features])
    ref_score = rf.predict(X)[0]

    return {
        "reference_score": ref_score,
        "raw_improvement": raw_improvement,
        "coherence": coherence,
        "filler": filler,
        "sentences": sent_cnt,
        "short_ratio": short_ratio,
        "text": text,
    }

def get_relative_improvement(current):
    baseline = st.session_state.history[0]["raw_improvement"]
    return max(0, current["raw_improvement"] - baseline)

# ---------------------
# UI
# ---------------------
st.title("Presentation practice")

st.write("Upload audio or paste your transcript. The app gives feedback on clarity, coherence, filler words, and pauses.")

input_mode = st.radio("Choose input type", ["Upload audio", "Paste transcript"])

# ---------------------
# Audio upload
# ---------------------
if input_mode == "Upload audio":
    audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a"])
    if audio_file:
        st.audio(audio_file)

        with NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        # Convert to WAV for pause analysis
        wav_path = convert_to_wav(tmp_path)

        st.write("Transcribing...")
        result = whisper_model.transcribe(tmp_path)
        transcript = result["text"]

        st.text_area("Transcript", transcript, height=200)

        if st.button("Analyze recording"):
            score = score_transcript(transcript)
            score["pauses"] = analyze_pauses(wav_path)
            st.session_state.history.append(score)

# ---------------------
# Text input
# ---------------------
if input_mode == "Paste transcript":
    text_input = st.text_area("Paste transcript here", height=200)
    if st.button("Analyze text") and text_input.strip():
        score = score_transcript(text_input)
        score["pauses"] = None
        st.session_state.history.append(score)

# ---------------------
# Display results
# ---------------------
if st.session_state.history:
    current = st.session_state.history[-1]
    current["improvement"] = get_relative_improvement(current)

    st.subheader("Current result")
    st.metric("Improvement", f"{current['improvement']:.2f}")
    st.metric("Coherence", f"{current['coherence']:.2f}")
    st.metric("Filler ratio", f"{current['filler']:.3f}")
    st.metric("Reference score", f"{current['reference_score']:.2f}")

    if current.get("pauses"):
        st.subheader("Pauses")
        st.write(f"Number of long pauses: {current['pauses']['num_pauses']}")
        st.write(f"Longest pause: {current['pauses']['longest_pause']:.2f}s")
        st.write(f"Average pause: {current['pauses']['avg_pause']:.2f}s")

    st.subheader("Notes")
    if current["filler"] > 0.05:
        st.write("- Try to reduce filler words")
    if current["coherence"] < 0.45:
        st.write("- Work on smoother transitions")
    if current.get("pauses") and current["pauses"]["longest_pause"] > 1.5:
        st.write("- Some long pauses detected")

# ---------------------
# History
# ---------------------
if len(st.session_state.history) >= 2:
    st.subheader("Progress over attempts")
    for i, h in enumerate(st.session_state.history):
        st.write(f"Attempt {i+1}: Improvement={h['improvement']:.2f}, Coherence={h['coherence']:.2f}, Filler={h['filler']:.3f}")
