import json
import pickle
import re
import html
import emoji
import numpy as np

import uvicorn 
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from symspellpy.symspellpy import SymSpell
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# === FastAPI app ===
app = FastAPI(title="Tweet Classifier with UI")

# Mount static folder (for CSS, images, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# === Load config, tokenizer, model at startup ===
MODEL_PATH = r"C:\Users\91781\OneDrive\Desktop\MLProjects\HateSpeechOffensive-LSTM\model.keras"  # adjust or use relative path
TOKENIZER_PATH = "tokenizer.pickle"
CONFIG_PATH = "config.json"
SYMSPELL_DICT = r"C:\Users\91781\OneDrive\Desktop\MLProjects\datasets\fdsymspelly.txt"  # adjust

model = None
tokenizer = None
max_len = None
class_labels = None
sym_spell = None

@app.on_event("startup")
def load_resources():
    global model, tokenizer, max_len, class_labels, sym_spell

    # Load model (use compile=False for prediction-only)
    model = load_model(MODEL_PATH, compile=False)

    # Load config
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    max_len = cfg["max_len"]
    class_labels = cfg["class_labels"]

    # Load tokenizer
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    # Initialize symspell
    sym_spell = SymSpell(max_dictionary_edit_distance=2)
    sym_spell.load_dictionary(SYMSPELL_DICT, term_index=0, count_index=1)


# === Preprocessing functions (same as your notebook) ===
def correct_text(text: str) -> str:
    corrected_words = []
    for word in text.split():
        if word.startswith("EMOJI_"):
            corrected_words.append(word)
        else:
            suggestions = sym_spell.lookup_compound(word, max_edit_distance=2)
            if suggestions:
                corrected_words.append(suggestions[0].term)
            else:
                corrected_words.append(word)
    return " ".join(corrected_words)

def clean_text_data(text: str) -> str:
    # Lowercase
    text = text.lower()
    text = html.unescape(text)

    # Emojis -> EMOJI_name
    text = emoji.demojize(text)
    text = re.sub(r":([a-zA-Z0-9_]+):", r" EMOJI_\1 ", text)

    # Strip HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove URLs, emails
    text = re.sub(r"(https?://\S+|www\.\S+|\S+@\S+\.\S+)", " ", text)

    # Mentions + RT
    text = re.sub(r"(@\w+|rt)", " ", text)

    # Remove '#' char only (optional)
    text = re.sub(r"#", "", text)

    # Spell correct
    text = correct_text(text)

    # Remove punctuation except underscore
    text = re.sub(r"[^a-zA-Z0-9_\s]", " ", text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [w for w in text.split() if w not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)


# === Routes ===

# Serve the frontend page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Renders the HTML page which has a textbox and a predict button.
    """
    return templates.TemplateResponse("index.html", {"request": request})


# JSON API predict endpoint (used by the frontend JS)
class TweetRequest(BaseModel):
    tweet: str

@app.post("/predict")
async def predict_api(req: TweetRequest):
    tweet_input = req.tweet.strip()
    if not tweet_input:
        return JSONResponse({"error": "Empty tweet"}, status_code=400)

    processed = clean_text_data(tweet_input)
    seq = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq, maxlen=max_len, padding="pre")
    pred_probs = model.predict(padded)
    pred_idx = int(np.argmax(pred_probs, axis=1)[0])
    pred_label = class_labels[str(pred_idx)]

    return {
        "tweet": tweet_input,
        "processed": processed,
        "predicted_label": pred_label,
        "probabilities": {class_labels[str(i)]: float(pred_probs[0][i]) for i in range(len(class_labels))}
    }

if __name__ == "__main__":
    print("\n🚀 Server running! Open this link in your browser:")
    print("👉 http://127.0.0.1:8000/\n")   # ⬅️ changed to `/` instead of `/docs`
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
