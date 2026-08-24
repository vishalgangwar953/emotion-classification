# 🎭 Emotion AI — Text Emotion Classifier

A deep learning NLP project that classifies free-form text into one of **six
basic emotions** — sadness, joy, love, anger, fear, and surprise. A
**Bidirectional GRU** network was trained on the
[`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) dataset
and served through an interactive Streamlit app.

**🔗 Live demo:** https://emotion-classification-qmdtqi4ujturfaxj5sjsog.streamlit.app/

<!-- Add a screenshot or GIF of the Predict tab here, e.g.: -->
<!-- ![App screenshot](assets/screenshot.png) -->

---

## Project structure

```
.
├── app.py                # Streamlit app (inference + UI)
├── code.ipynb            # Training notebook (EDA → model selection → export)
├── BiGRU_Model.keras     # Trained model weights (exported from code.ipynb)
├── tokenizer.pkl         # Fitted Keras tokenizer (exported from code.ipynb)
├── requirements.txt      # Python dependencies
└── README.md
```

> `app.py` resolves `BiGRU_Model.keras` and `tokenizer.pkl` relative to its
> own location, so all three files must live in the same folder — regardless
> of where you run `streamlit run` from.

## The app

The Streamlit app (`app.py`) has four tabs:

- **🔮 Predict** — enter any sentence (or pick a sample) and get the predicted
  emotion with a full confidence breakdown across all six classes.
- **📖 About the Model** — dataset details, preprocessing pipeline, and the
  two-phase methodology used to arrive at the final model.
- **🏗️ Architecture** — full layer-by-layer breakdown of the Bidirectional
  GRU, compilation settings, and a comparison against every architecture
  that was evaluated.
- **📊 Performance** — final held-out test accuracy and per-class
  precision/recall/F1.

## Dataset

[`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion), loaded
via Hugging Face `datasets`, with its official train/validation/test splits
used as-is:

| Split      | Samples |
|------------|---------|
| Train      | 16,000  |
| Validation | 2,000   |
| Test       | 2,000   |

Classes: `sadness`, `joy`, `love`, `anger`, `fear`, `surprise` — notably
imbalanced (`joy` and `sadness` dominate, `surprise` is the smallest class),
so **balanced class weights** (`sklearn.utils.class_weight`) were applied
during training. The test set was kept completely untouched until the final
evaluation, to avoid data leakage into model selection.

## Preprocessing

1. **Tokenization** — Keras `Tokenizer`, vocabulary capped at 10,000 words,
   `fit_on_texts` called **only on training text**
2. **OOV handling** — unseen words mapped to an `<OOV>` token
3. **Sequencing** — text converted to integer sequences
4. **Padding/truncation** — every sequence normalized to a fixed length of
   50 tokens (`post` padding & truncating)

## Methodology

Training was done in two phases (see `code.ipynb`), comparing models purely
on **validation** metrics — the test set was reserved for a single final
evaluation.

### Phase 1 — Foundational models

Plain (non-bidirectional) SimpleRNN, LSTM, and GRU, each with a 128‑dim
embedding and two stacked recurrent layers (128 → 64 units, dropout 0.5).

| Model     | Validation Accuracy | Validation Loss |
|-----------|---------------------|------------------|
| **LSTM**  | **89.85%**          | 0.3291           |
| GRU       | 35.10%              | 1.7695           |
| SimpleRNN | 19.95%              | 1.7841           |

LSTM was the clear winner; the plain SimpleRNN and GRU struggled to move
past the majority-class baseline within 20 epochs / early stopping.

### Phase 2 — Bidirectional models

The two strongest recurrent unit types (LSTM, GRU) were rebuilt as **stacked
Bidirectional** networks with larger 300‑dim embeddings (128 → 64 units,
dropout 0.5 each).

| Model                        | Validation Accuracy | Validation Loss |
|-------------------------------|---------------------|------------------|
| **Bidirectional GRU** ✅ (selected) | **93.10%**    | 0.1960           |
| Bidirectional LSTM             | 91.90%              | 0.2540           |

**Bidirectional GRU** was selected as the final model — it slightly
outperformed the Bidirectional LSTM while using a simpler gating mechanism
(2 gates vs. LSTM's 3), making it faster to train with fewer parameters.

## Final model architecture

```
Embedding(input_dim=10000, output_dim=300, input_length=50)
Bidirectional(GRU(128, return_sequences=True))
Dropout(0.5)
Bidirectional(GRU(64))
Dropout(0.5)
Dense(6, activation='softmax')
```

| Layer | Type              | Configuration                                     | Purpose |
|-------|-------------------|----------------------------------------------------|---------|
| 1     | Embedding          | input_dim=10,000, output_dim=300, input_length=50 | Maps each token to a 300-dim dense vector |
| 2     | Bidirectional GRU  | 128 units, return_sequences=True                  | Forward + backward pass, outputs full sequence (256-dim per step) |
| 3     | Dropout            | rate=0.5                                          | Regularization |
| 4     | Bidirectional GRU  | 64 units                                          | Forward + backward pass, outputs final context vector (128-dim) |
| 5     | Dropout            | rate=0.5                                          | Regularization |
| 6     | Dense (Output)     | 6 units, softmax                                  | Probability distribution over 6 emotion classes |

**Training setup:**
- Loss: sparse categorical cross-entropy
- Optimizer: Adam
- Class weighting: balanced
- Early stopping: monitors `val_loss`, patience 3, best weights restored
- Epochs: up to 20 (stopped early); batch size: 32

## Final test set performance

Evaluated **once**, on the held-out 2,000-sample test set, after model
selection was already finalized:

| Metric        | Value  |
|---------------|--------|
| Test Accuracy | 91.55% |
| Test Loss     | 0.2253 |

**Per-class classification report:**

| Emotion  | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Sadness  | 0.9818    | 0.9294 | 0.9549   | 581     |
| Joy      | 0.9722    | 0.9050 | 0.9374   | 695     |
| Love     | 0.7451    | 0.9560 | 0.8375   | 159     |
| Anger    | 0.9094    | 0.9491 | 0.9288   | 275     |
| Fear     | 0.8767    | 0.8571 | 0.8668   | 224     |
| Surprise | 0.6129    | 0.8636 | 0.7170   | 66      |

Macro avg F1: **0.8737** &nbsp;|&nbsp; Weighted avg F1: **0.9182**

- Strongest classes: **sadness** and **joy** (largest support, highest precision)
- Weakest class: **surprise** — smallest class (only 66 test samples), so
  lower precision is expected; recall is still strong at 86.4%
- **Love** shows high recall but lower precision — the model sometimes
  confuses affectionate language with joy
- The confusion matrix (see `code.ipynb`) shows most confusion between
  semantically related pairs — e.g. **love ↔ joy** and **fear ↔ sadness**

## Reproducing / retraining

The full pipeline — data loading, EDA, preprocessing, all five model
trainings (Phase 1 + Phase 2), evaluation, and artifact export — is in
[`code.ipynb`](./code.ipynb). It was originally run on Google Colab with a
T4 GPU; a GPU is recommended but not required (CPU training will just be
slower).

```bash
pip install tensorflow datasets scikit-learn pandas numpy matplotlib seaborn
jupyter notebook code.ipynb
```

Running the full notebook end-to-end regenerates `BiGRU_Model.keras` and
`tokenizer.pkl` inside an `ArtifactsFinal/` folder — copy both into the app
folder (next to `app.py`) to update the deployed model.

## Running the app locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Deployment

This app is deployable as-is on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this repo to GitHub (including `BiGRU_Model.keras` and `tokenizer.pkl`).
2. On Streamlit Community Cloud, create a new app pointing at this repo and `app.py`.
3. Streamlit installs everything from `requirements.txt` automatically.

## Tech stack

| Component          | Tool                                  |
|---------------------|----------------------------------------|
| Model               | Bidirectional GRU (TensorFlow/Keras)  |
| Training/EDA        | Jupyter, pandas, seaborn, matplotlib, scikit-learn |
| Dataset              | dair-ai/emotion (Hugging Face `datasets`) |
| App interface        | Streamlit                             |
| Charts (in-app)       | Plotly                                |

## License

_Add a license (e.g. MIT) here if you want the project to be reusable by others._