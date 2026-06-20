# 🔢 Handwritten Digit Recognizer
> CNN trained on MNIST · Flask + Canvas UI · ≥ 98 % test accuracy

---

## Project Structure

```
mnist_digit_recognizer/
│
├── model/
│   ├── train.py           ← CNN training script
│   └── mnist_cnn.h5       ← saved model (created after training)
│
├── app/
│   ├── app.py             ← Flask web application
│   ├── templates/
│   │   └── index.html     ← drawing UI (Canvas + JS)
│   └── static/
│       ├── training_history.png   ← generated after training
│       └── confusion_matrix.png  ← generated after training
│
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1 — Create a virtual environment (recommended)

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Train the model

```bash
python model/train.py
```

What this does:
- Downloads MNIST automatically (~11 MB, once only)
- Trains a CNN for up to 30 epochs with early stopping
- Saves `model/mnist_cnn.h5`
- Saves `app/static/training_history.png`
- Saves `app/static/confusion_matrix.png`

Expected output:
```
Test Accuracy : 99.20%
Test Loss     : 0.0261
```

### 4 — Run the web app

```bash
python app/app.py
```

Open your browser at **http://127.0.0.1:5000**

---

## How it works

```
User draws on canvas (280×280 px)
        │
        ▼ JavaScript
Canvas → PNG → base64 data-URL
        │
        ▼ POST /predict (JSON)
Flask receives data-URL
        │
        ▼ Pillow
Decode → grayscale → invert → resize 28×28 → normalise
        │
        ▼ TensorFlow
model.predict() → probabilities for digits 0–9
        │
        ▼ JSON response
{ digit: 7, confidence: 99.4, probabilities: [...] }
        │
        ▼ JavaScript
Update UI: big digit, confidence badge, bar chart
```

---

## Model Architecture

| Layer | Output Shape | Parameters |
|---|---|---|
| Conv2D(32, 3×3) + BN + ReLU | 28×28×32 | 320 |
| Conv2D(32, 3×3) + BN + ReLU | 28×28×32 | 9,248 |
| MaxPool2D + Dropout(0.25) | 14×14×32 | — |
| Conv2D(64, 3×3) + BN + ReLU | 14×14×64 | 18,496 |
| Conv2D(64, 3×3) + BN + ReLU | 14×14×64 | 36,928 |
| MaxPool2D + Dropout(0.25) | 7×7×64 | — |
| Flatten | 3136 | — |
| Dense(128) + BN + ReLU + Dropout(0.5) | 128 | 401,408 |
| Dense(10, softmax) | 10 | 1,290 |
| **Total** | | **~468K** |

Training tricks:
- **Data augmentation**: random rotations ±10°, shifts ±10%, zoom ±10%
- **Batch normalisation**: faster convergence, acts as regulariser
- **ReduceLROnPlateau**: halves LR when validation loss stalls
- **EarlyStopping**: restores best weights automatically

---

## Requirements

| Package | Purpose |
|---|---|
| tensorflow ≥ 2.12 | CNN training & inference |
| flask ≥ 3.0 | Web server |
| Pillow ≥ 10.0 | Canvas image → numpy |
| numpy | Array ops |
| scikit-learn | Confusion matrix, report |
| matplotlib / seaborn | Visualisation |

---

## Tips for better predictions

- Draw digits **large** and **centred** — same as MNIST style
- Use the full canvas width; avoid tiny strokes in one corner
- If accuracy seems low, re-train with more augmentation or extra epochs

---

*Built with TensorFlow 2.x · Flask · Vanilla JS — no heavy frontend framework needed.*
        