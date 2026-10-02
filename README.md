# Real or AI? Human vs. Model

HW1 Part 8. A game where you compete against our best model at telling real faces from AI-generated ones.

**Play it:** _paste your Streamlit Community Cloud URL here_

## How it works
- A held-out test face is shown; you click **Real** or **AI-Generated**.
- The model makes its prediction on the same face.
- The answer appears on the photo, which is outlined green if you were right and red if not. A card shows both verdicts and the model's confidence.
- A live scoreboard tracks both accuracies and the round number. Click **Next Image** to continue.
- After 10 rounds, a summary shows who won, both accuracies and every face you saw.

## Model
MobileNetV3Small pre-trained on ImageNet with a custom classification head: whichever of the Part 5 (frozen) and Part 6 (fine-tuned) models had the lower validation loss. Its name and test accuracy are in `model_info.json`.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app |
| `best_model.keras` | Trained model |
| `requirements.txt` | Dependencies (TensorFlow/Keras pinned to the training versions) |
| `test_images/` | 80 held-out test faces (40 real, 40 AI) + `labels.csv` (1 = real, 0 = AI) |
| `model_info.json` | Model name and test accuracy shown in the app |
| `.streamlit/config.toml` | Colour theme |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Design
Guided by Apple's Human Interface principles: the face is the content and gets most of the screen (clarity); the interface stays neutral and uses colour only for meaning, green for right and red for wrong (deference); results float above the page as frosted labels and cards (depth); and every tap gets an immediate, spring-like response, turned off when the device asks for reduced motion (feedback). The app uses the device's own system font.
