"""
Real or AI? — Human vs. Model
HW1 Part 8. Guess whether each test-set face is real or AI-generated,
then see whether you or the trained model got it right.
"""
import base64
import json
import os
import random

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "best_model.keras")
INFO_PATH = os.path.join(APP_DIR, "model_info.json")
IMG_DIR = os.path.join(APP_DIR, "test_images")
LABELS_PATH = os.path.join(IMG_DIR, "labels.csv")
DISPLAY_DIR = os.path.join(APP_DIR, "display_images")   # full-quality originals shown to players
N_ROUNDS = 10
NAMES = {1: "Real", 0: "AI-generated"}          # label convention from the notebook

st.set_page_config(page_title="Real or AI?", page_icon="🙂", layout="centered")

# Design tokens: one type family (system), one spacing scale, one radius scale, meaning-only colour.
st.markdown("""
<style>
:root{
  --bg:#F3F4F1; --surface:#FFFFFF; --fill:#E7E9E4; --ink:#16181A; --ink2:#5F6468; --ink3:#9A9FA3;
  --accent:#3B3AB5; --right:#1E8E5A; --wrong:#D04434;
  --r-lg:28px; --r-md:20px; --r-sm:12px;
  --shadow:0 1px 2px rgba(22,24,26,.06), 0 12px 32px -12px rgba(22,24,26,.18);
  --spring:cubic-bezier(.34,1.56,.64,1);
  --font:-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --font-display:-apple-system,BlinkMacSystemFont,"SF Pro Display","Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
}
.stApp{background:var(--bg);}
.stApp, .stApp p, .stApp label, .stApp button{font-family:var(--font);color:var(--ink);-webkit-font-smoothing:antialiased;}
#MainMenu, footer, header[data-testid="stHeader"]{display:none !important;}
.block-container{max-width:440px;padding:2.5rem 1.25rem 3rem;}

.title{font-family:var(--font-display);font-weight:700;font-size:clamp(2.2rem,7vw,2.9rem);letter-spacing:-.025em;line-height:1.05;text-align:center;margin:0;padding:0;color:var(--ink);}
.subtitle{text-align:center;color:var(--ink2) !important;font-size:1.05rem;line-height:1.45;margin:.5rem auto 1.75rem;max-width:26rem;}

.score{display:grid;grid-template-columns:1fr 1px 1fr;align-items:center;background:var(--surface);border-radius:var(--r-md);box-shadow:var(--shadow);padding:.9rem 0;}
.score .side{text-align:center;}
.score .who{font-size:.82rem;font-weight:600;color:var(--ink2);}
.score .num{font-family:var(--font-display);font-weight:700;font-size:1.9rem;letter-spacing:-.02em;line-height:1.1;font-variant-numeric:tabular-nums;color:var(--ink);}
.score .num span{color:var(--ink3);font-weight:600;font-size:1.2rem;}
.score .rule{height:2.4rem;background:var(--fill);}

.round{display:flex;justify-content:space-between;align-items:center;margin:1.4rem .2rem .55rem;font-size:.88rem;color:var(--ink2);}
.round b{color:var(--ink);font-weight:600;}
.track{display:flex;gap:4px;margin:0 0 1rem;}
.track i{flex:1;height:5px;border-radius:3px;background:var(--fill);transition:background .3s;}
.track i.r{background:var(--right);} .track i.w{background:var(--wrong);} .track i.now{background:var(--ink3);}

.photo{position:relative;width:256px;max-width:100%;margin:0 auto;border-radius:var(--r-md);overflow:hidden;box-shadow:var(--shadow);background:var(--fill);line-height:0;
  outline:0 solid transparent;outline-offset:0;transition:outline-color .3s;}
.photo img{width:100%;aspect-ratio:1;object-fit:cover;display:block;}
.photo.r{outline:4px solid var(--right);outline-offset:-4px;} .photo.w{outline:4px solid var(--wrong);outline-offset:-4px;}
.badge{position:absolute;left:50%;bottom:12px;transform:translateX(-50%);line-height:1.2;white-space:nowrap;
  padding:.45rem .9rem;border-radius:999px;font-weight:600;font-size:.9rem;color:var(--ink);
  background:rgba(255,255,255,.72);backdrop-filter:blur(20px) saturate(180%);-webkit-backdrop-filter:blur(20px) saturate(180%);
  box-shadow:0 4px 18px rgba(0,0,0,.14);display:flex;align-items:center;gap:.45rem;animation:pop .45s var(--spring) both;}
@keyframes pop{from{opacity:0;transform:translate(-50%,14px) scale(.9);}}

.sheet{background:var(--surface);border-radius:var(--r-md);box-shadow:var(--shadow);padding:1.1rem 1.25rem;margin:1rem 0 .9rem;animation:rise .4s var(--spring) both;}
@keyframes rise{from{opacity:0;transform:translateY(10px);}}
.sheet h3{font-family:var(--font-display);font-weight:700;font-size:1.3rem;letter-spacing:-.015em;margin:0 0 .75rem;padding:0;color:var(--ink);}
.line{display:flex;align-items:center;gap:.7rem;padding:.55rem 0;border-top:1px solid var(--fill);}
.line .ic{flex:none;width:1.6rem;height:1.6rem;border-radius:50%;display:grid;place-items:center;}
.line .ic svg{width:.9rem;height:.9rem;}
.line .ic.r{background:rgba(30,142,90,.13);color:var(--right);} .line .ic.w{background:rgba(208,68,52,.12);color:var(--wrong);}
.line .t{flex:1;font-size:.98rem;color:var(--ink);}
.line .t small{display:block;color:var(--ink2);font-size:.84rem;margin-top:.05rem;}

.stButton button{width:100%;min-height:3.25rem;border-radius:999px;border:0;font-weight:600;font-size:1.05rem;
  transition:transform .12s var(--spring), background .2s;box-shadow:none;}
.stButton button p{font-size:1.05rem;font-weight:600;color:inherit !important;}
.stButton button:active{transform:scale(.97);}
.stButton button:focus-visible{outline:3px solid var(--accent);outline-offset:3px;}
.st-key-real button, .st-key-ai button{background:var(--surface);color:var(--ink);box-shadow:var(--shadow);}
.st-key-real button:hover, .st-key-ai button:hover{background:#FAFBF9;color:var(--accent);border:0;}
.st-key-next button, .st-key-again button{background:var(--accent);color:#fff;}
.st-key-next button:hover, .st-key-again button:hover{background:#3231A0;color:#fff;border:0;}

.final{text-align:center;margin:.5rem 0 1.4rem;animation:rise .5s var(--spring) both;}
.final .eyebrow{font-size:.95rem;color:var(--ink2);margin:0;}
.final h2{font-family:var(--font-display);font-weight:700;font-size:clamp(2rem,6.5vw,2.6rem);letter-spacing:-.025em;line-height:1.08;margin:.3rem 0 0;padding:0;color:var(--ink);}
.vs{display:grid;grid-template-columns:1fr 1fr;gap:.75rem;margin:0 0 1rem;}
.vs > div{background:var(--surface);border-radius:var(--r-md);box-shadow:var(--shadow);padding:1rem;text-align:center;}
.vs .who{font-size:.85rem;font-weight:600;color:var(--ink2);}
.vs .big{font-family:var(--font-display);font-weight:700;font-size:2.6rem;letter-spacing:-.03em;line-height:1.05;font-variant-numeric:tabular-nums;color:var(--ink);}
.vs .sub{font-size:.85rem;color:var(--ink2);}
.vs > div.win{box-shadow:0 0 0 2px var(--accent), var(--shadow);}
.grid{background:var(--surface);border-radius:var(--r-md);box-shadow:var(--shadow);padding:1rem;margin-bottom:1rem;}
.grid h4{font-size:.95rem;font-weight:600;margin:0 0 .75rem;padding:0;color:var(--ink);}
.tiles{display:grid;grid-template-columns:repeat(5,1fr);gap:.55rem;}
.tile img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:var(--r-sm);display:block;}
.tile .k{display:flex;justify-content:center;align-items:center;gap:.25rem;margin-top:.35rem;font-size:.72rem;font-weight:600;color:var(--ink2);}
.tile .k b{width:.5rem;height:.5rem;border-radius:50%;display:inline-block;}
.tile .k em{font-style:normal;margin-right:.15rem;}
.stApp .legend{font-size:.8rem;color:var(--ink2);margin:.8rem 0 0;}
.foot{text-align:center;font-size:.82rem;color:var(--ink3) !important;margin-top:2rem;line-height:1.5;}
@media (prefers-reduced-motion:reduce){.badge,.sheet,.final{animation:none}.stButton button{transition:none}}
</style>
""", unsafe_allow_html=True)

CHECK = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8.5l3.2 3L13 4.5"/></svg>'
CROSS = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>'


def show(html):
    st.markdown(" ".join(line.strip() for line in html.strip().splitlines()), unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the model…")
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_labels():
    return pd.read_csv(LABELS_PATH)


@st.cache_data
def load_info():
    if os.path.exists(INFO_PATH):
        with open(INFO_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data
def b64(file):
    with open(os.path.join(IMG_DIR, file), "rb") as f:
        return base64.b64encode(f.read()).decode()


@st.cache_data
def photo(file):
    """Image shown to the player: the full-quality original if it was exported, else the 128x128 version.
    The model always predicts on the 128x128 version, which is what it was trained on."""
    hi = os.path.join(DISPLAY_DIR, os.path.splitext(file)[0] + ".jpg")
    if os.path.exists(hi):
        with open(hi, "rb") as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    return "data:image/png;base64," + b64(file)


@st.cache_data(show_spinner="Preparing the faces…")
def model_probs():
    """P(real) for every game image in one batch. MobileNetV3 takes raw [0, 255] RGB pixels."""
    files = load_labels()["file"]
    x = np.stack([np.asarray(Image.open(os.path.join(IMG_DIR, f)).convert("RGB"), dtype="float32") for f in files])
    p = np.asarray(load_model()(x, training=False)).ravel()
    return dict(zip(files, p.astype(float)))


def new_game():
    labels = load_labels()
    real = labels[labels["label"] == 1].sample(N_ROUNDS - N_ROUNDS // 2)
    ai = labels[labels["label"] == 0].sample(N_ROUNDS // 2)
    st.session_state.game = {"rounds": pd.concat([real, ai]).sample(frac=1).to_dict("records"),
                             "i": 0, "revealed": False, "log": []}


def guess(choice):
    g = st.session_state.game
    if g["revealed"] or g["i"] >= N_ROUNDS:      # ignore a second click before the page refreshes
        return
    r = g["rounds"][g["i"]]
    p = model_probs()[r["file"]]
    model_choice = int(p >= 0.5)
    g["log"].append({"file": r["file"], "truth": int(r["label"]), "you": choice, "model": model_choice,
                     "conf": p if model_choice == 1 else 1 - p})
    g["revealed"] = True


def next_image():
    g = st.session_state.game
    if not g["revealed"]:                        # ignore a second click before the page refreshes
        return
    g["i"] += 1
    g["revealed"] = False


def icon(ok):
    return f'<div class="ic {"r" if ok else "w"}">{CHECK if ok else CROSS}</div>'


# ------------------------------------------------------------------
if "game" not in st.session_state:
    new_game()
g = st.session_state.game
load_model()
model_probs()

log = g["log"]
played = len(log)
you_ok = sum(e["you"] == e["truth"] for e in log)
model_ok = sum(e["model"] == e["truth"] for e in log)

if g["i"] < N_ROUNDS:
    # ---------------- a round ----------------
    show('<h1 class="title">Real or AI?</h1>'
         '<p class="subtitle">Look closely at each face and decide. The model guesses too.</p>')
    show(f"""
    <div class="score">
      <div class="side"><div class="who">You</div><div class="num">{you_ok}<span>/{played}</span></div></div>
      <div class="rule"></div>
      <div class="side"><div class="who">Model</div><div class="num">{model_ok}<span>/{played}</span></div></div>
    </div>""")

    # progress: each finished round shows whether *you* were right
    ticks = "".join(
        f'<i class="{("r" if log[k]["you"] == log[k]["truth"] else "w") if k < played else ("now" if k == g["i"] else "")}"></i>'
        for k in range(N_ROUNDS))
    show(f'<div class="round"><span>Round <b>{g["i"] + 1}</b> of {N_ROUNDS}</span>'
         f'<span>{"Your call" if not g["revealed"] else "Answer"}</span></div><div class="track">{ticks}</div>')

    r = g["rounds"][g["i"]]
    ring, badge = "", ""
    if g["revealed"]:
        e = log[-1]
        ring = "r" if e["you"] == e["truth"] else "w"
        text = "Real photo" if e["truth"] == 1 else "AI-generated"
        badge = f'<div class="badge">{text}</div>'
    show(f'<div class="photo {ring}"><img src="{photo(r["file"])}" alt="Face to judge">{badge}</div>')

    if not g["revealed"]:
        st.write("")
        a, b = st.columns(2)
        a.button("Real", key="real", on_click=guess, args=(1,))
        b.button("AI-Generated", key="ai", on_click=guess, args=(0,))
    else:
        e = log[-1]
        you_right, model_right = e["you"] == e["truth"], e["model"] == e["truth"]
        if you_right and model_right:
            head = "You both got it."
        elif you_right:
            head = "You got it. The model didn't."
        elif model_right:
            head = "The model got this one."
        else:
            head = "This one fooled you both."
        show(f"""
        <div class="sheet">
          <h3>{head}</h3>
          <div class="line">{icon(you_right)}<div class="t">You said {NAMES[e['you']]}</div></div>
          <div class="line">{icon(model_right)}<div class="t">Model said {NAMES[e['model']]}
            <small>{e['conf']:.0%} confident</small></div></div>
        </div>""")
        last = g["i"] + 1 == N_ROUNDS
        st.button("See Results" if last else "Next Image", key="next", on_click=next_image)

else:
    # ---------------- final summary ----------------
    if you_ok > model_ok:
        eyebrow, head = "You win", "Sharper than the machine."
    elif you_ok < model_ok:
        eyebrow, head = "The model wins", "The machine had the edge."
    else:
        eyebrow, head = "It's a tie", "Evenly matched."
    both_wrong = sum(e["you"] != e["truth"] and e["model"] != e["truth"] for e in log)
    show(f'<div class="final"><p class="eyebrow">{eyebrow}</p><h2>{head}</h2></div>')
    show(f"""
    <div class="vs">
      <div class="{'win' if you_ok > model_ok else ''}"><div class="who">You</div><div class="big">{you_ok}/{N_ROUNDS}</div>
        <div class="sub">{you_ok / N_ROUNDS:.0%} accuracy</div></div>
      <div class="{'win' if model_ok > you_ok else ''}"><div class="who">Model</div><div class="big">{model_ok}/{N_ROUNDS}</div>
        <div class="sub">{model_ok / N_ROUNDS:.0%} accuracy</div></div>
    </div>""")

    def dot(ok):
        return f'<b style="background:{"var(--right)" if ok else "var(--wrong)"}"></b>'

    tiles = "".join(
        f'<div class="tile"><img src="{photo(e["file"])}" alt="{NAMES[e["truth"]]} face">'
        f'<div class="k"><em>{"Real" if e["truth"] == 1 else "AI"}</em>{dot(e["you"] == e["truth"])}{dot(e["model"] == e["truth"])}</div></div>'
        for e in log)
    fooled = f"{both_wrong} fooled you both." if both_wrong else "None fooled you both."
    show(f"""
    <div class="grid"><h4>Every face this game</h4><div class="tiles">{tiles}</div>
      <p class="legend">Under each face: the true answer, then your result and the model's (green right, red wrong).
      {fooled}</p></div>""")
    st.button("Play Again", key="again", on_click=new_game)

info = load_info()
if "test_acc" in info:
    show(f'<p class="foot">The model is {info["name"]}, {info["test_acc"]:.0%} accurate on {info.get("n_test", 300)} held-out test faces.<br>'
         'Every face here comes from that held-out set.</p>')
