# Viterbi Algorithm with Macro‑Driven Transitions

Applies the Viterbi algorithm to find the most likely sequence of hidden regimes (bull, bear, neutral) given ETF returns. Transition probabilities depend on macro variables (VIX, DXY, yields) via a composite factor. The per‑ETF score is the log‑likelihood of the most probable path – a measure of how well the regime model fits.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Hidden regimes with normal emissions
- Macro‑dependent transition probabilities
- Viterbi algorithm for optimal path inference
- Score = log‑likelihood of most probable path
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-viterbi-macro-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High path likelihood → the ETF's regime dynamics are well‑explained by the macro‑dependent HMM.
- Low likelihood → poor fit.

## Requirements

See `requirements.txt`.
