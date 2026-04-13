# NIDS Adversarial Robustness Dashboard

Streamlit dashboard for the paper **"Adversarial Robustness of Deep Learning-Based Network Intrusion Detection Systems: An Empirical Study on UNSW-NB15"**.

## Pages

| Page | Content |
|------|---------|
| Overview | Key metrics, pipeline summary, class distribution, clean F1 comparison |
| Attack Analysis | Macro F1 degradation curves, interactive per-class explorer (FGSM / PGD) |
| Robustness Comparison | Standard vs Robust model at any ε, per-class delta table |
| SHAP Explainability | Global importance, beeswarm plots, rank change chart + interactive rank table |
| Representation Space | t-SNE of penultimate-layer activations for both models |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select your repo → set **Main file path** to `app.py`
4. Deploy

## Repo structure

```
├── app.py
├── requirements.txt
├── README.md
└── assets/
    ├── phase2_macro_degradation.png
    ├── phase2_degradation_curves.png
    ├── phase4_global_importance.png
    ├── phase4_beeswarm.png
    ├── phase4_rank_change.png
    └── phase4_tsne.png
```
