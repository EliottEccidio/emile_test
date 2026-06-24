# Fine-tuning LoRA de Qwen2.5-1.5B-Instruct sur GSM8K + MLflow

Fine-tuning **LoRA** (PEFT/TRL) du modele `Qwen/Qwen2.5-1.5B-Instruct` sur le
benchmark [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), avec
**evaluation pre / post** et **observabilite via MLflow**.

## Structure

```
emile_test/
├── config.yaml            # source unique des hyperparametres et chemins
├── requirements.txt
├── Makefile               # raccourcis : setup / eval-pre / train / eval-post / pipeline
├── src/
│   ├── config.py          # config typee (dataclasses) chargee depuis config.yaml
│   ├── data.py            # chargement GSM8K + format chat Qwen
│   ├── metrics.py         # extraction reponse finale + exact-match (coeur de l'eval)
│   ├── model.py           # modele/tokenizer, device (cuda/mps/cpu), dtype, LoRA
│   ├── evaluate.py        # generation + scoring + logging MLflow (pre/post)
│   ├── train.py           # fine-tuning LoRA (SFT) + logging MLflow
│   └── pipeline.py        # orchestration pre -> train -> post (1 run parent MLflow)
├── tests/
│   └── test_metrics.py    # tests unitaires des metriques
├── outputs/               # adaptateur LoRA + rapports d'eval (gitignore)
├── mlflow.db              # tracking MLflow local, backend SQLite (gitignore)
└── mlartifacts/           # artefacts MLflow : rapports d'eval logges (gitignore)
```

## Installation

> Plateforme cible : Apple Silicon (MPS) ou GPU CUDA. **Pas de quantization 4-bit**
> (`bitsandbytes` est incompatible avec macOS/MPS), le code tourne en fp32 sur Mac.

```bash
make setup            # cree .venv et installe requirements.txt
# ou manuellement :
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

### Option 1 — pipeline complet (recommande)
Eval baseline, entrainement, eval fine-tune, le tout dans un seul run MLflow :

```bash
python -m src.pipeline           # ou : make pipeline
```

### Option 2 — etape par etape
```bash
python -m src.evaluate --run-name eval-pre                                   # 1. baseline
python -m src.train                                                          # 2. fine-tuning
python -m src.evaluate --adapter outputs/qwen2.5-1.5b-gsm8k-lora --run-name eval-post  # 3. post
```

### Smoke test rapide (verifier que tout s'enchaine)
```bash
python -m src.evaluate --limit 10 --run-name smoke   # eval sur 10 exemples
python -m src.train --train-limit 50 --epochs 1      # entrainement minuscule
```

## Observabilite (MLflow)

Tracking en local dans une base SQLite (`mlflow.db`), artefacts dans `mlartifacts/`
(le file store `./mlruns` est en *maintenance mode* depuis MLflow 3.x). Pour visualiser :

```bash
make mlflow-ui        # http://localhost:5000
# ou : mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Ce qui est logge :
- **train** : hyperparametres (lr, batch, r/alpha LoRA...), courbe de loss, `final_train_loss`.
- **eval-pre / eval-post** : `accuracy`, `n_correct`, parametres de generation, et un
  rapport JSON (`outputs/eval_*.json`) avec quelques exemples de predictions.
- **pipeline** : run parent `pre-vs-post` avec `pre_accuracy`, `post_accuracy`, `accuracy_delta`.

## Tests

```bash
make test    # ou : pytest -q
```

## Configuration

Tout se regle dans [`config.yaml`](config.yaml) : modele, LoRA (`r`, `alpha`, modules cibles),
hyperparametres d'entrainement, taille des sous-ensembles (`train_limit`, `eval_limit`) et
parametres de generation. Aucun parametre code en dur.
