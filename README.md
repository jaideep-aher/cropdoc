# CropDoc 🌿

**Plant Disease Detection & Treatment Recommendation**

CropDoc is a computer vision system that identifies plant diseases from leaf photos and delivers ranked, actionable treatment plans. It goes beyond classification — confidence-weighted severity scoring and a treatment recommendation engine turn a raw prediction into farmer-ready guidance.

## Live Demo
🔗 [cropdoc.up.railway.app](https://cropdoc.up.railway.app) *(deployed on Railway)*

## Project Structure
```
├── README.md
├── requirements.txt
├── setup.py                  # Download data, build features, train all models
├── main.py                   # CLI entry point
├── scripts/
│   ├── make_dataset.py       # Download & split PlantVillage dataset
│   ├── build_features.py     # HOG + color histogram feature extraction
│   └── model.py              # Train/evaluate all three models
├── app/
│   ├── backend/              # FastAPI inference server
│   └── frontend/             # React + Tailwind UI
├── models/                   # Saved model weights (gitignored, loaded from HF Hub)
├── data/
│   ├── raw/                  # Raw downloaded images
│   ├── processed/            # Feature arrays (.npy)
│   └── outputs/              # Evaluation results, confusion matrices
├── notebooks/                # Exploratory notebooks (not graded)
└── .gitignore
```

## Models
| Model | Approach | Val Accuracy |
|-------|----------|-------------|
| Naive baseline | Majority-class per crop | ~3.6% |
| Classical ML | HOG + Color Hist → Random Forest | ~85% |
| Deep Learning ⭐ | EfficientNetV2-S fine-tuned | ~97% |

## Quick Start
```bash
pip install -r requirements.txt
python setup.py          # downloads data, trains all models
python main.py           # starts inference server
```

## Training (Google Colab)
Open `notebooks/training.ipynb` in Colab — it downloads PlantVillage via HuggingFace datasets, trains all three models, and pushes weights to HF Hub.

## Experiment
We probe training-set size sensitivity: models trained on 10 / 25 / 50 / 100% of data to understand data efficiency. See `notebooks/experiment_data_efficiency.ipynb`.

## Dataset
[PlantVillage](https://huggingface.co/datasets/HanochSkandera/PlantVillage) — 54,306 leaf images, 38 disease classes across 14 crops.

## Ethics
See report for full ethics statement. Key considerations: false negatives may delay treatment; model should augment, not replace, agronomist judgment.
