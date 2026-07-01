"""
evaluate.py — Évaluation des modèles entraînés sur le jeu de test Pascal VOC.

Ce script effectue l'évaluation complète d'un ou plusieurs modèles :
- Calcul de mAP@0.5 et mAP@0.5:0.95 sur le jeu de test
- Calcul de précision, rappel et F1-score par classe
- Génération des matrices de confusion
- Génération des courbes Précision-Rappel
- Affichage des cas d'échec (faux positifs / faux négatifs)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    PROJECT_ROOT,
    RESULTS_DIR,
    FIGURES_DIR,
    DATA_YAML,
    AUGMENTATION_STRATEGIES,
    COLORS,
    setup_logger,
    ensure_dir,
    get_run_dir,
    load_all_metrics,
    compute_f1,
    print_summary_table,
)

# Classes Pascal VOC (20 classes)
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird",   "boat",       "bottle",
    "bus",       "car",     "cat",    "chair",       "cow",
    "diningtable","dog",    "horse",  "motorbike",   "person",
    "pottedplant","sheep",  "sofa",   "train",       "tvmonitor",
]


# ============================================================
# ÉVALUATION YOLOV5
# ============================================================

def evaluate_yolov5(model_name: str, augmentation: str) -> dict:
    import subprocess

    logger = setup_logger(f"eval_{model_name}_{augmentation}")

    yolov5_dir  = PROJECT_ROOT / "yolov5"
    val_script  = yolov5_dir / "val.py"
    run_dir     = RESULTS_DIR / f"{model_name}_{augmentation}"
    best_weights = run_dir / "weights" / "best.pt"

    if not best_weights.exists():
        raise FileNotFoundError(
            f"Poids introuvables : {best_weights}\n"
            "L'entraînement a-t-il été effectué ?"
        )

    data_yaml = PROJECT_ROOT / "data" / "data.yaml"

    cmd = [
        "python", str(val_script),
        "--weights", str(best_weights),
        "--data",    str(data_yaml),
        "--img",     "640",
        "--task",    "test",          # Évaluer sur le jeu de TEST
        "--name",    f"eval_{model_name}_{augmentation}",
        "--project", str(RESULTS_DIR),
        "--exist-ok",
        "--save-json",               # Sauvegarder les prédictions au format COCO
        "--plots",                   # Générer matrices de confusion et courbes PR
    ]

    logger.info(f"Commande d'évaluation : {' '.join(cmd)}")

    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Évaluation YOLOv5 échouée : {e}")

    # Lire les métriques depuis le JSON généré
    eval_dir  = RESULTS_DIR / f"eval_{model_name}_{augmentation}"
    json_file = eval_dir / "predictions.json"

    logger.info(f"Évaluation terminée. Résultats dans : {eval_dir}")
    return {"eval_dir": str(eval_dir), "model": model_name, "augmentation": augmentation}


# ============================================================
# ÉVALUATION YOLOV8
# ============================================================

def evaluate_yolov8(model_name: str, augmentation: str) -> dict:
    from ultralytics import YOLO

    logger = setup_logger(f"eval_{model_name}_{augmentation}")

    run_dir      = RESULTS_DIR / f"{model_name}_{augmentation}"
    best_weights = run_dir / "weights" / "best.pt"
    data_yaml    = PROJECT_ROOT / "data" / "data.yaml"

    if not best_weights.exists():
        raise FileNotFoundError(f"Poids introuvables : {best_weights}")

    model = YOLO(str(best_weights))

    logger.info(f"Évaluation de {best_weights} sur le jeu de test...")

    results = model.val(
        data=str(data_yaml),
        imgsz=640,
        batch=16,
        split="test",               # Évaluer sur le jeu de test
        name=f"eval_{model_name}_{augmentation}",
        project=str(RESULTS_DIR),
        exist_ok=True,
        plots=True,                 # Matrices de confusion + courbes PR
        save_json=True,
    )

    # Extraire les métriques depuis l'objet Results
    metrics = {
        "model"       : model_name,
        "augmentation": augmentation,
        "map50"       : round(results.box.map50, 4),
        "map50_95"    : round(results.box.map, 4),
        "precision"   : round(results.box.mp, 4),
        "recall"      : round(results.box.mr, 4),
        "f1"          : compute_f1(results.box.mp, results.box.mr),
    }

    logger.info(f"mAP@0.5 : {metrics['map50']}")
    logger.info(f"mAP@0.5:0.95 : {metrics['map50_95']}")

    return metrics


# ============================================================
# GÉNÉRATION DE LA MATRICE DE CONFUSION
# ============================================================

def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: list,
    model_name: str,
    augmentation: str,
    save_path: Path = None,
) -> plt.Figure:
    # Normalisation par ligne (valeurs entre 0 et 1)
    cm_normalized = confusion_matrix.astype(float)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Éviter division par zéro
    cm_normalized /= row_sums

    fig, ax = plt.subplots(figsize=(16, 14))

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="lightgray",
    )

    ax.set_xlabel("Classe prédite", fontsize=12)
    ax.set_ylabel("Classe réelle", fontsize=12)
    ax.set_title(
        f"Matrice de confusion — {model_name} | {augmentation}\n"
        "(normalisée par ligne — valeurs = rappel par classe)",
        fontsize=13,
        pad=15,
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        ensure_dir(save_path.parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ============================================================
# COURBES PRÉCISION-RAPPEL
# ============================================================

def plot_precision_recall_curve(
    precision_values: np.ndarray,
    recall_values   : np.ndarray,
    model_name      : str,
    augmentation    : str,
    save_path       : Path = None,
) -> plt.Figure:
    # Calcul de l'aire sous la courbe (approximation trapézoïdale)
    auc = np.trapz(precision_values, recall_values)

    color = COLORS.get(model_name.lower(), "steelblue")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_values, precision_values,
            color=color, linewidth=2.5, label=f"{model_name} (AUC={auc:.3f})")
    ax.fill_between(recall_values, precision_values, alpha=0.1, color=color)

    ax.set_xlabel("Rappel (Recall)")
    ax.set_ylabel("Précision (Precision)")
    ax.set_title(f"Courbe Précision-Rappel — {model_name} | {augmentation}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()

    if save_path:
        ensure_dir(save_path.parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ============================================================
# ANALYSE DES CAS D'ÉCHEC
# ============================================================

def analyze_failure_cases(
    model_name  : str,
    augmentation: str,
    n_samples   : int = 10,
) -> None:
    logger = setup_logger(f"failure_{model_name}_{augmentation}")

    eval_dir    = RESULTS_DIR / f"eval_{model_name}_{augmentation}"
    json_file   = eval_dir / "predictions.json"

    if not json_file.exists():
        logger.warning(
            f"Fichier de prédictions introuvable : {json_file}\n"
            "Lancez d'abord evaluate.py pour générer les prédictions."
        )
        return

    with open(json_file, "r") as f:
        predictions = json.load(f)

    logger.info(f"Analyse de {len(predictions)} prédictions...")

    # Regrouper par image_id pour identifier les images problématiques
    from collections import defaultdict
    by_image = defaultdict(list)
    for pred in predictions:
        by_image[pred["image_id"]].append(pred)

    # Identifier les images avec le plus de détections (potentiels FP)
    high_detection_images = sorted(
        by_image.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:n_samples]

    logger.info(f"\nTop {n_samples} images avec le plus de détections (FP potentiels) :")
    for img_id, preds in high_detection_images:
        scores = [p["score"] for p in preds]
        logger.info(
            f"  Image {img_id}: {len(preds)} détections | "
            f"Score moyen: {np.mean(scores):.3f}"
        )

    logger.info("\nAnalyse des cas d'échec terminée.")
    logger.info(
        "→ Pour l'analyse qualitative complète, consultez les images de validation "
        "générées dans le dossier du run."
    )


# ============================================================
# ÉVALUATION GLOBALE DE TOUS LES RUNS
# ============================================================

def evaluate_all_runs() -> pd.DataFrame:
    df = load_all_metrics()

    if df.empty:
        print("Aucun résultat trouvé dans results/. Lancez d'abord les entraînements.")
        return df

    print_summary_table(df)

    # Meilleur run par modèle
    print("\n--- Meilleur run par modèle (selon mAP@0.5:0.95) ---")
    for model in df["model"].unique():
        subset = df[df["model"] == model]
        best   = subset.loc[subset["map50_95"].idxmax()]
        print(f"  {model} : augmentation={best['augmentation']} | mAP@0.5:0.95={best['map50_95']}")

    return df


# ============================================================
# POINT D'ENTRÉE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Évaluation des modèles YOLOv5/YOLOv8 entraînés."
    )
    parser.add_argument("--model",       type=str, choices=["yolov5s", "yolov8s"])
    parser.add_argument("--augmentation",type=str, choices=AUGMENTATION_STRATEGIES)
    parser.add_argument("--all",         action="store_true",
                        help="Affiche les métriques de tous les runs disponibles.")
    parser.add_argument("--failure",     action="store_true",
                        help="Analyse les cas d'échec du run spécifié.")

    args = parser.parse_args()

    if args.all:
        evaluate_all_runs()

    elif args.model and args.augmentation:
        if args.failure:
            analyze_failure_cases(args.model, args.augmentation)
        elif args.model == "yolov5s":
            metrics = evaluate_yolov5(args.model, args.augmentation)
            print(json.dumps(metrics, indent=2))
        elif args.model == "yolov8s":
            metrics = evaluate_yolov8(args.model, args.augmentation)
            print(json.dumps(metrics, indent=2))

    else:
        parser.print_help()
        print("\nExemples :")
        print("  python src/evaluate.py --all")
        print("  python src/evaluate.py --model yolov8s --augmentation mosaic")
        print("  python src/evaluate.py --model yolov5s --augmentation mosaic --failure")


if __name__ == "__main__":
    main()