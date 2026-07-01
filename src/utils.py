"""
utils.py — Fonctions utilitaires partagées pour le projet YOLOv5 vs YOLOv8.

Ce module centralise :
- La configuration des chemins et constantes du projet
- La création de dossiers de résultats
- Le chargement et la sauvegarde des métriques
- La génération de graphiques réutilisables
- Les fonctions de logging
"""

import os
import json
import logging
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# ============================================================
# CONSTANTES DU PROJET
# ============================================================

# Répertoire racine du projet (deux niveaux au-dessus de src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Sous-dossiers principaux
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
DATA_DIR    = PROJECT_ROOT / "data"
MODELS_DIR  = PROJECT_ROOT / "models"

# Chemin vers le fichier de configuration du dataset (Pascal VOC)
DATA_YAML = DATA_DIR / "data.yaml"

# Paramètres d'entraînement partagés (conditions identiques pour les deux modèles)
SHARED_CONFIG = {
    "img_size"  : 640,    # Résolution d'entrée standard YOLO
    "epochs"    : 50,     # Suffisant pour la convergence sur Pascal VOC
    "batch_size": 16,     # Compatible GPU T4 (Colab)
    "workers"   : 4,      # Threads de chargement des données
}

# Stratégies d'augmentation testées
AUGMENTATION_STRATEGIES = [
    "no_augmentation",
    "flip_horizontal",
    "rotation",
    "mosaic",
    "mixup",
]

# Métriques collectées pour chaque run
METRICS_KEYS = [
    "model",
    "augmentation",
    "map50",           # mAP@0.5
    "map50_95",        # mAP@0.5:0.95
    "precision",
    "recall",
    "f1",
    "train_time_min",  # Temps d'entraînement en minutes
    "model_size_mb",   # Taille du fichier de poids (.pt)
    "epochs_done",
]

# Palette de couleurs cohérente dans tous les graphiques
# NOTE : les clés modèles incluent le suffixe "s" (yolov5s, yolov8s)
# pour matcher exactement les valeurs utilisées dans les DataFrames de métriques.
COLORS = {
    "yolov5"          : "#1f77b4",   # Bleu (alias sans suffixe)
    "yolov8"          : "#ff7f0e",   # Orange (alias sans suffixe)
    "yolov5s"         : "#1f77b4",   # Bleu
    "yolov8s"         : "#ff7f0e",   # Orange
    "no_augmentation" : "#7f7f7f",   # Gris
    "flip_horizontal" : "#2ca02c",   # Vert
    "rotation"        : "#d62728",   # Rouge
    "mosaic"          : "#9467bd",   # Violet
    "mixup"           : "#8c564b",   # Marron
}

# Style global Matplotlib pour tous les graphiques du projet
plt.rcParams.update({
    "figure.dpi"      : 150,
    "figure.facecolor": "white",
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
    "font.size"       : 11,
    "axes.titlesize"  : 13,
    "axes.labelsize"  : 11,
    "legend.fontsize" : 10,
})


# ============================================================
# CONFIGURATION DU LOGGING
# ============================================================

def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Format des messages
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler fichier (optionnel)
    if log_file:
        ensure_dir(Path(log_file).parent)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================
# GESTION DES DOSSIERS
# ============================================================

def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_run_dir(model_name: str, augmentation: str) -> Path:
    run_name = f"{model_name}_{augmentation}"
    run_dir = RESULTS_DIR / run_name
    return ensure_dir(run_dir)


# ============================================================
# SAUVEGARDE ET CHARGEMENT DES MÉTRIQUES
# ============================================================

def save_metrics(metrics: dict, model_name: str, augmentation: str) -> Path:
    run_dir = get_run_dir(model_name, augmentation)
    metrics_file = run_dir / "metrics.json"

    # Ajouter les métadonnées du run
    metrics["model"]        = model_name
    metrics["augmentation"] = augmentation
    metrics["timestamp"]    = datetime.now().isoformat()

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics_file


def load_all_metrics() -> pd.DataFrame:
    records = []

    if not RESULTS_DIR.exists():
        return pd.DataFrame(columns=METRICS_KEYS)

    for run_dir in sorted(RESULTS_DIR.iterdir()):
        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                records.append(json.load(f))

    if not records:
        return pd.DataFrame(columns=METRICS_KEYS)

    df = pd.DataFrame(records)
    return df


def append_to_csv(metrics: dict, csv_path: Path = None) -> None:
    if csv_path is None:
        csv_path = RESULTS_DIR / "all_results.csv"

    ensure_dir(csv_path.parent)
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_KEYS)
        if not file_exists:
            writer.writeheader()
        # Ne garder que les clés définies dans METRICS_KEYS
        row = {k: metrics.get(k, "") for k in METRICS_KEYS}
        writer.writerow(row)


# ============================================================
# FONCTIONS DE VISUALISATION
# ============================================================

def plot_metric_comparison(
    df: pd.DataFrame,
    metric: str,
    title: str = None,
    save_path: Path = None,
) -> plt.Figure:
    
    if df.empty:
        raise ValueError("Le DataFrame de métriques est vide.")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Pivoter pour avoir augmentation en X, modèle en couleur
    pivot = df.pivot(index="augmentation", columns="model", values=metric)
    pivot = pivot.reindex(AUGMENTATION_STRATEGIES)  # Ordre cohérent

    x = np.arange(len(pivot.index))
    width = 0.35
    models = pivot.columns.tolist()

    for i, model in enumerate(models):
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            pivot[model],
            width=width,
            label=model,
            color=COLORS.get(model, f"C{i}"),
            edgecolor="white",
            linewidth=0.5,
        )
        # Afficher la valeur au-dessus de chaque barre
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.005,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xlabel("Stratégie d'augmentation")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"Comparaison {metric.upper()} — YOLOv5 vs YOLOv8")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=15, ha="right")
    ax.legend(title="Modèle")
    ax.set_ylim(0, min(1.05, df[metric].max() * 1.15))

    plt.tight_layout()

    if save_path:
        ensure_dir(save_path.parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_curves(
    csv_path: Path,
    model_name: str,
    augmentation: str,
    save_path: Path = None,
) -> plt.Figure:
    
    df = pd.read_csv(csv_path)
    # Normaliser les noms de colonnes (supprimer les espaces)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"Courbes d'apprentissage — {model_name} | {augmentation}",
        fontsize=14,
        fontweight="bold",
    )

    color = COLORS.get(model_name.lower(), "steelblue")

    # Mapping des colonnes YOLOv5/YOLOv8 vers des noms standardisés
    # YOLOv5 : train/box_loss, metrics/mAP_0.5, etc.
    # YOLOv8 : train/box_loss, metrics/mAP50, etc.
    possible_cols = {
        "box_loss" : ["train/box_loss", "train/box_om"],
        "cls_loss" : ["train/cls_loss", "train/cls_om"],
        "map50"    : ["metrics/mAP_0.5", "metrics/mAP50"],
        "map50_95" : ["metrics/mAP_0.5:0.95", "metrics/mAP50-95"],
    }

    def find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    subplot_config = [
        (axes[0, 0], "box_loss", "Box Loss (train)"),
        (axes[0, 1], "cls_loss", "Class Loss (train)"),
        (axes[1, 0], "map50",    "mAP@0.5"),
        (axes[1, 1], "map50_95", "mAP@0.5:0.95"),
    ]

    for ax, key, label in subplot_config:
        col = find_col(possible_cols[key])
        if col:
            ax.plot(df.index + 1, df[col], color=color, linewidth=2)
            ax.set_xlabel("Époque")
            ax.set_ylabel(label)
            ax.set_title(label)
        else:
            ax.text(0.5, 0.5, f"Colonne '{key}' non trouvée",
                    ha="center", va="center", transform=ax.transAxes,
                    color="gray")
            ax.set_title(label)

    plt.tight_layout()

    if save_path:
        ensure_dir(save_path.parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_radar_chart(
    df: pd.DataFrame,
    save_path: Path = None,
) -> plt.Figure:
    
    metrics_to_plot = ["map50", "map50_95", "precision", "recall", "f1"]
    labels = ["mAP@0.5", "mAP@0.5:0.95", "Précision", "Rappel", "F1"]

    # Moyenne par modèle sur toutes les augmentations
    grouped = df.groupby("model")[metrics_to_plot].mean()

    num_vars = len(metrics_to_plot)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Fermer le polygone

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for model, row in grouped.iterrows():
        values = row[metrics_to_plot].tolist()
        values += values[:1]
        color = COLORS.get(model, "gray")
        ax.plot(angles, values, color=color, linewidth=2, label=model)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11)
    ax.set_ylim(0, 1)
    ax.set_title("Comparaison globale — YOLOv5 vs YOLOv8\n(moyenne sur toutes les augmentations)",
                pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()

    if save_path:
        ensure_dir(save_path.parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ============================================================
# FONCTIONS UTILITAIRES DIVERSES
# ============================================================

def get_model_size_mb(weights_path: Path) -> float:

    path = Path(weights_path)
    if path.exists():
        return round(path.stat().st_size / (1024 ** 2), 2)
    return 0.0


def compute_f1(precision: float, recall: float) -> float:

    if precision + recall == 0:
        return 0.0
    return round(2 * (precision * recall) / (precision + recall), 4)


def format_duration(seconds: float) -> str:
    
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def print_summary_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("Aucun résultat disponible.")
        return

    cols = ["model", "augmentation", "map50", "map50_95",
            "precision", "recall", "f1", "train_time_min"]
    available = [c for c in cols if c in df.columns]

    print("\n" + "=" * 80)
    print("TABLEAU RÉCAPITULATIF DES RÉSULTATS")
    print("=" * 80)
    print(df[available].to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Test rapide des fonctions utilitaires
    logger = setup_logger("utils_test")
    logger.info("Test de utils.py")

    # Vérification des chemins
    logger.info(f"PROJECT_ROOT : {PROJECT_ROOT}")
    logger.info(f"RESULTS_DIR  : {RESULTS_DIR}")
    logger.info(f"FIGURES_DIR  : {FIGURES_DIR}")

    # Test compute_f1
    f1 = compute_f1(0.85, 0.78)
    logger.info(f"F1 test (P=0.85, R=0.78) : {f1}")

    # Test format_duration
    duration = format_duration(3723)
    logger.info(f"Durée test (3723s) : {duration}")

    logger.info("utils.py — OK")