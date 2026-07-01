"""
compare_models.py — Comparaison scientifique complète YOLOv5 vs YOLOv8.

Ce script est le point central d'analyse du projet. Il génère :
- Tableau comparatif complet (toutes métriques × tous runs)
- Graphiques comparatifs par métrique
- Graphique radar (vue d'ensemble)
- Analyse de l'impact des augmentations
- Analyse statistique (gain relatif, classement)
- Rapport textuel d'analyse critique

Usage :
    python src/compare_models.py                   # Analyse complète
    python src/compare_models.py --report          # + Génère le rapport texte
    python src/compare_models.py --mock            # Avec données simulées (sans GPU)

Auteur : lepiso
Projet : Comparaison YOLOv5 vs YOLOv8 sur Pascal VOC
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    PROJECT_ROOT,
    RESULTS_DIR,
    FIGURES_DIR,
    AUGMENTATION_STRATEGIES,
    METRICS_KEYS,
    COLORS,
    setup_logger,
    ensure_dir,
    load_all_metrics,
    plot_metric_comparison,
    plot_radar_chart,
    print_summary_table,
    compute_f1,
)


logger = setup_logger("compare_models")


# ============================================================
# DONNÉES SIMULÉES (pour tests locaux sans GPU)
# ============================================================

def generate_mock_data() -> pd.DataFrame:
    np.random.seed(42)

    # Valeurs de référence (basées sur la littérature)
    # YOLOv8 légèrement supérieur à YOLOv5 (hypothèse H1)
    base_metrics = {
        "yolov5s": {
            "no_augmentation": {"map50": 0.621, "map50_95": 0.412, "precision": 0.678, "recall": 0.589},
            "flip_horizontal": {"map50": 0.649, "map50_95": 0.431, "precision": 0.701, "recall": 0.612},
            "rotation"       : {"map50": 0.638, "map50_95": 0.422, "precision": 0.689, "recall": 0.601},
            "mosaic"         : {"map50": 0.672, "map50_95": 0.451, "precision": 0.718, "recall": 0.634},
            "mixup"          : {"map50": 0.658, "map50_95": 0.439, "precision": 0.706, "recall": 0.619},
        },
        "yolov8s": {
            "no_augmentation": {"map50": 0.643, "map50_95": 0.431, "precision": 0.694, "recall": 0.608},
            "flip_horizontal": {"map50": 0.668, "map50_95": 0.449, "precision": 0.719, "recall": 0.631},
            "rotation"       : {"map50": 0.659, "map50_95": 0.441, "precision": 0.708, "recall": 0.621},
            "mosaic"         : {"map50": 0.681, "map50_95": 0.459, "precision": 0.729, "recall": 0.645},
            "mixup"          : {"map50": 0.674, "map50_95": 0.453, "precision": 0.722, "recall": 0.638},
        },
    }

    records = []
    for model, aug_dict in base_metrics.items():
        for aug, m in aug_dict.items():
            # Ajouter un léger bruit pour simuler la variabilité
            noise = np.random.normal(0, 0.005, 4)
            p = max(0, min(1, m["precision"] + noise[0]))
            r = max(0, min(1, m["recall"]    + noise[1]))

            record = {
                "model"         : model,
                "augmentation"  : aug,
                "map50"         : round(max(0, m["map50"]    + noise[2]), 4),
                "map50_95"      : round(max(0, m["map50_95"] + noise[3]), 4),
                "precision"     : round(p, 4),
                "recall"        : round(r, 4),
                "f1"            : compute_f1(p, r),
                "train_time_min": round(np.random.uniform(45, 95), 1),
                "model_size_mb" : 14.1 if model == "yolov5s" else 22.5,
                "epochs_done"   : 50,
            }
            records.append(record)

    return pd.DataFrame(records)


# ============================================================
# ANALYSE STATISTIQUE
# ============================================================

def compute_relative_gain(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["map50", "map50_95", "precision", "recall", "f1"]

    records = []
    for aug in AUGMENTATION_STRATEGIES:
        v5 = df[(df["model"] == "yolov5s") & (df["augmentation"] == aug)]
        v8 = df[(df["model"] == "yolov8s") & (df["augmentation"] == aug)]

        if v5.empty or v8.empty:
            continue

        v5 = v5.iloc[0]
        v8 = v8.iloc[0]

        for metric in metrics:
            if v5[metric] > 0:
                gain = (v8[metric] - v5[metric]) / v5[metric] * 100
            else:
                gain = 0.0

            records.append({
                "augmentation": aug,
                "metric"      : metric,
                "yolov5"      : round(v5[metric], 4),
                "yolov8"      : round(v8[metric], 4),
                "gain_pct"    : round(gain, 2),
            })

    return pd.DataFrame(records)


def rank_augmentations(df: pd.DataFrame, metric: str = "map50_95") -> pd.DataFrame:
    result = df.groupby(["model", "augmentation"])[metric].mean().reset_index()
    result = result.sort_values([metric], ascending=False)
    result["rank"] = result.groupby("model")[metric].rank(ascending=False).astype(int)
    return result


# ============================================================
# GRAPHIQUES SPÉCIALISÉS
# ============================================================

def plot_augmentation_impact(df: pd.DataFrame, save_dir: Path = None) -> None:
    if save_dir is None:
        save_dir = FIGURES_DIR

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Impact des techniques d'augmentation sur les performances\n"
        "YOLOv5s vs YOLOv8s — Pascal VOC",
        fontsize=14,
        fontweight="bold",
    )

    metrics_to_plot = [
        ("map50",    "mAP@0.5",    axes[0]),
        ("map50_95", "mAP@0.5:0.95", axes[1]),
    ]

    aug_labels = {
        "no_augmentation": "Baseline",
        "flip_horizontal": "Flip H.",
        "rotation"       : "Rotation",
        "mosaic"         : "Mosaic",
        "mixup"          : "MixUp",
    }

    x = np.arange(len(AUGMENTATION_STRATEGIES))

    for metric, label, ax in metrics_to_plot:
        for model in ["yolov5s", "yolov8s"]:
            subset = df[df["model"] == model].set_index("augmentation")
            y = [subset.loc[aug, metric] if aug in subset.index else np.nan
                for aug in AUGMENTATION_STRATEGIES]

            ax.plot(x, y,
                    marker="o",
                    linewidth=2.5,
                    markersize=8,
                    label=model,
                    color=COLORS.get(model, "gray"))

            # Annotations des valeurs
            for xi, yi in zip(x, y):
                if not np.isnan(yi):
                    ax.annotate(f"{yi:.3f}",
                                (xi, yi),
                                textcoords="offset points",
                                xytext=(0, 10),
                                ha="center",
                                fontsize=8)

        ax.set_xlabel("Stratégie d'augmentation")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels([aug_labels[a] for a in AUGMENTATION_STRATEGIES], rotation=15)
        ax.legend()
        ax.set_ylim(df[metric].min() * 0.95, df[metric].max() * 1.08)

    plt.tight_layout()
    save_path = save_dir / "augmentation_impact.png"
    ensure_dir(save_dir)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Figure sauvegardée : {save_path}")
    plt.close(fig)


def plot_heatmap_metrics(df: pd.DataFrame, save_dir: Path = None) -> None:
    if save_dir is None:
        save_dir = FIGURES_DIR

    metrics = ["map50", "map50_95", "precision", "recall", "f1"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Heatmap des métriques — YOLOv5s vs YOLOv8s", fontsize=14, fontweight="bold")

    for ax, model in zip(axes, ["yolov5s", "yolov8s"]):
        subset = df[df["model"] == model].set_index("augmentation")[metrics]
        subset = subset.reindex(AUGMENTATION_STRATEGIES)

        sns.heatmap(
            subset,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            ax=ax,
            vmin=0.5,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"shrink": 0.8},
        )

        ax.set_title(f"{model.upper()}", fontsize=13)
        ax.set_xlabel("Métrique")
        ax.set_ylabel("Augmentation")
        ax.set_xticklabels(["mAP@0.5", "mAP@0.5:0.95", "Précision", "Rappel", "F1"],
                        rotation=30, ha="right")

    plt.tight_layout()
    save_path = save_dir / "heatmap_metrics.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Figure sauvegardée : {save_path}")
    plt.close(fig)


def plot_training_time_comparison(df: pd.DataFrame, save_dir: Path = None) -> None:
    if save_dir is None:
        save_dir = FIGURES_DIR

    fig, ax = plt.subplots(figsize=(10, 6))

    pivot = df.pivot(index="augmentation", columns="model", values="train_time_min")
    pivot = pivot.reindex(AUGMENTATION_STRATEGIES)

    x = np.arange(len(pivot.index))
    width = 0.35

    for i, model in enumerate(pivot.columns):
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            pivot[model],
            width=width,
            label=model,
            color=COLORS.get(model, "gray"),
            edgecolor="white",
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.0f}m", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Stratégie d'augmentation")
    ax.set_ylabel("Temps d'entraînement (minutes)")
    ax.set_title("Comparaison des temps d'entraînement — 50 époques, GPU T4")
    ax.set_xticks(x)
    ax.set_xticklabels(AUGMENTATION_STRATEGIES, rotation=15)
    ax.legend(title="Modèle")

    plt.tight_layout()
    save_path = save_dir / "training_time_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Figure sauvegardée : {save_path}")
    plt.close(fig)


# ============================================================
# RAPPORT TEXTUEL D'ANALYSE CRITIQUE
# ============================================================

def generate_analysis_report(df: pd.DataFrame, output_path: Path = None) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("RAPPORT D'ANALYSE CRITIQUE — YOLOv5 vs YOLOv8 sur Pascal VOC")
    lines.append("=" * 70)

    # --- Métriques globales par modèle ---
    lines.append("\n1. PERFORMANCES GLOBALES (moyenne sur toutes les augmentations)\n")
    global_avg = df.groupby("model")[["map50", "map50_95", "precision", "recall", "f1"]].mean()
    lines.append(global_avg.to_string())

    # --- Validation H1 ---
    lines.append("\n\n2. VALIDATION DE L'HYPOTHÈSE H1")
    lines.append("H1 : YOLOv8 obtient un mAP@0.5:0.95 supérieur à YOLOv5.\n")

    v5_avg = df[df["model"] == "yolov5s"]["map50_95"].mean()
    v8_avg = df[df["model"] == "yolov8s"]["map50_95"].mean()
    gain   = (v8_avg - v5_avg) / v5_avg * 100

    if v8_avg > v5_avg:
        lines.append(
            f"H1 VALIDÉE : YOLOv8s ({v8_avg:.4f}) > YOLOv5s ({v5_avg:.4f})\n"
            f"Gain relatif : +{gain:.2f}% en mAP@0.5:0.95"
        )
    else:
        lines.append(
            f"H1 INFIRMÉE : YOLOv5s ({v5_avg:.4f}) >= YOLOv8s ({v8_avg:.4f})\n"
            f"Différence : {gain:.2f}%"
        )

    # --- Meilleure augmentation par modèle ---
    lines.append("\n\n3. IMPACT DES AUGMENTATIONS\n")
    lines.append("Meilleure augmentation par modèle (selon mAP@0.5:0.95) :\n")

    for model in ["yolov5s", "yolov8s"]:
        subset = df[df["model"] == model]
        best   = subset.loc[subset["map50_95"].idxmax()]
        worst  = subset.loc[subset["map50_95"].idxmin()]

        lines.append(f"  {model} :")
        lines.append(f"    Meilleure augmentation : {best['augmentation']} "
                    f"(mAP@0.5:0.95 = {best['map50_95']:.4f})")
        lines.append(f"    Moins bonne augmentation : {worst['augmentation']} "
                    f"(mAP@0.5:0.95 = {worst['map50_95']:.4f})")
        lines.append(f"    Gain vs baseline : "
                    f"+{(best['map50_95'] - subset[subset['augmentation']=='no_augmentation']['map50_95'].values[0]):.4f}")

    # --- Gain relatif détaillé ---
    lines.append("\n\n4. GAIN RELATIF YOLOv8 vs YOLOv5 PAR AUGMENTATION\n")
    gain_df = compute_relative_gain(df)
    for aug in AUGMENTATION_STRATEGIES:
        subset = gain_df[(gain_df["augmentation"] == aug) & (gain_df["metric"] == "map50_95")]
        if not subset.empty:
            row = subset.iloc[0]
            sign = "+" if row["gain_pct"] >= 0 else ""
            lines.append(f"  {aug:20s} : {sign}{row['gain_pct']:.2f}% "
                        f"(YOLOv5={row['yolov5']:.4f} → YOLOv8={row['yolov8']:.4f})")

    # --- Recommandation finale ---
    lines.append("\n\n5. RECOMMANDATION\n")

    best_combo = df.loc[df["map50_95"].idxmax()]
    lines.append(
        f"La combinaison optimale observée est :\n"
        f"  Modèle       : {best_combo['model']}\n"
        f"  Augmentation : {best_combo['augmentation']}\n"
        f"  mAP@0.5:0.95 : {best_combo['map50_95']:.4f}\n"
        f"  mAP@0.5      : {best_combo['map50']:.4f}\n"
        f"  F1-score     : {best_combo['f1']:.4f}"
    )

    lines.append("\n" + "=" * 70)

    report = "\n".join(lines)

    if output_path:
        ensure_dir(output_path.parent)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Rapport sauvegardé : {output_path}")

    return report


# ============================================================
# PIPELINE COMPLET D'ANALYSE
# ============================================================

def run_full_comparison(df: pd.DataFrame, mock: bool = False) -> None:
    if mock:
        logger.warning(
            "DONNÉES SIMULÉES — Ces figures sont générées pour tester le pipeline.\n"
            "Remplacez les données par vos vrais résultats d'entraînement."
        )

    ensure_dir(FIGURES_DIR)

    logger.info("Génération des figures comparatives...")

    # 1. Graphique d'impact des augmentations (figure centrale)
    plot_augmentation_impact(df, FIGURES_DIR)

    # 2. Graphiques en barres par métrique
    for metric in ["map50", "map50_95", "precision", "recall", "f1"]:
        fig = plot_metric_comparison(
            df,
            metric=metric,
            save_path=FIGURES_DIR / f"comparison_{metric}.png",
        )
        plt.close(fig)
        logger.info(f"Figure générée : comparison_{metric}.png")

    # 3. Graphique radar
    fig = plot_radar_chart(df, save_path=FIGURES_DIR / "radar_comparison.png")
    plt.close(fig)
    logger.info("Figure générée : radar_comparison.png")

    # 4. Heatmap
    plot_heatmap_metrics(df, FIGURES_DIR)

    # 5. Temps d'entraînement
    plot_training_time_comparison(df, FIGURES_DIR)

    # 6. Tableau récapitulatif
    print_summary_table(df)

    # 7. Analyse des gains relatifs
    gain_df = compute_relative_gain(df)
    logger.info("\nGain relatif YOLOv8 vs YOLOv5 (mAP@0.5:0.95) :")
    for _, row in gain_df[gain_df["metric"] == "map50_95"].iterrows():
        sign = "+" if row["gain_pct"] >= 0 else ""
        logger.info(f"  {row['augmentation']:20s} : {sign}{row['gain_pct']:.2f}%")

    # 8. Rapport textuel
    report = generate_analysis_report(
        df,
        output_path=PROJECT_ROOT / "reports" / "analysis_report.txt",
    )
    print(report)

    logger.info(f"\nAnalyse complète terminée. Figures dans : {FIGURES_DIR}")


# ============================================================
# POINT D'ENTRÉE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comparaison scientifique YOLOv5 vs YOLOv8."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Utiliser des données simulées (pour tester le pipeline sans GPU).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Générer uniquement le rapport d'analyse textuel.",
    )

    args = parser.parse_args()

    if args.mock:
        logger.info("Génération des données simulées...")
        df = generate_mock_data()
        logger.info(f"DataFrame créé : {len(df)} runs simulés")
    else:
        df = load_all_metrics()
        if df.empty:
            logger.warning(
                "Aucune métrique trouvée dans results/.\n"
                "Lancez d'abord les entraînements ou utilisez --mock pour les données simulées."
            )
            logger.info("Astuce : python src/compare_models.py --mock")
            return

    if args.report:
        report = generate_analysis_report(
            df,
            output_path=PROJECT_ROOT / "reports" / "analysis_report.txt",
        )
        print(report)
    else:
        run_full_comparison(df, mock=args.mock)


if __name__ == "__main__":
    main()