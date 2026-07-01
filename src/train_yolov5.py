"""
train_yolov5.py — Entraînement YOLOv5 sur Pascal VOC avec contrôle des augmentations.

Ce script gère :
- L'entraînement de YOLOv5s avec différentes stratégies d'augmentation
- Le logging des métriques à chaque run
- La sauvegarde des résultats dans la structure du projet

Prérequis :
    - yolov5/ cloné à la racine du projet
    - data/data.yaml disponible (généré par Roboflow)
    - Environnement virtuel activé avec dépendances installées
"""

import argparse
import os
import subprocess
import time
import yaml
from pathlib import Path

# Importer les utilitaires du projet
import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    PROJECT_ROOT,
    RESULTS_DIR,
    FIGURES_DIR,
    SHARED_CONFIG,
    AUGMENTATION_STRATEGIES,
    COLORS,
    setup_logger,
    ensure_dir,
    get_run_dir,
    save_metrics,
    append_to_csv,
    get_model_size_mb,
    compute_f1,
    format_duration,
    plot_training_curves,
)


# ============================================================
# CONFIGURATION YOLOV5
# ============================================================

MODEL_NAME   = "yolov5s"
YOLOV5_DIR   = PROJECT_ROOT / "yolov5"
TRAIN_SCRIPT = YOLOV5_DIR / "train.py"
DATA_YAML    = PROJECT_ROOT / "data" / "data.yaml"

# Fichiers d'hyperparamètres YOLOv5 (dans yolov5/data/hyps/)
# Ces fichiers contrôlent les augmentations natives de YOLOv5
HYPS_DIR = YOLOV5_DIR / "data" / "hyps"


# ============================================================
# GÉNÉRATION DES FICHIERS D'HYPERPARAMÈTRES
# ============================================================

def generate_hyp_file(augmentation: str) -> Path:
    logger = setup_logger("hyp_generator")

    # Fichier de base (hyperparamètres recommandés pour VOC)
    base_hyp_path = HYPS_DIR / "hyp.VOC.yaml"
    if not base_hyp_path.exists():
        # Fallback vers le fichier scratch si VOC n'est pas disponible
        base_hyp_path = HYPS_DIR / "hyp.scratch-low.yaml"

    if not base_hyp_path.exists():
        raise FileNotFoundError(
            f"Fichier d'hyperparamètres de base introuvable dans {HYPS_DIR}. "
            "Vérifiez que yolov5/ est bien cloné."
        )

    with open(base_hyp_path, "r") as f:
        hyp = yaml.safe_load(f)

    # --- Désactiver TOUTES les augmentations par défaut ---
    # On part d'une base propre, puis on active uniquement ce qui est demandé
    hyp["mosaic"]  = 0.0   # Désactivé
    hyp["mixup"]   = 0.0   # Désactivé
    hyp["fliplr"]  = 0.0   # Désactivé
    hyp["flipud"]  = 0.0   # Désactivé
    hyp["degrees"] = 0.0   # Désactivé
    hyp["translate"] = 0.0
    hyp["scale"]     = 0.0
    hyp["shear"]     = 0.0
    hyp["perspective"] = 0.0
    hyp["hsv_h"]     = 0.0
    hyp["hsv_s"]     = 0.0
    hyp["hsv_v"]     = 0.0

    # --- Activer uniquement la stratégie demandée ---
    if augmentation == "no_augmentation":
        # Tout reste à 0 — entraînement baseline pur
        logger.info("Stratégie : aucune augmentation (baseline)")

    elif augmentation == "flip_horizontal":
        hyp["fliplr"] = 0.5    # 50% de probabilité de flip horizontal
        logger.info("Stratégie : flip horizontal (fliplr=0.5)")

    elif augmentation == "rotation":
        hyp["degrees"] = 10.0  # Rotation aléatoire ±10°
        logger.info("Stratégie : rotation (degrees=10.0)")

    elif augmentation == "mosaic":
        hyp["mosaic"] = 1.0    # Mosaic activé à 100%
        logger.info("Stratégie : mosaic (mosaic=1.0)")

    elif augmentation == "mixup":
        hyp["mixup"] = 0.1     # MixUp activé à 10% (valeur standard)
        logger.info("Stratégie : mixup (mixup=0.1)")

    else:
        raise ValueError(
            f"Stratégie d'augmentation inconnue : '{augmentation}'. "
            f"Valeurs acceptées : {AUGMENTATION_STRATEGIES}"
        )

    # Sauvegarder dans le dossier du run
    run_dir = get_run_dir(MODEL_NAME, augmentation)
    hyp_output = run_dir / "hyp.yaml"
    with open(hyp_output, "w") as f:
        yaml.dump(hyp, f, default_flow_style=False)

    logger.info(f"Fichier hyp généré : {hyp_output}")
    return hyp_output


# ============================================================
# LECTURE DES MÉTRIQUES DEPUIS LE CSV YOLOV5
# ============================================================

def parse_yolov5_results(results_csv: Path) -> dict:
    import pandas as pd

    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv introuvable : {results_csv}")

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    # Dernière ligne = dernière époque
    last = df.iloc[-1]

    # Mapping robuste (compatible différentes versions de YOLOv5)
    def get_val(candidates):
        for c in candidates:
            if c in df.columns:
                return float(last[c])
        return 0.0

    precision = get_val(["metrics/precision", "   metrics/precision"])
    recall    = get_val(["metrics/recall",    "   metrics/recall"])
    map50     = get_val(["metrics/mAP_0.5",   "   metrics/mAP_0.5"])
    map50_95  = get_val(["metrics/mAP_0.5:0.95", "   metrics/mAP_0.5:0.95"])

    return {
        "precision"   : round(precision, 4),
        "recall"      : round(recall, 4),
        "map50"       : round(map50, 4),
        "map50_95"    : round(map50_95, 4),
        "f1"          : compute_f1(precision, recall),
        "epochs_done" : len(df),
    }


# ============================================================
# ENTRAÎNEMENT PRINCIPAL
# ============================================================

def train_yolov5(augmentation: str, dry_run: bool = False) -> dict:
    logger = setup_logger(
        f"train_yolov5_{augmentation}",
        log_file=str(get_run_dir(MODEL_NAME, augmentation) / "train.log"),
    )

    logger.info("=" * 60)
    logger.info(f"DÉMARRAGE — YOLOv5 | Augmentation : {augmentation}")
    logger.info("=" * 60)

    # --- 1. Validation des prérequis ---
    if not YOLOV5_DIR.exists():
        raise FileNotFoundError(
            f"Le dossier yolov5/ est introuvable à : {YOLOV5_DIR}\n"
            "Exécutez : git clone https://github.com/ultralytics/yolov5.git"
        )

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"train.py introuvable : {TRAIN_SCRIPT}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml introuvable : {DATA_YAML}\n"
            "Téléchargez Pascal VOC via Roboflow et placez data.yaml dans data/."
        )

    if augmentation not in AUGMENTATION_STRATEGIES:
        raise ValueError(
            f"Augmentation invalide : '{augmentation}'. "
            f"Valeurs acceptées : {AUGMENTATION_STRATEGIES}"
        )

    # --- 2. Génération des hyperparamètres ---
    hyp_file = generate_hyp_file(augmentation)
    run_dir  = get_run_dir(MODEL_NAME, augmentation)

    # --- 3. Construction de la commande d'entraînement ---
    cmd = [
        "python", str(TRAIN_SCRIPT),
        "--img",     str(SHARED_CONFIG["img_size"]),
        "--batch",   str(SHARED_CONFIG["batch_size"]),
        "--epochs",  str(SHARED_CONFIG["epochs"]),
        "--data",    str(DATA_YAML),
        "--weights", "yolov5s.pt",      # Téléchargé automatiquement si absent
        "--hyp",     str(hyp_file),
        "--name",    f"yolov5s_{augmentation}",
        "--project", str(RESULTS_DIR),
        "--workers", str(SHARED_CONFIG["workers"]),
        "--exist-ok",                    # Écraser si le dossier existe déjà
    ]

    logger.info(f"Commande : {' '.join(cmd)}")

    if dry_run:
        logger.info("MODE DRY-RUN : commande affichée, entraînement non lancé.")
        return {"model": MODEL_NAME, "augmentation": augmentation, "dry_run": True}

    # --- 4. Lancement de l'entraînement ---
    start_time = time.time()

    try:
        process = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Entraînement échoué avec le code : {e.returncode}")
        raise RuntimeError(f"Échec de l'entraînement YOLOv5 : {e}")

    end_time     = time.time()
    elapsed_secs = end_time - start_time
    elapsed_min  = round(elapsed_secs / 60, 2)

    logger.info(f"Entraînement terminé en {format_duration(elapsed_secs)}")

    # --- 5. Lecture des métriques ---
    # YOLOv5 sauvegarde dans : results/{name}/results.csv
    yolov5_run_dir = RESULTS_DIR / f"yolov5s_{augmentation}"
    results_csv    = yolov5_run_dir / "results.csv"

    metrics = parse_yolov5_results(results_csv)
    metrics["train_time_min"] = elapsed_min

    # Taille du meilleur modèle sauvegardé
    best_weights = yolov5_run_dir / "weights" / "best.pt"
    metrics["model_size_mb"] = get_model_size_mb(best_weights)

    # --- 6. Sauvegarde des métriques ---
    metrics_file = save_metrics(metrics, MODEL_NAME, augmentation)
    logger.info(f"Métriques sauvegardées : {metrics_file}")

    append_to_csv(metrics)
    logger.info("Métriques ajoutées au CSV global.")

    # --- 7. Génération des courbes d'apprentissage ---
    curves_path = FIGURES_DIR / f"curves_yolov5s_{augmentation}.png"
    try:
        plot_training_curves(
            csv_path=results_csv,
            model_name=MODEL_NAME,
            augmentation=augmentation,
            save_path=curves_path,
        )
        logger.info(f"Courbes sauvegardées : {curves_path}")
    except Exception as e:
        logger.warning(f"Génération des courbes échouée : {e}")

    # --- Résumé du run ---
    logger.info("-" * 40)
    logger.info(f"RÉSULTATS — {MODEL_NAME} | {augmentation}")
    logger.info(f"  mAP@0.5      : {metrics['map50']}")
    logger.info(f"  mAP@0.5:0.95 : {metrics['map50_95']}")
    logger.info(f"  Précision    : {metrics['precision']}")
    logger.info(f"  Rappel       : {metrics['recall']}")
    logger.info(f"  F1-score     : {metrics['f1']}")
    logger.info(f"  Durée        : {elapsed_min} min")
    logger.info(f"  Taille modèle: {metrics['model_size_mb']} Mo")
    logger.info("-" * 40)

    return metrics


# ============================================================
# POINT D'ENTRÉE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Entraînement YOLOv5 sur Pascal VOC avec stratégie d'augmentation."
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        choices=AUGMENTATION_STRATEGIES,
        help="Stratégie d'augmentation à utiliser.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Lance tous les runs d'augmentation séquentiellement.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche les commandes sans lancer l'entraînement (test local CPU).",
    )

    args = parser.parse_args()

    if args.all:
        # Lancer tous les runs séquentiellement
        all_metrics = []
        for aug in AUGMENTATION_STRATEGIES:
            print(f"\n{'='*60}")
            print(f"Run : YOLOv5 | {aug}")
            print(f"{'='*60}")
            try:
                metrics = train_yolov5(aug, dry_run=args.dry_run)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"❌ Échec du run {aug} : {e}")

        print(f"\n✅ Tous les runs YOLOv5 terminés ({len(all_metrics)}/{len(AUGMENTATION_STRATEGIES)})")

    elif args.augmentation:
        train_yolov5(args.augmentation, dry_run=args.dry_run)

    else:
        parser.print_help()
        print("\nExemples d'utilisation :")
        print("  python src/train_yolov5.py --augmentation no_augmentation")
        print("  python src/train_yolov5.py --augmentation mosaic")
        print("  python src/train_yolov5.py --all")
        print("  python src/train_yolov5.py --all --dry-run   # test sans GPU")


if __name__ == "__main__":
    main()