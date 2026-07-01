"""
train_yolov8.py — Entraînement YOLOv8 sur Pascal VOC avec contrôle des augmentations.

Ce script gère :
- L'entraînement de YOLOv8s via l'API Python Ultralytics
- Le contrôle fin des augmentations via les paramètres d'entraînement
- Le logging des métriques à chaque run
- La sauvegarde des résultats dans la structure du projet

Différence clé avec YOLOv5 :
    YOLOv8 utilise l'API Python Ultralytics (pas de subprocess).
    Les augmentations sont passées directement comme arguments à model.train().
"""

import argparse
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    PROJECT_ROOT,
    RESULTS_DIR,
    FIGURES_DIR,
    SHARED_CONFIG,
    AUGMENTATION_STRATEGIES,
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
# CONFIGURATION YOLOV8
# ============================================================

MODEL_NAME = "yolov8s"
DATA_YAML  = PROJECT_ROOT / "data" / "data.yaml"


# ============================================================
# CONFIGURATION DES AUGMENTATIONS YOLOV8
# ============================================================

def get_augmentation_params(augmentation: str) -> dict:
    # Base : toutes les augmentations désactivées
    base_params = {
        "fliplr"     : 0.0,
        "flipud"     : 0.0,
        "degrees"    : 0.0,
        "translate"  : 0.0,
        "scale"      : 0.0,
        "shear"      : 0.0,
        "perspective": 0.0,
        "mosaic"     : 0.0,
        "mixup"      : 0.0,
        "hsv_h"      : 0.0,
        "hsv_s"      : 0.0,
        "hsv_v"      : 0.0,
        "copy_paste" : 0.0,
    }

    if augmentation == "no_augmentation":
        # Baseline : aucune augmentation
        return base_params

    elif augmentation == "flip_horizontal":
        return {**base_params, "fliplr": 0.5}

    elif augmentation == "rotation":
        return {**base_params, "degrees": 10.0}

    elif augmentation == "mosaic":
        return {**base_params, "mosaic": 1.0}

    elif augmentation == "mixup":
        return {**base_params, "mixup": 0.1}

    else:
        raise ValueError(
            f"Stratégie d'augmentation inconnue : '{augmentation}'. "
            f"Valeurs acceptées : {AUGMENTATION_STRATEGIES}"
        )


# ============================================================
# LECTURE DES MÉTRIQUES YOLOV8
# ============================================================

def parse_yolov8_results(run_dir: Path) -> dict:
    import pandas as pd

    results_csv = run_dir / "results.csv"

    if not results_csv.exists():
        raise FileNotFoundError(
            f"results.csv introuvable dans {run_dir}. "
            "L'entraînement s'est-il terminé correctement ?"
        )

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    # Dernière ligne = métriques de la dernière époque
    last = df.iloc[-1]

    def get_val(candidates):
        """Cherche une colonne parmi plusieurs noms possibles."""
        for c in candidates:
            if c in df.columns:
                return float(last[c])
        return 0.0

    # YOLOv8 utilise des noms légèrement différents selon la version
    precision = get_val(["metrics/precision(B)", "metrics/precision"])
    recall    = get_val(["metrics/recall(B)",    "metrics/recall"])
    map50     = get_val(["metrics/mAP50(B)",     "metrics/mAP_0.5"])
    map50_95  = get_val(["metrics/mAP50-95(B)",  "metrics/mAP_0.5:0.95"])

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

def train_yolov8(augmentation: str, dry_run: bool = False) -> dict:
    logger = setup_logger(
        f"train_yolov8_{augmentation}",
        log_file=str(get_run_dir(MODEL_NAME, augmentation) / "train.log"),
    )

    logger.info("=" * 60)
    logger.info(f"DÉMARRAGE — YOLOv8 | Augmentation : {augmentation}")
    logger.info("=" * 60)

    # --- 1. Validation des prérequis ---
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml introuvable : {DATA_YAML}\n"
            "Téléchargez Pascal VOC via Roboflow et placez data.yaml dans data/."
        )

    if augmentation not in AUGMENTATION_STRATEGIES:
        raise ValueError(f"Augmentation invalide : '{augmentation}'")

    # --- 2. Import Ultralytics (import tardif pour ne pas bloquer les tests) ---
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "La bibliothèque 'ultralytics' n'est pas installée.\n"
            "Exécutez : pip install ultralytics"
        )

    # --- 3. Récupération des paramètres d'augmentation ---
    aug_params = get_augmentation_params(augmentation)
    logger.info(f"Paramètres d'augmentation : {aug_params}")

    # Nom du run (dossier de sortie)
    run_name = f"yolov8s_{augmentation}"

    if dry_run:
        logger.info("MODE DRY-RUN : configuration prête, entraînement non lancé.")
        logger.info(f"Commande équivalente : model.train(data='{DATA_YAML}', epochs={SHARED_CONFIG['epochs']}, ...)")
        return {"model": MODEL_NAME, "augmentation": augmentation, "dry_run": True}

    # --- 4. Chargement du modèle ---
    # yolov8s.pt est téléchargé automatiquement depuis le hub Ultralytics
    logger.info("Chargement du modèle YOLOv8s (téléchargement auto si absent)...")
    model = YOLO("yolov8s.pt")

    # --- 5. Lancement de l'entraînement ---
    start_time = time.time()

    try:
        model.train(
            # Dataset
            data=str(DATA_YAML),

            # Conditions expérimentales identiques à YOLOv5
            epochs=SHARED_CONFIG["epochs"],
            imgsz=SHARED_CONFIG["img_size"],
            batch=SHARED_CONFIG["batch_size"],
            workers=SHARED_CONFIG["workers"],

            # Sauvegarde des résultats
            name=run_name,
            project=str(RESULTS_DIR),
            exist_ok=True,

            # Device : 0 = GPU (Colab), 'cpu' = CPU (local)
            # Ultralytics détecte automatiquement le GPU disponible
            device=0 if _cuda_available() else "cpu",

            # Paramètres d'augmentation isolés
            **aug_params,

            # Options de stabilité
            cache=True,       # Mise en cache des images pour accélérer
            verbose=True,     # Afficher les logs d'entraînement
            save=True,        # Sauvegarder les checkpoints
            plots=True,       # Générer les graphiques automatiquement
        )
    except Exception as e:
        logger.error(f"Entraînement échoué : {e}")
        raise RuntimeError(f"Échec de l'entraînement YOLOv8 : {e}")

    end_time     = time.time()
    elapsed_secs = end_time - start_time
    elapsed_min  = round(elapsed_secs / 60, 2)

    logger.info(f"Entraînement terminé en {format_duration(elapsed_secs)}")

    # --- 6. Lecture des métriques ---
    yolov8_run_dir = RESULTS_DIR / run_name
    metrics = parse_yolov8_results(yolov8_run_dir)
    metrics["train_time_min"] = elapsed_min

    # Taille du meilleur modèle
    best_weights = yolov8_run_dir / "weights" / "best.pt"
    metrics["model_size_mb"] = get_model_size_mb(best_weights)

    # --- 7. Sauvegarde des métriques ---
    metrics_file = save_metrics(metrics, MODEL_NAME, augmentation)
    logger.info(f"Métriques sauvegardées : {metrics_file}")

    append_to_csv(metrics)
    logger.info("Métriques ajoutées au CSV global.")

    # --- 8. Génération des courbes d'apprentissage ---
    results_csv  = yolov8_run_dir / "results.csv"
    curves_path  = FIGURES_DIR / f"curves_yolov8s_{augmentation}.png"
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
# FONCTION UTILITAIRE INTERNE
# ============================================================

def _cuda_available() -> bool:
    """Vérifie si un GPU CUDA est disponible."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ============================================================
# POINT D'ENTRÉE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Entraînement YOLOv8 sur Pascal VOC avec stratégie d'augmentation."
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
        help="Initialise le modèle sans lancer l'entraînement (test local CPU).",
    )

    args = parser.parse_args()

    if args.all:
        all_metrics = []
        for aug in AUGMENTATION_STRATEGIES:
            print(f"\n{'='*60}")
            print(f"Run : YOLOv8 | {aug}")
            print(f"{'='*60}")
            try:
                metrics = train_yolov8(aug, dry_run=args.dry_run)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"❌ Échec du run {aug} : {e}")

        print(f"\n✅ Tous les runs YOLOv8 terminés ({len(all_metrics)}/{len(AUGMENTATION_STRATEGIES)})")

    elif args.augmentation:
        train_yolov8(args.augmentation, dry_run=args.dry_run)

    else:
        parser.print_help()
        print("\nExemples d'utilisation :")
        print("  python src/train_yolov8.py --augmentation no_augmentation")
        print("  python src/train_yolov8.py --augmentation mosaic")
        print("  python src/train_yolov8.py --all")
        print("  python src/train_yolov8.py --all --dry-run   # test sans GPU")


if __name__ == "__main__":
    main()