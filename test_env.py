"""
Script de vérification de l'environnement de développement.
"""
import sys

def check_import(module_name, alias=None):
    try:
        if alias:
            exec(f"import {module_name} as {alias}")
        else:
            exec(f"import {module_name}")
        print(f"  ✅ {module_name}")
    except ImportError as e:
        print(f"  ❌ {module_name} — {e}")

print(f"Python version : {sys.version}")
print("\nVérification des dépendances :")
check_import("torch")
check_import("torchvision")
check_import("ultralytics")
check_import("numpy")
check_import("pandas")
check_import("matplotlib")
check_import("seaborn")
check_import("cv2")
check_import("sklearn")

import torch
print(f"\nPyTorch version : {torch.__version__}")
print(f"CUDA disponible : {torch.cuda.is_available()}")
print("\nEnvironnement prêt pour le développement local.")