# Runs on a Kaggle GPU (Tesla T4) kernel.
# Clones the latest code and data from GitHub, trains with train.py,
# and copies the trained model into /kaggle/working so Jenkins can download it.
import os
import shutil
import subprocess

REPO = "https://github.com/abdullahasghar966/pdc-assignment2-gpu-pipeline.git"
WORK = "/kaggle/working"
REPO_DIR = os.path.join(WORK, "repo")

subprocess.run(["git", "clone", "--depth", "1", "-b", "main", REPO, REPO_DIR], check=True)
os.chdir(REPO_DIR)

import torch  # noqa: E402  (preinstalled on Kaggle)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

subprocess.run(["python", "train.py"], check=True)

shutil.copy(os.path.join(REPO_DIR, "models", "model.pt"), os.path.join(WORK, "model.pt"))
shutil.copy(os.path.join(REPO_DIR, "models", "metrics.json"), os.path.join(WORK, "metrics.json"))
print("Done. Model and metrics saved to", WORK)
