import subprocess, sys

result = subprocess.run(
    [sys.executable, "-c",
     "import sys, os; sys.path.insert(0, 'backend'); os.chdir('backend'); "
     "from config.settings import IMAGE_SIZE; print('config OK', IMAGE_SIZE); "
     "from preprocessing.image_loader import preprocess_image_bytes; print('preprocessing OK'); "
     "from preprocessing.ela import compute_ela; print('ela OK'); "
     "from models.densenet import ImprovedDenseNet; print('densenet OK'); "
     "from models.model_loader import get_model; print('model_loader OK'); "
     "from utils.image_utils import pil_to_base64; print('utils OK'); "
     "from inference.predictor import predict_image; print('inference OK'); "
     "m = get_model(pretrained=False); print('model instantiation OK');"
     ],
    capture_output=True, text=True, cwd=r"d:\MiniProject_MCA_final"
)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr[:3000] if result.stderr else "")
print("Return code:", result.returncode)
