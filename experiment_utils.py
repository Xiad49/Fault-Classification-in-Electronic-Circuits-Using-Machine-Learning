import os
from datetime import datetime

def create_experiment(base_dir: str):
    exp_id = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, exp_id)

    paths = {
        "root": exp_dir,
        "models": os.path.join(exp_dir, "models"),
        "results": os.path.join(exp_dir, "results"),
        "logs": os.path.join(exp_dir, "logs")
    }

    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    return exp_id, paths
