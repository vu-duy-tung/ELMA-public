from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'third_party/paper_runtime'
BUNDLE=Path(os.environ.get('ELMA_BUNDLE',ROOT/'assets/default'))
