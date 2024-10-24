from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class Data:
    y_train : pd.DataFrame | np.ndarray
    y_test : pd.DataFrame | np.ndarray
    X_train : pd.DataFrame | np.ndarray
    X_test : pd.DataFrame | np.ndarray
