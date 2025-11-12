import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
import pandas as pd
from xgboost import XGBClassifier

bst = XGBClassifier()
bst.load_model("results/model_ipi_only.json")

data = pd.read_csv("data/test_specific_ml_ipi_and_comparators.csv")
X_test = data.filter(like="pred_")
y_test = data["outc_treatment_failure_label_within_0_to_730_days_max_fallback_0"]

disp = CalibrationDisplay.from_estimator(bst, X_test, y_test, n_bins=10, name="ML$_{IPI}$")
plt.savefig("plots/calibration_ipi.png", dpi=300, bbox_inches="tight")
