import shap, pandas as pd, matplotlib.pyplot as plt
from xgboost import XGBClassifier

bst = XGBClassifier()
bst.load_model("results/model_ipi_only.json")

X_test = pd.read_pickle("data/WIDE_DATA.pkl").filter(like="pred_RKKP_")
explainer = shap.TreeExplainer(bst)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("plots/shap_values_ipi_only.png", dpi=300, bbox_inches="tight")
