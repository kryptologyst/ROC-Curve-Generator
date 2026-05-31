import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_data
from src.model import ROCCurveTool

st.set_page_config(page_title="ROC Curve", page_icon="📈", layout="wide")
st.title("📈 ROC Curve Generator")
st.markdown("Receiver Operating Characteristic curves with AUC scores.")

dataset_name = st.selectbox("Dataset", ["wine", "iris", "breast_cancer"])
X, y, fn, cn = load_data(dataset_name)

if st.button("Generate ROC", type="primary"):
    tool = ROCCurveTool()
    results = tool.train_and_compute(X, y, cn)
    auc_df = pd.DataFrame({"Class": list(results["auc"].keys()), "AUC": list(results["auc"].values())})
    st.dataframe(auc_df.set_index("Class"), use_container_width=True)
    import plotly.graph_objects as go
    fig = go.Figure()
    for label in results["fpr"]:
        fig.add_trace(go.Scatter(
            x=results["fpr"][label], y=results["tpr"][label],
            mode="lines", name=f"{label} (AUC={results['auc'][label]:.3f})",
            line=dict(width=2),
        ))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=500)
    st.plotly_chart(fig, use_container_width=True)
