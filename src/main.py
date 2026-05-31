import typer
import sys
from loguru import logger

from .config import settings
from .data import load_data
from .model import ROCCurveTool
from .visualizer import ROCVisualizer

app = typer.Typer(help="ROC Curve Generator CLI")
logger.remove()
logger.add(sys.stderr, level=settings.log_level)


@app.command()
def generate(dataset: str = typer.Option("wine", help="Dataset: iris, wine, breast_cancer")):
    logger.info(f"Generating ROC curve for {dataset}...")
    X, y, fn, cn = load_data(dataset)
    tool = ROCCurveTool()
    results = tool.train_and_compute(X, y, cn)
    logger.info(f"AUC scores: {results['auc']}")
    ROCVisualizer.plot(results["fpr"], results["tpr"], results["auc"], save_path=settings.plots_dir / "roc_curve.png")
    logger.success("Done!")


if __name__ == "__main__":
    app()
