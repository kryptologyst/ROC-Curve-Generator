import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from loguru import logger


class ROCCurveTool:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.fpr_: dict = {}
        self.tpr_: dict = {}
        self.auc_: dict = {}

    def compute(self, y_true: np.ndarray, y_score: np.ndarray, class_names: list = None) -> dict:
        n_classes = y_score.shape[1] if y_score.ndim > 1 else 2
        if class_names is None:
            class_names = [f"class_{i}" for i in range(n_classes)]
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_score[:, 1] if y_score.ndim > 1 else y_score)
            self.fpr_["macro"] = fpr.tolist()
            self.tpr_["macro"] = tpr.tolist()
            self.auc_["macro"] = float(auc(fpr, tpr))
        else:
            y_bin = label_binarize(y_true, classes=range(n_classes))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                self.fpr_[class_names[i]] = fpr.tolist()
                self.tpr_[class_names[i]] = tpr.tolist()
                self.auc_[class_names[i]] = float(auc(fpr, tpr))
        logger.info(f"ROC AUC: {self.auc_}")
        return {"fpr": self.fpr_, "tpr": self.tpr_, "auc": self.auc_}

    def train_and_compute(self, X, y, class_names=None):
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=self.random_state, stratify=y,
        )
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)
        return self.compute(y_test, y_score, class_names)
