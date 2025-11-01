"""
bayesian_ml.py

Consolidated Bayesian machine learning components.
Combines recursive Bayesian belief updating with data handling and persistence.
"""

import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BayesianPrediction:
    """Container for Bayesian prediction results"""
    prediction: float
    confidence: float
    variance: float
    timestamp: str
    metadata: Dict


@dataclass
class ModelState:
    """Container for Bayesian model state"""
    priors: Dict[str, float]
    likelihoods: Dict[str, float]
    posterior_history: List[Dict[str, float]]
    variance_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]


# =============================================================================
# BAYESIAN ML MODEL (Recursive Belief Updater)
# =============================================================================

class BayesianMLModel:
    """Recursive Bayesian belief updater with persistent state"""

    def __init__(self, data_path: str = "data/bayesian_model_data.pkl"):
        self.path = Path(data_path)
        self._load_or_init()

    def _load_or_init(self):
        """Load existing model state or initialize new one"""
        if self.path.exists():
            try:
                with open(self.path, "rb") as f:
                    self.state: ModelState = pickle.load(f)
            except:
                self._init_state()
        else:
            self._init_state()

    def _init_state(self):
        """Initialize default model state"""
        self.state = ModelState(
            {"default": 0.5},
            {},
            [],
            {"global_variance": 0.0},
            {"accuracy": 0.0, "confidence": 0.0},
        )
        with open(self.path, "wb") as f:
            pickle.dump(self.state, f)

    def update_belief(
        self, hypothesis: str, evidence: Dict[str, float]
    ) -> BayesianPrediction:
        """Update belief based on new evidence using Bayesian inference"""
        prior = self.state.priors.get(hypothesis, 0.5)
        lik = self._likelihood(hypothesis, evidence)
        marg = self._marginal(evidence)
        post = (prior * lik) / marg if marg else prior

        # Calculate confidence using geometric mean of factors
        conf = (
            post
            * np.mean(list(evidence.values()))
            * np.mean(list(self.state.priors.values()))
        ) ** (1 / 3)

        # Calculate variance from recent predictions
        vars_ = (
            np.var(
                [p["prediction"] for p in self.state.posterior_history[-10:]] + [post]
            )
            if self.state.posterior_history
            else 0.0
        )

        pred = BayesianPrediction(
            post,
            conf,
            vars_,
            datetime.now().isoformat(),
            {"evidence": evidence, "prior": prior},
        )

        # Update history
        self.state.posterior_history.append(
            {
                "prediction": pred.prediction,
                "confidence": pred.confidence,
                "variance": pred.variance,
                "timestamp": pred.timestamp,
            }
        )

        # Persist state
        with open(self.path, "wb") as f:
            pickle.dump(self.state, f)

        return pred

    def _likelihood(self, hypothesis: str, evidence: Dict[str, float]) -> float:
        """Calculate likelihood of evidence given hypothesis"""
        return (
            np.prod(
                [
                    stats.norm.pdf(
                        val,
                        loc=self.state.likelihoods.get(f"{hypothesis}|{k}", 0),
                        scale=0.1,
                    )
                    for k, val in evidence.items()
                ]
            )
            or 0.5
        )

    def _marginal(self, evidence: Dict[str, float]) -> float:
        """Calculate marginal likelihood"""
        return sum(
            self.state.priors[h] * self.update_belief(h, evidence).prediction
            for h in self.state.priors
        )


# =============================================================================
# DATA HANDLER (Persistence and Analysis)
# =============================================================================

class BayesianDataHandler:
    """Handles loading/saving of Bayesian prediction data"""

    def __init__(self, data_dir: str = "data/bayesian_ml"):
        self.data_dir = Path(data_dir)
        self.predictions_file = self.data_dir / "predictions.csv"
        self.metadata_file = self.data_dir / "metadata.json"
        self._init_storage()

    def _init_storage(self):
        """Initialize data storage directories and files"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.predictions_file.exists():
            pd.DataFrame(
                columns=[
                    "timestamp",
                    "prediction",
                    "confidence",
                    "variance",
                    "hypothesis",
                    "evidence",
                ]
            ).to_csv(self.predictions_file, index=False)
        if not self.metadata_file.exists():
            meta = {
                "model_version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "performance_metrics": {},
                "model_parameters": {},
            }
            self.save_metadata(meta)

    def save_prediction(self, prediction: BayesianPrediction, hypothesis: str) -> None:
        """Save a prediction to persistent storage"""
        row = {
            "timestamp": prediction.timestamp,
            "prediction": prediction.prediction,
            "confidence": prediction.confidence,
            "variance": prediction.variance,
            "hypothesis": hypothesis,
            "evidence": json.dumps(prediction.metadata["evidence"]),
        }
        pd.DataFrame([row]).to_csv(
            self.predictions_file, mode="a", header=False, index=False
        )

    def get_predictions(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> pd.DataFrame:
        """Retrieve predictions with optional filtering"""
        df = pd.read_csv(self.predictions_file, parse_dates=["timestamp"])
        if start_date:
            df = df[df.timestamp >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.timestamp <= pd.to_datetime(end_date)]
        if min_confidence > 0:
            df = df[df.confidence >= min_confidence]
        return df

    def save_metadata(self, metadata: Dict) -> None:
        """Save metadata to JSON file"""
        metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def get_metadata(self) -> Dict:
        """Load metadata from JSON file"""
        with open(self.metadata_file) as f:
            return json.load(f)

    def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """Remove data older than specified days"""
        df = pd.read_csv(self.predictions_file, parse_dates=["timestamp"])
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_to_keep)
        df = df[df.timestamp >= cutoff]
        df.to_csv(self.predictions_file, index=False)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data structures
    "BayesianPrediction",
    "ModelState",

    # ML Model
    "BayesianMLModel",

    # Data Handler
    "BayesianDataHandler",
]