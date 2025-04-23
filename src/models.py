from river import linear_model, metrics, preprocessing, drift, optim, base
from river.utils import Rolling
from collections import defaultdict
import numpy as np

class ADWINBaggingRegressor(base.Regressor):
    def __init__(self, model, n_models=10, seed=None):
        self.model = model
        self.n_models = n_models
        self.seed = seed
        
        self.models = [model.clone() for _ in range(n_models)]
        self.adwins = [drift.ADWIN() for _ in range(n_models)]
        self.rng = np.random.default_rng(seed=42)

    def learn_one(self, x, y):
        for i, model in enumerate(self.models):
            k = self.rng.poisson(lam=1.0)
            for _ in range(k):
                y_pred = model.predict_one(x)
                if y_pred is not None:
                    error = abs(y - y_pred)
                    self.adwins[i].update(error)
                    if self.adwins[i].drift_detected:
                        # positive novelty dont touch lmao
                        if error <= self.adwins[i].estimation:
                            continue
                        
                        # Reset the model if drift is detected
                        # print(f"🚨 Drift detected, resetting model {i}")
                        self.models[i] = model.clone()
                        self.adwins[i] = drift.ADWIN()
                model.learn_one(x, y)

    def predict_one(self, x):
        predictions = [model.predict_one(x) for model in self.models]
        # Filter out None predictions
        predictions = [pred for pred in predictions if pred is not None]
        if predictions:
            return sum(predictions) / len(predictions)
        else:
            return None
        
    def clone(self):
        return ADWINBaggingRegressor(self.model.clone(), self.n_models, seed=self.seed)