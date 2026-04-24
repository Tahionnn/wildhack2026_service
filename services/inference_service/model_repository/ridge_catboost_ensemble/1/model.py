
import io
import os

import numpy as np
import triton_python_backend_utils as pb_utils


FORECAST_POINTS = 10
EPS = 1e-6
TARGET_STEP_IDX = 3

class TritonPythonModel:
    def initialize(self, args):
        import joblib, json
        from catboost import CatBoostRegressor
        import os

        model_dir = os.path.join(args["model_repository"], args["model_version"])
        self.ridge = joblib.load(os.path.join(model_dir, "ridge_baseline.pkl"))

        with open(os.path.join(model_dir, "feature_cols.json")) as f:
            meta = json.load(f)
        self.feature_cols = meta["feature_cols"]

        self.cb_models = []
        for i in range(1, FORECAST_POINTS + 1):
            m = CatBoostRegressor()
            m.load_model(os.path.join(model_dir, f"catboost_step_{i}.cbm"))
            self.cb_models.append(m)
            
    def execute(self, requests):
        from catboost import Pool
        import pandas as pd
        import numpy as np
        import triton_python_backend_utils as pb_utils

        responses = []

        for request in requests:
            X_np = pb_utils.get_input_tensor_by_name(request, "INPUT_ARRAY").as_numpy()
            
            if X_np.size == 0:
                raise RuntimeError("[ridge_catboost_ensemble] INPUT_ARRAY is empty!")

            X = pd.DataFrame(X_np, columns=self.feature_cols)

            if X.shape[1] != len(self.feature_cols):
                raise RuntimeError(
                    f"[ridge_catboost_ensemble] Feature mismatch: expected {len(self.feature_cols)}, got {X.shape[1]}"
                )

            ridge_preds = self.ridge.predict(X) 

            step_preds = []
            for idx, model in enumerate(self.cb_models):
                baseline_step = ridge_preds[:, idx]  
                baseline_step = np.nan_to_num(baseline_step, nan=0.0, posinf=1e6, neginf=0.0)
                baseline_step = np.clip(baseline_step, EPS, None)
                baseline_step = np.log(baseline_step) * 0.7

                pool = Pool(X, baseline=baseline_step)
                preds = model.predict(pool).astype("float32")
                step_preds.append(preds)

            all_steps = np.column_stack(step_preds).astype("float32")

            out_tensor = pb_utils.Tensor("OUTPUT_PREDS", all_steps)
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses

    def finalize(self):
        pb_utils.Logger.log_info("[ridge_catboost_ensemble] finalize")
