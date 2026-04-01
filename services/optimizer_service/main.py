import sys
import pandas as pd
import json
from solver import DispatchConfig, optimize_dispatch

if __name__ == "__main__":
    config_path = sys.argv[1]
    data_path = sys.argv[2]
    output_path = sys.argv[3]

    with open(config_path, 'r') as f:
        cfg_dict = json.load(f)
    cfg = DispatchConfig(**cfg_dict)

    forecast = pd.read_csv(data_path)

    result = optimize_dispatch(forecast, cfg)
    
    result.to_csv(output_path, index=False)
