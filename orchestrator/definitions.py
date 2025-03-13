# definitions.py
from dagster import Definitions
from orchestrator.jobs import ml_pipeline_job  # Import the job from jobs.py
from orchestrator.assets import (
    processed_data_asset,
    trained_model_asset,
    predictions_asset,
    evaluation_metrics_asset,
)

defs = Definitions(
    assets=[
        processed_data_asset,
        trained_model_asset,
        predictions_asset,
        evaluation_metrics_asset,
    ],
    jobs=[ml_pipeline_job],  # Register the job here
)
