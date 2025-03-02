# orchestrator/definitions.py

from dagster import Definitions
from orchestrator.assets import (
    processed_data_asset,
    trained_model_asset,
    predictions_asset,
    evaluation_metrics_asset,
)
from orchestrator.jobs import ml_pipeline_job

defs = Definitions(
    assets=[
        processed_data_asset,
        trained_model_asset,
        predictions_asset,
        evaluation_metrics_asset,
    ],
    jobs=[ml_pipeline_job],
)
