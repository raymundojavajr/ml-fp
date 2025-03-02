# ML-Ops Predictive Maintenance Pipeline

A modular machine learning pipeline for predictive maintenance using XGBoost. This project includes data loading, processing, splitting, model training, evaluation, and prediction. It is built for reproducibility and easy integration with tools like Dagster, MLflow, and Docker.

## Folder Structure

```plaintext
.
├── README.md                # Project overview and instructions
├── data
│   ├── processed            # Processed data (model-ready CSV)
│   │   └── predictive_maintenance_processed.csv
│   └── raw                  # Raw data
│       └── predictive_maintenance.csv
├── dagster                  # Dagster assets, job, and definitions
│   ├── assets.py            # Dagster assets for data lineage
│   ├── definitions.py       # Registers assets and jobs
│   └── jobs.py              # Traditional op-based pipeline job
├── docs                     # Documentation & design notes
├── hello.py                 # Example script
├── main.py                  # Pipeline orchestrator (direct run)
├── notebooks                # Jupyter notebooks for exploration
│   └── 01-data-exploration.ipynb
├── pyproject.toml           # Dependency management via uv
├── reports                  # Generated reports
├── src                      # Source code
│   ├── data                 # Data loading, processing & splitting
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   ├── process_data.py
│   │   └── split_data.py
│   ├── features             # Feature engineering (if any)
│   └── models               # Model training, evaluation & prediction
│       ├── __init__.py
│       ├── evaluate_model.py
│       ├── predict_model.py
│       └── train_model.py
├── tests                    # Unit tests
└── visualizations           # Plots & visualizations
```

## Setup & Running

1. **Clone the repo & navigate to project root:**
   ```bash
   git clone <your-repo-url>
   cd ml-fp
   ```

2. **Activate your uv virtual environment:**
   ```bash
   uv
   ```
   *(Ensure you have uv installed and configured.)*

3. **Install dependencies using your pyproject.toml (uv will handle this).**

4. **Run the end-to-end pipeline (direct execution):**
   ```bash
   python main.py
   ```
   This script trains the model, saves it, and generates predictions.

5. **Run individual modules if needed:**
   - Data processing: `python -m src.data.process_data`
   - Model training: `python -m src.models.train_model`
   - Prediction: `python -m src.models.predict_model`

## Dagster Integration

Dagster is used to orchestrate and monitor your ML pipeline. The Dagster setup in this project includes:

- **Asset Definitions:**  
  Located in `dagster/assets.py`, assets such as `processed_data_asset`, `trained_model_asset`, `predictions_asset`, and `evaluation_metrics_asset` provide data lineage.

- **Job Pipeline:**  
  The traditional op-based pipeline is defined in `dagster/jobs.py` as `ml_pipeline_job`.

- **Definitions Registration:**  
  Both assets and the job are registered in `dagster/definitions.py`.

- **Workspace Configuration:**  
  The `workspace.yaml` file in the project root points Dagster to load definitions from `orchestrator.definitions` (or `dagster/definitions.py` if you haven't renamed the folder). For example:
  ```yaml
  load_from:
    - python_module: dagster.definitions
  ```
  *(Adjust the module path if you renamed the folder.)*

### To Test with Dagster:

1. **Set the DAGSTER_HOME Environment Variable:**
   ```bash
   export DAGSTER_HOME=$(pwd)/dagster_home
   ```

2. **Launch Dagit (Dagster UI):**
   ```bash
   dagit -w workspace.yaml
   ```
   Open your browser at [http://127.0.0.1:3000](http://127.0.0.1:3000).

3. **Explore Dagit:**
   - **Assets Tab:** View the asset lineage graph (showing processed data, trained model, predictions, and evaluation metrics).
   - **Jobs Tab:** View and launch the traditional pipeline job (`ml_pipeline_job`) to see the step-by-step op execution.
   - You can materialize assets individually or run the entire job.

## MLflow Integration (Upcoming)

The next steps include integrating MLflow for experiment tracking:
- **MLflow Setup:** Install MLflow and configure the tracking URI.
- **Logging Experiments:** Add MLflow logging calls in your model training and evaluation code.
- **Model Registry:** Optionally, register and manage your trained models via MLflow's model registry.

## Next Steps

- Integrate MLflow for experiment tracking.
- Containerize the application with Docker.
- Expand tests and documentation.

## Contributing

Fork the repository and create feature branches. Open pull requests with clear descriptions of your changes.

## License

Licensed under the MIT License.