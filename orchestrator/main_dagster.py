# orchestrator/main_dagster.py

from dagster import execute_job, DagsterInstance, reconstructable
from orchestrator.jobs import ml_pipeline_job

def main():
    # Obtain a persistent Dagster instance from DAGSTER_HOME
    instance = DagsterInstance.get()
    # Reconstruct the job for proper execution and history tracking
    result = execute_job(reconstructable(ml_pipeline_job), instance=instance)
    if result.success:
        print("Dagster pipeline executed successfully.")
    else:
        print("Dagster pipeline failed.")

if __name__ == "__main__":
    main()
