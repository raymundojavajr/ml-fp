# orchestrator/main_dagster.py

from dagster import execute_job, DagsterInstance, reconstructable
from orchestrator.jobs import ml_pipeline_job

def main():
    """Execute the ml_pipeline_job and report the outcome."""
    instance = DagsterInstance.get()
    result = execute_job(reconstructable(ml_pipeline_job), instance=instance)
    if result.success:
        print("Dagster pipeline executed successfully.")
    else:
        print("Dagster pipeline failed.")

if __name__ == "__main__":
    main()
