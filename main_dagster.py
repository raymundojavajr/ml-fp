# main_dagster.py

from dagster import execute_job, DagsterInstance, reconstructable
from pipelines.dagster_pipeline import ml_pipeline_job

def main():
    # Obtain a persistent instance from the DAGSTER_HOME environment variable
    instance = DagsterInstance.get()
    result = execute_job(reconstructable(ml_pipeline_job), instance=instance)
    if result.success:
        print("Dagster pipeline executed successfully.")
    else:
        print("Dagster pipeline failed.")

if __name__ == "__main__":
    main()
