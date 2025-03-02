# ML Project Tracker

**Team Members:**  
- Jeng  
- Karth  
- Shem

## Overview
This tracker outlines our project tasks, current statuses, and team assignments.  
- **Jeng:** Responsible for Project Scope & Data, Data Pipeline (Dagster), and MLflow integration.  
- **Karth:** Responsible for FastAPI endpoints and Containerization (Docker).  
- **Shem:** Responsible for Drift Detection (Evidently AI) and leads Testing & Continuous Integration.  
- **Shared:** Documentation & Notebook updates will be a collaborative effort among all team members.

| **Category**                      | **Task/Component**                                      | **Status**       | **Assigned To**                     | **Due Date**  | **Comments/Next Steps**                                               |
|-----------------------------------|---------------------------------------------------------|------------------|-------------------------------------|---------------|----------------------------------------------------------------------|
| **Project Scope & Data**          | Define ML problem and dataset documentation             | In Progress      | Jeng                                | 2025-03-10    | Initial draft complete; add detailed data dictionary.                |
| **Data Pipeline (Dagster)**       | Implement data ingestion step                           | Completed        | Jeng                                | 2025-03-05    | CSV data loaded successfully.                                        |
| **Data Pipeline (Dagster)**       | Implement feature engineering & transformation          | In Progress      | Jeng                                | 2025-03-12    | Handle missing values and document transformations.                  |
| **Experiment Tracking (MLflow)**  | Integrate MLflow logging (runs, hyperparameters, metrics, artifacts)  | Not Started      | Jeng                                | 2025-03-15    | Integrate MLflow into training/validation scripts.                   |
| **Model Serving (FastAPI)**       | Create `/predict` endpoint with input validation (using Pydantic)         | In Progress      | Karth                               | 2025-03-14    | Develop endpoint; ensure error handling and validation.              |
| **Model Serving (FastAPI)**       | Create `/model` endpoint for model info retrieval       | Not Started      | Karth                               | 2025-03-16    | Implement dynamic retrieval of hyperparameters and input schema.     |
| **Containerization (Docker)**     | Write Dockerfile for FastAPI service                    | Not Started      | Karth                               | 2025-03-20    | Decide on base image and dependency installation.                    |
| **Containerization (Docker)**     | Create `docker-compose.yml` for all services            | Not Started      | Karth                               | 2025-03-22    | Define networking between Dagster, MLflow, and FastAPI containers.    |
| **Drift Detection (Evidently AI)**| Implement Evidently AI drift reports and log to MLflow    | Not Started      | Shem                                | 2025-03-25    | Integrate drift detection in Dagster pipeline and log HTML reports.    |
| **Testing & CI**                  | Write unit tests (pytest) for data processing           | In Progress      | Shem (shared with Jeng & Karth)     | 2025-03-12    | Create tests covering preprocessing, training, and edge cases.       |
| **Testing & CI**                  | Set up GitHub Actions for CI automation                 | Not Started      | Shem (shared with Jeng & Karth)     | 2025-03-18    | Automate tests on pushes and pull requests.                          |
| **Notebook & Documentation**      | Update Jupyter Notebook for full system demo            | In Progress      | Shared (Shem lead, with Jeng & Karth)| 2025-03-14    | Include prediction demo, model info retrieval, drift detection, and clear documentation. |
