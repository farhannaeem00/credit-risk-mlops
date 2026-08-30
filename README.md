# Credit Risk MLOps — Home Credit Default Risk

End-to-end MLOps pipeline that scores loan-default risk on the Home Credit
Default Risk dataset. Built as a zero-cost, portfolio-grade project:

Kaggle dataset -> MongoDB Atlas (raw store) -> DVC/MLflow on DagsHub
(versioning + experiment tracking + model registry) -> FastAPI -> Docker ->
GitHub Actions -> Render (deployment) -> Evidently (monitoring).

See `projectflow.txt` for the full step-by-step build log and
`crashcourse.txt` for why the project is structured as an installable
local package.

## Local setup
```bash
conda create -n creditrisk python=3.10 -y
conda activate creditrisk
pip install -r requirements.txt
```

## Status
Scaffold only — components are being filled in phase by phase per
`projectflow.txt`.
