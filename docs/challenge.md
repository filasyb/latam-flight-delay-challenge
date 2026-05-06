# LATAM Flight Delay Challenge

## Overview

The goal of this challenge is to move the work delivered in
`challenge/exploration.ipynb` from an exploratory notebook into production-ready
Python code and expose it through a FastAPI service.

The implemented solution keeps the structure of the original challenge and
focuses on making the DS model reproducible, testable, and easy to deploy.

## Part I - Model Operationalization

### Notebook Findings

The notebook explores several feature engineering steps and model candidates.
The target variable is `delay`, defined as:

- `1` when the difference between `Fecha-O` and `Fecha-I` is greater than 15
  minutes.
- `0` otherwise.

The DS evaluated models using all generated features and then repeated the
training using the 10 most important features. The notebook conclusion is that:

- Reducing the model to the 10 most important features does not materially hurt
  performance.
- Balancing classes improves the model's ability to identify delayed flights.
- XGBoost and Logistic Regression produce very similar results after feature
  selection and class balancing.

### Selected Model

I selected a balanced `LogisticRegression` trained with the top 10 features.

The top 10 features are:

- `OPERA_Latin American Wings`
- `MES_7`
- `MES_10`
- `OPERA_Grupo LATAM`
- `MES_12`
- `TIPOVUELO_I`
- `MES_4`
- `MES_11`
- `OPERA_Sky Airline`
- `OPERA_Copa Air`

The balanced Logistic Regression model reported in the notebook has performance
very close to the balanced XGBoost alternative:

| Model | Features | Class balancing | Recall class 1 | F1 class 1 |
| --- | --- | --- | ---: | ---: |
| XGBoost | Top 10 | Yes | ~0.69 | ~0.37 |
| Logistic Regression | Top 10 | Yes | ~0.69 | ~0.36 |

Given this trade-off, Logistic Regression is the better production choice for
this challenge because it avoids adding XGBoost as an additional dependency,
keeps the Docker/runtime footprint smaller, and is easier to reason about during
API deployment. The small difference in F1 score does not justify the extra
operational complexity of XGBoost for this use case.

### Implementation Details

`challenge/model.py` implements the provided `DelayModel` interface without
changing method names or arguments.

The implementation:

- Computes `delay` from `Fecha-I` and `Fecha-O` when the target column is not
  already present in the input data.
- Encodes `OPERA`, `MES`, and `TIPOVUELO` using one-hot encoding.
- Reindexes training and serving inputs to the same fixed set of 10 production
  columns.
- Applies class weights based on the training distribution to improve recall for
  the minority class.
- Validates required input columns before preprocessing so failures are explicit.

This keeps preprocessing deterministic between training and serving, which is
especially important because unseen categories should not change the model input
shape.

### Model Validation

Official command:

```bash
make model-test
```

The model tests validate:

- The preprocessing output shape and columns.
- The target generation.
- The trained model metrics expected by the challenge.
- The prediction output type and length.

During local validation, the model passed the provided test suite:

```text
tests/model: 4 passed
```

## Part II - FastAPI Service

### API Design

The API is implemented in `challenge/api.py` using FastAPI, as required.

Available endpoints:

- `GET /health`
- `POST /predict`

The prediction endpoint accepts a batch of flights:

```json
{
  "flights": [
    {
      "OPERA": "Aerolineas Argentinas",
      "TIPOVUELO": "N",
      "MES": 3
    }
  ]
}
```

And returns:

```json
{
  "predict": [0]
}
```

### Runtime Behavior

The API trains one `DelayModel` instance when the module is loaded. For this
challenge, that is enough because the dataset is local, the model is lightweight,
and the test suite expects a ready-to-use API without an external model registry
or artifact store.

For a larger production system, the next iteration would be to train the model
offline, persist the artifact, and load it at API startup. That would reduce
startup cost and separate model training from model serving.

### Request Validation

The request body is validated with Pydantic models.

The API returns HTTP `400` for:

- Unknown airline in `OPERA`.
- Invalid flight type in `TIPOVUELO`.
- Invalid month outside the `1..12` range.

FastAPI normally returns `422` for request validation errors. The challenge tests
expect `400`, so the API includes a `RequestValidationError` handler that maps
validation failures to HTTP `400`.

### API Validation

Official command:

```bash
make api-test
```

During local validation, the API passed the provided test suite:

```text
tests/api: 4 passed
```

## Dependency Management

The original dependency set pins `fastapi~=0.86.0`, which depends on
`starlette==0.20.4`. That Starlette version uses the AnyIO 3 test client API.

Modern environments may resolve AnyIO 4, which removes
`anyio.start_blocking_portal` and breaks `fastapi.testclient.TestClient` with:

```text
AttributeError: module 'anyio' has no attribute 'start_blocking_portal'
```

To make local development and CI reproducible, `anyio<4` was added to
`requirements.txt`.

## Makefile Notes

The provided tests load the dataset using `../data/data.csv`. That path works
when tests are executed from the `challenge` directory. The `Makefile` was
therefore adjusted so the official test targets run from the expected working
directory while still being called from the repository root.

Official commands:

```bash
make model-test
make api-test
```
