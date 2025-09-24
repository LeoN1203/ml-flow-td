# MLflow Model Serving with Canary Deployment

A containerized MLflow model serving API with canary deployment capabilities for machine learning models. This project demonstrates MLOps best practices including experiment tracking, model versioning, and safe production deployments.

## Features

- **MLflow Tracking Server**: Complete experiment tracking and model registry
- **Model Serving API**: FastAPI-based REST service for model inference
- **Canary Deployment**: Gradual model rollouts with configurable traffic splitting
- **Docker Containerization**: Fully containerized setup with Docker Compose
- **Health Monitoring**: Built-in health checks and status endpoints

## Architecture

```
┌─────────────────┐    ┌──────────────────────┐
│   MLflow UI     │    │   Model Service      │
│   :5000         │◄───┤   :8000              │
│                 │    │                      │
│ • Experiments   │    │ • Current Model      │
│ • Models        │    │ • Next Model         │
│ • Artifacts     │    │ • Canary Logic       │
└─────────────────┘    └──────────────────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.9+ (for training scripts)

### 1. Launch the Services

```bash
# Clone or download the project files
# Start MLflow server and model service
docker compose up -d

# Check services are running
docker compose ps
```

### 2. Access the UIs

- **MLflow UI**: http://localhost:5000
- **Model API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 3. Train and Register a Model

```bash
# Run inside the model-service container
docker compose exec model-service python train_model.py
```

## API Endpoints

### Core Endpoints

| Endpoint   | Method | Description                          |
| ---------- | ------ | ------------------------------------ |
| `/`        | GET    | Service status and health            |
| `/predict` | POST   | Make predictions (uses canary logic) |
| `/health`  | GET    | Detailed health check                |

### Canary Deployment

| Endpoint             | Method | Description                        |
| -------------------- | ------ | ---------------------------------- |
| `/canary-status`     | GET    | View canary deployment status      |
| `/update-model`      | POST   | Update the next (canary) model     |
| `/accept-next-model` | POST   | Promote canary model to production |

### Model Management

| Endpoint      | Method | Description                         |
| ------------- | ------ | ----------------------------------- |
| `/model-info` | GET    | View current and next model details |

## Configuration

### Environment Variables

| Variable                | Default                | Description                             |
| ----------------------- | ---------------------- | --------------------------------------- |
| `MLFLOW_TRACKING_URI`   | `http://mlflow:5000`   | MLflow tracking server URL              |
| `DEFAULT_MODEL_NAME`    | `StudentExamPredictor` | Default model to load on startup        |
| `DEFAULT_MODEL_VERSION` | `latest`               | Default model version                   |
| `CANARY_PROBABILITY`    | `0.1`                  | Initial canary traffic percentage (10%) |

## Development

### Project Structure

```
├── app/
|   ├── main.py                     # FastAPI model serving application
|   ├── requirements.txt            # Python dependencies
|   ├── train_model.py              # Model training script
|   └── student_exam_scores.csv     # Sample dataset
├── test_predict.py                 # Test script for predictions
├── Dockerfile                      # Container definition
├── docker-compose.yml              # Multi-container setup
└── README.md                       # This file
```

## Monitoring and Troubleshooting

### Check Service Status

```bash
# View running containers
docker compose ps

# Check service logs
docker compose logs mlflow
docker compose logs model-service

# Check container health
docker compose exec model-service curl localhost:8000/health
```

## Canary Deployment Workflow

1. **Initial State**: Both current and next models are identical
2. **Deploy New Version**: Use `/update-model` to load new version into next slot
3. **Gradual Testing**:
   - Start with 5% canary traffic
   - Monitor metrics and logs
   - Gradually increase (10% → 25% → 50% → 100%)
4. **Validation**: Check prediction quality and system performance
5. **Promotion**: Use `/accept-next-model` to promote successful canary
6. **Rollback**: Set canary probability to 0 if issues detected

## License

This project is provided as-is for educational and demonstration purposes.
