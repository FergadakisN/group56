# Load Testing Guide

This guide covers load testing the Fish Species Classification API to ensure it can handle production traffic.

## Overview

Load testing helps identify:

- **Performance bottlenecks**: Slow endpoints or inefficient code
- **Capacity limits**: Maximum requests per second the system can handle
- **Resource requirements**: CPU, memory, and concurrency needs
- **Scalability behavior**: How the system responds to increasing load
- **Error handling**: System behavior under stress

We use [Locust](https://locust.io/) for load testing, which provides:

- Python-based test scenarios
- Web UI for monitoring
- Distributed load generation
- Detailed performance metrics

## Installation

```bash
# Install locust
pip install locust

# Or install all dev dependencies
pip install -e ".[dev]"
```

## Quick Start

### 1. Start the API Locally

```bash
# Start API server
uvicorn group_56.api:app --host 0.0.0.0 --port 8000
```

### 2. Run Load Tests

```bash
# Run with web UI (recommended for first-time use)
locust -f tests/load_test.py --host http://localhost:8000

# Open browser to http://localhost:8089
# Set users and spawn rate in the web UI
```

### 3. Run Headless Tests

```bash
# Run headless test (no UI)
locust -f tests/load_test.py \
    --host http://localhost:8000 \
    --users 50 \
    --spawn-rate 5 \
    --run-time 60s \
    --headless
```

## Test Scenarios

### FishClassifierUser (Default)

Simulates normal user behavior:

- Health checks: 25% of requests
- Root endpoint: 12.5% of requests
- Model info: 5% of requests
- Predictions: 50% of requests
- Predictions with top_k: 7.5% of requests

**Usage:**

```bash
locust -f tests/load_test.py --host http://localhost:8000
```

### StressTestUser

More aggressive testing with shorter wait times:

- Rapid predictions with varying image sizes
- Tests system under continuous load
- Wait time: 0.1-0.5 seconds

**Usage:**

```bash
locust -f tests/load_test.py \
    --class-picker \
    -H http://localhost:8000
# Then select "StressTestUser" in the web UI
```

### SpikeTestUser

Simulates sudden traffic spikes:

- Very short wait times (0-0.1 seconds)
- Tests autoscaling and resilience
- Useful for testing Cloud Run scaling

**Usage:**

```bash
locust -f tests/load_test.py \
    --class-picker \
    -H http://localhost:8000
# Select "SpikeTestUser"
```

## Load Test Patterns

### Baseline Performance Test

Establish baseline metrics with moderate load:

```bash
locust -f tests/load_test.py \
    --host http://localhost:8000 \
    --users 10 \
    --spawn-rate 2 \
    --run-time 5m \
    --headless \
    --html reports/baseline_performance.html
```

**Expected Results:**

- Average response time: < 500ms
- 95th percentile: < 1s
- 99th percentile: < 2s
- Error rate: < 1%

### Capacity Testing

Find the maximum sustainable load:

```bash
# Start with low load and gradually increase
locust -f tests/load_test.py \
    --host http://localhost:8000 \
    --users 100 \
    --spawn-rate 10 \
    --run-time 10m \
    --headless
```

Monitor for:

- Response time degradation
- Increasing error rates
- Resource saturation (CPU, memory)

### Stress Testing

Push beyond normal capacity to test failure modes:

```bash
locust -f tests/load_test.py \
    --user-classes StressTestUser \
    --host http://localhost:8000 \
    --users 200 \
    --spawn-rate 20 \
    --run-time 5m \
    --headless
```

### Spike Testing

Simulate sudden traffic spikes:

```bash
locust -f tests/load_test.py \
    --user-classes SpikeTestUser \
    --host http://localhost:8000 \
    --users 500 \
    --spawn-rate 100 \
    --run-time 2m \
    --headless
```

### Soak Testing

Long-duration test to detect memory leaks:

```bash
locust -f tests/load_test.py \
    --host http://localhost:8000 \
    --users 20 \
    --spawn-rate 5 \
    --run-time 2h \
    --headless \
    --html reports/soak_test.html
```

Monitor for:

- Gradual performance degradation
- Memory leaks
- Resource exhaustion

## Testing Cloud Run Deployment

### Prerequisites

1. Deploy API to Cloud Run (see [CLOUD_RUN_DEPLOYMENT.md](CLOUD_RUN_DEPLOYMENT.md))
2. Get service URL
3. Ensure model is loaded

### Run Tests Against Cloud Run

```bash
# Set your Cloud Run URL
export SERVICE_URL="https://fish-classifier-api-xxx.run.app"

# Run moderate load test
locust -f tests/load_test.py \
    --host $SERVICE_URL \
    --users 50 \
    --spawn-rate 10 \
    --run-time 5m \
    --headless \
    --html reports/cloud_run_performance.html
```

### Test Autoscaling

```bash
# Gradually increase load to trigger autoscaling
locust -f tests/load_test.py \
    --host $SERVICE_URL \
    --users 200 \
    --spawn-rate 20 \
    --run-time 10m
```

**Monitor in GCP Console:**

- Instance count increases
- Response times remain stable
- Cold start impact

### Test with Traffic from Multiple Regions

Use Locust's distributed mode:

```bash
# On master machine
locust -f tests/load_test.py \
    --master \
    --expect-workers 3 \
    --host $SERVICE_URL

# On worker machines (different regions)
locust -f tests/load_test.py \
    --worker \
    --master-host <master-ip>
```

## Analyzing Results

### Web UI Metrics

When using the web UI [http://localhost:8089](http://localhost:8089):

- **Charts tab**: Real-time performance graphs
- **Statistics tab**: Per-endpoint metrics
- **Failures tab**: Error details
- **Current ratio**: Request distribution

### Key Metrics

| Metric | Description | Target |
| -------- | ------------- | -------- |
| RPS | Requests per second | Depends on capacity |
| Response Time (avg) | Average response time | < 500ms |
| Response Time (p95) | 95th percentile | < 1s |
| Response Time (p99) | 99th percentile | < 2s |
| Failure Rate | Percentage of failed requests | < 1% |

### Generating Reports

```bash
# Generate HTML report
locust -f tests/load_test.py \
    --host http://localhost:8000 \
    --users 50 \
    --spawn-rate 10 \
    --run-time 5m \
    --headless \
    --html reports/load_test_$(date +%Y%m%d_%H%M%S).html \
    --csv reports/load_test_$(date +%Y%m%d_%H%M%S)
```

This generates:

- `load_test_YYYYMMDD_HHMMSS.html`: Interactive HTML report
- `load_test_YYYYMMDD_HHMMSS_stats.csv`: Per-endpoint statistics
- `load_test_YYYYMMDD_HHMMSS_failures.csv`: Failure details
- `load_test_YYYYMMDD_HHMMSS_stats_history.csv`: Time-series data

## Performance Optimization

### Common Bottlenecks

1. **Model Inference**
   - Solution: Batch predictions, use GPU, optimize model

2. **Image Processing**
   - Solution: Optimize transforms, use efficient image libraries

3. **Cold Starts (Cloud Run)**
   - Solution: Set min instances, optimize startup time

4. **Concurrency Limits**
   - Solution: Increase workers, use async processing

### Optimization Tips

```python
# Example: Add batching support to API
@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile]):
    # Process multiple images in one batch
    images = [load_image(f) for f in files]
    batch_tensor = torch.stack(images)
    predictions = model(batch_tensor)
    return predictions
```

## CI/CD Integration

Add load testing to GitHub Actions:

```yaml
# .github/workflows/load-test.yaml
name: Load Test

on:
  workflow_dispatch:
    inputs:
      target_url:
        description: 'URL to test'
        required: true
      users:
        description: 'Number of users'
        default: '50'
      duration:
        description: 'Test duration'
        default: '5m'

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install dependencies
        run: pip install locust
      
      - name: Run load test
        run: |
          locust -f tests/load_test.py \
            --host ${{ github.event.inputs.target_url }} \
            --users ${{ github.event.inputs.users }} \
            --spawn-rate 10 \
            --run-time ${{ github.event.inputs.duration }} \
            --headless \
            --html load_test_report.html
      
      - name: Upload report
        uses: actions/upload-artifact@v4
        with:
          name: load-test-report
          path: load_test_report.html
```

## Troubleshooting

### Connection Errors

```bash
# Increase timeout
locust -f tests/load_test.py \
    --host http://localhost:8000 \
    --timeout 30
```

### Rate Limiting

If hitting rate limits:

```bash
# Reduce spawn rate
locust -f tests/load_test.py \
    --users 100 \
    --spawn-rate 1  # Slower ramp-up
```

### Memory Issues (Locust)

For large-scale tests:

```bash
# Use distributed mode
locust -f tests/load_test.py \
    --master \
    --expect-workers 4

# On worker machines
locust -f tests/load_test.py --worker --master-host=<master-ip>
```

## Best Practices

1. **Start Small**: Begin with low load and gradually increase
2. **Test Locally First**: Validate tests work before testing production
3. **Monitor Resources**: Watch CPU, memory, and network during tests
4. **Document Baselines**: Record normal performance for comparison
5. **Test Regularly**: Run load tests after major changes
6. **Use Realistic Data**: Test with representative images and patterns
7. **Clean Up**: Remove test artifacts and stop services after testing

## Resources

- [Locust Documentation](https://docs.locust.io/)
- [Load Testing Best Practices](https://docs.locust.io/en/stable/writing-a-locustfile.html)
- [Cloud Run Optimization](https://cloud.google.com/run/docs/tips)
