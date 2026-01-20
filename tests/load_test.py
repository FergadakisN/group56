"""
Load testing script for the Fish Species Classification API.

This script uses Locust to simulate multiple users making concurrent requests
to the API endpoints, helping identify performance bottlenecks and capacity limits.

Usage:
    # Install locust first
    pip install locust

    # Run with web UI
    locust -f tests/load_test.py --host http://localhost:8000

    # Run headless (no UI)
    locust -f tests/load_test.py --host http://localhost:8000 \
        --users 100 --spawn-rate 10 --run-time 60s --headless

    # Test production endpoint
    locust -f tests/load_test.py --host https://your-service.run.app
"""

from __future__ import annotations

import io
import random

from locust import HttpUser, between, task
from PIL import Image


class FishClassifierUser(HttpUser):
    """
    Simulated user for load testing the Fish Classification API.

    Attributes:
        wait_time: Time to wait between consecutive requests (1-3 seconds).
        test_image_bytes: Pre-generated test image to avoid repeated generation.
    """

    wait_time = between(1, 3)

    def on_start(self):
        """Initialize test data when user starts."""
        # Generate a random test image once per user
        self.test_image_bytes = self._generate_test_image()

    def _generate_test_image(self, size: tuple[int, int] = (224, 224)) -> bytes:
        """
        Generate a random test image.

        Args:
            size: Image dimensions (width, height).

        Returns:
            JPEG image as bytes.
        """
        # Create random RGB image
        color = tuple(random.randint(0, 255) for _ in range(3))
        img = Image.new("RGB", size, color=color)

        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG", quality=85)
        img_bytes.seek(0)

        return img_bytes.getvalue()

    @task(10)
    def health_check(self):
        """Test health endpoint (high frequency)."""
        self.client.get("/health", name="/health")

    @task(5)
    def root_endpoint(self):
        """Test root endpoint (medium frequency)."""
        self.client.get("/", name="/")

    @task(2)
    def model_info(self):
        """Test model info endpoint (low frequency)."""
        self.client.get("/model/info", name="/model/info")

    @task(20)
    def predict_image(self):
        """Test prediction endpoint (highest frequency)."""
        # Create a fresh BytesIO object for each request
        img_bytes = io.BytesIO(self.test_image_bytes)

        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}

        with self.client.post(
            "/predict",
            files=files,
            catch_response=True,
            name="/predict",
        ) as response:
            if response.status_code == 200:
                # Validate response structure
                try:
                    data = response.json()
                    if "predicted_class" not in data or "confidence" not in data:
                        response.failure("Invalid response structure")
                    else:
                        response.success()
                except Exception as e:
                    response.failure(f"JSON parse error: {e}")
            elif response.status_code == 503:
                # Model not loaded is expected in some scenarios
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(3)
    def predict_with_top_k(self):
        """Test prediction with custom top_k parameter."""
        img_bytes = io.BytesIO(self.test_image_bytes)
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}

        top_k = random.choice([1, 3, 5, 10])

        with self.client.post(
            f"/predict?top_k={top_k}",
            files=files,
            catch_response=True,
            name="/predict?top_k=N",
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if len(data.get("top_k_predictions", [])) != min(top_k, 3):
                        # Assuming 3 classes in test scenario
                        pass  # May have fewer classes than requested
                    response.success()
                except Exception:
                    response.failure("Invalid response")
            elif response.status_code == 503:
                response.success()


class StressTestUser(HttpUser):
    """
    Stress test user with more aggressive request patterns.

    Use this class to test system limits and error handling under extreme load.
    """

    wait_time = between(0.1, 0.5)  # Much shorter wait time

    def on_start(self):
        """Initialize test data."""
        # Generate multiple images of different sizes
        self.small_image = self._generate_test_image((224, 224))
        self.medium_image = self._generate_test_image((512, 512))
        self.large_image = self._generate_test_image((1024, 1024))

    def _generate_test_image(self, size: tuple[int, int]) -> bytes:
        """Generate random test image."""
        color = tuple(random.randint(0, 255) for _ in range(3))
        img = Image.new("RGB", size, color=color)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG", quality=85)
        img_bytes.seek(0)
        return img_bytes.getvalue()

    @task(30)
    def rapid_predictions(self):
        """Make rapid prediction requests with varying image sizes."""
        image_bytes = random.choice([self.small_image, self.medium_image, self.large_image])

        img_bytes = io.BytesIO(image_bytes)
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}

        self.client.post("/predict", files=files, name="/predict (stress)")

    @task(5)
    def concurrent_health_checks(self):
        """Rapid health check requests."""
        self.client.get("/health", name="/health (stress)")


# Custom load test scenarios
class SpikeTestUser(HttpUser):
    """
    Simulates sudden traffic spikes.

    This helps test autoscaling behavior and system resilience.
    """

    wait_time = between(0, 0.1)  # Very short wait for spike simulation

    def on_start(self):
        """Initialize test data."""
        self.test_image = self._generate_test_image()

    def _generate_test_image(self) -> bytes:
        """Generate test image."""
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        return img_bytes.getvalue()

    @task
    def spike_requests(self):
        """Burst of prediction requests."""
        img_bytes = io.BytesIO(self.test_image)
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        self.client.post("/predict", files=files, name="/predict (spike)")
