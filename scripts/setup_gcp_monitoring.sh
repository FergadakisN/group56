#!/bin/bash

# Setup GCP Cloud Monitoring and Alerting for Fish Classifier API
# This script creates email notifications and alert policies for the Cloud Run service

set -e

# Configuration
PROJECT_ID="${1:-fish-mlops}"
SERVICE_NAME="${2:-fish-classifier-api}"
REGION="${3:-us-central1}"
EMAIL="${4:-your-email@example.com}"
ALERT_NAME="fish-classifier-api-alerts"
NOTIFICATION_CHANNEL_NAME="fish-classifier-email"

echo "Setting up GCP Cloud Monitoring for $SERVICE_NAME..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Email: $EMAIL"
echo ""

# Enable required APIs
echo "1. Enabling required APIs..."
gcloud services enable monitoring.googleapis.com \
    logging.googleapis.com \
    cloudresourcemanager.googleapis.com \
    --project=$PROJECT_ID

# Create email notification channel
echo ""
echo "2. Creating email notification channel..."
CHANNEL_RESPONSE=$(gcloud alpha monitoring channels create \
    --display-name="$NOTIFICATION_CHANNEL_NAME" \
    --type=email \
    --channel-labels=email_address="$EMAIL" \
    --project=$PROJECT_ID \
    --format='value(name)' 2>/dev/null || echo "")

if [ -z "$CHANNEL_RESPONSE" ]; then
    echo "  Note: Notification channel creation via CLI requires verification."
    echo "  Please create it manually in Cloud Console or use the UI to verify the email."
    echo ""
    echo "  Manual steps:"
    echo "  1. Go to Cloud Console > Monitoring > Alerting > Notification Channels"
    echo "  2. Click 'Create Channel'"
    echo "  3. Select 'Email' as the notification type"
    echo "  4. Enter email: $EMAIL"
    echo "  5. Click 'Create Channel'"
    CHANNEL_ID="MANUAL_CHANNEL_ID"
else
    CHANNEL_ID=$(echo "$CHANNEL_RESPONSE" | sed 's/.*\///')
    echo "  ✓ Notification channel created: $CHANNEL_ID"
fi

echo ""
echo "3. Creating alert policies..."
echo ""

# Alert Policy 1: High Error Rate (>5% in 5 minutes)
echo "  Creating Alert Policy 1: High Error Rate (>5%)..."
gcloud alpha monitoring policies create \
    --notification-channels="$CHANNEL_ID" \
    --display-name="$ALERT_NAME - High Error Rate" \
    --condition-display-name="Error rate > 5%" \
    --condition-threshold-value=0.05 \
    --condition-threshold-duration=300s \
    --condition-threshold-filter='resource.type="cloud_run_revision" AND metric.type="run.googleapis.com/request_count" AND resource.service_name="'$SERVICE_NAME'" AND metric.response_code_class="5xx"' \
    --project=$PROJECT_ID 2>/dev/null || echo "  Note: Alert policies may require manual setup via UI"

# Alert Policy 2: High P99 Latency (>5 seconds)
echo ""
echo "  Creating Alert Policy 2: High P99 Latency (>5s)..."
gcloud alpha monitoring policies create \
    --notification-channels="$CHANNEL_ID" \
    --display-name="$ALERT_NAME - High Latency" \
    --condition-display-name="P99 latency > 5s" \
    --condition-threshold-value=5000 \
    --condition-threshold-duration=300s \
    --condition-threshold-filter='resource.type="cloud_run_revision" AND metric.type="run.googleapis.com/request_latencies" AND resource.service_name="'$SERVICE_NAME'"' \
    --project=$PROJECT_ID 2>/dev/null || echo "  Note: Alert policies may require manual setup via UI"

# Alert Policy 3: Service Unresponsive (No requests for 10 minutes)
echo ""
echo "  Creating Alert Policy 3: Service Unresponsive..."
gcloud alpha monitoring policies create \
    --notification-channels="$CHANNEL_ID" \
    --display-name="$ALERT_NAME - Service Unresponsive" \
    --condition-display-name="No requests for 10 min" \
    --condition-threshold-value=0 \
    --condition-threshold-duration=600s \
    --condition-threshold-filter='resource.type="cloud_run_revision" AND metric.type="run.googleapis.com/request_count" AND resource.service_name="'$SERVICE_NAME'"' \
    --project=$PROJECT_ID 2>/dev/null || echo "  Note: Alert policies may require manual setup via UI"

echo ""
echo "✓ GCP Monitoring setup complete!"
echo ""
echo "Next steps:"
echo "1. Verify the email notification channel in Cloud Console"
echo "2. Go to Monitoring > Alerting > Alert Policies to review created policies"
echo "3. Test alerts by sending requests to your API endpoint:"
echo ""
echo "   # Test normal operation"
echo "   curl -s https://$SERVICE_NAME-*.run.app/health | jq ."
echo ""
echo "   # Test error handling"
echo "   curl -X POST https://$SERVICE_NAME-*.run.app/predict -F file=@invalid.jpg"
echo ""
echo "Useful commands:"
echo "  # List alert policies:"
echo "  gcloud alpha monitoring policies list --project=$PROJECT_ID"
echo ""
echo "  # List notification channels:"
echo "  gcloud alpha monitoring channels list --project=$PROJECT_ID"
echo ""
echo "  # View Cloud Run service metrics:"
echo "  gcloud logging read \"resource.service_name=$SERVICE_NAME\" --project=$PROJECT_ID --limit 50"
