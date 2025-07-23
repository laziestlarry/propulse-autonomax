#!/bin/bash

echo "🚀 Starting AutonomaX Cloud Deploy Script..."

# Load env vars from .env if exists
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Set defaults
PROJECT_ID=propulse-autonomax
REGION=us-central1
SERVICE_NAME=autonomax-service

echo "🧩 Enabling required GCP services..."
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com --project=$PROJECT_ID

echo "📁 Creating Artifact Registry for container builds..."
gcloud artifacts repositories create autonomax-repo --repository-format=docker   --location=$REGION --project=$PROJECT_ID || echo "✅ Repo exists, skipping."

echo "🛠️ Building Docker image..."
gcloud builds submit /Users/pq/__ProPulse_Group__/ProPulseApp_Final_Base \
--tag us-central1-docker.pkg.dev/$PROJECT_ID/autonomax-repo/autonomax-app:v$(date +%s) \
  --project=propulse-autonomax
  
echo "🚢 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME   --image=$REGION-docker.pkg.dev/$PROJECT_ID/autonomax-repo/autonomax-app   --platform=managed   --region=$REGION   --allow-unauthenticated   --set-env-vars "OPENAI_API_KEY=$OPENAI_API_KEY,SHOPIFY_ACCESS_TOKEN=$SHOPIFY_ACCESS_TOKEN,SHOPIFY_API_KEY=$SHOPIFY_API_KEY,SHOPIFY_SECRET=$SHOPIFY_SECRET,SHOP_URL=$SHOP_URL"   --project=$PROJECT_ID

echo "✅ Deployment complete."
gcloud run services describe $SERVICE_NAME --platform=managed --region=$REGION --project=$PROJECT_ID --format="value(status.url)"
