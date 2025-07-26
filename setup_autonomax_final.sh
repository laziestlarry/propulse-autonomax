#!/bin/bash

# Initial ProPulse setup tasks here

# ------------------------------------------
# ✅ SMART AUTONOMAX FINAL SETUP ACTIVATION
# ------------------------------------------

echo "🔐 Loading environment variables from .env if available..."
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✅ .env loaded"
else
    echo "⚠️ .env file not found, make sure to configure manually or use GCP secrets"
fi

echo "🐳 Building Docker Image..."
docker build -t gcr.io/propulse-autonomax/autonomax-dashboard .

echo "📤 Pushing Docker Image to Google Cloud Registry..."
docker push gcr.io/propulse-autonomax/autonomax-dashboard

echo "🚀 Deploying to Cloud Run (autonomax-service)..."
gcloud run deploy autonomax-service \
  --image gcr.io/propulse-autonomax/autonomax-dashboard \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=$OPENAI_API_KEY,SHOPIFY_ACCESS_TOKEN=$SHOPIFY_ACCESS_TOKEN,SHOPIFY_STORE_URL=$SHOPIFY_STORE_URL

echo "✅ Deployment complete. Visit your app at:"
gcloud run services describe autonomax-service --platform managed --region us-central1 --format="value(status.url)"

echo "🧠 ProPulse AutonomaX stack fully activated."
