#!/bin/bash

echo "🔐 Creating secrets in Google Cloud Secret Manager..."

echo -n "$OPENAI_API_KEY" | gcloud secrets create OPENAI_API_KEY --data-file=- --quiet
echo -n "$SHOPIFY_ACCESS_TOKEN" | gcloud secrets create SHOPIFY_ACCESS_TOKEN --data-file=- --quiet
echo -n "$SHOPIFY_STORE_URL" | gcloud secrets create SHOPIFY_STORE_URL --data-file=- --quiet

PROJECT_ID=$(gcloud config get-value project)
SERVICE_ACCOUNT="$PROJECT_ID@appspot.gserviceaccount.com"

echo "🔐 Granting Cloud Run access to secrets..."

gcloud secrets add-iam-policy-binding OPENAI_API_KEY \
  --member="serviceAccount:$SERVICE_ACCOUNT" --role="roles/secretmanager.secretAccessor" --quiet

gcloud secrets add-iam-policy-binding SHOPIFY_ACCESS_TOKEN \
  --member="serviceAccount:$SERVICE_ACCOUNT" --role="roles/secretmanager.secretAccessor" --quiet

gcloud secrets add-iam-policy-binding SHOPIFY_STORE_URL \
  --member="serviceAccount:$SERVICE_ACCOUNT" --role="roles/secretmanager.secretAccessor" --quiet

echo "✅ GCP secrets and access policies configured."
