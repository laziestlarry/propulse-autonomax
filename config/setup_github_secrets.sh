#!/bin/bash

echo "🔐 Adding GitHub secrets to repo: propulse-autonomax"

gh secret set OPENAI_API_KEY -b"$OPENAI_API_KEY" -R laziestlarry/propulse-autonomax
gh secret set SHOPIFY_ACCESS_TOKEN -b"$SHOPIFY_ACCESS_TOKEN" -R laziestlarry/propulse-autonomax
gh secret set SHOPIFY_STORE_URL -b"$SHOPIFY_STORE_URL" -R laziestlarry/propulse-autonomax

echo "✅ GitHub secrets added successfully."
