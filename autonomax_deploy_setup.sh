#!/bin/bash
# === AutonomaX Push + Deploy + Frontend Integration Script ===

set -e

# Project-specific setup
PROJECT_ID="propulse-autonomax"
REPO_NAME="autonomax-repo"
SERVICE_NAME="autonomax-service"
REGION="us-central1"
GITHUB_REPO="https://github.com/laziestlarry/propulse-autonomax.git"

# === Step 1: Push to GitHub ===
echo "\n🚀 Committing stable release to GitHub..."
git init || true
git remote add origin "$GITHUB_REPO" || true
git add .
git commit -m "Stable deployment: full-stack app with OpenAI + Shopify"
git branch -M main
git push -u origin main --force

# === Step 2: Configure Cloud Build + Artifact Registry ===
echo "\n🔧 Configuring Cloud Build + Docker registry..."
gcloud config set project "$PROJECT_ID"
gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com run.googleapis.com

gcloud artifacts repositories create "$REPO_NAME" \
  --repository-format=docker \
  --location="$REGION" \
  --description="Docker repo for AutonomaX App" || echo "✅ Repo exists"

# === Step 3: Build Docker image ===
echo "\n🐳 Building Docker image..."
gcloud builds submit --tag "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/autonomax-app"

# === Step 4: Deploy to Cloud Run ===
echo "\n🚢 Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/autonomax-app" \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated

# === Step 5: Frontend Install Scaffold (React) ===
echo "\n🎨 Creating React frontend scaffold..."
npx create-react-app autonomax-ui
cd autonomax-ui
echo "REACT_APP_BACKEND_URL=https://autonomax-service-$PROJECT_ID.a.run.app" > .env

# Placeholder component
mkdir -p src/components
cat <<EOF > src/components/ChatbotDemo.js
import React, { useState } from 'react';
export default function ChatbotDemo() {
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");

  const askBot = async () => {
    const res = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/chatbot`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    });
    const data = await res.json();
    setResponse(data.response);
  };

  return (
    <div>
      <h2>Chatbot Demo</h2>
      <input value={prompt} onChange={e => setPrompt(e.target.value)} />
      <button onClick={askBot}>Ask</button>
      <pre>{response}</pre>
    </div>
  );
}
EOF

# Update App.js
cat <<EOF > src/App.js
import React from 'react';
import ChatbotDemo from './components/ChatbotDemo';
function App() {
  return (
    <div className="App">
      <h1>AutonomaX Demo UI</h1>
      <ChatbotDemo />
    </div>
  );
}
export default App;
EOF

npm install && npm start
