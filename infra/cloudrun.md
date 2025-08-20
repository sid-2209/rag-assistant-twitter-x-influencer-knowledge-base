# Deploy to Cloud Run (manual)

1) Build and push to Artifact Registry or Docker Hub:
```bash
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT_ID/REPO/rag-assistant:latest .
```

2) Deploy to Cloud Run:
```bash
gcloud run deploy rag-assistant \
  --image REGION-docker.pkg.dev/PROJECT_ID/REPO/rag-assistant:latest \
  --platform managed \
  --allow-unauthenticated \
  --region REGION \
  --port 8000 \
  --set-env-vars=PYTHONUNBUFFERED=1
```

3) Set secrets (optional):
```bash
gcloud run services update rag-assistant \
  --update-secrets=OPENAI_API_KEY=projects/PROJECT_NUM/secrets/OPENAI_API_KEY:latest
```
