# 🚀 Deployment Guide

This guide will help you deploy RAG Assistant online so it's accessible to anyone.

## Quick Deploy Options

### 1. **Render (Recommended - Free & Easy)**

**Steps:**
1. Go to [render.com](https://render.com) and sign up
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `rag-assistant`
   - **Environment**: `Docker`
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Root Directory**: Leave empty
   - **Build Command**: Leave empty (uses Dockerfile)
   - **Start Command**: `uvicorn app.api:app --host 0.0.0.0 --port 10000`

5. **Environment Variables** (add these):
   ```
   PYTHONUNBUFFERED=1
   ENABLE_WEB_UI=true
   VECTOR_TOP_K=3
   USE_FAISS=auto
   OPENAI_API_KEY=your_openai_key_here (optional)
   ```

6. Click "Create Web Service"
7. Wait for deployment (5-10 minutes)
8. Your app will be live at: `https://rag-assistant.onrender.com`

**Features:**
- ✅ Free tier available
- ✅ Automatic deployments from GitHub
- ✅ Custom domain support
- ✅ SSL certificate included
- ✅ Persistent storage

### 2. **Railway (Alternative - Fast & Reliable)**

**Steps:**
1. Go to [railway.app](https://railway.app) and sign up
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway will automatically detect the `railway.json` configuration
5. **Environment Variables** (add these):
   ```
   ENABLE_WEB_UI=true
   PYTHONUNBUFFERED=1
   VECTOR_TOP_K=3
   USE_FAISS=auto
   OPENAI_API_KEY=your_openai_key_here (optional)
   ```
6. Click "Deploy Now"
7. Your app will be live at: `https://your-project-name.railway.app`

**Features:**
- ✅ Fast deployments
- ✅ Automatic scaling
- ✅ Custom domains
- ✅ SSL included
- ✅ Good free tier

### 3. **Google Cloud Run (Production Ready)**

**Prerequisites:**
- Google Cloud account
- `gcloud` CLI installed

**Steps:**
1. **Enable APIs:**
   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   ```

2. **Build and deploy:**
   ```bash
   # Build the image
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/rag-assistant .
   
   # Deploy to Cloud Run
   gcloud run deploy rag-assistant \
     --image gcr.io/YOUR_PROJECT_ID/rag-assistant \
     --platform managed \
     --allow-unauthenticated \
     --region us-central1 \
     --port 8000 \
     --set-env-vars=ENABLE_WEB_UI=true,PYTHONUNBUFFERED=1
   ```

3. Your app will be live at the provided URL

**Features:**
- ✅ Pay-per-use pricing
- ✅ Auto-scaling
- ✅ Global CDN
- ✅ Enterprise features

### 4. **Heroku (Classic Option)**

**Steps:**
1. Install Heroku CLI
2. Create `Procfile` in root:
   ```
   web: uvicorn app.api:app --host 0.0.0.0 --port $PORT
   ```
3. Deploy:
   ```bash
   heroku create your-app-name
   heroku config:set ENABLE_WEB_UI=true
   heroku config:set PYTHONUNBUFFERED=1
   git push heroku main
   ```

## Environment Variables

Set these environment variables in your hosting platform:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENABLE_WEB_UI` | Enable HTML web interface | `true` | No |
| `PYTHONUNBUFFERED` | Python output buffering | `1` | No |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | - | No |
| `VECTOR_TOP_K` | Number of results to retrieve | `3` | No |
| `USE_FAISS` | Use FAISS for vector search | `auto` | No |
| `VECTOR_BACKEND` | Vector store backend | `faiss` | No |

## Post-Deployment

### 1. **Test Your Deployment**
- Visit your app URL
- Check the health endpoint: `https://your-app.com/healthz`
- Test the web UI: `https://your-app.com/ui`

### 2. **Upload Sample Data**
- Go to the web UI
- Upload a JSON/CSV file with influencer data
- Test querying the data

### 3. **Custom Domain (Optional)**
- **Render**: Settings → Custom Domain
- **Railway**: Settings → Domains
- **Cloud Run**: Domain mappings

### 4. **Monitoring**
- Check logs in your hosting platform
- Monitor health checks
- Set up alerts if needed

## Troubleshooting

### Common Issues:

1. **App won't start:**
   - Check environment variables
   - Verify Dockerfile is correct
   - Check logs for errors

2. **Web UI not loading:**
   - Ensure `ENABLE_WEB_UI=true`
   - Check static files are included

3. **Vector store issues:**
   - Verify models directory exists
   - Check file permissions

4. **API errors:**
   - Verify OpenAI API key if using
   - Check network connectivity

### Getting Help:
- Check the logs in your hosting platform
- Review the `/healthz` endpoint
- Test locally first: `uvicorn app.api:app --reload`

## Cost Estimation

| Platform | Free Tier | Paid Plans |
|----------|-----------|------------|
| **Render** | $0/month | $7/month+ |
| **Railway** | $5/month | $20/month+ |
| **Cloud Run** | $0/month | Pay-per-use |
| **Heroku** | $0/month | $7/month+ |

## Security Notes

- Set `OPENAI_API_KEY` as a secret/environment variable
- Use HTTPS in production
- Consider rate limiting for public deployments
- Monitor usage and costs

---

**Recommended for most users:** Start with **Render** - it's free, easy, and reliable for getting started quickly!
