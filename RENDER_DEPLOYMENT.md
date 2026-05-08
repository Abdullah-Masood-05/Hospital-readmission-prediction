# 🚀 Deploy to Render

This guide explains how to deploy the Hospital Readmission Prediction app to Render.

## Prerequisites

1. **GitHub Account** - Render deploys from GitHub
2. **Render Account** - Free at https://render.com
3. **Your code pushed to GitHub**

## Step 1: Prepare for Deployment

Your app is already configured with:
- ✅ `render.yaml` - Render deployment configuration
- ✅ `requirements.txt` - All Python dependencies (updated)
- ✅ `.streamlit/config.toml` - Streamlit settings
- ✅ `app/app.py` - Main Streamlit application
- ✅ `data/preprocessed/` - Preprocessed datasets
- ✅ `results/` - Fairness metrics

## Step 2: Push to GitHub

```bash
# Stage all changes
git add .

# Commit with a deployment message
git commit -m "Prepare for Render deployment

- Updated requirements.txt with latest versions
- Added render.yaml deployment configuration
- Updated .streamlit/config.toml for production
- Configured environment variables for Render"

# Push to GitHub
git push origin master
```

## Step 3: Create Render Account & Link GitHub

1. Go to https://render.com
2. Sign up (free)
3. Click "Sign up with GitHub"
4. Authorize Render to access your GitHub repos

## Step 4: Deploy Your App

### Option A: Using render.yaml (Recommended)

1. Go to https://render.com/dashboard
2. Click **"New +"** → **"Web Service"**
3. Select your GitHub repository
4. Render will automatically detect `render.yaml`
5. Confirm the settings:
   - **Service name:** `hospital-readmission-predictor`
   - **Runtime:** Python 3.11
   - **Start command:** `streamlit run app/app.py --server.port=8501 --server.address=0.0.0.0`
6. Click **"Deploy"**

### Option B: Manual Configuration

If render.yaml isn't detected:

1. Click **"New +"** → **"Web Service"**
2. Select your GitHub repo
3. Fill in:
   - **Name:** `hospital-readmission-predictor`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app/app.py --server.port=8501 --server.address=0.0.0.0`
4. Add Environment Variables (optional):
   ```
   PYTHONUNBUFFERED=true
   STREAMLIT_SERVER_HEADLESS=true
   ```
5. Click **"Deploy"**

## Step 5: Wait for Deployment

- Build typically takes 3-5 minutes
- You'll see logs updating in real-time
- Once complete, you'll get a public URL like: `https://hospital-readmission-predictor.onrender.com`

## Step 6: Test Your App

1. Click the URL provided by Render
2. Test the prediction form:
   - Fill in patient demographics
   - Click "Predict Readmission Risk"
   - Verify the risk score displays correctly
   - Check fairness metrics in sidebar

## Troubleshooting

### 🔴 Build Fails with "Module not found"

**Solution:** Check `requirements.txt` has all packages

```bash
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update dependencies"
git push origin master
```

### 🔴 App Crashes After Deploy

**Check logs:**
1. Go to Render dashboard
2. Select your service
3. Click "Logs" tab
4. Look for error messages

**Common issues:**
- Missing data files: Ensure `data/preprocessed/` is committed to GitHub
- Python version: Verify Python 3.9+ is used
- Memory: Default Render free tier has 512MB - usually enough

### 🔴 Slow Loading

First load takes longer (cold start):
- Render free tier goes to sleep after 15 min inactivity
- First request wakes it up (~30 seconds)
- **Upgrade to Paid to avoid** (starts at $7/month)

### 🔴 Data Not Loading

Ensure data files are in git:

```bash
cd data/preprocessed
ls  # Verify files are here

# If files missing, add them
git add data/
git commit -m "Add preprocessed data"
git push
```

## Performance Tips

1. **First Load**: Takes 30-60 seconds (model loading + SHAP initialization)
2. **Predictions**: Subsequent predictions ~2-5 seconds
3. **Upgrade Plan**: Use Render Pro ($7+/month) to avoid cold starts

## Monitoring

Once deployed:

1. Go to Render dashboard
2. Select your service
3. View:
   - **Logs**: Real-time app output
   - **Metrics**: CPU, memory, requests
   - **Deploys**: Deployment history

## Update Your App

After making changes:

```bash
# Make your changes
git add .
git commit -m "Your change description"
git push origin master
```

Render will automatically rebuild and redeploy! 🚀

## Important Notes

### Data Size
- Your app uses ~4.7GB of preprocessed data
- **Render free tier limitation**: 512MB disk space
- **Solution**: Upgrade to paid tier (~$7/month gets 1GB)

### If Data Too Large

Compress data or use cloud storage:

```bash
# Option 1: Compress (if < 512MB after compression)
tar -czf data.tar.gz data/

# Option 2: Use external storage (S3, Google Drive)
# Add download logic to app.py
```

## Pricing

| Plan | Cost | Features |
|------|------|----------|
| **Free** | $0 | Public URL, auto-sleep after 15 min |
| **Starter** | $7/month | 512MB RAM, 1GB storage, no auto-sleep |
| **Professional** | $12+/month | 2GB RAM, 5GB storage, priority support |

## Support

- **Render Docs**: https://render.com/docs
- **Render Community**: https://render.com/community
- **Streamlit Docs**: https://docs.streamlit.io

---

## Quick Deploy Checklist

- [ ] Code pushed to GitHub
- [ ] `requirements.txt` updated
- [ ] `render.yaml` present in repo root
- [ ] Data files committed to git
- [ ] `.streamlit/config.toml` configured
- [ ] GitHub connected to Render
- [ ] Deploy triggered
- [ ] App loads and works
- [ ] Fairness metrics display correctly
- [ ] Predictions working

🎉 **Your app is now live on Render!**
