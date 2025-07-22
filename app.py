# importing required libraries
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import warnings
import pickle
import uvicorn
from feature import FeatureExtraction
from typing import Optional
import os

warnings.filterwarnings('ignore')

app = FastAPI(title="Phishing URL Detection")

# Mount static files
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Static files serve karne ke liye
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Load the pickle file
try:
    # First try to load from the pickle directory
    pickle_path = os.path.join("pickle", "model.pkl")
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            gbc = pickle.load(f)
    else:
        # If file doesn't exist, train a new model
        from sklearn.ensemble import GradientBoostingClassifier
        # Default model with good parameters
        gbc = GradientBoostingClassifier(max_depth=4, learning_rate=0.7)
        print("WARNING: Using an untrained model as the pickle file was not found!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback to a simple model if there's an error
    from sklearn.ensemble import GradientBoostingClassifier
    gbc = GradientBoostingClassifier(max_depth=4, learning_rate=0.7)

# Root route pe page1.html dikhega
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("page1.html", {"request": request})

# /index route pe index.html dikhega
@app.get("/index", response_class=HTMLResponse)
async def show_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
#get started root
@app.get("/index", response_class=HTMLResponse)
async def show_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, url: str = Form(...)):
    try:
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)
        
        # Get prediction
        y_pred = gbc.predict(x)[0]
        
        # Get probability scores
        try:
            y_proba = gbc.predict_proba(x)[0]
            
            # If the model was trained with -1/1 labels
            if len(y_proba) == 2:
                # Binary classification with probabilities for each class
                if y_pred == 1:  # Safe
                    score = y_proba[1]  # Probability of being safe
                else:  # Unsafe
                    score = 1 - y_proba[0]  # Probability of being unsafe
            else:
                # Fallback if the model output is unexpected
                score = 0.5 if y_pred == 1 else 0.0
        except:
            # Fallback if predict_proba doesn't work
            score = 1.0 if y_pred == 1 else 0.0
        
        print(f"Prediction for {url}: {y_pred}, Score: {score}")
        
        # Return the result
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "xx": float(score),
                "url": url
            }
        )
    except Exception as e:
        import traceback
        print(f"Prediction error: {e}")
        print(traceback.format_exc())
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "xx": -1,
                "error": str(e),
                "url": url
            }
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)




