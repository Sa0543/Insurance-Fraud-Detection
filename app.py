from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from io import BytesIO
from langchain_huggingface import HuggingFaceEndpoint
import torch
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Fraud Detection + Chatbot API", version="1.0")

# Mount static files for logo, css, etc.
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------- CHATBOT SETUP ----------------
device = 0 if torch.cuda.is_available() else -1

chat_model = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # ✅ lighter and conversational
    task="text-generation",
    temperature=0.7,
    max_new_tokens=200
)

class ChatInput(BaseModel):
    message: str

conversation_history = {"past_user_inputs": [], "generated_responses": []}

@app.post("/chat")
def chat(chat: ChatInput):
    """Insurance chatbot using Hugging Face conversational endpoint."""
    user_message = chat.message

    # Simple conversational-style prompt
    prompt = (
        "You are an insurance domain expert assistant. "
        "Answer clearly, concisely, and professionally.\n\n"
        f"User: {user_message}\nAssistant:"
    )

    try:
        response = chat_model.invoke(prompt)   # ✅ Send plain string
        reply = response if isinstance(response, str) else str(response)
    except Exception as e:
        reply = f"⚠️ Error: {e}"

    return {"reply": reply}


    # """Insurance chatbot using Hugging Face cloud endpoint."""
    # user_message = chat.message
    # prompt = f"You are an insurance expert assistant. Respond clearly and professionally.\n\nUser: {user_message}\nAssistant:"
    # try:
    #     response = chat_model.invoke(prompt)
    #     reply = response.strip()
    # except Exception as e:
    #     reply = f"Error: {e}"
    # return {"reply": reply}


# ---------------- FRAUD DETECTION SETUP ----------------
templates = Jinja2Templates(directory="templates")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

try:
    fraud_model = joblib.load(r"D:\project\Fraud_detection\model.pkl")
    print("✓ Fraud Detection Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading fraud model: {e}")
    fraud_model = None

class ClaimInput(BaseModel):
    PREMIUM_AMOUNT: float
    CLAIM_AMOUNT: float
    CLAIM_TO_PREMIUM_RATIO: float
    DAYS_POLICY_TO_LOSS: int
    DAYS_LOSS_TO_REPORT: int
    FLAG_SHORT_WINDOW: int
    FLAG_LONG_REPORT_DELAY: int
    FLAG_NIGHT_INCIDENT: int
    WEAK_SUSPECT_FLAG: int
    INSURANCE_LINE: str
    INCIDENT_SEVERITY: str
    RISK_SEGMENTATION: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home form page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict-form")
async def predict_from_form(
    request: Request,
    premium_amount: float = Form(...),
    claim_amount: float = Form(...),
    days_policy_to_loss: int = Form(...),
    days_loss_to_report: int = Form(...),
    insurance_line: str = Form(...),
    incident_severity: str = Form(...),
    risk_segmentation: str = Form(...),
    flag_short_window: int = Form(0),
    flag_long_report_delay: int = Form(0),
    flag_night_incident: int = Form(0),
    weak_suspect_flag: int = Form(0)
):
    if fraud_model is None:
        raise HTTPException(status_code=503, detail="Fraud detection model not loaded")

    try:
        claim_to_premium_ratio = claim_amount / premium_amount if premium_amount > 0 else 0

        input_data = pd.DataFrame([{
            'PREMIUM_AMOUNT': premium_amount,
            'CLAIM_AMOUNT': claim_amount,
            'CLAIM_TO_PREMIUM_RATIO': claim_to_premium_ratio,
            'DAYS_POLICY_TO_LOSS': days_policy_to_loss,
            'DAYS_LOSS_TO_REPORT': days_loss_to_report,
            'FLAG_SHORT_WINDOW': flag_short_window,
            'FLAG_LONG_REPORT_DELAY': flag_long_report_delay,
            'FLAG_NIGHT_INCIDENT': flag_night_incident,
            'WEAK_SUSPECT_FLAG': weak_suspect_flag,
            'INSURANCE_LINE': insurance_line,
            'INCIDENT_SEVERITY': incident_severity,
            'RISK_SEGMENTATION': risk_segmentation
        }])

        prediction = fraud_model.predict(input_data)[0]
        probability = fraud_model.predict_proba(input_data)[0]

        result = {
            "fraud_prediction": int(prediction),
            "fraud_label": "⚠️ FRAUDULENT" if prediction == 1 else "✓ NON-FRAUDULENT",
            "fraud_probability": round(float(probability[1]) * 100, 2),
            "risk_level": "High" if probability[1] > 0.7 else "Medium" if probability[1] > 0.4 else "Low"
        }

        return templates.TemplateResponse("result1.html", {
            "request": request,
            "result": result,
            "input_data": input_data.to_dict('records')[0]
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-csv")
async def predict_from_csv(request: Request, file: UploadFile = File(...)):
    if fraud_model is None:
        raise HTTPException(status_code=503, detail="Fraud detection model not loaded")

    try:
        contents = await file.read()
        buffer = BytesIO(contents)
        df = pd.read_csv(buffer)
        buffer.close()

        predictions = fraud_model.predict(df)
        probabilities = fraud_model.predict_proba(df)

        df['FRAUD_PREDICTION'] = predictions
        df['FRAUD_PROBABILITY'] = [round(prob[1] * 100, 2) for prob in probabilities]
        df['FRAUD_LABEL'] = ['FRAUDULENT' if p == 1 else 'NON-FRAUDULENT' for p in predictions]

        results = df.to_dict('records')

        return templates.TemplateResponse("csv_results.html", {
            "request": request,
            "results": results,
            "total": len(results),
            "fraudulent": sum(predictions),
            "non_fraudulent": len(predictions) - sum(predictions)
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")


@app.post("/api/predict")
def predict_api(claim: ClaimInput):
    if fraud_model is None:
        raise HTTPException(status_code=503, detail="Fraud detection model not loaded")

    try:
        input_data = pd.DataFrame([claim.dict()])
        prediction = fraud_model.predict(input_data)[0]
        probability = fraud_model.predict_proba(input_data)[0]

        return {
            "fraud_prediction": int(prediction),
            "fraud_label": "Fraudulent" if prediction == 1 else "Non-Fraudulent",
            "fraud_probability": float(probability[1]),
            "risk_level": "High" if probability[1] > 0.7 else "Medium" if probability[1] > 0.4 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/chat-widget", response_class=HTMLResponse)
async def chat_widget(request: Request):
    """Floating chatbot interface with logo"""
    return templates.TemplateResponse("chat_widget.html", {"request": request})
