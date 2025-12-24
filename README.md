**Fraud Detection Application 🚨**

This project is an Insurance Fraud Detection system built using Machine Learning and deployed as a FastAPI web application.
It allows users to predict whether an insurance claim is fraudulent or non-fraudulent using form input, CSV upload, or API requests.
It also includes a chatbot interface for insurance-related queries.

**🔹 Features**

Fraud prediction using a trained Machine Learning model

Web interface for manual claim input

Bulk prediction using CSV upload

REST API endpoint for predictions

Chatbot powered by Hugging Face LLM

HTML templates with FastAPI + Jinja2

Clean and modular project structure

**🔹 Tech Stack**

Backend: FastAPI

Machine Learning: Scikit-learn

Data Handling: Pandas, NumPy

Frontend: HTML, Jinja2, CSS

Model Serving: Joblib

Chatbot: Hugging Face (LangChain)

Environment Management: Python, dotenv

**🔹 Project Structure**
├── app.py                     # FastAPI application
├── fd.py                      # Fraud detection ML pipeline
├── fraud_detection.ipynb      # Model development notebook
├── requirements.txt           # Dependencies
├── templates/                 # HTML templates
│   ├── index.html
│   ├── result1.html
│   ├── result2.html
│   ├── chat_widget.html
│   └── csv_r.html
├── static/
│   └── logo.png
├── .gitignore
└── README.md

**🔹 How It Works**

User enters claim details or uploads a CSV file

Input data is processed and sent to the trained ML model

The model predicts:

Fraud / Non-Fraud

Fraud probability

Risk level (Low / Medium / High)

Results are displayed on the web UI or returned via API

**🔹 Installation & Setup**
1️⃣ Clone the repository
git clone <repo-url>
cd <repo-name>

2️⃣ Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add environment variables

Create a .env file:

HUGGINGFACEHUB_API_TOKEN=your_token_here

**🔹 Run the Application**
uvicorn app:app --reload


**Open in browser:**

http://127.0.0.1:8000

**🔹 API Endpoints**
Endpoint	Method	Description
/	GET	Home page
/predict-form	POST	Fraud prediction via form
/predict-csv	POST	Fraud prediction via CSV
/api/predict	POST	JSON API prediction
/chat	POST	Insurance chatbot
/chat-widget	GET	Chat UI
