import pickle
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

with open("decision_tree_1.pkl", "rb") as file:
    model = pickle.load(file)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(request: Request,
                  BMI: float = Form(...),
                  Smoking: int = Form(...),
                  AlcoholDrinking: int = Form(...),
                  Stroke: int = Form(...),
                  PhysicalHealth: int = Form(...),
                  MentalHealth: int = Form(...),
                  DiffWalking: int = Form(...),
                  Sex: int = Form(...),
                  AgeCategory: int = Form(...),
                  Race: int = Form(...),
                  Diabetic: int = Form(...),
                  PhysicalActivity: int = Form(...),
                  GenHealth: int = Form(...),
                  SleepTime: float = Form(...),
                  Asthma: int = Form(...),
                  KidneyDisease: int = Form(...),
                  SkinCancer: int = Form(...)):

    features = [BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, DiffWalking, Sex, AgeCategory, Race, Diabetic, PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease, SkinCancer]
    prediction = model.predict([features])[0]

    disease_status = "Heart Disease" if prediction == 1 else "Not Heart Disease"

    return templates.TemplateResponse("index.html", {"request": request, "prediction": disease_status})
