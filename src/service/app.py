from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
from s3_utils import upload_to_s3
from model_utils import predict_price, retrain_model, load_model_and_dependencies
from pydantic import BaseModel, Field
from typing import Optional

# Загружаем модели при старте приложения
app = FastAPI()

@app.on_event("startup")
def startup_event():
    print("Application startup: loading models...")
    try:
        load_model_and_dependencies()
        print("Models loaded successfully.")
    except Exception as e:
        print(f"WARNING: Could not load models on startup. "
              f"The '/predict' and '/retrain' endpoints may fail. Error: {e}")

class PredictRequest(BaseModel):
    # Поля, которые пользователь должен ввестиs
    marka: str = Field(..., example="Kia")
    model: str = Field(..., example="Rio")
    year: int = Field(..., example=2018)
    engine: float = Field(..., example=1.6)
    mileage: int = Field(..., example=100000)
    power: int = Field(..., example=123)
    color: str = Field(..., example="белый")
    
    # Необязательные поля с значениями по умолчанию
    kompl: Optional[str] = Field("не указано", example="Luxe")
    owners: Optional[int] = Field(1, example=2)
    body_type: Optional[str] = Field("седан", example="седан")
    drive: Optional[str] = Field("передний", example="передний")
    transmission: Optional[str] = Field("АКПП", example="АКПП")
    fuel_type: Optional[str] = Field("бензин", example="бензин")
    wheel: Optional[str] = Field("левый", example="левый")
    generation: Optional[int] = Field(0, example=4)


@app.get("/")
def root():
    return {"message": "Car Market Analysis API (Russia)"}

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        result = predict_price(request)
        return {"prediction": result}
    except FileNotFoundError as e:
         raise HTTPException(status_code=503, detail=f"Model artifacts not found: {e}. Please train the model first via /retrain endpoint.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

@app.post("/update-data")
async def update_data(file: UploadFile = File(...)):
    # Временное хранение файла для загрузки в S3
    temp_dir = "/tmp"
    os.makedirs(temp_dir, exist_ok=True)
    local_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(local_path, "wb") as f:
            f.write(await file.read())
        
        s3_key = f"data/{file.filename}"
        if not upload_to_s3(local_path, s3_key):
            raise HTTPException(status_code=500, detail="Ошибка загрузки файла в S3")
            
        return {"result": "Файл успешно загружен в Data Lake (S3)", "s3_key": s3_key}
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


@app.post("/retrain")
def retrain():
    try:
        retrain_model()
        return {"result": "Модель успешно переобучена и артефакты обновлены в S3"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during retraining: {e}")

