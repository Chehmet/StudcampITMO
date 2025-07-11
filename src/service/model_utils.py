import joblib
import os
import pandas as pd
import numpy as np
from datetime import datetime
from s3_utils import download_from_s3, upload_to_s3
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re

# Пути к артефактам
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "rus_cars_best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler_rus.pkl")
LE_MAKE_PATH = os.path.join(MODELS_DIR, "le_make_rus.pkl")
LE_MODEL_PATH = os.path.join(MODELS_DIR, "le_model_rus.pkl")
LE_KOMPL_PATH = os.path.join(MODELS_DIR, "le_kompl_rus.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "features_rus.pkl")

# Убедимся, что директория для моделей существует
os.makedirs(MODELS_DIR, exist_ok=True)

def preprocess_input(data: dict, le_make, le_model, le_kompl, features):
    df = pd.DataFrame([data])
    
    # Базовые и вычисляемые признаки
    df['Год выпуска'] = df['year']
    df['Мощность'] = df['power']
    df['Пробег'] = df['mileage']
    df['Владельцы'] = df['owners']
    df['Возраст'] = 2025 - df['year']
    df['Объем двигателя'] = df['engine']

    # Кодирование
    df['Марка_код'] = le_make.transform(df['marka'])
    df['Модель_код'] = le_model.transform(df['model'])
    df['Комплектация'] = le_kompl.transform(df['kompl'])
    
    # Признаки из даты (для предсказания берем текущую дату)
    now = datetime.now()
    df['год'] = now.year
    df['месяц'] = now.month
    df['день'] = now.day
    df['день_недели'] = now.weekday()
    df['квартал'] = (now.month - 1) // 3 + 1
    df['день_года'] = now.timetuple().tm_yday
    df['викенд'] = int(now.weekday() >= 5)
    df['сезон'] = (now.month % 12 // 3 + 1)
    df['дней_с_публикации'] = 0

    # Дополнительные вычисляемые признаки
    df['Поколение_номер'] = df.get('generation', 0)
    df['Мощность/Объем'] = df['Мощность'] / df['Объем двигателя'].replace(0, 1)
    df['Новый'] = (df['Пробег'] < 1000).astype(int)

    # One-Hot Encoding
    for col_prefix, value in [
        ('Тип кузова', data.get('body_type')),
        ('Привод', data.get('drive')),
        ('Тип КПП', data.get('transmission')),
        ('Тип топлива', data.get('fuel_type')),
        ('Руль', data.get('wheel')),
        ('Цвет', data.get('color'))
    ]:
        if value:
            column_name = f"{col_prefix}_{value}"
            if column_name in features:
                df[column_name] = 1

    # Убедимся, что все признаки из списка `features` присутствуют в DataFrame
    for feature in features:
        if feature not in df.columns:
            df[feature] = 0

    # Возвращаем DataFrame с колонками в правильном порядке
    return df[features]


def ensure_artifacts():
    s3_keys = {
        MODEL_PATH: 'rus_cars_best_model.pkl',
        SCALER_PATH: 'scaler_rus.pkl',
        LE_MAKE_PATH: 'le_make_rus.pkl',
        LE_MODEL_PATH: 'le_model_rus.pkl',
        LE_KOMPL_PATH: 'le_kompl_rus.pkl',
        FEATURES_PATH: 'features_rus.pkl'
    }
    for local_path, s3_key in s3_keys.items():
        if not os.path.exists(local_path):
            if not download_from_s3(s3_key, local_path):
                raise FileNotFoundError(f"Failed to download {s3_key} from S3. Please ensure artifacts are uploaded.")

def load_model_and_dependencies():
    ensure_artifacts()
    dependencies = {
        "model": joblib.load(MODEL_PATH),
        "scaler": joblib.load(SCALER_PATH),
        "le_make": joblib.load(LE_MAKE_PATH),
        "le_model": joblib.load(LE_MODEL_PATH),
        "le_kompl": joblib.load(LE_KOMPL_PATH),
        "features": joblib.load(FEATURES_PATH)
    }
    return dependencies

def predict_price(input_data):
    deps = load_model_and_dependencies()
    df_processed = preprocess_input(
        input_data.dict(), deps['le_make'], deps['le_model'], deps['le_kompl'], deps['features']
    )
    X_scaled = deps['scaler'].transform(df_processed)
    prediction = deps['model'].predict(X_scaled)
    return float(prediction[0])

def retrain_model():
    # 1. Скачиваем все json-файлы из S3 (data/*.json)
    import boto3
    S3_BUCKET = os.getenv("S3_BUCKET")
    S3_ENDPOINT = os.getenv("S3_ENDPOINT")
    S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
    S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
    
    session = boto3.session.Session()
    s3 = session.client(
        service_name='s3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY
    )
    
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="data/")
    if 'Contents' not in response:
        raise FileNotFoundError("No data files found in S3 bucket under 'data/' prefix.")
        
    files = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.json')]
    all_data = []
    
    # Создаем временную папку для скачиваемых данных
    temp_data_dir = "temp_data"
    os.makedirs(temp_data_dir, exist_ok=True)

    for key in files:
        local_path = os.path.join(temp_data_dir, os.path.basename(key))
        download_from_s3(key, local_path)
        with open(local_path, 'r', encoding='utf-8') as f:
            all_data.extend(json.load(f))
        os.remove(local_path)
    os.rmdir(temp_data_dir)
    
    df = pd.DataFrame(all_data)

    # 2. Препроцессинг (полностью повторяем логику из ноутбука)
    df = df.dropna(subset=["Марка", "Модель", "Год выпуска", "Цена", "Пробег", "Мощность"])
    df["Тип кузова"] = df["Тип кузова"].fillna("не указано")
    df["Комплектация"] = df["Комплектация"].fillna("не указано")
    df["Владельцы"] = df["Владельцы"].fillna("не указано")

    def clean_price(price_str):
        if pd.isna(price_str): return np.nan
        return int(str(price_str).replace(" ", "").replace("\xa0", "").replace("₽", ""))

    def clean_mileage(mileage_str):
        if pd.isna(mileage_str): return np.nan
        cleaned = ''.join(c for c in str(mileage_str) if c.isdigit())
        return int(cleaned) if cleaned else 0

    def clean_power(power_str):
        if pd.isna(power_str): return np.nan
        match = re.search(r'(\d+)', str(power_str))
        return int(match.group(1)) if match else np.nan
    
    df["Цена"] = df["Цена"].apply(clean_price)
    df["Пробег"] = df["Пробег"].apply(clean_mileage)
    df["Мощность"] = df["Мощность"].apply(clean_power)
    df['Год выпуска'] = pd.to_numeric(df['Год выпуска'], errors='coerce').fillna(0).astype(int)
    
    df["Возраст"] = 2025 - df["Год выпуска"]
    df["Объем двигателя"] = np.nan
    df["Тип топлива"] = np.nan

    for idx, engine in df["Двигатель"].items():
        if pd.isna(engine) or engine == "None": continue
        parts = [p.strip() for p in str(engine).split(",")]
        try:
            if len(parts) >= 2 and "л" in parts[1]:
                df.loc[idx, "Объем двигателя"] = float(parts[1].replace(" л", ""))
                df.loc[idx, "Тип топлива"] = parts[0]
            elif len(parts) >= 2 and ("гибрид" in parts[1].lower() or "электро" in parts[1].lower()):
                df.loc[idx, "Объем двигателя"] = float(parts[0].replace(" л", ""))
                df.loc[idx, "Тип топлива"] = "гибрид" if "гибрид" in parts[1].lower() else "электро"
            elif "л" in str(engine):
                df.loc[idx, "Объем двигателя"] = float(str(engine).replace(" л", ""))
                df.loc[idx, "Тип топлива"] = "не указано"
        except (ValueError, IndexError):
            pass

    df["Тип КПП"] = df["Коробка передач"].str.extract(r"(АКПП|МКПП|вариатор|робот)", flags=re.IGNORECASE)
    df['Владельцы'] = df['Владельцы'].replace({'1': 1, '2': 2, '3': 3, '4 и более': 4, 'не указано': -1}).astype(int)

    le_make = LabelEncoder()
    le_model = LabelEncoder()
    le_kompl = LabelEncoder()
    df['Марка_код'] = le_make.fit_transform(df['Марка'])
    df['Модель_код'] = le_model.fit_transform(df['Модель'])
    df['Комплектация'] = le_kompl.fit_transform(df['Комплектация'])
    
    cat_cols = ['Тип кузова', 'Привод', 'Тип КПП', 'Тип топлива', 'Руль', 'Цвет']
    for col in cat_cols:
        df[col] = df[col].fillna('не указано')
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    df['Поколение_номер'] = df['Поколение'].str.extract(r'(\d+)').fillna(0).astype(int)
    df['Мощность/Объем'] = df['Мощность'] / df['Объем двигателя'].replace(0, 1)
    df['Новый'] = (df['Пробег'] < 1000).astype(int)

    def preprocess_dates(df, date_col):
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df = df.dropna(subset=[date_col])
        df['год'] = df[date_col].dt.year
        df['месяц'] = df[date_col].dt.month
        df['день'] = df[date_col].dt.day
        df['день_недели'] = df[date_col].dt.dayofweek
        df['квартал'] = df[date_col].dt.quarter
        df['день_года'] = df[date_col].dt.dayofyear
        df['викенд'] = (df[date_col].dt.dayofweek >= 5).astype(int)
        df['сезон'] = (df[date_col].dt.month % 12 // 3 + 1)
        current_date = datetime.now()
        df['дней_с_публикации'] = (current_date - df[date_col]).dt.days
        return df.drop(columns=[date_col])

    df = preprocess_dates(df, 'Дата публикации')
    
    features = ['Год выпуска', 'Мощность', 'Пробег', 'Владельцы', 'Комплектация',
       'Возраст', 'Объем двигателя', 'Марка_код', 'Модель_код',
       'Тип кузова_джип/suv 5 дв.', 'Тип кузова_купе', 'Тип кузова_лифтбек',
       'Тип кузова_минивэн', 'Тип кузова_не указано', 'Тип кузова_открытый',
       'Тип кузова_седан', 'Тип кузова_универсал', 'Тип кузова_хэтчбек 3 дв.',
       'Тип кузова_хэтчбек 5 дв.', 'Привод_задний', 'Привод_передний',
       'Тип КПП_вариатор', 'Тип КПП_робот', 'Тип топлива_гибрид',
       'Тип топлива_дизель', 'Тип топлива_не указано', 'Руль_правый',
       'Цвет_белый', 'Цвет_бордовый', 'Цвет_голубой', 'Цвет_желтый',
       'Цвет_зеленый', 'Цвет_золотистый', 'Цвет_коричневый', 'Цвет_красный',
       'Цвет_оранжевый', 'Цвет_розовый', 'Цвет_серебристый', 'Цвет_серый',
       'Цвет_синий', 'Цвет_фиолетовый', 'Цвет_черный', 'Поколение_номер',
       'Мощность/Объем', 'Новый', 'год', 'месяц', 'день', 'день_недели',
       'квартал', 'день_года', 'викенд', 'сезон', 'дней_с_публикации']
       
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df.dropna(subset=['Цена'] + features)

    target = 'Цена'
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 3. Обучение модели
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='r2',
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    
    # 4. Сохраняем артефакты локально
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le_make, LE_MAKE_PATH)
    joblib.dump(le_model, LE_MODEL_PATH)
    joblib.dump(le_kompl, LE_KOMPL_PATH)
    joblib.dump(features, FEATURES_PATH)

    # 5. Загружаем артефакты в S3
    s3_keys = {
        MODEL_PATH: 'rus_cars_best_model.pkl',
        SCALER_PATH: 'scaler_rus.pkl',
        LE_MAKE_PATH: 'le_make_rus.pkl',
        LE_MODEL_PATH: 'le_model_rus.pkl',
        LE_KOMPL_PATH: 'le_kompl_rus.pkl',
        FEATURES_PATH: 'features_rus.pkl'
    }
    for local_path, s3_key in s3_keys.items():
        upload_to_s3(local_path, s3_key)

    return True

    