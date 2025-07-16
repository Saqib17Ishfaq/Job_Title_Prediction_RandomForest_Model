# job_predict_api.py

from fastapi import FastAPI, HTTPException, Request, status, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from http import HTTPStatus
import uvicorn

app = FastAPI()

# === Custom Error Responses ===
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": exc.status_code,
            "message": exc.detail,
        },
    )

@app.get("/status-codes")
def list_status_codes():
    return {
        200: "OK",
        201: "Created",
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        408: "Request Timeout",
        409: "Conflict",
        422: "Unprocessable Entity",
        500: "Internal Server Error",
        502: "Bad Gateway",
        504: "Gateway Timeout",
        505: "HTTP Version Not Supported",
        507: "Insufficient Storage",
        508: "Loop Detected"
    }

# === 1. Data Loading ===
df = pd.read_csv('Job Prediction By Resume.csv')

# === 2. Data Preparation ===
num_additional_rows = 160
if df.shape[0] < num_additional_rows + 10:
    for _ in range(num_additional_rows):
        base_row_index = np.random.randint(0, len(df['Name']))
        df_row = {k: df[k][base_row_index] if isinstance(df[k], list) else df[k] for k in df.keys()}
        df_row['Average CGPA or Percentage obtained in under graduation'] = np.random.uniform(6.0, 9.9)
        for text_col in ['Interests', 'Skills', 'Work in the past']:
            if np.random.rand() < 0.1:
                df_row[text_col] = np.nan
            else:
                if isinstance(df[text_col], list):
                    df_row[text_col] = np.random.choice([x for x in df[text_col] if pd.notna(x)])
        df = pd.concat([df, pd.DataFrame([df_row])], ignore_index=True)

# === 3. Feature Engineering ===
target_column_name = 'title job'
X = df.drop(columns=[target_column_name, 'Name'], errors='ignore').copy()
y = df[target_column_name]

categorical_features = ['Gender', 'Course in UG', ' UG specialization? Major Subject (Eg; Mathematics)', 
                       'Certificate course title']
numerical_features = ['Average CGPA or Percentage obtained in under graduation']
text_features = ['Interests', 'Skills', 'Work in the past']

def combine_text_columns(df_input):
    combined = df_input[text_features].fillna('').astype(str).apply(' '.join, axis=1)
    return combined.str.strip()

text_transformer = Pipeline([
    ('combiner', FunctionTransformer(combine_text_columns, validate=False)),
    ('tfidf', TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2)))
])

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numerical_features),
    ('text', text_transformer, text_features)
], remainder='drop')

classes = np.unique(y)
weights = compute_class_weight('balanced', classes=classes, y=y)
class_weights = dict(zip(classes, weights))

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        class_weight=class_weights,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

model_pipeline.fit(X_train, y_train)

# === 4. API Schema ===
class CandidateInput(BaseModel):
    Gender: str
    Course_in_UG: str
    UG_specialization: str
    Certificate_course_title: str
    Average_CGPA: float
    Interests: str
    Skills: str
    Work_in_the_past: str

class PredictionResponse(BaseModel):
    predicted_job: str
    confidence: float
    status_code: int
    status_message: str
    message: str

# === 5. POST Endpoint ===
@app.post("/predict", response_model=PredictionResponse)
def predict(candidate: CandidateInput):
    try:
        # Convert input to dictionary
        input_dict = candidate.dict()

        # Check for any missing or empty string fields
        empty_fields = [field for field, value in input_dict.items()
                        if value is None or (isinstance(value, str) and not value.strip())]

        if empty_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing or empty fields: {', '.join(empty_fields)}"
            )

        # Prepare DataFrame
        input_data = pd.DataFrame([{
            'Gender': candidate.Gender,
            'Course in UG': candidate.Course_in_UG,
            ' UG specialization? Major Subject (Eg; Mathematics)': candidate.UG_specialization,
            'Certificate course title': candidate.Certificate_course_title,
            'Average CGPA or Percentage obtained in under graduation': candidate.Average_CGPA,
            'Interests': candidate.Interests,
            'Skills': candidate.Skills,
            'Work in the past': candidate.Work_in_the_past
        }])

        # Make prediction
        pred = model_pipeline.predict(input_data)[0]
        prob = model_pipeline.predict_proba(input_data)[0].max()

        return PredictionResponse(
            predicted_job=pred,
            confidence=round(prob, 2),
            status_code=HTTPStatus.OK.value,
            status_message=HTTPStatus.OK.phrase,
            message="Prediction successful."
        )

    except ValueError:
        raise HTTPException(status_code=422, detail=HTTPStatus.UNPROCESSABLE_ENTITY.phrase)
    except KeyError:
        raise HTTPException(status_code=400, detail=HTTPStatus.BAD_REQUEST.phrase)
    except HTTPException as e:
        raise e  # re-raise manual exceptions like 400
    except Exception:
        raise HTTPException(status_code=500, detail=HTTPStatus.INTERNAL_SERVER_ERROR.phrase)


# === 6. Run the App ===
if __name__ == "__main__":
    uvicorn.run("job_predict_api:app", host="127.0.0.1", port=8000, reload=True)