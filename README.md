# Job_Title_Prediction_RandomForest_Model
Job Prediction API Documentation

Overview
This API predicts suitable job titles based on candidate resumes using machine learning. Built with FastAPI, it processes both structured (education details, grades) and unstructured data (skills, interests) to recommend job positions.

Key Features:
Job title prediction based on candidate qualifications
Handles text data (skills/interests/experience) and numerical data (grades)
Returns prediction confidence scores
Comprehensive error handling with HTTP status codes
Built-in data augmentation for better model performance

API Endpoints:
POST /predict
Predicts suitable job titles from candidate data.

Request Format: Requires JSON input with: Gender, UG course details, specialization, certificates, CGPA, interests, skills, and work experience.

Response: Returns predicted job title with confidence score (0-1), status code, and message.

GET /status-codes
Lists all supported HTTP status codes for reference.

Implementation Details:
Data Processing:
Combines text features using TF-IDF vectorization
One-hot encodes categorical variables
Scales numerical features
Uses median imputation for missing values
Machine Learning Model:
Random Forest Classifier with 100 trees
Custom class weights for balanced predictions
Trained on augmented dataset

Error Handling:
The API returns appropriate HTTP codes for:
400: Invalid/missing input fields
422: Unprocessable data
500: Server errors

Usage Example:
Send POST request to /predict with candidate data in JSON format to receive job predictions. The API runs locally on port 8000 by default.

Requirements:
Python 3.7+, FastAPI, scikit-learn, pandas, numpy
