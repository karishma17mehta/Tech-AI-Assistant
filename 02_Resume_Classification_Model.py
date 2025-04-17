import re
import nltk
import fitz
import pickle
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#!pip install pymupdf

# Function to clean text data and remove numbers
def clean_column_text(text):
    if not isinstance(text, str):
        return ""

    # Remove newline characters and extra spaces
    text = text.replace("\n", " ").strip()

    # Remove common prefixes like "Top Skills\n", "Experience\n", etc.
    text = re.sub(r"^(Contact|Top Skills|Languages|Certifications|Honors-Awards|Experience|Education)\s*", "", text, flags=re.IGNORECASE)

    # Remove brackets and their content (if any)
    text = re.sub(r"\[.*?\]|\(.*?\)|\{.*?\}", "", text)

    # Remove bullets, dashes, and special characters
    text = re.sub(r"•|-|\*", " ", text)

    # Remove punctuations
    text = re.sub(r"[^\w\s]", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

data = pd.read_csv("/Job_Description_Master_File.csv")
df = pd.DataFrame(data)

# Apply cleaning to all columns
df_cleaned = df.copy()
for col in df_cleaned.columns:
    df_cleaned[col] = df_cleaned[col].astype(str).apply(clean_column_text)

# Function to tokenize text using basic split
def tokenize_text_simple(text):
    if not isinstance(text, str):
        return []
    return text.lower().split()  # Basic whitespace-based tokenization

# Apply tokenization to each column except "Final_Category_Rule_Based" and "Final_Category_Actual"
df_tokenized = df_cleaned.copy()
columns_to_exclude = ["Final_Category_Rule_Based", "Final_Category_Actual"]

for col in df_tokenized.columns:
    if col not in columns_to_exclude:  # Exclude specific columns from tokenization
        df_tokenized[col] = df_tokenized[col].astype(str).apply(tokenize_text_simple)

# Download stopwords if not already available
nltk.download("stopwords")
nltk_stopwords = set(stopwords.words("english"))

# Manually define additional unwanted words
custom_stopwords = set([
    "skills", "experience", "certifications", "education", "years", "work", "working", "based",
    "learning", "knowledge", "ability", "strong", "including", "field", "industry", "role",
    "position", "year", "month", "team", "working", "requirement", "various", "related",
    "san", "francisco", "delhi", "california", "york", "new", "pennsylvania", "syracuse",
    "sacramento", "july","china","bloomington","los","bombay","hyderabad","map","students",
    "indore","september","prof","angeles","shenyang","philadelphia","arlington","santa","clara",
    "stony","brook","pradesh","iit","united","states","france","worked", "working","hours","india","ohio","pune", 
    "york","yale","page", "riverside","university","youth","young","masters","ms","accenture","meta","amazon","google"
])

# Combine NLTK and custom stopwords
all_stopwords = nltk_stopwords.union(custom_stopwords)

# Function to remove stopwords from a tokenized list
def remove_stopwords(token_list):
    if not isinstance(token_list, list):
        return []
    return [word for word in token_list if word.lower() not in all_stopwords]

# Apply stopword removal to each column except "Final_Category_Rule_Based" and "Final_Category_Actual"
df_no_stopwords = df_tokenized.copy()
columns_to_exclude = ["Final_Category_Rule_Based", "Final_Category_Actual"]

for col in df_no_stopwords.columns:
    if col not in columns_to_exclude:  # Exclude specific columns from stopword removal
        df_no_stopwords[col] = df_no_stopwords[col].apply(remove_stopwords)

df_no_stopwords["Combined_Text"] = df_no_stopwords["Experience"].astype(str) + " " + df_no_stopwords["Top Skills"].astype(str) + " " + df_no_stopwords["Certifications"].astype(str) + " " + df_no_stopwords["Education"].astype(str)

# Convert tokenized resumes back to text format for TF-IDF
df_text = df_no_stopwords["Combined_Text"].apply(lambda tokens: "".join(tokens))

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=3000)  

# Transform text data into TF-IDF vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(df_text)

# Convert to DataFrame for visualization
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())


valid_columns = [word for word in tfidf_vectorizer.get_feature_names_out() if not word.isdigit()]

# Filter out numeric-only columns
df_tfidf_clean = df_tfidf[valid_columns]

# Convert "Final_Category_Actual" into categorical format (numerical encoding)
label_encoder = LabelEncoder()
df_no_stopwords["Final_Category_Actual_Label"] = label_encoder.fit_transform(df_no_stopwords["Final_Category_Actual"])

# Store label mapping for reference
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Encoding Mapping:", label_mapping)


# TRAINING TEST SPLIT STRATIFIED

# Define features (TF-IDF vectors) and labels (encoded categories)
X = df_tfidf  # Ensure TF-IDF matrix is prepared
y = df_no_stopwords["Final_Category_Actual_Label"]  # Use encoded labels

# Perform stratified train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Display the split dataset details
split_info = {
    "Training Samples": len(X_train),
    "Testing Samples": len(X_test),
    "Total Samples": len(df_tfidf)
}

split_df = pd.DataFrame([split_info])

# TRAINING USING MULTINOMIAL LOGISTIC REGRESSION

# Initialize Logistic Regression for multi-class classification
log_reg = LogisticRegression(multi_class="multinomial",C=10, solver="lbfgs", max_iter=500, class_weight="balanced")

# Train the model on the training set
log_reg.fit(X_train, y_train)

training_status = {"Model": "Logistic Regression", "Training Status": "Completed"}
df_training_status = pd.DataFrame([training_status])

# Define file paths for saving
model_path = "logistic_regression_model.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"
encoder_path = "label_encoder.pkl"

# Save the trained Logistic Regression model
with open(model_path, 'wb') as model_file:
    pickle.dump(log_reg, model_file)

# Save the TF-IDF vectorizer
with open(vectorizer_path, 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

# Save the Label Encoder
with open(encoder_path, 'wb') as encoder_file:
    pickle.dump(label_encoder, encoder_file)

# Predict on the training set
y_train_pred = log_reg.predict(X_train)

# Evaluate accuracy on the training set
train_accuracy = accuracy_score(y_train, y_train_pred)
#print(f"Training Accuracy: {train_accuracy:.4f}")

# Generate classification report for training set
train_classification_report = classification_report(y_train, y_train_pred, target_names=label_encoder.classes_)
#print("Training Classification Report:\n", train_classification_report)

#Checking for Feature Importance (TF-IDF Weights in Logistic Regression)

# Get feature importance (absolute values of coefficients)
feature_importance = np.abs(log_reg.coef_)

# Get top 20 words for each category
top_n = 20
for i, category in enumerate(label_encoder.classes_):
    top_features_idx = np.argsort(feature_importance[i])[-top_n:]
    top_features = [tfidf_vectorizer.get_feature_names_out()[j] for j in top_features_idx]
    print(f"\nTop {top_n} Important Words for Category '{category}':")
    print(", ".join(top_features))

# Count the number of training samples per class
train_class_distribution = pd.Series(y_train).value_counts()
#print("Training Set Class Distribution:\n", train_class_distribution)

# List of unwanted words (e.g., country and city names)
unwanted_words = {"san", "francisco", "delhi", "california", "york", "new", "pennsylvania", "syracuse","sacramento", "july","china",
                 "bloomington","los","bombay","hyderabad","map","students","indore","september","prof","angeles","shenyang",
                  "philadelphia","arlington","santa","clara","stony","brook","pradesh","iit","united","states","france","worked",
                  "working","hours","india","ohio","pune"}

# Get feature importance (absolute values of coefficients)
feature_importance = np.abs(log_reg.coef_)

# Get top 20 words for each category after filtering out unwanted words
top_n = 20
filtered_top_words = {}

for i, category in enumerate(label_encoder.classes_):
    top_features_idx = np.argsort(feature_importance[i])[-(top_n + len(unwanted_words)):]  # Increase range to compensate for removed words
    top_features = [tfidf_vectorizer.get_feature_names_out()[j] for j in top_features_idx]
    
    # Filter out unwanted words
    filtered_features = [word for word in top_features if word not in unwanted_words][:top_n]

    filtered_top_words[category] = filtered_features


filtered_top_words

for category, words in filtered_top_words.items():
    print(f"\nTop 20 Important Words for '{category}':")
    print(", ".join(words))

#MODEL EVALUATION

# Predict on test data
y_test_pred = log_reg.predict(X_test)

# Evaluate accuracy on the test set
test_accuracy = accuracy_score(y_test, y_test_pred)

# Generate classification report for the test set
test_classification_report = classification_report(y_test, y_test_pred, target_names=label_encoder.classes_)

# Compute confusion matrix
conf_matrix_test = confusion_matrix(y_test, y_test_pred)

# Display test accuracy
test_metrics = {"Test Accuracy": test_accuracy}

df_test_metrics=pd.DataFrame([test_metrics])
#print(df_test_metrics)

# Print classification report
#print("Test Classification Report:\n", test_classification_report)

# Plot confusion matrix for test set
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Set Confusion Matrix")
plt.show()

# Define parameter grid for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'max_iter': [500, 1000, 2000],  # Maximum iterations
    'solver': ['lbfgs', 'saga']  # Different solvers for optimization
}

# Initialize GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',  # Optimize for accuracy
    verbose=1,  # Show progress
    n_jobs=-1  # Use all available cores for faster computation
)

# Run grid search on the training data
grid_search.fit(X_train, y_train)

# Get the best parameters from grid search
best_params = grid_search.best_params_
best_score = grid_search.best_score_

tuning_results = {
    "Best Regularization (C)": best_params['C'],
    "Best Max Iterations": best_params['max_iter'],
    "Best Solver": best_params['solver'],
    "Best Cross-Validation Accuracy": best_score
}

tuning_df=pd.DataFrame([tuning_results])
#print(tuning_df)

# ## PDF CONVERSION and USER INPUT

# Define keywords for early screening
data_scientist_keywords = {"data scientist", "machine learning engineer", "ml engineer"}
data_analytics_keywords = {"data analyst", "business analyst"}

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    """Extracts text content from a given PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"  # Extract text from each page
    return text

# Function to clean and preprocess text
def preprocess_resume_text(text):
    text = text.lower()
    
    # Remove special characters, numbers, and extra spaces
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text) 
    text = re.sub(r"\s+", " ", text).strip() 

    # Tokenization
    tokens = text.split()

    # Remove stopwords
    tokens = [word for word in tokens if word not in all_stopwords]

    return " ".join(tokens) 

# Load the saved model
with open("logistic_regression_model.pkl", 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Load the saved TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

# Load the saved Label Encoder
with open("label_encoder.pkl", 'rb') as encoder_file:
    loaded_encoder = pickle.load(encoder_file)


# Function to classify a resume using the saved model
def classify_resume_with_screening(pdf_path):
    """First checks for keywords in Experience, then classifies with the saved model if needed."""
    
    # Step 1: Extract text from the PDF
    raw_text = extract_text_from_pdf(pdf_path)

    # Step 2: Early Screening - Check Experience Section for Keywords
    experience_section = raw_text.lower()  # Convert to lowercase for easy matching

    # Directly classify based on keywords
    if any(keyword in experience_section for keyword in data_scientist_keywords):
        return "Data Scientist"
    elif any(keyword in experience_section for keyword in data_analytics_keywords):
        return "Data Analytics"
    
    # Step 3: If no direct classification, preprocess and classify using the loaded model
    cleaned_text = preprocess_resume_text(raw_text)
    resume_tfidf = loaded_vectorizer.transform([cleaned_text])  # Use loaded TF-IDF vectorizer
    prediction = loaded_model.predict(resume_tfidf)  # Use loaded Logistic Regression model
    
    # Convert numerical prediction back to category name
    predicted_category = loaded_encoder.inverse_transform(prediction)[0]

    return predicted_category  # Return the final category prediction


pdf_resume_path = "Data_Scientist_Masters_Resume.pdf"  # Replace with actual file path
predicted_category = classify_resume_with_screening(pdf_resume_path)
print(f"Predicted Category: {predicted_category}")

