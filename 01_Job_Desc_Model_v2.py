import re
import docx
import spacy
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy model 
nlp = spacy.load("en_core_web_sm")

# Load the pre-trained NLP model 
vectorizer = TfidfVectorizer(stop_words='english')

# Define regex-based classification for the job titles
def classify_with_regex(description):
    description = description.lower()

    if re.search(r'\b(data scientist|machine learning|deep learning|artificial intelligence|AI)\b', description) and not re.search(r'\bdata analyst\b', description):
        return "Data Scientist"
    elif re.search(r'\b(data analyst|business intelligence|reporting|excel|tableau|power bi|sql)\b', description) and not re.search(r'\bdata scientist\b', description):
        return "Data Analyst"
    else:
        return "Ambiguous"

# Load the dataset containing job descriptions
file_path = "01. Job Descriptions.xlsx" 
df = pd.read_excel(file_path)

#job descriptions are in a column named 'description'
df = df[['description']].dropna()

# Apply the regex classification
df['Regex_Prediction'] = df['description'].apply(classify_with_regex)

# Assign labels for training the NLP model
df['Label'] = df['Regex_Prediction'].map({'Data Scientist': 0, 'Data Analyst': 1, 'Ambiguous': 2})

# Train NLP Model
X_train, X_test, y_train, y_test = train_test_split(df[df['Label'] != 2]['description'],
                                                    df[df['Label'] != 2]['Label'],
                                                    test_size=0.2, random_state=42)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Process document and extract information
def extract_text_from_docx(file_path):
    """Extracts text from a DOCX file."""
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def extract_named_entities(text):
    """Extracts named entities using spaCy."""
    doc = nlp(text)

    extracted_info = {
        "title": "N/A",
        "company": "N/A",
        "location": "N/A",
        "seniority_level": "Senior" if "senior" in text.lower() else "N/A",
        "employment_type_full_contract": "N/A",
        "job_function": "N/A",
        "industries": "N/A",
        "required_degree": "N/A",
        "job_benefits": "N/A",
        "required_skills": "N/A",
        "recruiter_details": "N/A",
        "job_url": "N/A",
        "description": text[:500] 
    }

    for ent in doc.ents:
        if ent.label_ in ["ORG"]:
            extracted_info["company"] = ent.text
        elif ent.label_ in ["GPE"]:
            extracted_info["location"] = ent.text
        elif ent.label_ in ["PERSON"]:
            extracted_info["recruiter_details"] = ent.text
        elif ent.label_ in ["MONEY"]:
            extracted_info["job_benefits"] = ent.text
        elif ent.label_ in ["EDUCATION"]:
            extracted_info["required_degree"] = ent.text

    return extracted_info

# Define known categories for classification
employment_types = ["Full-Time", "Part-Time", "Contract", "Internship"]
job_functions = ["Data Science", "Data Analysis", "Machine Learning", "Software Engineering"]
industries = ["Technology", "Finance", "Healthcare", "Retail"]
required_skills_list = ["Python", "SQL", "Machine Learning", "Deep Learning", "Big Data"]
job_benefits_list = ["401k", "Health Insurance", "Remote Work", "Stock Options"]

# Process the document
doc_file_path = "Data_Scientist_Job_Desc.docx"
text = extract_text_from_docx(doc_file_path)
extracted_info = extract_named_entities(text)

# Predict the job title using regex
extracted_info["Regex_Prediction"] = classify_with_regex(extracted_info["description"])

# Use the pre-trained NLP model for ambiguous cases (if needed)
if extracted_info["Regex_Prediction"] == "Ambiguous":
    # Apply the NLP model to refine the job title prediction
    ambiguous_description = extracted_info["description"]
    prediction = model.predict(vectorizer.transform([ambiguous_description]))[0]
    if prediction == 0:
        extracted_info["Final_Prediction"] = "Data Scientist"
    elif prediction == 1:
        extracted_info["Final_Prediction"] = "Data Analyst"
else:
    extracted_info["Final_Prediction"] = extracted_info["Regex_Prediction"]

# Display the extracted and predicted information
extracted_info["description"] = text  # Add the full text for context
df_final = pd.DataFrame([extracted_info])
print(df_final[['description', 'Regex_Prediction', 'Final_Prediction']])

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)