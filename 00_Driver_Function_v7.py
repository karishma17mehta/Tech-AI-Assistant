import re
import docx
import fitz
import time
import spacy
import random
import pickle
import pandas as pd
from PIL import Image
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from DOCX
def extract_text_from_docx(file_path):
    """Extracts text from a DOCX file."""
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

# Function to classify using regex
def classify_with_regex(description):
    description = description.lower()
    
    if re.search(r'\b(data scientist|machine learning|deep learning|artificial intelligence|AI)\b', description) and not re.search(r'\bdata analyst\b', description):
        return "Data Scientist"
    elif re.search(r'\b(data analyst|business intelligence|reporting|excel|tableau|power bi|sql)\b', description) and not re.search(r'\bdata scientist\b', description):
        return "Data Analyst"
    else:
        return "Ambiguous"

# Function to process job description and return prediction
def predict_job_title_jd(doc_file_path):
    """Processes a job description DOCX file and returns the predicted job title, seniority"""

    # Load the trained model
    with open('JD_Identifier_Model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Load the same vectorizer used for training
    JD_vectorizer = TfidfVectorizer(stop_words='english')

    text = extract_text_from_docx(doc_file_path)
    
    # Apply regex classification
    regex_prediction = classify_with_regex(text)

    # Seniority Classification via keyword matching
    description_section = text.lower()  
    if "junior" in description_section:
        seniority = "Junior"
    elif "mid" in description_section or "intermediate" in description_section:
        seniority = "Mid"
    elif "senior" in description_section:
        seniority = "Senior"
    else:
        seniority = "Mid"  # Default assumption if unclear
    
    # Use the NLP model only if classification is ambiguous
    if regex_prediction == "Ambiguous":
        prediction = model.predict(JD_vectorizer.transform([text]))[0]
        final_prediction = "Data Scientist" if prediction == 0 else "Data Analyst"
    else:
        final_prediction = regex_prediction

    return [final_prediction, seniority]

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    """Extracts text content from a given PDF file."""
    doc = fitz.open(stream=pdf_path.read(), filetype="pdf")  # Use stream instead of path
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
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

    all_stopwords = {"the", "is", "in", "and", "to", "of", "for", "with", "on", "at", "a", "an", "this", "that", "by", "it"}

    # Remove stopwords
    tokens = [word for word in tokens if word not in all_stopwords]

    return " ".join(tokens)

# Function to classify a resume using the saved model
def classify_resume(pdf_path):
    """First checks for keywords in Experience, then classifies with the saved model if needed."""

    # Define keywords for early screening
    data_scientist_keywords = {"data scientist", "machine learning engineer", "ml engineer"}
    data_analytics_keywords = {"data analyst", "business analyst"}
    
    # Step 1: Extract text from the PDF
    raw_text = extract_text_from_pdf(pdf_path)

    # Step 2: Early Screening - Check Experience Section for Keywords
    experience_section = raw_text.lower()  # Convert to lowercase for easy matching

    # Seniority Classification via keyword matching
    experience_section = raw_text.lower()  
    if "junior" in experience_section:
        seniority = "Junior"
    elif "mid" in experience_section or "intermediate" in experience_section:
        seniority = "Mid"
    elif "senior" in experience_section:
        seniority = "Senior"
    else:
        seniority = "Mid"  # Default assumption if unclear

    # Directly classify based on keywords
    if any(keyword in experience_section for keyword in data_scientist_keywords):
        return ["Data Scientist", seniority]
    elif any(keyword in experience_section for keyword in data_analytics_keywords):
        return ["Data Analyst", seniority]
    
    with open("Resume_logistic_regression_model.pkl", "rb") as model_file:
        loaded_model = pickle.load(model_file)
    with open("Resume_tfidf_vectorizer.pkl", "rb") as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)
    with open("Resume_label_encoder.pkl", "rb") as encoder_file:
        loaded_encoder = pickle.load(encoder_file)

    # Step 3: If no direct classification, preprocess and classify using the loaded model
    cleaned_text = preprocess_resume_text(raw_text)
    resume_tfidf = loaded_vectorizer.transform([cleaned_text])  # Use loaded TF-IDF vectorizer
    prediction = loaded_model.predict(resume_tfidf)  # Use loaded Logistic Regression model
    
    # Convert numerical prediction back to category name
    predicted_category = loaded_encoder.inverse_transform(prediction)[0]

    return [predicted_category, seniority]  # Return the final category prediction

def process_files(job_desc, resume):
    """Process JD and Resume, match role, and fetch relevant questions."""
    
    # Step 1: Classify JD
    job_desc_category, job_desc_seniority = predict_job_title_jd(job_desc)

    # Step 2: Classify Resume
    resume_result = classify_resume(resume)
    resume_category, resume_seniority = resume_result[0], resume_result[1]

    #resume_category, resume_seniority = classify_resume(resume)

    return resume_category, resume_seniority, job_desc_category, job_desc_seniority

def fetch_questions(resume_role, resume_seniority, jd_role, jd_seniority):
    """Fetches 5 questions based on Role & Seniority filtering logic, with fallback to General DA (Mid)."""

    # Load the Excel file and ensure correct column naming
    df = pd.read_excel("Questions_Library.xlsx", header=1)
    df = df.rename(columns={"Unnamed: 2": "Role", "Unnamed: 3": "Question", "Unnamed: 5": "Seniority"})

    # Flow 1: Determine Role-Based Questions
    if resume_role == jd_role:
        selected_roles = [jd_role]  # Only JD Role
    else:
        selected_roles = [resume_role, jd_role]  # Mix of both roles

    # Flow 2: Determine Seniority-Based Questions
    filtered_df = df[df["Role"].isin(selected_roles)]
    filtered_df = filtered_df[filtered_df["Seniority"].isin([resume_seniority, jd_seniority])]

    # Fetch Resume & JD Role Questions (handling cases where fewer than expected exist)
    resume_filtered = filtered_df[filtered_df["Role"] == resume_role]
    jd_filtered = filtered_df[filtered_df["Role"] == jd_role]

    resume_questions = (
        resume_filtered.sample(n=min(2, len(resume_filtered)))["Question"].tolist()
        if not resume_filtered.empty
        else []
    )

    jd_questions = (
        jd_filtered.sample(n=min(3, len(jd_filtered)))["Question"].tolist()
        if not jd_filtered.empty
        else []
    )

    # Combine selected questions
    questions = resume_questions + jd_questions

    # Fallback: If fewer than 5 questions, add 'General DA' (Mid) level questions
    if len(questions) < 5:
        st.warning("Not enough role-specific questions. Adding General DA (Mid) questions.")
        general_questions = df[(df["Role"] == "General DA") & (df["Seniority"] == "Mid")]

        extra_questions = general_questions.sample(n=min(5 - len(questions), len(general_questions)))["Question"].tolist()
        questions.extend(extra_questions)

    return questions

def generate_summary(score):
    return f"You performed well with an overall score of {score}%. Keep improving on explaining projects with clarity!"

questions_df = pd.read_excel("Questions_Library.xlsx")
questions_df = questions_df.rename(columns={"Unnamed: 2": "Role", "Unnamed: 3": "Question", 
                                            "Unnamed: 4": "Answer", "Unnamed: 5": "Seniority"})

def get_expected_answer(question):
    """Fetch expected answer from the Questions Library DataFrame."""
    expected_answer_row = questions_df[questions_df["Question"] == question]
    if not expected_answer_row.empty:
        return expected_answer_row["Answer"].values[0]  
    return "No expected answer available"

def check_similarity(expected, user):
    """Compare expected answer and user answer using TF-IDF cosine similarity."""
    if not expected or not user:
        return 0  
    
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([expected, user])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    
    return round(similarity_score * 100, 2)  

def calculate_score(similarity):
    """Assigns a score based on similarity percentage."""
    if similarity > 80:
        return 10
    elif similarity > 60:
        return 8
    elif similarity > 40:
        return 6
    elif similarity > 20:
        return 4
    else:
        return 2

def generate_final_report():
    """Generates final summary report with evaluation and timing."""
    st.header("Final Interview Report")
    
    if "responses" not in st.session_state or not st.session_state.responses:
        st.error("No responses found.")
        return

    evaluation_data = []
    timing_data = []

    for idx, (question, user_answer, time_taken) in enumerate(st.session_state.responses):
        expected_answer = get_expected_answer(question)
        similarity_score = check_similarity(expected_answer, user_answer)
        final_score = calculate_score(similarity_score)

        evaluation_data.append([question, expected_answer, user_answer, f"{similarity_score}%", final_score])
        timing_data.append([question, f"{time_taken} seconds"])

    # Table 1: Answer Evaluation
    st.subheader("Answer Evaluation")
    eval_df = pd.DataFrame(evaluation_data, columns=["Question", "Expected Answer", "User's Answer", "Similarity", "Score"])
    st.dataframe(eval_df)

    # Table 2: Timing Report
    st.subheader("Timing Report")
    time_df = pd.DataFrame(timing_data, columns=["Question", "Time Taken"])
    st.dataframe(time_df)

    # Display Final Score
    total_score = sum([row[4] for row in evaluation_data])  
    max_possible_score = len(evaluation_data) * 10  
    final_percentage = round((total_score / max_possible_score) * 100, 2)

    st.metric(label="Final Performance Score", value=f"{final_percentage}%")
    st.write(f"Your final performance score is {final_percentage}%. Keep improving on explaining projects with clarity!")

def interview_ui():
    """Handles the interactive interview UI with timers and navigation."""
    
    if "question_index" not in st.session_state:
        st.session_state.question_index = 0
        st.session_state.start_time = time.time()
        st.session_state.responses = []
        st.session_state.show_report = False  

    if "questions" not in st.session_state:
        st.error("Error: No questions available. Please restart the interview.")
        return  

    questions = st.session_state.questions  
    total_questions = len(questions)
    current_index = st.session_state.question_index

    if current_index < total_questions:
        st.subheader(f"Question {current_index + 1} of {total_questions}")
        st.write("**Please provide a structured response to the following question:**")
        st.write(questions[current_index])
        
        user_answer = st.text_area(
            "Type your answer here...",
            key=f"answer_{current_index}",
            placeholder="Write a clear and concise response..."
        )

        if st.button("Next Question"):
            end_time = time.time()
            time_taken = round(end_time - st.session_state.start_time, 2)
            
            st.session_state.responses.append((questions[current_index], user_answer, time_taken))

            st.success(f"Response recorded. Time taken: {time_taken} seconds.")

            st.session_state.question_index += 1
            st.session_state.start_time = time.time()
            st.rerun()

    else:
        st.header("Interview Completed")
        st.write("Your responses have been submitted successfully. Generating performance report...")

        if st.button("Generate Report"):
            st.session_state.show_report = True  
            st.session_state.question_index = len(st.session_state.questions)  
            st.rerun()  

        if st.session_state.show_report:
            generate_final_report()
            return  

def main():
    st.title("AI Interview Assistant")

    st.markdown("""
    **Instructions:**
    - Upload your **Job Description (PDF)** to help the system determine the interview scope.
    - Upload your **Resume (PDF)** to align the interview questions with your experience level.
    - Supported formats: **PDF only**.
    """)
    
    st.header("Step 1: Upload Job Description & Resume")
    job_desc = st.file_uploader("Upload Job Description (DOCX)", type=["docx"])
    resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    
    if job_desc and resume:
        resume_category, resume_seniority, job_desc_category, job_desc_seniority = process_files(job_desc, resume)
        st.success(f"Resume Role: {resume_category}, Seniority: {resume_seniority}")
        st.success(f"Job Description Role: {job_desc_category}, Seniority: {job_desc_seniority}")
    else:
        st.warning("Please upload both Job Description and Resume.")
        return
    
    st.header("Step 2: Start the AI-Powered Interview")
    
    st.markdown("""
    **What to Expect:**
    - You will be asked **5 questions** based on your resume and job description.
    - Provide **clear and structured responses** in the text box.
    - Once completed, the system will **analyze your answers** and generate a performance report.
    """)


    if "questions" not in st.session_state:
        st.session_state.questions = fetch_questions(resume_category, resume_seniority, job_desc_category, job_desc_seniority)

    if "interview_started" not in st.session_state:
        st.session_state.interview_started = False

    if not st.session_state.interview_started:
        if st.button("Begin Interview"):
            st.session_state.interview_started = True
            st.rerun()
        return  

    interview_ui()  

if __name__ == "__main__":
    main()
