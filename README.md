<img width="1008" height="647" alt="ChatGPT Image Aug 11, 2025, 12_53_45 PM" src="https://github.com/user-attachments/assets/37d6a2ea-7023-434d-a2e6-b7451cc4cb03" />

# Smart Job Recommender - Gemini RAG System

📄 Upload your resume → 🧠 Let AI extract your skills → 🔍 Find the perfect jobs instantly.



## Project Overview
Finding a job or internship is time-consuming and frustrating for many students:You have to search on multiple websites (Google Jobs, LinkedIn, etc.).You see lots of irrelevant job posts that don’t match your skills.You waste time reading through dozens of descriptions.
What if you could have an AI assistant that:
- Reads your resume or skills.
- Searches for jobs online.
- Picks the ones that match you best.
- Explains why they’re a good fit.


## 🎯 Problem Statement

This project aims to build an AI-powered job recommendation web application that helps students and fresh graduates quickly find the most relevant job or internship opportunities.
The app will use LangChain, Google’s Gemini LLM, and Retrieval-Augmented Generation (RAG) techniques to:

- Understand the user’s skills, experience, and preferences from either a PDF resume or manual input.
- Retrieve relevant job postings from Google Jobs or LinkedIn Jobs using SerpAPI.
- Match and rank jobs based on skill relevance.
- Provide short, personalized explanations for each recommendation.

The application will be developed with Streamlit to deliver an interactive and user-friendly experience, making job searching smarter, faster, and more personalized.


## 🛠️ Tech Stack

Component	Technology
Frontend	Streamlit (Python)
AI Parsing	Gemini AI
Job Search	Google Custom Search API
Hosting	Streamlit Cloud / Local
Styling	Custom CSS + Background


## 💡 Key Idea

<img width="1095" height="655" alt="image" src="https://github.com/user-attachments/assets/2b368a82-f59f-4c89-b476-fbb34359e900" />

The core concept behind building **Smart Job Recommender** was to create a **seamless AI-powered pipeline** that transforms a simple resume upload into **tailored job opportunities** from multiple platforms in real-time.

### High-Level Workflow

1. **User Uploads Resume**  
   - The process starts when a user uploads their resume in PDF format.

2. **Document Loading**  
   - The uploaded PDF is read using **PyPDF Loader** from `langchain.document_loaders`.  
   - This extracts the **raw text** from the document.

3. **Text Splitting (Skipped for Resumes)**  
   - Normally, for large documents, the text would be split into smaller chunks.  
   - Since resumes typically have **low text volume**, chunking is **skipped** in this project to keep the pipeline lightweight.

4. **No Vector Store Required**  
   - Without chunks, there’s no need for a vector store or embedding search — the text can be processed directly.

5. **Passing Extracted Text to Gemini AI**  
   - The raw text is sent directly to **Gemini AI**.  
   - Gemini is instructed to **parse and extract structured details** such as:  
     - 📌 **Skills**  
     - 📌 **Experience**  
     - 📌 **Job Interests**  
     - 📌 **Preferred Locations** (if mentioned)

6. **Receiving JSON Output from Gemini**  
   - Gemini returns a **clean, structured JSON** containing the extracted information.

7. **Job Search with Google Custom Search API**  
   - This JSON data (especially the skills and job interests) is used to query **Google Custom Search API**.  
   - The API is configured to fetch results from:  
     - LinkedIn Jobs  
     - Google Jobs  
     - Indeed  
     - Naukri  
     - Internshala

8. **Real-Time Job Matching**  
   - The results are presented in the app UI in real-time, ensuring users see **fresh, relevant job listings** instead of outdated static data.
  
9. **finds matching jobs with:**

  Real job postings from Google Jobs
  Compatibility scores
  Required vs. your skills comparison
  Direct application links


## 🚀 How the App Works

The **Smart Job Recommender** offers two simple ways for users to find relevant jobs:

### 1️⃣ Upload Your Resume  
- **Step 1:** Click the **"Upload Resume"** option on the homepage.  
- **Step 2:** Select your PDF resume file.  
- **Step 3:** The app will automatically read your resume using **PyPDF Loader**, extract your **skills**, **experience**, and **job interests** using **Gemini AI**, and pass them to the **Google Custom Search API**.  
- **Step 4:** Instantly see **fresh job listings** from LinkedIn, Google Jobs, Indeed, Naukri, and Internshala.
- 
https://github.com/user-attachments/assets/a253b522-5eda-4302-a5b2-b2574137c127



### 2️⃣ Enter Details Manually  
- **Step 1:** Choose the **"Enter Details Manually"** option.  
- **Step 2:** Type in your **skills**, **preferred location**, and **experience level**.  
- **Step 3:** The app will directly use your inputs to search for jobs through the **Google Custom Search API**.  
- **Step 4:** Browse the latest and most relevant job postings in real-time.

https://github.com/user-attachments/assets/c391de19-c8e0-4796-8ba8-91b236de0879



 











