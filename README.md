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
- Retrieve relevant job postings from Google Jobs or LinkedIn Jobs using Google Custom Search API .
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


1️⃣ Resume Upload / Manual Skills Entry 
    ↓
2️⃣ Gemini AI (Skill & Preference Extraction) 
    ↓
3️⃣ Google Custom Search API 
    ↓
4️⃣ Job Platforms (LinkedIn, Naukri, Indeed, Internshala) 
    ↓
5️⃣ Matching & Ranking Engine 
    ↓
6️⃣ Streamlit UI (Job Results & Explanations)


## 🚀 How the App Works


The **Smart Job Recommender** offers two simple ways for users to find relevant jobs:

**APP LINK** : https://smartjobrecommenderusinglangchain-3r597cctohrhdbqvzubyd9.streamlit.app/

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


## Folder / Code Structure

Here’s how our project is organized:

<img width="986" height="536" alt="image" src="https://github.com/user-attachments/assets/59310454-4f4a-44e0-aeff-2a8f66d2e761" />



## ⚡ Challenges & How I Solved Them

1. **Parsing Resume Data Accurately**  
   - *Challenge:* Resumes come in different formats and layouts, making extraction tricky.  
   - *Solution:* Used **PyPDF Loader** for consistent text extraction and avoided chunking for small text sizes.

2. **Avoiding Irrelevant Job Results**  
   - *Challenge:* Google Custom Search API sometimes returned unrelated links.  
   - *Solution:* Fine-tuned search queries using extracted **skills** and **job titles** from Gemini AI output.

3. **Fetching Jobs from Multiple Platforms**  
   - *Challenge:* APIs for many job portals are paid or unavailable.  
   - *Solution:* Leveraged **Google Custom Search API** to scrape results from LinkedIn, Indeed, Naukri, and Internshala in one request.

4. **Maintaining Real-Time Job Listings**  
   - *Challenge:* Many job listings go stale quickly.  
   - *Solution:* Always query live data instead of storing results, ensuring users see only **fresh, active jobs**.
   

## 🔮 Future Enhancements

Here are some planned improvements and ideas to make **Smart Job Recommender** even more powerful:

1. **Advanced Job Filtering**  
   - Add filters for salary range, remote/hybrid/on-site preference, company size, and industry type.

2. **User Profile & History**  
   - Let users create accounts and save their job search history, shortlisted jobs, and application status.

3. **Multiple Resume Support**  
   - Allow users to upload and compare results from multiple resumes to see which one gets better matches.

4. **AI-Powered Resume Feedback**  
   - Provide automated suggestions to improve resumes before job searching.

5. **More Job Sources**  
   - Integrate APIs from platforms like Glassdoor, Monster, AngelList, and others.

6. **Email Alerts**  
   - Send daily or weekly job recommendations directly to a user’s inbox.

7. **Job Application Tracker**  
   - Help users keep track of which jobs they’ve applied for and their application status.

8. **Multilingual Support**  
   - Expand compatibility for resumes in different languages.


## 🙏 Big Thanks for Reading!  
Thank you for taking the time to explore **Smart Job Recommender**.  
Your interest and support mean a lot — feel free to ⭐ the repo if you found it useful, and contributions are always welcome!

 













