"""
Smart Job Recommender - Streamlit Cloud Deployment Version
=============================================

Updated version using Google Custom Search JSON API with GOOGLE_API_KEY and SEARCH_ENGINE_ID.
Compatible with streamlit>=1.48.0, requests>=2.32.0, google-generativeai==0.8.0, pypdf==5.9.0, beautifulsoup4>=4.12.3, lxml>=5.2.2.

Main changes (v2.9):
- Integrated a web scraping layer using requests and BeautifulSoup to fetch accurate job details.
- Added scrape_job_details_from_url function to parse live job pages from LinkedIn, Indeed, etc.
- This new method directly extracts Company, Location, and Salary from the source page, fixing the "Unknown" data issue.
- The system now uses scraping as the primary data source and falls back to snippet parsing if scraping fails.
- Added necessary imports (BeautifulSoup) and updated requirements.

Author: AI Assistant (updated)
Version: 2.9 (Web Scraping for Accurate Data Extraction)
"""

import streamlit as st
import requests
import time
import os
from typing import List, Dict, Any
import tempfile
from urllib.parse import quote_plus, urlparse
import re  # For parsing salary, location, and company

# Import web scraping and AI libraries with error handling
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    st.error("BeautifulSoup not available. Please install: pip install beautifulsoup4 lxml")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.error("Google Generative AI not available. Please install: pip install google-generativeai==0.8.0")

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    st.error("PyPDF not available. Please install: pip install pypdf==5.9.0")

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Smart Job Recommender",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CORE RAG SYSTEM CLASS
# ============================================================================

class SmartJobRecommenderRAG:
    """Enhanced RAG system for job recommendations using Gemini Flash and Web Scraping"""

    def __init__(self):
        self.gemini_client = None
        self.initialize_gemini()

    def initialize_gemini(self) -> bool:
        """Initialize Gemini AI client"""
        try:
            try:
                gemini_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
            except Exception:
                gemini_key = os.environ.get("GEMINI_API_KEY")

            if gemini_key and GEMINI_AVAILABLE:
                try:
                    genai.configure(api_key=gemini_key)
                    self.gemini_client = genai.GenerativeModel('gemini-1.5-flash')
                    # Suppress success message on every run
                    # st.success("AI system initialized successfully")
                    return True
                except Exception as e:
                    st.error(f"❌ Error initializing Gemini client: {e}")
                    return False
            else:
                st.error("❌ Gemini API key required. Please add GEMINI_API_KEY to your Streamlit secrets.")
                return False
        except Exception as e:
            st.error(f"❌ Error initializing Gemini: {e}")
            return False

    def load_document_with_pypdf(self, uploaded_file) -> List:
        """Load PDF document using PyPDF (defensive against None pages)"""
        if not PYPDF_AVAILABLE:
            st.error("PyPDF not available for document processing")
            return []

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name

            documents = []
            reader = PdfReader(temp_file_path)

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() if hasattr(page, 'extract_text') else None
                if text and text.strip():
                    doc_obj = type('Document', (), {
                        'page_content': text,
                        'metadata': {'page': page_num + 1}
                    })()
                    documents.append(doc_obj)

            try:
                os.unlink(temp_file_path)
            except Exception:
                pass

            st.success(f"✅ Loaded {len(documents)} pages from PDF")
            return documents

        except Exception as e:
            st.error(f"❌ Error loading PDF: {e}")
            return []

    def call_direct_gemini(self, prompt: str) -> Dict[str, Any]:
        """Call Gemini directly for text analysis"""
        if not self.gemini_client:
            return {"skills": [], "job_interests": [], "experience_level": "entry"}

        try:
            response = self.gemini_client.generate_content(prompt)
            response_text = response.text

            skills = []
            job_interests = []
            experience_level = "entry"

            lines = response_text.split('\n')
            for line in lines:
                if line.startswith('SKILLS:'):
                    skills_text = line.replace('SKILLS:', '').strip()
                    skills = [s.strip() for s in skills_text.split(',') if s.strip()]
                elif line.startswith('JOB_INTERESTS:'):
                    interests_text = line.replace('JOB_INTERESTS:', '').strip()
                    job_interests = [i.strip() for i in interests_text.split(',') if i.strip()]
                elif line.startswith('EXPERIENCE_LEVEL:'):
                    experience_level = line.replace('EXPERIENCE_LEVEL:', '').strip().lower()

            return {
                "skills": skills[:10],
                "job_interests": job_interests[:5],
                "experience_level": experience_level
            }

        except Exception as e:
            st.error(f"❌ Error calling Gemini: {e}")
            return {"skills": [], "job_interests": [], "experience_level": "entry"}

    def scrape_job_details_from_url(self, job_url: str) -> Dict[str, str]:
        """
        Scrapes detailed job info (company, location, salary) from the job URL.
        Contains specific logic for different job platforms.
        """
        if not job_url or not BS4_AVAILABLE:
            return {}

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        scraped_data = {}
        
        try:
            response = requests.get(job_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            
            domain = urlparse(job_url).netloc

            # --- LinkedIn Scraping Logic ---
            if 'linkedin.com' in domain:
                company_tag = soup.select_one('a.topcard__org-name-link') or soup.select_one('span.topcard__flavor')
                if company_tag:
                    scraped_data['company'] = company_tag.get_text(strip=True)
                
                location_tag = soup.select_one('span.topcard__flavor--bullet')
                if location_tag:
                    scraped_data['location'] = location_tag.get_text(strip=True)
            
            # --- Indeed Scraping Logic ---
            elif 'indeed.com' in domain:
                company_tag = soup.select_one('div[data-company-name="true"]')
                if company_tag:
                    scraped_data['company'] = company_tag.get_text(strip=True)

                location_tag = soup.find('div', {'data-testid': 'inlineHeader-companyLocation'})
                if location_tag:
                    scraped_data['location'] = location_tag.get_text(strip=True)

                salary_tag = soup.find('div', id='salaryInfoAndJobType')
                if salary_tag:
                    salary_span = salary_tag.find('span')
                    if salary_span:
                        scraped_data['salary'] = salary_span.get_text(strip=True)
                        
            # Add more 'elif' blocks here for Glassdoor, Monster, etc. as needed

        except requests.exceptions.RequestException as e:
            # Silently fail if scraping doesn't work, will fall back to snippet
            # print(f"Warning: Could not scrape {job_url}. Reason: {e}")
            pass
        except Exception as e:
            # print(f"Warning: An error occurred during scraping of {job_url}. Reason: {e}")
            pass
            
        return scraped_data

    def sanitize_link(self, link: Any) -> str:
        """Return cleaned link or empty string if invalid"""
        if not link:
            return ""
        try:
            cleaned = str(link).strip()
            if cleaned == "#" or cleaned.lower() == "none":
                return ""
            return cleaned
        except Exception:
            return ""

    def get_best_apply_link(self, job: Dict[str, Any], response_data: Dict[str, Any] = None) -> str:
        """Try many possible fields for an application/website link"""
        return self.sanitize_link(job.get('link'))

    def extract_salary_from_snippet(self, snippet: str) -> str:
        """Extract salary from snippet using regex (FALLBACK METHOD)"""
        salary_pattern = r'(\$|USD|EUR|₹|INR)?\s*([\d,]+\s*(?:to|-)?\s*[\d,]+)\s*(?:per|a)\s*(?:year|month|annum|hr|hour)|(\d{1,3}(?:,\d{3})?(?:-\d{1,3}(?:,\d{3})?)?)\s*(LPA|lakh|crore)'
        match = re.search(salary_pattern, snippet)
        return match.group(0).strip() if match else "Not specified"

    def extract_location_from_snippet(self, snippet: str) -> str:
        """Extract location from snippet (FALLBACK METHOD)"""
        location_pattern = r'\b(in|at|near)\s+([A-Z][a-z]+(?:,\s*[A-Z]{2})?|[A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*)'
        match = re.search(location_pattern, snippet)
        return match.group(2) if match else "Unknown Location"

    def extract_company_from_snippet(self, title: str, snippet: str) -> str:
        """Extract company name from snippet if not available in metadata (FALLBACK METHOD)"""
        # Company names are often at the start of the title, like "Python Developer - Google"
        parts = title.split('-')
        if len(parts) > 1:
            return parts[-1].strip()
        # Fallback to checking the snippet
        company_pattern = r'(\w+ Inc\.|\w+ Corporation|\w+ Ltd\.|\w+ Solutions)'
        match = re.search(company_pattern, snippet)
        return match.group(1) if match else "Unknown Company"

    def search_jobs_with_custom_search_api(self, skills: List[str], job_interests: List[str]) -> Dict[str, List]:
        """Search jobs using Google Custom Search JSON API"""
        return self._perform_job_search(skills, job_interests)

    def search_jobs_with_custom_search_api_location(self, skills: List[str], job_interests: List[str], location: str) -> Dict[str, List]:
        """Search jobs with location preference using Google Custom Search JSON API"""
        return self._perform_job_search(skills, job_interests, location)

    def _perform_job_search(self, skills: List[str], job_interests: List[str], location: str = "") -> Dict[str, List]:
        """Unified job search function"""
        try:
            google_api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            search_engine_id = st.secrets.get("SEARCH_ENGINE_ID") or os.environ.get("SEARCH_ENGINE_ID")

            if not google_api_key or not search_engine_id:
                st.error("❌ Google API key and Search Engine ID required.")
                return {"jobs": [], "internships": [], "search_queries": []}

            search_queries = []
            location_query = f"in {location}" if location.strip() else ""
            
            base_skills = skills[:3]
            base_interests = job_interests[:3]

            for item in base_skills + base_interests:
                search_queries.append(f'"{item}" job openings {location_query}')
            
            if not search_queries:
                search_queries = [f"software developer jobs {location_query}", f"python developer jobs {location_query}"]

            all_jobs = []
            all_internships = []
            url = "https://www.googleapis.com/customsearch/v1"

            for query in list(set(search_queries))[:5]:
                try:
                    params = {
                        "key": google_api_key, "cx": search_engine_id, "q": query,
                        "num": 10, "safe": "off"
                    }
                    st.info(f"🔍 Searching Google for: '{query}'...")
                    response = requests.get(url, params=params, timeout=15)
                    
                    if response.status_code != 200:
                        st.warning(f"API Error for query '{query}': {response.text}")
                        continue
                    
                    data = response.json()
                    if "items" not in data or not data["items"]:
                        continue

                    for item in data["items"]:
                        apply_link = self.get_best_apply_link(item, response_data=data)
                        if not apply_link:
                            continue

                        # --- NEW: Scrape details from the actual job page ---
                        scraped_details = self.scrape_job_details_from_url(apply_link)

                        # --- Use scraped data first, then fallback to API/snippet data ---
                        snippet = item.get("snippet", "")
                        title = item.get("title", "Unknown Title")
                        
                        company = scraped_details.get('company') or \
                                  item.get("pagemap", {}).get("metatags", [{}])[0].get("og:site_name") or \
                                  self.extract_company_from_snippet(title, snippet)
                        
                        location_val = scraped_details.get('location') or self.extract_location_from_snippet(snippet)
                        
                        salary = scraped_details.get('salary') or self.extract_salary_from_snippet(snippet)

                        job_data = {
                            "title": title,
                            "company": company,
                            "location": location_val,
                            "description": snippet,
                            "apply_link": apply_link,
                            "salary": salary,
                            "source": urlparse(apply_link).netloc.replace('www.', ''),
                            "match_score": self.calculate_match_score(skills, snippet),
                            "required_skills": self.extract_skills_from_description(snippet)
                        }

                        if any(word in title.lower() for word in ["intern", "internship", "trainee"]):
                            all_internships.append(job_data)
                        else:
                            all_jobs.append(job_data)
                    
                    time.sleep(0.5)

                except Exception as e:
                    st.warning(f"⚠️ Error during search for query '{query}': {str(e)}")
                    continue

            unique_jobs = self.remove_duplicates(all_jobs)
            unique_internships = self.remove_duplicates(all_internships)

            unique_jobs.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            unique_internships.sort(key=lambda x: x.get("match_score", 0), reverse=True)

            st.success(f"✅ Found {len(unique_jobs)} jobs and {len(unique_internships)} internships.")
            return {"jobs": unique_jobs[:20], "internships": unique_internships[:10]}

        except Exception as e:
            st.error(f"❌ Critical error during job search: {e}")
            return {"jobs": [], "internships": [], "search_queries": []}

    def calculate_match_score(self, user_skills: List[str], job_description: str) -> int:
        """Calculate match percentage between user skills and job requirements"""
        if not user_skills or not job_description:
            return 0
        job_desc_lower = (job_description or "").lower()
        matched_skills = sum(1 for skill in user_skills if skill and skill.lower() in job_desc_lower)
        return int((matched_skills / len(user_skills)) * 100) if user_skills else 0

    def extract_skills_from_description(self, description: str) -> List[str]:
        """Extract skills from job description"""
        if not description:
            return []
        tech_skills = ["python", "java", "javascript", "react", "sql", "aws", "docker", "kubernetes", "git", "machine learning", "ai", "data analysis", "pandas", "numpy", "tensorflow", "pytorch", "tableau", "power bi", "agile", "scrum"]
        return list(set([skill.title() for skill in tech_skills if skill in description.lower()]))[:10]

    def remove_duplicates(self, jobs: List[Dict]) -> List[Dict]:
        """Remove duplicate jobs based on title and company"""
        seen = set()
        unique_jobs = []
        for job in jobs:
            key = (job.get("title", "").lower(), job.get("company", "").lower())
            if key not in seen:
                seen.add(key)
                unique_jobs.append(job)
        return unique_jobs

# ============================================================================
# STREAMLIT UI COMPONENTS (UNCHANGED)
# ============================================================================

def main():
    """Main application function"""
    # Add background image CSS for light theme
    background_css = """
    <style>
    body, .stApp, .css-1aumxhk, .st-emotion-cache-1aumxhk {
        background-color: #FFFFFF; color: #333333;
    }
    .stSidebar { background-color: #FFFFFF; color: #333333; }
    .stSidebar * { color: #333333; }
    .stApp *, body *, .css-1aumxhk *, .st-emotion-cache-1aumxhk * { color: #333333; text-shadow: none; }
    .stButton>button { background-color: #D3D3D3; color: #333333; border: 1px solid #CCCCCC; border-radius: 5px; padding: 8px 16px; }
    .stButton>button:hover { background-color: #C0C0C0; color: #333333; }
    [data-testid="stNotification"], .stAlert { background-color: #DFF5E1; color: #333333; border-radius: 8px; padding: 8px 12px; font-weight: 500; border: 1px solid #BEE3BE; }
    [data-testid="stFileUploaderDropzone"] { background-color: #F8F9FA; border: 2px dashed #CCCCCC; border-radius: 8px; padding: 20px; color: #333333; }
    header[data-testid="stHeader"] { background-color: #FFFFFF; color: #333333; }
    header[data-testid="stHeader"] * { color: #333333; }
    hr { border-top: 1px solid #E0E0E0; }
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)

    if "rag_system" not in st.session_state:
        st.session_state.rag_system = SmartJobRecommenderRAG()

    st.title("💼 Smart Job Recommender")
    st.markdown("### AI-Powered Job Matching with Real-Time Search")
    st.markdown("---")

    with st.sidebar:
        st.header("🔧 Configuration")
        st.subheader("API Status")

        gemini_ok = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        st.markdown(f"{'✅' if gemini_ok else '❌'} Gemini AI: {'Connected' if gemini_ok else 'API key required'}")

        google_ok = st.secrets.get("GOOGLE_API_KEY") and st.secrets.get("SEARCH_ENGINE_ID")
        st.markdown(f"{'✅' if google_ok else '❌'} Google Custom Search: {'Connected' if google_ok else 'API key & ID required'}")

        st.markdown("---")
        st.subheader("📋 Instructions")
        st.markdown("""
        **Setup Required:**
        1. Add `GEMINI_API_KEY` to Streamlit secrets.
        2. Add `GOOGLE_API_KEY` and `SEARCH_ENGINE_ID` to Streamlit secrets.
        **How to Use:**
        1. Upload your resume PDF, OR
        2. Enter your skills manually.
        3. Get personalized job recommendations.
        """)

    tab1, tab2 = st.tabs(["📄 Resume Upload", "✍️ Manual Entry"])

    with tab1:
        st.header("📄 Upload Your Resume")
        st.markdown("Upload your resume in PDF format for AI-powered skill extraction and job matching.")
        uploaded_file = st.file_uploader("Choose your resume PDF file", type="pdf")
        if uploaded_file:
            if st.button("🚀 Analyze Resume & Find Jobs", type="primary"):
                process_resume_and_find_jobs(uploaded_file)

    with tab2:
        st.header("✍️ Manual Skills Entry")
        st.markdown("Enter your skills and preferences manually to find matching job opportunities.")
        with st.form("manual_skills_form"):
            skills_input = st.text_area("Your Skills (comma-separated)", placeholder="e.g., Python, React, Machine Learning, SQL")
            col1, col2 = st.columns(2)
            with col1:
                job_interests = st.text_input("Job Interests (comma-separated)", placeholder="e.g., Software Developer, Data Scientist")
            with col2:
                location_pref = st.text_input("Preferred Location (Optional)", placeholder="e.g., United States, Remote, New York")
            submitted = st.form_submit_button("🔍 Find Matching Jobs", type="primary")
            if submitted and skills_input.strip():
                skills_list = [s.strip() for s in skills_input.split(',') if s.strip()]
                interests_list = [i.strip() for i in job_interests.split(',') if i.strip()]
                manual_data = {"skills": skills_list, "job_interests": interests_list, "experience_level": "entry"}
                process_manual_skills_and_find_jobs(manual_data, location_pref)
            elif submitted:
                st.error("Please enter at least one skill.")

def process_resume_and_find_jobs(uploaded_file):
    rag_system = st.session_state.rag_system
    with st.spinner("Analyzing your resume and searching for jobs... This may take a moment."):
        documents = rag_system.load_document_with_pypdf(uploaded_file)
        if not documents:
            st.error("❌ Failed to load PDF.")
            return
        all_text = "\n\n".join([doc.page_content for doc in documents])
        prompt = f"Based on this resume, extract: SKILLS: [list], JOB_INTERESTS: [list], EXPERIENCE_LEVEL: [entry/mid/senior]. Resume: {all_text}"
        extracted_data = rag_system.call_direct_gemini(prompt)
        job_results = rag_system.search_jobs_with_custom_search_api(extracted_data["skills"], extracted_data["job_interests"])
    display_results(extracted_data, job_results)

def process_manual_skills_and_find_jobs(manual_data: Dict[str, Any], location_pref: str):
    rag_system = st.session_state.rag_system
    with st.spinner("Searching for jobs based on your skills... This may take a moment."):
        if location_pref.strip():
            job_results = rag_system.search_jobs_with_custom_search_api_location(manual_data["skills"], manual_data["job_interests"], location_pref)
        else:
            job_results = rag_system.search_jobs_with_custom_search_api(manual_data["skills"], manual_data["job_interests"])
    display_results(manual_data, job_results)

def display_results(extracted_data: Dict[str, Any], job_results: Dict[str, List]):
    st.markdown("---")
    st.header("📊 Analysis Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🛠️ Your Skills")
        st.markdown("\n".join(f"• {s}" for s in extracted_data.get("skills", [])))
    with col2:
        st.subheader("💼 Your Interests")
        st.markdown("\n".join(f"• {i}" for i in extracted_data.get("job_interests", [])))
    with col3:
        st.subheader("📈 Experience Level")
        st.markdown(f"**{(extracted_data.get('experience_level', 'N/A')).title()}**")

    st.markdown("---")
    st.header("💼 Job Recommendations")
    jobs = job_results.get("jobs", [])
    internships = job_results.get("internships", [])

    if jobs:
        st.subheader(f"🎯 Found {len(jobs)} Job Matches")
        for i, job in enumerate(jobs):
            display_job_card(job, "job", i, extracted_data)
    
    if internships:
        st.subheader(f"🎓 Found {len(internships)} Internship Matches")
        for i, intern in enumerate(internships):
            display_job_card(intern, "intern", i, extracted_data)

    if not jobs and not internships:
        st.info("🔍 No job matches found. Try broadening your skills or checking API configurations in the sidebar.")

def display_job_card(job: Dict, job_type: str, index: int, user_data: Dict):
    """Reusable function to display a job or internship card."""
    with st.expander(f"#{index + 1} {job.get('title', 'N/A')} at {job.get('company', 'N/A')} - {job.get('match_score', 0)}% Match"):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**Company:** {job.get('company', 'N/A')}")
            st.write(f"**Location:** {job.get('location', 'N/A')}")
            st.write(f"**Salary:** {job.get('salary', 'Not specified')}")
            st.write(f"**Description:** {job.get('description', '')[:200]}...")
            if job.get('required_skills'):
                st.write("**Mentioned Skills:** " + ", ".join(job['required_skills']))
        with col2:
            st.metric("Match Score", f"{job.get('match_score', 0)}%")
            st.write(f"**Source:** {job.get('source', 'Unknown')}")
            apply_link = job.get('apply_link', '').strip()
            if apply_link and apply_link != "#":
                st.link_button("🚀 Apply Now", apply_link)
            else:
                st.warning("No direct link found.")

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()```
