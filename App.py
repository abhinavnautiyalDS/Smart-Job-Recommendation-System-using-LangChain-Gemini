# Smart Job Recommender - Streamlit Cloud Deployment Version
# =============================================

# Updated version using Google Custom Search JSON API with GOOGLE_API_KEY and SEARCH_ENGINE_ID.
# Compatible with streamlit>=1.48.0, requests>=2.32.0, google-generativeai==0.8.0, pypdf==5.9.0.

# Main changes (v2.11):
# - Enhanced site-specific parsing for LinkedIn and Indeed to accurately extract title, company, location, salary from title and snippet structures.
# - Added st.spinner for beautiful loading interface during analysis and job search.
# - Improved regex patterns for location and salary to match screenshot formats (e.g., "Seattle, WA", "$118.7K/yr - $168.2K/yr").
# - Increased accuracy in extraction by prioritizing snippet line parsing for Indeed and title splitting for LinkedIn.
# - Removed debug st.write from display to clean up UI; debug now optional via env var.
# - Version bump to v2.11.

# Author: AI Assistant (updated)
# Version: 2.11 (Accurate Extraction & Loading UI)
# """

import streamlit as st
import requests
import time
import os
import re
from typing import List, Dict, Any
import tempfile
from urllib.parse import quote_plus

# Import AI libraries with error handling
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

DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

# ============================================================================
# CORE RAG SYSTEM CLASS
# ============================================================================

class SmartJobRecommenderRAG:
    """Enhanced RAG system for job recommendations using Gemini Flash"""

    def __init__(self):
        self.gemini_client = None
        self.initialize_gemini()

    def initialize_gemini(self) -> bool:
        """Initialize Gemini AI client"""
        try:
            gemini_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if gemini_key and GEMINI_AVAILABLE:
                genai.configure(api_key=gemini_key)
                self.gemini_client = genai.GenerativeModel('gemini-1.5-flash')
                st.success("AI system initialized successfully")
                return True
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
                text = page.extract_text() if hasattr(page, 'extract_text') else ""
                if text and text.strip():
                    doc_obj = type('Document', (), {
                        'page_content': text,
                        'metadata': {'page': page_num + 1}
                    })()
                    documents.append(doc_obj)

            os.unlink(temp_file_path)
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
        """Try multiple fields for an application/website link"""
        candidates = [
            'link', 'url', 'apply_link', 'application_link', 'apply_url', 'job_posting_url',
            'canonical_url', 'destination', 'job_link', 'website', 'company_website', 'company_url'
        ]

        for key in candidates:
            if key in job:
                s = self.sanitize_link(job.get(key))
                if s:
                    return s

        if response_data:
            maybe = response_data.get('website_link') or response_data.get('website')
            if maybe:
                s = self.sanitize_link(maybe)
                if s:
                    return s
            search_meta = response_data.get('search_metadata') if isinstance(response_data, dict) else None
            if search_meta and isinstance(search_meta, dict):
                for k in ['source', 'website', 'source_url']:
                    if k in search_meta:
                        s = self.sanitize_link(search_meta.get(k))
                        if s:
                            return s

        return ""

    def parse_job_data(self, item: Dict, skills: List[str], default_location: str = "Unknown Location") -> Dict:
        """Parse job data from Google Custom Search item based on site with high accuracy"""
        title_str = item.get("title", "Unknown Title")
        snippet = item.get("snippet", "No description")
        link = item.get("link", "")
        pagemap = item.get("pagemap", {})
        metatags = pagemap.get("metatags", [{}])[0]

        site = "other"
        if 'linkedin.com' in link:
            site = "linkedin"
            source = "LinkedIn"
        elif 'indeed.com' in link:
            site = "indeed"
            source = "Indeed"
        elif 'glassdoor.com' in link:
            site = "glassdoor"
            source = "Glassdoor"
        else:
            source = "Google Custom Search"

        job_title = "Unknown Title"
        company = "Unknown Company"
        location = default_location
        salary = "Not specified"
        description = snippet
        apply_link = self.get_best_apply_link(item)
        match_score = self.calculate_match_score(skills, snippet)
        required_skills = self.extract_skills_from_description(snippet)

        # Site-specific parsing for high accuracy
        if site == "linkedin":
            # Example title: "ByteDance hiring Machine Learning Engineer Graduate (E-Commerce Risk Control) - 2026 Start (BS/MS) in Seattle, WA"
            pattern = r"^(.*?)\s+hiring\s+(.*?)\s+in\s+(.*?)(?:\s+-\s+LinkedIn|\s+\|\s+LinkedIn)?$"
            match = re.match(pattern, title_str, re.IGNORECASE)
            if match:
                company = match.group(1).strip()
                job_title = match.group(2).strip()
                location = match.group(3).strip()

            # Snippet for salary and additional details
            salary_match = re.search(r'\$[\d.]+[Kk]?/yr\s*-\s*\$[\d.]+[Kk]?/yr', snippet, re.IGNORECASE)
            if salary_match:
                salary = salary_match.group(0)

        elif site == "indeed":
            # Example title: "Senior Machine Learning Engineer - #TeamGoHealth - Hybrid work in Chicago, IL"
            pattern = r"^(.*?)\s*-\s*(.*?)\s*-\s*(.*?)$"
            match = re.match(pattern, title_str, re.IGNORECASE)
            if match:
                job_title = match.group(1).strip()
                company = match.group(2).strip()
                location = match.group(3).strip()

            # Snippet for salary and description
            snippet_lines = snippet.split('...')
            if snippet_lines:
                first_line = snippet_lines[0].strip()
                parts = first_line.split(' - ')
                if len(parts) >= 2:
                    company = parts[0].strip() if company == "Unknown Company" else company
                    location = parts[1].strip() if location == default_location else location
                if len(snippet_lines) > 1:
                    salary_match = re.search(r'\$[\d,]+ - \$[\d,]+ a year', snippet, re.IGNORECASE)
                    if salary_match:
                        salary = salary_match.group(0)

        elif site == "glassdoor":
            # Example title: "Machine Learning Engineer - Company - Location - Glassdoor"
            pattern = r"^(.*?)\s*-\s*(.*?)\s*-\s*(.*?)\s*-\s*Glassdoor$"
            match = re.match(pattern, title_str, re.IGNORECASE)
            if match:
                job_title = match.group(1).strip()
                company = match.group(2).strip()
                location = match.group(3).strip()

        # Fallback extractions for 100% coverage
        if company == "Unknown Company":
            company = self.extract_company_from_snippet(snippet, title_str) or metatags.get("og:site_name", "Unknown Company")

        if location == "Unknown Location" or location == default_location:
            location = self.extract_location_from_snippet(snippet, title_str)

        if salary == "Not specified":
            salary = pagemap.get("offer", [{}])[0].get("salary", "Not specified")

        # Description cleanup
        description = re.sub(r'\s*\.\.\.$', '', description).strip()

        if DEBUG:
            st.write(f"Debug: Parsed from {site} - Title: {job_title}, Company: {company}, Location: {location}, Salary: {salary}")

        return {
            "title": job_title,
            "company": company,
            "location": location,
            "description": description,
            "apply_link": apply_link,
            "salary": salary,
            "source": source,
            "match_score": match_score,
            "required_skills": required_skills
        }

    def extract_company_from_snippet(self, snippet: str, title: str) -> str:
        """Extract company name from snippet or title, avoiding job board names"""
        if not snippet and not title:
            return "Unknown Company"

        job_boards = [
            "indeed", "linkedin", "glassdoor", "monster", "careerbuilder",
            "simplyhired", "dice", "jobstreet", "jobsdb"
        ]

        patterns = [
            r'at\s+([A-Za-z0-9\s&-]+?)(?:\s*[-|]\s*(?:jobs|careers|apply))?',  
            r'([A-Za-z0-9\s&-]+?)\s*is\s+hiring',  
            r'([A-Za-z0-9\s&-]+?)\s*(?:Careers|Jobs)',  
            r'([A-Za-z0-9\s&-]+?)\s*[-|]\s*\w+\s*(?:Engineer|Developer|Manager)'  
        ]

        combined_text = title + " " + snippet
        for pattern in patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match and match.group(1).strip():
                company_name = match.group(1).strip()
                if not any(jb.lower() in company_name.lower() for jb in job_boards):
                    return company_name

        common_companies = [
            "Google", "Amazon", "Microsoft", "Apple", "Meta", "Tesla", "IBM",
            "Oracle", "Accenture", "Deloitte", "Salesforce", "Adobe"
        ]
        for company in common_companies:
            if company.lower() in combined_text.lower():
                return company

        return "Unknown Company"

    def extract_location_from_snippet(self, snippet: str, title: str) -> str:
        """Extract location from snippet or title with site-specific patterns"""
        if not snippet and not title:
            return "Unknown Location"

        patterns = [
            r'in\s+([A-Za-z\s,]+),\s*[A-Z]{2}',  
            r'([A-Za-z\s]+),\s*[A-Z]{2}',  
            r'based\s+in\s+([A-Za-z\s]+)',  
            r'([A-Za-z\s]+)\s+\(Remote\)',  
            r'remote\s+([A-Za-z\s]+)',  
            r'location:\s*([A-Za-z\s,]+)',  
            r'- Hybrid work in ([A-Za-z\s,]+)',  # Indeed specific
            r'in ([A-Za-z\s,]+) \('  # LinkedIn specific
        ]

        combined_text = title + " " + snippet
        for pattern in patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match and match.group(1).strip():
                return match.group(1).strip()

        common_locations = [
            "New York", "San Francisco", "Seattle", "Austin", "Boston", "Chicago",
            "Los Angeles", "Remote", "London", "Toronto", "Bangalore", "Mumbai"
        ]
        for loc in common_locations:
            if loc.lower() in combined_text.lower():
                return loc

        return "Unknown Location"

    def search_jobs_with_custom_search_api(self, skills: List[str], job_interests: List[str]) -> Dict[str, List]:
        """Search jobs using Google Custom Search JSON API"""
        try:
            google_api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            search_engine_id = st.secrets.get("SEARCH_ENGINE_ID") or os.environ.get("SEARCH_ENGINE_ID")

            if not google_api_key or not search_engine_id:
                st.error("❌ Google API key and Search Engine ID required. Please add GOOGLE_API_KEY and SEARCH_ENGINE_ID to your Streamlit secrets.")
                return {"jobs": [], "internships": [], "search_queries": []}

            search_queries = []
            if skills:
                primary_skills = skills[:3]
                for skill in primary_skills:
                    search_queries.extend([
                        f"{skill} developer jobs",
                        f"{skill} engineer jobs",
                        f"{skill} full-time jobs",
                        f"{skill} job openings"
                    ])

            if job_interests:
                for interest in job_interests[:3]:
                    search_queries.extend([
                        f"{interest} jobs",
                        f"{interest} careers",
                        f"{interest} opportunities"
                    ])

            if not search_queries:
                search_queries = ["software developer jobs", "python developer jobs", "data scientist jobs"]

            st.write(f"🔍 Generated search queries: {search_queries}")

            all_jobs = []
            all_internships = []
            url = "https://www.googleapis.com/customsearch/v1"

            for query in search_queries[:5]:
                try:
                    params = {
                        "key": google_api_key,
                        "cx": search_engine_id,
                        "q": query + " site:*.linkedin.com | site:*.indeed.com | site:*.glassdoor.com | site:*.monster.com | site:*.careerbuilder.com",
                        "num": 10,
                        "safe": "off"
                    }

                    st.info(f"🔍 Searching Google Custom Search for '{query}'...")
                    response = requests.get(url, params=params, timeout=15)
                    st.write(f"API Response Status: {response.status_code}" if DEBUG else "")

                    if response.status_code != 200:
                        st.warning(f"API Error: {response.text}")
                        continue

                    data = response.json()
                    items = data.get("items", [])
                    st.write(f"API Response Items: {len(items)}" if DEBUG else "")

                    if not items:
                        st.warning(f"No results found for query: {query}")
                        continue

                    for item in items:
                        job_data = self.parse_job_data(item, skills)

                        title_lower = job_data["title"].lower()
                        if any(word in title_lower for word in ["intern", "internship", "trainee"]):
                            all_internships.append(job_data)
                        else:
                            all_jobs.append(job_data)

                        time.sleep(0.5)

                except Exception as e:
                    st.warning(f"⚠️ Error searching Google Custom Search: {str(e)}")
                    continue

            unique_jobs = self.remove_duplicates(all_jobs)
            unique_internships = self.remove_duplicates(all_internships)

            unique_jobs.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            unique_internships.sort(key=lambda x: x.get("match_score", 0), reverse=True)

            st.success(f"✅ Found {len(unique_jobs)} jobs and {len(unique_internships)} internships")

            return {
                "jobs": unique_jobs[:20],
                "internships": unique_internships[:10],
                "search_queries": search_queries[:8]
            }

        except Exception as e:
            st.error(f"❌ Error with job search: {e}")
            return {"jobs": [], "internships": [], "search_queries": []}

    def search_jobs_with_custom_search_api_location(self, skills: List[str], job_interests: List[str], location: str) -> Dict[str, List]:
        """Search jobs with location preference using Google Custom Search JSON API"""
        try:
            google_api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            search_engine_id = st.secrets.get("SEARCH_ENGINE_ID") or os.environ.get("SEARCH_ENGINE_ID")

            if not google_api_key or not search_engine_id:
                st.error("❌ Google API key and Search Engine ID required. Please add GOOGLE_API_KEY and SEARCH_ENGINE_ID to your Streamlit secrets.")
                return {"jobs": [], "internships": [], "search_queries": []}

            search_queries = []
            if skills:
                primary_skills = skills[:3]
                for skill in primary_skills:
                    search_queries.extend([
                        f"{skill} jobs {location}",
                        f"{skill} developer {location}",
                        f"{skill} engineer {location}",
                        f"{skill} full-time {location}"
                    ])

            if job_interests:
                for interest in job_interests[:3]:
                    search_queries.extend([
                        f"{interest} {location}",
                        f"{interest} jobs {location}"
                    ])

            if any(word in location.lower() for word in ["india", "mumbai", "delhi", "bangalore", "chennai", "pune", "hyderabad"]):
                search_queries.append(f"internship {location}")

            if not search_queries:
                search_queries = [f"software developer jobs {location}", f"python developer jobs {location}"]

            st.write(f"🔍 Generated location-based search queries: {search_queries}")

            all_jobs = []
            all_internships = []
            url = "https://www.googleapis.com/customsearch/v1"

            for query in search_queries[:5]:
                try:
                    params = {
                        "key": google_api_key,
                        "cx": search_engine_id,
                        "q": query + " site:*.linkedin.com | site:*.indeed.com | site:*.glassdoor.com | site:*.monster.com | site:*.careerbuilder.com",
                        "num": 10,
                        "safe": "off"
                    }

                    st.info(f"🔍 Searching Google Custom Search in {location} for '{query}'...")
                    response = requests.get(url, params=params, timeout=15)
                    st.write(f"API Response Status: {response.status_code}" if DEBUG else "")

                    if response.status_code != 200:
                        st.warning(f"API Error: {response.text}")
                        continue

                    data = response.json()
                    items = data.get("items", [])
                    st.write(f"API Response Items: {len(items)}" if DEBUG else "")

                    if not items:
                        st.warning(f"No results found for query: {query}")
                        continue

                    for item in items:
                        job_data = self.parse_job_data(item, skills, location)

                        title_lower = job_data["title"].lower()
                        if any(word in title_lower for word in ["intern", "internship", "trainee"]):
                            all_internships.append(job_data)
                        else:
                            all_jobs.append(job_data)

                        time.sleep(0.5)

                except Exception as e:
                    st.warning(f"⚠️ Error searching Google Custom Search: {str(e)}")
                    continue

            unique_jobs = self.remove_duplicates(all_jobs)
            unique_internships = self.remove_duplicates(all_internships)

            unique_jobs.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            unique_internships.sort(key=lambda x: x.get("match_score", 0), reverse=True)

            st.success(f"✅ Found {len(unique_jobs)} jobs and {len(unique_internships)} internships in {location}")

            return {
                "jobs": unique_jobs[:20],
                "internships": unique_internships[:10],
                "search_queries": search_queries[:8]
            }

        except Exception as e:
            st.error(f"❌ Error with location-based job search: {e}")
            return {"jobs": [], "internships": [], "search_queries": []}

    def calculate_match_score(self, user_skills: List[str], job_description: str) -> int:
        """Calculate match percentage between user skills and job requirements"""
        if not user_skills or not job_description:
            return 0

        job_desc_lower = job_description.lower()
        matched_skills = 0

        for skill in user_skills:
            if skill and skill.lower() in job_desc_lower:
                matched_skills += 1

        try:
            return int((matched_skills / len(user_skills)) * 100) if user_skills else 0
        except Exception:
            return 0

    def extract_skills_from_description(self, description: str) -> List[str]:
        """Extract skills from job description"""
        if not description:
            return []

        tech_skills = [
            "python", "java", "javascript", "typescript", "c++", "c#", "php", "ruby", "go", "rust", "swift", "kotlin",
            "react", "angular", "vue.js", "node.js", "express", "django", "flask", "fastapi", "spring boot",
            "html", "css", "sass", "bootstrap", "tailwind", "jquery",
            "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "oracle", "sqlite",
            "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "github", "gitlab",
            "terraform", "ansible", "linux", "bash",
            "machine learning", "ai", "data analysis", "pandas", "numpy", "tensorflow", "pytorch",
            "tableau", "power bi", "excel", "r", "spark",
            "communication", "leadership", "project management", "agile", "scrum", "problem solving",
            "teamwork", "time management"
        ]

        found_skills = []
        desc_lower = description.lower()

        for skill in tech_skills:
            if skill in desc_lower:
                found_skills.append(skill.title())

        return list(set(found_skills))[:10]

    def remove_duplicates(self, jobs: List[Dict]) -> List[Dict]:
        """Remove duplicate jobs based on apply_link, title, and company"""
        seen = set()
        unique_jobs = []

        for job in jobs:
            apply_link = (job.get("apply_link") or "").lower()
            title = (job.get("title") or "").lower()
            company = (job.get("company") or "").lower()
            key = (apply_link, title, company) if apply_link else (title, company)
            if key not in seen:
                seen.add(key)
                unique_jobs.append(job)

        return unique_jobs

# ============================================================================
# STREAMLIT UI COMPONENTS
# ============================================================================

def main():
    """Main application function"""
    # Add background image CSS for light theme
    background_css = """
    <style>
    /* Set background for entire app including top and browser file areas */
    body, .stApp, .css-1aumxhk, .st-emotion-cache-1aumxhk {
        background-color: #FFFFFF; /* White background */
        background-image: url("https://images.unsplash.com/photo-1519120944692-1a8d8cfc107f?q=80&w=1936&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-position: center;
        color: #000000; /* Light black text for contrast */
    }

    /* Ensure sidebar matches light theme */
    .stSidebar {
        background-color: #FFFFFF; /* White sidebar */
        color: #000000; /* Light black text */
    }
    .stSidebar * {
        color: #000000;
    }

    /* Improve text readability */
    .stApp *, body *, .css-1aumxhk *, .st-emotion-cache-1aumxhk * {
        color: #000000; /* Consistent light black text */
        text-shadow: none;
    }

    /* Style buttons with light grey background and light black text */
    .stButton>button {
        background-color: #D3D3D3; /* Light grey background */
        color: #000000; /* Light black text */
        border: 1px solid #CCCCCC;
        border-radius: 5px;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #C0C0C0; /* Slightly darker grey on hover */
        color: #000000;
    }

    /* Success banner */
    [data-testid="stNotification"], .stAlert {
        background-color: #DFF5E1;
        color: #000000;
        border-radius: 8px;
        padding: 8px 12px;
        font-weight: 500;
        border: 1px solid #BEE3BE;
    }

    /* File uploader box */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #F8F9FA;
        border: 2px dashed #CCCCCC;
        border-radius: 8px;
        padding: 20px;
        color: #333333;
    }

    /* Browse files button */
    [data-testid="stFileUploaderBrowseButton"] > div:first-child {
        background-color: #D3D3D3;
        color: #333333;
        border: 1px solid #CCCCCC;
        border-radius: 5px;
        padding: 4px 12px;
    }
    [data-testid="stFileUploaderBrowseButton"] > div:first-child:hover {
        background-color: #C0C0C0;
        color: #333333;
    }

    /* Top navigation bar */
    header[data-testid="stHeader"] {
        background-color: #FFFFFF;
        color: #333333;
    }
    header[data-testid="stHeader"] * {
        color: #333333;
    }

    /* Lines */
    hr {
        border-top: 1px solid #E0E0E0;
    }

    /* Style selectbox (dropdown) */
    .stSelectbox div[role="listbox"] * {
        color: #000000 !important;
    }
    .stSelectbox div[role="option"] {
        color: #000000 !important;
        background-color: #FFFFFF; /* White background to match theme */
    }

    /* Text Input & Text Area */
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background-color: #F8F9FA !important;
        color: #000000 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 5px !important;
    }
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {
        color: #666666 !important;
    }

    /* Select Dropdown (Experience Level) */
    [data-testid="stSelectbox"] div[data-baseweb="select"] {
        background-color: #F8F9FA !important;
        color: #000000 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 5px !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] * {
        color: #000000 !important;
    }

    /* File uploader text inside drop area */
    [data-testid="stFileUploader"] section div {
        color: #000000 !important;
    }

    /* Custom loading spinner style */
    .stSpinner {
        color: #007BFF;
        font-size: 18px;
        text-align: center;
    }
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

        try:
            gemini_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if gemini_key:
                st.success("✅ Gemini AI: Connected")
            else:
                st.error("❌ Gemini AI: API key required")
        except Exception:
            st.error("❌ Gemini AI: API key required")

        try:
            google_api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            search_engine_id = st.secrets.get("SEARCH_ENGINE_ID") or os.environ.get("SEARCH_ENGINE_ID")
            if google_api_key and search_engine_id:
                st.success("✅ Google Custom Search: Connected")
            else:
                st.error("❌ Google Custom Search: API key and Search Engine ID required")
        except Exception:
            st.error("❌ Google Custom Search: API key and Search Engine ID required")

        st.markdown("---")
        st.subheader("📋 Instructions")
        st.markdown("""
        **Setup Required:**
        1. Add GEMINI_API_KEY to Streamlit secrets
        2. Add GOOGLE_API_KEY and SEARCH_ENGINE_ID to Streamlit secrets

        **How to Use:**
        1. Upload your resume PDF, OR
        2. Enter your skills manually
        3. Get personalized job recommendations
        4. Click 'Apply Now' to apply directly
        """)

        st.markdown("---")
        st.subheader("🎯 Features")
        st.markdown("""
        - Resume PDF analysis
        - Manual skill entry
        - Real-time job search via Google Custom Search
        - Real-time matching scores
        - Clickable application links
        - Location-based search
        """)

    tab1, tab2 = st.tabs(["📄 Resume Upload", "✍️ Manual Entry"])

    with tab1:
        st.header("📄 Upload Your Resume")
        st.markdown("Upload your resume in PDF format for AI-powered skill extraction and job matching.")

        uploaded_file = st.file_uploader(
            "Choose your resume PDF file",
            type="pdf",
            help="Upload a clear, text-readable PDF resume for best results."
        )

        if uploaded_file is not None:
            st.success(f"✅ Uploaded: {uploaded_file.name}")

            if st.button("🚀 Analyze Resume & Find Jobs", type="primary"):
                process_resume_and_find_jobs(uploaded_file)

    with tab2:
        st.header("✍️ Manual Skills Entry")
        st.markdown("Enter your skills and preferences manually to find matching job opportunities.")

        with st.form("manual_skills_form"):
            skills_input = st.text_area(
                "Your Skills (comma-separated)",
                placeholder="e.g., Python, React, Machine Learning, SQL, Project Management",
                height=100,
                help="Enter your technical and soft skills separated by commas"
            )

            col1, col2 = st.columns(2)

            with col1:
                job_interests = st.text_input(
                    "Job Interests (comma-separated)",
                    placeholder="e.g., Software Developer, Data Scientist, Product Manager",
                    help="Enter your job roles or industries of interest"
                )

                experience_level = st.selectbox(
                    "Experience Level",
                    options=["entry", "mid", "senior"],
                    index=0,
                    help="Select your experience level"
                )

            with col2:
                location = st.text_input(
                    "Preferred Location",
                    placeholder="e.g., Bangalore, Remote, New York",
                    help="Enter your preferred job location"
                )

            submitted = st.form_submit_button("🔍 Search Jobs")

            if submitted:
                skills = [s.strip() for s in skills_input.split(',') if s.strip()]
                job_interests_list = [j.strip() for j in job_interests.split(',') if j.strip()] if job_interests else []

                if not skills and not job_interests_list:
                    st.error("Please enter at least one skill or job interest.")
                else:
                    with st.spinner("Analyzing your skills and searching for jobs..."):
                        rag_system = st.session_state.rag_system
                        if location and location.strip():
                            search_results = rag_system.search_jobs_with_custom_search_api_location(skills, job_interests_list, location.strip())
                        else:
                            search_results = rag_system.search_jobs_with_custom_search_api(skills, job_interests_list)

                    jobs = search_results.get("jobs", [])
                    internships = search_results.get("internships", [])
                    search_queries = search_results.get("search_queries", [])

                    if jobs or internships:
                        st.success("✅ Job search completed! Here are your recommendations:")

                        if jobs:
                            st.subheader("💼 Recommended Jobs")
                            for job in jobs:
                                with st.expander(f"{job['title']} at {job['company']} ({job['location']}) - Match: {job['match_score']}%"):
                                    st.write(f"**Company:** {job['company']}")
                                    st.write(f"**Location:** {job['location']}")
                                    st.write(f"**Salary:** {job['salary']}")
                                    st.write(f"**Description:** {job['description']}")
                                    st.write(f"**Required Skills:** {', '.join(job['required_skills'])}")
                                    if job['apply_link']:
                                        st.markdown(f"[Apply Now]({job['apply_link']})", unsafe_allow_html=True)
                                    else:
                                        st.write("**Apply Link:** Not available")

                        if internships:
                            st.subheader("🌱 Recommended Internships")
                            for internship in internships:
                                with st.expander(f"{internship['title']} at {internship['company']} ({internship['location']}) - Match: {internship['match_score']}%"):
                                    st.write(f"**Company:** {internship['company']}")
                                    st.write(f"**Location:** {internship['location']}")
                                    st.write(f"**Salary:** {internship['salary']}")
                                    st.write(f"**Description:** {internship['description']}")
                                    st.write(f"**Required Skills:** {', '.join(internship['required_skills'])}")
                                    if internship['apply_link']:
                                        st.markdown(f"[Apply Now]({internship['apply_link']})", unsafe_allow_html=True)
                                    else:
                                        st.write("**Apply Link:** Not available")
                    else:
                        st.warning("⚠️ No jobs or internships found matching your criteria.")

def process_resume_and_find_jobs(uploaded_file):
    """Process uploaded resume and find jobs"""
    with st.spinner("Analyzing your resume and searching for jobs..."):
        rag_system = st.session_state.rag_system
        documents = rag_system.load_document_with_pypdf(uploaded_file)

        if documents:
            full_text = " ".join(doc.page_content for doc in documents)
            prompt = f"Analyze this resume text and extract the following:\n- SKILLS: List relevant technical and soft skills (comma-separated)\n- JOB_INTERESTS: List job roles or industries of interest (comma-separated)\n- EXPERIENCE_LEVEL: One of 'entry', 'mid', 'senior'\n\nResume: {full_text}"
            analysis = rag_system.call_direct_gemini(prompt)

            skills = analysis.get("skills", [])
            job_interests = analysis.get("job_interests", [])
            location = st.session_state.get("last_location", "Unknown Location")

            if location != "Unknown Location":
                search_results = rag_system.search_jobs_with_custom_search_api_location(skills, job_interests, location)
            else:
                search_results = rag_system.search_jobs_with_custom_search_api(skills, job_interests)

            jobs = search_results.get("jobs", [])
            internships = search_results.get("internships", [])

            if jobs or internships:
                st.success("✅ Job search completed! Here are your recommendations based on your resume:")

                if jobs:
                    st.subheader("💼 Recommended Jobs")
                    for job in jobs:
                        with st.expander(f"{job['title']} at {job['company']} ({job['location']}) - Match: {job['match_score']}%"):
                            st.write(f"**Company:** {job['company']}")
                            st.write(f"**Location:** {job['location']}")
                            st.write(f"**Salary:** {job['salary']}")
                            st.write(f"**Description:** {job['description']}")
                            st.write(f"**Required Skills:** {', '.join(job['required_skills'])}")
                            if job['apply_link']:
                                st.markdown(f"[Apply Now]({job['apply_link']})", unsafe_allow_html=True)
                            else:
                                st.write("**Apply Link:** Not available")

                if internships:
                    st.subheader("🌱 Recommended Internships")
                    for internship in internships:
                        with st.expander(f"{internship['title']} at {internship['company']} ({internship['location']}) - Match: {internship['match_score']}%"):
                            st.write(f"**Company:** {internship['company']}")
                            st.write(f"**Location:** {internship['location']}")
                            st.write(f"**Salary:** {internship['salary']}")
                            st.write(f"**Description:** {internship['description']}")
                            st.write(f"**Required Skills:** {', '.join(internship['required_skills'])}")
                            if internship['apply_link']:
                                st.markdown(f"[Apply Now]({internship['apply_link']})", unsafe_allow_html=True)
                            else:
                                st.write("**Apply Link:** Not available")
            else:
                st.warning("⚠️ No jobs or internships found matching your resume.")
        else:
            st.error("❌ Failed to process the resume. Please upload a valid PDF.")

if __name__ == "__main__":
    main()
