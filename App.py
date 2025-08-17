"""
Smart Job Recommender - Streamlit Cloud Deployment Version
=============================================

Updated version using Google Custom Search JSON API with GOOGLE_API_KEY and SEARCH_ENGINE_ID.
Compatible with streamlit>=1.48.0, requests>=2.32.0, google-generativeai==0.8.0, pypdf==5.9.0, selenium==4.10.0.

Main changes (v2.9):
- Added Selenium for accurate web scraping of job details (company, position, location, salary).
- Enhanced regex for extracting company, salary, and location from snippets to handle formats like "Company · Location".
- Added automation on "Apply Now" click: Send job and user data to n8n webhook for cover letter generation and spreadsheet storage.
- Retained light theme and styling.

Author: AI Assistant (updated)
Version: 2.9 (Added Selenium for Web Scraping)
"""

import streamlit as st
import requests
import time
import os
from typing import List, Dict, Any
import tempfile
from urllib.parse import quote_plus
import re  # For parsing salary, location, and company

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

# Import Selenium for web scraping
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    st.error("Selenium not available. Please install: pip install selenium==4.10.0 and ensure ChromeDriver is installed")

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
    """Enhanced RAG system for job recommendations using Gemini Flash and Selenium"""

    def __init__(self):
        self.gemini_client = None
        self.initialize_gemini()
        self.driver = None

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
                    st.success("AI system initialized successfully")
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

    def initialize_selenium(self):
        """Initialize Selenium WebDriver"""
        if not SELENIUM_AVAILABLE:
            st.error("Selenium not available for web scraping")
            return False

        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # Run in headless mode
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            service = Service(executable_path="/usr/bin/chromedriver")  # Adjust path as needed
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            st.success("✅ Selenium WebDriver initialized successfully")
            return True
        except Exception as e:
            st.error(f"❌ Error initializing Selenium: {e}")
            return False

    def cleanup_selenium(self):
        """Cleanup Selenium WebDriver"""
        if self.driver:
            self.driver.quit()

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
                text = page.extract_text_lines() if hasattr(page, 'extract_text_lines') else page.extract_text() if hasattr(page, 'extract_text') else None
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

    def extract_salary_from_snippet(self, snippet: str) -> str:
        """Extract salary from snippet using regex"""
        salary_pattern = r'(\$|USD|EUR)?\s*(\d{1,3}(?:,\d{3})?)(?:\s*-\s*\$?\d{1,3}(?:,\d{3})?)?(?:k|K| per year| annually| /yr)?'
        match = re.search(salary_pattern, snippet)
        return match.group(0) if match else "Not specified"

    def extract_location_from_snippet(self, snippet: str) -> str:
        """Extract location from snippet"""
        location_pattern = r'· ([\w\s,]+) \('
        match = re.search(location_pattern, snippet)
        return match.group(1) if match else "Unknown Location"

    def extract_company_from_snippet(self, snippet: str) -> str:
        """Extract company name from snippet if not available in metadata"""
        company_pattern = r'(\w+) ·'
        match = re.search(company_pattern, snippet)
        return match.group(1) if match else "Unknown Company"

    def scrape_job_details(self, apply_link: str) -> Dict[str, Any]:
        """Scrape job details using Selenium"""
        if not self.driver or not apply_link:
            return {"company": "Unknown Company", "title": "Unknown Title", "location": "Unknown Location", "salary": "Not specified"}

        try:
            self.driver.get(apply_link)
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            # Extract company name (example: look for common class names or tags)
            try:
                company = self.driver.find_element(By.CLASS_NAME, "company").text
            except:
                company = self.driver.find_element(By.XPATH, "//*[contains(text(), 'Company')]").text.split(':')[1].strip() if self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Company')]") else "Unknown Company"

            # Extract job title
            try:
                title = self.driver.find_element(By.TAG_NAME, "h1").text
            except:
                title = "Unknown Title"

            # Extract location
            try:
                location = self.driver.find_element(By.CLASS_NAME, "location").text
            except:
                location = self.driver.find_element(By.XPATH, "//*[contains(text(), 'Location')]").text.split(':')[1].strip() if self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Location')]") else "Unknown Location"

            # Extract salary
            try:
                salary = self.driver.find_element(By.CLASS_NAME, "salary").text
            except:
                salary = self.driver.find_element(By.XPATH, "//*[contains(text(), 'Salary')]").text.split(':')[1].strip() if self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Salary')]") else "Not specified"

            return {
                "company": company,
                "title": title,
                "location": location,
                "salary": salary
            }

        except Exception as e:
            st.warning(f"⚠️ Error scraping {apply_link}: {str(e)}")
            return {"company": "Unknown Company", "title": "Unknown Title", "location": "Unknown Location", "salary": "Not specified"}

    def search_jobs_with_custom_search_api(self, skills: List[str], job_interests: List[str]) -> Dict[str, List]:
        """Search jobs using Google Custom Search JSON API and enhance with Selenium"""
        if not self.initialize_selenium():
            return {"jobs": [], "internships": [], "search_queries": []}

        try:
            try:
                google_api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                search_engine_id = st.secrets.get("SEARCH_ENGINE_ID") or os.environ.get("SEARCH_ENGINE_ID")
            except Exception:
                google_api_key = os.environ.get("GOOGLE_API_KEY")
                search_engine_id = os.environ.get("SEARCH_ENGINE_ID")

            if not google_api_key or not search_engine_id:
                st.error("❌ Google API key and Search Engine ID required. Please add GOOGLE_API_KEY and SEARCH_ENGINE_ID to your Streamlit secrets.")
                self.cleanup_selenium()
                return {"jobs": [], "internships": [], "search_queries": []}

            search_queries = []
            if skills:
                primary_skills = skills[:3]  # Use top 3 skills for diversity
                for skill in primary_skills:
                    search_queries.extend([
                        f"{skill} developer jobs",
                        f"{skill} engineer jobs",
                        f"{skill} full-time jobs",
                        f"{skill} job openings"
                    ])

            if job_interests:
                for interest in job_interests[:3]:  # Use top 3 interests
                    search_queries.extend([
                        f"{interest} jobs",
                        f"{interest} careers",
                        f"{interest} opportunities"
                    ])

            if not search_queries:
                search_queries = ["software developer jobs", "python developer jobs", "data scientist jobs"]

            st.write(f"🔍 Generated search queries: {search_queries}")  # Debug

            all_jobs = []
            all_internships = []

            url = "https://www.googleapis.com/customsearch/v1"

            for query in search_queries[:5]:  # Increased to 5 queries
                try:
                    params = {
                        "key": google_api_key,
                        "cx": search_engine_id,
                        "q": query + " site:*.linkedin.com | site:*.indeed.com | site:*.glassdoor.com | site:*.monster.com | site:*.careerbuilder.com",
                        "num": 10,  # Max results per query
                        "safe": "off"  # Disable SafeSearch for broader results
                    }

                    st.info(f"🔍 Searching Google Custom Search for '{query}'...")

                    response = requests.get(url, params=params, timeout=15)
                    st.write(f"API Response Status: {response.status_code}")  # Debug
                    if response.status_code != 200:
                        st.warning(f"API Error: {response.text}")
                        continue

                    data = response.json()
                    st.write(f"API Response Items: {len(data.get('items', []))}")  # Debug

                    items_key = "items"
                    if items_key not in data or not data[items_key]:
                        st.warning(f"No results found for query: {query}")
                        continue

                    for item in data[items_key]:
                        snippet = item.get("snippet", "")
                        apply_link = self.get_best_apply_link(item, response_data=data)
                        scraped_data = self.scrape_job_details(apply_link) if apply_link else {}

                        job_data = {
                            "title": scraped_data.get("title", item.get("title", "Unknown Title")) or "Unknown Title",
                            "company": scraped_data.get("company", item.get("pagemap", {}).get("metatags", [{}])[0].get("og:site_name", "")) or self.extract_company_from_snippet(snippet) or "Unknown Company",
                            "location": scraped_data.get("location", self.extract_location_from_snippet(snippet)) or "Unknown Location",
                            "description": snippet or "No description",
                            "apply_link": apply_link,
                            "salary": scraped_data.get("salary", self.extract_salary_from_snippet(snippet)) or "Not specified",
                            "source": "Google Custom Search",
                            "match_score": self.calculate_match_score(skills, snippet),
                            "required_skills": self.extract_skills_from_description(snippet)
                        }

                        title_lower = (job_data["title"] or "").lower()
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
            self.cleanup_selenium()
            return {
                "jobs": unique_jobs[:20],
                "internships": unique_internships[:10],
                "search_queries": search_queries[:8]
            }

        except Exception as e:
            st.error(f"❌ Error with job search: {e}")
            self.cleanup_selenium()
            return {"jobs": [], "internships": [], "search_queries": []}

    def search_jobs_with_custom_search_api_location(self, skills: List[str], job_interests: List[str], location: str) -> Dict[str, List]:
        """Search jobs with location preference using Google Custom Search JSON API and enhance with Selenium"""
        if not self.initialize_selenium():
            return {"jobs": [], "internships": [], "search_queries": []}

        try:
            try:
                google_api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                search_engine_id = st.secrets.get("SEARCH_ENGINE_ID") or os.environ.get("SEARCH_ENGINE_ID")
            except Exception:
                google_api_key = os.environ.get("GOOGLE_API_KEY")
                search_engine_id = os.environ.get("SEARCH_ENGINE_ID")

            if not google_api_key or not search_engine_id:
                st.error("❌ Google API key and Search Engine ID required. Please add GOOGLE_API_KEY and SEARCH_ENGINE_ID to your Streamlit secrets.")
                self.cleanup_selenium()
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

            st.write(f"🔍 Generated location-based search queries: {search_queries}")  # Debug

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
                    st.write(f"API Response Status: {response.status_code}")  # Debug
                    if response.status_code != 200:
                        st.warning(f"API Error: {response.text}")
                        continue

                    data = response.json()
                    st.write(f"API Response Items: {len(data.get('items', []))}")  # Debug

                    items_key = "items"
                    if items_key not in data or not data[items_key]:
                        st.warning(f"No results found for query: {query}")
                        continue

                    for item in data[items_key]:
                        snippet = item.get("snippet", "")
                        apply_link = self.get_best_apply_link(item, response_data=data)
                        scraped_data = self.scrape_job_details(apply_link) if apply_link else {}

                        job_data = {
                            "title": scraped_data.get("title", item.get("title", "Unknown Title")) or "Unknown Title",
                            "company": scraped_data.get("company", item.get("pagemap", {}).get("metatags", [{}])[0].get("og:site_name", "")) or self.extract_company_from_snippet(snippet) or "Unknown Company",
                            "location": scraped_data.get("location", self.extract_location_from_snippet(snippet)) or "Unknown Location",
                            "description": snippet or "No description",
                            "apply_link": apply_link,
                            "salary": scraped_data.get("salary", self.extract_salary_from_snippet(snippet)) or "Not specified",
                            "source": "Google Custom Search",
                            "match_score": self.calculate_match_score(skills, snippet),
                            "required_skills": self.extract_skills_from_description(snippet)
                        }

                        title_lower = (job_data["title"] or "").lower()
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
            self.cleanup_selenium()
            return {
                "jobs": unique_jobs[:20],
                "internships": unique_internships[:10],
                "search_queries": search_queries[:8]
            }

        except Exception as e:
            st.error(f"❌ Error with location-based job search: {e}")
            self.cleanup_selenium()
            return {"jobs": [], "internships": [], "search_queries": []}

    def calculate_match_score(self, user_skills: List[str], job_description: str) -> int:
        """Calculate match percentage between user skills and job requirements"""
        if not user_skills or not job_description:
            return 0

        job_desc_lower = (job_description or "").lower()
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
        background-image: none; /* Remove any previous background image */
        color: #333333; /* Light black text for contrast */
    }

    /* Ensure sidebar matches light theme */
    .stSidebar {
        background-color: #FFFFFF; /* White sidebar */
        color: #333333; /* Light black text */
    }
    .stSidebar * {
        color: #333333;
    }

    /* Improve text readability */
    .stApp *, body *, .css-1aumxhk *, .st-emotion-cache-1aumxhk * {
        color: #333333; /* Consistent light black text */
        text-shadow: none;
    }

    /* Style buttons with light grey background and light black text */
    .stButton>button {
        background-color: #D3D3D3; /* Light grey background */
        color: #333333; /* Light black text */
        border: 1px solid #CCCCCC;
        border-radius: 5px;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #C0C0C0; /* Slightly darker grey on hover */
        color: #333333;
    }

    /* Success banner */
    [data-testid="stNotification"], .stAlert {
        background-color: #DFF5E1;
        color: #333333;
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

    /* Style selectbox (dropdown) for experience level to have white text */
    .stSelectbox div[role="listbox"] * {
        color: #FFFFFF !important;
    }
    .stSelectbox div[role="option"] {
        color: #FFFFFF !important;
        background-color: #333333; /* Dark background to contrast white text */
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
        3. Install ChromeDriver and ensure it's in PATH

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
        - Real-time matching
