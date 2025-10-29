import os
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads .env if present

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT") or "knowledge-collection-service/0.1"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- connect to Postgres ---
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="knowledgebase",
    user="stiw_user",
    password="stiw_pwd"
)
cursor = conn.cursor()
print("✅ Connected to Postgres and OpenAI")

