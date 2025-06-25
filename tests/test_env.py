from dotenv import load_dotenv
import os

load_dotenv()

print("TG_API_ID:", os.getenv("TG_API_ID"))
print("TG_API_HASH:", os.getenv("TG_API_HASH"))
print("Phone:", os.getenv("phone"))
