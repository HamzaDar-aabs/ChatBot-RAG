from dotenv import load_dotenv
import os


load_dotenv()  # Load environment variables from .env file


class Config:
    MODEL_NAME = os.getenv("MODEL_NAME")  # Default model if not set
    