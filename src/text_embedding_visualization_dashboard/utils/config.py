from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()


class Configs(BaseModel):
    VECTOR_DB_HTTP_PORT: str = os.getenv("VECTOR_DB_HTTP_PORT", 8800)


cfg = Configs()
