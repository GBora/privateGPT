from pydantic import BaseModel


class Question(BaseModel):
    text: str
    language: str = "en"
