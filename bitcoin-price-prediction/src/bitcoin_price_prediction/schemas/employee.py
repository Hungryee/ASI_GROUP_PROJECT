
from pydantic import BaseModel

class Employee(BaseModel):
    name: str
    pesel: str
    salary: float