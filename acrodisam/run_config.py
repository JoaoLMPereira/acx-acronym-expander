'''
Run configuration class
'''
from typing import Optional

from pydantic import BaseModel

class RunConfig(BaseModel):
    """
    Configuration class with general specifications to run acronym expander server or experiments
    """

    name: str = ""
    save_and_load: bool = False
    persistent_articles: Optional[str] = None
