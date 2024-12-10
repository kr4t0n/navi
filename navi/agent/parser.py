from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class ActionGeneration(BaseModel):
    element: str = Field(
        description="element to be operated",
    )
    element_type: str = Field(
        description="element type to be operated, should be one of BUTTON, TEXTBOX, SELECTBOX, LINK",
    )
    action: str = Field(
        description="next action to perform, should be one of CLICK, TYPE, SELECT",
    )
    value: str = Field(
        description=(
            "if action is TYPE, specify the text to be typed, \n"
            "if action is SELECT, specify the option to be chosen, \n"
            "if action is CLICK, not value is needed, write 'None'\n"
        ),
    )