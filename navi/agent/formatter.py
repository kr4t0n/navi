from langchain_core.pydantic_v1 import BaseModel, Field


class ActionGeneration(BaseModel):
    element: str = Field(
        description="element to be operated",
    )
    element_type: str = Field(
        description="element type to be operated, should be one of BUTTON, TEXTBOX, SELECTBOX, LINK",
    )
    action: str = Field(
        description="next action to perform, should be one of CLICK, HOVER, TYPE, ENTER, SELECT",
    )
    value: str = Field(
        description=(
            "if action is TYPE, specify the text to be typed, \n"
            "if action is ENTER, no value is needed, write 'None', \n"
            "if action is SELECT, specify the option to be chosen, \n"
            "if action is CLICK, no value is needed, write 'None'\n"
            "if action is HOVER, no value is needed, write 'None'\n"
        ),
    )

    def __repr__(self):
        return (
            f"My next action is to operate on element {self.element} "
            f"with type of {self.element_type}. The detailed action is {self.action} "
            f"with value of {self.value}."
        )


class GroundedActionGeneration(BaseModel):
    element_idx: int = Field(
        description="index of the element to be operated",
    )
    element_name: str = Field(
        description="a simple descriptive name of the element, do not use an instatiation name",
    )
    element_position: str = Field(
        description="relative position of the element in the screeshot",
    )
    variables: str = Field(
        description=(
            "variables inside the element, placeholder inside the element, if no placeholder is needed, write 'None'"
        ),
    )
    action: str = Field(
        description="next action to perform, should be one of CLICK, HOVER, TYPE, ENTER, SELECT",
    )
    value: str = Field(
        description=(
            "if action is TYPE, specify the text to be typed, \n"
            "if action is ENTER, no value is needed, write 'None', \n"
            "if action is SELECT, specify the option to be chosen, \n"
            "if action is CLICK, no value is needed, write 'None'\n"
            "if action is HOVER, no value is needed, write 'None'\n"
        ),
    )

    def __repr__(self):
        return (
            f"My next action is to operate on element idx {self.element_idx}. "
            f"The element name is {self.element_name}. "
            f"The element position is {self.element_position}. "
            f"The variables inside are {self.variables}. "
            f"The detailed action is {self.action} with value of {self.value}."
        )


class DifferenceGeneration(BaseModel):
    description: str = Field(
        description="detailed description of the outcome of operated element",
    )
    is_effective: str = Field(
        description="whether the action has any effect, return TRUE or FALSE",
    )

    def __repr__(self):
        return (
            f"The detailed description of the outcome: {self.description} "
            f"The outcome effectiveness: {self.is_effective}"
        )


class FinishGeneration(BaseModel):
    is_finished: str = Field(
        description="if the whole process is finished, return TRUE or FALSE",
    )


class SummaryGeneration(BaseModel):
    task_name: str = Field(description="task name")
    task_description: str = Field(description="task description")

    def __repr__(self):
        return (
            f"The overall task name is: {self.task_name}. "
            f"The detailed task description is: {self.task_description}."
        )


class CriticGeneration(BaseModel):
    achieved: str = Field(description="if the task is achieved by performed actions, return TRUE or FALSE")
