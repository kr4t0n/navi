import json

from typing import TypeVar
from abc import ABC, abstractmethod
from langchain_core.messages.base import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field

from navi.agent.messages import Action, Summary

T = TypeVar("T", bound="Knowledge")


class Knowledge(BaseModel):
    app_name: str = Field(description="app name")
    app_version: str = Field(description="app version", default_factory=str)
    source: str = Field(description="knowledge source")
    task: str = Field(description="task name")
    task_description: str = Field(description="task description")
    trajectories: list[str] = Field(description="trajectories to complete the task")
    screenshots: list[str] = Field(description="trajectories screenshots", default_factory=list)

    @classmethod
    def from_json(
        cls: T,
        json_obj: dict,
    ) -> T:
        return cls(**json_obj)

    @classmethod
    def history_to_knowledge(
        cls: T,
        app_name: str,
        source: str,
        history: list[BaseMessage],
        screenshots: list[str],
    ) -> T:
        # TODO: support version in future, currently set an empty value
        app_version = ""
        trajectories = []
        task, task_description = "", ""

        # iterate over history to get full trajectories
        for message in history:
            if isinstance(message, Action):
                action_content = json.loads(message.content)
                # convert action node into natural language description
                action_description = (
                    f"{len(trajectories) + 1}. "
                    f"{action_content['action']} element {action_content['element_name']}. "
                    f"The element is located at {action_content['element_position']}."
                )
                trajectories.append(action_description)
            elif isinstance(message, Summary):
                summary_content = json.loads(message.content)
                # convert summary node into task and task_description
                task, task_description = summary_content["task_name"], summary_content["task_description"]

        knowledge = Knowledge(
            app_name=app_name,
            app_version=app_version,
            source=source,
            task=task,
            task_description=task_description,
            trajectories=trajectories,
            screenshots=screenshots,
        )

        return knowledge


class KnowledgeBase(ABC):
    @abstractmethod
    def insert(self, knowledge: Knowledge) -> dict: ...

    @abstractmethod
    def retrieve(self, query: str, top_n: int = 5) -> list[Knowledge]: ...
