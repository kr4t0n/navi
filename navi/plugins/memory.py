import json

from langchain_core.messages.base import BaseMessage

from navi.agent.messages import Description, Action, Difference, Summary


def memory_to_traj(
    source: str,
    app_name: str,
    memory: list[BaseMessage],
) -> dict:
    # define traj
    traj = {
        "source": source,
        "app_name": app_name,
        "nodes": [],
    }

    # iterate over memory
    for message in memory:
        if isinstance(message, Action):
            action_content = json.loads(message.content)
            # convert action node to traj node
            node = {
                "obj_name": action_content["element_name"],
                "action": action_content["action"],
                "relative_position": action_content["element_position"],
                "variables": action_content["variables"],
            }
            traj["nodes"].append(node)
        elif isinstance(message, Difference):
            difference_content = json.loads(message.content)
            # difference observed from last action
            node = traj["nodes"][-1]
            node.update(
                {
                    "description": difference_content["description"],
                }
            )
        elif isinstance(message, Summary):
            traj["task"] = message.content
        else:
            continue

    return traj


def memory_to_markdown(
    memory: list[BaseMessage],
) -> str:
    # define markdown content
    markdown_content = []
    state_idx, action_idx = 0, 0

    # iterate over memory
    for message in memory:
        if isinstance(message, Description):
            markdown_content.append(f"## State {state_idx}")
            state_idx += 1
        elif isinstance(message, Action):
            markdown_content.append(f"## Action {action_idx}")
            action_idx += 1
        elif isinstance(message, Difference):
            markdown_content.append("### Action Effect")
        elif isinstance(message, Summary):
            markdown_content.append("## Task")
        else:
            continue

        markdown_content.append(message.content)

    return "\n".join(markdown_content)
