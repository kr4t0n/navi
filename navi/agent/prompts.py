# TODO: make prompt into langchain prompt template

DEFAULT_DESCRIBE_QUERY = "describe the webpage"

DEFAULT_EXPLORER_SYSTEM_PROMPT = """You are doing web navigation to explore the function and meaning of a specfied website.
After each action during exploration, you can see the webpage by a screenshot and know all the previous actions.
You need to describe the result of the current action and what you have observed.
"""

DEFAULT_EXPLORER_QUERY = "Generate an action to explore the website"

DEFAULT_EXPLORER_GROUNDED_QUERY = """Generate next action to explore the website, the action must performed on a provided element.
Try not to repeat previous actions. The generated action should serve a consistent goal with previous actions.
You should provide element index and detailed relative position of element on the screenshot.

Available potential elements are
{elements_desc}
"""

DEFAULT_TASK_QUERY = "Generate ONE and ONLY ONE action at a time to achieve the task {task}"

DEFAULT_TASK_GROUNDED_QUERY = """Generate next action to achieve the task {task}, the action must performed on a provided element.
You should provide element index and detailed relative position of element on the screenshot.

Available potential elements are
{elements_desc}
"""

DEFAULT_TASK_FEW_SHOT_GROUNDED_QUERY = """Generate next action to achieve the task {task}, the action must performed on a provided element.
Try not to repeat previous actions. If CLICK does not make any effect, you can try using HOVER instead.

Some example trajectories to achieve the task as references are:
{few_shots}

You should provide element index and detailed relative position of element on the screenshot.

Available potential elements are
{elements_desc}
"""

DEFAULT_TASK_CRITIC_QUERY = """Decide if the task {task} is achieved.

Some example trajectories to achieve the task as references are:
{few_shots}

The history performed actions are:
{actions}
"""

DEFAULT_MULTICHOICE_QUERY = """{action_repr}.

Available potential elements are
{elements_desc}

Select the element meet your need, ONLY return the desired index:
"""

DEFAULT_DIFF_QUERY = """According to the difference of the two screenshots caused by the action.
Summarize what the action does.
"""

DEFAULT_FINISH_QUERY = """Decide if the process is finished or not"""

DEFAULT_SUMMARY_QUERY = """
According to the full actions you have performed, come up with a task name and a task description.
The task name should be precised, while the description should be simple and useful.
"""
