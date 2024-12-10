DEFAULT_DESCRIBE_QUERY = "describe the webpage"

DEFAULT_EXPLORER_SYSTEM_PROMPT = """You are doing web navigation to explore the function and meaning of a specfied website.
After each action during exploration, you can see the webpage by a screenshot and know all the previous actions.
You need to describe the result of the current action and what you have observed.
"""

DEFAULT_EXPLORER_QUERY = "generate an action to explore the website"

DEFAULT_MULTICHOICE_QUERY = """{action_repr}.

Available potential elements are
{elements_desc}

Select the element meet your need, ONLY return the desired index:
"""

DEFAULT_DIFF_QUERY = """According to the difference of the two descriptions caused by the action.
Summarize what the action does.
"""