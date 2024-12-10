import json
import base64
import asyncio

from playwright.async_api import Playwright, Browser, BrowserContext, Page, Locator, async_playwright
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, BaseChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_openai.chat_models.base import BaseChatOpenAI

from navi.core.pw import list_action_elements, list_all_elements
from navi.agent.prompts import (
    DEFAULT_EXPLORER_SYSTEM_PROMPT,
    DEFAULT_EXPLORER_QUERY,
    DEFAULT_EXPLORER_GROUNDED_QUERY,
    DEFAULT_TASK_QUERY,
    DEFAULT_TASK_GROUNDED_QUERY,
    DEFAULT_TASK_FEW_SHOT_GROUNDED_QUERY,
    DEFAULT_TASK_CRITIC_QUERY,
    DEFAULT_DESCRIBE_QUERY,
    DEFAULT_MULTICHOICE_QUERY,
    DEFAULT_DIFF_QUERY,
    DEFAULT_FINISH_QUERY,
    DEFAULT_SUMMARY_QUERY,
)
from navi.agent.messages import Description, Action, ActionFailure, Difference, Summary
from navi.agent.formatter import (
    ActionGeneration,
    GroundedActionGeneration,
    DifferenceGeneration,
    FinishGeneration,
    SummaryGeneration,
    CriticGeneration,
)
from navi.plugins.knowledge import KnowledgeBase, Knowledge


class NaviAgent:
    playwright: Playwright
    browser: Browser
    context: BrowserContext
    page: Page
    homepage: str
    headless: bool = False
    # this is for the page waiting for initialization
    # useful when some page redirecting happened as first
    init_waiting: int = 30

    llm: BaseChatOpenAI
    history: list[BaseMessage] | None
    screenshots: list[str]
    # external knowledge base
    knowledge_base: KnowledgeBase | None
    knowledges: list[Knowledge]

    async def _handle_popup(self, popup: Page) -> None:
        await popup.wait_for_load_state()
        await asyncio.sleep(self.init_waiting)
        self.page = popup

    async def _setup(self) -> None:
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context(viewport=self.viewport)
        if self.cookies:
            await self.context.add_cookies(self.cookies)

        self.page = await self.context.new_page()
        self.page.on("popup", self._handle_popup)

    async def _browse(self, page: str) -> None:
        await self.page.goto(page, timeout=0)
        await asyncio.sleep(self.init_waiting)

    def __init__(
        self,
        homepage: str,
        llm: BaseChatOpenAI,
        cookies: list[dict] | None = None,
        headless: bool = False,
        viewport: dict = {"width": 1280, "height": 720},
        history: bool = True,
        knowledge_base: KnowledgeBase | None = None,
    ) -> None:
        self.homepage = homepage
        self.cookies = cookies
        self.headless = headless
        self.viewport = viewport

        self.llm = llm
        self.history = [] if history else None
        self.screenshots = []
        self.knowledge_base = knowledge_base
        self.knowledges = []

    async def start(self) -> None:
        await self._setup()
        await self._browse(self.homepage)

    async def stop(self) -> None:
        await self.browser.close()

    async def clear(self) -> None:
        await self._browse(self.homepage)
        if self.history is not None:
            self.history = []
        self.screenshots = []
        self.knowledges = []

    async def _take_screenshot(self) -> str:
        screenshot = await self.page.screenshot()
        return base64.b64encode(screenshot).decode()

    async def _save_screenshot(self) -> None:
        self.screenshots.append(await self._take_screenshot())

    async def _query(
        self,
        system_prompt: str | None = None,
        history: list[BaseMessage] | None = None,
        query: str | None = None,
        output_parser: BaseTransformOutputParser = StrOutputParser(),
    ) -> str | ActionGeneration | GroundedActionGeneration | FinishGeneration:
        messages: list[BaseMessage | BaseChatPromptTemplate] = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        if history:
            messages.extend(history)

        if query:
            messages.append(HumanMessagePromptTemplate.from_template("{query}"))
        else:
            raise ValueError("Please provide desired query.")

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm | output_parser
        result = await chain.ainvoke({"query": query})

        return result

    async def _query_with_screenshot(
        self,
        system_prompt: str | None = None,
        history: list[BaseMessage] | None = None,
        query: str | None = None,
        output_parser: BaseTransformOutputParser = StrOutputParser(),
    ) -> str | ActionGeneration | GroundedActionGeneration | FinishGeneration:
        screenshot = await self.page.screenshot()
        screenshot = base64.b64encode(screenshot).decode()

        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        if history:
            messages.extend(history)

        if query:
            template = HumanMessagePromptTemplate.from_template(
                template=[
                    {"type": "text", "text": "{query}"},
                    {"type": "image_url", "image_url": "data:image/jpeg;base64,{screenshot}"},
                ]
            )
            messages.append(template)
        else:
            raise ValueError("Please provide desired query.")

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm | output_parser
        result = await chain.ainvoke({"query": query, "screenshot": screenshot})

        return result

    async def _query_with_screenshots(
        self,
        system_prompt: str | None = None,
        history: list[BaseMessage] | None = None,
        query: str | None = None,
        output_parser: BaseTransformOutputParser = StrOutputParser(),
        screenshots: list[str] | None = None,
    ) -> str:
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        if history:
            messages.extend(history)

        if query:
            template = HumanMessagePromptTemplate.from_template(
                template=[
                    *[
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{{sc_{i}}}"}
                        for i in range(len(screenshots))
                    ],
                    {"type": "text", "text": "{query}"},
                ]
            )
            messages.append(template)
        else:
            raise ValueError("Please provide desired query.")

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm | output_parser
        result = await chain.ainvoke(
            {
                "query": query,
                **{f"sc_{i}": sc for i, sc in enumerate(screenshots)},
            }
        )

        return result

    async def describe(
        self,
        system_prompt: str | None = None,
        query: str | None = DEFAULT_DESCRIBE_QUERY,
        output_parser: BaseTransformOutputParser = StrOutputParser(),
    ) -> str:
        # take screenshot and save
        await self._save_screenshot()
        # query description with the last screenshot
        result = await self._query_with_screenshots(
            system_prompt=system_prompt,
            history=self.history,
            query=query,
            output_parser=output_parser,
            screenshots=[self.screenshots[-1]],
        )
        # concat history
        if self.history is not None:
            self.history.append(HumanMessage(content=query))
            self.history.append(Description(content=result))

        return result

    def save_knowledge(self, app_name: str, source: str) -> None:
        if self.knowledge_base is None:
            raise ValueError("You have to set knowledge base to interact with knowledge.")

        # convert history to knowledge
        knowledge = Knowledge.history_to_knowledge(
            app_name=app_name,
            source=source,
            history=self.history,
            screenshots=self.screenshots,
        )
        self.knowledge_base.insert(knowledge=knowledge)

    def load_knowledges(self, query: str, top_n: int = 1) -> list[Knowledge]:
        if self.knowledge_base is None:
            raise ValueError("You have to set knowledge base to interact with knowledge.")

        self.knowledges.extend(self.knowledge_base.retrieve(query=query, top_n=top_n))

    def get_few_shots(self) -> str:
        few_shots = [
            f"Task: {knowledge.task}\nTrajectories:\n{"\n".join(knowledge.trajectories)}\n"
            for knowledge in self.knowledges
        ]
        few_shots = "\n".join([few_shot.replace("{", "").replace("}", "") for few_shot in few_shots])
        return few_shots


class NaviActionAgent(NaviAgent):
    num_iters: int
    num_max_iters: int
    num_total_iters: int

    num_failures: int
    num_max_failures: int

    def __init__(
        self,
        num_max_iters: int,
        num_max_failures: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_max_iters = num_max_iters
        self.num_iters = 0
        self.num_total_iters = 0

        self.num_max_failures = num_max_failures
        self.num_failures = 0

    async def clear(self) -> None:
        await super().clear()
        self.num_iters = 0
        self.num_failures = 0

    async def _act(self, element: Locator, action: ActionGeneration) -> None:
        if action.action.upper() == "CLICK":
            await element.click()
        elif action.action.upper() == "HOVER":
            await element.hover()
        elif action.action.upper() == "TYPE":
            await element.fill(action.value)
        elif action.action.upper() == "ENTER":
            await element.press("Enter")

    async def _generate_grounded_action(
        self,
        elements_desc: list[str],
        system_prompt: str | None = DEFAULT_EXPLORER_SYSTEM_PROMPT,
        query: str | None = None,
        output_parser: BaseTransformOutputParser = PydanticOutputParser(pydantic_object=GroundedActionGeneration),
    ) -> tuple[GroundedActionGeneration, str]:
        # prepare query
        multichoice_elements_desc = "\n".join((f"{i}: {desc}" for i, desc in enumerate(elements_desc)))
        query = query.format(
            elements_desc=multichoice_elements_desc,
        )

        if hasattr(output_parser, "get_format_instructions"):
            query = f"{query}\n\n{output_parser.get_format_instructions()}"

        # generate grounded action
        result = await self._query_with_screenshots(
            system_prompt=system_prompt,
            history=self.history,
            query=query,
            output_parser=output_parser,
            screenshots=[await self._take_screenshot()],
        )
        result_repr = repr(result)

        element_desc = elements_desc[result.element_idx]
        result_repr = f"{result_repr} Element is {element_desc}."

        return result, result_repr

    async def _generate_ungrounded_action(
        self,
        system_prompt: str | None = DEFAULT_EXPLORER_SYSTEM_PROMPT,
        query: str | None = None,
        output_parser: BaseTransformOutputParser = PydanticOutputParser(pydantic_object=ActionGeneration),
    ) -> tuple[ActionGeneration, str]:
        # prepare query
        if hasattr(output_parser, "get_format_instructions"):
            query = f"{query}\n\n{output_parser.get_format_instructions()}"

        # generate action
        result = await self._query_with_screenshots(
            system_prompt=system_prompt,
            history=self.history,
            query=query,
            output_parser=output_parser,
            screenshots=[await self._take_screenshot()],
        )
        result_repr = repr(result)

        return result, result_repr

    async def _choose(
        self,
        action_repr: str,
        elements_desc: list[str],
        system_prompt: str | None = None,
        query: str | None = DEFAULT_MULTICHOICE_QUERY,
        output_parser: BaseTransformOutputParser = StrOutputParser(),
    ) -> int:
        multichoice_elements_desc = "\n".join((f"{i}: {desc}" for i, desc in enumerate(elements_desc)))

        query = query.format(
            action_repr=action_repr,
            elements_desc=multichoice_elements_desc,
        )
        result = await self._query(
            system_prompt=system_prompt,
            query=query,
            output_parser=output_parser,
        )

        return result

    async def _generate_action(
        self,
        query: str,
        grounded: bool = False,
        verbose: bool = False,
    ) -> tuple[Locator, ActionGeneration, str]:
        if grounded:
            # directly generate an action given a screenshot and a list of elements
            # list all elements
            elements, elements_desc = await list_all_elements(page=self.page)
            # generate next action
            next_action, next_action_repr = await self._generate_grounded_action(
                elements_desc=elements_desc,
                query=query,
            )
            if verbose:
                print(next_action_repr)
            # choosed element
            element = elements[next_action.element_idx]
        else:
            # first generate an action, then choose from a list of elements
            # generate next action according to query
            next_action, next_action_repr = await self._generate_ungrounded_action(
                query=query,
            )
            if verbose:
                print(next_action_repr)
            # list all elements related to next action
            elements, elements_desc = await list_action_elements(page=self.page, action=next_action.action)
            # mulichoice selection
            choice = int(await self._choose(next_action_repr, elements_desc))
            # choosed element
            element = elements[choice]

        return element, next_action, next_action_repr

    async def step(
        self,
        query: str,
        grounded: bool = False,
        verbose: bool = False,
    ) -> None:
        if grounded:
            # directly generate an action given a screenshot and a list of elements
            # list all elements
            elements, elements_desc = await list_all_elements(page=self.page)
            # generate next action
            next_action, next_action_repr = await self._generate_grounded_action(
                elements_desc=elements_desc,
                query=query,
            )
            if verbose:
                print(next_action_repr)
            # choosed element
            element = elements[next_action.element_idx]
        else:
            # first generate an action, then choose from a list of elements
            # generate next action according to query
            next_action, next_action_repr = await self._generate_ungrounded_action(
                query=query,
            )
            if verbose:
                print(next_action_repr)
            # list all elements related to next action
            elements, elements_desc = await list_action_elements(page=self.page, action=next_action.action)
            # mulichoice selection
            choice = int(await self._choose(next_action_repr, elements_desc))
            # choosed element
            element = elements[choice]
        # act
        try:
            # screenshot before action
            sc1 = await self._take_screenshot()
            # take action
            await self._act(element=element, action=next_action)
            # screenshot after action
            sc2 = await self._take_screenshot()
            # update history
            if self.history is not None:
                self.history.append(HumanMessage(content=query))
                self.history.append(Action(content=next_action.json()))
            # update screenshots
            self.screenshots.extend([sc1, sc2])
        except Exception:
            self.num_failures += 1
            # avoid repeatedly perform failed actions
            content = f"Failed to execute\n{next_action_repr}. Avoid this action."
            if verbose:
                print(content)
            if self.history is not None:
                self.history.append(ActionFailure(content=content))
        # add an iteration
        self.num_iters += 1
        self.num_total_iters += 1

    async def is_finished(
        self,
        system_prompt: str | None = DEFAULT_EXPLORER_SYSTEM_PROMPT,
        query: str | None = DEFAULT_FINISH_QUERY,
        output_parser: BaseTransformOutputParser = PydanticOutputParser(pydantic_object=FinishGeneration),
    ) -> bool:
        if self.num_iters >= self.num_max_iters or self.num_failures >= self.num_max_failures:
            return True

        if hasattr(output_parser, "get_format_instructions"):
            query = f"{query}\n\n{output_parser.get_format_instructions()}"

        result = await self._query(
            system_prompt=system_prompt,
            history=self.history,
            query=query,
            output_parser=output_parser,
        )
        result_repr = result.is_finished.upper() == "TRUE"
        return result_repr

    def get_actions(self) -> str:
        # prepare history actions
        actions = []
        for message in self.history:
            if isinstance(message, Action):
                action_content = json.loads(message.content)
                # convert action node into natural language description
                action_description = (
                    f"{len(actions) + 1}. "
                    f"{action_content['action']} element {action_content['element_name']}. "
                    f"The element is located at {action_content['element_position']}."
                )
                actions.append(action_description)

        return "\n".join(actions)


class NaviTaskAgent(NaviActionAgent):
    async def _criticize(
        self,
        task: str,
        system_prompt: str | None = None,
        query: str | None = DEFAULT_TASK_CRITIC_QUERY,
        output_parser: BaseTransformOutputParser = PydanticOutputParser(pydantic_object=CriticGeneration),
    ) -> bool:
        # critize if task has been achieved
        # prepare history actions
        actions = self.get_actions()
        # prepare query
        query = query.format(
            task=task,
            few_shots=self.get_few_shots(),
            actions=actions,
        )
        # final query
        if hasattr(output_parser, "get_format_instructions"):
            query = f"{query}\n\n{output_parser.get_format_instructions()}"

        # critize
        result = await self._query(
            system_prompt=system_prompt,
            history=None,  # we do not need history here
            query=query,
            output_parser=output_parser,
        )

        return result.achieved.upper() == "TRUE"

    async def run(
        self,
        task: str,
        grounded: bool = False,
        verbose: bool = False,
    ) -> bool:
        # HACK: load knowledge, only use the most relevant knowledge
        if self.knowledge_base is not None:
            self.load_knowledges(query=task, top_n=1)
        # main loop for task completion
        while True:
            if verbose:
                print(f"action {self.num_iters}:")
            # generate one action according to task
            if grounded:
                if self.knowledge_base is not None:
                    query = DEFAULT_TASK_FEW_SHOT_GROUNDED_QUERY
                else:
                    query = DEFAULT_TASK_GROUNDED_QUERY
            else:
                query = DEFAULT_TASK_QUERY
            # assign the task
            if grounded:
                if self.knowledge_base is not None:
                    query = query.format(task=task, few_shots=self.get_few_shots(), elements_desc="{elements_desc}")
                else:
                    query = query.format(task=task, elements_desc="{elements_desc}")
            else:
                query = query.format(task=task)
            # next element and action
            element, action, action_repr = await self._generate_action(
                query=query,
                grounded=grounded,
                verbose=verbose,
            )
            # take action
            try:
                # act
                await self._act(element=element, action=action)
                if self.history is not None:
                    # action generation history
                    self.history.append(Action(content=action.json()))
            except Exception:
                self.num_failures += 1
                # avoid repeatedly perform failed actions
                content = f"Failed to execute {action_repr}.\nAvoid this action."
                if verbose:
                    print(content)
                if self.history is not None:
                    self.history.append(ActionFailure(content=content))
            # add an iteration
            self.num_iters += 1
            self.num_total_iters += 1
            # finished?
            if await self.is_finished():
                break
        # critize if succeed
        result = await self._criticize(task=task)

        return result


class NaviSelfExploreAgent(NaviActionAgent):
    async def _diff(
        self,
        system_prompt: str | None = DEFAULT_EXPLORER_SYSTEM_PROMPT,
        query: str | None = DEFAULT_DIFF_QUERY,
        output_parser: BaseTransformOutputParser = PydanticOutputParser(pydantic_object=DifferenceGeneration),
        screenshots: list[str] | None = None,
        verbose: bool = False,
    ) -> DifferenceGeneration:
        # prepare query
        if hasattr(output_parser, "get_format_instructions"):
            query = f"{query}\n\n{output_parser.get_format_instructions()}"

        # use last two screenshots to find the difference
        result = await self._query_with_screenshots(
            system_prompt=system_prompt,
            history=self.history,
            query=query,
            output_parser=output_parser,
            screenshots=screenshots,
        )
        result_repr = repr(result)
        if verbose:
            print(result_repr)

        return result

    async def _summarize(
        self,
        system_prompt: str | None = DEFAULT_EXPLORER_SYSTEM_PROMPT,
        query: str | None = DEFAULT_SUMMARY_QUERY,
        output_parser: BaseTransformOutputParser = PydanticOutputParser(pydantic_object=SummaryGeneration),
        verbose: bool = False,
    ) -> str:
        # prepare query
        if hasattr(output_parser, "get_format_instructions"):
            query = f"{query}\n\n{output_parser.get_format_instructions()}"

        result = await self._query(
            system_prompt=system_prompt,
            history=self.history,
            query=query,
            output_parser=output_parser,
        )
        result_repr = repr(result)
        if verbose:
            print(result_repr)

        if self.history is not None:
            self.history.append(HumanMessage(content=query))
            self.history.append(Summary(content=result.json()))

        return result

    async def run(
        self,
        grounded: bool = False,
        verbose: bool = False,
    ) -> None:
        # main loop for one exploration
        while True:
            if verbose:
                print(f"action {self.num_iters}, total actions {self.num_total_iters}:")
            # generate one action
            query = DEFAULT_EXPLORER_GROUNDED_QUERY if grounded else DEFAULT_EXPLORER_QUERY
            # next element and action
            element, action, action_repr = await self._generate_action(
                query=query,
                grounded=grounded,
                verbose=verbose,
            )
            # take action
            try:
                # screenshot before action
                sc1 = await self._take_screenshot()
                # act
                await self._act(element=element, action=action)
                # screenshot after action
                sc2 = await self._take_screenshot()
                # effect caused by the action
                diff_query = DEFAULT_DIFF_QUERY
                difference = await self._diff(query=diff_query, screenshots=[sc1, sc2], verbose=verbose)
                # check difference
                if difference.is_effective.upper() == "TRUE":
                    self.screenshots.extend([sc1, sc2])
                    if self.history is not None:
                        # action generation history
                        self.history.append(HumanMessage(content=query))
                        self.history.append(Action(content=action.json()))
                        # difference generation history
                        self.history.append(HumanMessage(content=diff_query))
                        self.history.append(Difference(content=difference.json()))
            except Exception:
                self.num_failures += 1
                # avoid repeatedly perform failed actions
                content = f"Failed to execute {action_repr}.\nAvoid this action."
                if verbose:
                    print(content)
                if self.history is not None:
                    self.history.append(ActionFailure(content=content))
            # add an iteration
            self.num_iters += 1
            self.num_total_iters += 1
            # finished?
            if await self.is_finished():
                break
        await self._summarize(verbose=verbose)
