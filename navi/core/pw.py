from playwright.async_api import Page, Locator

DEFAULT_CLICKABLE_TYPES = [
    "a",
    "button",
    "select",
]
EXTRA_CLICKABLE_TYPES = [
    # google compatiable
    "input",
]
DEFAULT_EDITABLE_TYPES = [
    "input",
    "textarea",
]
EXTRA_EDITABLE_TYPES = []
ALL_TYPES = set(
    [
        *DEFAULT_CLICKABLE_TYPES,
        *EXTRA_CLICKABLE_TYPES,
        *DEFAULT_EDITABLE_TYPES,
        *EXTRA_EDITABLE_TYPES,
    ]
)

DEFAULT_SALIENT_ATTRS = [
    "alt",
    "aria-describedby",
    "aria-label",
    "aria-role",
    "input-checked",
    "label",
    "name",
    "option_selected",
    "placeholder",
    "readonly",
    "text-value",
    "title",
    "value",
]
EXTRA_SALIENT_ATTRS = [
    # lark compatiable
    "data-text",
]


async def generate_elements_desc(
    element_type: str,
    elements: list[Locator],
    salient_attributes: list[str] = DEFAULT_SALIENT_ATTRS,
    extra_salient_attributes: list[str] = EXTRA_SALIENT_ATTRS,
) -> list[str | None]:
    elements_desc = []
    for element in elements:
        if await element.is_hidden() or await element.is_disabled():
            element_desc = None
        else:
            attr_desc = []
            for attr in [*salient_attributes, *extra_salient_attributes]:
                attr_value = await element.get_attribute(attr)
                if attr_value:
                    attr_desc.append(f'{attr}="{attr_value.strip()}"')

            # get text
            if text := await element.inner_text():
                attr_desc.append(f'text="{text}"')

            attr_desc = " ".join(attr_desc)
            if attr_desc:
                element_desc = f"<{element_type}> {attr_desc}"
            else:
                element_desc = None
        elements_desc.append(element_desc)

    return elements_desc


async def list_editable_elements(
    page: Page,
    editable_types: list[str] = DEFAULT_EDITABLE_TYPES,
    extra_editable_types: list[str] = EXTRA_EDITABLE_TYPES,
) -> tuple[list[Locator], list[str]]:
    elements, elements_desc = [], []

    for editable_type in editable_types:
        editable_elements = await page.locator(editable_type).all()
        editable_elements_desc = await generate_elements_desc(
            element_type=editable_type,
            elements=editable_elements,
        )

        elements.extend(editable_elements)
        elements_desc.extend(editable_elements_desc)

    for editable_type in extra_editable_types:
        editable_elements = await page.locator(editable_type).all()
        editable_elements = [
            element for element in editable_elements if await element.get_attribute("contenteditable") == "true"
        ]
        editable_elements_desc = await generate_elements_desc(
            element_type=editable_type,
            elements=editable_elements,
        )

        elements.extend(editable_elements)
        elements_desc.extend(editable_elements_desc)

    # filter all elements without a proper description
    elements, elements_desc = zip(
        *filter(lambda x: x[1] is not None, zip(elements, elements_desc)),
    )

    return elements, elements_desc


async def list_clickable_elements(
    page: Page,
    clickable_types: list[str] = DEFAULT_CLICKABLE_TYPES,
    extra_clickable_types: list[str] = EXTRA_CLICKABLE_TYPES,
) -> tuple[list[Locator], list[str]]:
    elements, elements_desc = [], []

    for clickable_type in [*clickable_types, *extra_clickable_types]:
        clickable_elements = await page.locator(clickable_type).all()
        clickable_elements_desc = await generate_elements_desc(
            element_type=clickable_type,
            elements=clickable_elements,
        )

        elements.extend(clickable_elements)
        elements_desc.extend(clickable_elements_desc)

    # filter all elements without a proper description
    elements, elements_desc = zip(
        *filter(lambda x: x[1] is not None, zip(elements, elements_desc)),
    )

    return elements, elements_desc


async def list_action_elements(
    page: Page,
    action: str,
) -> tuple[list[Locator], list[str]]:
    if action == "CLICK":
        return await list_clickable_elements(page)
    elif action == "TYPE" or action == "ENTER":
        return await list_editable_elements(page)
    else:
        raise ValueError(f"Do not support to list elements related to action {action}.")


async def list_all_elements(page: Page) -> tuple[list[Locator], list[str]]:
    elements, elements_desc = [], []

    for all_type in ALL_TYPES:
        all_elements = await page.locator(all_type).all()
        all_elements_desc = await generate_elements_desc(element_type=all_type, elements=all_elements)

        elements.extend(all_elements)
        elements_desc.extend(all_elements_desc)

    # filter all elements without a proper description
    elements, elements_desc = zip(
        *filter(lambda x: x[1] is not None, zip(elements, elements_desc)),
    )

    return elements, elements_desc
