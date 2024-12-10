from playwright.async_api import Page, Locator

DEFAULT_INTERACTIVE_TYPES = [
    "a",
    "button",
    "input",
    "select",
    "textarea",
]

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


async def generate_elements_desc(
    element_type: str,
    elements: list[Locator],
    salient_attributes: list[str] = DEFAULT_SALIENT_ATTRS,
) -> list[str | None]:
    elements_desc = []
    for element in elements:
        attr_desc = []
        for attr in salient_attributes:
            attr_value = await element.get_attribute(attr)
            if attr_value:
                attr_desc.append(f'{attr}="{attr_value.strip()}"')

        attr_desc = " ".join(attr_desc)
        if attr_desc:
            element_desc = f"<{element_type}> {attr_desc}"
        else:
            element_desc = None
        elements_desc.append(element_desc)

    return elements_desc


async def list_interactive_elements(
    page: Page,
    interative_types: list[str] = DEFAULT_INTERACTIVE_TYPES,
) -> tuple[list[Locator], list[str | None]]:
    elements, elements_desc = [], []
    for interative_type in interative_types:
        interactive_elements = await page.locator(interative_type).all()
        interactive_elements_desc = await generate_elements_desc(
            element_type=interative_type,
            elements=interactive_elements,
        )

        elements.extend(interactive_elements)
        elements_desc.extend(interactive_elements_desc)

    # filter all elements without a proper description
    elements, elements_desc = zip(
        *filter(lambda x: x[1] is not None, zip(elements, elements_desc)),
    )

    return elements, elements_desc