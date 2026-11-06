"""Pure key-name translation: agent vocabulary to Playwright key names.

``to_playwright_key`` maps the agent's key names (``up``, ``space``, ...)
to the names Playwright's ``keyboard`` API expects (``ArrowUp``, ``Space``,
...). Single characters and unknown names pass through unchanged, so a bad
name surfaces as a Playwright error rather than being silently remapped.
"""

_KEY_MAP: dict[str, str] = {
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "space": "Space",
    "enter": "Enter",
    "escape": "Escape",
    "shift": "Shift",
    "tab": "Tab",
}


def to_playwright_key(name: str) -> str:
    """Translate an agent key name to its Playwright key name.

    Known agent names are mapped; anything else is returned unchanged.
    """
    return _KEY_MAP.get(name, name)
