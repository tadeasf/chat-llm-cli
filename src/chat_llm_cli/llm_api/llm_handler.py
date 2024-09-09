from typing import Dict, Any, List, Optional, Tuple
from chat_llm_cli.prompt.prompt import console
from prompt_toolkit import PromptSession
from chat_llm_cli.config.config import get_api_key, budget_manager
from litellm.budget_manager import BudgetManager
import litellm
from rich.panel import Panel
from rich.text import Text

SYSTEM_MARKDOWN_INSTRUCTION = """
# Role and Expertise
You are an expert software developer specializing in Python, TypeScript/JavaScript (preferring TS-bun), and occasionally Golang. Your code is clean, secure, and follows DRY principles. You prioritize simplicity over over-engineered solutions.

# Coding Principles
- Write modular, maintainable, and production-ready code
- Prioritize security in all implementations
- Follow best practices and check for potential bugs
- Suggest simpler solutions when possible

# Markdown Formatting
Always use proper Markdown formatting in your responses:

1. Headers: Use `#`, `##`, `###`, etc.
2. Emphasis: Use `*italic*` or `_italic_`, `**bold**` or `__bold__`
3. Lists: Use `-` or `*` for unordered, `1.`, `2.`, `3.` for ordered
4. Blockquotes: Use `>`
5. Code blocks: Use triple backticks with language specification:

```python
def example_function():
    print("This is a Python code block")
```

```javascript
function exampleFunction() {
    console.log("This is a JavaScript code block");
}
```

6. Inline code: Use single backticks, like `this`
7. Links: `[link text](URL)`
8. Images: `![alt text](image URL)`

# Conversation Structure
1. Review conversation history to avoid repetition
2. For recurring issues, suggest brainstorming instead of immediate fixes
3. Break down complex tasks into discrete changes
4. Suggest small tests after each stage

# Special Tags
Use these tags to structure your responses:
- <CODE_REVIEW>: Describe existing code functionality
- <PLANNING>: Outline changes, consider trade-offs, and suggest relevant frameworks
- <OUTPUT>: Produce agreed-upon code changes
- <SECURITY_REVIEW>: Analyze security implications for sensitive changes

# Response Guidelines
- Provide code examples only when necessary or requested
- Prefer explanations without code when possible
- Request clarification for ambiguities
- When naming by convention, use ::UPPERCASE::
- Maintain existing code style and use language-appropriate idioms
- Always specify the language after opening backticks in code blocks

Remember to consistently apply these guidelines throughout our conversation.
"""

# os.environ["LITELLM_LOG"] = "DEBUG"


def chat_with_context(
    config: Dict[str, Any],
    messages: List[Dict[str, str]],
    session: PromptSession,
    proxy: Optional[Dict[str, str]],
    show_spinner: bool,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Sends a message to the LLM with the given context and returns the response.

    Args:
        config: The configuration dictionary.
        messages: The list of messages to send to the LLM.
        session: The prompt session.
        proxy: The proxy configuration.
        show_spinner: Whether to show a spinner while waiting for the response.

    Returns:
        A tuple containing the response content and the response object, or None if an error occurred.
    """
    user = config["budget_user"]

    try:
        api_messages = messages.copy()
        api_key = get_api_key(config)

        # Handle Anthropic models
        if config["provider"] == "anthropic":
            for message in api_messages:
                if message["role"] == "assistant" and "prefix" in message:
                    message["content"] = f"{message['content']} {message['prefix']}"
                    del message["prefix"]

        # Ensure all messages have valid roles
        valid_roles = ["system", "assistant", "user", "function", "tool"]
        for message in api_messages:
            if message["role"] not in valid_roles:
                message["role"] = "user"  # Default to user if role is invalid

        completion_kwargs = {
            "model": config["model"],
            "messages": api_messages,
            "api_key": api_key,
        }

        if show_spinner:
            with console.status(
                "[bold #a6e3a1]Waiting for response...",
                spinner="bouncingBar",  # Catppuccin Green
            ) as status:
                response = litellm.completion(**completion_kwargs)
                status.update(
                    status="[bold #a6e3a1]Response received!"
                )  # Catppuccin Green

            response_content, response_obj = handle_response(
                response, budget_manager, config, user
            )
        else:
            response = litellm.completion(**completion_kwargs)
            response_content, response_obj = handle_response(
                response, budget_manager, config, user
            )

        if response_content is None:
            return None

        # Update cost and save data
        budget_manager.update_cost(user=user, completion_obj=response)
        budget_manager.save_data()

    except KeyboardInterrupt:
        return None
    except Exception as e:
        console.print(
            Panel(
                Text(f"An error occurred: {str(e)}", style="white"),
                title="Error",
                border_style="bold #f38ba8",  # Catppuccin Red
                expand=False,
            )
        )
        return None
    return response_content, response_obj


def handle_response(
    response: Any, budget_manager: BudgetManager, config: Dict[str, Any], user: str
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Handles the response from the LLM and updates the budget.

    Args:
        response: The response object from the LLM.
        budget_manager: The budget manager.
        config: The configuration dictionary.
        user: The user's name.

    Returns:
        A tuple containing the response content and the response object, or None if an error occurred.
    """
    try:
        budget_manager.update_cost(user=user, completion_obj=response)
    except Exception as budget_error:
        console.print(
            Panel(
                Text(f"Budget update error: {str(budget_error)}", style="white"),
                title="Error",
                border_style="bold #f38ba8",  # Catppuccin Red
                expand=False,
            )
        )

    if hasattr(response, "choices") and len(response.choices) > 0:
        response_content = response.choices[0].message.content
        usage = response.usage
        response_obj = {
            "choices": [
                {
                    "message": {
                        "content": choice.message.content,
                        "role": choice.message.role,
                    },
                    "finish_reason": choice.finish_reason,
                    "index": choice.index,
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
            "model": response.model,
        }
        return response_content, response_obj
    else:
        console.print(
            Panel(
                Text(f"Unexpected response format: {response!r}", style="white"),
                title="Error",
                border_style="bold #f38ba8",  # Catppuccin Red
                expand=False,
            )
        )
        return None, None
