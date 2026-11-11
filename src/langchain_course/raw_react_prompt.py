import re
import inspect
import ollama

MAX_ITERATIONS = 10
MODEL = "llama3.2:3b"

# ----------------- @tool - Tool decorator -------------------------


def get_product_price(product: str) -> float:
    """Look up the price of the product in the catalog"""
    print(f"Executing  get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"Executing apply_discount(price='{price}', discount_tier='{discount_tier}')")
    price = float(price)
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools = {"get_product_price": get_product_price, "apply_discount": apply_discount}

# CHANGE 3: Delete the JSON schemas. Tools now live inside the prompt as plain text.
# We derive descriptions from the functions themselves using inspect.


def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        signature = inspect.signature(tool_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)


tool_descriptions = get_tool_descriptions(tools)
tool_names = ",".join(tools.keys())

react_prompt = f"""
You are a ReAct agent. You can use the following tools to answer questions.

TOOLS:
{tool_descriptions}

AVAILABLE TOOL NAMES:
{tool_names}

RULES:
1. Never guess a product price. Retrieve it using get_product_price.
2. Apply discounts using apply_discount. Never calculate discounts yourself.
3. If the user does not specify a discount tier, ask which tier they want.
4. Examine previous observations before deciding what to do next.
5. You may call the same tool multiple times if necessary.
6. Do not repeat a tool call unless there is a reason.
7. When you have enough information to answer, return a Final Answer.
8. Do not invent tool results or observations.

RESPONSE FORMAT:

Thought: Explain briefly what you need to do.
Action: exact_tool_name
Action Input: arguments

OR

Thought: Explain briefly why the task is complete.
Final Answer: Your answer to the user.

IMPORTANT:
- Action must contain ONLY the exact tool name.
- Action Input must contain ONLY the arguments.
- Do not generate Observation yourself. The Python agent will provide it.
- After receiving an Observation, decide what to do next.
"""

# CHANGE 4: Drop tools= from ollama.chat(). The LLM has no idea it's an agent —
# all agency comes from the prompt above and our regex parsing below.


def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


# ------------------------ Agent Loop -----------------------


# @traceable(name="LangChain Agent Loop") # name used in LangSmith
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 60)

    # Maintain one conversation throughout the agent execution.
    messages = [
        {
            "role": "system",
            "content": react_prompt
        },
        {
            "role": "user",
            "content": question
        }
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):

        print(f"\n---------Iteration {iteration} ---------")

        # Ask the LLM to continue from the existing conversation.
        response = ollama_chat_traced(
            model=MODEL,
            messages=messages,
            options={"temperature": 0}
        )

        output = response.message.content.strip()

        print(f"LLM Output: {output}")

        # Store the LLM response in conversation history.
        messages.append({
            "role": "assistant",
            "content": output
        })

        # Check whether the LLM has completed the task.
        final_answer_match = re.search(
            r"Final Answer:\s*(.+)",
            output,
            re.DOTALL
        )

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()

            print("=" * 60)
            print(f"Final Answer: {final_answer}")

            return final_answer

        # Extract the tool name.
        action_match = re.search(
            r"^Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*$",
            output,
            re.MULTILINE
        )

        # Extract tool arguments.
        action_input_match = re.search(
            r"^Action Input:\s*(.+)$",
            output,
            re.MULTILINE
        )

        if not action_match or not action_input_match:

            print("[Parsing] Could not find a valid tool call.")

            # Ask the LLM to correct its response.
            messages.append({
                "role": "user",
                "content": (
                    "Your response did not contain a valid Action and "
                    "Action Input, or a Final Answer. Please continue "
                    "using the required response format."
                )
            })

            continue

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"[Tool selected] {tool_name}")
        print(f"[Raw arguments] {tool_input_raw}")

        # Parse the comma-separated arguments.
        raw_args = [
            x.strip()
            for x in tool_input_raw.split(",")
        ]

        args = [
            x.split("=", 1)[-1].strip().strip("'\"")
            for x in raw_args
        ]

        # Execute the selected tool.
        if tool_name not in tools:

            observation = (
                f"Error: Tool '{tool_name}' not found. "
                f"Available tools: {list(tools.keys())}"
            )

        else:

            print(f"[Tool Executing] {tool_name}({args})")

            try:
                observation = tools[tool_name](*args)
            except Exception as e:
                observation = f"Tool execution failed: {str(e)}"

        print(f"[Tool Result] {observation}")

        # Return the actual tool result to the LLM.
        messages.append({
            "role": "user",
            "content": f"Observation: {observation}\nContinue with the task."
        })

    print("ERROR: Maximum iterations reached without a final answer.")
    return None


def main() -> None:
    print("Hello from langchain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
    print(result)


if __name__ == "__main__":
    main()
