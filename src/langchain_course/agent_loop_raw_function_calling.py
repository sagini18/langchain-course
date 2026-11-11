import ollama

MAX_ITERATIONS = 10
MODEL = "llama3.2:3b"

# ----------------- @tool - Tool decorator -------------------------

def get_product_price(product:str) -> float:
    """Look up the price of the product in the catalog"""
    print(f"Executing  get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

def apply_discount(price:float, discount_tier:str) -> float:
    """ Apply a discount to the product price.

    Args:
        price: The actual numeric product price returned by
               get_product_price.
        discount_tier: The discount category. Must be one of:
                       bronze, silver, gold.

    Returns:
        The final price after applying the discount."""
    print(f"Executing apply_discount(price='{price}', discount_tier='{discount_tier}')")
    if not isinstance(price, (int, float)):
        raise TypeError("Price must be a numeric value")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# Difference 1: Without @tool, we must manually define JSON Schema for each function.
# This is exactly what LangChain's tool decorator generates automatically 
# from the function's type hints and docstring.

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of the product in the catalog",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'"
                    }
                },
                "required": ["product"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        }
    }
]

# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.

# --- Helper: traced Ollama call ---
# Difference 2: Without LangChain, we must manually trace LLM calls for LangSmith.

def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)
    
# ------------------------ Agent Loop -----------------------

# @traceable(name="LangChain Agent Loop") # name used in LangSmith
def run_agent(question:str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "You have access to two tools:\n"
                "1. get_product_price: Retrieves the actual product price.\n"
                "2. apply_discount: Applies a discount to a given price.\n\n"
                "WORKFLOW:\n"
                "Step 1: ALWAYS call get_product_price first.\n"
                "Step 2: WAIT for the tool result.\n"
                "Step 3: Use the returned numeric price from get_product_price as the "
                "price argument when calling apply_discount. you MUST provide "
                "the argument named 'price'. Never use 'discount' instead of 'discount_tier'. "
                "discount_tier must be one of: bronze, silver, gold.\n"
                "Step 4: WAIT for the discount result.\n"
                "Step 5: Present the final price to the user.\n\n"

                "Never call apply_discount before receiving the "
                "get_product_price result.\n"
                "Never invent or calculate prices yourself.\n"

                "IMPORTANT:\n"
                "Do not provide a final answer after get_product_price.\n"
                "You MUST call apply_discount before responding to the user.\n"
                "Never invent or calculate prices yourself."
            )
        },
        { "role": "user", "content": question}
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n---------Iteration {iteration} ---------")

        # Difference 3: ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer 
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        # Process only the FIRST tool call - force one tool per iteration
        tool_call = tool_calls[0]
        # Difference 4: Attribute access (.function.name) instead of dict access (.get("name"))
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"  [Tool selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} is not found")
        
        # Difference 7: Direct function call instead of tool.invoke()
        observation = tool_to_use(**tool_args)

        print(f"Tool Result: {observation}")

        messages.append(ai_message)
        messages.append({"role": "tool", "tool_name": tool_name, "content": str(observation)})

    print(f"ERROR: Max iterations reached without a final answer")
    return None


def main() -> None:
    print("Hello from langchain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
    print(result)

if __name__ == "__main__":
    main()