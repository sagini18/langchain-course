from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage, ToolMessage

MAX_ITERATIONS = 10
MODEL = "llama3.2:3b"

# ----------------- @tool - Tool decorator -------------------------

@tool
def get_product_price(product:str) -> float:
    """Look up the price of the product in the catalog"""
    print(f"Executing  get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@tool
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
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)
    
# ------------------------ Agent Loop -----------------------

# @traceable(name="LangChain Agent Loop") # name used in LangSmith
def run_agent(question:str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
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
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n---------Iteration {iteration} ---------")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer 
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        # Process only the FIRST tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} is not found")
        
        observation = tool_to_use.invoke(tool_args)

        print(f"Tool Result: {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print(f"ERROR: Max iterations reached without a final answer")
    return None


def main() -> None:
    print("Hello from langchain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
    print(result)

if __name__ == "__main__":
    main()