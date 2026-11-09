import os

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    print("Hello from langchain-course!")
    print("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))


if __name__ == "__main__":
    main()
