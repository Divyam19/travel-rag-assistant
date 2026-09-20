"""Talk to the agent in the terminal. Usage: python -m travelrag.chat   (blank line or Ctrl-D quits)"""

from .agent import run_agent


def main() -> None:
    history: list[dict] = []
    print("Travel assistant. Ask about travel news, advisories, flights, destinations.")
    while True:
        try:
            question = input("\nyou> ").strip()
        except EOFError:
            break
        if not question:
            break
        result = run_agent(question, history, run_label="chat")
        print(f"\n{result.answer}")
        for s in result.sources:
            print(f"  [{s.n}] {s.title[:70]} ({s.source}) {s.url}")
        print(f"\n[{result.path}] " + " | ".join(result.trace)
              + f" | tokens {result.prompt_tokens}+{result.completion_tokens}")
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": result.answer}]


if __name__ == "__main__":
    main()
