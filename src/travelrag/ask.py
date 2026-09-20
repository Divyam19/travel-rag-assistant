"""Ask one question. Usage: python -m travelrag.ask "your question" """

import sys

from .rag import answer_question


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage: python -m travelrag.ask "your question"')
    result = answer_question(" ".join(sys.argv[1:]))
    print(result.text)
    print("\nSources:")
    for s in result.sources:
        print(f"  [{s.n}] {s.similarity:.3f}  {s.title[:70]}  ({s.source})")
    print(f"\ntokens: prompt={result.prompt_tokens} completion={result.completion_tokens}")


if __name__ == "__main__":
    main()
