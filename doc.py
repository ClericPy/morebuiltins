import importlib
import re
from textwrap import indent

import morebuiltins


def make_docs():
    with open("doc.md", "w", encoding="u8") as f:
        indent_text = " " * 4
        for index1, name in enumerate(morebuiltins.__all__, 1):
            module = importlib.import_module(name)
            print("=" * 22, file=f, end="\n\n")
            print(f"### {index1}.", name, file=f, end="\n\n")
            print("=" * 22, file=f, end="\n\n")
            for index2, name in enumerate(module.__all__, 1):
                member = vars(module)[name]
                doc = member.__doc__
                if doc:
                    title = f"{index1}.{index2} {name}"
                    lines = re.split("\n", doc, maxsplit=1)
                    head = lines[0]
                    if len(lines) > 1:
                        tail = lines[1]
                    else:
                        tail = ""
                    line = f"\n{title} - {head}\n{tail}\n"
                    print(
                        indent(line, indent_text),
                        "\n---\n",
                        sep="",
                        file=f,
                    )


def main():
    make_docs()


if __name__ == "__main__":
    main()
