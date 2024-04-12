import importlib
import re
from pathlib import Path
from textwrap import indent

import morebuiltins


def make_docs():
    short_doc = ""
    with open("doc.md", "w", encoding="u8") as f:
        indent_text = " " * 4
        for index1, name in enumerate(morebuiltins.__all__, 1):
            module = importlib.import_module(name)
            print("=" * 22, file=f, end="\n\n")
            name1 = f"## {index1}. {name}\n\n"
            short_doc += name1
            print(name1, file=f)
            print("=" * 22, file=f, end="\n\n")
            for index2, name in enumerate(module.__all__, 1):
                member = vars(module)[name]
                doc = member.__doc__
                if doc:
                    title = f"{index1}.{index2} `{name}`"
                    lines = re.split("\n", doc, maxsplit=1)
                    head = "%s - %s" % (title, lines[0])
                    short_doc += head + "\n\n"
                    if len(lines) > 1:
                        tail = lines[1]
                    else:
                        tail = ""
                    line = f"\n{head}\n{tail}\n"
                    print(
                        indent(line, indent_text),
                        "\n---\n",
                        sep="",
                        file=f,
                    )
            short_doc += "\n"
    path = Path("README.md")
    old_readme = path.read_text("u8")
    start = old_readme.find("<!-- start -->\n") + len("<!-- start -->\n")
    end = old_readme.find("<!-- end -->")
    path.write_text(old_readme[:start] + short_doc + old_readme[end:], encoding="u8")


def main():
    make_docs()


if __name__ == "__main__":
    main()
