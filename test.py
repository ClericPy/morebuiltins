import doctest
import importlib
import sys
import time

sys.path.insert(0, "./")


def zipapp_module():
    from morebuiltins.zipapps.main import create_app

    create_app(
        "./morebuiltins",
        main="morebuiltins",
        output="./morebuiltins.pyz",
        compressed=True,
    )


def test_all():
    # test local usage and pyz usage
    for path in ["./", "./morebuiltins.pyz"]:
        sys.modules.pop("morebuiltins")
        sys.path.insert(0, path)

        import morebuiltins

        assert path.replace("./", "") in morebuiltins.__file__, morebuiltins.__file__

        for name in morebuiltins.__all__:
            module = importlib.import_module(name)
            print(
                time.strftime("%Y-%m-%d %H:%M:%S"),
                "[TEST]",
                module.__name__,
                flush=True,
            )
            if hasattr(module, "test"):
                module.test()
            else:
                result = doctest.testmod(module)
                if result.failed:
                    raise RuntimeError
            print(
                time.strftime("%Y-%m-%d %H:%M:%S"),
                "[PASS]",
                module.__name__,
                flush=True,
            )
            time.sleep(1)
        print("all test ok", flush=True)


def make_docs():
    import importlib
    import re
    from pathlib import Path
    from textwrap import indent

    import morebuiltins

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
                    doc = doc.lstrip()
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
    zipapp_module()
    test_all()


if __name__ == "__main__":
    main()
