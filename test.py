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
        print("all test ok", flush=True)


def update_doc():
    from doc import make_docs

    make_docs()


def main():
    update_doc()
    zipapp_module()
    test_all()


if __name__ == "__main__":
    main()
