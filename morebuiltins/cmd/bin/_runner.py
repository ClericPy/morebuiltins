import sys
import runpy

# Usage:
#   python runner.py <module> [args...]
# Example:
#   python runner.py morebuiltins.cmd.log_server
# uvx:
#   uvx morebuiltins morebuiltins.cmd.log_server


import importlib.util


def runner(mod_name: str | None = None, argv: list[str] | None = None) -> None:
    # Read module name and args from command line if not provided
    if mod_name is None:
        if len(sys.argv) < 2:
            print("Usage: python runner.py <module> [args...]")
            raise SystemExit(2)
        mod_name = sys.argv[1]
        argv = sys.argv[2:]
    if argv is None:
        argv = []

    # Mimic `python -m <module>` by setting sys.argv
    orig_argv = sys.argv[:]
    sys.argv = [mod_name] + argv
    try:
        # Prefer private API to replicate `python -m` semantics (incl. package __main__)
        run_as_main = getattr(runpy, "_run_module_as_main", None)
        if run_as_main is not None:
            # We already set sys.argv above; prevent _run_module_as_main from altering it again
            run_as_main(mod_name, alter_argv=False)
            return

        # Fallback: manually handle package __main__
        spec = importlib.util.find_spec(mod_name)
        if spec is None:
            raise ModuleNotFoundError(f"No module named '{mod_name}'")

        target = mod_name
        if spec.submodule_search_locations is not None:
            # Package: run <pkg>.__main__
            main_name = f"{mod_name}.__main__"
            if importlib.util.find_spec(main_name) is None:
                raise ModuleNotFoundError(f"No module named '{main_name}'")
            target = main_name

        # Run target module as __main__
        runpy.run_module(target, run_name="__main__", alter_sys=True)
    finally:
        # Restore original argv to avoid side effects
        sys.argv = orig_argv


if __name__ == "__main__":
    runner()
