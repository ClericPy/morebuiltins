import ast
import re
import sys
from pathlib import Path

__all__ = ["parse_deps"]

try:
    stdlibs = set(sys.stdlib_module_names)
except AttributeError:
    # fix python3.9 sys.stdlib_module_names not found
    import os
    import pkgutil

    stdlib_path = os.path.dirname(os.__file__)
    stdlibs = {name for _, name, _ in pkgutil.iter_modules([stdlib_path])}

stds_libs = stdlibs | set(sys.builtin_module_names)


def parse_deps(project_dir, ignore_stds=True, format_path=True, pattern_list=("*.py",)):
    r"""Parse dependencies of a project directory, and find circular dependencies.

    Args:
          project_dir (str): Path to the project directory.
          ignore_stds (bool, optional): Whether to ignore dependencies from the standard library. Defaults to True.
          format_path (bool, optional): Whether to format the paths. Defaults to True.
          pattern_list (tuple, optional): List of patterns to match files. Defaults to ("*.py",).
    Returns:
            dict: A dictionary containing the project directory, circular dependencies, and dependencies.

    Demo::

        import json
        import multiprocessing
        from pathlib import Path

        project_dir = Path(multiprocessing.__file__).parent
        result = parse_deps(
            project_dir,
            ignore_stds=True,
            format_path=True,
            pattern_list=("*.py",),
        )
        dependencies = sorted(
            result["dependencies"].items(),
            key=lambda i: (len(i[1]), i[0]),
            reverse=True,
        )
        print("project_dir:", project_dir.as_posix(), flush=True)
        print("circular_dependency:", result["circular_dependency"], flush=True)
        for source, deps in dependencies:
            print(source, f"({len(deps)})", flush=True)
            for i in deps:
                print("\t", i, flush=True)
        # project_dir: D:/python311/Lib/multiprocessing
        # circular_dependency: [('./connection.py', './context.py'), ('./context.py', './forkserver.py'), ('./context.py', './managers.py'), ('./context.py', './popen_forkserver.py'), ('./context.py', './popen_spawn_posix.py'), ('./context.py', './popen_spawn_win32.py'), ('./context.py', './sharedctypes.py'), ('./context.py', './spawn.py'), ('./dummy/__init__.py', './pool.py')]
        # ./context.py (13)
        # 	 ./connection.py
        # 	 ./forkserver.py
        # 	 ./managers.py
        # 	 ./pool.py
        # 	 ./popen_fork.py
        # 	 ./popen_forkserver.py
        # 	 ./popen_spawn_posix.py
        # 	 ./popen_spawn_win32.py
        # 	 ./queues.py
        # 	 ./sharedctypes.py
        # 	 ./spawn.py
        # 	 ./synchronize.py
        # 	 ./util.py
        # ./synchronize.py (2)
        # 	 ./heap.py
        # 	 ./resource_tracker.py
        # ./resource_sharer.py (2)
        # 	 ./connection.py
        # 	 ./context.py
        # ./queues.py (2)
        # 	 ./synchronize.py
        # 	 ./util.py
        # ./pool.py (2)
        # 	 ./connection.py
        # 	 ./dummy/__init__.py
        # ./dummy/__init__.py (2)
        # 	 ./dummy/connection.py
        # 	 ./pool.py
        # ./util.py (1)
        # 	 test
        # ./spawn.py (1)
        # 	 ./context.py
        # ./sharedctypes.py (1)
        # 	 ./context.py
        # ./reduction.py (1)
        # 	 ./resource_sharer.py
        # ./process.py (1)
        # 	 ./context.py
        # ./popen_spawn_win32.py (1)
        # 	 ./context.py
        # ./popen_spawn_posix.py (1)
        # 	 ./context.py
        # ./popen_forkserver.py (1)
        # 	 ./context.py
        # ./managers.py (1)
        # 	 ./context.py
        # ./heap.py (1)
        # 	 ./context.py
        # ./forkserver.py (1)
        # 	 ./context.py
        # ./connection.py (1)
        # 	 ./context.py

    """

    def get_dependencies(file_path: Path, ignore_stds=True):
        dependencies: set = set()
        try:
            tree = ast.parse(file_path.read_text("utf-8", "ignore"))
        except SyntaxError:
            return dependencies
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name and name.split(".", 1)[0] not in stds_libs:
                        dependencies.add(name)
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ""
                if not name:
                    continue
                if ignore_stds and name.split(".", 1)[0] in stds_libs:
                    continue
                dependencies.add("." * (node.level) + name)
        return dependencies

    def fix_path(file_path: Path, name: str, project_dir: Path):
        match = re.match(r"^(\.+)(.+)", name)
        if match:
            # a/b/c.py + ..d => a/d.py
            # fix relative import .. to absolute import
            level = len(match[1])
            tail = match[2]
            base = file_path
            for _ in range(level):
                base = base.parent
            name = ".".join([base.name, tail])
        # name to path
        path_tail = name.replace(".", "/") + ".py"
        head_name, _, tail = path_tail.partition("/")
        dir_path = file_path.parent
        for _ in range(len(file_path.parts)):
            if dir_path.name == head_name:
                result_path = dir_path.joinpath(tail)
                if result_path.is_file():
                    result = "./" + result_path.relative_to(project_dir).as_posix()
                    return result
                else:
                    maybe_dir = result_path.with_suffix("")
                    if maybe_dir.is_dir():
                        maybe_file = maybe_dir.joinpath("__init__.py")
                        if maybe_file.is_file():
                            result = (
                                "./" + maybe_file.relative_to(project_dir).as_posix()
                            )
                            return result
            dir_path = dir_path.parent
        return name

    project_dir = Path(project_dir)
    if not project_dir.is_dir():
        raise ValueError(f"{project_dir} is not a directory")
    dependencies = {}
    for pattern in pattern_list:
        for file_path in project_dir.rglob(pattern):
            deps = get_dependencies(file_path, ignore_stds=ignore_stds)
            if deps:
                key = "./" + file_path.relative_to(project_dir).as_posix()
                if key in dependencies:
                    continue
                if format_path:
                    deps = {fix_path(file_path, name, project_dir) for name in deps}
                    deps = {i for i in deps if i}
                    if not deps:
                        continue
                dependencies[key] = sorted(deps, key=lambda i: (i.startswith("./"), i))
    cur_deps = set()
    for source, deps in dependencies.items():
        for dep in deps:
            if source in dependencies.get(dep, []):
                cur_deps.add(tuple(sorted([source, dep])))
    cur_deps = sorted(cur_deps)
    result = {
        "project_dir": project_dir.resolve().as_posix(),
        "circular_dependency": cur_deps,
        "dependencies": dependencies,
    }
    return result


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser()

    parser.add_argument("--project-dir", default=r".")
    parser.add_argument(
        "--ignore-stds",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to ignore dependencies from the standard library, default to 1",
    )
    parser.add_argument(
        "--format-path",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to format the paths, default to 1",
    )
    parser.add_argument(
        "--pattern-list",
        nargs="+",
        default=("*.py",),
        help="List of patterns to match files, default to ['*.py']",
    )
    parser.add_argument("--print-fmt", choices=["default", "json"], default="default")
    args = parser.parse_args()
    project_dir = Path(args.project_dir)
    result = parse_deps(
        project_dir,
        ignore_stds=args.ignore_stds,
        format_path=args.format_path,
        pattern_list=args.pattern_list,
    )

    if args.print_fmt == "json":
        print(json.dumps(result, ensure_ascii=False), flush=True)
    else:
        dependencies = sorted(
            result["dependencies"].items(),
            key=lambda i: (len(i[1]), i[0]),
            reverse=True,
        )
        print("project_dir:", project_dir.as_posix(), flush=True)
        print("circular_dependency:", result["circular_dependency"], flush=True)
        for source, deps in dependencies:
            print(source, f"({len(deps)})", flush=True)
            for i in deps:
                print("\t", i, flush=True)


if __name__ == "__main__":
    main()
