import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

__all__ = ["service_handler"]


def run_systemctl(command: str, service_name: str, dry_run: bool = False) -> int:
    """Execute systemctl command"""
    cmd = ["systemctl", command, service_name]
    template = f"""
# {"=" * 40}
# Execute systemctl {command}
>>> {" ".join(cmd)}"""
    print(template)

    if dry_run:
        return 0

    print(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603  # fixed systemctl argv, list form, never shell-interpolated
        print(f"Successfully executed: systemctl {command} {service_name}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return 1


def _reload_daemon() -> int:
    """Run `systemctl daemon-reload`, returning 0 on success."""
    print("Reloading systemd daemon...")
    if subprocess.run(["systemctl", "daemon-reload"]).returncode != 0:  # noqa: S607  # systemctl is a system CLI resolved from PATH
        print("Error: systemctl daemon-reload failed")
        return 1
    return 0


def _create_service_file(service_path: Path, content: str, dry_run: bool) -> int:
    """Print the creation preview, then write the unit file unless dry-run."""
    posix_path = service_path.as_posix()
    template = f"""
# {"=" * 40}
# Create service file: {posix_path}
# (run as root)
>>> vim {posix_path}
{content}

# {"=" * 40}
# Reload systemd daemon (run as root)
>>> systemctl daemon-reload"""
    print(template)

    if dry_run:
        return 0

    print(f"Writing service file to {service_path}")
    try:
        service_path.write_text(content, encoding="utf-8")
    except Exception as e:
        print(f"Error creating service file: {e}")
        return 1
    return _reload_daemon()


def _delete_service_file(service_path: Path, dry_run: bool) -> int:
    """Print the removal preview, then unlink the unit file unless dry-run."""
    posix_path = service_path.as_posix()
    template = f"""
# {"=" * 40}
# Remove service file (run as root)
>>> rm {posix_path}

# {"=" * 40}
# Reload systemd daemon (run as root)
>>> systemctl daemon-reload"""

    if dry_run:
        print(template)
        return 0

    print(f"Removing service file {service_path}")
    try:
        if not service_path.exists():
            print(f"Service file {service_path} does not exist.")
            return 0
        service_path.unlink()
    except Exception as e:
        print(f"Error removing service file: {e}")
        return 1
    return _reload_daemon()


def manage_service_file(service_name: str, content: str = "", action: str = "create", dry_run: bool = False) -> int:
    """Manage systemd service file"""
    service_path = Path("/etc/systemd/system") / f"{service_name}.service"
    if action == "create":
        return _create_service_file(service_path, content, dry_run)
    if action == "delete":
        return _delete_service_file(service_path, dry_run)
    return 0


def create_service_file(args: Dict[str, Any]) -> str:  # noqa: C901  # linear systemd section builder; splitting the merge logic would obscure the emitted layout
    """Generate systemd service file content.

    Custom parameters of the form `Section__Key=value` are appended to the
    matching section header, which is looked up ignoring the leading newline
    of the generated headers (the first header, `[Unit]`, has none). Adding a
    key that is already present appends a second directive rather than
    replacing the first: for scalar keys the later line wins, for list-type
    keys the values accumulate.
    """
    # Validate required parameter
    if not args.get("ExecStart"):
        raise ValueError("Error: ExecStart is required for service file creation")

    content = ["[Unit]"]

    # Unit section configuration - keep only common params
    unit_params = ["Description", "After", "Requires", "Wants"]
    content.extend(f"{param}={args[param]}" for param in unit_params if args.get(param))

    content.append("\n[Service]")
    # Service section configuration - keep only common params
    service_params = [
        "Type",
        "ExecStart",
        "ExecStop",
        "User",
        "Group",
        "WorkingDirectory",
        "Restart",
        "RestartSec",
        "Environment",
        "EnvironmentFile",
        "StandardOutput",
        "StandardError",
    ]
    for param in service_params:
        value = args.get(param)
        if value:
            if isinstance(value, list):
                content.extend(f"{param}={v}" for v in value)
            else:
                content.append(f"{param}={args[param]}")

    content.append("\n[Install]")
    content.append(f"WantedBy={args.get('WantedBy', 'multi-user.target')}")
    # Install section directives which support multiple values, space separated
    for param in ("RequiredBy", "Also", "Alias"):
        value = args.get(param)
        if value:
            if isinstance(value, list):
                value = " ".join(value)
            content.append(f"{param}={value}")

    # Handle custom parameters
    custom_sections: dict = {}
    for key, value in args.items():
        if "__" in key and value is not None:
            section, param = key.split("__", 1)
            if section not in custom_sections:
                custom_sections[section] = []
            custom_sections[section].append(f"{param}={value}")

    for section, params in custom_sections.items():
        if not params:
            continue
        header = f"[{section}]"
        # match the header regardless of the newline the generator prepends,
        # otherwise the first element ("[Unit]") is never found and a second
        # [Unit] section is emitted
        section_index = None
        for index, line in enumerate(content):
            if str(line).strip() == header:
                section_index = index
                break
        if section_index is None:
            content.append(f"\n{header}")
            content.extend(params)
            continue
        # merge into the existing section instead of adding a duplicated
        # header: insert before the next section header, or at the end
        insert_at = len(content)
        for index in range(section_index + 1, len(content)):
            item = str(content[index]).strip()
            # a section header is a bare "[Name]" line; a directive such as
            # "Environment=[x]" must not be mistaken for one
            if item.startswith("[") and item.endswith("]") and "=" not in item:
                insert_at = index
                break
        content[insert_at:insert_at] = params

    return "\n".join(content)


def _print_error_and_help(parser, message: str) -> int:
    print(f"Error: {message}")
    parser.print_help()
    return 1


def _handle_disable(args) -> int:
    return_codes = [
        run_systemctl("stop", f"{args.name}.service", dry_run=args.dry_run),
        run_systemctl("disable", f"{args.name}.service", dry_run=args.dry_run),
        manage_service_file(args.name, action="delete", dry_run=args.dry_run),
    ]
    failed_codes = [code for code in return_codes if code]
    if failed_codes:
        print(f"Error: failed with return codes {return_codes}")
        return 1
    return 0


def _handle_enable(args, parser) -> int:
    if not args.ExecStart:
        return _print_error_and_help(parser, "-ExecStart is required for enabling service")

    service_content = create_service_file(vars(args))
    if manage_service_file(args.name, service_content, "create", dry_run=args.dry_run) != 0:
        return 1

    return_codes = [
        run_systemctl("enable", f"{args.name}.service", dry_run=args.dry_run),
        run_systemctl("start", f"{args.name}.service", dry_run=args.dry_run),
    ]
    failed_codes = [code for code in return_codes if code]
    if failed_codes:
        print(f"Error: failed with return codes {return_codes}")
        return 1
    return 0


def service_handler():
    """Generate and manage systemd service files

    Example usage:

    1. Create, enable and start service:
        python -m morebuiltins.cmd.systemd.service -name myservice -enable -Description "My service" -ExecStart "/bin/bash myscript.sh"
    2. Stop, disable and remove service:
        python -m morebuiltins.cmd.systemd.service -name myservice -disable
    3. Print service file content:
        python -m morebuiltins.cmd.systemd.service -name myservice -Description "My service" -ExecStart "/bin/bash myscript.sh" -Type simple
    4. Dry run (print actions without executing):
        python -m morebuiltins.cmd.systemd.service -name myservice -enable --dry-run -Description "My service" -ExecStart "/bin/bash myscript.sh"
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, usage=service_handler.__doc__)

    # Basic arguments
    parser.add_argument("-name", "--name", help="Service name")
    parser.add_argument(
        "-enable",
        "--enable",
        "-start",
        "--start",
        action="store_true",
        help="Create, enable and start service",
    )
    parser.add_argument(
        "-disable",
        "--disable",
        "-stop",
        "--stop",
        action="store_true",
        help="Stop, disable and remove service",
    )
    parser.add_argument(
        "--dry-run",
        "--test",
        action="store_true",
        help="Print actions without executing them",
    )

    # Unit section - common arguments
    parser.add_argument("-Description", "--Description", help="Service description")
    parser.add_argument("-After", "--After", help="Start after specified units")
    parser.add_argument("-Requires", "--Requires", help="Required units")
    parser.add_argument("-Wants", "--Wants", help="Wanted units")

    # Service section - common arguments
    parser.add_argument("-Type", "--Type", choices=["simple", "forking", "oneshot", "notify"])
    parser.add_argument("-ExecStart", "--ExecStart", help="Start command")
    parser.add_argument("-ExecStop", "--ExecStop", help="Stop command")
    parser.add_argument("-User", "--User", help="User to run service")
    parser.add_argument("-Group", "--Group", help="Group to run service")
    parser.add_argument("-WorkingDirectory", "--WorkingDirectory", help="Working directory")
    parser.add_argument("-Restart", "--Restart", choices=["no", "always", "on-success", "on-failure"])
    parser.add_argument("-RestartSec", "--RestartSec", help="Restart interval (seconds)")
    parser.add_argument("-Environment", "--Environment", help="Environment variables", action="append")
    parser.add_argument("-EnvironmentFile", "--EnvironmentFile", help="Environment file")
    parser.add_argument(
        "-StandardOutput",
        "--StandardOutput",
        help="Standard output, e.g. syslog, journal, append:/tmp/app.log, file:/tmp/app.log",
        default="",
    )
    parser.add_argument(
        "-StandardError",
        "--StandardError",
        help="Standard error, e.g. syslog, journal, append:/tmp/app.log, file:/tmp/app.log",
        default="",
    )

    # Install section arguments
    parser.add_argument("-WantedBy", "--WantedBy", default="multi-user.target")
    parser.add_argument("-RequiredBy", "--RequiredBy", help="Required by units (space separated)")
    parser.add_argument("-Also", "--Also", help="Also enable these units (space separated)")
    parser.add_argument("-Alias", "--Alias", help="Unit aliases (space separated)")

    # Support custom section parameters in format: -Section__Key=value
    # Example: -Network__IPForward=yes will add IPForward=yes under [Network] section
    args, unknown = parser.parse_known_args()

    # Handle unknown parameters
    for arg in unknown:
        if arg.startswith(("-", "--")) and "=" in arg:
            key = arg.lstrip("-").split("=")[0]
            value = arg.split("=", 1)[1]
            setattr(args, key, value)
    # Force dry-run on Windows by default
    if os.name == "nt":
        if not args.dry_run:
            print("[Info] Windows detected: Forcing --dry-run mode.")
            args.dry_run = True

    # Show help if name is not provided
    if not args.name:
        parser.print_help()
        return 0

    if args.disable:
        return _handle_disable(args)
    if args.enable:
        return _handle_enable(args, parser)
    if not args.ExecStart:
        return _print_error_and_help(parser, "-ExecStart is required for service file creation")
    service_content = create_service_file(vars(args))
    print(service_content)
    return 0


if __name__ == "__main__":
    sys.exit(service_handler())
