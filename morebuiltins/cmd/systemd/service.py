import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

__all__ = ["service_handler"]


def run_systemctl(command: str, service_name: str) -> int:
    """Execute systemctl command"""
    cmd = ["systemctl", command, service_name]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Successfully executed: systemctl {command} {service_name}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return 1


def manage_service_file(
    service_name: str, content: str = "", action: str = "create"
) -> int:
    """Manage systemd service file"""
    service_path = Path("/etc/systemd/system") / f"{service_name}.service"

    if action == "create":
        try:
            service_path.write_text(content)
            subprocess.run(["systemctl", "daemon-reload"])
            return 0
        except Exception as e:
            print(f"Error creating service file: {e}")
            return 1
    elif action == "delete":
        try:
            if service_path.exists():
                service_path.unlink()
                subprocess.run(["systemctl", "daemon-reload"])
            return 0
        except Exception as e:
            print(f"Error removing service file: {e}")
            return 1
    return 0


def create_service_file(args: Dict[str, Any]) -> str:
    """Generate systemd service file content"""
    # Validate required parameter
    if not args.get("ExecStart"):
        raise ValueError("Error: ExecStart is required for service file creation")

    content = ["[Unit]"]

    # Unit section configuration - keep only common params
    unit_params = ["Description", "After", "Requires", "Wants"]
    for param in unit_params:
        if args.get(param):
            content.append(f"{param}={args[param]}")

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
                for v in value:
                    content.append(f"Environment={v}")
            else:
                content.append(f"{param}={args[param]}")

    content.append("\n[Install]")
    content.append(f"WantedBy={args.get('WantedBy', 'multi-user.target')}")

    # Handle custom parameters
    custom_sections: dict = {}
    for key, value in args.items():
        if "__" in key and value is not None:
            section, param = key.split("__", 1)
            if section not in custom_sections:
                custom_sections[section] = []
            custom_sections[section].append(f"{param}={value}")

    for section, params in custom_sections.items():
        if params:
            content.append(f"\n[{section}]")
            content.extend(params)

    return "\n".join(content)


def service_handler():
    """Generate and manage systemd service files

    Example usage:

    1. Create, enable and start service:
        python -m morebuiltins.cmd.systemd.service -name myservice -enable -Description "My service" -ExecStart "/bin/bash myscript.sh"
    2. Stop, disable and remove service:
        python -m morebuiltins.cmd.systemd.service -name myservice -disable
    3. Print service file content:
        python -m morebuiltins.cmd.systemd.service -name myservice -Description "My service" -ExecStart "/bin/bash myscript.sh" -Type simple
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, usage=service_handler.__doc__
    )

    # Basic arguments
    parser.add_argument(
        "-name", "--name", help="Service name (required)", required=True
    )
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

    # Unit section - common arguments
    parser.add_argument("-Description", "--Description", help="Service description")
    parser.add_argument("-After", "--After", help="Start after specified units")
    parser.add_argument("-Requires", "--Requires", help="Required units")
    parser.add_argument("-Wants", "--Wants", help="Wanted units")

    # Service section - common arguments
    parser.add_argument(
        "-Type", "--Type", choices=["simple", "forking", "oneshot", "notify"]
    )
    parser.add_argument("-ExecStart", "--ExecStart", help="Start command")
    parser.add_argument("-ExecStop", "--ExecStop", help="Stop command")
    parser.add_argument("-User", "--User", help="User to run service")
    parser.add_argument("-Group", "--Group", help="Group to run service")
    parser.add_argument(
        "-WorkingDirectory", "--WorkingDirectory", help="Working directory"
    )
    parser.add_argument(
        "-Restart", "--Restart", choices=["no", "always", "on-success", "on-failure"]
    )
    parser.add_argument(
        "-RestartSec", "--RestartSec", help="Restart interval (seconds)"
    )
    parser.add_argument(
        "-Environment", "--Environment", help="Environment variables", action="append"
    )
    parser.add_argument(
        "-EnvironmentFile", "--EnvironmentFile", help="Environment file"
    )
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
    parser.add_argument("-RequiredBy", "--RequiredBy", help="Required by units")
    parser.add_argument("-Also", "--Also", help="Also enable these units")
    parser.add_argument("-Alias", "--Alias", help="Unit aliases")

    # Support custom section parameters in format: -Section__Key=value
    # Example: -Network__IPForward=yes will add IPForward=yes under [Network] section
    args, unknown = parser.parse_known_args()

    # Handle unknown parameters
    for arg in unknown:
        if arg.startswith(("-", "--")) and "=" in arg:
            key = arg.lstrip("-").split("=")[0]
            value = arg.split("=")[1]
            setattr(args, key, value)

    if args.disable:
        run_systemctl("stop", f"{args.name}.service")
        run_systemctl("disable", f"{args.name}.service")
        return manage_service_file(args.name, action="delete")

    elif args.enable:
        if not args.ExecStart:
            print("Error: -ExecStart is required for enabling service")
            parser.print_help()
            return 1

        service_content = create_service_file(vars(args))
        if manage_service_file(args.name, service_content, "create") != 0:
            return 1

        run_systemctl("enable", f"{args.name}.service")
        run_systemctl("start", f"{args.name}.service")
        return 0
    else:
        service_content = create_service_file(vars(args))
        print(service_content)
        return 0


if __name__ == "__main__":
    sys.exit(service_handler())
