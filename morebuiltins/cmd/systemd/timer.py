import argparse
import sys
from typing import Any, Dict, Tuple
import subprocess
from pathlib import Path

__all__ = ["timer_handler"]


def run_systemctl(command: str, timer_name: str) -> int:
    """Execute systemctl command for timer"""
    cmd = ["systemctl", command, timer_name]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Successfully executed: systemctl {command} {timer_name}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return 1


def manage_timer_files(
    timer_name: str,
    timer_content: str = "",
    service_content: str = "",
    action: str = "create",
) -> int:
    """Manage systemd timer and its associated service file"""
    timer_path = Path("/etc/systemd/system") / f"{timer_name}.timer"
    service_path = Path("/etc/systemd/system") / f"{timer_name}.service"

    if action == "create":
        try:
            timer_path.write_text(timer_content)
            service_path.write_text(service_content)
            subprocess.run(["systemctl", "daemon-reload"])
            return 0
        except Exception as e:
            print(f"Error creating timer files: {e}")
            return 1
    elif action == "delete":
        try:
            if timer_path.exists():
                timer_path.unlink()
            if service_path.exists():
                service_path.unlink()
            subprocess.run(["systemctl", "daemon-reload"])
            return 0
        except Exception as e:
            print(f"Error removing timer files: {e}")
            return 1
    return 0


def create_timer_file(args: Dict[str, Any]) -> Tuple[str, str]:
    """Generate systemd timer and service file content"""
    # Generate timer content
    timer_content = ["[Unit]"]
    if args.get("Description"):
        timer_content.append(f"Description={args['Description']} (Timer)")

    timer_content.append("\n[Timer]")
    # Timer section configuration
    timer_params = [
        "OnCalendar",
        "OnBootSec",
        "OnUnitActiveSec",
        "OnUnitInactiveSec",
        "OnStartupSec",
        "AccuracySec",
        "RandomizedDelaySec",
        "Unit",
        "Persistent",
        "WakeSystem",
    ]
    for param in timer_params:
        if args.get(param):
            timer_content.append(f"{param}={args[param]}")

    timer_content.append("\n[Install]")
    timer_content.append(f"WantedBy={args.get('WantedBy', 'timers.target')}")

    # Generate service content
    service_content = ["[Unit]"]
    if args.get("Description"):
        service_content.append(f"Description={args['Description']}")

    service_content.append("\n[Service]")
    if not args.get("Type"):
        service_content.append("Type=oneshot")
    service_params = [
        "Type",
        "ExecStart",
        "User",
        "Group",
        "WorkingDirectory",
        "Environment",
        "EnvironmentFile",
    ]
    for param in service_params:
        if args.get(param):
            service_content.append(f"{param}={args[param]}")

    return "\n".join(timer_content), "\n".join(service_content)


def timer_handler():
    """Parse arguments and manage systemd timer files.
    If -enable or -disable is not set, print timer and service file content.

    Examples usage:

        1. enable & start timer
            - python -m morebuiltins.cmd.systemd.timer -name mytimer -enable -OnCalendar '*:0/15' -ExecStart '/bin/echo hello'
        2. disable & stop timer
            - python -m morebuiltins.cmd.systemd.timer -name mytimer -disable
        3. print timer and service file content
            - python -m morebuiltins.cmd.systemd.timer -name mytimer -OnCalendar '*:0/15' -ExecStart '/bin/echo hello'
    """

    parser = argparse.ArgumentParser(
        description="Generate and manage systemd timer files",
        formatter_class=argparse.RawTextHelpFormatter,
        usage=__doc__,
    )

    # Basic arguments
    parser.add_argument("-name", "--name", help="Timer name (required)", required=True)
    parser.add_argument(
        "-enable",
        "--enable",
        "-start",
        "--start",
        action="store_true",
        help="Create, enable and start timer, print content if not set",
    )
    parser.add_argument(
        "-disable",
        "--disable",
        "-stop",
        "--stop",
        action="store_true",
        help="Stop, disable and remove timer, print content if not set",
    )

    # Timer specific arguments
    parser.add_argument(
        "-OnCalendar",
        "--OnCalendar",
        help="""Calendar event expression. Common examples:
'*:0/15'          - Every 15 minutes
'*-*-* 02:00'     - Every day at 2am
'Mon *-*-* 09:00' - Every Monday at 9am
'*-*-1 00:00'     - First day of every month
'Sat,Sun 14:00'   - Weekends at 2pm
'Mon..Fri 09:00'  - Weekdays at 9am
daily             - Every day at midnight
weekly            - Every Monday at midnight
monthly           - Every first day of month
hourly            - Every hour at minute 0""",
    )

    parser.add_argument(
        "-OnBootSec",
        "--OnBootSec",
        help="""Time to wait after boot. Examples:
'15s'   - 15 seconds after boot
'5min'  - 5 minutes after boot
'1h'    - 1 hour after boot""",
    )

    parser.add_argument(
        "-OnUnitActiveSec",
        "--OnUnitActiveSec",
        help="""Time between activations. Examples:
'15min' - Run every 15 minutes
'1h'    - Run hourly
'1d'    - Run daily""",
    )

    parser.add_argument(
        "-OnStartupSec",
        "--OnStartupSec",
        help="""Time to wait after startup. Examples:
'1min'  - 1 minute after startup
'10min' - 10 minutes after startup""",
    )

    parser.add_argument(
        "-AccuracySec",
        "--AccuracySec",
        help="""Timer accuracy (default: 1min). Examples:
'1s'    - Second accuracy
'1min'  - Minute accuracy (default)
'5min'  - 5-minute accuracy (saves power)""",
    )

    parser.add_argument(
        "-RandomizedDelaySec",
        "--RandomizedDelaySec",
        help="""Random delay to add. Examples:
'30s'   - Add up to 30 seconds delay
'5min'  - Add up to 5 minutes delay""",
    )

    parser.add_argument(
        "-Persistent",
        "--Persistent",
        choices=["yes", "no"],
        help="Run missed executions: yes - run immediately, no - skip (default)",
    )

    parser.add_argument(
        "-WakeSystem",
        "--WakeSystem",
        choices=["yes", "no"],
        help="Wake from suspend: yes - wake up, no - wait for wake (default)",
    )

    # Service related arguments
    parser.add_argument("-Description", "--Description", help="Timer description")
    parser.add_argument("-ExecStart", "--ExecStart", help="Command to execute")
    parser.add_argument("-User", "--User", help="User to run service")
    parser.add_argument("-Group", "--Group", help="Group to run service")
    parser.add_argument("-Environment", "--Environment", help="Environment variables")
    parser.add_argument(
        "-EnvironmentFile", "--EnvironmentFile", help="Environment file"
    )
    parser.add_argument(
        "-WorkingDirectory", "--WorkingDirectory", help="Working directory"
    )

    args = parser.parse_args()

    if args.disable:
        run_systemctl("stop", f"{args.name}.timer")
        run_systemctl("disable", f"{args.name}.timer")
        return manage_timer_files(args.name, action="delete")

    elif args.enable:
        if not args.ExecStart:
            print("Error: -ExecStart is required for enabling timer")
            parser.print_help()
            return 1

        if not (
            args.OnCalendar
            or args.OnBootSec
            or args.OnUnitActiveSec
            or args.OnUnitInactiveSec
            or args.OnStartupSec
        ):
            print("Error: At least one timing option is required")
            parser.print_help()
            return 1

        timer_content, service_content = create_timer_file(vars(args))
        if manage_timer_files(args.name, timer_content, service_content, "create") != 0:
            return 1

        run_systemctl("enable", f"{args.name}.timer")
        run_systemctl("start", f"{args.name}.timer")
        return 0
    else:
        timer_content, service_content = create_timer_file(vars(args))
        print("=== Timer File ===")
        print(timer_content)
        print("\n=== Service File ===")
        print(service_content)
        return 0


if __name__ == "__main__":
    sys.exit(timer_handler())
