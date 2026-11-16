import contextlib
import io
import smtplib
import typing
import zipfile
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

__all__ = ["SimpleEmail"]


class SimpleEmail:
    """SimpleEmail Sender.

    Demo::

        with SimpleEmail("smtp.gmail.com", 465, "someone@gmail.com", "PASSWORD") as s:
            print(
                s.send_message(
                    "This is Title",
                    "This is body text or file path(.md/.txt)",
                    "Author<someone@gmail.com>",
                    "anybody@gmail.com",
                    files="a.py,b.py,c.txt",
                    filename="files.zip",
                    encoding="u8",
                )
            )

    Notes:
        - encoding defaults to utf-8 and is used to read a .txt/.md body file;
          GBK or other non-utf-8 files must pass encoding explicitly.
        - When the body path exists, a read failure raises instead of silently
          sending the path string as the body.
        - Attachment files are zipped and 0-byte files are still attached;
          prepare_attach("") returns None (no attachment part).
    """

    def __init__(self, host, port, user, pwd, smtp_cls=smtplib.SMTP_SSL):
        self.host = host
        self.port = port
        self.user = user
        self.pwd = pwd
        self.smtp_cls = smtp_cls

        self.server = None

    def __enter__(self):
        self.server = self.smtp_cls(self.host, self.port)
        self.server.__enter__()
        try:
            self.server.login(self.user, self.pwd)
        except Exception:
            # login failed: close the connection instead of leaking it
            with contextlib.suppress(Exception):
                self.server.quit()
            raise
        return self

    def __exit__(self, *_):
        if self.server:
            return self.server.__exit__(*_)

    def _send_message(self, msg):
        if not self.server:
            raise RuntimeError("with context not entered")
        return self.server.send_message(msg)

    def prepare_attach(self, files: str, filename: typing.Union[str, tuple] = "files.zip"):
        # only an empty files string means no attachment; 0-byte files are
        # attached normally instead of being silently dropped
        if not files:
            return None
        with io.BytesIO() as zip_buffer:
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                for _path in [p.strip() for p in files.split(",")]:
                    file_path = Path(_path)
                    if file_path.is_file():
                        zipf.write(file_path.as_posix(), arcname=file_path.name)
                    else:
                        raise FileNotFoundError(file_path.as_posix())
            zip_buffer.seek(0)
            part = MIMEBase("application", "octet-stream")
            content = zip_buffer.read()
            part.set_payload(content)
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment", filename=filename)
            return part

    def send_message(
        self,
        title: str,
        body: typing.Union[str, Path],
        from_email: str,
        to_emails: str,
        files="",
        filename: typing.Union[str, tuple] = "files.zip",
        encoding="utf-8",
    ):
        to_mail_list = [i.strip() for i in to_emails.split(",")]
        if not to_mail_list or not to_mail_list[0]:
            raise ValueError(f"email is {to_emails}")
        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = ", ".join(to_mail_list)
        msg["Subject"] = title
        # get text from the file path if body is an existing .txt/.md file
        b_path = Path(body)
        try:
            is_file = b_path.is_file()
        except (OSError, ValueError):
            # body is not a usable path (e.g. plain text), keep it as the body
            is_file = False
        if is_file and b_path.suffix in (".txt", ".md"):
            # the file is confirmed to exist: read errors must propagate
            # instead of silently sending the path string as the body
            body = b_path.read_text(encoding=encoding)
        if files:
            attach_part = self.prepare_attach(files, filename=filename)
        else:
            attach_part = None
        msg.attach(MIMEText(str(body), "plain"))
        if attach_part:
            msg.attach(attach_part)
        return self._send_message(msg)
