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
        self.server.login(self.user, self.pwd)
        return self

    def __exit__(self, *_):
        if self.server:
            return self.server.__exit__(*_)

    def _send_message(self, msg):
        if not self.server:
            raise RuntimeError("with context not entered")
        return self.server.send_message(msg)

    def prepare_attach(self, files: str, filename="files.zip"):
        with io.BytesIO() as zip_buffer:
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                total_size = 0
                for _path in files.split(","):
                    file_path = Path(_path)
                    if file_path.is_file():
                        zipf.write(file_path.as_posix(), arcname=file_path.name)
                        total_size += file_path.stat().st_size
                    else:
                        raise FileNotFoundError(file_path.as_posix())
                if not total_size:
                    return None
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
        encoding=None,
    ):
        to_mail_list = [i.strip() for i in to_emails.split(",")]
        if not to_mail_list or not to_mail_list[0]:
            raise ValueError("email is %s" % to_emails)
        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = ", ".join(to_mail_list)
        msg["Subject"] = title
        try:
            # get text from path
            b_path = Path(body)
            if b_path.is_file() and b_path.suffix in (".txt", ".md"):
                body = b_path.read_text(encoding=encoding)
        except (UnicodeDecodeError, OSError):
            pass
        if files:
            attach_part = self.prepare_attach(files, filename=filename)
        else:
            attach_part = None
        msg.attach(MIMEText(str(body), "plain"))
        if attach_part:
            msg.attach(attach_part)
        return self._send_message(msg)
