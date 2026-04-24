import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# Your email credentials
SENDER_EMAIL = "s26099984@gmail.com"
APP_PASSWORD = "kots mjfo zges msve"


def send_attendance_mail(receiver_email, name):
    """Send attendance confirmation email to the user."""
    now = datetime.now()
    subject = f"Attendance Recorded - {name}"
    body = (
        f"Hello {name},\n\n"
        f"Your attendance has been successfully recorded.\n\n"
        f"Date: {now.strftime('%d-%m-%Y')}\n"
        f"Time: {now.strftime('%H:%M:%S')}\n\n"
        f"Thank you!"
    )

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    server = None
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)
        print(f"[OK] Email sent to {receiver_email} for {name}")
        return True
    except Exception as e:
        print(f"[ERR] Email error for {name}: {e}")
        return False
    finally:
        if server:
            try:
                server.quit()
            except Exception:
                pass