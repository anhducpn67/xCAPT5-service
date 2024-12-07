from fastapi import FastAPI, Form, HTTPException, Request
from pydantic import BaseModel
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = FastAPI()

# Store verification codes
verification_codes = {}

# Email configuration
SMTP_SERVER = "smtp.example.com"  # Replace with your SMTP server
SMTP_PORT = 587
EMAIL_ADDRESS = "your_email@example.com"  # Replace with your email address
EMAIL_PASSWORD = "your_password"  # Replace with your email password


# Use Pydantic BaseModel for consistent validation
class SubmitFormData(BaseModel):
    email: str
    sequenceA: str
    sequenceB: str
    target: str = None


@app.post("/submit_form")
async def submit_form(request: Request):
    try:
        form = await request.form()
        print(form)
        email = form.get("email")
        SEQUENCEA = form.get("SEQUENCEA")
        SEQUENCEB = form.get("SEQUENCEB")
        TARGET = form.get("TARGET", None)

        if not email or not SEQUENCEA or not SEQUENCEB:
            raise HTTPException(status_code=400, detail="All fields except 'ID' are mandatory.")

        # Generate a verification code
        verification_code = random.randint(100000, 999999)
        verification_codes[email] = verification_code

        # Send email
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = email
            msg['Subject'] = "Your xCAPT5 Verification Code"

            body = f"Your verification code is: {verification_code}. Please enter this code on the website to continue."
            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.send_message(msg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

        return {"message": "Verification email sent.", "email": email}

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid input: {str(e)}")


@app.post("/verify_code")
async def verify_code(request: Request):
    form = await request.form()
    email = form.get("email")
    code = form.get("code")

    if not email or not code:
        raise HTTPException(status_code=400, detail="Email and verification code are required.")

    try:
        code = int(code)  # Convert code to int for validation
    except ValueError:
        raise HTTPException(status_code=400, detail="Verification code must be a number.")

    if email not in verification_codes or verification_codes[email] != code:
        raise HTTPException(status_code=400, detail="Invalid verification code.")

    # Clear the verification code after successful verification
    del verification_codes[email]
    return {"message": "Verification successful. You can now process your request."}