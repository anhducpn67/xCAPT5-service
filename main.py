import random

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins. Replace with specific origins for better security.
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods.
    allow_headers=["*"],  # Allow all headers.
)


# Store verification codes and outputs
verification_codes = {}
outputs = {}


# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
EMAIL_ADDRESS = "bacn27890@gmail.com"
EMAIL_PASSWORD = "zqxq vomm ypmk lddo"


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
        email = form.get("email")
        sequence_a = form.get("sequenceA")
        sequence_b = form.get("sequenceB")
        _ = form.get("target", None)

        if not email or not sequence_a or not sequence_b:
            raise HTTPException(status_code=400, detail="All fields except 'ID' are mandatory.")

        # Generate a verification code
        verification_code = random.randint(100000, 999999)
        verification_codes[email] = verification_code

        # Send email
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = email
            msg['Subject'] = "A-B - query received by xCAPT5"

            body = (f"Dear user,\n"
                    f"This email confirms that your have submitted A-B to xCAPT5.\n"
                    f"Your verification code is: {verification_code}. Please enter this code on the website to continue.\n"
                    f"\n"
                    f"AIDANTE Lab\n"
                    f"Department of Computational Science and Engineering\n"
                    f"VNU University of Engineering and Technology")
            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
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

    # Generate unique output code
    output_code = f"PPI{random.randint(10000, 99999)}"
    output_info = verification_codes[email]
    outputs[output_code] = {
        "sequenceA": output_info["sequenceA"],
        "sequenceB": output_info["sequenceB"],
        "result": f"Log(LR) = {round(random.uniform(4.0, 6.0), 3)}\nThe chains are predicted to interact.",
    }

    # Clear the verification code after successful verification
    del verification_codes[email]
    return {"message": "Verification successful.", "output_url": f"/output/{output_code}"}


@app.get("/output/{output_code}", response_class=HTMLResponse)
async def get_output(output_code: str):
    if output_code not in outputs:
        raise HTTPException(status_code=404, detail="Output not found.")

    output_data = outputs[output_code]
    sequence_a = output_data["sequenceA"]
    sequence_b = output_data["sequenceB"]
    result = output_data["result"]

    html_content = f"""
    <html>
        <head>
            <title>Result for {output_code}</title>
        </head>
        <body>
            <h1>PEPPI Results for {output_code}</h1>
            <p>{result}</p>
            <h2>Input Sequence in FASTA Format</h2>
            <div>
                <pre>{sequence_a}</pre>
                <pre>{sequence_b}</pre>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)