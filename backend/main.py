import os
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


def model_predict(sequence_a, sequence_b, output_code):
    # Create a folder for the output
    output_folder = f"backend/output/{output_code}"
    os.makedirs(output_folder, exist_ok=True)

    # Save sequenceA to a file A.seq
    with open(os.path.join(output_folder, "A.seq"), "w") as file:
        file.write(sequence_a)

    # Save sequenceB to a file B.seq
    with open(os.path.join(output_folder, "B.seq"), "w") as file:
        file.write(sequence_a)


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

        # Model predict
        output_code = f"PPI{random.randint(10000, 99999)}"  # TODO: Check duplicate output code
        model_predict(sequence_a, sequence_b, output_code)

        # Send email
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = email
            msg['Subject'] = "A-B - query received by xCAPT5"

            body = (f"Dear user,\n"
                    f"This email confirms that your have submitted A-B to xCAPT5.\n"
                    f"You can track job status at /output/{output_code}\n"
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

        return {"message": "Job status sent.", "email": email}

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid input: {str(e)}")


@app.get("/output/{output_code}", response_class=HTMLResponse)
async def get_output(output_code: str):
    # Define the folder path for the output
    output_folder = f"backend/output/{output_code}"

    # Check if the folder exists
    if not os.path.exists(output_folder):
        raise HTTPException(status_code=404, detail="Output not found.")

    # Load sequences from files
    try:
        with open(os.path.join(output_folder, "A.seq"), "r") as file_a:
            sequence_a = file_a.read()
        with open(os.path.join(output_folder, "B.seq"), "r") as file_b:
            sequence_b = file_b.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Sequence files not found.")

    # Generate the HTML response
    html_content = f"""
    <html>
        <head>
            <title>Result for {output_code}</title>
        </head>
        <body>
            <h1>PEPPI Results for {output_code}</h1>
            <h2>Input Sequence in FASTA Format</h2>
            <div>
                <pre>{sequence_a}</pre>
                <pre>{sequence_b}</pre>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)
