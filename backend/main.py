import random

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.model import *

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
        prediction = predict(sequence_a, sequence_b, output_code)

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
    # Define the folder path and output.html template
    output_folder = f"backend/output/{output_code}"
    template_path = "frontend/output.html"

    # Check if the folder and template exist
    if not os.path.exists(output_folder):
        raise HTTPException(status_code=404, detail="Output folder not found.")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=500, detail="Template not found.")

    # Load sequence A and B from their files
    try:
        with open(os.path.join(output_folder, "A.seq"), "r") as file_a:
            sequence_a = file_a.read()
        with open(os.path.join(output_folder, "B.seq"), "r") as file_b:
            sequence_b = file_b.read()
        with open(os.path.join(output_folder, "result.txt"), "r") as file_result:
            prediction = file_result.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Sequence files not found.")

    # Load the HTML template and replace placeholders
    with open(template_path, "r") as template_file:
        html_template = template_file.read()

    # Replace placeholders in the template
    html_content = html_template.replace("{output_code}", output_code)
    html_content = html_content.replace("{sequence_a}", sequence_a)
    html_content = html_content.replace("{sequence_b}", sequence_b)
    html_content = html_content.replace("{prediction}", prediction)

    # Return the customized HTML response
    return HTMLResponse(content=html_content)

app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
