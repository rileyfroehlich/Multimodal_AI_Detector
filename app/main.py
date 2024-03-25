##### Import models ######
from models.audio import audio_detection
from models.image import image_pipeline
from models.text import text_pipeline

# Imports
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
import uvicorn
import base64

app = FastAPI()

# Stolen from https://stackoverflow.com/questions/61641514/css-not-recognized-when-served-with-fastapi
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="html")

#Home page
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request}
    )

@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    file_type = file.filename.split(".")[-1]  # File extension determines next steps
    file_contents = ''
    percent_score = -15 #Default negative percent indicates error
    ai_bool = False

    #Check for file upload
    if file_type == '':
        return templates.TemplateResponse(
            "index.html"
        )
    
    # Validate file type
    allowed_types = ['txt', 'pdf', 'docx', 'jpg', 'jpeg', 'png', 'heic', 'mp3', 'wav', 'm4a', 'flac']
    if file_type not in allowed_types:
        return {"error": "Invalid file type"}
    
    #Text
    if file_type in ["txt", "pdf", ".docx"]:
        contents = await file.read()
        ai_bool, percent_score = text_pipeline(file.file, file_type)
        file_contents = file_contents.decode()
        file_extension = file_type
        file_type = "text"
    #Image
    elif file_type in ["jpg", "jpeg", "png", "heic"]:
        #Put model_prediction here
        ai_bool, percent_score = image_pipeline(file.file, file_type)
        file_contents = base64.b64encode(await file.read()).decode("utf-8")
        file_extension = file_type
        file_type = "image"
    #Audio
    elif file_type in ["mp3", "wav", "m4a", "flac"]:
        ai_bool, percent_score = audio_detection(file.file, file_type)
        file_contents = base64.b64encode(await file.read()).decode("utf-8")
        file_extension = file_type
        file_type = 'audio'

    #Other (Throw error) This should not run
    else:
        return {"filename": file.filename, "file_type": "unknown"}
    
    #Format Percent Score if of type float
    percent_score = f"{percent_score:.1%}"

    #Dynamic data for score.html
    response_data = {
        "filename": file.filename,
        "file_extension": file_extension,
        "file_type": file_type,
        "file_contents": file_contents,
        "ai_bool": ai_bool,
        "confidence": percent_score
    }

    #Load scores.html with dynamic data
    return templates.TemplateResponse(
        "score.html",
        {
            "request": request,
            "data": response_data
        }
    )

# Exception handler for RequestValidationError (e.g., validation errors in request body)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return templates.TemplateResponse("error.html", {"request": request, "error_message": "Validation error"})

# Exception handler for HTTPException (e.g., 404 errors)
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return templates.TemplateResponse("error.html", {"request": request, "error_message": exc.detail})

# Generic exception handler for all other unhandled errors
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return templates.TemplateResponse("error.html", {"request": request, "error_message": "An unexpected error occurred"})

#Main function for app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)