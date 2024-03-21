##### TODO Import models ######
from models.audio import audio_detection

# Imports
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from fastapi.responses import JSONResponse


app = FastAPI()

# Stolen from https://stackoverflow.com/questions/61641514/css-not-recognized-when-served-with-fastapi
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="html")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request}
    )

@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    file_type = file.filename.split(".")[-1]  # File extension determines next steps
    percent_score = -15

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
        file_type = "text"
        file_name = file.filename
        contents = contents.decode()
        percent_score = 15
    #Image
    elif file_type in ["jpg", "jpeg", "png", "heic"]:
        file_type = "image"
        file_name = file.filename
        percent_score = 10
    #Audio
    elif file_type in ["mp3", "wav", "m4a", "flac"]:
        percent_score = audio_detection(file.file, file_type)
#        try:
#            percent_score = audio_detection(file.file, file_type)
#        except:
#            return {"error": "audio_error"}
    #Other (Throw error) This should not run
    else:
        return {"filename": file.filename, "file_type": "unknown"}
    
    # Process file
    file_details = {
        "filename": file.filename,
        "file_type": file_type,
        "file_contents": "base64encodedimagestring"
    }

    response_data = {
        "file_details": file_details,
        "score": percent_score
    }

    return templates.TemplateResponse(
        "score.html",
        {
            "request": request,
            "data": response_data
        }
    )

#Main function for app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)