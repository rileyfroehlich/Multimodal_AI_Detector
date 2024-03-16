##### TODO Import models ######

# Imports
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

app = FastAPI(title="Please upload a file")

# Stolen from https://stackoverflow.com/questions/61641514/css-not-recognized-when-served-with-fastapi
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.absolute() / "static"),
    name="static",
)
templates = Jinja2Templates(directory="html")

# HTML form to upload files ##### TODO THIS COULD BE IMPROVED? ##### CAN I ADD CSS?
upload_form = FileResponse("")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request}
    )

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_type = file.filename.split(".")[-1]  # File extension determines next steps
    
    # Validate file type
    allowed_types = ['txt', 'pdf', 'docx', 'jpg', 'jpeg', 'png', 'heic', 'mp3', 'wav', 'm4a', 'flac']
    if file_type not in allowed_types:
        return {"error": "Invalid file type"}

    #Text
    if file_type in ["txt", "pdf", ".docx"]:
        contents = await file.read()
        return {"filename": file.filename, "file_contents": contents.decode()}
    #Image
    elif file_type in ["jpg", "jpeg", "png", "heic"]:
        return {"filename": file.filename, "file_type": "image"}
    #Audio
    elif file_type in ["mp3", "wav", "m4a", "flac"]:
        return {"filename": file.filename, "file_type": "audio"}
    #Other (Throw error)
    else:
        return {"filename": file.filename, "file_type": "unknown"}

#Main function for app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)