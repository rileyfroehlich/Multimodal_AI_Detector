##### TODO Import models ######

# Imports
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

app = FastAPI()

# HTML form to upload files
upload_form = """
<form method="post" enctype="multipart/form-data">
<input type="file" name="file">
<input type="submit">
</form>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return upload_form

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_type = file.filename.split(".")[-1]  # Get file extension to determine file type
    if file_type in ["txt", "pdf"]:
        contents = await file.read()
        return {"filename": file.filename, "file_contents": contents.decode()}
    elif file_type in ["jpg", "jpeg", "png"]:
        return {"filename": file.filename, "file_type": "image"}
    elif file_type in ["mp3", "wav"]:
        return {"filename": file.filename, "file_type": "audio"}
    else:
        return {"filename": file.filename, "file_type": "unknown"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)