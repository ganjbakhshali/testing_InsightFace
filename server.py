from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
from src.face_recognizer import Recognizer
from src.utils import *
import config

app = FastAPI()

# Mount the 'io' directory as static files to serve images
app.mount("/io", StaticFiles(directory="io"), name="io")

recognizer = Recognizer(model_name=config.model_name)
targets, names = load_face_bank("face_bank_CALFW.npy")

@app.post("/recognize")
async def recognize(file: UploadFile = File(...), update: bool = False, origin_size: bool = True,
                    tta: bool = False, show: bool = False, save: bool = False):
    contents = await file.read()
    image = cv2.imdecode(np.fromstring(contents, np.uint8), cv2.IMREAD_COLOR)
    
    if not origin_size:
        image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results, bboxes = recognizer.recognize(image_rgb, targets, tta)
    name = 'Unknown'
    for idx, bbox in enumerate(bboxes):
        if results[idx] != -1:
            name = names[results[idx] + 1]
        else:
            name = 'Unknown'
        image = draw_box_name(image, bbox.astype("int"), name)
    
    if show:
        cv2.imshow('face Capture', image)
        cv2.waitKey(0)
    output_file_path = f"io/output/{file.filename}"
    
    if save:
        cv2.imwrite(output_file_path, image)
    
    return {"message": "Recognition completed", "file_path": output_file_path, "name_recognized":name}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
