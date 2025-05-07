from fastapi import FastAPI, UploadFile
import tempfile 
import cv2 
from ultralytics import YOLO
from main import preprocessPlate, getCharOnPlate

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "OK!"}

@app.post("/violate")
async def uploadVideo(videoFile: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as mainVideo:
        mainVideo.write(await videoFile.read())
    
    videoPath = mainVideo.name
    modelPath = "best.pt"
    boxColor = (0, 0, 255)

    cap = cv2.VideoCapture(videoPath)
    detectModel = YOLO(modelPath)
    if not cap.isOpened():
        return {"error": "We can't open this video"}

    violationRes = []
    while cap.isOpened():
        isSuccess, vidFrame = cap.read()
        if isSuccess == True:
            isWithoutHelmet, plateBox = False, None # isWithoutHelmet: Check person (people) who do not wear helmet (default False) + plateBox: bounding box coordinate of license plate
            detectRes = detectModel(vidFrame)
    
            if detectRes[0].boxes != None:
                resCoordOfBox = detectRes[0].boxes.xyxy.cpu().numpy().astype(int)
                resCls = detectRes[0].boxes.cls.cpu().numpy().astype(int)
                resConf = detectRes[0].boxes.conf.cpu().numpy().astype(int)
                
                for box, clsIdx, conf in zip(resCoordOfBox, resCls, resConf):
                    if clsIdx == 3: # Without helmet
                        isWithoutHelmet = True
                    elif clsIdx == 0: # License plate
                        plateBox = box
    
                    if isWithoutHelmet == True and plateBox is not None:
                        x1,y1,x2,y2 = plateBox
                        plateFrameCrop = vidFrame[y1:y2, x1:x2] # crop plate
                        plateOcrRes = getCharOnPlate(plateFrameCrop, True) # OCR with agreeing to preprocess
        
                        cv2.rectangle(vidFrame, (x1, y1), (x2, y2), boxColor, 2)
                        if plateOcrRes != []:
                            violationRes.append(plateOcrRes)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    return {"results": violationRes}