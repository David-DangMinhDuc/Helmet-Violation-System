import easyocr
from ultralytics import YOLO
import cv2


# Preprocessing from cropped license Plate
def preprocessPlate(plateFrameCrop):
    grayPlate = cv2.cvtColor(plateFrameCrop, cv2.COLOR_BGR2GRAY)
    blurPlate = cv2.GaussianBlur(grayPlate, (5,5), 0)
    plateThresh = cv2.adaptiveThreshold(blurPlate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return plateThresh

# OCR from preprocessed license plate
def getCharOnPlate(plateFrameCrop, isAgree):
    plateOcrRes = None
    if isAgree == True:
        preprocessedPlate = preprocessPlate(plateFrameCrop)
        plateOcrRes = easyocr.Reader(['en']).readtext(preprocessedPlate, detail=0)
    else:
        plateOcrRes = easyocr.Reader(['en']).readtext(plateFrameCrop, detail=0)
    
    return plateOcrRes

if __name__ == '__main__':
    modelPath = "best.pt"
    videoPath = "demo.mp4"
    nameCls = ['number plate', 'rider', 'with helmet', 'without helmet']
    boxColor = (0, 255, 0)

    cap = cv2.VideoCapture(videoPath)
    detectModel = YOLO(modelPath)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {videoPath}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

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
                        plateOcrRes = getCharOnPlate(plateFrameCrop, False) # OCR
        
                        cv2.rectangle(vidFrame, (x1, y1), (x2, y2), boxColor, 2)
                        if plateOcrRes != []:
                            cv2.putText(
                                vidFrame,
                                f"{' '.join(plateOcrRes)}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                boxColor,
                                2
                            )
                        else:
                            cv2.putText(
                                vidFrame,
                                "Unknown plate",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                boxColor,
                                2
                            )
        
            cv2.imshow(vidFrame)
            out.write(vidFrame)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
