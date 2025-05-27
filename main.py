import easyocr
from ultralytics import YOLO
import cv2


# Preprocessing from cropped license Plate
def preprocessPlate(plateFrameCrop):
    grayPlate = cv2.cvtColor(plateFrameCrop, cv2.COLOR_BGR2GRAY)
    blurPlate = cv2.GaussianBlur(grayPlate, (5,5), 0)
    plateThresh = cv2.adaptiveThreshold(blurPlate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return plateThresh

# Check above characters in license plate
def checkAboveCharsInPlate(plateL):
    checkL = True
    
    if len(plateL) == 2:
        if len(plateL[0]) != 2 or len(plateL[1]) != 2:
            checkL = False
        elif plateL[0].isnumeric() == False:
            checkL = False
        elif plateL[1][0].isalpha() == False or plateL[1][1].isnumeric() == False:
            checkL = False
    else:
        for i in range (len(plateL[0])):
            if i == 2:
                if plateL[0][i].isalpha() == False:
                    checkL = False
            elif plateL[0][i].isnumeric() == False:
                checkL = False
    
    return checkL

# Check under characters in license plate
def checkUnderCharsInPlate(plateR):
    checkR = True
    
    if len(plateR) == 2:
        if len(plateR[0]) != 3 or len(plateR[1]) != 2:
            checkL = False
        elif plateR[0].isnumeric() == False or plateR[1].isnumeric() == False:
            checkR = False
    else:
        if len(plateR[0]) != 5:
            checkR = False
        elif plateR[0].isnumeric() == False:
            checkR = False
    
    return checkR

# Check is correct format plate after recognizing
def checkCorrectFormAfterOcr(plateOcrRes):
    if len(plateOcrRes) != 2:
        return False
    elif (len(plateOcrRes[0]) < 4 and len(plateOcrRes[0]) > 5) or (len(plateOcrRes[1]) < 5 and len(plateOcrRes[1]) > 6):
        return False
    else:
        plateL = plateOcrRes[0].split("-")
        plateR = plateOcrRes[1].split(".")
        checkL, checkR = checkAboveCharsInPlate(plateL), checkUnderCharsInPlate(plateR)
        return checkL and checkR
        
# OCR from preprocessed license plate
def getCharOnPlate(plateFrameCrop, isAgree):
    plateOcrRes = None
    if isAgree == True:
        preprocessedPlate = preprocessPlate(plateFrameCrop)
        plateOcrRes = easyocr.Reader(['en']).readtext(preprocessedPlate, detail=0)
    else:
        plateOcrRes = easyocr.Reader(['en']).readtext(plateFrameCrop, detail=0)

    # Uppercase characters and remove space on the right of the right string (under characters in plate)
    for part in plateOcrRes:
        part = part.upper()
        part = part.strip()
    
    plateInfoRes = ""
    if plateOcrRes != [] and checkCorrectFormAfterOcr(plateOcrRes) == True:
        plateInfo = ' '.join(plateOcrRes)
        plateInfoRes = plateInfo.upper()
        
    return plateInfoRes

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
                        if plateOcrRes != "":
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
