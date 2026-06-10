import cv2
import numpy as np

def createMessageFrame(
        width=640,
        height=480,
        msg='no image data',
        savePath=None,
        textColor=(255, 255, 255),
        fontScale=1,
        thickness=2
    ):
    # Create a message frame
    messageFrame = np.zeros([height, width, 3], dtype='uint8')
    font = cv2.FONT_HERSHEY_SIMPLEX
    (textWidth, textHeight), baseline = cv2.getTextSize(msg, font, fontScale, thickness)
    origin = ((width - textWidth)//2, (height + textHeight)//2)
    messageFrame = cv2.putText(messageFrame, msg, origin, font, fontScale, textColor, thickness)
    if savePath is not None:
        cv2.imwrite(savePath, messageFrame)
    return messageFrame
