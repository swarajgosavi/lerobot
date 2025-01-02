import cv2

cap = cv2.VideoCapture(0)  # here it throws an error


import json
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    retval, buffer_img = cv2.imencode('.jpg', frame)

    # resdata = base64.b64encode(buffer_img)

    # resdata = "data:image/png;base64,"+ str(resdata.decode("utf-8"))
    # PARAMS = {'image': resdata}

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()