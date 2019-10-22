import cv2

# VideoCapture オブジェクトを取得
capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    cv2.imshow('Raw Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
