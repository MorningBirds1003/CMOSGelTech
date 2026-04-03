import cv2
# camera test code to find which index it is
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Index {i}: not available")
        continue

    ret, frame = cap.read()
    if ret and frame is not None:
        cv2.imshow(f"Camera {i}", frame)
        print(f"Showing camera index {i}. Press any key for next.")
        cv2.waitKey(0)
        cv2.destroyWindow(f"Camera {i}")
    else:
        print(f"Index {i}: opened but no frame")

    cap.release()

cv2.destroyAllWindows()