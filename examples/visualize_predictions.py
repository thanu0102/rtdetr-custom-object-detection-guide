import cv2

image = cv2.imread("sample_prediction.jpg")

cv2.imshow("Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()