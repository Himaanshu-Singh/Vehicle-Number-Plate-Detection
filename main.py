import cv2
import imutils
import pytesseract

pytesseract.pytesseract.tesseracts_cmd = 'C:\\Program Files\\Tesseracts-OCR\\tesseracts.exe'
img = cv2.imread('c.jpg', cv2.IMREAD_COLOR)
img = imutils.resize(img, width=500)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)
# find contours from the edged image and keep only the largest
# ones, and initialize our screen contour
cnt, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
img1 = img.copy()
cv2.drawContours(img1, cnt, -1, (0, 255, 0), 3)
cv2.imshow("img1", img1)
cv2.waitKey(0)

cnt = sorted(cnt, key=cv2.contourArea, reverse=True)[:30]
screenCnt = None
img2 = img.copy()
cv2.drawContours(img2, cnt, -1, (0, 255, 0), 3)
cv2.imshow("img2", img2)
cv2.waitKey(0)
count = 0
idx = 7
# loop over contours
for c in cnt :

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4 :  # chooses contours with 4 corners
        screenCnt = approx
        x, y, w, h = cv2.boundingRect(c)  # finds co-ordinates of the plate
        new_img = img[y :y + h, x :x + w]
        cv2.imwrite('./' + str(idx) + '.png', new_img)  # stores the new image
        idx += 1
        break

cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("Final image with plate detected", img)
cv2.waitKey(0)
Cropped_loc = './7.png'
cv2.imshow("cropped", cv2.imread(Cropped_loc))
text = pytesseract.image_to_string(Cropped_loc, lang='eng')
print("Number is:" , text)
cv2.waitKey(0)
cv2.destroyAllWindows()
