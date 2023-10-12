import cv2
import matplotlib.pyplot as plt

img = cv2.imread('stopsign.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

stop_data = cv2.CascadeClassifier('stop_sign.xml')

found = stop_data.detectMultiScale(img_gray, minSize=(20, 20))
amount_found = len(found)

if amount_found != 0:

    for(x, y, width, height) in found:
        cv2.rectangle(img_rgb, (x, y), (x + width, y + height), (0, 0, 255), 5)
        
plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()