import cv2

src_image = cv2.imread('/Users/matt/Documents/udacity-self-driving/Behavioral_Cloning/data/IMG/center_2017_07_16_20_20_24_765.jpg')
print(src_image.shape)

r = 100.0 / src_image.shape[1]
dim = (100, int(src_image.shape[0] * r))

# perform the actual resizing of the image and show it
resized = cv2.resize(src_image, dim, interpolation=cv2.INTER_AREA)

print(resized.shape)

cv2.imshow("resized", resized)
