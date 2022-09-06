from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature
from tqdm import tqdm
import glob
import cv2

initialblurr = 1
contrastfactor = 1.5
cannyfactor = [2, 3, 4, 5, 7]

filenames = glob.glob("images/*.jpg")
for cfilename in tqdm(filenames):
    for ccannyf in cannyfactor:
        image = Image.open(cfilename).convert('L')
        image = image.filter(ImageFilter.GaussianBlur(radius=initialblurr))

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrastfactor)
        image = np.array(image)

        image = feature.canny(image, sigma=ccannyf)

        # Plot with faces
        plt.imshow(image)
        plt.savefig("results/{}_edges_canny_{}.jpg"
                    .format(cfilename.split('images\\')[1].split(".jpg")[0], ccannyf))

        # Plot without (detected) faces
        # Does not really work well... Maybe replace with deep segmentation using unet/pix2pix
        originalimg = cv2.imread(cfilename)
        gray = cv2.cvtColor(originalimg, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, minNeighbors=7)
        binaryimage = np.zeros((originalimg.shape[0], originalimg.shape[1]))
        for (x, y, w, h) in faces:
            binaryimage[y:y + h, x:x + w] = 255
            binaryimage = binaryimage.astype('float32')

            kernel = np.ones((9, 9), np.uint8)
            binaryimage = cv2.erode(binaryimage, kernel, iterations=3)
            binaryimage = np.invert(binaryimage.astype(bool))

        image = np.logical_and(image, binaryimage)
        plt.imshow(image)
        plt.savefig("results/{}_edges_canny_{}_nofaces.jpg"
                    .format(cfilename.split('images\\')[1].split(".jpg")[0], ccannyf))
