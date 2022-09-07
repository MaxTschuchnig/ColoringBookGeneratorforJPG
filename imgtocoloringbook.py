from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from skimage import feature
from skimage.morphology import dilation
from tqdm import tqdm
import glob

initialblurr = 1
contrastfactor = 1.5
cannyfactor = [1.5, 2, 3, 4, 5]
dilationsize = [2, 3, 5, 7]

filenames = glob.glob("images/*.jpg")
for cfilename in tqdm(filenames):
    for ccannyf in cannyfactor:
        for cdilationsize in dilationsize:
            image = Image.open(cfilename).convert('L')
            image = image.filter(ImageFilter.GaussianBlur(radius=initialblurr))

            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrastfactor)
            image = np.array(image)

            image = feature.canny(image, sigma=ccannyf)
            image = dilation(image, np.ones((cdilationsize, cdilationsize)))
            image = np.invert(image)

            # Plot with faces
            dpi = mpl.rcParams['figure.dpi']
            height, width = image.shape
            figsize = width / float(dpi), height / float(dpi)
            fig = plt.figure(figsize=figsize)
            plt.imshow(image, cmap='gray', interpolation='nearest')
            plt.axis('off')
            plt.savefig("results/{}_edges_canny_{}_dil_{}.jpg"
                        .format(cfilename.split('images\\')[1].split(".jpg")[0], ccannyf, cdilationsize),
                        bbox_inches='tight')
