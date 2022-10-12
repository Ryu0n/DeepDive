# reference : https://yunwoong.tistory.com/51
# reference : https://tariat.tistory.com/703
# brew install tesseract
# brew install tesseract-lang
import re
import cv2
import pytesseract
import matplotlib.pyplot as plt
from preprocess import get_images


def text_extract(image):
    config = r'--psm 4'
    text = pytesseract.image_to_string(image, config=config, lang='kor+eng')
    return text


if __name__ == "__main__":
    img_paths, images = get_images(num_images_per_dir=1, rescaled_pixels=800)
    for image in images:
        text = text_extract(image)
        if text:
            print('\n', '='*30)
            print(text)
            plt.imshow(image)
            plt.show()
