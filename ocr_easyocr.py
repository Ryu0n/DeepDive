import easyocr
import matplotlib.pyplot as plt
from preprocess import get_images

if __name__ == "__main__":
    reader = easyocr.Reader(['ko'])
    batches = list(zip(*get_images(scaling=False)))
    for i, (img_path, image) in enumerate(batches):
        result = reader.readtext(img_path)
        result = [box for box in result if box[-1] > 0.9]
        print(f'\n[{i+1} / {len(batches)}] {img_path}')
        if result:
            print(result)
            plt.imshow(image)
            plt.show()
