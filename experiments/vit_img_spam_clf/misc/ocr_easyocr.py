import easyocr
import matplotlib.pyplot as plt
from utils import get_images

if __name__ == "__main__":
    reader = easyocr.Reader(['ko'])
    batches = list(zip(*get_images()))
    for i, (img_path, image) in enumerate(batches):
        result = reader.readtext(img_path)
        result = [box for box in result if box[-1] > 0.9]
        print(f'\n[{i+1} / {len(batches)}] {img_path}')
        if result:
            print(result)
            plt.imshow(image)
            plt.show()
