# ImageSpamClassifier  
## Usage  

- Collect data 
  - execute `s3_image.py`. It will download images from AWS S3 bucket.
- Caching data for sampling
  - execute `preprocess.py`. It will create `cache.py` that contains information about sampled images from whole datasets.
- Label data 
  - execute `labelilng.py`. You can label image manually.

## Misc.  

- `dimensionality_reduction.py`
  - flatten image features to vectorize. 
  - take `PCA` & `t-SNE` for image clustering.
- `clustering.py`
  - `KMeans` for vectorized images.
- `ocr_*.py`
  - sample for `pytesseract` and `EasyOCR`