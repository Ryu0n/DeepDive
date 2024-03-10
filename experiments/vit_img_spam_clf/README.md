# ImageSpamClassifier  
## Usage  

- Collect data 
  - execute `downloader_s3_image.py`. It will download images from AWS S3 bucket. (`instagram/date/*`)
  - execute `downloader_mongo_image.py`. It will download images from mongoDB. (`mongo_images/*`)
- Label data 
  - execute `labelilng.py`. You can label image manually.
- Model training & evaluation
  - execute `model.py`. It will save models' checkpoints per every epochs.

## Misc.  

- `dimensionality_reduction.py`
  - flatten image features to vectorize. 
  - take `PCA` & `t-SNE` for image clustering.
- `clustering.py`
  - `KMeans` for vectorized images.
- `ocr_*.py`
  - sample for `pytesseract` and `EasyOCR`