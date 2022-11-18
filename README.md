# TextDimensionalityReducer
Reduce text embeddings' dimension via Auto Encoder  

# Future Work  
- 전송 과정 중간에 바이트로 변환하면서 ```np.ndarray```의 shape정보가 사라는 현상 해결  
- 현재는 입력 벡터의 차원이 고정적으로 300차원이지만, 추후에는 동적으로 받아서 축소하도록 변경
- 현재는 오토인코더로 차원축소중인데, 오토인코더 이후로 나온 성능이 좋은 모델로 대체
- ```Dockerfile```, ```docker-compose.yml``` 파일 작성