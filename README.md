# wine_finder_api
## Introduction
We implement 'wine_finder_api' system to recommend wines for a stranger.
It performed by content-based filtering (K-Means clustering). 
The datasets used in this system are crawled from 'https://www.vivino.com/' and 'https://www.winespectator.com/'.

## How to use
```python
df_top_wine, df_image_lnk = WineRecommender.recommend(light=50, smooth=0, dry=80, soft=0, top=10, threshold=4)
```
WineRecommender class doesn't have to be initialized. It just has 'recommend' static method.

### Parameters
* light, smooth, dry, soft  
Each parameters are need a value between 0 and 100. They are used to finding similar wines or cluster.
  
* top  
This parameter is a number that you want to get the rows. (sort by ScoreCount and AvgScore descending)
  
* threshold  
This parameter is between 0 and 5 (`int` or `float`). It abandons the rows that has ratings score under threshold from wine dataframe.
  
### Return
```python
# df_top_wine
                                              WineName  AvgScore  ScoreCount  \
213         Zenato Amarone Della Valpolicella Classico       4.4       38621   
326             Tormaresca Primitivo Salento Torcicoda       4.0       23453   
104  Vigneti Del Vulture Aglianico Del Vulture Pian...       4.2       12971   
26                  COS Cerasuolo Di Vittoria Classico       4.0        8710   
27                                       Shafer Merlot       4.2        5146   
330                                      Polkura Syrah       4.0        4337   
338                                      Polkura Syrah       4.0        4337   
92                                    Pahlmeyer Merlot       4.4        2666   
172                                      Oberon Merlot       4.0        2144   
234                            Limerick Lane Zinfandel       4.1        1500   

       Light   Smooth       Dry      Soft  cluster  
213  80.0000  20.9347  49.00260  20.05500        3  
326  64.3827  20.3368  37.09400  11.50430        3  
104  80.0000  39.2658  30.17100  22.02970        3  
26   75.6232  38.7995  20.53330  31.64040        3  
27   64.0293  17.7211  17.81770   8.19227        3  
330  80.0000  37.5359   9.50501  23.97120        3  
338  80.0000  37.5359   9.50501  23.97120        3  
92   63.2534  19.0811  16.88510  10.33250        3  
172  62.8807  16.7873  15.06190  10.37450        3  
234  80.0000  30.6511  24.76990  15.16260        3    

# df_image_lnk
                                              WineName
18                  COS Cerasuolo Di Vittoria Classico   
19                                       Shafer Merlot   
67                                    Pahlmeyer Merlot   
76   Vigneti Del Vulture Aglianico Del Vulture Pian...   
125                                      Oberon Merlot   
156         Zenato Amarone Della Valpolicella Classico   
170                            Limerick Lane Zinfandel   
233             Tormaresca Primitivo Salento Torcicoda   
236                                      Polkura Syrah   
242                                      Polkura Syrah   

                                             ImageLink  
18   https://images.vivino.com/thumbs/OB_z6iMASVuu4...  
19   https://images.vivino.com/thumbs/ElDpjEtgQG2DE...  
67   https://images.vivino.com/thumbs/1g7rWrVDRg-1y...  
76   https://images.vivino.com/thumbs/rdIlT73STcqes...  
125  https://images.vivino.com/thumbs/sjJL09pATD66j...  
156  https://images.vivino.com/thumbs/Es37D9nmRfa3B...  
170  https://images.vivino.com/thumbs/8jS6YA9-Rjqzr...  
233  https://images.vivino.com/thumbs/SzuXju7yTWi16...  
236  https://images.vivino.com/thumbs/pXpQB3IaSXSfY...  
242  https://images.vivino.com/thumbs/pXpQB3IaSXSfY...
```