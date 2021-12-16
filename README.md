# wine_finder_api
## Introduction
We implement 'wine_finder_api' system to recommend wines for a stranger.
It performed by content-based filtering (K-Means clustering). 
The datasets used in this system are crawled from 'https://www.vivino.com/' and 'https://www.winespectator.com/'.

## How to use
### Method 'Search'
```python
df_wine, df_image_lnk = WineRecommender.search(wine_name='Zenato')
```
This method returns wine's information by name. (it can return when keyword contains in name.)

#### Parameters
* wine_name  
name of wine that you want to search.
  
#### Return
```python
# df_wine
                                       WineName  AvgScore  ...      Dry    Soft
156  Zenato Amarone Della Valpolicella Classico       4.4  ...  49.0026  20.055

[1 rows x 8 columns]


# df_image_lnk
                                       WineName                                          ImageLink
156  Zenato Amarone Della Valpolicella Classico  https://images.vivino.com/thumbs/Es37D9nmRfa3B...)
```

### Method 'recommend'
```python
df_top_wine, df_image_lnk = WineRecommender.recommend(light=50, smooth=0, dry=80, soft=0, top=10, threshold=4)
```
WineRecommender class doesn't have to be initialized. It just has 'recommend' static method.

#### Parameters
* light, smooth, dry, soft  
Each parameters are need a value between 0 and 100. They are used to finding similar wines or cluster.
  
* top  
This parameter is a number that you want to get the rows. (sort by ScoreCount and AvgScore descending)
  
* threshold  
This parameter is between 0 and 5 (`int` or `float`). It abandons the rows that has ratings score under threshold from wine dataframe.
  
#### Return
```python
# df_top_wine
                                              WineName  AvgScore  ScoreCount  
213         Zenato Amarone Della Valpolicella Classico       4.4       38621   
326             Tormaresca Primitivo Salento Torcicoda       4.0       23453   
144                            Domaine Bousquet Malbec       3.7       17970   
119                 Tenuta Delle Terre Nere Etna Rosso       3.9       13666   
104  Vigneti Del Vulture Aglianico Del Vulture Pian...       4.2       12971   

       Light   Smooth      Dry     Soft  Cluster  Similarity  
213  80.0000  20.9347  49.0026  20.0550        0    0.999957  
326  64.3827  20.3368  37.0940  11.5043        0    0.996450  
144  59.2256  24.6369  14.6889  33.2949        0    0.915605  
119  71.6743  42.7492  20.0338  33.6724        0    0.912588  
104  80.0000  39.2658  30.1710  22.0297        0    0.961557                                               WineName  \

# cluster : wine's cluster
# Similarity : cosine similarity with input factors.


# df_image_lnk
                                              WineName 
76   Vigneti Del Vulture Aglianico Del Vulture Pian...   
87                  Tenuta Delle Terre Nere Etna Rosso   
109                            Domaine Bousquet Malbec   
156         Zenato Amarone Della Valpolicella Classico   
233             Tormaresca Primitivo Salento Torcicoda   

                                             ImageLink  
76   https://images.vivino.com/thumbs/rdIlT73STcqes...  
87   https://images.vivino.com/thumbs/KBWsgxMeTB6I3...  
109  https://images.vivino.com/thumbs/1wFapPiVSN2fD...  
156  https://images.vivino.com/thumbs/Es37D9nmRfa3B...  
233  https://images.vivino.com/thumbs/SzuXju7yTWi16... 
```