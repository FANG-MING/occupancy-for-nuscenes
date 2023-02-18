# 3Doccupancy
3D occupancy 
# Dadaset
<img src="./assets/demo.gif" width="696px">

# Prediction
<img src="./assets/prediction.gif" width="696px">
## Evaluation Metrics
Leaderboard ranking for this challenge is by the intersection-over-union (mIoU) over all classes. 

### mIoU

Let $C$ be he number of classes. 

$$
    mIoU=\frac{1}{C}\displaystyle \sum_{c=1}^{C}\frac{TP_c}{TP_c+FP_c+FN_c},
$$

where $TP_c$ , $FP_c$ , and $FN_c$ correspond to the number of true positive, false positive, and false negative predictions for class $c_i$.
### Results in validate

| barrier | bicycle | bus | car | construction_vehicle | motorcycle | pedestrian | traffic_cone | trailer |  truck | driveable_surface | other_flat | sidewalk | terrain | manmade | vegetation |  miou |
| -- | --|--| -- | --|--|--|--|--|--|--|--| --|----------------------|---|------ | -------------------------------- |
| 15.12 | 8.55 | 28.78 | 28.06 | 10.36 | 13.42 | 9.22 | 4.57 | 17.38 | 22.56 | 48.38 | 22.57 | 29.11 | 25.81 | 16.22 |20.77 | 20.056  

[clik here download mini occupancy dataset for nuscenes v1.0-mini](https://drive.google.com/file/d/1n48IIy1poOOusHujyGhuDx_QZkPeI6ki/view?usp=sharing)