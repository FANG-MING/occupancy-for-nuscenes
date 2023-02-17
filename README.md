# 3Doccupancy
3D occupancy 
# Dadaset
<img src="./assets/demo.gif" width="696px">

# A demo of it
<img src="./assets/prediction.gif" width="696px">
## Evaluation Metrics
Leaderboard ranking for this challenge is by the intersection-over-union (mIoU) over all classes. 
### mIoU

Let $C$ be he number of classes. 

$$
    mIoU=\frac{1}{C}\displaystyle \sum_{c=1}^{C}\frac{TP_c}{TP_c+FP_c+FN_c},
$$

where $TP_c$ , $FP_c$ , and $FN_c$ correspond to the number of true positive, false positive, and false negative predictions for class $c_i$.

### F1 Score
We also measure the F-score as the harmonic mean of the completeness $P_c$ and the accuracy $P_a$.

$$
    F-score=\left( \frac{P_a^{-1}+P_c^{-1}}{2} \right) ^{-1} ,
$$

where $P_a$ is the percentage of predicted voxels that are within a distance threshold to the ground truth voxels, and $P_c$ is the percentage of ground truth voxels that are within a distance threshold to the predicted voxels.