# 3Doccupancy
Generate occupancy label for secondary processing of nuscenes dataset, We use the internal and external parameters of the camera and the 3D bounding box of the object to generate dense point cloud and their labels. 
Simply speak, For key frames, we directly use the label of the point cloud and internal and external parameters to align static objects. For dynamic objects, we use his label information (position and rotation angle) for alignment. For non-key frames, since the label of the point cloud cannot be obtained, we first align the dynamic objects according to the above method, and then use the nearest neighbor search for the remaining point clouds to generate their point cloud labels.

# Dadaset
<img src="./assets/demo.gif" width="696px">

# Prediction
<img src="./assets/prediction.gif" width="696px">

# Model
Similar to BEVFormer, 3Doccupancy network has 3 encoder layers, each of which follows the conventional structure of transformers, except for three tailored designs, namely BEV queries, spatial cross-attention, and  self-attention. Specifically, BEV queries are grid-shaped learnable parameters, which is designed to query features in BEV space from multi-camera views via attention mechanisms. Spatial cross-attention and self-attention are attention layers working with BEV queries, which are used to lookup and aggregate spatial features from multi-camera images, according to the BEV query. Since the BEVfeature is two-dimensional, an embedding in the z-axis direction is added to turn it into a three-dimensional space feature, and then a convolutional neural network is used to generate the semantics of each position in the three-dimensional space.


## Evaluation Metrics

### mIoU

Let $C$ be he number of classes. 

$$
    mIoU=\frac{1}{C}\displaystyle \sum_{c=1}^{C}\frac{TP_c}{TP_c+FP_c+FN_c},
$$

where $TP_c$ , $FP_c$ , and $FN_c$ correspond to the number of true positive, false positive, and false negative predictions for class $c_i$.
### Results in val set

| barrier | bicycle | bus | car | construction_vehicle | motorcycle | pedestrian | traffic_cone | trailer |  truck | driveable_surface | other_flat | sidewalk | terrain | manmade | vegetation |  miou |
| -- | --|--| -- | --|--|--|--|--|--|--|--| --|----------------------|---|------ | -------------------------------- |
| 15.12 | 8.55 | 28.78 | 28.06 | 10.36 | 13.42 | 9.22 | 4.57 | 17.38 | 22.56 | 48.38 | 22.57 | 29.11 | 25.81 | 16.22 |20.77 | 20.056  

[**clik here download mini occupancy dataset for nuscenes v1.0-mini**](https://drive.google.com/file/d/1n48IIy1poOOusHujyGhuDx_QZkPeI6ki/view?usp=sharing)

## Acknowledgement

Many thanks to these excellent open source projects:

- [open-mmlab](https://github.com/open-mmlab)
- [CenterPoint](https://github.com/tianweiy/CenterPoint)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [TPVFormer](https://github.com/wzzheng/TPVFormer)

Most thanks to nuscenes dataset:
- [nuscenes](https://www.nuscenes.org/nuscenes)
