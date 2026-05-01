<div align="center">

<h1>Cross-Domain Spatial Matching for Camera and Radar Sensor Data Fusion in Autonomous Vehicle Perception System</h1>

<div>
Daniel Dworak&emsp;Mateusz Komorkiewicz&emsp;Pawel Skruch&emsp;Jerzy Baranowski
</div>
<div>
    AGH University of Krakow &emsp;<br/>
</div>

---

### Abstract

> In this paper, we propose a novel approach for camera–radar sensor fusion aimed at 3D object detection in autonomous vehicle perception systems. Our method leverages recent advances in deep learning to take advantage of the complementary strengths of both sensors, thereby enhancing detection performance. Specifically, we extract 2D features from camera images using a state-of-the-art deep neural network and then employ a Cross-Domain Spatial Matching (CDSM) transformation to map these features into 3D space. These transformed features are subsequently fused with radar-derived features through a complementary fusion strategy, producing a unified 3D object representation. To evaluate the effectiveness of the proposed approach, we conduct experiments on the nuScenes dataset and compare our method against both single-sensor baselines and current state-of-the-art fusion techniques. The results demonstrate that our approach outperforms single-sensor solutions and achieves competitive performance relative to other top-level fusion methods..

### Source code
The repository contains four folders:
1. cdsmnet - the final implementation of the method described as CDSM Fusion with modalities Camera+Radar (3D)
2. linet - the method described as Pointcloud model with modalities Radar (3D) 
3. vinet - the method described as Vision model with modality Camera (2D)
4. vinet_3d - the method described as Vision model with modality Camera (3D)

### License
CDSM © 2026 by AGH University of Krakow is licensed under CC BY-NC-ND 4.0. 
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-nd/4.0/ 

