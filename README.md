# DDistill-SR
Codes for TMM paper "[DDistill-SR: Reparameterized Dynamic Distillation Network for Lightweight Image Super-Resolution](https://ieeexplore.ieee.org/document/9939085)".

> Recent research on deep convolutional neural networks (CNNs) has provided a significant performance boost on efficient super-resolution (SR) tasks by trading off the performance and applicability. However, most existing methods focus on subtracting feature processing consumption to reduce the parameters and calculations without refining the immediate features, which leads to inadequate information in the restoration. In this paper, we propose a lightweight network termed DDistill-SR, which significantly improves the SR quality by capturing and reusing more helpful information in a static-dynamic feature distillation manner. Specifically, we propose a plug-in reparameterized dynamic unit (RDU) to promote the performance and inference cost trade-off. During the training phase, the RDU learns to linearly combine multiple reparameterizable blocks by analyzing varied input statistics to enhance layer-level representation. In the inference phase, the RDU is equally converted to simple dynamic convolutions that explicitly capture robust dynamic and static feature maps. Then, the information distillation block is constructed by several RDUs to enforce hierarchical refinement and selective fusion of spatial context information. Furthermore, we propose a dynamic distillation fusion (DDF) module to enable dynamic signals aggregation and communication between hierarchical modules to further improve performance. Empirical results show that our DDistill-SR outperforms the baselines and achieves state-of-the-art results on most super-resolution domains with much fewer parameters and less computational overhead.
```
@ARTICLE{9939085,
  author={Wang, Yan and Su, Tongtong and Li, Yusen and Cao, Jiuwen and Wang, Gang and Liu, Xiaoguang},
  journal={IEEE Transactions on Multimedia}, 
  title={DDistill-SR: Reparameterized Dynamic Distillation Network for Lightweight Image Super-Resolution}, 
  year={2022},
  pages={1-13},
  doi={10.1109/TMM.2022.3219646}}
```
---
### Training and Testing
* Training with [EDSR framework](https://github.com/sanghyun-son/EDSR-PyTorch) or [BasicSR framework](https://github.com/XPixelGroup/BasicSR).
  * Download training data (800 + 2650 images) from [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar).
  * Prepare LR-HR pairs with **BI**, **BN**, and **DN** methods. 
  
* Testing with five commenly used datasets.

  * [Set5](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html): Bevilacqua *et al*. BMVC 2012.
  * [Set14](https://sites.google.com/site/romanzeyde/research-interests): Zeyde *et al*. LNCS 2010.
  * [B100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/): Martin *et al*. ICCV 2001.
  * [Urban100](https://sites.google.com/site/jbhuang0604/publications/struct_sr): Huang *et al*. CVPR 2015.
  * [Manga109](http://www.manga109.org/en/): Matsui *et al*. MTA.
  
### Results and Models
* Pretrained models available at [model-backup](https://github.com/icandle/DDistill-SR/tree/main/model-backup).    
* Visual results available at [Baidu Pan](https://pan.baidu.com/s/1FpD5ucp_G31TQoxZDa5acQ?pwd=ddsr).

### Acknowledgement 
Our RDU is based on existing dynamic and reparameterized methods, thanks for their enlightening work！
