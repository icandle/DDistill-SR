# DDistill-SR
Codes for TMM paper "[DDistill-SR: Reparameterized Dynamic Distillation Network for Lightweight Image Super-Resolution](https://ieeexplore.ieee.org/document/9939085)".

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

---
### Acknowledgement 
Our RDU is based on existing dynamic and reparameterized methods, thanks for their enlightening work！
