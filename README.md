# RSSM：道路影像切割模型 (Road Semantic Segmentation Model)

## Overview
此模型為台灣郊區、山區道路俯視角空拍語意切割模型。  
由政大無人機團隊 starLab 使用 [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch "link") 進行訓練。  
訓練、驗證與測試資料集：[郊區、山區道路俯視角空拍語意切割資料集](https://github.com/nccudrone/SMRAVSSD "link")。  
訓練GPU：RTX3070Ti-8G  
模型支援之框架與版本： python3.6+、Pytorch 1.4+。  

用途為對影像進行語意切割出郊、山區道路輪廓。  
Pretrained Model放置在model/裡，測試影像請放在test/裡。  
下面為模型預測展示，可以使用[road_predict.py](https://github.com/nccudrone/RSSM/blob/main/road_predict.py "link")來獲得相似結果，會儲存在test/裡，以_predict做為後綴字：  
<img src="https://github.com/nccudrone/RSSM/blob/main/image/segm1.png" width="428" height="240"/>  
<img src="https://github.com/nccudrone/RSSM/blob/main/image/segm2.png" width="428" height="240"/><br/>
## Notes
* 此repo僅包含pretrained model做為免費開放使用
* 此模型用於學術研究
* 訓練、驗證與測試資料集[郊區、山區道路俯視角空拍語意切割資料集](https://github.com/nccudrone/SMRAVSSD "link") 僅包含部分免費開放使用資料集
* road_predict.py需安裝segmentation_models_pytorch、torch、albumentations、numpy、cv2才能執行，如執行發現有缺少套件請再自行安裝
## Pretrained Model Download
[link](http://140.119.164.183:5000/sharing/WCMCj4Vs9 "link")
## Licence
請參考 [LICENSE](https://github.com/nccudrone/RSSM/blob/main/LICENSE "link")
