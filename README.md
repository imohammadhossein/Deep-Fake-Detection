<div align="center">
    <img src="/images/index.jpg" width="200">
</div>

<h1 align="center">Kaggle Deep Fake Detection Challenge</h1>


These codes have implemented for  **[Kaggle Deep Fake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge "Kaggle Deep Fake Detection Challenge").**  There are two main idea behind the story, one is finding the structure of deep fake just by CNN features; second, adding a LSTM network probably will have positive effects on it. 
The sequential procedure is respected inside codes.
this implementation was among the top ranks in the challenge. 


[**Crop Faces**](https://github.com/imohammadhossein/Deep-Fake-Detection/blob/develop/src/face_extractor.ipynb "face extractor mtcnn") helps to save the videos into sequential frames with your desired interval. the rationale behind the scripts follows two main branches: 

* **CNN Based Algorithms** 
> In this approach by inspiring from the strnegth of deep neural networks, we attempted to figure out the structure and texture of Deep-fake videos using convolutional layers. 
> There are some [**CNN-based networks**](https://github.com/imohammadhossein/Deep-Fake-Detection/blob/develop/src/CNNs/ "CNN-Based approachs"), (EfficientNet ResNet, VGG, ...), each with different results.
> 
> 
* **RNN Based Algorithms**
> [**RNN-based networks**](https://github.com/imohammadhossein/Deep-Fake-Detection/blob/develop/src/Recurrents/ "RNN-Based approach") utilizes the vital features hidden in sequences of frames in a fake video and recognizing it after passing the output of CNN layers towards LSTM layers. This approach has got much better result.

## Feedback
If you had any feedback or question, please reach out to me at mh.nikimaleki@gmail.com
## 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mhnikimaleki/)
