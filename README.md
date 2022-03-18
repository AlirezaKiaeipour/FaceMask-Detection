# Face Mask Detection

  **Dataset Link:** [Face Mask](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset)
  
  **Model Link:** [Model](https://drive.google.com/file/d/12g18xGpmCgUQ4nAQLvW_hUnyUSTasA50/view?usp=sharing)

  - Face Mask Detection using TensorFlow and Keras
    
  - Model:

    - [x] MobileNetV2


  - Accuracy & Loss:

    Algorithm | Accuracy | Loss |
    ------------- | ------------- | ------------- |
    MobileNetV2 | **99.19 %** | **0.0276** |
    

  - Inference:

      ## RUN
      You can run  Inference with the following command
      
      **Please download the [Model](https://drive.google.com/file/d/12g18xGpmCgUQ4nAQLvW_hUnyUSTasA50/view?usp=sharing) first**

      ```
      python inference_image.py [--input_model INPUT] [--input_image INPUT]
      
      python inference_webcam.py [--input_model INPUT]
      
      python inference_qt.py [--input_model INPUT]
      ```
      
      ![1](https://user-images.githubusercontent.com/88143329/157250672-d1c343c6-b224-401e-939b-e230d1bd3335.png)
      
      ![2](https://user-images.githubusercontent.com/88143329/157250764-3d5d9c16-fbe7-48d4-a2f4-fac2ba447cfa.png)
      
      
      ### Demo
      
      https://user-images.githubusercontent.com/88143329/157250846-56fe558c-b0c9-45a0-a158-03c5996a8933.mp4
