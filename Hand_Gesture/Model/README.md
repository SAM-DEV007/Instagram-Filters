# Model
  The model is trained and, then converted and saved as `tflite` model along with the unconverted one, saved under the 
  [Model_Data](/Hand_Gesture/Model/Model_Data) folder. The [Dataset](/Hand_Gesture/Model/Dataset)
  is a single `csv` file. It consists of the position of various hand points or landmarks, along with the gesture.
  
  The model consists of a single `Input` layer, with three `Hidden` layers and a single `Output` layer. The output optimizer is
  `softmax`, and loss function is `sparse_categorical_crossentropy`.
  
  [Train_Model.py](/Hand_Gesture/Model/Train_Model.py) is used here to train, save and convert the model.
