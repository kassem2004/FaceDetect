# FaceDetect— Face Classification with VGGFace2 Encoder + MTCNN (PyTorch)

This project trains a lightweight classifier head on top of a frozen **InceptionResnetV1** (pretrained on **VGGFace2**) for closed-set face identification. Faces are detected/aligned with **MTCNN** at load time.

Key training flow:
- **Data split**: 90% train/validation, **10% held-out test** (stratified by class).
- **Model selection**: **5-fold cross-validation** on the 90% portion; pick the fold with the **best validation accuracy**.
- **Final report**: After training, the chosen best model is evaluated on the 10% test set and the script **prints only the test accuracy**

---

## Project Layout

.
src/
  train_vggface2_transfer_pytorch.py # main script (train & predict)
  data_raw/
    person_a/ img1.jpg, img2.jpg, ...
    person_b/
  models_vggface2/ # outputs (checkpoints per fold)

## Environment

- Python 3.9+
- PyTorch, TorchVision
- `facenet-pytorch`
- Pillow, NumPy

Sources Studied:

The Essential Guide to K-Fold Cross Validation
https://medium.com/@bididudy/the-essential-guide-to-k-fold-cross-validation-in-machine-learning-2bcb58c50578

Multi-task Cascaded Convolutional Neural Network (MTCNN)
https://medium.com/the-modern-scientist/multi-task-cascaded-convolutional-neural-network-mtcnn-a31d88f501c8

Implementing Face Recognition Using Deep Learning and SVMs
https://www.codemag.com/Article/2205081/Implementing-Face-Recognition-Using-Deep-Learning-and-Support-Vector-Machines
