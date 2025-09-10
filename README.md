# CNN --> (mnist -> Conv -> ReLU -> Pool -> FC -> Softmax) 

<img width="1048" height="217" alt="image" src="https://github.com/user-attachments/assets/ab5f3270-96ab-438e-bdfa-0eb9573c6016" />

## 1- MNIST Data : Input Image(batch_size, 1, 28, 28) , Each image = 28 × 28 pixels (1 channel)

## 2- CNN structure :
#### (Conv) automatic Detect features(edges, small curves, blobs)
#### (ReLU) is important for Non-linearity and Sparsity and Avoids vanishing gradients
#### (Pool) Reduce spatial size (less computation, more robust to small shifts).
#### Conv2D(1 → 32, kernel=3, padding='same') output (batch_size, 32, 28, 28) --> MaxPool2D(kernel=2, stride=2) output (batch_size, 32, 14, 14)
#### Conv2D(32 → 64, kernel=3, padding='same') output (batch_size, 64, 14, 14) --> MaxPool2D (kernel=2, stride=2) output (batch_size, 64, 7, 7)

## 3- Flatten : The feature maps are turned into a single vector containing all extracted features.
#### (batch_size, 64, 7, 7) --> (batch_size, 3136)

## 4- Fully Connected Layers structure ( 2 FC ) : 
#### (Linear: 3136 → 128) output (batch_size, 128) --> ReLU(batch_size, 128)
#### (Linear: 128 → 10) output (batch_size, 10) --> Output vector(logits) of 10 class(numbers from 0 - 9)





