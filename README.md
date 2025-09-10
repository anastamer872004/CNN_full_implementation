# (MNIST -> Conv -> ReLU -> Pool -> FC -> Softmax) 

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

## 5- Training Pipeline (with batch size = 64) :
#### Dataset: 60,000 MNIST images (28x28)

#### ↓ DataLoader splits into batches
####   Batch 1 → 64 images + 64 labels
####   Batch 2 → 64 images + 64 labels
####   ...
####   Batch 938 → 64 images + 64 labels
####   ( 938 batches per epoch)

#### ↓ Each batch goes into the CNN
####   inputs: [64, 1, 28, 28]
####   → Conv1 + ReLU + Pool
####   → Conv2 + ReLU + Pool
####   → Flatten to [64, 3136]
####   → Fully connected layers ( 2 FC )
####   → Output: [64, 10]

####   loss function : that Compare predictions with labels by using CrossEntropyLoss (better predictions → smaller loss)
####   - CrossEntropyLoss = Applies softmax to outputs , then Computes negative log likelihood.

####   Backpropagation : computes the gradients of the loss w.r.t. all model weights automatically (using autograd), update weights.

####   Repeat for all 938 batches = 1 epoch
####   - At the end of the epoch, print average loss (total loss ÷ number of batches).
  







