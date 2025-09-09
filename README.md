# CNN --> (Conv → ReLU → Pooling) Visualization of Implementation

Input Image(batch_size, 1, 28, 28) 
####  Conv2D(1 → 32, kernel=3, padding='same') output (batch_size, 32, 28, 28) -> MaxPool2D(kernel=2, stride=2) output (batch_size, 32, 14, 14)
#### Conv2D(32 → 64, kernel=3, padding='same') output (batch_size, 64, 14, 14) -> MaxPool2D (kernel=2, stride=2) output (batch_size, 64, 7, 7)

####### Flatten
(batch_size, 3136)   ← (64 × 7 × 7)
        │
        ▼
Fully Connected (Linear: 3136 → 128)
(batch_size, 128)
        │
        ▼
ReLU
(batch_size, 128)
        │
        ▼
Fully Connected (Linear: 128 → 10)
(batch_size, 10)
        │
        ▼
Output = logits → احتمالية كل رقم من 0 إلى 9
