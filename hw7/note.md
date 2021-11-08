Hi

Just to be clear, is my understanding of the training and testing process for this assignment correct? 

## Training

Let us assume that the name in question we are training on is **brycen**
I assume that the loss will calculated as `loss_function(actual_output, expected_output)`.

| Iteration number      | Input | Actual Output | Expected Output |
| :---        |    :----   |          :--- | :--- |
| 1    |  b -> Tensor of dimen `(1,1,27)`    |  ? Tensor of dimen `(1,1,27)` | r -> Tensor of dimen `(1,1,27)` | 
| 2    |  br -> Tensor of dimen `(1,2,27)`   |  ? Tensor of dimen `(1,2,27)` |ry -> Tensor of dimen `(1,2,27)` |
| 3    |  bry -> Tensor of dimen `(1,3,27)`  |  ? Tensor of dimen `(1,3,27)` | ryc ->Tensor of dimen `(1,3,27)` |
| 4    |  bryc -> Tensor of dimen `(1,4,27)` |  ? Tensor of dimen `(1,4,27)` | ryce-> Tensor of dimen `(1,4,27)` |
| 5    |  bryce -> Tensor of dimen `(1,5,27)`|  ? Tensor of dimen `(1,5,27)` | rycen -> Tensor of dimen `(1,5,27)` |
| 6    |  brycen -> Tensor of dimen `(1,6,27)` |  ? Tensor of dimen `(1,6,27)` | rycen EON -> Tensor of dimen `(1,6,27)` |
| 7 |  brycen EON -> Tensor of dimen `(1,7,27)` |  ? Tensor of dimen `(1,7,27)` | rycen EON EON -> Tensor of dimen `(1,7,27)` |
and so on


## Testing (Generation of names)

| Iteration number      | Input | Actual Output |
| :---        |    :----   |          :--- |
| 1      |  b -> Tensor of dimen `(1,1,27)`        |  r -> Tensor of dimen `(1,1,27)` | 
| 2      |  br -> Tensor of dimen `(1,2,27)`       |  ry -> Tensor of dimen `(1,2,27)` | 
| 3      |  bry -> Tensor of dimen `(1,3,27)`      |  rya -> Tensor of dimen `(1,3,27)` | 
| 4      |  brya -> Tensor of dimen `(1,4,27)`     |  ryan -> Tensor of dimen `(1,4,27)` |
| 5      |  bryan -> Tensor of dimen `(1,5,27)`    |  ryan EON -> Tensor of dimen `(1,5,27)` | 
and so on





