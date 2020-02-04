# IE 498 - HW1

> Julius Olson


## Implementation

## Instructions & requirements


```sh

# Train
python3 nn.py --mode train --data <path-to-dataset>


# Load and evaluate
python3 nn.py --mode load --model <path-to-model> --data <path-to-dataset>

```

Requires python ver `3.6` or higher and numpy.


## Result

An accuracy of XX percent was achieved using

Param      | Value
-----------|--------
`d_hidden` | 100
`epochs`   | 15
`LR`       | Piecewise constant

```

Progress: [============================================================]
Epoch 1/10 done!, Training Accuracy: 0.934, Test= 0.935 - 0.01
Progress: [============================================================]
Epoch 2/10 done!, Training Accuracy: 0.95335, Test= 0.9508 - 0.01
Progress: [============================================================]
Epoch 3/10 done!, Training Accuracy: 0.9612166666666667, Test= 0.9588 - 0.01
Progress: [============================================================]
Epoch 4/10 done!, Training Accuracy: 0.96985, Test= 0.9623 - 0.01
Progress: [============================================================]
Epoch 5/10 done!, Training Accuracy: 0.9751666666666666, Test= 0.9677 - 0.01
Progress: [============================================================]
Epoch 6/10 done!, Training Accuracy: 0.9774166666666667, Test= 0.9693 - 0.001
Progress: [============================================================]
Epoch 7/10 done!, Training Accuracy: 0.9779833333333333, Test= 0.9701 - 0.001
Progress: [============================================================]
Epoch 8/10 done!, Training Accuracy: 0.9780166666666666, Test= 0.9702 - 0.001
Progress: [============================================================]
Epoch 9/10 done!, Training Accuracy: 0.9782333333333333, Test= 0.9706 - 0.0001
Progress: [============================================================]
Epoch 10/10 done!, Training Accuracy: 0.9784166666666667, Test= 0.9709 - 0.0001
Accuracy: 0.9709

```