# IE 498 Homework 3

> Julius Olson

## Result

A test accuracy of `82.3%` was reached using the following parameters in combination with monte carlo. `80.4%` Was reached using the same parameters but with heuristic eval

Param      | Value
-----------|--------
Epochs   | `15`
LR       | `0.0005`
Activation | `RELU`
Monte Carlo Iterations | `10`

![res](dropout-res.png)


As observed from the plots above, the process of training seemed to run a bit smoother when utlizing monte carlo simulations instead of the heurisitcs. 

## Implementation

The structure of the net is as follows. The convolutional blocks are named in accordance with their order and the linear block is the final one in the structure:

```python
self.conv_block1 = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2,stride=2),
			nn.Dropout(0.05),
		)

		self.conv_block2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=5, padding=2),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=5, padding=2),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2,stride=2),
			nn.Dropout(0.05)
		)

		self.conv_block3 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=5, padding=2),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=5, padding=2),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2,stride=2),
			nn.Dropout(0.1),
		)

		self.linear_block = nn.Sequential(
			nn.Linear(4096, 512),
			nn.ReLU(inplace=True),
			nn.Dropout(0.3),
			nn.Linear(512, 256),
			nn.ReLU(inplace=True),
			nn.Linear(256, 10),
		)
```


## Instructions & requirements

```sh
# Run the code:

python net.py

``` 