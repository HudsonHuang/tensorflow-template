model
========
Define model network here.  

Usage:  
```
def model_a(input_tensor):                        # typically, input_tensor would be input placeholder
    output_tensor = some_function(input_tensor)   # you can modules as funtction in ../module  
    return output_tensor                          # typically, output_tensor would be train_step
```
