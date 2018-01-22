module
========
Define reusable modules of network here.  
> "Every feature has one implement structure."  
It's suggested that any structure should write as a "Module", under here.

e.g:  
Basically, you can write module like this:  
```
def module_a(input_tensor):                       
    output_tensor = some_function(input_tensor)   
    return output_tensor                          
```  
Or, for custom unit which has given feature, you can define it in either layer, loss or util like this:  
```
def MAP_Inference(input_tensor): 
    q_h_under_v = Dirac_distribution(some_tensor)                      
    output_tensor = some_function(q_h_under_v)   
    return output_tensor                          
```