# Quantization module

1. Quantizing using k-mean clustering 
    - quantization is performed on Linear, Conv2d and btachNorm layers of model
    - every layer weights are accessed and quantized by replacing weights from clusters with cluster centroids.
    - Number of clusters can be controlled using num_cluster parameter while calling the quantization function
    - for our approach we have used number of cluster as 5.
    
2. Quantizing into 8 bit
    - we have used standard 8 bit quantization algorithm to calculate zero points and scaling factor using pytorch functions.
    - this further reduces the memory footprint of the model while impacting model accuracy by few fractions.
   
Note: Detail results are present in the Quantization_and_testing.ipynb Notebook
# Running the quantization Module
```
   # import the module using 
   # from quantization import QuantizeNetwork
   
   q_net = QuantizeNetwork(verbose=True)
   num_cluster = 5
   quanitzed_model = q_net(model , num_cluster)
```

 
# Visualization module

1. Display Kernels 
    - function display_kernel renders all kernel/filters from the model
    - if the kernel is 3*3 then it will take average of all different channels and render single heatmap
    
2. Post quantize weight sharing scatter plot
    - After quantization how the weights gets distributed is hard task to guess and render from.
    - method display_scatter_plot is implemented to complete this task by comparing two models unique weights
    - it plots frequency vs weight scatter plot for both model in orange and blue
    - this shows the distribution in weights matrix
   
3. Rendering Activations
   - Rendering activations is an great way to learn about model's learning
   - method display_activations can be called with model and one sample input image/instance
   - this will plot input image and all RELU activation.

Note: Detail results are present in the visualization_model_test.ipynb notebook
   
# Running the Visualization Module
```
   # import the module using 
   # from quantization import VisualizeNetwork
   
   v_net = VisualizeNetwork()
   
   # model object tranfer to cpu
   post_process_net.cpu()
   
   # displaying kernels from all convolution layers of model
   vn.display_kernel(post_process_net)
   
   # displaying scatter plot for all weights of Linear, conv2d and batchnorm layers of model
   # input: initial_net : reference model 1
   #        post_process_net: reference model 2
   vn.display_scatter_plot(initial_net,post_process_net)
   
   # displaying/simulating activation layers of model with input vector
   # input:- model, inputimage :x
   # x can be taken from loader with below command 
   x = next(iter(train_loader))[0]
   vn.display_activations(post_process_net,x)
   
```


