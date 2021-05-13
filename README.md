# Pruning Module

# Reference:  
A. original SNIP Paper implementation: "https://github.com/mil-ad/snip"<br/> 
B. Even though referenced implementation is done from scratch.<br/> 
C. Done Layerwise and global pruning.<br/>


# Single Shot Network Pruning:
A short description of our approach can be found below:

![image](https://user-images.githubusercontent.com/37202614/118071422-bbe7ac80-b375-11eb-9c3a-447c56bca7c4.png)


Train the model on first batch of the dataset<br/>
create a mask based on the global threshold or layerwise threshold<br/>
Use this mask for the rest of the batches in all epochs.



# Running baseline models:
```
models:                         --model : ["VGG16", "ResNet34", "ResNet50"] :default - VGG16 <br/> 
datasets:                     --dataset :["CIFAR10", "CIFAR100"] : default - CIFAR10<br/> 
pruning type(global/layerwise): --prunetype: [True, False]  : default - False # layerwise pruning is not the default instead global pruning is default<br/> 
learning rate:                   --lr    :                   : default - 0.1<br/> 
num_epochs :                    --epochs :                    :defualt - 70<br/> 
weightdecay:                    --weightdecay:                :default - 0.0005<br/> 
step_size:                      --step:                       : default - 20<br/> 
batch_size                      --batchsize                   :default - 128<br/> 
retaining weight fraction:      --retrain                     :default - 0.05(only 5% of the weights are retained)<br/> 

Run command:<br/>   
E.g.:<br/>   
python3 baseline.py --model = "ResNet34" --dataset="CIFAR10"<br/> 
```

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


