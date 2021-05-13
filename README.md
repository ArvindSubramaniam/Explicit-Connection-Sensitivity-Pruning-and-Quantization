# Learning Compact and Efficient Networks through Explicit Connection Sensitivity Pruning and Quantization

Hi! Welcome to the code for our project! We perform pruning followed by quantization to obtain compact and efficient Deep Neural Networks! For more details, please refer our Project report.

Pruning File Structure

```
src
└── pruning
       ├── ECS.py
       ├── baseline.py
       ├── train.py          
       └── main.py        
```
 
 ```ECS.py``` - Code for our pruning approach.<br/>
 ```baseline.py``` - Code for the baseline pruning approach (SNIP).<br/>
 ```train.py``` - Code for training and testing models.<br/>
 ```main.py``` - Code for running pruning (SNIP and our approach).<br/>


The following figure serves as a high-level representation of our work.


![image](https://user-images.githubusercontent.com/37202614/118072436-c86d0480-b377-11eb-881e-1a6d4bb8f7dc.png)

# Pruning Module

# Baseline:  
A. SNIP Paper implementation: "https://github.com/mil-ad/snip" (This is not the official implementation, but its the one we referred)<br/> 
B. Even though referenced implementation is done from scratch.<br/> 
C. Done Layerwise and global pruning.<br/>


# Explicit Connection Sensitivity Pruning (ECS):

Our approach uses both weights and gradients to decide which parameters are redundant. After training the model for 1 epoch, we jointly exploit information from the weights and the gradients. Only parameters with a low weight magnitude and a low gradient magnitude are deemed unimpportant and removed. For instance, as shown in the figure below, a parameter with a low weight magnitude but a high gradient magnitude is still considered important.

![image](https://user-images.githubusercontent.com/37202614/118072561-14b84480-b378-11eb-9c67-f8f8ec9c80d1.png)



# Usage:
```
models:                         --model : ["VGG16", "ResNet34", "ResNet50"] :default - VGG16 
datasets:                     --dataset :["CIFAR10", "CIFAR100"] : default - CIFAR10
pruning type(global/layerwise): --prunetype: [True, False]  : default - False # layerwise pruning is not the default instead global pruning is default<br/> 
learning rate:                   --lr    :                   : default - 0.1
num_epochs :                    --epochs :                    :defualt - 70
weightdecay:                    --weightdecay:                :default - 0.0005
step_size:                      --step:                       : default - 20
batch_size                      --batchsize                   :default - 128
retaining weight fraction:      --retrain                     :default - 0.05(only 5% of the weights are retained)<br/> 
```

Navigate to the _pruning_ directory. 

Baseline (SNIP)
```
python3 baseline.py --model = "ResNet34" --dataset="CIFAR10"
```

Explicit Connection Sensitivity Pruning (ECS)
```
python3 ECS.py 
python3 main.py --model = "ResNet34" --dataset="CIFAR10"
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


