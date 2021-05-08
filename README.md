# Deep-Pruning-Quantization-Gradient-approach
Explicit Connection Sensitivity Pruning and Quantization



####Running baseline models:\
models:                         --model : ["VGG16", "ResNet34", "ResNet50"] :default - VGG16 <br/> 
datasets:                     --dataset :["CIFAR10", "CIFAR100"] : default - ResNet34<br/> 
pruning type(global/layerwise): --prunetype: [True, False]  : default - False # layerwise pruning is not the default instead global pruning is default<br/> 
learning rate:                   --lr    :                   : default - 0.1<br/> 
num_epochs :                    --epochs :                    :defualt - 70<br/> 
weightdecay:                    --weightdecay:                :default - 0.0005<br/> 
step_size:                      --step:                       : default - 20<br/> 
batch_size                      --batchsize                   :default - 128<br/> 
retaining weight fraction:      --retrain                     :default - 0.05(only 5% of the weights are retained)<br/> 

Run command:<br/>   
E.g.:<br/>   
python3 main.py --model = "ResNet34" --dataset="CIFAR10"<br/> 
 
 
