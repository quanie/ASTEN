# ASTEN
ASTEN is an algorithm to perform Coupled Tensor Factorization in an accurate and scalable manner. Unlike other existing algorithms, ASTEN optimizes every single tensor and matrix having one mode in common by enabling each of them to have its own discriminative factors on the shared mode. It is thus capable of finding the accurate approximation of every tensor. We also designed it to be scaled up with respect to the number of tensors, their dimensions, their sizes and the number of data partitions.

The details of our idea and its implementation can be found in the below reference. 

#Usage
java -cp ASTEN.jar edu.uts.Main [Parameters]

Parameters: [Number of Tensor] <[Tensor 1's Mode] [Tensor 1's Length] ... [Tensor N's Mode] [Tensor N's Length]> <[Tensor 1's parts] ... [Tensor N's parts]> [Rank] <[Tensor 1's filename] ... [Tensor N's filename]> [Out_Key] [Learning rate] [Stoping condition] [Optional: Maximum running hour]
  
Please refer to run.sh for an example of how to specify the parameters.

#Author
Quan Do, University of Technology Sydney - https://sites.google.com/site/minhquandd/

Wei Liu, University of Technology Sydney - https://sites.google.com/site/weiliusite/

#Reference
Quan Do, Wei Liu, ASTEN: an Accurate and Scalable Approach to Coupled Tensor Factorization, in Proceedings of the International Joint Conference in Neural Networks (IJCNN), 2016 

#Copyright
This software is free for research projects. If you publish your results obtained from this software, please cite our paper:

@inproceedings{Do16,

    author = {Quan Do and Wei Liu},
    title = {ASTEN: An Accurate and Scalable Approach to Coupled Tensor Factorization},
    booktitle = {IEEE International Joint Conference in Neural Networks (IJCNN)},
    year = {2016},

}

Contributing back bug fixes and improvements is polite and encouraged. If you have any question, feel free to contact Feng Zhou.
