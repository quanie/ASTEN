# ASTEN
ASTEN is an algorithm to perform Coupled Tensor Factorization in an accurate and scalable manner. Unlike other existing algorithms, ASTEN optimizes every single tensor and matrix having one mode in common by enabling each of them to have its own discriminative factors on the shared mode. It is thus capable of finding the accurate approximation of every tensor. We also designed it to be scaled up with respect to the number of tensors, their dimensions, their sizes and the number of data partitions.

The details of our idea and its implementation can be found in the below reference. 

#Usage
java -cp ASTEN.jar edu.uts.Main [Parameters]

Parameters: [Number of Tensor] <[Tensor 1's Mode] [Tensor 1's Length] ... [Tensor N's Mode] [Tensor N's Length]> <[Tensor 1's parts] ... [Tensor N's parts]> [Rank] <[Tensor 1's filename] ... [Tensor N's filename]> [Out_Key] [Learning rate] [Stoping condition] [Optional: Maximum running hour]
  
Please refer to run.sh for an example of how to specify the parameters.

#Authors
Quan Do, University of Technology Sydney - https://sites.google.com/site/minhquandd/

Wei Liu, University of Technology Sydney - https://sites.google.com/site/weiliusite/

* Feel free to contact Quan for any questions, bug fixes or improvements.

#Reference
Please reference as: Quan Do and Wei Liu, ASTEN: an Accurate and Scalable Approach to Coupled Tensor Factorization, in Proceedings of the International Joint Conference on Neural Networks (IJCNN), 2016 

    @inproceedings{Do_ASTEN16,
      author = {Do, Quan and Liu, Wei},
      title = {ASTEN: An Accurate and Scalable Approach to Coupled Tensor Factorization},
      booktitle = {IEEE International Joint Conference on Neural Networks (IJCNN)},
      year = {2016},
    }


#Copyright
This software is free for research purposes. For commercial purposes, please contact the authors.
