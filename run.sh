#java -cp ASTEN.jar edu.uts.Main [Number of Tensor] <[Tensor 1's Mode] [Tensor 1's Length] ... [Tensor N's Mode] [Tensor N's Length]> <[Tensor 1's parts] ... [Tensor N's parts]> [Rank] <[Tensor 1's filename] ... [Tensor N's filename]> [Out_Key] [Learning rate] [Stoping condition] [Optional: Maximum running hour]
#Ex: decomposing 2 tensors: mode-3 tensor of 3x3x3 coupled with a matrix of 3x2
#					Rank: 3
#					Tensor 1's filename: "testtensor.txt" 
#					Tensor 2's filename: "testmatrix.txt"
#					Output Key: "ASTEN_out"
# 					Learning rate: 0.1
#					Stoping condition: 10^-5
#					Maximun running hours: 4
java -cp ASTEN.jar edu.uts.Main 2 3 3 3 3 2 3 2 3 3 3 3 2 3 "testtensor.txt" "testmatrix.txt" "ASTEN_out" 0.1 0.00001 4
