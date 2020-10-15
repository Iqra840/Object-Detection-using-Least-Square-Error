# Object-Detection-using-Least-Square-Error
The program uses the least square error formula to determine which digit an image represents. The least square formula produces a numeric weight; the higher the weight, the more likely that it represents a particular digit. Of all tested digits, the digit with the largest weight should be the one that is represented on the image.
The least square formula is shown below: 



weight = (((img-transpose) img)-inverse)img) * labels matrix

where img is a matrix contained the flattened image we are dealing with, and the labels matrix is an array which labels an image through a series of 0s and 1s. After calculating the weight, we also run a confusion matrix to see how many numbers are correctly identifies by the program.
