## Unsupervised Learning Assignment
> Implementation for K-Means and Fuzzy C-Means by Bradley Thompson

#### Course Details:  

     CS 545 - Intro to Machine Learning
     Prof. Anthony Rhodes
     Fall Term (2020) - Portland State University
     
#### Overview
This is a manual implementation of the two-aforementioned algorithms.
I implemented the algorithms, instead of re-using some library implementation 
because that was a requirement, this is a homework assignment!

###### K-Means
This implementation randomly assigns centroids before running the algorithm.
It saves the cluster states across multiple iterations and then
plots that data using matplotlib.pyplot.

###### Fuzzy C-Means
This implementation differs in that a datapoint can be
part of multiple clusters at once. To simplify graphing the data,
I assigned datapoints to clusters based on their maximally weighted relationship.
    