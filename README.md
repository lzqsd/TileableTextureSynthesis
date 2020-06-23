# Tileable Texture Synthesis 
This is the texture synthesis method used in the paper:
* Li, Z., Shafiei, M., Ramamoorthi, R., Sunkavalli, K., & Chandraker, M. (2020). Inverse rendering for complex indoor scenes: Shape, spatially-varying lighting and svbrdf from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 2475-2484).

Please cite this paper if you use the code.

## Method overview
We adopt a graph-cut-based [2] methods to perform tileable texture synthesis, which can keep the structure of the texture. Please refer to [1] for more details. A demonstration of our pipeline is shown below. 
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/texture_method.png)
Some example results are shown below.
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/texture_result.png)

## Instructions
* First, download the Adobe Stock materials from this [link](https://stock.adobe.com/). 
* Open `runTextureSynthesis.m`. At line 13, set `root` to be the directory you have the Adobe Stock materials. 
* Go to directory `Bk_matlab`. Compile the code as instructed.
* Run `runTextureSynthesis.m`.

## References
* [1] Li, Z., Shafiei, M., Ramamoorthi, R., Sunkavalli, K., & Chandraker, M. (2020). Inverse rendering for complex indoor scenes: Shape, spatially-varying lighting and svbrdf from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 2475-2484).
* [2] Boykov, Y., Veksler, O., & Zabih, R. (2001). Fast approximate energy minimization via graph cuts. IEEE Transactions on pattern analysis and machine intelligence, 23(11), 1222-1239.

