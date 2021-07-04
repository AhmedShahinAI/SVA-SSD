SVASSD is a Matlab Implementation for Saliency Visual Attention Single Shot Detector for building detection in low contrast images.

SVA-SSD proposes a modified version for single shot multi-box detector based on a visual saliency mechanism. The modification is based on three approaches

    1- Modify the SSD backbone based on saliency attention.
    2- Modify the SSD head based on optimized Anchorboxes.
    3- SVA-SSD is fed with a special augmentation process to overcome the low dataset size and decrease the overfitting problems.


<img width="634" alt="GraphicalAbstract" src="https://user-images.githubusercontent.com/24828652/124371872-af554580-dc8e-11eb-8b77-053c4aee068a.png">



o establish our research idea for building detection inside a desert environment, we have collected a
dataset for buildings in high-resolution satellite images from Riyadh city, Saudi Arabia. The image size is
3.6 Gigabytes in ECW compression format. We have utilized QGIS Pro to extract the small patches from
the large image in JPEG compression format with a zoom ratio of 100% and a scale of 1:1582. The total number of
images was 500 images with a resolution of 300×300. The total number of buildings in our dataset reached
3878 buildings. The annotated buildings have large variations in size, color, overcrowding, and shape
inside the dataset. The dataset is available on request>>> dr.eng.ahmedshahin@gmail.com.

A sample of the dataset with several data augmentation processes.

![image](https://user-images.githubusercontent.com/24828652/124371903-f17e8700-dc8e-11eb-9967-61ecd08cabac.png)

