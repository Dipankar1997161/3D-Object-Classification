# 3D-Object-Classification

### The following repository focuses on Point-cloud based Object Classification.
#### The model was trained under the following configurations:

###### The Dataset used in Princepton ModelNet10 which consists of objects such as "Dresser", "Bed", "Toilet", "Monitor", "Desk", "Table", "Bathtub, "Chair", "Sofa", "Night-stand". The point-clouds are avialble in .off format
###### Frameworks: Tensorflow, Trimesh and Open3D
##### Config 1: Trained with Learning rate 1e-3 for 20 epochs. Validation Accuracy for such stage is 0.757

##### Config 2: Trained with Learning rate 5e-4 for 30 epochs. Vallidation Accuracy for such stage is 0.711

### Trimesh
Trimesh is a pure Python (2.7-3.5+) library for loading and using triangular meshes with an emphasis on watertight surfaces. The goal of the library is to provide a full featured and well tested Trimesh object which allows for easy manipulation and analysis, in the style of the Polygon object in the Shapely library.
#### Installation: pip install trimesh
