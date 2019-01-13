## DeepDefacer: Automatic Removal of Facial Features via Deep Learning.
 
 <div align="center">
<img style="float:center;margin:0px" src="diffsizeundefaced.png"> <img style="float: center;" src="exampledefacedresized.png"> 
</div>

DeepDefacer is a MRI anonymization tool written in Python, on top of Tensorflow and Keras, that was developed in partnership with the Poldrack Lab at Stanford University. It can be used to quickly deface 3D MRI images of any resolution and size on commercial CPUs and GPUs. Its goal is to provide the community with an easy to use and efficient tool for defacing medical images that require anonymization for compliance with federal privacy laws (e.g HIPAA). 

### Referencing and citing DeepDefacer
If you use DeepDefacer in your work, please refer to this citation for the current version:

```
@article{khazane2019state,
  title={DeepDefacer: Automatic Removal of Facial Features from MR Scans Via Deep Learning},
  author={Anish Khazane, Julien Hoachuck, Dr. Chris Gorgowelski, Dr. Russell Poldrack},
  journal={in proceedings, arXiv preprint},
  year={2019}
}
```
If you use any of the architecture code from the [ARFF-CNN](https://github.com/AKhazane/ARFF-CNN.git), please also use the citation above to comply with its authors' instructions on referencing.


### Requirements 

* Any Python version between 2.7 and 3.6.
* If you are using the GPU version of this library, please ensure that your GPU drivers are correctly installed and up to date. Please reference [GPU Support for Tensorflow-GPU](https://www.tensorflow.org/install/gpu) for further details on GPU setup. 
* Input MRI images must have 3D structure and be saved in either .nii or .nii.gz format.

### Installation

Deepdefacer can be easily installed on any operating system via Pypi. There are two versions of this package; CPU or GPU support. Please enter **one** of the following commands into your terminal window to begin installation, depending on your system specifications and desired python version. 

CPU Support
```
pip install deepdefacer[tf_cpu] / pip3 install deepdefacer[tf_cpu]
```

GPU Support
```
pip install deepdefacer[tf_gpu] / pip3 install deepdefacer[tf_gpu]
```

**Note: If you are using a ZSH-type shell, you may need to wrap the package name in quotations in order to successfully initiate the Pip installation. (e.g ```pip install "deepdefacer[...]"```).**

### Usage and Documentation

Once installed, please enter ```deepdefacer --help``` into your terminal window to see a list of available tools within this program. Defacing a 3D MRI image is extremely simple, and can be done with the following command:

 ```deepdefacer <input filename> ```
 
 The program will output a defaced image in the **same directory as the input file.** 

