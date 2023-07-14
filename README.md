![Psyche AI Inc release](./psy_logo.png)

# CVCUDA_FaceStoreHelper
CVCUDA version of FaceStoreHelper, suitable for super-resolution, face restoration, and other face extraction and reattachment procedures.

[![License ↗](https://img.shields.io/badge/License-CCBYNC4.0-blue.svg)](LICENSE)

## Project Introduction

This project is designed to extract faces from images, perform image restoration/super-resolution operations, and then merge the restored face back into the original image. The code provides a simple, fast, and accurate method for face extraction and merging back into the original image, suitable for various application scenarios.

We have undertaken a comprehensive **rewrite of the original OpenCV implementation**, **replacing all serial CPU operations** with their corresponding **[CVCUDA operators](https://github.com/CVCUDA/CV-CUDA/blob/release_v0.3.x/DEVELOPER_GUIDE.md)**. This optimization enables our project to leverage the power of GPU acceleration, resulting in significantly improved performance and efficiency in computer vision tasks. By harnessing CVCUDA's capabilities, we have successfully enhanced the processing speed of our application, providing a more responsive and robust user experience.

## update

- [x] python serial version：The blocks are executed sequentially
- [ ] python Coroutine version: The blocks are executed in parallel，closed source
- [x] support for [codeformer](https://github.com/sczhou/CodeFormer/blob/master/inference_codeformer.py)
- [ ] Chinese readme
- [ ] Integrated Self-developed Lip driving talking head algorithm, closed source
- [ ] Implementation of cvcuda process of face normalization
- [x] Put the face back on and morphologically blur the edges
- [x] cvcuda reconstruction of batch affine transform

## Features

- Widely applicable for post-processing of face image super-resolution/restoration networks, such as GFPGAN and CodeFormer.

## Quick Start


### Install Dependencies

Before running this project, please make sure the following CVCUDA are download and installed:
please refer https://github.com/CVCUDA/CV-CUDA

Configure cvcuda in prebuild mode and download it in the release file 
https://github.com/CVCUDA/CV-CUDA/tree/dd1b6cae076b0d284e042b3dda42773a5816f1c8

installation example：
```bash
pip install nvcv_python-0.3.1-cp38-cp38-linux_x86_64.whl
```

Add /your-cvcuda-path to PYTHONPATH in the bashrc file
```
export PYTHONPATH=/your-cvcuda-path/CV-CUDA/build-rel/lib/python:$PYTHONPATH
```

Solution to show so file not found:
```
export LD_LIBRARY_PATH=your-path/CV-CUDA/build-rel/lib:$LD_LIBRARY_PATH
```


Install the necessary python packages by executing the following command:

``` shell
pip install -r requirements.txt
```

Go to codeformer/ and run the following command to install the basicsr package:

``` shell
cd ./codeformer/
python basicsr/setup.py develop
cd ..
```

After installing the basicsr package, add the project root directory, and codeformer directory to the system's environment variables:

``` shell
vim ~/.bashrc
```

``` 
export PYTHONPATH=$PYTHONPATH:/path/to/talking_lip:/path/to/talking_lip/codeformer
```

Note that /path/to/talking_lip is the absolute path of your project in the system.

 Save and exit. Run the following command to make the configuration take effect:

``` shell
source ~/.bashrc
```

### Downloading pretraining models

download the checkpoints and put it into ./checkpoint
link: https://pan.baidu.com/s/1ZPfLnXS5oGDawqualhXCrQ?pwd=psya 
password: psya

Google drive link: https://drive.google.com/drive/folders/1pwadwZLJt0EQUmjS7u4lUofYiLIfAjGj?usp=sharing


### Usage Example

1. Clone this repository:

```bash
git clone https://github.com/your-username/your-repo.git
```

2. Enter the project directory:

```bash
cd your-repo
```

3. Run the script:

Generate resource packs that provide codeformer with the necessary face landmarks and affine transformation matrices

When you run gen_resource_pkg.py, place a video in the `testdata/video_based` folder to extract the face resource pack.

``` shell
python gen_resource_pkg.py
```

inference with cvcuda accelerated codeformer network
``` shell
python serial_pipeline.py --input_path = your_video_path
```

4. The program will save `./outlip.mp4` , your video will be enhanced by codeformer, only for face area.

## Contribution

- If you encounter any problems, you can report them in this project's GitHub Issue.
- If you want to contribute code, please follow these steps:
  1. Clone this repository.
  2. Create a new branch:
     ```bash
     git checkout -b new-feature
     ```
  3. Make changes and commit:
     ```bash
     git commit -m 'Add some feature'
     ```
  4. Push to the remote branch:
     ```bash
     git push origin new-feature
     ```
  5. Submit a pull request.
  
## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. Please read the [LICENSE](LICENSE) file for more information.

## Authors

Code contributor: [Junli Deng](https://github.com/cucdengjunli), [Xueting Yang](https://github.com/yxt7979), [Xiaotian Ren](https://github.com/csnowhermit)

If you would like to work with us further, please contact this guy：
- Contact Author Name: [Jason Zhaoxin Fan](https://github.com/FANzhaoxin666)
- Contact Author Email: fanzhaoxin@psyai.net
- Any other contact information: [psyai.com](https://www.psyai.com/home)

## Acknowledgments

This project incorporates the following methods:

1. **CVCUDA**: [Project Link](https://github.com/CVCUDA/CV-CUDA)

2. **GFPGAN**: [Project Link](https://github.com/TencentARC/GFPGAN)

3. **CodeFormer**: [Project Link](https://github.com/sczhou/CodeFormer)


## Invitation

We invite you to join [Psyche AI Inc](https://www.psyai.com/home) to conduct cutting-edge research and business implementation together. At Psyche AI Inc, we are committed to pushing the boundaries of what's possible in the fields of artificial intelligence and computer vision, especially their applications in avatars. As a member of our team, you will have the opportunity to collaborate with talented individuals, innovate new ideas, and contribute to projects that have a real-world impact.

If you are passionate about working on the forefront of technology and making a difference, we would love to hear from you. Please visit our website at [Psyche AI Inc](https://www.psyai.com/home) to learn more about us and to apply for open positions. You can also contact us by fanzhaoxin@psyai.net.

Let's shape the future together!!
