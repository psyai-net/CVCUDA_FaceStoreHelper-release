![Psyche AI Inc release](./psy_logo.png)

# CVCUDA_FaceStoreHelper
CVCUDA version of FaceStoreHelper, suitable for super-resolution, face restoration, and other face extraction and reattachment procedures.

[![License ↗](https://img.shields.io/badge/License-CCBYNC4.0-blue.svg)](LICENSE)

## Project Introduction

This project is designed to extract faces from images, perform image restoration/super-resolution operations, and then merge the restored face back into the original image. The code provides a simple, fast, and accurate method for face extraction and merging back into the original image, suitable for various application scenarios.

We have undertaken a comprehensive rewrite of the original OpenCV implementation, replacing all serial CPU operations with their corresponding CVCUDA operators. This optimization enables our project to leverage the power of GPU acceleration, resulting in significantly improved performance and efficiency in computer vision tasks. By harnessing CVCUDA's capabilities, we have successfully enhanced the processing speed of our application, providing a more responsive and robust user experience.

## Features

- Widely applicable for post-processing of face image super-resolution/restoration networks, such as GFPGAN and CodeFormer.

## Quick Start

### Install Dependencies

Before running this project, please make sure the following dependencies are installed:
```bash
pip install nvcv_python-0.3.1-cp38-cp38-linux_x86_64.whl
```

### Downloading pretraining models

download the checkpoints and put it into ./checkpoint
link: https://pan.baidu.com/s/1ZPfLnXS5oGDawqualhXCrQ?pwd=psya 
password: psya

link: https://drive.google.com/drive/folders/1pwadwZLJt0EQUmjS7u4lUofYiLIfAjGj?usp=sharing


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
``` shell
python gen_resource_pkg.py
```

inference with cvcuda accelerated codeformer network
``` shell
python cvcuda_facestorehelper.py --input_path = your_images_path
```

4. The program will save the images with extracted faces in the specified path and draw the restored faces on the original images.

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


- Contact Author Name: Jason Zhaoxin Fan
- Contact  Author Email: fanzhaoxin@psyai.net
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
