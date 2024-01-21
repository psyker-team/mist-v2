<p align="center">
<br>
<!-- <img  src="mist_logo.png"> -->
<img  src="assets/MIST_V2_LOGO.png">
<br>
</p>


[![project page](https://img.shields.io/badge/homepage-mist--project.io-blue.svg)](https://mist-project.github.io/index_en.html)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1k5tLNsWTTAkOlkl5d9llf93bJ6csvMuZ?usp=sharing)
[![One-click Package](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1vg8oK2BUOla5adaJcFYx5QMq0-MoP8kk?usp=drive_link)


# Handbook: Mist-v2 Free Version

Mist-v2 free version allows you to have Mist-v2 deployed at your local devices without connecting to the network. This instruction focuses
on how to deploy and run Mist-v2 free version.

## Download

Download and install the following Windows official kit: [.NET](https://dotnet.microsoft.com/zh-cn/download/dotnet/thank-you/runtime-desktop-7.0.13-windows-x64-installer).


Access our [Google Drive](https://drive.google.com/drive/folders/1vg8oK2BUOla5adaJcFYx5QMq0-MoP8kk?usp=drive_link).


<p align="center">
<img  src="../assets/tutorial_0_download.png">
</p>


Download the file *mist-v2_gui_free_version* .


<p align="center">
<img  src="../assets/tutorial_1_download.png">
</p>


Unzip the file to see the following file structure.

<p align="center">
<img  src="../assets/tutorial_2_main.png">
</p>


## Run

**Note: This step does not require network connection.**


Create a folder called `img` in `mist-v2_gui_free_version/mist-v2/src/data/` and put your to-be-Misted images in the folder.


<p align="center">
<img  src="../assets/tutorial_3_img.png">
</p>


Click on `Mist_GUI_Booter`. 


<p align="center">
<img  src="../assets/tutorial_4_boot.png">
</p>



Click *Run Mist* in the booter.


<p align="center">
<img  src="../assets/tutorial_5_boot_.png">
</p>


See the GUI of Mist-v2. 



<p align="center">
<img  src="../assets/tutorial_6_gui.png">
</p>


Click on `Mist` to Mist your images.


<p align="center">
<img  src="../assets/tutorial_7_gui_.png">
</p>


If Mist-v2 is currently running, you can see the following messages in the command window:


<p align="center">
<img  src="../assets/tutorial_8_running.png">
</p>


After seeing the following messages, the output will be placed in the `mist-v2_gui_free_version/mist-v2/src/output/`.


<p align="center">
<img  src="../assets/tutorial_9_running_.png">
</p>


Open `mist-v2_gui_free_version/mist-v2/src/output/` to check the output images.


<p align="center">
<img  src="../assets/tutorial_10_result.png">
</p>


## Hints

**Device requirements**：Windows system and an Nvidia GPU with at least 6 GB VRAM. 

**How to pick image batch for one shot of running Mist**：Select 10-20 images with similar contents or styles. 

**Running time**：3 minutes for one image on average. Note that it is not recommended to Mist one image separately.

