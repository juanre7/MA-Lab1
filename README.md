# Lab 1: Image as a 2D Signal


## 0) Load and inspect an image

Loads a sample image, shows the RGB version, converts it to grayscale, and prints class, range, and size. This establishes the starting point for all operations.

**Screenshots**

* Original RGB
  <img width="749" height="691" alt="Figure_1" src="https://github.com/user-attachments/assets/2f32707e-8e6c-4cde-8c25-43aa07dad3d6" />

* Grayscale
  <img width="749" height="691" alt="Figure_2" src="https://github.com/user-attachments/assets/2a15fea9-74fc-48a4-a593-edda6ac97223" />


---

## 1) Quantization and dynamic range

Creates lower bit depth versions of the image to visualize banding and posterization. This links quantization levels to perceived quality.

**What to look for**

* Smooth regions turning into visible bands as bit depth decreases.

**Screenshot**

* <img width="1045" height="282" alt="Figure_3" src="https://github.com/user-attachments/assets/ceace1a0-d169-4c9b-b1b4-9039c48f31f8" />


---

## 2) Histogram and contrast stretching

Plots the original histogram and the histogram after linear contrast stretching. Also shows the original, a normalized version, and the stretched image.

**What to look for**

* Histogram spreading after stretching.
* Enhanced visibility of midtone details.

**Screenshots**

* Histograms
 <img width="1120" height="751" alt="Figure_4" src="https://github.com/user-attachments/assets/978f536c-dbb4-4afe-ae26-5b283185e3a9" />

* Images: original vs normalized vs stretched
 <img width="1045" height="282" alt="Figure_5" src="https://github.com/user-attachments/assets/5416c958-4660-490b-a12c-54ecf0ae7291" />

---

## 3) Gamma correction

Applies nonlinear gamma correction. Gamma less than 1 brightens shadows. Gamma greater than 1 darkens highlights.

**What to look for**

* Recovery of shadow detail with low gamma.
* Taming of bright regions with high gamma.

**Screenshot**

 <img width="1045" height="282" alt="Figure_6" src="https://github.com/user-attachments/assets/48e72bb3-d6c5-40b6-8ece-0ad56e83e746" />


---

## 4) Sampling and aliasing

Downsamples aggressively using nearest neighbor, then upsamples back to the original size. Demonstrates how undersampling produces aliasing artifacts.

**What to look for**

* Loss of fine detail at low resolution.
* Blockiness and jagged patterns after upscaling.

**Screenshot**

 <img width="1065" height="282" alt="Figure_7" src="https://github.com/user-attachments/assets/4d204a4b-0c29-4e6f-aa73-84e97a90dd1f" />

