t# Face Detection

  We have developed a real-time (and static) face detector. We didn't use any augmentation and if needs it can be developed. The training dataset we used is sourced from [UTKFace](https://susanqq.github.io/UTKFace/). To contribute testing (described below), we have made a small testing dataset available in our repository (`./data/images`).

We utilized the pretrained *SSD MobileNet V2 FPNLite 320x320* model for detection. The training dataset consisted of approximately 400 images, which we manually annotated.

Instructions for usage:

Optional:
You may create a new environment.

Step 1: *Install requirements*
Navigate to the `requirements` directory, located at `./face_detection/requirements`, and use the command 
`pip install object_detection`
(do not use requirements.txt) to install the necessary dependencies.

Step 2: *Running*
Navigate to the `face_detection` directory, located at `./face_detection`, and use the command 
`python face_detection.py --mode {real-time/static}`.

For real-time mode, the program will utilize your camera to detect faces in real time.
  Use `q` to interrupt.

For static mode, the program will use images from `./face_detection/data/images` to detect faces and save the results to `./face_detection/data/output`.

Note: You may also add images to `./face_detection/data/images`

# Examples below
Static:

![friend-group-selfie-1024x683](https://github.com/SamvelKarapetyaan/aca_homeworks/assets/130899451/fe0a22d8-081a-4eff-aef9-50a39fa74e77)
![friend-group-selfie-1024x683](https://github.com/SamvelKarapetyaan/aca_homeworks/assets/130899451/7981517a-32df-4459-8549-7e385d1d88d4)

Real-time:

![ezgif com-optimize](https://github.com/SamvelKarapetyaan/aca_homeworks/assets/130899451/3b40d1b0-ffa3-471a-8057-232b224f5e32)


