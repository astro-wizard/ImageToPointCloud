# Real-Time Depth Estimation and 3D Visualization

This project captures video from a webcam, estimates depth using a MiDaS model, and visualizes the resulting point cloud in real-time using Open3D.

![""](images/3dreconstruct.PNG)

## Prerequisites
* Python 3.6+
* OpenCV
* PyTorch
* NumPy
* Open3D


## Installation
### Clone the repository:

```bash
git clone https://github.com/ImageToPointCloud.git

cd your-repository
```
Install the required packages:
```
pip install opencv-python torch numpy open3d
```
Running the Code
Open up a terminal in the project directory.

Run the script:
```
python depth_estimation_visualization.py
```
## Project Explanation
The script captures video frames from a webcam, processes each frame to estimate depth, and visualizes the resulting point cloud in real-time.

### Load the MiDaS Model: 

* The script uses the MiDaS v2.1 Small model for depth estimation.
* The model is moved to the GPU if available for faster processing.
* Set Up Video Capture and Open3D Visualization:

### Video frames are captured from the default webcam.
An Open3D visualization window is created to display the point cloud.
### Process Each Frame:

* Capture a frame and convert it to RGB.
* Apply necessary transforms and predict the depth map using the MiDaS model.
* Normalize the depth map and convert it for display.
* Create an RGBD image and a point cloud from the depth map.
* Apply a transformation to flip and rotate the point cloud.
* Update the Open3D visualization.
### Point Cloud Transformation:

A transformation matrix is used to flip the point cloud to match the correct orientation.
A rotation matrix is applied to the point cloud to provide different viewing angles.
## Controls
* Press q to quit the application.
* The point cloud rotates continuously, allowing you to view it from different angles.
## Customization
* Adjust the camera intrinsic parameters and depth scale values to match your specific camera setup.
* Ensure your webcam is properly connected and functioning.
