# Autonomous car in real time road detection simulation with 
This is source code of the part of my research project in ISAE SUPAERO master program. The final goal of the code is to simulate the auto pilot behaviours of autonomous vehicle by considering pictures capturing by front camera as inputs and steering command as ouput. In this project, I explore a transformative approach - converting manual cars into autonomous ones, paving the way for an eco-friendly transportation ecosystem. The key focus areas are visual perception and motion planning, tackling the challenges of end-to-end self-driving systems.For image semantic segmentation, I apply Unet convolutional neural network model that outperforms previous models carried out by my project successor, achieving exceptional Intersection over Union scores. The implementation of image filtering techniques plays a crucial role in streamlining computational resources while maintaining high-quality results. In the realm of motion planning, we leverage a dataset comprising video and steering commands from the control wheel. This data trains a neural network model to mimic human driving behavior. This unique combination of convolutional and fully connected neural networks yields remarkable results, albeit with minor discrepancies in certain frames that don't significantly impact vehicle behavior.

# intro
Welcome to our exciting autonomous car development project! Our ultimate aim is to transform a manual car into a fully autonomous vehicle. The project encompasses comprehensive phases including research, simulation, assembly, and rigorous testing. Currently, we are at the initial stages, primarily focused on research and simulation. One key aspect we're delving into is the implementation of visual perception. This empowers the car to intelligently perceive its surroundings, encompassing the road, ground, and potential obstacles. Additionally, we're employing an alternative method to generate depth maps, enhancing the accuracy of our algorithms.
List of tasks in this code: 
- Implementing object detection utilizing convolutional neural networks.
- Producing a detailed depth map of the environment.
- Proposing and evaluating a cutting-edge image semantic segmentation model.
- Utilizing a combination of image convolutional neural networks and fully connected neural networks to enable the car to predict steering commands.
- Thoroughly analyzing results with various models and datasets.
- Harnessing the power of depth map generation and image segmentation to refine steering command predictions.



# Neural network learning model 

This model is inspired by the Dave system of NVIDIA https://developer.nvidia.com/blog/deep-learning-self-driving-cars/ with DAVE system with 2 cameras to
perform steering and adjusting thrust based on pre-trained model. However, NVIDIA has not
clarified the process of training this model architecture as well as evaluate its performance in
general. The model is described in the picture below.

![Screenshot 2023-10-17 002425](https://github.com/nguyensonnnn/autonomous_vehicle/assets/140680983/c37602bc-b478-4cb3-8ba9-3e911fad9da4)

• The first set of layers contain of 5 convolutional neural layers, which is used for extracting
the basic information relating to the surrounding environment of images perceived from the
camera is called the sensory neurons. Sensory neurons are performed convolution by the
kernel of size 3x3 with the stride of 2 in the first three layers and the kernel of size 2x2 in
two last layers with the stride of 1.
• The second set of layers contain of 4 fully connected neural layers to interpret the extract
feature from the images to the steering command.
In order to research the different between these layers, there are some modifications is applied to
research the optimal result of this model such as add more layer, dropout at different stage of
model. The final results is demonstrated on chapter 5. These modifications are:
• Adding 1 more convolutional layer and 1 more neural network layer with the purpose of
achieving better understanding the image and more accurate interpretation.
• Adding 1 more neural network layer and delete one convolutional neural network. In this
case, filtered dataset is used so the image has already been splitted into different pattern and
it becomes redundant if there are two many sensory neurons on the networks instead of
more regular neurons to interpret the image feature

# Result 
The error of the prediction is given as below 
![Screenshot 2023-10-17 002014](https://github.com/nguyensonnnn/autonomous_vehicle/assets/140680983/897d2dbf-471b-4f13-83a3-95af613ef234)

It is crucial to mention that this error is not big enough to affect the safety of the car because, the
predicted values are recorded from the control wheel and based on the normal steering ratio of
ordinary cars nowadays, the car in real life only change its steering angle 0.243-0.275 degrees. In
addition, although there are some large errors at certain frame such as 300 degrees and 350 degrees,
this error only occurs for 1 frame lasting 33 milliseconds. Nevertheless, there must be some
augmentations need to be applied to enhance the overall performance
