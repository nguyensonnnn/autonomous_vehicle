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

<iframe width="560" height="315" src="https://www.youtube.com/embed/S_aP-RwHZb0?si=3S-V2P_Y3QRSxC78" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

