Manuscript title: A multi-agent social interaction model for autonomous vehicle testing


Running the Code
For code implementing MA-AIRL, please visit multi-agent-irl folder.
For the OpenAI particle environment code, please visit multi-agent-particle-envs folder.
Training: please run \multi-agent-irl\irl\mack\run_mack_airl.py
Testing: please run \multi-agent-irl\evaluate_int.py
Obtaining SVO for expert scenario: please run \multi-agent-irl\social_pre_expert.py
Obtaining SVO for generated scenario: please run \multi-agent-irl\social_pre_generated.py

Requirements and Installation
python	3.6.15
tensorflow	1.8.0
baselines	0.1.6		
box2d	2.3.10		
click	8.0.4	
geopandas	0.9.0		
grpcio	1.48.2	
gym	0.15.7			
matplotlib	3.3.2		
numpy	1.16.0		
numpy-stl	3.0.1	
opencv-python	3.4.2.16		
pandas	1.1.5		
pip	21.3.1	
protobuf	3.19.6	
pyglet	1.5.0	
pymysql	1.0.2	
pyparsing	3.1.1	
pyproj	3.0.1		
scipy	1.5.4	
seaborn	0.11.2		
sklearn	0.0.post9		
tensorboard	1.8.0	
tensorflow-estimator	1.14.0				
typing-extensions	4.1.1		
wheel	0.37.1
xlrd	1.2.0
xlsxwriter	3.1.9	

Data description: 
The data is derived from a publicly available dataset: SinD dataset, and data that can be used directly for training and testing has been uploaded. The data includes three categories, which are explained in the following.
1. Expert data: 
①sinD.pkl The data for each expert scenario agent, including the observations and actions of the vehicle at each timestep (the selection of observations and actions aligns with the descriptions in the paper), is stored in a PKL file.
②init_sinD.npy The data for each expert scenario agent, including the vehicle's observations at the first timestep, the time of entering the scenario (the selection of observations aligns with the descriptions in the paper), is stored in an NPY file.
③landmarks.pkl The data for each expert scenario background vehicle, including its position, speed, heading angle, and acceleration at each timestep, is stored in a PKL file.
2. Lane data: csv The boundary of left-turn lanes and through lanes
3. SinD data: csv SinD raw data for lane drawing
