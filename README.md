# SmartPlanAI
This project aims to use artificial intelligence to streamline the process of creating custom floorplans by using inputs such as the desired number of rooms, functionality of said rooms and the requirements of the space. 


## Dataset
The dataset consists of 8 parquet files sourced from zimhe/sudo-floor-plan-12k from HuggingFace (https://huggingface.co/datasets/zimhe/sudo-floor-plan-12k). The dataset is structured with 6 columns with the headings: Indices (string), plans (image), walls (image), colors (image), footprint (image), and plan_captions (string).
Each item in the dataset is unique in layout, orientation, size, orientation or a combination of any of the aforementioned properties.

## Process to be followed
1) Parse and preprocess the dataset to separate the elements
2) Procure a model suitable for the training task
3) Proceed to train the model to learn key features and recognize the different regions of a given floorplan
4) Evaluate the model on its ability to predict a floorplan layout and compare its efficacy against the ground truths
5) Test the model on its ability to generate a viable floorplan given a random or new footprint and evaluate its accuracy.
