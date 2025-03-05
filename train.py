!pip install wandb -qU
!pip install numpy
!pip install matplotlib
!pip install pandas

import wandb
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#login to wandb
wandb.login()

# Initialize wandb
wandb.init(
    project="DL_A1" ,resume=True
)

#loading the dataset
from keras.datasets import fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
class_names= ['T-shirt', 'Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']


# i want to convert the dataset into sub dataset based obn common labels

label_grp={label:[] for label in range(10)}

for img,label in zip(X_train,y_train):
  label_grp[label].append(img)
for label in label_grp:
    label_grp[label] = np.array(label_grp[label])
sweep_config = {
    "method": "random",  # Random search for hyperparameters
    "metric": {"name": "accuracy", "goal": "maximize"},  # Example metric
    "parameters": {
        "learning_rate": {"values": [0.001, 0.01, 0.1]},
        "batch_size": {"values": [32, 64, 128]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="DL_A1")

def train():
    # Initialize W&B for this run
    wandb.init(project="DL_A1", config=sweep_config)

    sweep_run_id = wandb.run.sweep_id  # Get the sweep ID
    run_name = wandb.run.name  # Run name

    for run_id in range(10):  # Each run will log different random images

        for step in range(3):  # Log over 3 steps
            selected_images = []
            selected_labels = []

            for label in labels_grp:
                img = random.choice(labels_grp[label])  # Pick a random image
                selected_images.append(img)
                selected_labels.append(class_names[label])  # Store class name
            image_data = []
            for index, (img, label) in enumerate(zip(selected_images, selected_labels)):
                # Include Sweep ID and Run Name in the caption
                image_data.append(wandb.Image(img, caption=f"{label} (Index: {index}) - Sweep: {sweep_run_id} - {run_name}"))

            # Log images at the current step
            wandb.log({"examples": image_data, "Step": step})

    # Finish logging
    wandb.finish()


# Run the Sweep for 150 Runs
wandb.agent(sweep_id, function=train, count=150)
