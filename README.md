# CS-168-Final-Project

I have created this repository to host the code for the CS 168 Final Project. As we go along, this file will be updated.

### General Info

I have created a deep learning model based off on the [Pytorch Model](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html). This has nost of the info necessary to understand whats going on behind the scenes in this project. The implementation contains a few models (resnet, alexnet, etc.) to use in training and validation sets. The best validation error along with the weights is returned. 

I have set up and argument parser with the model name, number of epochs, and batch size as arguments (Only model name is required). You can use the help option (-h) for more info. I was able to get 86% validation accuracy for resnet and 81% for alexnet. We can work on other methods prom the paper if necessary or just use an array of models.

Let me know if you have any questions about thew code or implementation and I'll try my best to answer, some of it is still a little over my head.

### GIT HELP:

##### initialize the directory as a git repo
- git init
##### pull the repository
- git pull <ssh> # if it prompts a password try: CS168COVID
##### to upload files
- git add . # or git add * 
- git commit -m "insert message"
##### if it troubles you make sure you are verified
- git config --global user.email "your@email.com"
- git config --global user.name "username"
##### and make sure the remote add is set up
- git remote add CS-168-Final-Project "git@github.com:davisgomes/CS-168-Final-Project.git" 
##### it will prompt you, use the password above: CS168COVID
##### push it to github
- git push CS-168-Final-Project master
	
 
