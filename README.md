# Machine Learning Demos

This repo is a collection of some machine learning code I wrote and played with in the past. The sole purpose is to demostrate my coding experience and knowledge of the field. All code here are short and simple, because why not?

## Vision

* [vision/oxford_iiit_pet_unet.py](vision/oxford_iiit_pet_unet.py): U-Net, transfer learning, image segmentation
* [vision/cats_vs_dogs.py](vision/cats_vs_dogs.py): ResNet, transfer learning, image classification
* [vision/cats_vs_dogs_heatmap.py](vision/cats_vs_dogs_heatmap.py): ResNet, CNN activation map, transfer learning
* [vision/mnist_cnn.py](vision/mnist_cnn.py): simple CNN, image classification
* [vision/mnist.py](vision/mnist.py): DNN, Dropout, image classification

## Natural Language Processing

* [nlp/imdb_bert.py](nlp/imdb_bert.py): BERT, sentiment analysis, transfer learning, text classification
* [nlp/imdb_lstm.py](nlp/imdb_lstm.py): LSTM, embedding, text classification
* [nlp/imdb_tf_hub.py](nlp/imdb_tf_hub.py): embedding, TF-Hub, text classification
* [nlp/imdb.py](nlp/imdb.py): embedding, text classification

## Reinforcement Learning

All were written with PyTorch and run with OpenAI Gym.

* [rl/cartpole_pytorch_dqn.py](rl/cartpole_pytorch_dqn.py): Deep Q Learning, experience replay
* [rl/cartpole_pytorch_actor_critic.py](rl/cartpole_pytorch_actor_critic.py): Actor-Critic architecture
* [rl/cartpole_pytorch_ddpg.py](rl/cartpole_pytorch_ddpg.py): DDPG, Actor-Critic architecture
* [rl/cartpole_pytorch_policy_gradient.py](rl/cartpole_pytorch_policy_gradient.py): vanilla policy gradient method
* [rl/cartpole_random_pytorch.py](rl/cartpole_random_pytorch.py): randomized weights (toy algorithm)
* [rl/mountain_car.py](rl/mountain_car.py): Deep Q Learning, experience replay
* [rl/mountain_car_continuous_ddpg.py](rl/mountain_car_continuous_ddpg.py): DDPG, Actor-Critic architecture
* [rl/robotics.py](rl/robotics.py): an attempt to get some more complex cases work. Failed due to not having enough computing power. Used stable_baselines, hindsight experience replay and DDPG.

## Machine Learning Fundementals

[ml-fundementals](ml-fundementals/): A series of scripts to demostrate the core and heart of Machine Learning: backpropagation, SGD, losses, optimizers, layers, ReLU and automatic differentiation. All were written with PyTorch.
