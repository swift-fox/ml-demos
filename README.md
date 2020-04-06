# Machine Learning Demos

This repo is a collection of some machine learning code I wrote and played with in the past. The sole purpose is to demostrate my coding experience and knowledge of the field.

## Image Classification

* [image/mnist.py](image/mnist.py): basic DNN, Dropout, TensorFlow
* [image/mnist_cnn.py](image/mnist_cnn.py): basic CNN, TensorFlow
* [image/cats_vs_dogs.py](image/cats_vs_dogs.py): ResNet, transfer learning, TensorFlow

## Natural Language Processing

* [nlp/imdb.py](nlp/imdb.py): embedding, TensorFlow
* [nlp/imdb_lstm.py](nlp/imdb_lstm.py): LSTM, RNN, embedding, TensorFlow
* [nlp/imdb_tf_hub.py](nlp/imdb_tf_hub.py): embedding, TF-Hub, TensorFlow

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

A series of scripts to demostrate the core and heart of Machine Learning: backpropagation, SGD, losses, optimizers, layers, ReLU and automatic differentiation. All were written with PyTorch.
