# Machine Learning Demos

This repo is a collection of code snippets to demostrate common models and applications of Machine Learning. Each script is a standalone program. The intention is to capture the core of the idea with the minimal amount of code. Hope you find it helpful as a quick reference to understand ML techniques.

## Natural Language Processing

* [nlp/text_generation_gpt_2_beam_search.py](nlp/text_generation_gpt_2_beam_search.py): GPT-2, text generation, pre-trained model, beam search
* [nlp/text_generation_gpt_2_top_k_top_p_filtering.py](nlp/text_generation_gpt_2_top_k_top_p_filtering.py): GPT-2, text generation, pre-trained model, top-k top-p filtering
* [nlp/text_generation_gpt_2_greedy_search.py](nlp/text_generation_gpt_2_greedy_search.py): GPT-2, text generation, pre-trained model, greedy search
* [nlp/transformer.py](nlp/transformer.py): the original Transformer model implemented with PyTorch
* [nlp/imdb_bert.py](nlp/imdb_bert.py): BERT, sentiment analysis, transfer learning, text classification
* [nlp/imdb_lstm.py](nlp/imdb_lstm.py): LSTM, embedding, text classification
* [nlp/imdb_tf_hub.py](nlp/imdb_tf_hub.py): embedding, TF-Hub, text classification
* [nlp/imdb.py](nlp/imdb.py): embedding, text classification

## Speech

* [speech/rnnt.py](speech/rnnt.py): RNN transducer, speech recognition, end-to-end model. This is **only** to demostrate the implemenation of the model, because I don't have the massive hardware resource required to train it (300GB data + 1 GPU month).

## Vision

* [vision/oxford_iiit_pet_unet.py](vision/oxford_iiit_pet_unet.py): U-Net, transfer learning, image segmentation
* [vision/cats_vs_dogs.py](vision/cats_vs_dogs.py): ResNet, transfer learning, image classification
* [vision/cats_vs_dogs_heatmap.py](vision/cats_vs_dogs_heatmap.py): ResNet, CNN activation map, transfer learning
* [vision/mnist_cnn.py](vision/mnist_cnn.py): simple CNN, image classification
* [vision/mnist.py](vision/mnist.py): DNN, Dropout, image classification

## Generative Adversarial Networks

* [gan/mnist.py](gan/mnist.py): Deep Convolutional GAN, image generation

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
