# Art Title Generator

The goal of this project is to generate a text caption based on the artwork image.  

## Context
Machine learning algorithm has proven to be able to read the structural patterns of the language and able to generate text sequence that follows the syntactic rules  and imitate the semantics of human language. This is especially done with the help of the recurrent neural network, which is essentially very similar to one way humans learn languages. We learn individual words, then learn to tie different words through trial and error. (A child may learn to tie 'give' - 'me' - 'food' by experiencing the reaction (often rewards) from saying different combinations of words.)

Similary, machine learning is also able to recognize the scenes. It can visually parition a scene into meaningful components and associate it to a label through hierarchical and sequential steps. Just like how human vision works. 

But human cognition goes beyond this limit. Humans make connection beyond the preassigned categories and also break rules. We abstract our scene beyond the universal comprehension just like we abstract language beyond its universality to be more exclusive. We make art and write poem.

Interestingly, interpreting art is a learnable skill. For the first-timers, some abstract artworks may seem out of the boundary of human comprehension. After enough training, we start making more and more confident prediction about what the art is about. We get better at it. Such expertise might be just a simple extension of the fundamental process of comprehending visual inputs and translating them into words. What happens when we extend the simple boundary of object classificaion and sequential language generation in machine learning?

## Overview
For this project, I explore the deep learning models to generate captioning for paintings. I explore this with the assumption that art interpretation is an extension of ability to describe any visual scene. Therefore, I built the models on top of a simple real-life image captioning model and incorporated art data and extending the complexity of models. The basis of the image captioning model I wrote here is an adaptation of codes by Jason Brownlee PhD at [Machine Learning Mastery](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/). 

## Repo Structure

## Data
There are 2 large category of data used in this project. First is the real-life photo images with caption from Flickr provided by [Jason Brownlee PhD](https://github.com/jbrownlee/Datasets). 
Second is the set of art images with their text information collected from 3 different APIs as below.
1. Harvard Museum ()
2. Rhode Island School of Design ()
3. Museum of Modern Art ()

Our final data included
_____ number of images from Flickr with multiple captions
_____ number of images of paintings from museum APIs with title

## Preprocessing (NLP)
Standard preprocessing was done to both art titles and object captions, including turning them into lowercase, removing special characters and digits (or changing to syntactically meaningful words.
Dimensions of art title tokens were further reduced by consodliating words with at least .8 cosine similarity using [Spacy](https://spacy.io/usage/vectors-similarity).

## Distribution of Art Data
Our data only included paintings dated post-1900s to current. 

some info about data here

## Deep Learning




