# Art Title Generator

The goal of this project is to develop a natural language generator (NLG) model to generate a text caption from an artwork image.  

## Context
1. Art Classification   
    The world of art is closely intertwined, the influence from one to another is sometimes clear yet the connection is never exclusive. Because the transition of art style is continuous, always imminent, and never simple, it's hard to classify the style of art. For this very reason, art interpretation for the research in the field of art and art history relies on the number of trained personnels and is prone to biases. This can also create discrepancy between art interpretation from the eyes of beholders (actual perceptual experience) and the research. If we are able to eliminate the bias of individual's perception and apply machine learning to art classification, we can advance art research more efficiently and more objectively.

2. Neuroaesthetics (Perception, Abstract Representation)  
    Machine learning algorithm has proven to be able to read the structural patterns of the language and to generate text sequence that follows the syntactic rules and also imitates the semantics of human language. Such function is established in a similar manner as how humans learn languages. We learn individual words, then learn to tie different words through trial and error. Similary, machine learning is also able to recognize the scenes. It can visually partition a scene into meaningful components and associate it to a label through hierarchical and sequential steps. Just like human vision.

    But human cognition goes beyond this limit. Humans make connection beyond the pre-assigned categories and also break rules. We abstract our scene beyond the universal comprehension just like we abstract language beyond its universality to be more exclusive. We make art and write poem. Interpreting art is a learnable skill. Such expertise is a simple extension of the fundamental process of comprehending visual inputs and translating them into words. What happens when we extend the simple boundary of object classificaion and sequential language generation in machine learning? Can the machine incorporate what it learned from everyday scene to interpreting art just like humans?

## Overview
For this project, I create deep learning models to generate captioning for paintings. I explore this with the assumption that art interpretation is an extension of ability to describe any visual scene. Therefore, I built the models on top of a simple real-life image captioning model and incorporated art data and extended the complexity of models. The basis of the image captioning model is an adaptation of outlines by Jason Brownlee PhD at [Machine Learning Mastery](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/). 

## Repo Structure

## Data
There are 2 large category of data used in this project. First is the real-life photo images with caption from Flickr provided by [Jason Brownlee PhD](https://github.com/jbrownlee/Datasets). 
Second is the set of art images with their text information collected from 3 different APIs as below.
1. Harvard Museum (https://github.com/harvardartmuseums/api-docs)
2. Rhode Island School of Design (https://risdmuseum.org/art-design/projects-publications/articles/risd-museum-collection-api)
3. Museum of Modern Art (https://github.com/MuseumofModernArt/collection)

![image_count](/PNG/image_count.png) 

Our final data included
8,091 real-life photographs from Flickr with multiple captions
3,067 images of paintings (post 1990) from museum APIs with their titles

## Preprocessing (NLP)
Standard preprocessing was done to both art titles and object captions, including turning them into lowercase, removing special characters and digits (or changing them to syntactically meaningful words). Dimensions of art title tokens were further reduced by consodliating words with at least .8 cosine similarity using [Spacy](https://spacy.io/usage/vectors-similarity). There were total 9,161 unique vocabularies (7,451 unique words in Flickr set, 2,893 unique words in art set).  

![total word frequencies](/PNG/top_20_total.png)

## Example Data
Both Flickr and art datasets included images with large variability. 

![example_flickr](/PNG/example_flickr.png)

Examples of Flickr image dataset

![example art](/PNG/example_art.png)  

Examples of art dataset

Flickr dataset contained 4-5 human written descriptions per image. For example, one image might be described as...
1. one dog biting at another dog face in grassy field
2. there are two brown dogs playing in field
3. two brown dogs with blue collars are running in the grass
4. two dogs are fighting and playing with each other while running through some grass
5. two dogs are playing in the grass

Art description only contained its title as its description. When it contains only subject's name with years, it was changed to be 'portrait of a person'. 
Some of the example titles are as below.
1. still life with vase of flowers
2. the large spoon
3. view from the elevated
4. equestrian portrait with gun and sword

## Evaluation Metrics
The BLEU (Bilingual Evaluation Understudy) score was used to evaluate the quality of predicted sentence compared to the references. BLEU evaluates how many n-grams in predicted sentence matches the references. 

## Model
The basic approach involves two parts: feature extraction using a pre-trained network (NASNetLarge) then sequence prediction for text descriptions using LSTM. 
Final model architecture: 

![architecture](/PNG/iter6_arch.png)


| Model | Unigram BLEU | Bigram BLEU | 3-Gram BLEU | 4-Gram BLEU |
| --- | --- | --- | --- | --- |
| Baseline | 0.32 | 0.15 | 0.09 | 0.03 |
| Final Model | 0.47 | 0.27 | 0.18 | 0.08 | 

## Example Performance
Compared to the baseline model with the minimum structure, the final model showed at least 47% increase in BLEU scores in 1 to 4 n-gram matches. Detailed look at the individual predictions showed that the model did good job in creating a syntactically accurate sentences for Flickr images, even though it often failed in referring to the word with correct semantics. On the other hand, its performance on describing art was still weak with many instances of incomplete sentences that failed to follow the correct syntax. Even though it did provide feasible descriptions for some of the items, a deeper training is necessary.

## Limitations & Future Directions
Essentially BLEU score assumes that the text references contain all the essence of that image. This is a feasible case for our Flickr data, which has 4-5 descriptions per image, but not the artwork data with only a title per image. At the current stage, the model is evaluated based on how well it classifies the real-life images then used to generate art title. It does not evaluate how accurately (or human-like, since art description is subjective) model describes art. To solve this problem, in the future, it will be worthwhile to collect numbers of human generated art descriptions to evaluate how model's description compares to human's. 
