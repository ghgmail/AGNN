# AGNN

## Introduction 

Attentive Graph Neural Network (AGNN) is a new recommendation framework tailored to knowledge graph-based personalized recommendation. Built upon the graph neural network framework, AGNN explicitly models the semantics encoded in KGs with expressive subgraphs to provide better recommendation with side information.

## Environment Requirement
+ Python 2.7
+ Pytorch (conda 4.5.11)
+ numpy == 1.15.4

## Datasets

+ MovieLens-1M
   + For the MoiveLens dataset, we crawl the corresponding IMDB dataset as movie auxiliary information, including genre, director, and actor. Note that we automatically remove the movies without auxilairy information. We then combined MovieLens and IMDB by movie title and released year. The combined data is save in a txt file (auxiliary.txt) and the format is as follows:    
   
         id:1|genre:Animation,Adventure,Comedy|director:John Lasseter|actors:Tom Hanks,Tim Allen,Don Rickles,Jim Varney
   
   + For the original user-movie rating data, we remove all items without auxiliary information. The data is save in a txt file (rating-delete-missing-itemid.txt) and the format is as follows:  
   
         userid itemid rating timestamp
   
+ Last-FM
   + This is the music listening dataset collected from Last.fm online music systems. Wherein, the tracks are viewed as the items. In particular, we take the subset of the dataset where the timestamp is from Jan, 2015 to June, 2015. For Last-FM,we map items into Freebase entities via title matching if there isa mapping available. 
+ Yelp
   + It records user ratings on local business scaled from 1-5. Additionally, social relations as well as business attributes (e.g., category, city) are also included. For Yelp, we extract item knowledge from the local business information network (e.g., category, location,
and attribute) as KG data. The format is as follows:

         id:11163|genre:Accountants,Professional Services,Tax Services,Financial Services|city:Peoria
      
## Modules 

For clarify, hereafter we use movieLens dataset as a toy example to demonstrate the detailed modules of AGNN. For Last-FM and yelp dataset, you need to do some adaptation for the code and tune some parameters.

+ Data Split (data-split.py)

   + Split the user-movie rating data into training and test data

   + Input Data: rating-delete-missing-itemid.txt

   + Output Data: training.txt, test.txt

+ Negative Sample (negative-sample.py)

   + Sample negative movies for each user to balance the model training
   
   + Input Data: training.txt
   
   + Output Data: negative.txt

+ Path Extraction （path-extraction.py）

   + Extract paths for both positive and negative user-moive interaction

   + Input Data: training.txt, negative.txt, auxiliary-mapping.txt,

   + Output Data: positive-path.txt, negative-path.txt

+ Attentive Graph Neural Network (model.py)

   + Feed both postive and negative path into the recurrent neural network, train and evaluate the model
   
   + Input Data: positive-path.txt, negative-path.txt, training.txt, test.txt, pre-train-user-embedding.txt, pre-train-movie-embedding.txt (To speed up model training process, the user and movie embedding is pre-trained via TransR[1]. You may also use TransE [2] or TransH [3] to pre-train the embeddings).

   + Output Data: results.txt

+ References

   [1] Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu. 2015. Learningentity and relation embeddings for knowledge graph      completion. In AAAI.

   [2] Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Ok-sana Yakhnenko. 2013.  Translating embeddings for        modeling multi-relationaldata. In NIPS. 2787–2795.

   [3] Zhen Wang, Jianwen Zhang, Jianlin Feng, and Zheng Chen. 2014. Knowledgegraph embedding by translating on hyperplanes. In AAAI.
