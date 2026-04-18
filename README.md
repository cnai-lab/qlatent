# Latent Constructs for Assessing the Psychology of Large AI Models  

In natural language processing (NLP), semantic relationships between words can be captured using  
a variety of different approaches, such as semantic word embeddings, transformer-based language models (a la BERT), encoder-decoder models (a la T5 and BART), and others.
While most embedding techniques consider the contexts of words, some consider sub-word components or even phonetics.[^1] 
Learning contextual language representations using transformers[^2] drove rapid progress in NLP and led to the development of tools readily accessible for researchers in a variety of disciplines.  
In this project, we refer to the various tools used to represent natural language collectively as **NLP models**.  

Most NLP models allow represeting words, phrases, sentences, and documents using mutidimentional coordinates, so called embedding. 
A vector in this coordinate system represents some concept. 
Similarity of concepts can be measured by, for example, the cosine similarity.[^3][^4]    
Coordinates of words may change depending on the language style, mood, and associations prevalent in the corpus on which the NLP models were trained.  


*Consider, for example, two chatbots - one trained using free text from the [SuicideWatch](https://www.reddit.com/r/SuicideWatch/) peer support group on Reddit and the other with free text from [partymusic](https://www.reddit.com/r/partymusic/) on the same platform. 
Intuitively, the answers of the two chatbots to the question `How do you feel today?` would be different. 
Now consider the kind of answers these two chatbots would provide to anxiety and depression questionnaires.*

The above example is overly simplistic in the sense that NLP models cannot be trained on the small amount of data of one subreddit, and the models' behavior depends on a variety of factors. 
We use this example only to illustrate the idea of querying an NLP model fitted to a corpus of messages produced by a specific population or after a specific event. 
Intuitively, the outputs of NLP models are biased toward associations prevalent in the training corpus.  

The main working hypothesis driving this library that **NLP models can capture – to a measurable extent – the emotional states reflected in the training corpus.**
Under the emotional state we include depression, anxiety, stress and burnout. 
We also include the positive aspects of wellbeing such as sense of coherence,[^5] professional fulfillment,[^6] and various coping strategies[^7] all collectively referred to as **Indicators of Resilience (IoRs)**.   

Traditionally IoRs are measured using questionnairs such as [GAD](https://www.hiv.uw.edu/page/mental-health-screening/gad-2), [PHQ](https://www.hiv.uw.edu/page/mental-health-screening/phq-2), [SPF](https://wellmd.stanford.edu/self-assessment.html#professional-fulfillment), and others. 
This library provides the toolset and guidelines to translating validated psychological questionnairs into querried for trained NLP models.  


[^1]: Ling, S., Salazar, J., Liu, Y., Kirchhoff, K., & Amazon, A. (2020). Bertphone: Phonetically-aware encoder representations for utterance-level speaker and language recognition. In Proc. Odyssey 2020 the speaker and language recognition workshop (pp. 9-16).
[^2]: Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
[^3]: Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems (pp. 3111-3119).
[^4]: Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. Science, 356(6334), 183-186.‏
[^5]: Antonovsky, A. (1987). Unraveling the mystery of health: How people manage stress and stay well. Jossey-bass.
[^6]: Trockel, M., Bohman, B., Lesure, E., Hamidi, M. S., Welle, D., Roberts, L., & Shanafelt, T. (2018). A brief instrument to assess both burnout and professional fulfillment in physicians: reliability and validity, including correlation with self-reported medical errors, in a sample of resident and practicing physicians. Academic Psychiatry, 42(1), 11-24.
[^7]: Lazarus, R. S., & Folkman, S. (1984). Stress, appraisal, and coping. Springer publishing company.

# The NLP Tasks Currently Supported by the `qlatent` Package

The "qlatent" package supports the following types of NLP tasks:
- MLM
- (M)NLI
- NSP
- CoLA

Descriptions of the available NLP tasks are provided below, along with steps and examples of how to implement them.

**IMPORTANT NOTE**: Results of items are displayed after using softmax or no softmax and filters.<br>
Item variations are first split into 1 or 2 groups according to the softmax arguments, and then within each group, the splits are created according to the filters.

## MLM - Masked Language Modeling

A masked language model (MLM) is a type of model commonly used in NLP. Given a text with some tokens masked (i.e., missing), the MLM predicts the probabilities of various tokens filling these masked positions, based on the corpus it was trained on.

In our package, we use the MLM task to evaluate the relative probability of a token to fill a masked token in a given text. The higher the score a token receives, the higher the probability that it correctly fills the masked token. Conversely, a lower score indicates a lower probability.

### Steps for Defining and Running a QMLM Question

In this guide, we'll construct the ASIQ2 question as a QMLM question.

1. Import the QMLM class and methods:
   ```
   from qlatent.qmlm.qmlm import *
   ```
2. Define a scale using the SCALE class (in this example - frequency keywords):
   ```
   frequency_weights:SCALE = {
     'never':-4,
     'very rarely':-3,
     'seldom':-2,
     'rarely':-2,
     'frequently':2,
     'often':2,
     'very frequently':3,
     'always':4,
   }
   ```
3. Define a class for your question:
   ```
   class ASIQ2(QMLM):
   ```
4. Define a dictionary (inside the class) of words that make the question's score positive and words that make it negative.<br>
   Make sure that there are at least 2 words with a positive value on the score and at least 2 with a negative value:
   ```
   kw_fairness = ["fairness", "equal chances", "equality"]
   kw_favors   = ['special favors', "privileges"]
   dict_objective = dict_pos_neg(kw_favors, kw_fairness,1)
   ```
5. Define your index and scale inside the class:
   ```
   index=["index"]
   scale="frequency"
   ```
6. Define an ```__init__``` function:
   
   6.1 Define a context template:
   ```
   template="Women are actually seeking for {index} that favor them over men under the guise of asking for equality. It is {frequency} correct."
   ```
   6.2 Define the question's dimensions (index and scale):
   ```
   dimensions={
               "index":self.dict_objective,
               "frequency":frequency_weights
   }
   ```
   6.3 Define the question's descriptor:
   ```
   descriptor = {"Questionnair":"ASI",
      "Factor":"H",
      "Ordinal":2,
      "Original":'H 2. Many women are actually seeking special favors, such as hiring policies that favor them over men, under the guise of asking for "equality."'
   }
   ```
   6.4 Put it all together, along with keyword arguments and use of the QMLM's ```__init___``` function:
   ```
   def __init__(self, **kwargs):
      super().__init__(
         template="Women are actually seeking for {index} that favor them over men under the guise of asking for equality. It is {frequency} correct.",
         dimensions={
                     "index":self.dict_objective,
                     "frequency":frequency_weights
         },
         descriptor = {"Questionnair":"ASI",
            "Factor":"H",
            "Ordinal":2,
            "Original":'H 2. Many women are actually seeking special favors, such as hiring policies that favor them over men, under the guise of asking for "equality."'
         },
         **kwargs,
      )
   ```
7. Here is how our code looks like after steps 1-6:
   ```
   from qlatent.qmlm.qmlm import *

   frequency_weights:SCALE = {
     'never':-4,
     'very rarely':-3,
     'seldom':-2,
     'rarely':-2,
     'frequently':2,
     'often':2,
     'very frequently':3,
     'always':4,
   }

   class ASIQ2(QMLM):

      kw_fairness = ["fairness", "equal chances", "equality"]
      kw_favors   = ['special favors', "privileges"]
      dict_objective = dict_pos_neg(kw_favors, kw_fairness,1)

      index=["index"]
      scale="frequency"
   
      def __init__(self, **kwargs):
         super().__init__(
            template="Women are actually seeking for {index} that favor them over men under the guise of asking for equality. It is {frequency} correct.",
            dimensions={
                        "index":self.dict_objective,
                        "frequency":frequency_weights
            },
            descriptor = {"Questionnair":"ASI",
               "Factor":"H",
               "Ordinal":2,
               "Original":'H 2. Many women are actually seeking special favors, such as hiring policies that favor them over men, under the guise of asking for "equality."'
            },
            **kwargs,
         )
   ```

8. Create a question object (note that parentheses aren't required here):
   ```
   Q = ASIQ2
   ```
9. Decide whether you'd like softmaxed results, raw results or both. The input-output order relationship is FIFO (first in first out):
  ```
  # Only softmaxed results: [True]
  # Only raw results: [False]
  # Softmaxed results before raw: [True, False]
  # Raw results before softmaxed: [False, True]
  ```
10. Decide on filters you'd like to use. You can use more than one filter. The input-output order relationship is FIFO.<br>
    All filters must be inside a dictionary. Here are a couple of examples:
  ```
  # Unfiltered filter: {"unfiltered" : {}}
  # Only positive keywords: {"positiveonly": Q.get_filter_for_postive_keywords()}
  # Both of the filters together: {"unfiltered" : {}, "positiveonly": Q.get_filter_for_postive_keywords()}
  ```
11. Create splits of the questions using the ```split_question``` function and everything we did in steps 8-10:
  ```
  Qs = split_question(Q,
                      index=Q.index,
                      scales=[Q.scale],
                      softmax=[True],
                      filters={'unfiltered':{},
                              "positiveonly":Q().get_filter_for_postive_keywords()
                              },
                      )
  ```
12. Create a MLM pipeline (in this example we used "distilbert/distilbert-base-uncased" as our MLM model):
  ```
  device = 0 if torch.cuda.is_available() else -1
   
  p = "distilbert/distilbert-base-uncased"
  mlm_pipeline = pipeline("fill-mask", device=device, model=p)
  mlm_pipeline.model_identifier = p
  ```
13. Run the question on the split you want to inspect. If you would like to inspect more than one split, you will have to run each split individually:
   ```
   # Run specific split (in this case - the split at index 0):
   Qs[0].run(mlm_pipeline)

   # Run all splits:
   for split in Qs:
      split.run(mlm_pipeline)

   # You can also print a report of the run by using report()
   Qs[0].run(mlm_pipeline).report()
   ```
14. Finally, after steps 1-13, our code looks like this:
   ```
   from qlatent.qmlm.qmlm import *

   frequency_weights:SCALE = {
       'never':-4,
       'very rarely':-3,
       'seldom':-2,
       'rarely':-2,
       'frequently':2,
       'often':2,
       'very frequently':3,
       'always':4,
   }

   class ASIQ2(QMLM):

      kw_fairness = ["fairness", "equal chances", "equality"]
      kw_favors   = ['special favors', "privileges"]
      dict_objective = dict_pos_neg(kw_favors, kw_fairness,1)

      index=["index"]
      scale="frequency"
   
      def __init__(self, **kwargs):
         super().__init__(
            template="Women are actually seeking for {index} that favor them over men under the guise of asking for equality. It is {frequency} correct.",
            dimensions={
                        "index":self.dict_objective,
                        "frequency":frequency_weights
            },
            descriptor = {"Questionnair":"ASI",
               "Factor":"H",
               "Ordinal":2,
               "Original":'H 2. Many women are actually seeking special favors, such as hiring policies that favor them over men, under the guise of asking for "equality."'
            },
            **kwargs,
         )

   Q = ASIQ2

   Qs = split_question(Q,
                       index=Q.index,
                       scales=[Q.scale],
                       softmax=[True],
                       filters={'unfiltered':{},
                                "positiveonly":Q().get_filter_for_postive_keywords()
                               },
                      )

   device = 0 if torch.cuda.is_available() else -1
      
   p = "distilbert/distilbert-base-uncased"
   mlm_pipeline = pipeline("fill-mask", device=device, model=p)
   mlm_pipeline.model_identifier = p

   Qs[0].run(mlm_pipeline)
   ```

## (M)NLI - (Multi-Genre) Natural Language Inference

NLI is a text classification NLP task that assigns a label or class to text.

When given 2 sentences, the NLI task assigns 1 of 3 labels to describe the relationship between them - either entailment, neutral, or contradiction.
- Entailment: The 2nd sentence is entailed by the 1st sentence.
- Neutral: The 2nd sentence is neither entailed by the 1st sentence nor contradicts it.
- Contradiction: The 2nd sentence contradicts the 1st sentence.

MNLI refers to the NLI task performed on sentences from numerous distinct genres, such as movie reviews, text messages, political statements, etc.

In our package, we use the MNLI task to evaluate the relative probability of a label to represent a relationship between 2 sentences. The higher the score a label receives, the higher the probability that it correctly represents the relationship. Conversely, a lower score indicates a lower probability.

**IMPORTANT NOTE**: There are 2 types of MNLI questions in this package: QMNLI and _QMNLI.<br>
A _QMNLI question is a simplified version of a QMNLI question, using positive and negative emotions as indexes instead of any index.

### Steps for Defining and Running a QMNLI Question

In this guide, we'll construct the SOCQ4 question as a QMNLI question.

1. Import the QMNLI classes and methods:
   ```
   from qlatent.qmnli.qmnli import *
   ```
2. Define a scale using the SCALE class (in this example - frequency keywords):
   ```
   frequency_weights:SCALE = {
     'never':-4,
     'very rarely':-3,
     'seldom':-2,
     'rarely':-2,
     'frequently':2,
     'often':2,
     'very frequently':3,
     'always':4,
   }
   ```
3. Define a class for your question:
   ```
   class SOCQ4(QMNLI):
   ```
4. Define your index and scale inside the class:
   ```
   index=["index"]
   scale="frequency"
   ```
5. Define a dictionary (inside the class) of words that make the question's score positive and words that make it negative.<br>
   Make sure that there are at least 2 words with a positive value on the score and at least 2 with a negative value:
   ```
   kw_attitude_neg = ["meaningless", "dull", "aimless", 'boring']
   kw_attitude_pos = ["meaningful", "interesting", "fulfilling", 'fascinating']
   dict_attitude = dict_pos_neg(kw_attitude_pos,kw_attitude_neg, 1.0)
   ```
6. Define an ```__init__``` function:
   
   6.1 Define a context template and an answer template:
   ```
   context_template="What goes around me is {index} to me.
   answer_template="It is {frequency} correct."
   ```
   6.2 Define the question's dimensions (index and scale):
   ```
    dimensions={
      "frequency":frequency_weights,
      "index":self.dict_attitude,
      }
   ```
   6.3 Define the question's descriptor:
   ```
   descriptor = {"Questionnair":"SOC",
     "Factor":"Meaningfulness",
     "Ordinal":4,
     "Original":"Do you have the feeling that you don’t really care what goes on around you? "
     }
   ```
   6.4 Put it all together, along with keyword arguments and use of the QMNLI's ```__init___``` function:
   ```
   def __init__(self, **kwargs):
     super().__init__(
       context_template="What goes around me is {index} to me.",
       answer_template="It is {frequency} correct.",
       dimensions={
         "frequency":frequency_weights,
         "index":self.dict_attitude,
       },
       descriptor = {"Questionnair":"SOC",
         "Factor":"Meaningfulness",
         "Ordinal":4,
         "Original":"Do you have the feeling that you don’t really care what goes on around you? "
       },
       **kwargs,
     )
   ```
7. Here is how our code looks like after steps 1-6:
   ```
   from qlatent.qmnli.qmnli import *

   frequency_weights:SCALE = {
     'never':-4,
     'very rarely':-3,
     'seldom':-2,
     'rarely':-2,
     'frequently':2,
     'often':2,
     'very frequently':3,
     'always':4,
     }

   class SOCQ4(QMNLI):
  
      index=["index"]
      scale="frequency"
   
      kw_attitude_neg = ["meaningless", "dull", "aimless", 'boring']
      kw_attitude_pos = ["meaningful", "interesting", "fulfilling", 'fascinating']
      dict_attitude = dict_pos_neg(kw_attitude_pos,kw_attitude_neg, 1.0)
   
      def __init__(self, **kwargs):
         super().__init__(
            context_template="What goes around me is {index} to me.",
            answer_template="It is {frequency} correct.",
            dimensions={
               "frequency":frequency_weights,
               "index":self.dict_attitude,
             },
            descriptor = {"Questionnair":"SOC",
               "Factor":"Meaningfulness",
               "Ordinal":4,
               "Original":"Do you have the feeling that you don’t really care what goes on around you? "
            },
            **kwargs,
         )
   ```
8. Create a question object (note that parentheses aren't required here):
   ```
   Q = SOCQ4
   ```
9. Decide whether you'd like softmaxed results, raw results or both. The input-output order relationship is FIFO (first in first out):
  ```
  # Only softmaxed results: [True]
  # Only raw results: [False]
  # Softmaxed results before raw: [True, False]
  # Raw results before softmaxed: [False, True]
  ```
10. Decide on filters you'd like to use. You can use more than one filter. The input-output order relationship is FIFO.<br>
    All filters must be inside a dictionary. Here are a couple of examples:
  ```
  # Unfiltered filter: {"unfiltered" : {}}
  # Only positive keywords: {"positiveonly": Q.get_filter_for_postive_keywords()}
  # Both of the filters together: {"unfiltered" : {}, "positiveonly": Q.get_filter_for_postive_keywords()}
  ```
11. Create splits of the questions using the ```split_question``` function and everything we did in steps 8-10:
  ```
  Qs = split_question(Q,
                      index=Q.index,
                      scales=[Q.scale],
                      softmax=[True],
                      filters={'unfiltered':{},
                              "positiveonly":Q().get_filter_for_postive_keywords()
                              },
                      )
  ```
12. Create a NLI pipeline (in this example we will use "typeform/distilbert-base-uncased-mnli" as our NLI model):
  ```
  device = 0 if torch.cuda.is_available() else -1
   
  p = "typeform/distilbert-base-uncased-mnli"
  nli_pipeline = pipeline("zero-shot-classification",device=device, model=p)
  nli_pipeline.model_identifier = p
  ```
13. Run the question on the split you want to inspect. If you would like to inspect more than one split, you will have to run each split individually:
   ```
   # Run specific split (in this case - the split at index 0):
   Qs[0].run(nli_pipeline)

   # Run all splits:
   for split in Qs:
      split.run(nli_pipeline)

   # You can also print a report of the run by using report()
   Qs[0].run(nli_pipeline).report()
   ```
14. Finally, after steps 1-13, our code looks like this:
   ```
   from qlatent.qmnli.qmnli import *

   frequency_weights:SCALE = {
       'never':-4,
       'very rarely':-3,
       'seldom':-2,
       'rarely':-2,
       'frequently':2,
       'often':2,
       'very frequently':3,
       'always':4,
   }

   class SOCQ4(QMNLI):
  
      index=["index"]
      scale="frequency"
   
      kw_attitude_neg = ["meaningless", "dull", "aimless", 'boring']
      kw_attitude_pos = ["meaningful", "interesting", "fulfilling", 'fascinating']
      dict_attitude = dict_pos_neg(kw_attitude_pos,kw_attitude_neg, 1.0)
   
      def __init__(self, **kwargs):
         super().__init__(
            context_template="What goes around me is {index} to me.",
            answer_template="It is {frequency} correct.",
            dimensions={
               "frequency":frequency_weights,
               "index":self.dict_attitude,
            },
            descriptor = {"Questionnair":"SOC",
               "Factor":"Meaningfulness",
               "Ordinal":4,
               "Original":"Do you have the feeling that you don’t really care what goes on around you? "
            },
            **kwargs,
         )

   Q = SOCQ4

   Qs = split_question(Q,
                       index=Q.index,
                       scales=[Q.scale],
                       softmax=[True],
                       filters={'unfiltered':{},
                                "positiveonly":Q().get_filter_for_postive_keywords()
                               },
                      )

   device = 0 if torch.cuda.is_available() else -1
      
   p = "typeform/distilbert-base-uncased-mnli"
   nli_pipeline = pipeline("zero-shot-classification",device=device, model=p)
   nli_pipeline.model_identifier = p

   Qs[0].run(nli_pipeline)
   ```

### Steps for Defining and Running a _QMNLI Question

In this guide, we'll construct the GAD7Q1 question as a _QMNLI question.

1. Follow steps 1-2 of the "Steps for Defining and Running a QMNLI Question" guide.
2. Define a class for your question:
   ```
   class GAD7Q1(_QMNLI):
   ```
3. Follow step 6 of the "Steps for Defining and Running a QMNLI Question" guide in the following manner:
   
   3.1 Define a context template and an answer template:
   ```
   context="Over the last 2 weeks, I feel {emotion}."
   template="It is {intensifier} correct."
   ```
   3.2 Define 2 lists: a list of emotions that provide positive scores, and another list of emotions that provide negative scores:
   ```
   emo_pos=['nervous', 'anxious', 'on edge']
   emo_neg=['calm', 'peaceful', 'relaxed']
   ```
   3.3 Define the question's intensifiers:
   ```
   intensifiers=frequency_weights
   ```
   3.4 Define the question's descriptor:
   ```
   descriptor = {"Questionnair":"GAD7",
     "Factor":"GAD",
     "Ordinal":1,
     "Original":"Over the last 2 weeks, how often have you been bothered by the following problems? Feeling nervous, anxious or on edge"
     }
   ```
   3.5 Put it all together, along with keyword arguments and use of the _QMNLI's ```__init___``` function:
   ```
   def __init__(self, **kwargs):
      super().__init__(
         context="Over the last 2 weeks, I feel {emotion}.",
         template="It is {intensifier} correct.",
         emo_pos=['nervous', 'anxious', 'on edge'],
         emo_neg=['calm', 'peaceful', 'relaxed'],
         intensifiers=frequency_weights,
         descriptor = {"Questionnair":"GAD7",
                       "Factor":"GAD",
                       "Ordinal":1,
                       "Original":"Over the last 2 weeks, how often have you been bothered by the following problems? Feeling nervous, anxious or on edge"
                      }
      **kwargs,
      )
   ```
4. Here is how our code looks like after steps 1-3:
   ```
   from qlatent.qmnli.qmnli import *

   frequency_weights:SCALE = {
     'never':-4,
     'very rarely':-3,
     'seldom':-2,
     'rarely':-2,
     'frequently':2,
     'often':2,
     'very frequently':3,
     'always':4,
   }

   class GAD7Q1(_QMNLI):

      def __init__(self, **kwargs):
         super().__init__(
            context="Over the last 2 weeks, I feel {emotion}.",
            template="It is {intensifier} correct.",
            emo_pos=['nervous', 'anxious', 'on edge'],
            emo_neg=['calm', 'peaceful', 'relaxed'],
            intensifiers=frequency_weights,
            descriptor = {"Questionnair":"GAD7",
                          "Factor":"GAD",
                          "Ordinal":1,
                          "Original":"Over the last 2 weeks, how often have you been bothered by the following problems? Feeling nervous, anxious or on edge"
                         }
         **kwargs,
         )
   ```
5. Create a question object (note that parentheses aren't required here):
   ```
   Q = GAD7Q1
   ```
6. Follow steps 9-13 of the "Steps for defining and running a QMNLI question" guide.
7. Finally, after steps 1-6, our code looks like this:
   ```
   from qlatent.qmnli.qmnli import *

   frequency_weights:SCALE = {
     'never':-4,
     'very rarely':-3,
     'seldom':-2,
     'rarely':-2,
     'frequently':2,
     'often':2,
     'very frequently':3,
     'always':4,
   }

   class GAD7Q1(_QMNLI):

      def __init__(self, **kwargs):
         super().__init__(
            context="Over the last 2 weeks, I feel {emotion}.",
            template="It is {intensifier} correct.",
            emo_pos=['nervous', 'anxious', 'on edge'],
            emo_neg=['calm', 'peaceful', 'relaxed'],
            intensifiers=frequency_weights,
            descriptor = {"Questionnair":"GAD7",
                          "Factor":"GAD",
                          "Ordinal":1,
                          "Original":"Over the last 2 weeks, how often have you been bothered by the following problems? Feeling nervous, anxious or on edge"
                         }
         **kwargs,
         )

   Q = GAD7Q1

   Qs = split_question(Q,
                       index=Q.index,
                       scales=[Q.scale],
                       softmax=[True],
                       filters={'unfiltered':{},
                                "positiveonly":Q().get_filter_for_postive_keywords()
                               },
                      )

   device = 0 if torch.cuda.is_available() else -1
      
   p = "typeform/distilbert-base-uncased-mnli"
   nli_pipeline = pipeline("zero-shot-classification",device=device, model=p)
   nli_pipeline.model_identifier = p

   Qs[0].run(nli_pipeline)
   ```

## NSP - Next Sequence Predicion

Given 2 sequences of characters, NSP is an NLP task that checks how good the 2nd sequence is as a follow up for the 1st sequence.<br>
In the qlatent package's case, the sequences are sentences that represent questions and their corresponding possible answers. The higher the score the 2nd sequence gets - the higher the likelyhood that the 2nd sequence will come after the 1st sequence. Conversely, a lower score indicates a lower likelyhood for the 2nd sequence to follow the 1st sequence.

### Steps for Defining and Running a QNSP Question

In this guide, we'll construct the ASIQ2 question as a QNSP question.

1. Import the QNSP class and methods:
   ```
   from qlatent.qnsp.qnsp import *
   ```
2. Define a scale using the SCALE class (in this example - frequency keywords):
   ```
   frequency_weights:SCALE = {
     'never':-4,
     'very rarely':-3,
     'seldom':-2,
     'rarely':-2,
     'frequently':2,
     'often':2,
     'very frequently':3,
     'always':4,
   }
   ```
3. Define a class for your question:
   ```
   class ASIQ2(QNSP):
   ```
4. Define a dictionary (inside the class) of words that make the question's score positive and words that make it negative.<br>
   Make sure that there are at least 2 "positive words" and at least 2 "negative words":
   ```
   kw_fairness = ["equal chances", "equality", 'fair opportunities']
   kw_favors   = ['special favors', "privileges", 'preferential treatment']
   dict_objective = dict_pos_neg(kw_favors, kw_fairness,1)
   ```
5. Define an ```__init__``` function:
   
   5.1 Define a context template and a next sentence template:
   ```
   prompt='Women seek {index} under the guise of asking for equality.'
   next_sentence="It is {frequency} correct"
   ```
   5.2 Define the question's dimensions (index and scale):
   ```
   dimensions={
               'frequency':frequency_weights,  
               'index':self.dict_objective,
   }
   ```
   5.3 Define the question's descriptor:
   ```
   descriptor = {"Questionnair":"ASI",
                 "Factor":"H",
                 "Ordinal":2,
                 "Original":'H 2. Many women are actually seeking special favors, such as hiring policies that favor them over men, under the guise of asking for "equality."'
   }
   ```
   5.4 Put it all together, along with a few additions:
   ```
   def __init__(self, **kwargs):
      super().__init__(
         prompt='Women seek {index} under the guise of asking for equality.',
         next_sentence="It is {frequency} correct",
         dimensions={
                     "index":self.dict_objective,
                     "frequency":frequency_weights
         },
         descriptor = {"Questionnair":"ASI",
            "Factor":"H",
            "Ordinal":2,
            "Original":'H 2. Many women are actually seeking special favors, such as hiring policies that favor them over men, under the guise of asking for "equality."'
         },
         **kwargs,
      )
   ```
6. Your code should look like this after following steps 1-5:
   ```
   from qlatent.qnsp.qnsp import *

   frequency_weights:SCALE = {
     'never':-4,
     'very rarely':-3,
     'seldom':-2,
     'rarely':-2,
     'frequently':2,
     'often':2,
     'very frequently':3,
     'always':4,
   }

   class ASIQ2(QNSP):

      kw_fairness = ["equal chances", "equality", 'fair opportunities']
      kw_favors   = ['special favors', "privileges", 'preferential treatment']
      dict_objective = dict_pos_neg(kw_favors, kw_fairness,1)

      def __init__(self, **kwargs):
         super().__init__(
            prompt='Women seek {index} under the guise of asking for equality.',
            next_sentence="It is {frequency} correct",
            dimensions={
                        "index":self.dict_objective,
                        "frequency":frequency_weights
            },
            descriptor = {"Questionnair":"ASI",
               "Factor":"H",
               "Ordinal":2,
               "Original":'H 2. Many women are actually seeking special favors, such as hiring policies that favor them over men, under the guise of asking for "equality."'
            },
            **kwargs,
         )
   ```
7. Create a question object (note that parentheses aren't required here):
   ```
   Q = ASIQ2
   ```
8. Decide whether you'd like softmaxed results, raw rwsults or both. Order matters, meaning whatever you'll put first will come out first too:
   ```
   # Only softmaxed results: [True]
   # Only raw results: [False]
   # Softmaxed results before raw: [True, False]
   # Raw results before softmaxed: [False, True]
   ```
9. Decide on filters you'd like to use. You can use more than one filter, and filters will be displayed according to the order in which you provided them.
   All filters must be inside a dictionary. Here are a couple of examples:
   ```
   # Unfiltered filter: {"unfiltered" : {}}
   # Only positive keywords: {"positiveonly": Q.get_filter_for_postive_keywords()}
   # Both of the filters together: {"unfiltered" : {}, "positiveonly": Q.get_filter_for_postive_keywords()}
   ```
10. Create splits of the questions using the ```split_question``` function and everything we did in steps 7-9:
   ```
   Qs = split_question(Q,
                      index=["index",],
                      scales=['frequency'],
                      softmax=[True],
                      filters={'unfiltered':{},
                              "positiveonly":Q().get_filter_for_postive_keywords()
                              },
                      )
   ```
11. Create a NSP pipeline (in this example we will use "google-bert/bert-base-uncased" as our NSP model):
   ```
   device = 0 if torch.cuda.is_available() else -1
   
   p = "google-bert/bert-base-uncased"
   nsp_pipeline = NextSentencePredictionPipeline(p)
   nsp_pipeline.model_identifier = p
   ```
12. Run the question on the split you'd want to inspect. If you'd like to inspect more than one split, you'll have to run each split individually:
   ```
   # Run specific split (in this case - the split at index 0):
   Qs[0].run(nsp_pipeline)
   
   # Run all splits:
   for split in Qs:
      split.run(nsp_pipeline)

   # You can also print a report of the run by using report()
   Qs[0].run(nsp_pipeline).report()
   ```
13. In the end (after steps 1-12), your code should look like this:
   ```
   from qlatent.qnsp.qnsp import *

   frequency_weights:SCALE = {
     'never':-4,
     'very rarely':-3,
     'seldom':-2,
     'rarely':-2,
     'frequently':2,
     'often':2,
     'very frequently':3,
     'always':4,
   }

   class ASIQ2(QNSP):

      kw_fairness = ["equal chances", "equality", 'fair opportunities']
      kw_favors   = ['special favors', "privileges", 'preferential treatment']
      dict_objective = dict_pos_neg(kw_favors, kw_fairness,1)

      def __init__(self, **kwargs):
         super().__init__(
            prompt='Women seek {index} under the guise of asking for equality.',
            next_sentence="It is {frequency} correct",
            dimensions={
                        "index":self.dict_objective,
                        "frequency":frequency_weights
            },
            descriptor = {"Questionnair":"ASI",
               "Factor":"H",
               "Ordinal":2,
               "Original":'H 2. Many women are actually seeking special favors, such as hiring policies that favor them over men, under the guise of asking for "equality."'
            },
            **kwargs,
         )
   
   Q = ASIQ2
   
   Qs = split_question(Q,
                       index=["index",],
                       scales=['frequency'],
                       softmax=[True],
                       filters={'unfiltered':{},
                                "positiveonly":Q().get_filter_for_postive_keywords()
                               },
                      )
   
   device = 0 if torch.cuda.is_available() else -1
      
   p = "google-bert/bert-base-uncased"
   nsp_pipeline = NextSentencePredictionPipeline(p)
   nsp_pipeline.model_identifier = p
   
   Qs[0].run(nsp_pipeline)
   ```

## CoLA - Corpus of Linguistic Acceptability

The CoLA task is an NLP task that checks the grammatical acceptability of English text. The higher the score a text gets - the more grammatically correct the text is. Conversely, a lower score indicates a more grammatically incorrect text.

### Steps for Defining and Running a QCOLA Question

In this guide, we'll construct the ASIQ2 question as a QCOLA question.

1. Import the QCOLA class and methods:
   ```
   from qlatent.qcola.qcola import *
   ```
2. Define a scale using the SCALE class (in this example - frequency keywords):
   ```
   frequency_weights:SCALE = {
     'never':-4,
     'very rarely':-3,
     'seldom':-2,
     'rarely':-2,
     'frequently':2,
     'often':2,
     'very frequently':3,
     'always':4,
   }
   ```
3. Define a class for your question:
   ```
   class ASIQ2(QCOLA):
   ```
4. Define a dictionary (inside the class) of words that make the question's score positive and words that make it negative.<br>
   Make sure that there are at least 2 "positive words" and at least 2 "negative words":
   ```
   kw_fairness = ["equal rights", 'impartiality', 'equality']
   kw_favors   = ['special favors', 'privileges', 'preferential treatment']
   dict_objective = dict_pos_neg(kw_favors, kw_fairness,1)
   ```
5. Define an ```__init__``` function:
   
   5.1 Define a context template:
   ```
   template='Women {frequency} seek {index} when asking for equality.',
   ```
   5.2 Define the question's dimensions (index and scale):
   ```
   dimensions={
               'frequency':frequency_weights,  
               'index':self.dict_objective,
   }
   ```
   5.3 Define the question's descriptor:
   ```
   descriptor = {"Questionnair":"ASI",
                 "Factor":"H",
                 "Ordinal":2,
                 "Original":'H 2. Many women are actually seeking special favors, such as hiring policies that favor them over men, under the guise of asking for "equality."'
   }
   ```
   5.4 Put it all together, along with a few additions:
   ```
   def __init__(self, **kwargs):
      super().__init__(
         template='Women {frequency} seek {index} when asking for equality.',
         dimensions={
                     "index":self.dict_objective,
                     "frequency":frequency_weights
         },
         descriptor = {"Questionnair":"ASI",
            "Factor":"H",
            "Ordinal":2,
            "Original":'H 2. Many women are actually seeking special favors, such as hiring policies that favor them over men, under the guise of asking for "equality."'
         },
         **kwargs,
      )
   ```
6. Your code should look like this after following steps 1-5:
   ```
   from qlatent.qcola.qcola import *

   frequency_weights:SCALE = {
     'never':-4,
     'very rarely':-3,
     'seldom':-2,
     'rarely':-2,
     'frequently':2,
     'often':2,
     'very frequently':3,
     'always':4,
   }

   class ASIQ2(QCOLA):

      kw_fairness = ["equal rights", 'impartiality', 'equality']
      kw_favors   = ['special favors', 'privileges', 'preferential treatment']
      dict_objective = dict_pos_neg(kw_favors, kw_fairness,1)

      def __init__(self, **kwargs):
      super().__init__(
         template='Women {frequency} seek {index} when asking for equality.',
         dimensions={
                     "index":self.dict_objective,
                     "frequency":frequency_weights
         },
         descriptor = {"Questionnair":"ASI",
            "Factor":"H",
            "Ordinal":2,
            "Original":'H 2. Many women are actually seeking special favors, such as hiring policies that favor them over men, under the guise of asking for "equality."'
         },
         **kwargs,
      )
   ```
7. Create a question object (note that parentheses aren't required here):
   ```
   Q = ASIQ2
   ```
8. Decide whether you'd like softmaxed results, raw rwsults or both. Order matters, meaning whatever you'll put first will come out first too:
   ```
   # Only softmaxed results: [True]
   # Only raw results: [False]
   # Softmaxed results before raw: [True, False]
   # Raw results before softmaxed: [False, True]
   ```
9. Decide on filters you'd like to use. You can use more than one filter, and filters will be displayed according to the order in which you provided them.
   All filters must be inside a dictionary. Here are a couple of examples:
   ```
   # Unfiltered filter: {"unfiltered" : {}}
   # Only positive keywords: {"positiveonly": Q.get_filter_for_postive_keywords()}
   # Both of the filters together: {"unfiltered" : {}, "positiveonly": Q.get_filter_for_postive_keywords()}
   ```
10. Create splits of the questions using the ```split_question``` function and everything we did in steps 8-10:
   ```
   Qs = split_question(Q,
                       index=["index",],
                       scales=['frequency'],
                       softmax=[True],
                       filters={'unfiltered':{},
                                "positiveonly":Q().get_filter_for_postive_keywords()
                               },
                      )
   ```
11. Create a CoLA pipeline (in this example we will use "mrm8488/deberta-v3-small-finetuned-cola" as our NLI model):
   ```
   device = 0 if torch.cuda.is_available() else -1
   
   p = "mrm8488/deberta-v3-small-finetuned-cola"
   cola_pipeline = pipeline("text-classification", device=device, model = p)
   cola_pipeline.model_identifier = p
   ```
12. Run the question on the split you'd want to inspect. If you'd like to inspect more than one split, you'll have to run each split individually:
   ```
   # Run specific split (in this case - the split at index 0):
   Qs[0].run(cola_pipeline)
   
   # Run all splits:
   for split in Qs:
      split.run(cola_pipeline)
   
   # You can also print a report of the run by using report()
   Qs[0].run(cola_pipeline).report()
   ```
13. In the end (after steps 1-12), your code should look like this:
   ```
   from qlatent.qcola.qcola import *

   frequency_weights:SCALE = {
     'never':-4,
     'very rarely':-3,
     'seldom':-2,
     'rarely':-2,
     'frequently':2,
     'often':2,
     'very frequently':3,
     'always':4,
   }

   class ASIQ2(QCOLA):

      kw_fairness = ["equal rights", 'impartiality', 'equality']
      kw_favors   = ['special favors', 'privileges', 'preferential treatment']
      dict_objective = dict_pos_neg(kw_favors, kw_fairness,1)

      def __init__(self, **kwargs):
         super().__init__(
            template='Women {frequency} seek {index} when asking for equality.',
            dimensions={
                        "index":self.dict_objective,
                        "frequency":frequency_weights
            },
            descriptor = {"Questionnair":"ASI",
               "Factor":"H",
               "Ordinal":2,
               "Original":'H 2. Many women are actually seeking special favors, such as hiring policies that favor them over men, under the guise of asking for "equality."'
            },
            **kwargs,
         )
   
   Q = ASIQ2
   
   Qs = split_question(Q,
                       index=["index",],
                       scales=['frequency'],
                       softmax=[True],
                       filters={'unfiltered':{},
                                "positiveonly":Q().get_filter_for_postive_keywords()
                               },
                      )
   
   device = 0 if torch.cuda.is_available() else -1
      
   p = "mrm8488/deberta-v3-small-finetuned-cola"
   cola_pipeline = pipeline("text-classification", device=device, model = p)
   cola_pipeline.model_identifier = p
   
   Qs[0].run(cola_pipeline)
   ```

# Model Training Utility

This class is a utility for training language models using the Hugging Face Transformers library. It supports training with both Masked Language Modeling (MLM) and Natural Language Inference (NLI) heads, and includes functionality to handle dataset formats, copying weights, and saving checkpoints.

## Table of Contents

- [Installation](#installation)
- [Callbacks](#callbacks)
  - [__init__](#init)
  - [on_epoch_end](#on_epoch_end)
- [DataLoader](#dataloader)
  - [__init__](#init-DataLoader)
  - [_print_dataset_status](#_print_dataset_status)
  - [_convert_labels_to_numeric](#_convert_labels_to_numeric)
  - [_load_csv_data](#_load_csv_data)
  - [_prepare_dict_dataset](#_prepare_dict_dataset)
- [Public Methods](#public-methods)
  - [train_head](#train_head)
  - [fix_model_embedding_layer](#fix_model_embedding_layer)
  - [init_head](#init_head)
  - [get_non_base_layers](#get_non_base_layers)
  - [attach_head_to_model](#attach_head_to_model)
- [Formats and Parameter Constraints](#formats-and-parameter-constraints)
  - [Dataset Format](#dataset-format)
  - [CSV Format](#csv-format)
  - [Dictionary Format](#dictionary-format)
  - [Parameter Constraints](#parameter-constraints)
- [Usage Examples](#usage-examples)
  - [Example of an MLM head loaded from an MLM model, trained with dataset (saving checkpoint).](#example-of-an-mlm-head-loaded-from-an-mlm-model-trained-with-dataset-saving-checkpoint)
  - [Example of an MLM head loaded from an MLM model, trained with CSV.](#example-of-an-mlm-head-loaded-from-an-mlm-model-trained-with-csv)
  - [Example of an MLM head loaded from an MLM model, trained with a dictionary of lists.](#example-of-an-mlm-head-loaded-from-an-mlm-model-trained-with-a-dictionary-of-lists)
  - [Example of an MLM head loaded from an NLI model with copied weights and biases from a trained MLM head, trained on dataset, but saving the NLI head instead of the trained MLM head.](#example-of-an-mlm-head-loaded-from-an-nli-model-with-copied-weights-and-biases-from-a-trained-mlm-head-trained-on-dataset-but-saving-the-nli-head-instead-of-the-trained-mlm-head)
  - [Example of an MLM head loaded from an NLI model with weights and biases initialized randomly, trained with dataset.](#example-of-an-mlm-head-loaded-from-an-nli-model-with-weights-and-biases-initialized-randomly-trained-with-dataset)
  - [Example of an MLM head loaded from an NLI model with weights and biases initialized randomly, trained with CSV.](#example-of-an-mlm-head-loaded-from-an-nli-model-with-weights-and-biases-initialized-randomly-trained-with-csv)
  - [Example of an MLM head loaded from an NLI model with copied weights and biases from a trained MLM head, trained on CSV.](#example-of-an-mlm-head-loaded-from-an-nli-model-with-copied-weights-and-biases-from-a-trained-mlm-head-trained-on-csv)
  - [Example of an MLM head loaded from an NLI model with copied weights and biases from a trained MLM head, trained on dataset.](#example-of-an-mlm-head-loaded-from-an-nli-model-with-copied-weights-and-biases-from-a-trained-mlm-head-trained-on-dataset)
  - [Example of an NLI head loaded from an NLI model, trained on CSV.](#example-of-an-nli-head-loaded-from-an-nli-model-trained-on-csv)
  - [Example of an NLI head loaded from an NLI model, trained on dataset.](#example-of-an-nli-head-loaded-from-an-nli-model-trained-on-dataset)
  - [Example of an NLI head loaded from an NLI model, trained on a dictionary of lists.](#example-of-an-nli-head-loaded-from-an-nli-model-trained-on-a-dictionary-of-lists)
  - [Example of an NLI head loaded from an MLM model with weights and biases initialized randomly.](#example-of-an-nli-head-loaded-from-an-mlm-model-with-weights-and-biases-initialized-randomly)
  - [Example of an NLI head loaded from an MLM model with copied weights and biases from a trained NLI head.](#example-of-an-nli-head-loaded-from-an-mlm-model-with-copied-weights-and-biases-from-a-trained-nli-head)


## Installation

To use this code, you need to install the required packages. You can do this by running:

```bash
pip install qlatent transformers datasets torch numpy
```

To import the ModelTrainer utility do the following:
```bash
from qlatent.utils import ModelTrainer
```

## Callbacks

The `SaveCheckpointByEpochCallback` is a callback class that facilitates saving model and tokenizer states at the end of each training epoch. It allows periodic checkpoints during training, storing them in a specified directory for later use or evaluation.

#### Functions:

#### __init__
`__init__(self, output_dir: str, tokenizer, save_checkpoint: bool, epochs_to_save: list[int], head_to_save)`

This is the initialization method for the callback.

##### Parameters:

- `output_dir (str)`: The directory where the checkpoints will be saved.
- `tokenizer`: The tokenizer associated with the model being trained.
- `save_checkpoint (bool)`: A flag indicating whether checkpoints should be saved. Default is `False`.
- `epochs_to_save (list[int])`: A list specifying which epochs to save checkpoints for. If empty or `None`, checkpoints will be saved at every epoch.
- `head_to_save`: If specified, the head model to save during the checkpoint.

#### on_epoch_end
`on_epoch_end(self, args, state, control, model=None, **kwargs)`

This method is automatically called by the Trainer at the end of each epoch. It saves the model and tokenizer to a subdirectory named after the current epoch.

##### Parameters:

- `args`: The training arguments.
- `state`: The current state of the Trainer.
- `control`: The current control object.
- `model`: The model being trained. Default is `None`.
- `**kwargs`: Additional keyword arguments.


## Dataloader

---

The `DataLoader` class is designed to load, process, and prepare datasets for two types of natural language processing tasks: Natural Language Inference (NLI) and Masked Language Modeling (MLM). It supports loading data from CSV files using the Hugging Face `load_dataset` function or directly from dictionary objects, converting the data into a Hugging Face `DatasetDict` for seamless integration with NLP pipelines.

---

### __init__-DataLoader
`__init__(self, label2_id)`
- **Description:**  
  Initializes the `DataLoader` instance with label-to-ID mapping used for converting string labels into numeric values in NLI tasks.
- **Parameters:**  
  - `label2_id`: A dictionary mapping textual labels to numeric IDs (e.g., `{'entailment': 0, 'neutral': 1, 'contradiction': 2}`).

---

## Methods

### _print_dataset_status 
`_print_dataset_status(self, dataset, task_type)`
- **Description:**  
  Prints the number of samples in the training split and, if available, the validation split.
- **Parameters:**  
  - `dataset`: A dictionary containing the data splits (typically with keys such as `'train'` and optionally `'validation'`).
  - `task_type` (str): Specifies the type of task.  
    - For NLI tasks (`'nli'`), the sample count is determined by the length of the `'premise'` list.
    - For MLM tasks (`'mlm'`), the sample count is determined by the length of the dataset.
- **Behavior:**  
  - Displays the number of training samples.
  - If a validation split exists, displays the number of validation samples.

---

### _convert_labels_to_numeric 
`_convert_labels_to_numeric(self, dataset, task_type)`
- **Description:**  
  Converts non-numeric labels in the dataset to numeric labels using the provided `label2_id` mapping. This conversion is applied only for NLI tasks when `label2_id` is available.
- **Parameters:**  
  - `dataset`: A Hugging Face `Dataset` object.
  - `task_type` (str): The type of task; conversion is applied only if `task_type` is `'nli'`.
- **Returns:**  
  - The modified dataset with labels converted to numeric values where applicable.

---

### _load_csv_data 
`_load_csv_data(self, dataset_path, task_type, num_percentage_validation, val_dataset)`
- **Description:**  
  Loads CSV data using the Hugging Face `load_dataset` function and prepares it for the specified task type (NLI or MLM). Supports splitting the training data into a validation set based on a specified percentage if a separate validation dataset is not provided.
- **Parameters:**  
  - `dataset_path` (str): Path to the CSV file containing the training data.
  - `task_type` (str): Indicates the task type; accepted values are `'nli'` and `'mlm'`.
  - `num_percentage_validation` (float): A float between 0 and 1 representing the proportion of the training data to use for validation if a separate validation file is not provided.
  - `val_dataset` (optional): Path to a CSV file containing a separate validation dataset.
- **Behavior:**  
  - Loads the training (and optionally validation) data using the `load_dataset` function.
  - If a separate validation dataset is not provided and a valid percentage is specified, the training data is split into training and validation sets.
  - Applies numeric conversion of labels for NLI tasks.
- **Returns:**  
  - A Hugging Face `DatasetDict` with `'train'` and, if applicable, `'validation'` splits.

---

### _prepare_dict_dataset
`_prepare_dict_dataset(self, data_dict, task_type, num_percentage_validation, val_dataset)`
- **Description:**  
  Prepares a dataset from a provided dictionary, ensuring it is in the correct format for either NLI or MLM tasks. It can handle cases where a validation set is provided separately or needs to be generated by splitting the training data.
- **Parameters:**  
  - `data_dict` (dict):  
    - For NLI tasks, it must include a `'train'` key with sub-keys `'premise'`, `'hypothesis'`, and `'label'`. Optionally, a `'validation'` key may also be provided.
    - For MLM tasks, the `'train'` key should map to a list of texts. A separate `'validation'` key can also be provided.
  - `task_type` (str): Specifies the task type (`'nli'` or `'mlm'`).
  - `num_percentage_validation` (float): A float between 0 and 1 that determines the percentage of the training data to use for validation if a separate validation set is not provided.
  - `val_dataset` (optional): A separate validation dataset (as a dictionary) if available.
- **Behavior:**  
  - Validates the format of the input dictionary based on the task type.
  - **For NLI:**
    - Ensures the presence of required keys (`'premise'`, `'hypothesis'`, and `'label'`).
    - Converts the dictionary splits into Hugging Face `Dataset` objects.
    - If no validation set is provided and a valid percentage is specified, it splits the training data accordingly.
  - **For MLM:**
    - Checks that the training data is provided as a list of texts.
    - Processes the available splits similarly, including an optional split for validation.
  - Converts non-numeric labels to numeric values where necessary.
- **Returns:**  
  - A Hugging Face `DatasetDict` containing the prepared `'train'` split and, if applicable, the `'validation'` split.


---

## Public Methods

### train_head
Trains a model head for either MLM or NLI tasks. Depending on the parameters provided, it calls the internal `_train_mlm` or `_train_nli` methods. This method also supports copying weights from a pre-initialized model head.

**Arguments:**
- `model`: The model to be trained.
- `tokenizer`: The tokenizer used for preprocessing the text.
- `dataset`: The dataset for training. It can be a CSV file path, a `DatasetDict`, or a dictionary.
- `label2_id` (dict, optional): Mapping of label names to IDs for NLI tasks.
- `nli_head` (bool, optional): Set to `True` to train an NLI head.
- `mlm_head` (bool, optional): Set to `True` to train an MLM head.
- `model_to_copy_weights_from` (optional): A model from which to copy head weights.
- `num_samples_train` (int, optional): Number of training samples to use.
- `num_samples_validation` (int, optional): Number of validation samples to use.
- `num_percentage_validation` (float, optional): Fraction of training data to reserve for validation if no separate validation set is provided.
- `shuffle_dataset` (bool, optional): Whether to shuffle the dataset before training.
- `val_dataset` (optional): A separate validation dataset (CSV path, `DatasetDict`, or dictionary).
- `batch_size` (int, optional): Batch size for training (default is 16).
- `num_epochs` (int, optional): Number of training epochs (default is 10).
- `learning_rate` (float, optional): Learning rate (default is 2e-5).
- `training_model_max_tokens` (int, optional): Maximum token length for training. If not specified, it is computed from the dataset.
- `freeze_base` (bool, optional): Whether to freeze the base model layers during training.
- `copy_weights` (bool, optional): Whether to initialize the head by copying weights from another model.
- `save_checkpoint` (bool, optional): Whether to save model checkpoints during training.
- `checkpoint_path` (str, optional): Directory path to save checkpoints.
- `epochs_to_save` (list[int], optional): List of epoch numbers at which checkpoints will be saved.
- `head_to_save` (str, optional): Specifies which head to save in the checkpoint.
- `fix_model_embedding_layer` (bool, optional): Whether to fix embedding layer in NLI models.

**Returns:**
- The trained model.

---

### fix_model_embedding_layer
Adjusts the token type embedding layer for NLI models. If the model's `type_vocab_size` is less than 2, this method updates the configuration and creates a new embedding layer with two token types by duplicating the original embedding.

**Arguments:**
- `nli_model`: The NLI model whose embedding layer needs to be modified.

**Returns:**
- None (modifies the model in place).

---

### init_head
Initializes an uninitialized model head by copying specified layers from an initialized head. This is useful for transferring learned weights between models that share the same architecture.

**Arguments:**
- `uninitialized_head`: The model head to be initialized.
- `initialized_head`: The model head from which to copy weights.
- `layers_to_init` (list[str]): A list of layer names (or nested attribute paths) to be copied.

**Returns:**
- None.

---

### get_non_base_layers
Retrieves the layers that are not part of the base model. This helps identify the head layers that can be trained or updated separately from the base model parameters.

**Arguments:**
- `model`: The model from which to extract the non-base (head) layers.

**Returns:**
- A list of layer names corresponding to the head layers.

---

### attach_head_to_model
Attaches one model head to another by setting a specified attribute. This function facilitates combining or updating model heads.

**Arguments:**
- `head1`: The destination head that will receive the attribute.
- `head2`: The source head from which the attribute is taken.
- `model_identifier` (str): The attribute name that identifies the model head.

**Returns:**
- None.

---

## Formats and Parameter Constraints

### Dataset Format
- **For MLM:**  
  The dataset should be a `DatasetDict` with keys `'train'` and optionally `'validation'`. The data should include a key `'text'` containing the text strings.
- **For NLI:**  
  The dataset should be a `DatasetDict` with keys `'train'` and optionally `'validation'`. The training data must include the keys `'premise'`, `'hypothesis'`, and `'label'`.

---

### CSV Format
- **For MLM:**  
  Provide the path to a CSV file that where the data lies under one column: `'text'`
- **For NLI:**  
  Provide the path to a CSV file that where the data lies under three columns: `'premise'`, `'hypothesis'`, and `'label'`.

---

### Dictionary Format
- **For MLM:**  
  A dictionary with a `'train'` key (and optionally a `'validation'` key) where the value is a list of text strings.
- **For NLI:**  
  A dictionary where the `'train'` (and optionally `'validation'`) key contains sub-keys `'premise'`, `'hypothesis'`, and `'label'` mapping to lists of strings and integers respectively.

---

### Parameter Constraints
- **num_samples_train** and **num_samples_validation:**  
  Specify the number of samples to use for training and validation. If `num_samples_validation` is provided, either `num_percentage_validation` or `val_dataset` must be specified. Be aware that the sampling occurs after the split of the dataset.
- **num_percentage_validation:**  
  A float between 0 and 1 that indicates the fraction of training data to be used for validation if no separate validation set is provided.
- **val_dataset:**  
  Use this parameter to provide a separate validation dataset when needed.
- **batch_size, num_epochs, and learning_rate:**  
  These parameters control the training process and should be set according to your system capabilities and training requirements.

## Usage Examples
This section can include various usage examples demonstrating different functionalities and configurations of the ModelTrainer class.
### Example of an MLM head loaded from an MLM model, trained with dataset (saving checkpoint).
```python
base_model_name = "distilbert/distilbert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
mlm_model = AutoModelForMaskedLM.from_pretrained(base_model_name)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", ignore_verifications=True)
trainer = ModelTrainer()

trainer.train_head(model=mlm_model, tokenizer=tokenizer,dataset=dataset,val_dataset=dataset,
                   num_samples_train=10, num_samples_validation=2, mlm_head=True, save_checkpoint=True)
```
### Example of an MLM head loaded from an MLM model, trained with csv.
```python
base_model_name = "distilbert/distilbert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
mlm_model = AutoModelForMaskedLM.from_pretrained(base_model_name)
dataset = "../AMI/datasets/ami_hostility_towards_men.csv"
trainer = ModelTrainer()

trainer.train_head(model=mlm_model, tokenizer=tokenizer,dataset=dataset, num_samples_train=50, num_samples_validation=10, num_percentage_validation=0.2, mlm_head=True)
```
### Example of an MLM head loaded from an MLM model, trained with a dictionary of lists.
```python
base_model_name = "distilbert/distilbert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
mlm_model = AutoModelForMaskedLM.from_pretrained(base_model_name)
dataset_file_path = "../AMI/datasets/ami_hostility_towards_men.csv"
df = pd.read_csv(dataset_file_path, header=None)
text_list = df[0].tolist()

dataset = {
    "train": text_list[:80],
    "validation": text_list[80:]

}

trainer = ModelTrainer()
trainer.train_head(model=mlm_model, tokenizer=tokenizer,dataset=dataset, mlm_head=True)
```
### Example of an MLM head loaded from an NLI model with copied weights and biases from a trained MLM head, trained on dataset, but saving the NLI head instead of the trained MLM head.
```python
trainer = ModelTrainer()
dataset = "../AMI/datasets/ami_hostility_towards_men.csv"
nli_model_path="typeform/distilbert-base-uncased-mnli" 
mlm_initialized_head = AutoModelForMaskedLM.from_pretrained("distilbert/distilbert-base-uncased")

nli_head_to_save=AutoModelForSequenceClassification.from_pretrained(nli_model_path)

distilbert_model_mlm = AutoModelForMaskedLM.from_pretrained(nli_model_path)
distilbert_model_tokenizer = AutoTokenizer.from_pretrained(nli_model_path)

trainer.attach_head_to_model(distilbert_model_mlm, nli_head_to_save, "distilbert") # distilbert_model_mlm base model is referenced to nli_head_to_save base model.
trainer.train_head(model=distilbert_model_mlm, tokenizer=distilbert_model_tokenizer, model_to_copy_weights_from=mlm_initialized_head, copy_weights=True, mlm_head=True, 
                   dataset=dataset, val_dataset=dataset, save_checkpoint=True, checkpoint_path="./phq_trained_models/nli_then_mlm_domain_adaptation_depressive_saved_nli/", head_to_save=nli_head_to_save)
```
### Example of an MLM head loaded from an NLI model with weights and biases initialized randomly, trained with dataset.
```python
base_model_name = "typeform/distilbert-base-uncased-mnli" 
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
mlm_model = AutoModelForMaskedLM.from_pretrained(base_model_name)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
trainer = ModelTrainer()
trainer.train_head(model=mlm_model, tokenizer=tokenizer, dataset=dataset, mlm_head=True, num_samples_train=100, num_samples_validation=20, num_percentage_validation=0.2)
```
### Example of an MLM head loaded from an NLI model with weights and biases initialized randomly, trained with csv.
```python
base_model_name = "typeform/distilbert-base-uncased-mnli" 
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
mlm_model = AutoModelForMaskedLM.from_pretrained(base_model_name)
dataset = "../AMI/datasets/ami_hostility_towards_men.csv"
trainer = ModelTrainer()

trainer.train_head(model=mlm_model, tokenizer=tokenizer,dataset=dataset,num_samples_train=100, mlm_head=True, batch_size=4)
```
### Example of an MLM head loaded from an NLI model with copied weights and biases from a trained MLM head, trained on csv.
```python
base_model_name = "typeform/distilbert-base-uncased-mnli" 
mlm_initialized_head = AutoModelForMaskedLM.from_pretrained("distilbert/distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
mlm_model = AutoModelForMaskedLM.from_pretrained(base_model_name)
dataset = "../AMI/datasets/ami_hostility_towards_men.csv"
trainer = ModelTrainer()

trainer.train_head(model=mlm_model, tokenizer=tokenizer, dataset=dataset, mlm_head=True, num_samples_train=90, num_samples_validation=10, model_to_copy_weights_from=mlm_initialized_head, copy_weights=True, batch_size=4)
```
### Example of an MLM head loaded from an NLI model with copied weights and biases from a trained MLM head, trained on dataset.
```python
base_model_name = "typeform/distilbert-base-uncased-mnli" 
mlm_initialized_head = AutoModelForMaskedLM.from_pretrained("distilbert/distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
mlm_model = AutoModelForMaskedLM.from_pretrained(base_model_name)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
trainer = ModelTrainer()

trainer.train_head(model=mlm_model, tokenizer=tokenizer,dataset=dataset, mlm_head=True, model_to_copy_weights_from=mlm_initialized_head, copy_weights=True,num_samples_train=100, num_samples_validation=20, num_percentage_validation=0.2)
```
### Example of an NLI head loaded from an NLI model, trained on csv.
```python
base_model_name = "typeform/distilbert-base-uncased-mnli"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
dataset = "./test_nli.csv"

trainer = ModelTrainer()
trainer.train_head(nli_model, tokenizer, nli_head=True, dataset=dataset, num_samples_train=100, num_samples_validation=10, num_percentage_validation=0.2, label2_id={"entailment":0, "neutral":1, "contradiction":2})
```
### Example of an NLI head loaded from an NLI model, trained on dataset.
```python
base_model_name = "typeform/distilbert-base-uncased-mnli"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
dataset = load_dataset('multi_nli')

dataset = DatasetDict({
    'train': dataset['train'],
    'validation': dataset['validation_matched']
})

trainer = ModelTrainer()
trainer.train_head(nli_model, tokenizer, nli_head=True, dataset=dataset, num_samples_train=2000,label2_id={"entailment":0, "neutral":1, "contradiction":2})
```
### Example of an NLI head loaded from an NLI model, trained on a dictionary of lists.
```python
base_model_name = "typeform/distilbert-base-uncased-mnli"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
dataset_file_path = "./test_nli.csv"

df = pd.read_csv(dataset_file_path)
premise_list = df['premise'].to_list()
hypothesis_list = df['hypothesis'].to_list()
label_list = df['label'].to_list()


dataset = {
    "train": {"premise":premise_list[:400],"hypothesis":hypothesis_list[:400],"label":label_list[:400]},
    "validation": {"premise":premise_list[400:450],"hypothesis":hypothesis_list[400:450],"label":label_list[400:450]}

}

trainer = ModelTrainer()
trainer.train_head(nli_model, tokenizer, nli_head=True, dataset=dataset, num_samples_train=100, num_samples_validation=10, num_percentage_validation=0.2, label2_id={"entailment":0, "neutral":1, "contradiction":2})
```
### Example of an NLI head loaded from an MLM model with weights and biases initialized randomly.
```python
config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", num_labels = 3)
base_model_name = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(base_model_name , config=config)
trainer = ModelTrainer()
dataset = load_dataset('multi_nli')

dataset = DatasetDict({
    'train': dataset['train'],
    'validation': dataset['validation_matched']
})

trainer.train_head(nli_model, tokenizer,dataset=dataset, nli_head=True, num_samples_train=100, num_samples_validation=20, num_percentage_validation=0.2, label2_id={"entailment":0, "neutral":1, "contradiction":2})
```
### Example of an NLI head loaded from an MLM model with copied weights and biases from a trained NLI head.
```python
config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", num_labels = 3)
base_model_name = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(base_model_name , config=config)
nli_initialized_head = AutoModelForSequenceClassification.from_pretrained("typeform/distilbert-base-uncased-mnli")
trainer = ModelTrainer()
dataset = load_dataset('multi_nli')

dataset = DatasetDict({
    'train': dataset['train'],
    'validation': dataset['validation_matched']
})

trainer.train_head(nli_model, tokenizer,dataset=dataset, nli_head=True, model_to_copy_weights_from=nli_initialized_head, num_samples_train=100, num_samples_validation=10, copy_weights=True, num_percentage_validation=0.2, label2_id={"entailment":0, "neutral":1, "contradiction":2})
```

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
