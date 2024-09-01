# Indicators of Resilience (IoR)

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

# The NLP Tasks Supported by the "qlatent" Package

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

**IMPORTANT NOTE**: There are 2 types of MNLI questions in this package: QMNLI and _QMNLI.

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

Given 2 sequences of characters, NSP is an NLP task that checks how good the 2nd sequence is as a follow-up for the 1st sequence.<br>
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

This class is a utility for training language models using the Hugging Face Transformers library. It supports training with both Masked Language Modeling (MLM) and Natural Language Inference (NLI) heads.

## Table of Contents

- [Installation](#installation)
- [Public Methods](#public-methods)
  - [train_head](#train-head)
  - [attach_head_to_model](#attach-head-to-model)
  - [get_non_base_layers](#get-non-base-layers)
  - [init_head](#init-head)
- [Usage](#usage)
  - [Initialization](#initialization)
  - [Training an MLM Model](#training-an-mlm-model)
  - [Training an NLI Model](#training-an-nli-model)
  - [Copying Weights](#copying-weights)
- [Callbacks](#callbacks)
- [Formats and parameter constraints](#constraints)
  - [Dataset Format](#dataset-format)
  - [CSV Format](#csv-format)
  - [Parameter Constraints](#parameter-constraints)
- [Usage Examples](#usage-examples)
  - [Example of an MLM head loaded from an MLM model trained with dataset](#example-of-an-mlm-head-loaded-from-an-mlm-model-trained-with-dataset)
  - [Example of an MLM head loaded from an MLM model trained with csv](#example-of-an-mlm-head-loaded-from-an-mlm-model-trained-with-csv)
  - [Example of an MLM head loaded from an NLI model with copied weights and biases from a trained MLM head, trained on dataset, but saving an MNLI head instead of the trained MLM head](#example-of-an-mlm-head-loaded-from-an-nli-model-with-copied-weights-and-biases-from-a-trained-mlm-head-trained-on-dataset-but-saving-an-mnli-head-instead-of-the-trained-mlm-head)
  - [Example of an MLM head loaded from an NLI model with weights and biases initialized randomly, trained with dataset](#example-of-an-mlm-head-loaded-from-an-nli-model-with-weights-and-biases-initialized-randomly-trained-with-dataset)
  - [Example of an MLM head loaded from an NLI model with copied weights and biases from a trained MLM head, trained on dataset](#example-of-an-mlm-head-loaded-from-an-nli-model-with-copied-weights-and-biases-from-a-trained-mlm-head-trained-on-dataset)
  - [Example of an MNLI head loaded from an MNLI model trained with csv](#example-of-an-mnli-head-loaded-from-an-mnli-model-trained-with-csv)
  - [Example of an NLI loaded from an NLI model, trained on dataset](#example-of-an-nli-loaded-from-an-nli-model-trained-on-dataset)
  - [Example of an NLI head loaded from an MLM model with weights and biases initialized randomly](#example-of-an-nli-head-loaded-from-an-mlm-model-with-weights-and-biases-initialized-randomly)
  - [Example of an NLI head loaded from MLM model with copied weights and biases from a trained NLI head](#example-of-an-nli-head-loaded-from-mlm-model-with-copied-weights-and-biases-from-a-trained-nli-head)


## Installation

To use this code, you need to install the required packages. You can do this by running:

```bash
pip install transformers datasets torch numpy
```

## Public Methods

### train_head
Trains the specified head (NLI or MLM) of the model.

**Arguments:**
- `model`: The model to be trained.
- `tokenizer`: The tokenizer to be used for training.
- `dataset`: The dataset or path to the dataset for training.
- `nli_head` (bool): Whether to train the NLI head.
- `mlm_head` (bool): Whether to train the MLM head.
- `model_to_copy_weights_from`: Model from which to copy the weights for initialization (optional).
- `num_samples_train`: Number of training samples to use (optional).
- `num_samples_validation`: Number of validation samples to use (optional).
- `val_dataset`: Validation dataset (optional).
- `validate` (bool): Whether to use validation during training.
- `training_model_max_tokens` (int): Maximum number of tokens for the training model.
- `batch_size` (int): The batch size for training.
- `num_epochs` (int): Number of epochs for training.
- `learning_rate` (float): Learning rate for training.
- `freeze_base` (bool): Whether to freeze the base model parameters.
- `copy_weights` (bool): Whether to copy weights from another model.
- `checkpoint_path`: Path to save checkpoints.
- `head_to_save`: The model head to save during the checkpoint.

**Returns:**
- None.

### attach_head_to_model
Attaches a model head to a specified model.

**Arguments:**
- `head1`: The head to attach to.
- `head2`: The head to attach from.
- `model_identifier` (str): Identifier for the model.

**Returns:**
- None.

### get_non_base_layers
Gets the non-base layers of the model.

**Arguments:**
- `model`: The model from which to get the non-base layers.

**Returns:**
- List of non-base layers.

### init_head
Initializes the head of the model by copying specified layers from another model.

**Arguments:**
- `uninitialized_head`: The head to initialize.
- `initialized_head`: The head from which to copy the layers.
- `layers_to_init` (list[str]): List of layers to initialize.

**Returns:**
- None.

## Usage

### Initialization

First, initialize the `ModelTrainer` class:

```python
from qmnli.utils import ModelTrainer

trainer = ModelTrainer()
```

### Training an MLM Model

To train a model with an MLM head:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)

trainer.train_head(
    model=model,
    tokenizer=tokenizer,
    dataset='path/to/dataset.csv',
    mlm_head=True,
    num_samples_train=10000,
    num_samples_validation=2000,
    validate=True,
    training_model_max_tokens=128,
    batch_size=16,
    num_epochs=5,
    learning_rate=5e-5,
    freeze_base=False,
    copy_weights=False,
    checkpoint_path="./mlm_checkpoints"
)
```

### Training an NLI Model

To train a model with an NLI head:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

trainer.train_head(
    model=model,
    tokenizer=tokenizer,
    dataset='path/to/nli_dataset.csv',
    nli_head=True,
    num_samples_train=10000,
    num_samples_validation=2000,
    validate=True,
    training_model_max_tokens=128,
    batch_size=16,
    num_epochs=5,
    learning_rate=5e-5,
    freeze_base=False,
    copy_weights=False,
    checkpoint_path="./nli_checkpoints",
    head_to_save=None
)
```

### Copying Weights

To copy weights from an initialized model head to another model head:
- Choose a model with the desired initialized head with the same architecture.
- Set copy_weights to True and provide the intialized_model you want to copy the parameters from, model_to_copy_weights_from = intialized_model


### Callbacks

This utility includes a callback class `SaveCheckpointByEpochCallback` to save model checkpoints at the end of each epoch:

```python
from transformers import TrainerCallback

class SaveCheckpointByEpochCallback(TrainerCallback):
    def __init__(self, output_dir: str, tokenizer):
        self.output_dir = output_dir
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = state.epoch
        checkpoint_dir = f"{self.output_dir}/checkpoint-epoch-{int(epoch)}"
        model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
```

## Formats and parameter constraints

#### Dataset Format

- **For MLM:**
  - The dataset object should have `'train'` and `'validation'` keys. To access the sentences within these datasets, ensure there is a key `'text'` that contains sentences as a list value.

- **For NLI:**
  - The dataset object should have `'train'` and `'validation_matched'` keys. To access the sentences within these datasets, ensure there are keys `'premise'`, `'hypothesis'`, and `'label'`.

#### CSV Format

- **For MLM:**
  - The CSV file should contain only sentences in the first column, without any headers.

- **For NLI:**
  - The CSV file should consist of three columns: `'premise'`, `'hypothesis'`, and `'label'`. Make sure the label ids are integers and aligned with the ids the original model is trained on.

#### Parameter Constraints
- **`num_samples_train` and `num_samples_validation`:**
  - Using these parameters, you can specify the amount of data taken from each dataset, if unspecified it takes the whole dataset.
  - if `num_samples_validation` is provided, then `num_samples_train` must be provided as well.

- **val_dataset:**
  - In case you want a seperate dataset for validation, provide the path to the validation dataset (`val_dataset`) if validation is required (`validate=True`).

- **validate:**
  - Set `validate=True` if validation is wanted during training.

- **training_model_max_tokens:**
  - Set the maximum token length (`training_model_max_tokens`) for training samples.
- **batch_size:**
  - Specify the batch size (`batch_size`) for training. Adjust based on your hardware capabilities and training performance.

- **num_epochs:**
  - Set the number of training epochs (`num_epochs`).

- **learning_rate:**
  - Specify the learning rate (`learning_rate`) for training.

- **freeze_base:**
  - Set `freeze_base=True` to freeze the base model parameters during training if required.

- **copy_weights:**
  - Set `copy_weights=True` if you need to copy weights from another model (`model_to_copy_weights_from`). Ensure `model_to_copy_weights_from` is correctly specified and compatible with the current model architecture.

- **head_to_save:**
  - Set head_to_save to a specific head you want to save, otherwise it saves the head used for the training by default.

## Use Examples
This section can include various usage examples demonstrating different functionalities and configurations of the ModelTrainer class.

### Example of an MLM head loaded from an MLM model trained with dataset.
```python
base_model_name = "distilbert/distilbert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
mlm_model = AutoModelForMaskedLM.from_pretrained(base_model_name)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
trainer = ModelTrainer()

trainer.train_head(model=mlm_model, tokenizer=tokenizer,dataset=dataset, mlm_head=True, num_samples_train=10000, num_samples_validation=2000)
```
### Example of an MLM head loaded from an MLM model trained with csv.
```python
base_model_name = "distilbert/distilbert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
mlm_model = AutoModelForMaskedLM.from_pretrained(base_model_name)
dataset = "../AMI/datasets/ami_hostility_towards_men.csv"
trainer = ModelTrainer()

trainer.train_head(model=mlm_model, tokenizer=tokenizer,dataset=dataset, mlm_head=True,validate=False, batch_size=4)
```
### Example of an MLM head loaded from an NLI model with copied weights and biases from a trained MLM head, trained on dataset, but saving an MNLI head instead of the trained MLM head.
```python
trainer = ModelTrainer()
dataset = './datasets/depressive_dataset.csv'
p="./asi_trained_models/just_nli/checkpoint-24544/"
mlm_initialized_head = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")

nli_head_to_save=AutoModelForSequenceClassification.from_pretrained(p)

bert_model_mlm = AutoModelForMaskedLM.from_pretrained(p)
bert_model_tokenizer = AutoTokenizer.from_pretrained(p)

trainer.attach_head_to_model(bert_model_mlm, nli_head_to_save, "bert") # bert_model_mlm base model is referenced to nli_head_to_save base model.
trainer.train_head(model=bert_model_mlm, tokenizer=bert_model_tokenizer, model_to_copy_weights_from=mlm_initialized_head, copy_weights=True, mlm_head=True, dataset=dataset,val_dataset=dataset, checkpoint_path="./phq_trained_models/nli_then_mlm_domain_adaptation_depressive_saved_nli/", head_to_save=nli_head_to_save, batch_size=8, num_epochs=20)
```
### Example of an MLM head loaded from an NLI model with weights and biases initialized randomly, trained with dataset.
```python
base_model_name = "typeform/distilbert-base-uncased-mnli" 
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
mlm_model = AutoModelForMaskedLM.from_pretrained(base_model_name)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
trainer = ModelTrainer()
trainer.train_head(model=mlm_model, tokenizer=tokenizer,dataset=dataset, mlm_head=True, copy_weights=False,num_samples_train=10000, num_samples_validation=2000)
```
### Example of an MLM head loaded from an NLI model with copied weights and biases from a trained MLM head, trained on dataset.
```python
base_model_name = "typeform/distilbert-base-uncased-mnli" 
mlm_initialized_head = AutoModelForMaskedLM.from_pretrained("distilbert/distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
mlm_model = AutoModelForMaskedLM.from_pretrained(base_model_name)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
trainer = ModelTrainer()

trainer.train_head(model=mlm_model, tokenizer=tokenizer,dataset=dataset, mlm_head=True, model_to_copy_weights_from=mlm_initialized_head, copy_weights=True,num_samples_train=10000, num_samples_validation=2000)
```
### Example of an MNLI head loaded from an MNLI model trained with csv.
```python
base_model_name = "typeform/distilbert-base-uncased-mnli"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
dataset = "./jsonl_to_csv.csv"
trainer = ModelTrainer()
trainer.train_head(nli_model, tokenizer, nli_head=True, dataset=dataset, num_samples_train=4000, num_samples_validation=1000)
```
### Example of an NLI loaded from an NLI model, trained on dataset.
```python
base_model_name = "typeform/distilbert-base-uncased-mnli"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
dataset = load_dataset('multi_nli')
trainer = ModelTrainer()
trainer.train_head(nli_model, tokenizer, nli_head=True, dataset=dataset, num_samples_train=10000, num_samples_validation=2000)
```
### Example of an NLI head loaded from an MLM model with weights and biases initialized randomly.
```python
config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", num_labels = 3)
base_model_name = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(base_model_name , config=config)
trainer = ModelTrainer()
dataset = load_dataset('multi_nli')

trainer.train_head(nli_model, tokenizer,dataset=dataset, nli_head=True, num_samples_train=10000, num_samples_validation=2000)
```
### Example of an NLI head loaded from MLM model with copied weights and biases from a trained NLI head.
```python
config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", num_labels = 3)
base_model_name = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(base_model_name , config=config)
nli_initialized_head = AutoModelForSequenceClassification.from_pretrained("typeform/distilbert-base-uncased-mnli")
trainer = ModelTrainer()
dataset = load_dataset('multi_nli')
trainer.train_head(nli_model, tokenizer,dataset=dataset, nli_head=True, model_to_copy_weights_from=nli_initialized_head, num_samples_train=10000, num_samples_validation=2000, copy_weights=True)
```
