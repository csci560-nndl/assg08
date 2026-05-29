---
title: 'Assignment 08: Neural Machine Translation and Attention'
author: 'CSci 560: Neural Networks and Deep Learning'
date: ''
---

# Description

Welcome to our next assignment over Text/Sequence deep learning systems.  In this assignment you will be turning your attention to Machine Translation using an encoder / decoder, as we discussed in our class.  You are going
to implement the translator using Keras LSTM recurrent layers.  But you will be adding neural attention
mechanisms by hand to your encoder / decoder, instead of using the Keras attention layers.

The model you will build here could be used to translate from one language to another, such as
translating from English to Hindi.  However, language translation requires massive datasets and
usually takes days of training on GPUs. To give you a place to experiment with these models without
using massive datasets, we will perform a simpler "date translation" task. The network will input
a date written in a variety of possible formats
(*e.g. "the 29th of August 1958", "03/30/1968", "24 JUNE 1987"*).
The network will translate them into standardized, machine readable dates
(*e.g. "1958-08-29", "1968-03-30", "1987-06-24"*).
We will have the network learn to output dates in the common machine-readable format YYYY-MM-DD.
 

**Instructions:**

- As with the previous assignment, you will need to create the function declarations asked for
  in `src/assg_tasks.py`.  Make sure you use
  [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/) and are generally
  following [Pep8 Python Style Guide](https://peps.python.org/pep-0008/) for your code.
- Cells with `### TESTED` comment contain unit tests that are run on your implementation.  You will
  need to uncomment the call to the unit tests, but otherwise need to stay as given in the original
  notebook.
- Likewise since you need to write your declaration of the functions asked for the tasks, don't forget
  to uncomment/add the appropriate `from assg_src include X` statements in both this notebook and
  in the `../src/test_assg_tasks.py`

# Objectives

**You will learn:**

- Learn more about how machine translation and the basic encoder / decoder architecture pattern
  work in practice.
- Implement by hand the neural attention mechanism described in our course materials, to add attention
  to "word" (date token) embeddings.
- Get some experience with a "text" dataset but with different normalization and tokenization requirements
  than the examples we have used in our course.

# Overview and Setup

# Assignment Tasks

# Assignment Submission

# Additional Information



