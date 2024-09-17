import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras_nlp
import keras
import tensorflow as tf
from keras import layers
from keras.layers import TextVectorization
from dataclasses import dataclass
import pandas as pd
import numpy as np
import glob
import re
from pprint import pprint

print(keras_nlp.encoders.)
# Unbatched input.
tokenizer = keras_nlp.models.GemmaTokenizer.from_preset("gemma_2b_en")
tokenizer("The quick brown fox jumped.")