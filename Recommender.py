# import libraries
import pandas as pd
import numpy as np
import re
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel
import neattext.functions as nfx

