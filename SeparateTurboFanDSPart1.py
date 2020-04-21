import pandas as pd

import csv
import logging
  

from pade.misc.utility import display_message
from pade.core.agent import Agent
from pade.acl.aid import AID

from logging.handlers import RotatingFileHandler

import subprocess
import traceback
import weka.core.jvm as jvm
from weka.core.converters import load_any_file
from weka.classifiers import Classifier
import weka.plot.graph as plot_graph
from PIL import Image

import string

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph  
from networkx.drawing.nx_pydot import write_dot
from subprocess import check_call


File="C:/Users/Tesnim Touil/Downloads/CMAPSSData/test_FD004.txt"
DF = pd.read_csv(File, index_col = False, delim_whitespace=True)
DF.drop([DF.columns[1]], axis='columns', inplace=True)#DF.columns[4], DF.columns[5], DF.columns[9], DF.columns[20], DF.columns[22], DF.columns[23]], axis='columns', inplace=True)
DF = DF.loc[:,DF.apply(pd.Series.nunique) != 1]
l = 0
for j in range(1, DF.iloc[len(DF)-1,0]+1):
	for i in range(len(DF)):
		if DF.iloc[i,0]==j:
			k = i+1
	DF1 = DF.iloc[l:k, :]
	l = k
	s = str(j)
	DF1.to_csv('C:/PythonProjects/TurboFanTest/Test4/TurboFanTest4SMotor'+s+'.csv', index=False)
	


	



