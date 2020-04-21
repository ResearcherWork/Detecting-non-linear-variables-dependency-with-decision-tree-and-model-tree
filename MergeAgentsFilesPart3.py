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

import os
import glob
from csvsorter import csvsort
from more_itertools import unique_everseen

File="C:/PythonProjects/TurboFanTest/Test4/GetColumns.csv"

DF = pd.read_csv(File, delimiter=",")
ListColumns=DF.columns

for a in ListColumns:
	path = "C:/PythonProjects/AgentsTurboFan/Test/Test4/Agent"+a+""
	os.chdir(path)
	results = pd.DataFrame()
	counter = range(1, 101)
	for i in counter:
		j = str(i)
		namedf = pd.read_csv("C:/PythonProjects/AgentsTurboFan/Test/Test4/Agent"+a+"/Motor"+j+"Agent"+a+".csv", header=None, sep=",")
		print(namedf)
		results = pd.concat([results, namedf])

	results.to_csv('C:/PythonProjects/AgentsTurboFan/Test/Test4/Agent'+a+'.csv', index=None, header=None, sep=",")

	
	file_name = 'C:/PythonProjects/AgentsTurboFan/Test/Test4/Agent'+a+'.csv'
	file_name_output = 'C:/PythonProjects/AgentsTurboFan/Test/Test4/Agent'+a+'.csv'

	df = pd.read_csv(file_name, sep=",")

	df.drop_duplicates(subset=None, keep=False, inplace=True)

	df.to_csv(file_name_output, index=None)
