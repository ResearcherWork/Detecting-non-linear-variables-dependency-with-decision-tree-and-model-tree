import pandas as pd

import csv
import logging
from fastnumbers import isreal
  

from pade.misc.utility import display_message
from pade.core.agent import Agent
from pade.acl.aid import AID

from logging.handlers import RotatingFileHandler
import wekaexamples.helper as helper
import subprocess
import traceback
import weka.core.jvm as jvm
from weka.core.converters import load_any_file
from weka.classifiers import Classifier, Evaluation, PredictionOutput
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

class A(Agent):
    def __init__(self, aid): 
        super(A, self).__init__(aid=aid, debug=False)
        display_message(self.aid.localname, 'The Agents were started !')

if __name__ == '__main__':

    

    agents = list()

    Agent_s = A(AID(name='Agent_s'))
    Agent_s.ams = {'name': 'localhost', 'port': 8000}
    agents.append(Agent_s)
 

    jvm.start()

    File="C:/PythonProjects/TurboFanTest/Test4/GetColumns.csv"

    DF = pd.read_csv(File, delimiter=",")
    ListColumns=DF.columns
    G = nx.MultiDiGraph()
    
    labelsD1={}
    labels1 = []
    NewListColumns = []
    for i in range(len(ListColumns)):
        NewListColumns.append(ListColumns[i])

    print("NewListColumns :")
    print(NewListColumns)
    for l in (NewListColumns):

        data_file = 'C:/PythonProjects/AgentsTurboFan/Test/Test4/Agent'+l+'.csv'
        
        test = []
        test.append(l) 
        print("test :")
        print(test)

        print("\n--> loading:\n")
        print(data_file)
        dataA= load_any_file(data_file)
        dataA.class_is_last()

        DFC = pd.read_csv('C:/PythonProjects/AgentsTurboFan/Test/Test4/Agent'+l+'.csv', delimiter=",")
        for a in range(len(DFC)):
            classvar = DFC.iloc[a,len(DFC.columns)-1]
            classvarStr = str(classvar)

        print('classvarStr :', classvarStr)
        print('isreal(classvarStr) :', isreal(classvarStr))

        if isreal(classvarStr)==True:

            classifier = Classifier(classname="weka.classifiers.trees.M5P", options=["-U", "-M", "500.0"])
            print("\n--> building:")
            print(classifier.to_commandline())
            classifier.build_classifier(dataA)
            print("\n--> classifier:\n")
            print(classifier)
            print("\n--> graph:\n")
            print(classifier.graph)

            outputfile = helper.get_tmp_dir() + "/result.csv"
            output = PredictionOutput(classname='weka.classifiers.evaluation.output.prediction.CSV', options=["-distribution", "-suppress", "-file", outputfile])
            print("\n--> Output:\n")
            output.header = dataA
            output.print_all(classifier, dataA)
            helper.print_info("Predictions stored in:" + outputfile)
            print(output.buffer_content())
            Eval = Evaluation(dataA)
            Eval.test_model(classifier, dataA, output=output)
            print(Eval.summary())
            ListEval = []
            Corr = []
            Corrf = []
            ListEval = Eval.summary().split('Mean absolute error')
            print("ListEval :")
            print(ListEval)
            Corr = ListEval[0].split('\n')
            Corrf = Corr[1].split('Correlation coefficient                  ')
            print("Corrf :")
            print(Corrf[1])

            ListEvalRAE = []
            RA = []
            RAE = []
            ListEvalRAE = Eval.summary().split('Root relative squared error')
            print("ListEvalRAE :")
            print(ListEvalRAE)
            RA = ListEvalRAE[0].split('\n')
            print("RA :")
            print(RA)
            RAE = RA[4].split('Relative absolute error                ')
            #print(RAE)
            #RAE = RA[1].split('\n')
            print("RAE :")
            print(RAE[1])
            RAEfinal = RAE[1].split(' %')
            print("RAEfinal :")
            print(RAEfinal[0])

            ListEvalRSE = []
            RS = []
            RSE = []
            ListEvalRSE = Eval.summary().split('Total Number of Instances')
            print("ListEvalRSE :")
            print(ListEvalRSE)
            RS = ListEvalRSE[0].split('\n')
            print("RS :")
            print(RS)
            RSE = RS[5].split('Root relative squared error             ')
            #print(RAE)
            #RAE = RA[1].split('\n')
            print("RSE :")
            print(RSE[1])
            RSEfinal = RSE[1].split(' %')
            print("RSEfinal :")
            print(RSEfinal[0])


            tclassifier=[]
            Listclassifiers=[]
            t1f=[]
            t1final=[]
            t1=[]
            tclassifier.append(str(classifier))
            
            for k in range(len(tclassifier)):
                t1 = tclassifier[k].split(" ")
            

           
            alphabet = []
            for letter in range(65, 91):
                alphabet.append(chr(letter))
            for letter in range(97,123):
                alphabet.append(chr(letter))

            for i in range(len(t1)):
                for j in alphabet:

                    if t1[i].endswith("t1") and t1[i].startswith(j):
                        t1f.append(t1[i])
            print("t1f :")
            print(t1f)               
            t1fi=[]
            if t1f !=[]:
                t1fi = t1f[0].split("\n")
                print("t1fi :")
                print(t1fi)
                t1f[0]=t1fi[len(t1fi)-1]
                print("t1f :")
                print(t1f)
                t1final=list(set(t1f))
            
        
            DF = pd.read_csv('C:/PythonProjects/AgentsTurboFan/Test/Test4/Agent'+l+'.csv', delimiter=",")
            lc1 = []
            lc1 = list(DF.columns)
            ClassVariable = lc1[len(lc1)-1]
            print("ClassVariable :")
            print(ClassVariable)
            print("t1final :")
            print(t1final)

            q=1
            cv1=[]
            cv1 = ClassVariable.split('t0')
            G.add_node(q, color='blue')
            labelsD1[q] = cv1[0]
            print("cv1 :")
            print(cv1)

            ls11=[]
            for t in range(len(t1final)):
                ls11.append(t1final[t].split('pt1'))
            
            ls12=[]
            for b in range(len(ls11)):
                for c in range(1):
                    ls12.append(ls11[b][c].split('t1'))
            
            #print(ls11)
            print("ls12 :")
            print(ls12)
            
            
            

            lscv1=[]
            for a in range(len(t1final)):
                lscv1.append(t1final[a].split(cv1[0]))
            print("lscv1 :")
            print(lscv1)

            lsavt11=[]
            for c in range(len(t1final)):
                lsavt11.append(t1final[c].split('t1'))

            #print(lsavt11)

            lsavpt11=[]
            for d in range(len(t1final)):
                lsavpt11.append(t1final[d].split('pt1'))
            print("lsavt11 :")
            print(lsavt11)
            print("lsavpt11 :")
            print(lsavpt11)
            
            ls12norepeat = []
            for o in range(len(ls12)):
                ls12norepeat.append(ls12[o][0]) 
                
            ls12norepeat = list(dict.fromkeys(ls12norepeat))
            
            print("ls12norepeat :")    
            print(ls12norepeat)


            DFFile = pd.DataFrame()
            DFFile.loc[0,0] = cv1[0]
            DFFile.loc[0,1] = Corrf[1]
            DFFile.loc[0,2] = RAE[0]
            DFFile.loc[0,3] = RSE[0]
            with open('C:/PythonProjects/AgentsTurboFan/Test/Test4/QueryNodesTFTest4.cypher', 'a') as f:
                f.write("CREATE(vtest4:VariableTest4 { Name: '"+cv1[0]+"', Correlation: toFloat("+Corrf[1]+"), RelativeAbsoluteError: toFloat("+RAEfinal[0]+"), RootRelativeSquaredError: toFloat("+RSEfinal[0]+")});\n")
                f.close

            DepValue=''
            for e in range(len(ls12norepeat)):
                DepValue = ls12norepeat[e]
                for g in range(len(t1final)):
                    if t1final[g].startswith(DepValue+'pt1'):
                        with open('C:/PythonProjects/AgentsTurboFan/Test/Test4/QueryArcsTFTest4.cypher', 'a') as f:
                            f.write("MATCH (vtest41:VariableTest4),(vtest42:VariableTest4) \n WHERE vtest41.Name = '"+cv1[0]+"' AND vtest42.Name = '"+DepValue+"' \n CREATE (vtest41)-[r:pt1]->(vtest42) \n RETURN type(r); \n")
                            f.close
                        
                        with open('C:/PythonProjects/AgentsTurboFan/Test/Test4/CorrelationScoreInfoVar.csv', 'a') as f:
                            f.write(cv1[0]+','+Corrf[1]+','+DepValue+'\n')
                            f.close
                        with open('C:/PythonProjects/AgentsTurboFan/Test/Test4/RelativeAbsoluteErrorScoreInfoVar.csv', 'a') as f:
                            f.write(cv1[0]+','+RAEfinal[0]+','+DepValue+'\n')
                            f.close
                        with open('C:/PythonProjects/AgentsTurboFan/Test/Test4/RootRelativeSquaredErrorScoreInfoVar.csv', 'a') as f:
                            f.write(cv1[0]+','+RSEfinal[0]+','+DepValue+'\n')
                            f.close

                    if t1final[g].startswith(DepValue+'t1') and t1final[g].endswith('pt1')==False:
                        with open('C:/PythonProjects/AgentsTurboFan/Test/Test4/QueryArcsTFTest4.cypher', 'a') as f:
                            f.write("MATCH (vtest41:VariableTest4),(vtest42:VariableTest4) \n WHERE vtest41.Name = '"+cv1[0]+"' AND vtest42.Name = '"+DepValue+"' \n CREATE (vtest41)-[r:t1]->(vtest42) \n RETURN type(r); \n")
                            f.close

                        with open('C:/PythonProjects/AgentsTurboFan/Test/Test4/CorrelationScoreInfoVar.csv', 'a') as f:
                            f.write(cv1[0]+','+Corrf[1]+','+DepValue+'\n')
                            f.close
                        with open('C:/PythonProjects/AgentsTurboFan/Test/Test4/RelativeAbsoluteErrorScoreInfoVar.csv', 'a') as f:
                            f.write(cv1[0]+','+RAEfinal[0]+','+DepValue+'\n')
                            f.close
                        with open('C:/PythonProjects/AgentsTurboFan/Test/Test4/RootRelativeSquaredErrorScoreInfoVar.csv', 'a') as f:
                            f.write(cv1[0]+','+RSEfinal[0]+','+DepValue+'\n')
                            f.close

            with open('C:/PythonProjects/AgentsTurboFan/Test/Test4/CorrelationScore.csv', 'a') as f:
                f.write(cv1[0]+','+Corrf[1]+'\n')
                f.close

            with open('C:/PythonProjects/AgentsTurboFan/Test/Test4/RelativeAbsoluteErrorScore.csv', 'a') as f:
                f.write(cv1[0]+','+RAEfinal[0]+'\n')
                f.close

            with open('C:/PythonProjects/AgentsTurboFan/Test/Test4/RootRelativeSquaredErrorScore.csv', 'a') as f:
                f.write(cv1[0]+','+RSEfinal[0]+'\n')
                f.close


            graph_file = "C:/PythonProjects/graphAgentTest4"+l+".dot"
            png_file = "C:/PythonProjects/graphAgentTest4"+l+".png"
            print("\n--> writing graph file:\n")
            with open(graph_file, "w") as text_file:
                text_file.write(classifier.graph)
            params = ["dot.exe", "-Tpng", "-o" + png_file, graph_file]
            print("\n--> calling dot.exe\n")
            print(params)
            try:
                completed = subprocess.run(params)
                if completed.returncode == 0:
                    print("--> successful!")
                    image = Image.open(png_file)
                    image.show()
                else:
                    print("--> exit code:", completed.returncode)
                    print("--> stdout")
                    print(completed.stdout)
                    print("--> stderr")
                    print(completed.returncode)
            except:
                print(traceback.format_exc())

        print('classvarStr :', classvarStr)
        print('isreal(classvarStr) :', isreal(classvarStr))

        if isreal(classvarStr)==False:

            classifier = Classifier(classname="weka.classifiers.trees.J48")
            print("\n--> building:")
            print(classifier.to_commandline())
            classifier.build_classifier(dataA)
            print("\n--> classifier:\n")
            print(classifier)
            print("\n--> graph:\n")
            print(classifier.graph)

            outputfile = helper.get_tmp_dir() + "/result.csv"
            output = PredictionOutput(classname='weka.classifiers.evaluation.output.prediction.CSV', options=["-distribution", "-suppress", "-file", outputfile])
            print("\n--> Output:\n")
            output.header = dataA
            output.print_all(classifier, dataA)
            helper.print_info("Predictions stored in:" + outputfile)
            print(output.buffer_content())
            Eval = Evaluation(dataA)
            Eval.test_model(classifier, dataA, output=output)
            print(Eval.summary())
            ListEval = []
            Corr = []
            Corrf = []
            ListEval = Eval.summary().split('Mean absolute error')
            print("ListEval :")
            print(ListEval)
            Corr = ListEval[0].split('\n')
            Corrf = Corr[1].split('Correlation coefficient                  ')
            print("Corrf :")
            print(Corrf[1])

            ListEvalRAE = []
            RA = []
            RAE = []
            ListEvalRAE = Eval.summary().split('Root relative squared error')
            print("ListEvalRAE :")
            print(ListEvalRAE)
            RA = ListEvalRAE[0].split('                 Relative absolute error % ')
            print(RA)
            RAE = RA[1].split('  %\n')
            print("RAE :")
            print(RAE[0])

            ListEvalRSE = []
            RS = []
            RSE = []
            ListEvalRSE = Eval.summary().split('Total Number of Instances')
            print("ListEvalRSE :")
            print(ListEvalRSE)
            RS = ListEvalRSE[0].split('             Root relative squared error % ')
            print(RS)
            RSE = RS[1].split('  %\n')
            print("RSE :")
            print(RSE[0])

            tclassifier=[]
            Listclassifiers=[]
            t1f=[]
            t1final=[]
            t1=[]
            tclassifier.append(str(classifier))
            
            for k in range(len(tclassifier)):
                t1 = tclassifier[k].split(" ")
            

           
            alphabet = []
            for letter in range(65, 91):
                alphabet.append(chr(letter))
            for letter in range(97,123):
                alphabet.append(chr(letter))

            for i in range(len(t1)):
                for j in alphabet:

                    if t1[i].endswith("t1") and t1[i].startswith(j):
                        t1f.append(t1[i])
            print("t1f :")
            print(t1f)               
            t1fi=[]
            if t1f !=[]:
                t1fi = t1f[0].split("\n")
                print("t1fi :")
                print(t1fi)
                t1f[0]=t1fi[len(t1fi)-1]
                print("t1f :")
                print(t1f)
                t1final=list(set(t1f))
            
        
            DF = pd.read_csv('C:/PythonProjects/AgentsTurboFan/Test/Test4/Agent'+l+'.csv', delimiter=",")
            lc1 = []
            lc1 = list(DF.columns)
            ClassVariable = lc1[len(lc1)-1]
            print("ClassVariable :")
            print(ClassVariable)
            print("t1final :")
            print(t1final)

            q=1
            cv1=[]
            cv1 = ClassVariable.split('t0')
            G.add_node(q, color='blue')
            labelsD1[q] = cv1[0]
            print("cv1 :")
            print(cv1)

            ls11=[]
            for t in range(len(t1final)):
                ls11.append(t1final[t].split('pt1'))
            
            ls12=[]
            for b in range(len(ls11)):
                for c in range(1):
                    ls12.append(ls11[b][c].split('t1'))
            
            #print(ls11)
            print("ls12 :")
            print(ls12)
            
            
            

            lscv1=[]
            for a in range(len(t1final)):
                lscv1.append(t1final[a].split(cv1[0]))
            print("lscv1 :")
            print(lscv1)

            lsavt11=[]
            for c in range(len(t1final)):
                lsavt11.append(t1final[c].split('t1'))

            #print(lsavt11)

            lsavpt11=[]
            for d in range(len(t1final)):
                lsavpt11.append(t1final[d].split('pt1'))
            print("lsavt11 :")
            print(lsavt11)
            print("lsavpt11 :")
            print(lsavpt11)
            
            ls12norepeat = []
            for o in range(len(ls12)):
                ls12norepeat.append(ls12[o][0]) 
                
            ls12norepeat = list(dict.fromkeys(ls12norepeat))
            
            print("ls12norepeat :")    
            print(ls12norepeat)


            DFFile = pd.DataFrame()
            DFFile.loc[0,0] = cv1[0]
            DFFile.loc[0,1] = Corrf[1]
            DFFile.loc[0,2] = RAE[0]
            DFFile.loc[0,3] = RSE[0]
            with open('C:/PythonProjects/AgentsTurboFan/Test/Test4/QueryNodesTFTest4.cypher', 'a') as f:
                f.write("CREATE(vtest4:VariableTest4 { Name: '"+cv1[0]+"', Correlation: toFloat("+Corrf[1]+"), RelativeAbsoluteError: toFloat("+RAE[0]+"), RootRelativeSquaredError: toFloat("+RSE[0]+")});\n")
                f.close

            DepValue=""
            for e in range(len(ls12norepeat)):
                DepValue = ls12norepeat[e]
                for g in range(len(t1final)):
                    if t1final[g].startswith(DepValue) and t1final[g].endswith('pt1'):
                        with open('C:/PythonProjects/AgentsTurboFan/Test/Test4/QueryArcsTFTest4.cypher', 'a') as f:
                            f.write("MATCH (vtest41:VariableTest4),(vtest42:VariableTest4) \n WHERE vtest41.Name = '"+cv1[0]+"' AND vtest42.Name = '"+DepValue+"' \n CREATE (vtest41)-[r:pt1]->(vtest42) \n RETURN type(r); \n")
                            f.close
                    if (t1final[g].startswith(DepValue)) and (t1final[g].endswith('pt1') == False):
                        with open('C:/PythonProjects/AgentsTurboFan/Test/Test4/QueryArcsTFTest4.cypher', 'a') as f:
                            f.write("MATCH (vtest41:VariableTest4),(vtest42:VariableTest4) \n WHERE vtest41.Name = '"+cv1[0]+"' AND vtest42.Name = '"+DepValue+"' \n CREATE (vtest41)-[r:t1]->(vtest42) \n RETURN type(r); \n")
                            f.close

            

            graph_file = "C:/PythonProjects/graphAgentTest4"+l+".dot"
            png_file = "C:/PythonProjects/graphAgentTest4"+l+".png"
            print("\n--> writing graph file:\n")
            with open(graph_file, "w") as text_file:
                text_file.write(classifier.graph)
            params = ["dot.exe", "-Tpng", "-o" + png_file, graph_file]
            print("\n--> calling dot.exe\n")
            print(params)
            try:
                completed = subprocess.run(params)
                if completed.returncode == 0:
                    print("--> successful!")
                    image = Image.open(png_file)
                    image.show()
                else:
                    print("--> exit code:", completed.returncode)
                    print("--> stdout")
                    print(completed.stdout)
                    print("--> stderr")
                    print(completed.returncode)
            except:
                print(traceback.format_exc())