# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:52:22 2015

@author: Sanghita
"""

#!/bin/sh
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import operator
import unicodedata
from collections import namedtuple
import sys
import xlrd
import xlwt
import math
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from time import time
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer 
from collections import Counter

def probability_dist(a,search):
    tfidf_vectorizer = TfidfVectorizer() 
    rank=-1
    #print a
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(a)      
        ldict=tfidf_vectorizer.vocabulary_  
        print ldict        
        for keys in ldict.keys():            
            if keys.find(search)>=0 or search.find(keys)>=0:
                if rank==-1:
                    rank=ldict[keys]
                elif rank>ldict[keys]:
                    rank=ldict[keys]                    
        return rank                    
    except ValueError:
        print "Value Error"                
        return rank

          
def readXL():
	reload(sys)  
	sys.setdefaultencoding('utf8')

	stop = stopwords.words('english')

	xl_user = xlrd.open_workbook("Data.xlsX")

	sheet_user = xl_user.sheet_names()

	sheetu = xl_user.sheet_by_name(sheet_user[0])	

	
	print "USER :: ROWS :: " +str(sheetu.nrows - 1)
	print "USER :: COLS :: " +str(sheetu.ncols - 1)

	taskTable={}
 
	# Extracting features min and max values for each feature in set for users 
	for j in xrange(0,sheetu.ncols):
		value=str(sheetu.col(j)[1].value)      
		value=value.split('_')[0]
		taskTable[value]=[]		
		if value=="Routine":  
			for i in xrange(2,sheetu.nrows):
				vList=str(sheetu.col(j)[i].value).split('-')
				pVal=0
				if len(vList)==3:
					try:        
						pVal=(int(math.pow(10,5))*int(vList[0]))+(int(math.pow(10,3))*int(vList[1]))+(int(vList[2]))
					except ValueError:    
						pVal=0
				else:      
					pVal=0        
				#print pVal          
				taskTable[value].append(pVal)
		elif value=="Task":       
			for i in xrange(2,sheetu.nrows):     
				try:        
					pVal=int(sheetu.col(j)[i].value)
				except ValueError:        
					pVal=0        
				#print pVal     
				taskTable[value].append(pVal)        
		elif value=="Category":       
			for i in xrange(2,sheetu.nrows):     
				try:        
					pVal=int(sheetu.col(j)[i].value)
				except ValueError:        
					pVal=0        
				#print pVal     
				taskTable[value].append(pVal)            
		elif value=="Description":           
			for i in xrange(2,sheetu.nrows):     
				word = sheetu.col(j)[i].value
				wordL = word.split()           
				for w in wordL:        
					w=w.lower()        
					x=re.sub('[0-9]+',"",w)
					if x==w and w not in stop and w.isdigit() is False:
						print w         
    				index=0
    				sindex=-1
    				eindex=-1       
				for w in wordL:        
					x=re.sub('[^0-9]+',"",w)
     					if x!='' and index<len(wordL)-1:
						if wordL[index+1].lower()=='am' or wordL[index+1].lower()=='pm':
							wordL[index]=wordL[index]+wordL[index+1]
							word=word.replace(wordL[index+1],"")				
       
					index += 1	
    				index=0    
				wordL = word.split()                   
				#print word    
				for w in wordL:    
					x = re.sub('[^0-9]+','',w)       
					if x!='':             
						if sindex==-1:
    							sindex=index
						elif eindex==-1 and index-sindex==2:           
    							eindex=index          
						else:
           						sindex=-1
					index +=1				
				s = re.sub('[^0-9]+','',wordL[sindex])       
				e = re.sub('[^0-9]+','',wordL[eindex])            
				startTime=""
				endTime=""    
				if sindex!=-1 and eindex!=-1:    
					#print (29*'-')      
					#print word            
					startTime=str(s)
					endTime=str(e)
				elif sindex!=-1:        
					#print "start :: "+str(wordL[sindex-1])     
					if wordL[sindex-1]=="at" or wordL[sindex-1]=="from":    
						startTime=str(s)
						endTime='0'    
					#	print (29*'-')      
					#	print word            
					#	print "at "+str(s)
				print (29*'-')    
				print word
				tokens = nltk.word_tokenize(word.lower())    
				text = nltk.Text(tokens)
				tags = nltk.pos_tag(text)
				print (29*'+')           
				print tags    
				noun=[]
				for t in tags: 					
					if t[1]=='NN'or t[1]=='NNS':         
						noun.append(t[0])         
				#counts = Counter(tag for word,tag in tags)
				print str(startTime)+" :: "+str(endTime)      
				print noun
				print (29*'+')        

    
if __name__ == '__main__':
    readXL()
    
