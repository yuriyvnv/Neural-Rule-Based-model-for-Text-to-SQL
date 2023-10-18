#### GENERAL IMPORTS
import pandas as pd
import numpy as np 
import torch
import time

#### STANZA LIB (POS EXTRACTION AND OTHERS)
import stanza

#stanza.download('en') 
nlp = stanza.Pipeline(lang='en')
import regex as re 



#### FOR TEXT PREPROCESSING
#pip install word2number
from word2number import w2n 
#!pip install nltk
#!pip install textblob
#!pip install autocorrect



### SIMILARITY CALCULUS
#!pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer, util
SimilarityModel = SentenceTransformer("sentence-transformers/sentence-t5-xl")

#### SQLITE 
import sqlite3

def question(text):
    global text1
    text1= text
    global questionPrep
    questionPrep=[]
    ### this part of speech tags would be important to know dependencies and lexical context in whole question
    global goldPOS
    goldPOS=[]
    global goldDoc
    goldDoc=nlp(text)
    for sent in goldDoc.sentences:
        for word in sent.words:
            goldPOS.append([word.id, f'{word.text}']+ [f'{word.upos}'] )
    return goldPOS
###################################################
#    VOCABULARY
selectClause=['select','get','choose','obtain','take','generate','retrieve','give','show','how','count','group','what']
w_pron=['which','where','whose','who','whom','that','there','with']
allCOL=['all','everything','every','each','whole','complete']
groupCOL=['group','cluster','aglomerate','per','by',"groupped"]
equal=['equal','equivalent','corresponding']
bigger= ['bigger','above','greater','over','larger','more','>', "higher", "after"]
lower=['lower','under','smaller','less','<', "below", "before", "fewer"]
order=["order", "ordered", "sorted", "sort"]
groupBy= ["different"]
aggregations=['average','avg',"between",'mean','biggest','lowest','max','maximum','largest','min','minimum','smallest','sum']
avg= ['average','avg','mean']
between= ["between"]
maximum=['maximum','max','largest','biggest', "most", "highest", "greatest", "longest", "latest"]
minimum=['minimum','min','smallest','lowest', "least", "fewest", "shortest", "earliest", "fewest"]
sum=['sum']
to_ignore=["frequency","count","id","ids","sorted", "ones","corresponding","fewer","records","record","fewest","shortest","longest","frequent","in","one","two","youngest","oldest","before","after","three","four", "five", "six", "seven", "eight", "nine","ten","how_many","ascending", "descending","decreasing","below","higher","least","but","highest", "most",'smallest','lowest','largest','biggest','bigger',"is","and", "are",'above',"older","younger",'greater','over','larger','more','lower','under','smaller','less',"of",",","number", "have","has","average","start","ended", "end","many","maximum","common", "minimum","distinct", "different", "descending", "ascending","alphabetical", "order", "greatest", "most", "kinds","newest", "oldest"]
special_cases=["older", "younger", "oldest", "youngest"]
count= ["count", "how_many"]
notIN= ["not", "neither", "nor", "no", "except", "dont"]
having = ["having", "has", "had", "have", "shared"]
punctPhrase= [".", "!", "?"]
phraseBroke= ["were", "was", "are", "has", "have", "do", "does", "for", "from","did"]
###################################################
# In case it is necessary to check the tables here goes the snippet to it.
# connection=sqlite3.connect('/Users/yuriyperezhohin2/Desktop/spider/database/climbing/climbing.sqlite')
# c= connection.cursor()
goldPOS=[]
ColumnEmbeddings=[]
tableColumnEmbeddings=[]
names={}
AllColumns= []
AllTableColumns= {}
tables=[]
memory={"goldPOScopy":goldPOS.copy(), "idxfirst" : None, "interIDX" : None, "idxSim1": None, "idxSim2": None, "exceptionSimilarity":None, "idxSim1Table": None, "idxSim2Table": None}
tableColumns={}
###################################################
################################################
#Functions for formulating queries based on a database
def sqlDatabase(pathToConnect):
    connection= sqlite3.connect(pathToConnect)
    connection.row_factory= lambda cursor, row:row[0]
    c= connection.cursor()
    sql= 'Select name FROM sqlite_schema;'
    result= c.execute(sql).fetchall()
    for name in result:
        if (name.startswith('sqlite')) or (name.startswith('idx_')) or (name.startswith('index_')):
            pass
        else:
            sql= c.execute(f'SELECT * from {name}')
            columns=[]
            for column in sql.description:
                columns.append(column[0])
            names[name]=columns
    connection.commit()
    connection.close()
    return 

# Generating embeddings for each table with respective columns

def sqlEmbeddings(names):
    columnsForEmbedding=[]
    pattern = re.compile('.*id$')
    for name in names:
        all= []
        columns=[]
        namesList=[]
        namesList.append(name)
        for id in names[name]:
            if ('_id' in str.lower(id)) or ('id_' in str.lower(id)) or (("id" in str.lower(id)) & (len(id)==2)) or (pattern.match(str.lower(id))):
                all.append(id)
                pass
            else:
                all.append(id)
                namesList.append(id)
                columns.append(id)
                AllColumns.append(id)
        if len(namesList)==1:
            namesList.append(namesList[0])
        if len(columns)>1:
            columnsForEmbedding.append(columns)
        if len(columns)==1:
            columnsForEmbedding.append(columns[0])
        AllTableColumns[name]= all
        tableColumns[name]=namesList
    processedNames = list(map(list, tableColumns.values()))
    tableList=list((tableColumns.keys()))
    tables.append(tableList)
    embedding= SimilarityModel.encode(processedNames,convert_to_tensor=True, normalize_embeddings=True)
    colEmbedding=SimilarityModel.encode(columnsForEmbedding,convert_to_tensor=True, normalize_embeddings=True)
    ColumnEmbeddings.append(colEmbedding)
    tableColumnEmbeddings.append(embedding)
    return 


def phraseCouter(goldPOS, memory):
    memory["phrases"]=0
    memory["idxEndPhrase"]=[]
    for x in goldPOS:
        try:
            if (x[1] in punctPhrase) & (goldPOS[x[0]][0]==1):
                memory["phrases"]+=1
                memory["idxEndPhrase"].append(x[0]-1)
                break
        except:pass
        try:
            if x[1] in punctPhrase:
                memory["phrases"]+=1
                memory["idxEndPhrase"].append(x[0]-1)
        except:pass
    if len(memory["idxEndPhrase"])==0:
        memory["idxEndPhrase"]= [goldPOS[-1][0]-1]
    if memory["phrases"]== 0 :
        memory["phrases"]=1
    return

def truncateSpecialChars(goldPOS):
    i = 0
    idx = []
    if memory["phrases"]>1:
        index= memory["idxEndPhrase"][0]
    if memory["phrases"]==1:
        index= memory["idxEndPhrase"][0]
    while i < index:
        if (goldPOS[i][2] == "PUNCT") and (goldPOS[i][1] in ['"', "'"]):
            idx.append(goldPOS[i][0]-1)
            start_index = i + 1
            end_index = start_index
            while end_index < index and not ((goldPOS[end_index][2] == "PUNCT") and (goldPOS[end_index][1] in ['"', "'"])):
                end_index += 1
            if end_index - start_index > 1:
                new_wordlist=[]
                for j in range(start_index, end_index):
                    if ((goldPOS[j][2] != "PUNCT") & (goldPOS[j][1]!="-")):
                        new_wordlist.append((goldPOS[j][1]))
                    if (goldPOS[j][2]=="PUNCT") & (goldPOS[j][1]=="-"):
                        new_wordlist.append((goldPOS[j][1]))
                if "-" in new_wordlist:
                    new_word= "".join(new_wordlist)
                if "-" not in new_wordlist:
                    new_word= " ".join(new_wordlist)
#                new_word = "_".join([goldPOS[j][1] for j in range(start_index, end_index) if ((goldPOS[j][2] != "PUNCT") | (goldPOS[j][1]=="-"))])
                goldPOS[i+1:end_index] = [[_+1, "ignore", "deleted"] for _ in range(start_index, end_index)]
                goldPOS[i+1] = [goldPOS[i+1][0], new_word, "Value"]
            if ((end_index - start_index) == 1 )& (goldPOS[end_index-1][2] not in ["PUNCT"]): 
                if goldPOS[i+1][1] in ["and", "or"]:
                    pass
                else:
                    goldPOS[i+1] = [goldPOS[i+1][0], goldPOS[i+1][1], "Value"]
        i += 1
    if len(idx)>1:   
        memory["quotationMarkIDX"]= idx 
    if len(idx)==0:
        memory["quotationMarkIDX"]= [-1, -1]
    if len(idx)==1:
        idx.append(goldPOS[-1][0]-1)
        memory["quotationMarkIDX"]= idx
    return 


def propnTruncation(goldPOS):
    for x in goldPOS:
        try:
            if (x[2] =='PROPN') & (goldPOS[x[0]][2] != "PROPN") & (x[0]-1 not in range(memory["quotationMarkIDX"][0], memory["quotationMarkIDX"][-1])):
                memory["propnPattern"]= x[1]
                goldPOS[x[0]-1]= [x[0], x[1], "propnPattern"]
            if ((x[2] =='PROPN') & (goldPOS[x[0]][2] =="PROPN")) & (x[0]-1 not in range(memory["quotationMarkIDX"][0], memory["quotationMarkIDX"][-1])):
                memory["propnPattern"]= x[1]
                goldPOS[x[0]-1]= [x[0], f'{x[1]} {goldPOS[x[0]][1]}', "propnPattern"]
                goldPOS[x[0]]= [x[0]+1, "ignore", "deleted by propn Truncation"]
        except:pass
    return



memory["tripleCompound"]=None
def tripleCompoundDepencies(text1, goldPOS):
    doc = nlp(text1)
    dep= []
    for sentence in doc.sentences:
        for word in sentence.words:
            dep.append([word.id, word.text, word.head, word.deprel, word.upos])
    for x in dep:
        if (x[3] == "compound") & (x[4]!="PROPN") & (x[0]-1 not in range(memory["quotationMarkIDX"][0], memory["quotationMarkIDX"][-1])) & (x[2] not in to_ignore):
            head= x[2]
            id=x[0]
            for z in dep:
                try:
                    if (z[0]>=id) & (dep[z[0]-1][3]=="compound") & (dep[z[0]][3]=="compound") & (dep[dep[z[2]-1][2]-1][0] == z[0]+2) & (z[0]<= head+2):
                        goldPOS[id-1]=[id, f'{z[1]}_{dep[z[0]][1]}_{dep[dep[z[2]-1][2]-1][1] }', "PatternEmbed"]
                        goldPOS[z[0]]= [z[0]+1,"ignore", "Deleted by multiple Compound dependencies"]
                        goldPOS[dep[z[0]][0]]= [dep[z[0]][0], "ignore", "Deleted by multiple Compound dependencies"]
                        memory["tripleCompound"]= [z[0]-1, dep[z[0]][0]-1, dep[dep[z[2]-1][2]-1][0]-1]
                except:pass
    return



def compoundDependencies(text1,goldPOS):
    if memory["tripleCompound"]==None:
        memory["tripleCompound"]= [-1]
    else: pass
    memory["compoundIDX"]= []
    doc=nlp(text1)
    dep=[]
    for sentence in doc.sentences:
        for word in sentence.words:
            dep.append([word.id, word.text, word.head])
    for sentence in doc.sentences:
        for word in sentence.words:
            if (word.deprel=='compound') & (word.head==word.id+1)&(word.id not in memory["tripleCompound"]) & (word.upos not in ["PROPN"])& (word.id-1 not in range(memory["quotationMarkIDX"][0], memory["quotationMarkIDX"][-1])) & (word.text not in to_ignore) & (goldPOS[word.head-1][1] not in to_ignore):
                goldPOS[word.id-1]= [word.id, f'{word.text}_{dep[word.head-1][1]}', "PatternEmbed"]
                goldPOS[word.head-1]= [word.head, "ignore", "deleted by Compound"]
                memory["compoundIDX"].append(word.id-1)
                memory["compoundIDX"].append(word.head-1)
    return

def amodDependencies(text1,goldPOS):
    doc=nlp(text1)
    dep=[]
    for sentence in doc.sentences:
        for word in sentence.words:
            dep.append([word.id, word.text, word.head])
    for sentence in doc.sentences:
        for word in sentence.words:
            if (word.deprel=='amod') &(goldPOS[word.head-1][1]!="number") &(word.text !="number") & (word.text not in to_ignore) & (word.id-1 not in memory["compoundIDX"])&(word.upos!= "PROPN") & (word.id-1 not in range(memory["quotationMarkIDX"][0], memory["quotationMarkIDX"][-1])):
                goldPOS[word.id-1]= [word.id, f'{word.text}_{dep[word.head-1][1]}', "PatternEmbed"]
                goldPOS[word.head-1]= [word.head, "ignore", "deleted by Adjectival modifier"]
    return

def conjuctionDependencies(text1, goldPOS):
    doc = nlp(text1)
    dep=[]
    for sentence in doc.sentences:
        for word in sentence.words:
            dep.append([word.id, word.text, word.head, word.upos])
    for sentence in doc.sentences:
        for word in sentence.words:
            try:
                if (word.deprel=="conj") &(dep[dep[word.head][2]][3] in ["NOUN"]) &  (word.upos not in ["PROPN", "NUM", "PUNCT"]) & (dep[dep[word.head][2]][1] not in to_ignore) & (word.text not in to_ignore) & (bool(goldPOS[dep[word.head][2]-1][2]!="deleted by Compound")) & (word.id-1 not in range(memory["quotationMarkIDX"][0], memory["quotationMarkIDX"][-1])):
                    goldPOS[word.id-1]= [word.id, f'{word.text}_{dep[dep[word.head][2]][1]}', "PatternEmbed"]
            except:pass
    return

#### Tenho q inventar uma coisa nova para estas dependencias, MUITAS vezes junta verbos com nomes desnecessários.
# def aclDependencies(text1, goldPOS):
#     doc = nlp(text1)
#     dep= []
#     for sentence in doc.sentences:
#         for word in sentence.words:
#             dep.append([word.id, word.text, word.head])
#     for sentence in doc.sentences:
#         for word in sentence.words:
#             if (word.deprel=="acl:relcl") & (word.upos != "PROPN")&  (word.text not in to_ignore) &(goldPOS[word.head-1][1] not in to_ignore) & (goldPOS[word.id-2][1] not in ['"', "'"]):
#                 goldPOS[word.head-1]= [word.head, f'{word.text}_{dep[word.head-1][1]}', "PatternEmbed"]
#                 goldPOS[word.id-1]= [word.id, "ignore", "deleted by ACL modifier"]

#     return
# aclDependencies(text1, goldPOS)

def mixedDependencies(text1, goldPOS):
    doc= nlp(text1)
    dep=[]
    for sentence in doc.sentences:
        for word in sentence.words:
            dep.append([word.id,word.text,word.head, word.deprel, word.upos])
    for x in dep:
        if (x[3] in["amod", "compound"]) & (x[0]-1 not in memory["compoundIDX"] )&  (x[0]-1 not in memory["tripleCompound"]) &(x[1] not in to_ignore) & (x[1] != "total") & (x[4] not in ["PROPN"]) & (x[0]-1 not in range(memory["quotationMarkIDX"][0], memory["quotationMarkIDX"][-1])):
            head = x[2]
            id= x[0]
            for z in dep:
                if (z[0]>id) & (z[2]==head) & (z[3] in["amod", "compound"]):
                    goldPOS[id-1]= [id, f'{x[1]}_{z[1]}_{dep[head-1][1]}', f'PatternEmbed']
                    goldPOS[z[0]-1]= [z[0], "ignore", "Deleted by multiple Dependencies"]
                    goldPOS[head-1]= [head, "ignore", "Deleted by multiple Dependencies"]
    return



def CountTruncate(goldPOS):
    try:
        for x in goldPOS:
            if (str.lower(x[1])== "how") & (str.lower(goldPOS[x[0]][1])=="many") :
                goldPOS[x[0]-1]= [x[0], "how_many", "Counter"]
                goldPOS[x[0]]= [x[0]+1, "ignore", "deleted by counter"]
    except:pass
    return



def ofCountPattern(goldPOS,memory):
    memory["ofCount"]=0
    idx=[]
    for x in goldPOS:
        if x[1]=='of':
            memory["ofCount"]+=1
            idx.append(x[0]-1)
    if memory["ofCount"]==1:
        memory["idxfirst"]= idx[0]
    if memory["ofCount"]==1:
        for x in goldPOS:
            if (x[0]>memory["idxfirst"]+1) &  (x[0]<= memory["idxfirst"]+2) & (x[2] not in ["NOUN","PROPN","ADJ","DET"]):
                pass
            else:
                if (x[0] > memory["idxfirst"]+1) & ((x[2] not in ["NOUN","PROPN","ADJ","DET"]) | ((x[2] not in ["NOUN","PROPN","ADJ","DET", "PRON"]) & (x[1] not in ["where", "whose", "which", "who"]))):
                    memory["interIDX"]=x[0]-1
                    break
    if memory["ofCount"]>=2:
        memory["interIDXSecondOF"]= None
        memory["idxfirst"]= idx[0]
        memory["ofIdxSecond"]= idx[1]
        for x in goldPOS:
            if (x[0]>=memory["idxfirst"]+1)  & (x[2] not in ["NOUN","PROPN","ADJ","DET", "PatternEmbed"]):
                memory["interIDXdoubleOF"]= x[0]-1
        for x in goldPOS:
            if (x[0]> memory["ofIdxSecond"]+1) & (x[2] not in ["NOUN","PROPN","ADJ","DET", "PatternEmbed"]):
                memory["interIDXSecondOF"]= x[0]-1
        if memory["interIDXSecondOF"]==None:
            memory["interIDXSecondOF"]= goldPOS[-1][0]-1
    return

def OfexceptionFirst(memory, goldPOS,tables):
    if memory["ofCount"]==1:
        if (memory ["interIDX"]==None):
            memory["interIDX"]=goldPOS[-1][0]-1
        for x in goldPOS:
            if (x[0]>memory["idxfirst"]) & (x[2]in["NOUN","PatternEmbed"]) & (x[0]<=memory["interIDX"]+1):
                embedtables= SimilarityModel.encode(tables[0],convert_to_tensor=True,normalize_embeddings=True)
                embedNoun= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                sim1 = util.dot_score(embedNoun,embedtables)
                memory["idxSim1"]=x[0]-1
            if (x[0] <= memory["idxfirst"]) & (x[2] in ["NOUN","PatternEmbed"]):
                embedtables1=SimilarityModel.encode(tables[0],convert_to_tensor=True, normalize_embeddings=True)
                embedNoun1=SimilarityModel.encode(x[1],convert_to_tensor=True, normalize_embeddings=True)
                sim2= util.dot_score(embedNoun1,embedtables1)
                memory["idxSim2"]=x[0]-1
            if (x[0]<= memory["idxfirst"]) & (x[2].startswith("deleted")) & (memory["idxSim2"]==None):
                memory["idxSim2"]= x[0]-1
        if memory["idxSim1"]==None:
            memory["idxSim1"]=-1
            sim1=torch.tensor(0)
        if (goldPOS[memory["idxSim1"]][2]=='PatternEmbed') & (goldPOS[memory["idxSim2"]][2].startswith('deleted')):
            memory["idxSim2"]= None
        if (memory["idxSim2"]== None):
            sim2= torch.tensor(0)
            for x in goldPOS:
                if (x[0]<= memory["idxfirst"]) & (x[2].startswith("deleted")):
                    memory["idxSim2"]= x[0]-1
                    memory["exceptionSimilarity"] = 1
        if (torch.max(sim1) >0.85) & (torch.max(sim2)>0.85):
            #Function to check if the two nouns referenced before and after "of" are tables, "show me the heads of the departments"
            #print(tables[0][torch.argmax(sim2)],"both")
            memory["idxSim2Table"]=tables[0][torch.argmax(sim2)]
            memory["idxSim1Table"]=tables[0][torch.argmax(sim1)]
            memory["ofExceptionBoth"]="both"
        if (torch.max(sim1)>0.85) & (torch.max(sim2)<0.85):
            #print(tables[0][torch.argmax(sim1)],"second")
            memory["idxSim1Table"]=tables[0][torch.argmax(sim1)]
            memory["ofExceptionSecond"]='second'
        if (torch.max(sim1)<0.85) & (torch.max(sim2)>0.85):
            #print(tables[0][torch.argmax(sim2)],"first")
            memory["idxSim1Table"]=tables[0][torch.argmax(sim2)]
            memory["ofExceptionFirst"]='first'
        if (torch.max(sim1)<0.85) & (torch.max(sim2)<0.85):
            #print(goldPOS[memory["idxSim1"]][1], goldPOS[memory["idxSim2"]][1],"neither")
            memory["ofExceptionNeither"]="neither"
        if memory["idxSim1"]== -1:
            memory["idxSim1"]= None
    return

memory["idxAdpositionInclusiveLast"]= None
memory["idxAdpositionInclusive"]= None
def OfFirstexceptionSimilarity(memory, goldPOS):
    if (memory["exceptionSimilarity"]==1) & (memory["ofCount"]==1):
        for x in goldPOS: 
            if (x[0]> memory["idxfirst"]) & (x[1] in ["in", "on", "at", "for"]): 
                memory["idxAdpositionInclusive"]= x[0]-1
        if memory["idxAdpositionInclusive"]!=None:
            for x in goldPOS:
                if (x[0]> memory["idxAdpositionInclusive"]+1) & (x[2] not in ["DET", "NOUN", "PROPN", "ADJ", "PatternEmbed"]):
                    memory["idxAdpositionInclusiveLast"]= x[0]-1
                    break
            if memory["idxAdpositionInclusiveLast"]==None:
                memory["idxAdpositionInclusiveLast"]= goldPOS[-1][0]-1
            for x in goldPOS:
                if (x[0]> memory["idxAdpositionInclusive"]+1) & (x[0]<= memory["idxAdpositionInclusiveLast"]+1) & (x[2] in ["NOUN", "PROPN", "PatternEmbed"]):
                    embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                    embedTables= SimilarityModel.encode(tables[0],convert_to_tensor=True,  normalize_embeddings=True)
                    sim1= util.dot_score(embed, embedTables)
                    if torch.max(sim1)>0.85:
                        memory["idxSim1"]= x[0]-1
                        goldPOS[x[0]-1]= [x[0], f'Table:{tables[0][torch.argmax(sim1)]}', "Main table after 'of"]
                        memory["idxSim1Table"]= tables[0][torch.argmax(sim1)]
        else:pass
    return


def bothExceptionOf(memory,goldPOS,tableColumns):
    #### AS the two nouns are tables we first check if there are any reference to the Noun before "of" in the table after "of"
    ## in this case "show me the heads of the departments" heads do not pursue any refernces in the department table
    try:
        if (memory["ofCount"]==1) & (memory["exceptionSimilarity"]!=1):
            if memory["ofExceptionBoth"]=='both':
                columns= tableColumns[memory["idxSim1Table"]][1:]
                embedPattern=SimilarityModel.encode(goldPOS[memory["idxSim2"]][1], convert_to_tensor=True, normalize_embeddings=True)
                embedColumns= SimilarityModel.encode(columns,convert_to_tensor=True, normalize_embeddings=True)
                similarity= util.dot_score(embedPattern, embedColumns)
                if torch.max(similarity)<0.75:
                    columns=tableColumns[memory["idxSim2Table"]][1:]
                    embedColumns=SimilarityModel.encode(columns,convert_to_tensor=True, normalize_embeddings=True)
                    similarity=util.dot_score(embedPattern, embedColumns)
                    goldPOS[memory["idxSim2"]]=[memory["idxSim2"], f'Table:{memory["idxSim2Table"]}',f'Column:{columns[torch.argmax(similarity)]}']
                    goldPOS[memory["idxSim1"]]=[memory["idxSim1"], f'Table:{memory["idxSim1Table"]}',f'Is referencing the column:{columns[torch.argmax(similarity)]} from {memory["idxSim2Table"]}']
                if torch.max(similarity)>0.75:
                    goldPOS[memory["idxSim2"]]=[memory["idxSim2"], f'Table:{memory["idxSim1Table"]}',f'Column:{columns[torch.argmax(similarity)]}']
                    goldPOS[memory["idxSim1"]]=[memory["idxSim1"], f'Table:{memory["idxSim1Table"]}',f'Reference Table']            
    except:pass
    return


def secondExceptionOf(memory,goldPOS,tableColumns):
    try:
        if (memory["ofCount"]==1) & (memory["exceptionSimilarity"]!=1):
            if memory["ofExceptionSecond"]=='second':
                goldPOS[memory["idxSim1"]] = [memory["idxSim1"]+1, f'Table:{memory["idxSim1Table"]}',f'Main table after "of"']
                for x in goldPOS:
                    if (x[0]<=memory["idxfirst"]) & (x[1] not in ["number"]) & (x[2] in ['NOUN','PatternEmbed']):
                        columns=tableColumns[memory["idxSim1Table"]][1:]
                        embedAllCols= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                        TabelsEmbed= SimilarityModel.encode(tables[0],convert_to_tensor=True,normalize_embeddings=True)
                        embed= SimilarityModel.encode(x[1], convert_to_tensor=True,normalize_embeddings=True)
                        embedTable= SimilarityModel.encode(columns, convert_to_tensor=True, normalize_embeddings=True)
                        colSim= util.dot_score(embed, embedAllCols)
                        column= AllColumns[torch.argmax(colSim)]
                        sim= util.dot_score(embed, embedTable)
                        tablesSim= util.dot_score(embed, TabelsEmbed)
                        if (torch.max(tablesSim)>=0.8) & (torch.max(colSim)<0.75):
                            goldPOS[x[0]-1]= [x[0], f'{tables[0][torch.argmax(tablesSim)]}', "Reference table before 'of'"]
                            memory["idxSim1Table"]= tables[0][torch.argmax(tablesSim)]
                        if (torch.max(sim)>0.45) & (torch.max(tablesSim)<0.8) & (torch.max(colSim)<0.75):
                            goldPOS[x[0]-1]=[x[0], f'Table:{memory["idxSim1Table"]}',f'Column:{columns[torch.argmax(sim)]}']
                        if (torch.max(sim)<0.45) & (torch.max(tablesSim)<0.8) & (torch.max(colSim)>=0.75):
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[x[0]-1] = [x[0], f'Table:{table}', f'Column:{column}']
                        if (torch.max(sim)>0.45) & (torch.max(colSim)>0.75):
                            if (np.round(torch.max(sim).item(), decimals=3)) >= (np.round(torch.max(colSim).item(),decimals=3)):
                                goldPOS[x[0]-1]= [x[0], f"Table:{memory['idxSim1Table']}", f"Column:{columns[torch.argmax(sim)]}"]
                            if (np.round(torch.max(sim).item(), decimals=3)) < (np.round(torch.max(colSim).item(),decimals=3)):
                                table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                                goldPOS[x[0]-1] = [x[0], f'Table:{table}', f'Column:{column}']
    except:pass
    return    


def firstExceptionOf(memory,goldPOS):
    try:
        if (memory["ofCount"]==1)& (memory["exceptionSimilarity"]!=1):
            if memory["ofExceptionFirst"]=='first':
                goldPOS[memory["idxSim2"]]= [memory["idxSim2"], f'Table:{memory["idxSim1Table"]}','Main table before "of"']
                for x in goldPOS:
                    if (x[0]<= memory["idxfirst"]) & (x[1] not in to_ignore) &(x[2] in ["NOUN", "PatternEmbed"]):  
                        embed= SimilarityModel.encode(x[1], convert_to_tensor=True,normalize_embeddings=True)
                        EmbedTables= SimilarityModel.encode(tables[0], convert_to_tensor=True, normalize_embeddings=True)
                        TablesSim= util.dot_score(embed, EmbedTables)
                        if torch.max(TablesSim)>=0.8:
                            goldPOS[x[0]-1]= [x[0], f'Table:{tables[0][torch.argmax(TablesSim)]}', "Reference table before'of'"]
                            memory["idxSim2Table"]= tables[0][torch.argmax(TablesSim)]
                        if torch.max(TablesSim)<0.8:
                            embedCols= SimilarityModel.encode(tableColumns[memory["idxSim2Table"]][1:],normalize_embeddings=True, convert_to_tensor=True)
                            sim = util.dot_score(embed, embedCols)
                            if torch.max(sim)>=0.65:
                                goldPOS[x[0]-1]= [x[0], f'Table:{memory["idxSim1Table"]}', f'Column:{tableColumns[memory["idxSim1Table"]][1:][torch.argmax(sim)]}']
                            if torch.max(sim)<0.65:
                                sim2= util.dot_score(embed, ColumnEmbeddings[0])
                                table= tables[0][torch.argmax(sim2)]
                                encodeTableCol= SimilarityModel.encode(tableColumns[table], convert_to_tensor=True, normalize_embeddings=True)
                                sim3= util.dot_score(embed,encodeTableCol) 
                                if torch.max(sim3)>0.45:
                                    goldPOS[x[0]-1]= [x[0], f'Table:{table}', f'Column:{tableColumns[table][torch.argmax(sim3)]}']
                    if (x[0]>= memory["idxfirst"]) & (x[1] not in to_ignore) &(x[2] in ["NOUN", "PatternEmbed"]):
                        embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                        EmbedTables= SimilarityModel.encode(tables[0], convert_to_tensor=True, normalize_embeddings=True)
                        TablesSim= util.dot_score(embed, EmbedTables)
                        if torch.max(TablesSim)>=0.8:
                            goldPOS[x[0]-1]= [x[0], f'Table:{tables[0][torch.argmax(TablesSim)]}', "Reference table after'of'"]
                        if torch.max(TablesSim)<0.8:
                            embedCols= SimilarityModel.encode(tableColumns[memory["idxSim1Table"]][1:],convert_to_tensor=True, normalize_embeddings=True)
                            sim = util.dot_score(embed, embedCols)
                            if torch.max(sim)>=0.65:
                                goldPOS[x[0]-1]= [x[0], f'Table:{memory["idxSim1Table"]}', f'Column:{tableColumns[memory["idxSim1Table"]][1:][torch.argmax(sim)]}']
                            if torch.max(sim)<0.65:
                                sim2= util.dot_score(embed, ColumnEmbeddings[0])
                                table= tables[0][torch.argmax(sim2)]
                                encodeTableCol= SimilarityModel.encode(tableColumns[table], convert_to_tensor=True, normalize_embeddings=True)
                                sim3= util.dot_score(embed,encodeTableCol) 
                                if torch.max(sim3)>0.45:
                                    goldPOS[x[0]-1]= [x[0], f'Table:{table}', f'Column:{tableColumns[table][torch.argmax(sim3)]}']
    except:pass
    return

def neitherExceptionOf(memory, goldPOS):
    try:
        if(memory["ofCount"]==1) & (memory["exceptionSimilarity"]!=1)  :
            if memory["ofExceptionNeither"]=='neither':
                embed= SimilarityModel.encode(goldPOS[memory["idxSim1"]][1], convert_to_tensor=True, normalize_embeddings=True)
                embedTables= SimilarityModel.encode(tables[0], convert_to_tensor=True, normalize_embeddings=True)
                embedALLCols= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                tablesim= util.dot_score(embed,embedTables)
                AllColsSim= util.dot_score(embed, embedALLCols)
                if (torch.max(tablesim)>0.55) & (torch.max(AllColsSim)<0.70):
                    embedColumns = SimilarityModel.encode(tableColumns[tables[0][torch.argmax(tablesim)]][1:], convert_to_tensor=True,normalize_embeddings=True)
                    colsim= util.dot_score(embed, embedColumns)
                    if torch.max(colsim)>=0.75:
                        goldPOS[memory["idxSim1"]]=[memory["idxSim1"], f"Table:{tables[0][torch.argmax(tablesim)]}", f"Column:{tableColumns[tables[0][torch.argmax(tablesim)]][1:][torch.argmax(colsim)]}"] 
                        memory["idxSim1Table"]= tables[0][torch.argmax(tablesim)]
                    if torch.max(colsim)<0.75:
                        goldPOS[memory["idxSim1"]] = [memory["idxSim1"],f'Table:{tables[0][torch.argmax(tablesim)]}',"Reference table after 'of' "]
                        columns= tableColumns[tables[0][torch.argmax(tablesim)]][1:]
                        memory["idxSim1Table"]= tables[0][torch.argmax(tablesim)]
                    for x in goldPOS:
                        if (x[0]<= memory["idxfirst"]) & (x[2]in ["NOUN", "PatternEmbed"]) & (x[1] not in to_ignore):
                            embed=SimilarityModel.encode(x[1],convert_to_tensor=True, normalize_embeddings=True)
                            columnEmbed= SimilarityModel.encode(columns, convert_to_tensor=True, normalize_embeddings=True)
                            sim = util.dot_score(embed, columnEmbed)
                            TablesSim= util.dot_score(embed, embedTables)
                            if torch.max(TablesSim)>=0.85:
                                goldPOS[x[0]-1]= [x[0], f'Table:{tables[0][torch.argmax(TablesSim)]}', "Reference Table before 'of'"]
                                memory["idxSim2Table"]=tables[0][torch.argmax(TablesSim)]
                            if (torch.max(sim)>0.6) & (torch.max(TablesSim)<0.85):
                                goldPOS[x[0]-1] = [x[0]-1, f'Table:{tables[0][torch.argmax(tablesim)]}', f'Column:{columns[torch.argmax(sim)]}' ]
                            if (torch.max(sim)<=0.6) & (torch.max(TablesSim)<0.85):
                                allCollEmbed= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                                simCol= util.dot_score(allCollEmbed, embed)
                                if torch.max(simCol)>0.4:
                                    column= AllColumns[torch.argmax(simCol)]
                                    table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                                    goldPOS[x[0]-1]= [x[0], f'Table:{table}', f"Column:{column}"]
                if (torch.max(tablesim)>0.55) & (torch.max(AllColsSim)>=0.70):
                    column= AllColumns[torch.argmax(AllColsSim)]
                    table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                    goldPOS[goldPOS[memory["idxSim1"]-1][0]]= [goldPOS[memory["idxSim1"]][0], f'Table:{table}', f"Column:{column}"]
                if (torch.max(tablesim)<=0.55):
                    for x in goldPOS:
                        if (x[0]>=memory["idxSim1"]) & (x[1] in ["in", "at", "on", "for"]):
                            memory["idxAdpositionInclusive"]= x[0]-1
                            break
                    for x in goldPOS:
                        if memory["idxAdpositionInclusive"]!=None:
                            if (x[0]> memory["idxAdpositionInclusive"]) & (x[2] not in ["ADJ", "DET", "NOUN", "PROPN", "PatternEmbed"]):
                                memory["idxAdpositionInclusiveLast"]= x[0]-1
                    if (memory["idxAdpositionInclusiveLast"]==None) & (memory["idxAdpositionInclusive"]!= None):
                        memory["idxAdpositionInclusiveLast"]= goldPOS[-1][0]-1
                    if (memory["idxAdpositionInclusiveLast"]!=None) & (memory["idxAdpositionInclusive"]!=None):
                        for x in goldPOS:
                            if (x[0]>=memory["idxAdpositionInclusive"]+1) & (x[0]<=memory["idxAdpositionInclusiveLast"]+1) & (x[2] in ["NOUN", "PatternEmbed"]) & (x[1] not in to_ignore):
                                embedPattern= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                                embedTable= SimilarityModel.encode(tables[0],convert_to_tensor=True, normalize_embeddings=True)
                                #### add similarity to search table, if big enough add main table after of, if not search for column 
                                TableSim= util.dot_score(embedPattern, embedTable)
                                if torch.max(TableSim)>=0.85:
                                    goldPOS[x[0]-1]= [x[0], f"Table:{tables[0][torch.argmax(TableSim)]}", "Main Table after 'of"]
                                    memory["idxSim1Table"]= tables[0][torch.argmax(TableSim)]
                                if torch.max(TableSim)<0.85:
                                    allColumnEmbedding= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                                    colsim= util.dot_score(embedPattern, allColumnEmbedding)
                                    if torch.max(colsim)>0.4:
                                        column= AllColumns[torch.argmax(colsim)]
                                        table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                                        goldPOS[x[0]-1]= [x[0], f'Table:{table}', f"Column:{column}"]
                        for x in goldPOS:
                            if (x[0]<= memory["idxAdpositionInclusive"]+1)& (x[0]>= memory["idxfirst"]+1) &(x[2]in ["NOUN", "PatternEmbed"]) & (x[1] not in to_ignore):
                                embedPattern = SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                                # Here we slice it from the first index as it represent the table name
                                embedCols= SimilarityModel.encode(tableColumns[tables[0][torch.argmax(TableSim)]][1:],convert_to_tensor=True, normalize_embeddings=True)
                                ColSim= util.dot_score(embedPattern, embedCols)
                                if torch.max(ColSim)>=0.65:
                                    goldPOS[x[0]-1]= [x[0]-1, f'Table:{tables[0][torch.argmax(TableSim)]}', f'Column:{tableColumns[tables[0][torch.argmax(TableSim)]][1:][torch.argmax(ColSim)]}']
                                if torch.max(ColSim)<0.65:
                                    TableSim= util.dot_score(embedPattern, tableColumnEmbeddings[0])
                                    # here we ponderate to all names inside the list as we dont know from which table it follows.
                                    tableCols= tableColumns[tables[0][torch.argmax(TableSim)]][1:]
                                    embedTableCols= SimilarityModel.encode(tableCols, convert_to_tensor=True, normalize_embeddings=True)
                                    exceptSim= util.dot_score(embedPattern, embedTableCols)
                                    if torch.max(exceptSim)>0.4:
                                        goldPOS[x[0]-1] = [x[0]-1, f'Table:{tables[0][torch.argmax(TableSim)]}', f'Column:{tableCols[torch.argmax(exceptSim)]}']
                    if (memory["idxAdpositionInclusiveLast"]==None) & (memory["idxAdpositionInclusive"]== None):
                        allCollEmbed= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                        colsim= util.dot_score(embed, allCollEmbed)
                        if torch.max(colsim)>0.4:
                            column= AllColumns[torch.argmax(colsim)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[memory["idxSim1"]] = [memory["idxSim1"], f'Table:{table}', f'Column:{column}']

    except:pass           
    return 


memory["interOfidx"]=None
def neitherSimilarityException(memory, goldPOS):
    try:
        if (memory["ofCount"]==1) & (memory["ofExceptionNeither"]== "neither") & (memory["exceptionSimilarity"]==1):
            for x in goldPOS:
                if (x[0]>= memory["idxSim1"]) &(x[2] not in ["ADJ", "DET", "NOUN", "PROPN"]):
                    memory["interOfidx"]= x[0]-1
            if memory["interOfidx"]==None:
                memory["interOfidx"]= goldPOS[-1][0]-1
            for x in goldPOS:
                if (x[0]>= memory["idxSim1"]) & (x[0]<= memory["interOfidx"]+1) & (x[2] in ["NOUN", "PatternEmbed"]) & (x[1] not in to_ignore ):
                    embedTables= SimilarityModel.encode(tables[0], convert_to_tensor=True, normalize_embeddings=True)
                    embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                    if torch.max(util.dot_score(embed, embedTables))>=0.8:
                        goldPOS[x[0]-1]= [x[0], f'Table:{tables[0][torch.argmax(util.dot_score(embed, embedTables))]}', "Reference Table after 'of"]
                        memory["idxSim1Table"]= tables[0][torch.argmax(util.dot_score(embed, embedTables))]
                    if torch.max(util.dot_score(embed, embedTables))<0.8:
                        colEmbeddings= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                        ColSim= util.dot_score(embed, colEmbeddings)
                        column= AllColumns[torch.argmax(ColSim)]
                        table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                        goldPOS[x[0]-1] = [x[0]-1, f'Table:{table}', f"Column:{column}"]

            for x in goldPOS:
                if (x[0]<=memory["idxSim1"])&(x[2] in ["NOUN","PROPN", "PatternEmbed"]) &(x[1] not in to_ignore):
                    embedTables= SimilarityModel.encode(tables[0], convert_to_tensor=True, normalize_embeddings=True)
                    embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                    if torch.max(util.dot_score(embed, embedTables))>=0.8:
                        goldPOS[x[0]-1]= [x[0], f'Table:{tables[0][torch.argmax(util.dot_score(embed, embedTables))]}', "Reference Table before 'of"]
                        memory["idxSim1Table"]= tables[0][torch.argmax(util.dot_score(embed, embedTables))]
                    if torch.max(util.dot_score(embed, embedTables))<0.8:
                        colEmbeddings= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                        ColSim= util.dot_score(embed, colEmbeddings)
                        column= AllColumns[torch.argmax(ColSim)]
                        table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                        goldPOS[x[0]-1] = [x[0]-1, f'Table:{table}', f"Column:{column}"]
    except:pass
    return


## idxfirst
## ofIdxSecond
memory["doubleOfExceptionBoth"]= None
def doubleOfExceptionBoth(goldPOS, memory):
    try:
        if memory["ofCount"]>=2:
            if (memory ["interIDXdoubleOF"]==None):
                memory["interIDXdoubleOF"] = memory["ofIdxSecond"]
            if (memory["interIDXdoubleOF"]> memory["ofIdxSecond"]):
                memory["interIDXdoubleOF"] = memory["ofIdxSecond"]
            sim1 = None
            for x in goldPOS:
                if (x[2] in ["NOUN", "PatternEmbed"]) & (x[0]> memory["ofIdxSecond"]) & (x[1] not in to_ignore) & (x[0]<= memory["interIDXSecondOF"]+1):
                    embed= SimilarityModel.encode(x[1], convert_to_tensor=True,normalize_embeddings=True)
                    tablesEmbed= SimilarityModel.encode(tables[0], convert_to_tensor=True, normalize_embeddings=True)
                    sim= util.dot_score(embed, tablesEmbed)
                    if torch.max(sim)>0.85:
                        memory["SecondIndexLast"]= x[0]-1
                        for z in goldPOS:
                            if (z[0]<= memory["interIDXdoubleOF"]) & (z[0]>memory["idxfirst"]) & (z[2] in ["NOUN", "PatternEmbed"]) & (z[1] not in to_ignore):
                                embed1= SimilarityModel.encode(z[1], convert_to_tensor=True, normalize_embeddings=True)
                                tablesEmbed= SimilarityModel.encode(tables[0], convert_to_tensor=True, normalize_embeddings=True)
                                sim1 = util.dot_score(embed1, tablesEmbed)
                                if torch.max(sim1)>0.75:
                                    memory["SecondIndexFirst"]= z[0]-1
                                    goldPOS[memory["SecondIndexLast"]]= [memory["SecondIndexLast"], f"Table:{tables[0][torch.argmax(sim)]}", f"Is referencing to the table {tables[0][torch.argmax(sim1)]}"]
                                    goldPOS[memory["SecondIndexFirst"]]= [z[0], f'Table:{tables[0][torch.argmax(sim1)]}', "Main table between 'of's' adpostions"]
                                    memory["idxSim1Table"]= tables[0][torch.argmax(sim1)]
                                    memory["doubleOfExceptionBoth"]= 1
                                if torch.max(sim1)<0.75:
                                    goldPOS[memory["SecondIndexLast"]]= [memory["SecondIndexLast"], f'Table:{tables[0][torch.argmax(sim)]}', f'Main Table after double "of"']
                                    colEmbeddings= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                                    ColSim= util.dot_score(embed1, colEmbeddings)
                                    column= AllColumns[torch.argmax(ColSim)]
                                    table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                                    memory["idxSim2Table"]= table
                                    embedMainTableCols= SimilarityModel.encode(tableColumns[tables[0][torch.argmax(sim)]][1:],convert_to_tensor=True,normalize_embeddings=True)
                                    mainTableColSimilarity= util.dot_score(embed1, embedMainTableCols)
                                    if torch.max(mainTableColSimilarity)>0.8:
                                        goldPOS[z[0]-1]= [z[0], f'Table:{tables[0][torch.argmax(sim)]}', f'Column:{tableColumns[tables[0][torch.argmax(sim)]][1:][torch.argmax(mainTableColSimilarity)]}']
                                        memory["doubleOfExceptionBoth"]= 1
                                        memory["idxSim2Table"]= tables[0][torch.argmax(sim)]
                                    if (torch.max(ColSim)>=0.45)& (torch.max(ColSim)<=0.8):
                                        goldPOS[z[0]-1]= [z[0], f'Table:{table}', f'Column:{column}']
                                        memory["idxSim2Table"]= table
                                        memory["doubleOfExceptionBoth"]= 1
        for x in goldPOS:
            if (x[0]<= memory["ofIdxSecond"]) & (x[2] in ["NOUN", "PatternEmbed"]) & (x[1] not in to_ignore):
                embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                columns= tableColumns[tables[0][torch.argmax(sim1)]]
                embedTable= SimilarityModel.encode(columns, convert_to_tensor=True, normalize_embeddings=True)
                sim2 = util.dot_score(embed, embedTable)
                if torch.max(sim2)>=0.6:
                    goldPOS[x[0]-1]= [x[0], f"Table:{tables[0][torch.argmax(sim1)]}", f"Column:{columns[torch.argmax(sim2)]}"]
                if torch.max(sim2)<0.6:
                    sim3= util.dot_score(embed, tableColumnEmbeddings[0])
                    table= tables[0][torch.argmax(sim3)]
                    embedCol= SimilarityModel.encode(tableColumns[table], convert_to_tensor=True, normalize_embeddings=True)
                    sim4= util.dot_score(embed, embedCol)
                    goldPOS[x[0]-1]= [x[0], f'Table:{table}', f'Column:{tableColumns[table][torch.argmax(sim4)]}']
                    

    except: pass                            
    return


memory["doubleOfExceptionSecond"]=None
def doubleOfExceptionSecond(goldPOS, memory):
    try:
        if (memory["ofCount"]>=2) & (memory["doubleOfExceptionBoth"]==None):
            if (memory ["interIDXdoubleOF"] == None):
                memory["interIDXdoubleOF"] = memory["ofIdxSecond"]
            if (memory["interIDXdoubleOF"]> memory["ofIdxSecond"]):
                memory["interIDXdoubleOF"] = memory["ofIdxSecond"]
            memory["mainTableAfterDoubleOf"] = None
            for x in goldPOS:
                if (x[0] > memory["interIDXdoubleOF"]) &(x[0]<= memory["interIDXSecondOF"])& (x[2] in ["NOUN", "PatternEmbed"]) & (x[1] not in ["number", "different","distinct","order"]):
                    embed= SimilarityModel.encode(x[1],convert_to_tensor=True, normalize_embeddings=True)
                    TablesEmbed= SimilarityModel.encode(tables[0],convert_to_tensor=True, normalize_embeddings=True)
                    sim= util.dot_score(embed, TablesEmbed)
                    if torch.max(sim)>0.80:
                        goldPOS[x[0]-1]= [x[0], f'Table:{tables[0][torch.argmax(sim)]}', "Main table after double 'of'"]
                        memory["idxSim2Table"]= tables[0][torch.argmax(sim)]
                        memory["mainTableAfterDoubleOf"]= tables[0][torch.argmax(sim)]
                        memory["doubleOfExceptionSecond"]=1
            if memory["mainTableAfterDoubleOf"]!= None:
                for x in goldPOS:
                    if (x[0] >= memory["idxfirst"]+1) & (x[0]<= memory["ofIdxSecond"]) & (x[2] in ["NOUN", "PatternEmbed"]) & (x[1] not in ["number","different", "distinct","order"]): 
                        embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                        columns= tableColumns[tables[0][torch.argmax(sim)]]
                        embedColumns= SimilarityModel.encode(columns, convert_to_tensor=True, normalize_embeddings=True)
                        sim1= util.dot_score(embed, embedColumns)
                        goldPOS[x[0]-1]= [x[0]-1, f'Table:{tables[0][torch.argmax(sim)]}', f'Column:{columns[torch.argmax(sim1)]}']
                for x in goldPOS: 
                    if (x[0]<= memory["idxfirst"]) & (x[2] in ["NOUN", "PatternEmbed"]) & (x[1] not in ["number","different", "distinct","order"]):
                        embed= SimilarityModel.encode(x[1],convert_to_tensor=True, normalize_embeddings=True)
                        TablesEmbed= SimilarityModel.encode(tables[0], convert_to_tensor=True, normalize_embeddings=True)
                        embedTableCol= SimilarityModel.encode(tableColumns[memory["mainTableAfterDoubleOf"]][1:],convert_to_tensor=True, normalize_embeddings=True)
                        sim= util.dot_score(embed, embedTableCol)
                        tableSim= util.dot_score(embed,TablesEmbed)
                        if torch.max(tableSim)>=0.85:
                            goldPOS[x[0]-1]= [x[0], f'{tables[0][torch.argmax(tableSim)]}', f'Reference Table before "of"']
                        if (torch.max(sim)>0.65) & (torch.max(tableSim)<0.85):
                            goldPOS[x[0]-1]= [x[0]-1, f'Table:{memory["mainTableAfterDoubleOf"]}', f'Column:{tableColumns[memory["mainTableAfterDoubleOf"]][1:][torch.argmax(sim)]}']
                        else:
                            sim1= util.dot_score(embed, tableColumnEmbeddings[0])
                            columns= tableColumns[tables[0][torch.argmax(sim1)]][1:]
                            embedCol= SimilarityModel.encode(columns, convert_to_tensor=True, normalize_embeddings=True)
                            sim2= util.dot_score(embed, embedCol)
                            goldPOS[x[0]-1]= [x[0]-1, f'Table:{tables[0][torch.argmax(sim1)]}', f'Column:{tableColumns[tables[0][torch.argmax(sim1)]][1:][torch.argmax(sim2)]}']
            
    except:pass
    return 


memory["doubleOfExceptionFirst"]= None
def doubleOfExceptionFirst(goldPOS, memory):
    try:
        if (memory["ofCount"]>=2) & (memory["doubleOfExceptionBoth"]==None)  & (memory["doubleOfExceptionSecond"]==None):
            if (memory ["interIDXdoubleOF"]==None):
                memory["interIDXdoubleOF"] = memory["ofIdxSecond"]
            if (memory["interIDXdoubleOF"]> memory["ofIdxSecond"]):
                memory["interIDXdoubleOF"] = memory["ofIdxSecond"]     
        for x in goldPOS: 
            if (x[0]>memory["idxfirst"]) & (x[0]<= memory["interIDXdoubleOF"]) & (x[1] not in to_ignore) & (x[2] in ["NOUN", "PatternEmbed"]):
                embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                embedTables= SimilarityModel.encode(tables[0], convert_to_tensor=True, normalize_embeddings=True)
                sim= util.dot_score(embed, embedTables)
                table= tables[0][torch.argmax(sim)]
                if torch.max(sim)>0.85:
                    goldPOS[x[0]-1]= [x[0]-1, f'Table:{table}', "Main table between double 'of"]
                    memory["idxSim1Table"]= table
                    memory["mainTableBetweenDoubleOf"]= 1
                    memory["doubleOfExceptionFirst"]=1
        if memory["mainTableBetweenDoubleOf"]==1:
            for x in goldPOS:
                if (x[0]>=memory["interIDXdoubleOF"]) & (x[0]<=memory["interIDXSecondOF"]) & (x[1] not in to_ignore) & (x[2] in ["NOUN", "PatternEmbed"]):
                    embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                    embedCols= SimilarityModel.encode(tableColumns[memory["idxSim1Table"]][1:], convert_to_tensor=True, normalize_embeddings=True)
                    sim2= util.dot_score(embed, embedCols)
                    if torch.max(sim2)>=0.55:
                        goldPOS[x[0]-1]= [x[0],f'Table:{memory["idxSim1Table"]}', f'Column:{tableColumns[memory["idxSim1Table"]][1:][torch.argmax(sim2)]}']
                    if (torch.max(sim2)<0.55) & (torch.max(sim2)>0.40):
                        sim3= util.dot_score(embed, ColumnEmbeddings[0])
                        differentTable= tables[0][torch.argmax(sim3)]
                        embedCols= SimilarityModel.encode(tableColumns[differentTable][1:], convert_to_tensor=True, normalize_embeddings=True)
                        sim4= util.dot_score(embed, embedCols)
                        if (x[2]== "PatternEmbed") & (torch.max(sim3)>=0.6):
                            splitted_string= x[1].split("_")
                            first= SimilarityModel.encode(splitted_string[0], convert_to_tensor=True, normalize_embeddings=True)
                            last= SimilarityModel.encode(splitted_string[-1], convert_to_tensor=True, normalize_embeddings=True)
                            simFirst= util.dot_score(first, embedCols)
                            simLast= util.dot_score(last, embedCols)
                            if (torch.max(simFirst)>torch.max(simLast)) & (torch.max(simFirst)>torch.max(sim4)):
                                z= simFirst
                            if (torch.max(simLast)>torch.max(simFirst)) & (torch.max(simLast)>torch.max(sim4)):
                                z= simLast  
                            if (torch.max(sim4)>torch.max(simFirst)) & (torch.max(sim4)>torch.max(simLast)):
                                z= sim4
                            goldPOS[x[0]-1]= [x[0], f'Table:{differentTable}', f'Column:{tableColumns[differentTable][1:][torch.argmax(z)]}']                         
                        if (torch.max(sim4)>0.55) & (x[2] != "PatternEmbed"):
                            goldPOS[x[0]-1]= [x[0],f'Table:{differentTable}', f'Column:{tableColumns[differentTable][1:][torch.argmax(sim4)]}']
                        else:pass
                    else:pass
            for x in goldPOS:
                if (x[0]<= memory["idxfirst"]) & (x[1] not in to_ignore) & (x[2] in ["NOUN", "PatternEmbed"]):
                    embed = SimilarityModel.encode(x[1], convert_to_tensor=True,normalize_embeddings=True)
                    embedCols= SimilarityModel.encode(tableColumns[memory["idxSim1Table"]][1:], convert_to_tensor=True, normalize_embeddings=True)
                    sim2= util.dot_score(embed, embedCols)
                    if torch.max(sim2)>=0.55:
                        goldPOS[x[0]-1]= [x[0],f'Table:{memory["idxSim1Table"]}', f'Column:{tableColumns[memory["idxSim1Table"]][1:][torch.argmax(sim2)]}']
                    if (torch.max(sim2)<0.55):
                        sim3= util.dot_score(embed, ColumnEmbeddings[0])
                        differentTable= tables[0][torch.argmax(sim3)]
                        embedCols= SimilarityModel.encode(tableColumns[differentTable][1:], convert_to_tensor=True,normalize_embeddings=True)
                        sim4= util.dot_score(embed, embedCols)
                        if torch.max(sim4) >=0.5:
                            goldPOS[x[0]-1]= [x[0],f'Table:{differentTable}', f'Column:{tableColumns[differentTable][1:][torch.argmax(sim4)]}']
    except:pass
    return


def doubleOfExceptionNeither(goldPOS, memory):
    if (memory["ofCount"]>=2) & (memory["doubleOfExceptionBoth"]==None)  & (memory["doubleOfExceptionSecond"]==None) & (memory["doubleOfExceptionFirst"]==None) :
        try:
            for x in goldPOS:
                if (x[0]<= memory["idxfirst"])  & (x[1] not in to_ignore) & (x[2] in ["NOUN", "PatternEmbed"]):
                    embed= SimilarityModel.encode(x[1], convert_to_tensor= True, normalize_embeddings=True)
                    embedAllCols= SimilarityModel.encode(AllColumns, convert_to_tensor= True, normalize_embeddings=True)
                    embedTables = SimilarityModel.encode(tables[0], convert_to_tensor=True, normalize_embeddings=True)
                    simTable= util.dot_score(embed, embedTables)
                    sim1= util.dot_score(embed, embedAllCols)
                    table= tables[0][torch.argmax(simTable)]
                    if (torch.max(sim1)>=0.65) & (torch.max(simTable)>0.75):
                        column= AllColumns[torch.argmax(sim1)]
                        table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                        goldPOS[x[0]-1]= [x[0], f"Table:{table}", f"Column:{column}"]
                    if (torch.max(sim1)>=0.65) & (torch.max(simTable)<0.75):
                        column= AllColumns[torch.argmax(sim1)]
                        table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                        goldPOS[x[0]-1]= [x[0], f"Table:{table}", f"Column:{column}"]
                    if (torch.max(sim1)<0.65) & (torch.max(simTable)<0.75):
                        pass
                    if (torch.max(sim1)<0.65) & (torch.max(simTable)>=0.75):
                        goldPOS[x[0]-1]= [x[0], f"Table:{table}", "Reference table before of"]
        except:pass
    return



def vocabularyInsert(goldPOS, memory):
    memory["each"]= False
    for x in goldPOS:
        if x[1] in bigger:
            goldPOS[x[0]-1]= [x[0], ">", "SQL function"]
        if x[1] in lower:
            goldPOS[x[0]-1]= [x[0], "<", "SQL function"]
        if x[1] in avg:
            goldPOS[x[0]-1]= [x[0], "AVG", "SQL function"]
        if x[1] in maximum:
            goldPOS[x[0]-1]= [x[0], "MAX", "SQL function"]
        if x[1] in minimum:
            goldPOS[x[0]-1]= [x[0], "MIN", "SQL function"]
        if x[1] in between:
            goldPOS[x[0]-1]= [x[0], "BETWEEN", "SQL function"]
        if x[1] in ["inside"]:
            goldPOS[x[0]-1]= [x[0],"=", "SQL function"]
        if x[1] in ["outside"]:
            goldPOS[x[0]-1]= [x[0], "!=", "SQL function"]
        if x[1] in ["distinct", "unique", "different"]:
            for c in goldPOS: 
                if c[1] == "each":
                    memory["each"]= True
            if memory["each"]== False:
                goldPOS[x[0]-1]= [x[0], "DISTINCT", "SQL function"]
        if x[1] in sum:
            goldPOS[x[0]-1]= [x[0], "SUM", "SQL function"]
    return


def notIdentifiedOF(goldPOS, memory):
    if memory["ofCount"]>0:
        for x in goldPOS:
            if x[1].startswith("Table"):
                memory["not_identified"]= 0
                break
            else: 
                memory["not_identified"]=1
    return


def specialCaseWords(goldPOS, memory):
    if memory["ofCount"]>0:
        for x in goldPOS:
            if x[1] in special_cases:
                if x[1] in ["older"]:
                    embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                    if memory["idxSim1Table"]!=None:
                        tableCols= tableColumns[memory["idxSim1Table"]][1:]
                        embedTableCols= SimilarityModel.encode(tableCols, convert_to_tensor=True, normalize_embeddings=True)
                        sim1 = util.dot_score(embed, embedTableCols)
                        if torch.max(sim1)>=0.45:
                            goldPOS[x[0]-1]= [x[0], f'Table:{memory["idxSim1Table"]}', f"Column:{tableCols[torch.argmax(sim1)]}"]
                            goldPOS.insert(x[0],[x[0], '>', "SQL function"])
                            i=1
                            for z in goldPOS:
                                goldPOS[i-1]= [i, z[1], z[2]]
                                i+=1
                        if torch.max(sim1)<0.45:
                            embedAllCols= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                            sim2 =util.dot_score(embed, embedAllCols)
                            column = AllColumns[torch.argmax(sim2)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[x[0]-1]= [x[0], f'Table:{table}', f'Column:{column}']
                            goldPOS.insert(x[0],[x[0], '>', "SQL function"])
                            i=1
                            for z in goldPOS:
                                goldPOS[i-1]= [i, z[1], z[2]]
                                i+=1
                    if memory["idxSim1Table"] == None:
                        embedAllCols= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                        sim1 = util.dot_score(embed, embedAllCols)
                        column = AllColumns[torch.argmax(sim1)]
                        table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                        goldPOS[x[0]-1]= [x[0], f'Table:{table}', f"Column:{column}"]
                        goldPOS.insert(x[0]-1,[x[0], '>', "SQL function"])
                        i = 1
                        for z in goldPOS:
                            goldPOS[i-1]= [i, z[1], z[2]]
                            i+=1
                if x[1] in ["younger"]:
                    embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                    if memory["idxSim1Table"]!=None:
                        tableCols= tableColumns[memory["idxSim1Table"]][1:]
                        embedTableCols= SimilarityModel.encode(tableCols, convert_to_tensor=True, normalize_embeddings=True)
                        sim1 = util.dot_score(embed, embedTableCols)
                        if torch.max(sim1)>=0.45:
                            goldPOS[x[0]-1]= [x[0], f'Table:{memory["idxSim1Table"]}', f"Column:{tableCols[torch.argmax(sim1)]}"]
                            goldPOS.insert(x[0],[x[0], '<', "SQL function"])
                            i=1
                            for z in goldPOS:
                                goldPOS[i-1]= [i, z[1], z[2]]
                                i+=1
                            
                        if torch.max(sim1)<0.45:
                            embedAllCols= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                            sim2 =util.dot_score(embed, embedAllCols)
                            column = AllColumns[torch.argmax(sim2)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[x[0]-1]= [x[0], f'Table:{table}', f'Column:{column}']
                            goldPOS.insert(x[0],[x[0], '<', "SQL function"])
                            i=1
                            for z in goldPOS:
                                goldPOS[i-1]= [i, z[1], z[2]]
                                i+=1
                    if memory["idxSim1Table"] == None:
                        embedAllCols= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                        sim1 = util.dot_score(embed, embedAllCols)
                        column = AllColumns[torch.argmax(sim1)]
                        table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                        goldPOS[x[0]-1]= [x[0], f'Table:{table}', f"Column:{column}"]
                        goldPOS.insert(x[0]-1,[x[0], '<', "SQL function"])
                        i = 1
                        for z in goldPOS:
                            goldPOS[i-1]= [i, z[1], z[2]]
                            i+=1
                        
                if x[1] in ["youngest"]:
                    embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                    if memory["idxSim1Table"]!=None:
                        tableCols= tableColumns[memory["idxSim1Table"]][1:]
                        embedTableCols= SimilarityModel.encode(tableCols, convert_to_tensor=True,normalize_embeddings=True)
                        sim1 = util.dot_score(embed, embedTableCols)             
                        if torch.max(sim1)>=0.45:
                            goldPOS[x[0]-1]= [x[0], f'Table:{memory["idxSim1Table"]}', f"Column:{tableCols[torch.argmax(sim1)]}"]
                            goldPOS.insert(x[0]-1,[x[0], 'MIN', "SQL function"])
                            i=1
                            for z in goldPOS:
                                goldPOS[i-1]= [i, z[1], z[2]]
                                i+=1
                        if torch.max(sim1)<0.45:
                            embedAllCols= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                            sim2 =util.dot_score(embed, embedAllCols)
                            column = AllColumns[torch.argmax(sim2)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[x[0]-1]= [x[0], f'Table:{table}', f'Column:{column}']
                            goldPOS.insert(x[0]-1,[x[0], 'MIN', "SQL function"])
                            i=1
                            for z in goldPOS:
                                goldPOS[i-1]= [i, z[1], z[2]]
                                i+=1
                    if memory["idxSim1Table"] == None:
                        embedAllCols= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                        sim1 = util.dot_score(embed, embedAllCols)
                        column = AllColumns[torch.argmax(sim1)]
                        table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                        goldPOS[x[0]-1]= [x[0], f'Table:{table}', f"Column:{column}"]
                        goldPOS.insert(x[0]-1,[x[0], 'MIN', "SQL function"])
                        i = 1
                        for z in goldPOS:
                            goldPOS[i-1]= [i, z[1], z[2]]
                            i+=1
                if x[1] in ["oldest"]:
                    embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                    if memory["idxSim1Table"]!= None:
                        tableCols= tableColumns[memory["idxSim1Table"]][1:]
                        embedTableCols= SimilarityModel.encode(tableCols, convert_to_tensor=True, normalize_embeddings=True)
                        sim1 = util.dot_score(embed, embedTableCols)
                        if torch.max(sim1)>=0.45:
                            goldPOS[x[0]-1]= [x[0], f'Table:{memory["idxSim1Table"]}', f"Column:{tableCols[torch.argmax(sim1)]}"]
                            goldPOS.insert(x[0]-1,[x[0], 'MAX', "SQL function"])
                            i = 1
                            for z in goldPOS:
                                goldPOS[i-1]= [i, z[1], z[2]]
                                i+=1
                        if torch.max(sim1)<0.45:
                            embedAllCols= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                            sim2 =util.dot_score(embed, embedAllCols)
                            column = AllColumns[torch.argmax(sim2)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[x[0]-1]= [x[0], f'Table:{table}', f'Column:{column}']
                            goldPOS.insert(x[0]-1,[x[0], 'MAX', "SQL function"])
                            i=1
                            for z in goldPOS:
                                goldPOS[i-1]= [i, z[1], z[2]]
                                i+=1
                    if memory["idxSim1Table"] == None:
                        embedAllCols= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                        sim1 = util.dot_score(embed, embedAllCols)
                        column = AllColumns[torch.argmax(sim1)]
                        table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                        goldPOS[x[0]-1]= [x[0], f'Table:{table}', f"Column:{column}"]
                        goldPOS.insert(x[0]-1,[x[0], 'MAX', "SQL function"])
                        i = 1
                        for z in goldPOS:
                            goldPOS[i-1]= [i, z[1], z[2]]
                            i+=1

            
    return

def yearOF(goldPOS, memory):
    if memory["ofCount"]>0:
        for x in goldPOS:
            if (x[0]>memory["idxfirst"]) & (x[2].startswith("Column")):
                break
            if (x[0]>memory["idxfirst"]) & (x[2]== "NUM") & (len(x[1])==4):
                match = re.match(r'.*([1-9][0-9]{3})', x[1])
                if match != None:
                    num = match.group(1)
                    embed = SimilarityModel.encode(num, convert_to_tensor=True,normalize_embeddings=True)
                    embedAllCols= SimilarityModel.encode(AllColumns,  convert_to_tensor=True,normalize_embeddings=True)
                    sim = util.dot_score(embed, embedAllCols)
                    if torch.max(sim)>0.45:
                        column = AllColumns[torch.argmax(sim)]
                        table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                        goldPOS.insert(x[0]-1,[x[0], f'Table:{table}', f"Column:{column}"])
                        i=1
                        for z in goldPOS:
                            goldPOS[i-1]= [i, z[1], z[2]]
                            i+=1
                        break
    return




def TruncateAggColumns(memory, goldPOS):
    if memory["ofCount"]>0:
        idx= []
        for x in goldPOS:
            if (x[2] =="SQL function") & (x[1] in ["AVG", "MAX", "MIN", "SUM"]) :
                idx.append(x[0]-1)
        for index in idx:
            for x in goldPOS:
                if ((x[0]-1)>=index ) & (x[2] in ["NOUN", "PatternEmbed"]) & (x[1] not in to_ignore):
                    if goldPOS[index][1]== "AVG":
                        new_word= f"average_{x[1]}"
                        new_word2= f"mean_{x[1]}"
                    if goldPOS[index][1]== "SUM":
                        new_word= f"total_{x[1]}"
                        new_word2= f"sum_{x[1]}"
                    if goldPOS[index][1] not in ["AVG", "SUM"]: 
                        break
                    embedingOriginalWord= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                    embedNewWord=SimilarityModel.encode(new_word, convert_to_tensor=True, normalize_embeddings=True)
                    embedAllcolumns= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                    simOriginal= util.dot_score(embedingOriginalWord, embedAllcolumns)
                    simNewWord= util.dot_score(embedNewWord, embedAllcolumns)
                    if torch.max(simOriginal)> torch.max(simNewWord) :
                        pass
                    if torch.max(simNewWord)> torch.max(simOriginal):
                        embedNewWord2= SimilarityModel.encode(new_word2, convert_to_tensor=True, normalize_embeddings=True)
                        simNewWord2= util.dot_score(embedAllcolumns, embedNewWord2)
                        if torch.max(simNewWord2)> torch.max(simNewWord):
                            goldPOS[index]= [index, "ignore", "Delted by Aggregation Truncation"]
                            column = AllColumns[torch.argmax(simNewWord2)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[x[0]-1]= [x[0], f"Table:{table}", f"Column:{column}"]
                        if torch.max(simNewWord)> torch.max(simNewWord2):
                            goldPOS[index]= [index, "ignore", "Delted by Aggregation Truncation"]
                            column = AllColumns[torch.argmax(simNewWord)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[x[0]-1]= [x[0], f"Table:{table}", f"Column:{column}"]
                    break
                if ((x[0]-1)>= index ) & (x[2].startswith("Column:")):
                    if goldPOS[index][1]== "AVG":
                        new_word= f"average_{x[2][7:]}"
                        new_word2= f"mean_{x[2][7:]}"
                    if goldPOS[index][1]== "SUM":
                        new_word= f"total_{x[2][7:]}"
                        new_word2= f"sum_{x[2][7:]}"
                    if goldPOS[index][1] not in ["SUM", "AVG"]: 
                        new_word= f"{goldPOS[index][1]}_{x[1][6:]}"
                        new_word2= f"{goldPOS[index][1]}_{x[1][6:]}"
                    embedAllcolumns = SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                    originalWord= memory["goldPOScopy"][x[0]-1][1]
                    embedingOriginalWord= SimilarityModel.encode(originalWord, convert_to_tensor=True,normalize_embeddings=True)
                    embedingNewWord= SimilarityModel.encode(new_word, convert_to_tensor=True,normalize_embeddings=True)
                    simOriginalWord= util.dot_score(embedingOriginalWord, embedAllcolumns)
                    simNewWord= util.dot_score(embedingNewWord, embedAllcolumns)
                    if torch.max(simOriginalWord)> torch.max(simNewWord):
                        pass
                    if torch.max(simNewWord)> torch.max(simOriginalWord):
                        embedNewWord2= SimilarityModel.encode(new_word2, convert_to_tensor=True,normalize_embeddings=True)
                        simNewWord2= util.dot_score(embedAllcolumns, embedNewWord2)
                        if torch.max(simNewWord2)> torch.max(simNewWord):
                            goldPOS[index]= [index, "ignore", "Delted by Aggregation Truncation"]
                            column = AllColumns[torch.argmax(simNewWord2)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[x[0]-1]= [x[0], f"Table:{table}", f"Column:{column}"]
                        if torch.max(simNewWord)> torch.max(simNewWord2):
                            goldPOS[index]= [index, "ignore", "Delted by Aggregation Truncation"]
                            column = AllColumns[torch.argmax(simNewWord)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[x[0]-1]= [x[0], f"Table:{table}", f"Column:{column}"]
                    break
    return


def continuedPatternsAfterOf(goldPOS, memory):
    try:
        if (memory["ofCount"]==1):
            for x in goldPOS:
                if (x[0]>memory["idxfirst"]) & (x[2] in ["NOUN", "PatternEmbed"])  & (x[1] not in to_ignore) :
                    embedTables= SimilarityModel.encode(tables[0],convert_to_tensor=True, normalize_embeddings=True)
                    embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                    tableSim= util.dot_score(embed, embedTables)
                    if memory["idxSim1Table"]!=None:
                        table= memory["idxSim1Table"]
                        cols= tableColumns[table][1:]
                        embedCols= SimilarityModel.encode(cols, convert_to_tensor=True, normalize_embeddings=True)
                        sim1 = util.dot_score(embed, embedCols)
                        allColumnsEmbeding= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                        colSim= util.dot_score(embed, allColumnsEmbeding)
                        if torch.max(tableSim)<=0.8:
                            if (torch.max(colSim)>=0.75) & (torch.max(sim1)<=0.55):
                                column = AllColumns[torch.argmax(colSim)]
                                table1= [key for key, value_list in tableColumns.items() if column in value_list][0]
                                goldPOS[x[0]-1]= [x[0], f'Table:{table1}', f"Column:{column}"]
                            if (torch.max(sim1)>=0.55) &(torch.max(colSim)<0.75):
                                goldPOS[x[0]-1]=[x[0], f'Table:{table}', f"Column:{tableColumns[table][1:][torch.argmax(sim1)]}"]
                            if (torch.max(sim1)>=0.55) & (torch.max(colSim)>0.75):
                                goldPOS[x[0]-1]=[x[0], f'Table:{table}', f"Column:{tableColumns[table][1:][torch.argmax(sim1)]}"]
                            if (torch.max(sim1)<0.55 )  & (torch.max(colSim)<0.75):
                                allColumnsEmbeding= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                                colSim= util.dot_score(embed, allColumnsEmbeding)
                                if torch.max(colSim)>=0.45:
                                    column = AllColumns[torch.argmax(colSim)]
                                    table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                                    goldPOS[x[0]-1]= [x[0], f'Table:{table}', f'Column:{column}']
                        if (torch.max(tableSim)>0.8) & (torch.max(sim1)>0.8):
                            goldPOS[x[0]-1]= [x[0], f"Table:{table}", f"Column:{tableColumns[table][1:][torch.argmax(sim1)]}"]
                        if (torch.max(tableSim)>=0.8) & (torch.max(colSim)<0.8):
                            goldPOS[x[0]-1]= [x[0], f'Table:{tables[0][torch.argmax(tableSim)]}', "Reference Table after 'of'"]
                        if (torch.max(tableSim)>=0.8) & (torch.max(colSim)>0.8):
                            goldPOS[x[0]-1]= [x[0], f'Table:{tables[0][torch.argmax(tableSim)]}', "Reference Table after 'of'"]
                    if (memory["idxSim1Table"]== None):
                        embedAllCols= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                        colSim= util.dot_score(embed, embedAllCols)
                        if torch.max(tableSim)<=0.8:
                            if torch.max(colSim)>=0.45:
                                column = AllColumns[torch.argmax(colSim)]
                                table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                                goldPOS[x[0]-1]= [x[0], f'Table:{table}' , f'Column:{column}']
                        if (torch.max(tableSim)>=0.8) & (torch.max(colSim)>=0.8):
                            column = AllColumns[torch.argmax(colSim)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[x[0]-1]= [x[0], f"Table:{table}", f"Column:{column}"]
                        if (torch.max(tableSim)>=0.8) & (torch.max(colSim)<0.8):
                            goldPOS[x[0]-1]= [x[0], f'Table:{tables[0][torch.argmax(tableSim)]}', "Reference Table before 'of'"]
    except:pass
    return


def continuedPatternsBeforeOf(goldPOS, memory):
    try: 
        if (memory["ofCount"]==1):
            for x in goldPOS:
                if (x[0]<=memory["idxfirst"]) & (x[2] in ["NOUN", "PatternEmbed"])  & (x[1] not in to_ignore) :
                    embedTables= SimilarityModel.encode(tables[0],convert_to_tensor=True, normalize_embeddings=True)
                    embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                    tableSim= util.dot_score(embed, embedTables)
                    if memory["idxSim1Table"]!=None:
                        table= memory["idxSim1Table"]
                        cols= tableColumns[table][1:]
                        embedCols= SimilarityModel.encode(cols, convert_to_tensor=True, normalize_embeddings=True)
                        sim1 = util.dot_score(embed, embedCols)
                        if torch.max(tableSim)<=0.8:
                            if (torch.max(tableSim)<=0.8):
                                if (torch.max(sim1)>=0.49) : 
                                    goldPOS[x[0]-1]=[x[0], f'Table:{table}', f"Column:{tableColumns[table][1:][torch.argmax(sim1)]}"]
                                if (torch.max(sim1)<0.49) :
                                    allColumnsEmbeding= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                                    colSim= util.dot_score(embed, allColumnsEmbeding)
                                    if torch.max(colSim)>=0.45:
                                        column = AllColumns[torch.argmax(colSim)]
                                        table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                                        goldPOS[x[0]-1]= [x[0], f'Table:{table}', f'Column:{column}']
                        if (torch.max(tableSim)>=0.8) & (torch.max(sim1)>=0.8):
                            goldPOS[x[0]-1]= [x[0], f"Table:{table}", f"Column:{tableColumns[table][1:][torch.argmax(sim1)]}"]
                        if (torch.max(tableSim)>=0.8) & (torch.max(sim1)<0.8):
                            goldPOS[x[0]-1]= [x[0], f'Table:{tables[0][torch.argmax(tableSim)]}', "Reference Table before 'of'"]
                    if (memory["idxSim1Table"]== None):
                        allColumnsEmbeding= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                        colSim= util.dot_score(embed, allColumnsEmbeding)
                        if torch.max(tableSim)<=0.8:
                            if torch.max(colSim)>=0.45:
                                column = AllColumns[torch.argmax(colSim)]
                                table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                                goldPOS[x[0]-1]= [x[0], f'Table:{table}' , f'Column:{column}']
                        if (torch.max(tableSim)>=0.8) & (torch.max(colSim)>=0.8):
                            column = AllColumns[torch.argmax(colSim)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[x[0]-1]= [x[0], f"Table:{table}", f"Column:{column}"]
                        if (torch.max(tableSim)>=0.8) & (torch.max(colSim)<0.8):
                            goldPOS[x[0]-1]= [x[0], f'Table:{tables[0][torch.argmax(tableSim)]}', "Reference Table before 'of'"]

    except:pass
    return


def continuedPatternsAfterDoubleOf(goldPOS, memory):
    if memory["ofCount"]==2:
        try:
            for x in goldPOS:
                if (x[0]>memory["ofIdxSecond"]) & (x[2] in ["PatternEmbed", "NOUN"])&(x[1] not in to_ignore):
                    try:
                        table = memory["idxSim1Table"]
                        cols = tableColumns[table][1:]
                        embedCols= SimilarityModel.encode(cols, convert_to_tensor=True, normalize_embeddings=True)
                        embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                        sim= util.dot_score(embed, embedCols)
                        embedAllcols= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                        simAllCols= util.dot_score(embed, embedAllcols)
                        if (torch.max(sim)<=0.75) & (torch.max(simAllCols)>=0.75):
                            column = AllColumns[torch.argmax(simAllCols)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS [x[0]-1] = [x[0], f"Table:{table}", f"Column:{column}"]
                        if (torch.max(sim)>=0.75) & (torch.max(simAllCols)<=0.75):
                            goldPOS[x[0]-1]=[x[0], f'Table:{table}' , f'Column:{tableColumns[table][1:][torch.argmax(sim)]}']
                        if (torch.max(sim)>=0.75) & (torch.max(simAllCols)>=0.75):
                            goldPOS[x[0]-1]= [x[0], f"Table:{table}", f"Column:{tableColumns[table][1:][torch.argmax(sim)]}"]

                        if (torch.max(sim)<0.75) & ((torch.max(simAllCols)<=0.75)):
                            tablesEmbed= SimilarityModel.encode(tables[0],convert_to_tensor=True, normalize_embeddings=True)
                            TableSim= util.dot_score(embed, tablesEmbed)
                            if torch.max(TableSim)>=0.75:
                                goldPOS[x[0]-1]= [x[0], f'Table:{tables[0][torch.argmax(TableSim)]}', "Reference table after double 'of'"]
                            if torch.max(TableSim)<0.75:
                                AllColsEmbeding=SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                                colSim= util.dot_score(embed, AllColsEmbeding)
                                if torch.max(colSim)>0.4:
                                    column = AllColumns[torch.argmax(colSim)]
                                    table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                                    goldPOS[x[0]-1]= [x[0], f'Table:{table}', f"Column:{column}"]
                    except:pass
                    try:
                        table= memory["idxSim2Table"]
                        cols = tableColumns[table][1:]
                        embedCols= SimilarityModel.encode(cols, convert_to_tensor=True, normalize_embeddings=True)
                        embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                        sim= util.dot_score(embed, embedCols)
                        embedAllcols= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                        simAllCols= util.dot_score(embed, embedAllcols)
                        if (torch.max(sim)<=0.75) & (torch.max(simAllCols)>=0.75):
                            column = AllColumns[torch.argmax(simAllCols)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS [x[0]-1] = [x[0], f"Table:{table}", f"Column:{column}"]
                        if (torch.max(sim)>=0.75) & (torch.max(simAllCols)<=0.75):
                            goldPOS[x[0]-1]=[x[0], f'Table:{table}' , f'Column:{tableColumns[table][1:][torch.argmax(sim)]}']
                        if (torch.max(sim)>0.75) & (torch.max(simAllCols)>=0.75):
                            goldPOS[x[0]-1]=[x[0], f'Table:{table}' , f'Column:{tableColumns[table][1:][torch.argmax(sim)]}']
                        if (torch.max(sim)<=0.75) & (torch.max(simAllCols)<=0.75):
                            tablesEmbed= SimilarityModel.encode(tables[0],convert_to_tensor=True, normalize_embeddings=True)
                            TableSim= util.dot_score(embed, tablesEmbed)
                            if torch.max(TableSim)>=0.75:
                                goldPOS[x[0]-1]= [x[0], f'Table:{tables[0][torch.argmax(TableSim)]}', "Reference table after double 'of'"]
                            if torch.max(TableSim)<0.75:
                                AllColsEmbeding=SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                                colSim= util.dot_score(embed, AllColsEmbeding)
                                if torch.max(colSim)>0.4:
                                    column = AllColumns[torch.argmax(colSim)]
                                    table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                                    goldPOS[x[0]-1]= [x[0], f'Table:{table}', f"Column:{column}"]
                    except:pass
                    if (memory["idxSim1Table"]==None) & (memory["idxSim2Table"]== None):
                        embed = SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                        AllColsEmbeding=SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                        colSim= util.dot_score(embed, AllColsEmbeding)
                        if torch.max(colSim)>0.4:
                            column = AllColumns[torch.argmax(colSim)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[x[0]-1]= [x[0], f'Table:{table}', f"Column:{column}"]
        except:pass
                

    return


def continuedPatternsBeforeDoubleOf(goldPOS, memory):
    if memory["ofCount"]==2:
        try:
            for x in goldPOS:
                if (x[0]<memory["ofIdxSecond"]) & (x[2] in ["PatternEmbed", "NOUN"])&(x[1] not in to_ignore):
                    try:
                        table = memory["idxSim1Table"]
                        cols = tableColumns[table][1:]
                        embedCols= SimilarityModel.encode(cols, convert_to_tensor=True, normalize_embeddings=True)
                        embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                        sim= util.dot_score(embed, embedCols)
                        if torch.max(sim)>=0.52:
                            goldPOS[x[0]-1]=[x[0], f'Table:{table}' , f'Column:{tableColumns[table][1:][torch.argmax(sim)]}']
                        if torch.max(sim)<0.52:
                            tablesEmbed= SimilarityModel.encode(tables[0],convert_to_tensor=True, normalize_embeddings=True)
                            TableSim= util.dot_score(embed, tablesEmbed)
                            if torch.max(TableSim)>=0.65:
                                goldPOS[x[0]-1]= [x[0], f'Table:{tables[0][torch.argmax(TableSim)]}', "Reference table after double 'of'"]
                            if torch.max(TableSim)<0.65:
                                AllColsEmbeding=SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                                colSim= util.dot_score(embed, AllColsEmbeding)
                                if torch.max(colSim)>0.4:
                                    column = AllColumns[torch.argmax(colSim)]
                                    table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                                    goldPOS[x[0]-1]= [x[0], f'Table:{table}', f"Column:{column}"]
                    except:pass
                    try:
                        table= memory["idxSim2Table"]
                        cols = tableColumns[table][1:]
                        embedCols= SimilarityModel.encode(cols, convert_to_tensor=True, normalize_embeddings=True)
                        embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                        sim= util.dot_score(embed, embedCols)
                        if torch.max(sim)>=0.52:
                                goldPOS[x[0]-1]=[x[0], f'Table:{table}' , f'Column:{tableColumns[table][1:][torch.argmax(sim)]}']
                        if torch.max(sim)<0.52:
                            tablesEmbed= SimilarityModel.encode(tables[0],convert_to_tensor=True, normalize_embeddings=True)
                            TableSim= util.dot_score(embed, tablesEmbed)
                            if torch.max(TableSim)>=0.8:
                                goldPOS[x[0]-1]= [x[0], f'Table:{tables[0][torch.argmax(TableSim)]}', "Reference table after double 'of'"]
                            if torch.max(TableSim)<0.8:
                                AllColsEmbeding=SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                                colSim= util.dot_score(embed, AllColsEmbeding)
                                if torch.max(colSim)>0.4:
                                    column = AllColumns[torch.argmax(colSim)]
                                    table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                                    goldPOS[x[0]-1]= [x[0], f'Table:{table}', f"Column:{column}"]
                    except:pass
                    if (memory["idxSim1Table"]==None) & (memory["idxSim2Table"]== None):
                        embed = SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                        AllColsEmbeding=SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                        colSim= util.dot_score(embed, AllColsEmbeding)
                        if torch.max(colSim)>0.4:
                            column = AllColumns[torch.argmax(colSim)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[x[0]-1]= [x[0], f'Table:{table}', f"Column:{column}"]
        except:pass
    return


def swapNumberTable(goldPOS, memory):
    if memory["ofCount"]>0:
        Value= False
        for x in goldPOS:
            if (x[0]<=memory["idxfirst"]) & (x[1].startswith("Table:")):
                Value=True
        if Value==False:
            if goldPOS[memory["idxfirst"]+1][2].startswith("Column:"):
                goldPOS.insert((memory["idxfirst"]), goldPOS[memory["idxfirst"]+1])
                goldPOS[memory["idxfirst"]+2]= [memory["idxfirst"]+2, goldPOS[memory["idxfirst"]+2][1], "Reference Table after 'of'"]
                for i, sublist in enumerate(goldPOS):
                    sublist[0] = i + 1

    return

def idRecognizer(goldPOS, memory):
    table = None
    for x in goldPOS:
        if x[1].startswith("Table:"):
            table= x[1][6:]
            break
    if table!= None:
        for x in goldPOS:
            if x[1] in["id", "ids"]:
                goldPOS[x[0]-1]= [x[0], f"Table:{table}", f"Column:{AllTableColumns[table][0]}"]

    return 






def SelectClauseOF(goldPOS, memory):
    memory["SelectClause"]= []
    memory["Tables"]= []
    cols=[]
    if memory["ofCount"]>0:
        for x in goldPOS:
            if (x[0]<= memory["idxfirst"]):
                if (x[2].startswith("Column:")) & (x[2][7:] not in cols):
                    column= str.lower(x[2][7:])
                    table= x[1][6:]
                    cols.append(column)
                    if column in  memory["SelectClause"]:
                        pass
                    else: memory["SelectClause"].append(column)
                    if table not in memory["Tables"]:
                        memory["Tables"].append(table)
            if (x[0]> memory["idxfirst"]):
                if (x[2] in ["VERB"]) | (x[1] in ["which", "whose", "where", "what", "with", "from"]):
                    break
                if x[1]== "and":
                    for c in goldPOS:
                        if (c[0]>x[0])&(c[2].startswith("Column")):
                            if (str.lower(f"{c[2][7:]}") in memory["SelectClause"]) | (f"{c[1][6:]}.{str.lower(c[2][7:])}" in memory["SelectClause"]):
                                pass
                            else:
                                memory["SelectClause"].append(f"{c[1][6:]}.{str.lower(c[2][7:])}")
                            break            
        
    
        if len(memory["Tables"])>1:
            i = 0
            while i<len(memory["Tables"]):
                memory["SelectClause"][i]= f'{memory["Tables"][i]}.{memory["SelectClause"][i]}'
                i+=1
        if len(memory["Tables"])==1:
            memory["SelectClause"][0]= f'{memory["Tables"][0]}.{memory["SelectClause"][0]}'

        for x in goldPOS:
            if (x[0]> memory["idxfirst"]) & (x[2].startswith("Column:") | (x[2].startswith("Reference")) | ((x[2].startswith("Main")))):
                if x[1][:6] not in memory["Tables"]:
                    memory["Tables"].append(x[1][6:])
            
    return


def andOrWhere(goldPOS, memory):
    if memory["ofCount"]>0:
        memory["or"] = False
        for x in goldPOS:
            if (x[0]>memory["idxfirst"]) & (x[1].startswith("Table")| (x[2] == "PROPN")) :
                for c in goldPOS:
                    if (c[0]>=x[0]) & (c[1]in ["or", "either"]):
                        memory["or"] = True
                        break

    return



def WhereClauseOF(goldPOS, memory):
    memory["WhereClause"]= ["WHERE"]
    if memory["ofCount"]>0:
        for x in goldPOS:
            where= None
            values= []
            if (x[0]> memory["idxfirst"]) & (x[2] == "SQL function") & (x[1] not in ["=", "!="]):
                if goldPOS[x[0]][1] in ["!","?", "."]:
                    break
                if goldPOS[x[0]-2][1] == "at":
                    continue
                if goldPOS[x[0]-2][1] in having:
                    break
                if goldPOS[x[0]-2][2] == "NUM":
                    break
                if x[1] == "BETWEEN":
                    break
                for z in goldPOS:
                    # if (z[0]>memory["idxfirst"]) & (z[0]< x[0]) &( z[1] in ["which", "whose", "where", "what", "with"]):
                    #     break
                    if (z[0]>memory["idxfirst"]) & (z[0]< x[0]) & (z[1].startswith("Table:")) & (z[2].startswith("Column:")) & (f"{z[1][6:]}.{z[2][7:]}" not in memory["WhereClause"]):
                        table= z[1][6:]
                        column = z[2][7:]
                        where= f"{table}.{column}"
                    if (z[0]>x[0]) :
                        if z[2] == "SQL function":
                            break
                    if (z[0]>x[0]) & (z[2] in ["NUM", "Value"]):
                        values.append(z[1])
                if (len(values)==1) & (where != None):
                    memory["WhereClause"].append(f"{where}")
                    memory["WhereClause"].append(f"{x[1]} '{values[0]}' ")
                    values.pop(0)

                if where == None:
                    for z in goldPOS:
                        if (z[0]>=x[0]) & (z[1].startswith("Table:")) & (z[2].startswith("Column:")):
                            table= z[1][6:]
                            column = z[2][7:]
                            where= f"{table}.{column}"
                            break
                    if where is not None:
                        memory["WhereClause"].append(where)
                if (len(memory["WhereClause"])==2) & (len(values)>=1):
                    flag= False
                    for z in memory["WhereClause"]:
                        if ("MAX" in z) or ("MIN" in z) or ("SUM" in z):
                            flag=True
                            break
                    if flag== False:
                        memory["WhereClause"].append(x[1])
                        memory["WhereClause"].append(values[0])
                if len(memory["WhereClause"])<3 :
                    if (where !=None ) & (memory["WhereClause"][-1]!= "WHERE"):
                        memory["SelectClause"].append(f"{x[1]}({memory['WhereClause'][-1]})")
                        memory["WhereClause"]= ["WHERE"]
                    if where== None:
                      pass
                                
        if len(memory["WhereClause"])==1:
            func = None
            internalFlagValue= False
            internalFlagColumn= False
            for c in goldPOS:
                where= None
                values= []

                if (c[0]>memory["idxfirst"]) & (c[2] in  ["Value"]):
                    val = c[1]
                    validx= c[0]
                    internalFlagValue= True
                if (c[0]>memory["idxfirst"]) & (c[2] == "SQL function"):
                    break
                if (c[0]> memory["idxfirst"]) & (c[2].startswith("Column:")):
                    table= c[1][6:]
                    column = c[2][7:]
                    where= f"{table}.{column}"
                    whereInternalFlag= f"{table}.{column}"
                    internalFlagColumn= True
                    for y in goldPOS:
                        if (y[0]>c[0]) & (y[1]=="BETWEEN"):
                            break
                        if (y[0]>c[0])&(y[2].startswith("Column:")):
                            break
                        if(y[0]>c[0])& (y[2] in ["NUM", "Value"]):
                            values.append(y[1])
                            if goldPOS[y[0]-2][1] in having:
                                break
                            for z in goldPOS:
                                if ((z[0]>memory["idxfirst"]) & (z[0]<y[0]) & (z[1] in notIN)):
                                    func= "!="
                                if (z[0]>memory["idxfirst"]) & (z[0]<y[0]) & (z[1] in ["<", ">"]):
                                    func= f"{z[1]}"
                    if len(values)>=1:
                       for value in values:
                            if func==None:
                                memory["WhereClause"].append(f"{where}")
                                memory["WhereClause"].append(f"= '{value}' ")
                            if func!=None:
                                memory["WhereClause"].append(f"{where}")
                                memory["WhereClause"].append(f"{func} '{value}' ")

            if (internalFlagValue==True) & (internalFlagColumn==False):
                    for z in goldPOS:
                        if (z[0]<= validx) &( z[0] > memory["idxfirst"]) & (z[1] in notIN):
                            func= "!="
                    if memory["idxSim1Table"]!= None:
                        embed = SimilarityModel.encode(val, convert_to_tensor=True, normalize_embeddings=True)
                        embedMainTable = SimilarityModel.encode(tableColumns[memory["idxSim1Table"]], convert_to_tensor=True, normalize_embeddings=True)
                        sim= util.dot_score(embed, embedMainTable)
                        if torch.max(sim)>0.3:
                            table = memory["idxSim1Table"]
                            column = tableColumns[memory["idxSim1Table"]][torch.argmax(sim)]
                            where= f"{table}.{column}"
                            if func== None:
                                memory["WhereClause"].append(f"{where}")
                                memory["WhereClause"].append(f" = {val}")
                            if func!= None:
                                memory["WhereClause"].append(f"{where}")
                                memory["WhereClause"].append(f" {func} {val}")
                    if memory["idxSim1Table"]== None:
                        embed = SimilarityModel.encode(val, convert_to_tensor=True, normalize_embeddings=True)
                        embedAllColumns= SimilarityModel.encode(AllColumns,convert_to_tensor=True, normalize_embeddings=True)
                        sim= util.dot_score(embed, embedAllColumns)
                        if torch.max(sim)>0.4:
                            column = AllColumns[torch.argmax(sim)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            where= f"{table}.{column}"
                            if func== None:
                                memory["WhereClause"].append(f"{where}")
                                memory["WhereClause"].append(f" = {val}")
                            if func!= None:
                                memory["WhereClause"].append(f"{where}")
                                memory["WhereClause"].append(f" {func} {val}")
            if (internalFlagColumn==True) & (internalFlagValue==False):
                flag=False
                val = None
                for z in goldPOS:
                    if (z[0]>memory["idxfirst"]) & (z[2]=="SQL function"):
                        break
                    if (z[0]>memory["idxfirst"]) & (z[2] =="NUM"):
                        if goldPOS[z[0]-2][1] in having:
                            break
                        val = z[1]
                        validx= z[0]
                        internalFlagValue=True
                if internalFlagValue==True:
                    for z in goldPOS:
                            if (z[0]<= validx) &( z[0] > memory["idxfirst"]) & (z[1] in notIN):
                                func= "!="
                    for value in memory["WhereClause"]:
                            if val in value:
                                flag= True
                    if (func== None) & (flag==False):
                        memory["WhereClause"].append(f"{whereInternalFlag}")
                        memory["WhereClause"].append(f" = {val}")
                    if (func!= None )& (flag==False):
                        memory["WhereClause"].append(f"{whereInternalFlag}")
                        memory["WhereClause"].append(f" {func} {val}")
    return


def OrderByOF(goldPOS,memory):
    if memory["ofCount"]>0:
        orderBy= None
        memory["orderBy"]= []   
        flag= False   
        numIDX= None  
        memory["orderAggFlag"] = False
        memory["specialFlagOrder"]= False
        for x in goldPOS:
            if (x[0]>=memory["idxfirst"]) & (x[1] in order):
                orderBy= x[0]-1
                for z in goldPOS:
                    if (z[0]>x[0]) & (z[2]== "SQL function"):
                        memory["orderAggFlag"] = True
                        break
                    if (z[0]>x[0]) & ((z[1] in count) | (z[1] in ["number"])):
                        flag= True
                        numIDX=z[0]
                        break
                    if (z[0]>x[0]) & (z[1].startswith("Table:"))&(z[2].startswith("Column:")):
                        table= z[1][6:]
                        column= z[2][7:]
                        memory["orderBy"].append(f"ORDER BY {table}.{column}")
        if flag== True:
            for x in goldPOS:
                if (x[0]>=numIDX): 
                    if x[1].startswith("Table:") & (not x[2].startswith("Column:")):
                        memory["orderBy"].append(f"ORDER BY COUNT(*) ASC")
                        break
                    if x[1].startswith("Table:") & x[2].startswith("Column:"):
                        table= z[1][6:]
                        column= z[2][7:] 
                        memory["orderBy"].append(f"ORDER BY COUNT({table}.{column}) ASC")

        if len(memory["orderBy"])>0:
            for z in goldPOS:
                if (z[0]>memory["idxfirst"]) & (z[1] in ["ascending", "descending"]):
                    if z[1] == "ascending":
                        memory["orderBy"].append(f"ASC")
                    if z[1] == "descending":
                        memory["orderBy"].append(f"DESC")
        table = None
        column= None
        if (bool(orderBy) ==True ) & (len(memory["orderBy"])==0) & (memory["orderAggFlag"] == False):
            for x in goldPOS:
                if (x[2].startswith("Column:")) & (x[0]<=orderBy):
                    table = x[1][6:]
                    column = x[2][7:]
            for z in goldPOS:
                if  (z[0]>= memory["idxfirst"])  & (z[1] in ["alphabetical", "alphabetic", "alphabetically"]) :
                    memory["orderBy"]=[f"ORDER BY {table}.{column}"]
                if (z[0]>= memory["idxfirst"])  & (z[1] in ["ascending", "descending"]):
                    if z[1] in [ "ascending"]:
                        memory["orderBy"].append(f"ORDER BY  {table}.{column} ASC")
                    if z[1] in ["descending"]:
                        memory["orderBy"].append(f"ORDER BY {table}.{column} DESC")
            if len(memory["orderBy"])==0:
                for x in goldPOS:
                    if ((x[1] in count) or (x[1] in "number")):
                        flag= True
                        numIDX= x[0]
                    if ((x[2].startswith("Column:")) & (x[0]<=memory["idxfirst"])):
                        table=x[1][6:]
                        column= x[2][7:]
                if flag== True:
                    for x in goldPOS:
                        if (x[0]>=numIDX): 
                            if (x[1].startswith("Table:") & (not x[2].startswith("Column:"))):
                                memory["orderBy"].append(f"ORDER BY COUNT(*) ASC")
                                break
                            if (x[2].startswith("Column:")):
                                table= z[1][6:]
                                column= z[2][7:] 
                                memory["orderBy"].append(f"ORDER BY COUNT({table}.{column}) ASC")
            
                for z in goldPOS:
                    if (z[0]>= memory["idxfirst"])  & (z[1] in ["ascending", "descending"]):
                        if z[1] == "ascending":
                            memory["orderBy"].append(f"ORDER BY  {table}.{column} ASC")
                        if z[1] == "descending":
                            memory["orderBy"].append(f"ORDER BY {table}.{column} DESC")
        
        
        if (bool(orderBy) == False ) & (len(memory["orderBy"])==0) & (memory["orderAggFlag"]  == False):
            for x in goldPOS:
                if (x[1] in  order):
                    orderBy = x[0]
                    break
            if bool(orderBy) == True:
                numIDX= []
                for x in goldPOS:
                    if (x[0]>=orderBy) & (x[1] in ["number"]):
                        numIDX.append(x[0])
                        break
                if len(numIDX)>0:
                    for x in goldPOS:
                        if (x[0]>numIDX[0]) & ((x[2].startswith("Reference")) | (x[2].startswith("Main "))):
                            memory["orderBy"].append(f"ORDER BY COUNT(*)ASC")
                            break
                        if (x[0]>numIDX[0])&(x[1].startswith("Table:") & x[2].startswith("Column:")):
                            table= x[1][6:]
                            column= x[2][7:] 
                            memory["orderBy"].append(f"ORDER BY COUNT({table}.{column}) ASC")
                            break
        
        if (bool(orderBy) == True) & (memory["orderAggFlag"] == True):
            for c in goldPOS:
                if c[2] == "SQL function":
                    if c[1] in ["MAX", "MIN"]:
                        for z in goldPOS:
                            if (z[0]>=c[0]) & (z[2].startswith("Column:")):
                                table = z[1][6:]
                                column= z[2][7:]
                                break
                        if c[1]== "MAX":
                            memory["orderBy"]= []
                            memory["orderBy"].append(f"ORDER BY MAX({table}.{column})")
                        if c[1]== "MIN":
                            memory["orderBy"]= []
                            memory["orderBy"].append(f"ORDER BY MIN({table}.{column})")
                        for z in goldPOS:
                            if z[1] in ["ascending", "descending"]:
                                if z[1] == "ascending":
                                    memory["orderBy"].append("ASC")
                                if z[1]== "descending":
                                    memory["orderBy"].append("DESC")
        if (bool(orderBy)== False) & (memory["orderAggFlag"]== False):
            groupTable=None
            groupCol=None
            func= None
            flagMAXMIN= False
            for x in goldPOS:
                if x[2] == "SQL function":
                    if x[1] in ["MAX", "MIN"]:
                        func=x[1]
                        for z in goldPOS:
                            if (z[0]>= x[0]) & (z[1] == "number"):
                                if (goldPOS[z[0]][2].startswith("Column")):
                                    break
                                for c in goldPOS:
                                    if (c[0]>z[0])&(c[1].startswith("Table")) & (not c[2].startswith("Column")):
                                        table= c[1][6:]
                                        break
                                    if (c[0]>z[0])&(c[1].startswith("Table")) & (c[2].startswith("Column")):
                                        groupTable= c[1][6:]
                                        groupCol= c[2][7:]
                                        flagMAXMIN= True
                                        break
                                if flagMAXMIN== False:
                                    for c in goldPOS:
                                        if(c[0]<z[0])& (c[2].startswith("Column")):
                                            groupTable= c[1][6:]
                                            groupCol= c[2][7:]
                                            break
                                break
            if(groupCol!=None)& (groupTable!=None) & (func!=None):
                if func== "MAX":
                    memory["orderBy"].append(f"ORDER BY COUNT(*)")
                    memory["orderBy"].append(f"DESC LIMIT 1")
                    memory["specialFlagOrder"]= True
                    memory["groupByClause"]= f"GROUP BY {groupTable}.{str.lower(groupCol)}"
                if func== "MIN":
                    memory["orderBy"].append(f"ORDER BY COUNT(*)")
                    memory["orderBy"].append(f"ASC LIMI 1")
                    memory["specialFlagOrder"]= True
                    memory["groupByClause"]= f"GROUP BY {groupTable}.{str.lower(groupCol)}"



                        
 
        if len(memory["orderBy"])>=1:
            memory["orderBy"] = " ".join(memory["orderBy"])
            

    return


def aggregationsOF(goldPOS,memory):
    if memory["ofCount"]>0:
        specialFlag=False
        memory["specialFlag"] = False
    ############# Before of 
        memory["aggregations"]= {}
        for x in goldPOS:
            col = None
            colonly=None
            if (x[2] == "SQL function") & (x[0]<= memory["idxfirst"]):
                if (x[1]  in ["MIN", "MAX", "AVG", "DISTINCT", "SUM"]):
                    for z in goldPOS:
                        if (z[1] in ["common", "usual", "frequent", "frequently", "commonly","usually", "widely"]):
                            break
                        if(z[0]>=x[0]) & (z[2].startswith("Column:")):
                            col = f"{z[1][6:]}.{z[2][7:]}"
                            colonly= f"{z[2][7:]}"
                            memory["aggregations"][x[1]]= f"{z[1][6:]}.{z[2][7:]}"
                            break
                    if (col!= None) & (colonly!=None):
                        if str.lower(col) in memory["SelectClause"]:
                            memory["SelectClause"].pop(memory["SelectClause"].index(str.lower(col)))
                        if  (str.lower(colonly) in memory["SelectClause"]):
                            memory["SelectClause"].pop(memory["SelectClause"].index(str.lower(colonly)))
        if len(memory["aggregations"])>=1:
            select_clause = memory["SelectClause"]
            aggregations = memory.get('aggregations', {})
            agg_values = []
            for agg_func, agg_key in aggregations.items():
                if agg_key in select_clause:
                    select_clause[select_clause.index(agg_key)] = f"{agg_func}({agg_key})"
                    specialFlag=True
                if ((list(agg_key.split("."))[1]) in select_clause) & (specialFlag==False):
                    select_clause[select_clause.index(list(agg_key.split("."))[1])] = f"{agg_func}({agg_key})"
                else:
                    if f"{agg_func}({agg_key})" not in select_clause:
                        agg_values.append(f"{agg_func}({agg_key})")
            select_clause.extend(agg_values)
    ################# After OF
        memory["aggregationsAfterOf"]= {}
        num= None
        internalFlag= False
        for x in goldPOS:
            if (x[2] == "SQL function") & (x[0]>memory["idxfirst"]) & ((x[1] in ["MIN", "MAX", "AVG", "DISTINCT"])):
                if (goldPOS[x[0]-2][1]== "at"):
                            break
                if (goldPOS[x[0]][2].startswith("Reference")) & (x[1] in ["MIN", "MAX"]):
                        for c in goldPOS:
                            if (c[0]<x[0]) & (c[2].startswith("Column")):
                                if x[1] == "MAX":
                                    table = c[1][6:]
                                    column= c[2][7:]
                                    memory["groupByClause"] =f"GROUP BY {table}.{str.lower(column)}"
                                    memory["orderBy"].append("ORDER BY COUNT(*) DESC LIMIT 1")
                                    memory["specialFlag"] = True
                                if x[1] == "MIN":
                                    table = c[1][6:]
                                    column = c[2][7:]
                                    memory["groupByClause"]= f"GROUP BY {table}.{str.lower(column)}"
                                    memory["orderBy"].append("ORDER BY COUNT(*) ASC LIMIT 1")
                                    memory["specialFlag"] = True
                        break
                for z in goldPOS:
                    if (z[0]>x[0])& (z[1] in ["MAX", "MIN", "AVG", "SUM"]):
                        func = x[1]
                        internalFlag= True
                        internalfunc= z[1]
                        break
                    if (z[0]<x[0]) & (z[2] in ["NUM"]):
                        num= z[1]
                    if (z[0]>=x[0]) & (z[2] in ["common", "usual", "frequent", "frequently", "commonly","usually", "widely"]):
                        break
                    if(z[0]>=x[0]) & (z[2].startswith("Column:")):
                        memory["aggregationsAfterOf"][x[1]]= f"{z[1][6:]}.{z[2][7:]}"
                        break
        if (len(memory["aggregationsAfterOf"])>=1) & (memory["orderAggFlag"] ==False):
            memory["orderBy"]= []
            oderby_clause = memory["orderBy"]
            aggregations = memory.get("aggregationsAfterOf", {})
            agg_values = []
            for agg_func, agg_key in aggregations.items():
                if agg_func in["MAX", "MIN"]:                            
                    if (agg_func == "MAX"):
                        if num == None:
                            agg_values.append(f"ORDER BY {agg_key}")
                            agg_values.append(f"DESC LIMIT 1")
                        if num != None:
                            agg_values.append(f"ORDER BY {agg_key}")
                            agg_values.append(f"DESC LIMIT {w2n.word_to_num(num)}")
                        for v in memory["SelectClause"]:
                            if v.startswith("MAX"):
                                idx = memory["SelectClause"].index(v)
                                memory["SelectClause"].pop(idx)
                    if agg_func == "MIN":
                        if num == None:
                            agg_values.append(f"ORDER BY {agg_key}")
                            agg_values.append(f"ASC LIMIT 1")
                        if num != None:
                            agg_values.append(f"ORDER BY {agg_key}")
                            agg_values.append(f"ASC LIMIT {w2n.word_to_num(num)}")
                        for v in memory["SelectClause"]:
                            if v.startswith("MIN"):
                                idx = memory["SelectClause"].index(v)
                                memory["SelectClause"].pop(idx)
                if (agg_func in ["AVG", "DISTINCT"]) & (internalFlag==True):
                    if func == "MAX":
                        if num == None:
                            agg_values.append(f"ORDER BY {internalfunc}({agg_key})")
                            agg_values.append(f"DESC LIMIT 1")
                        if num != None:
                            agg_values.append(f"ORDER BY {internalfunc}({agg_key})")
                            agg_values.append(f"DESC LIMIT {w2n.word_to_num(num)}")
                    if func == "MIN":
                        if num == None:
                            agg_values.append(f"ORDER BY {internalfunc}({agg_key})")
                            agg_values.append(f"ASC LIMIT 1")
                        if num != None:
                            agg_values.append(f"ORDER BY {internalfunc}({agg_key})")
                            agg_values.append(f"ASC LIMIT {w2n.word_to_num(num)}")
            oderby_clause.extend(agg_values)
            if len(memory["orderBy"])>1:
                memory["orderBy"] = " ".join(memory["orderBy"])


    return




def specialAggregations(goldPOS,memory):
    if (memory["ofCount"]>0) & (len(memory["WhereClause"])==1):
        values= []
        where= None
        flag= False
        for x in goldPOS:
            if (x[0]>memory["idxfirst"])& (x[1]=="BETWEEN") & (x[2]=="SQL function"):
                betweenIDX= x[0]
                for z in goldPOS:
                    if (z[0]<x[0]) & (z[2].startswith("Column")):
                        table= z[1][6:]
                        column= z[2][7:]
                        where= f"{table}.{column}"
                    if (z[0]>x[0]) & (z[2] in ["NUM"]):
                        values.append(z[1])
        if len(values)>0:
            memory["WhereClause"].append(f"{table}.{column} ")
            for c in values:
                if "." in c:
                    flag = True  
                    break
            if flag== False:
                memory["WhereClause"].append(f"BETWEEN {w2n.word_to_num(values[0])} AND {w2n.word_to_num(values[-1])}")
            if flag == True:
                memory["WhereClause"].append(f"BETWEEN {values[0]} AND {values[-1]}")
        if (len(values)>0) & (where == None) :
            for z in goldPOS:
                if (z[0]>betweenIDX) & z[2].startswith("Column"):
                    table= z[1][6:]
                    column= z[2][7:]
                    where= f"{table}.{column}"
            if where!=None:
                memory["WhereClause"].append(f"{where} ")
                for c in values:
                    if "." in c:
                        flag = True  
                        break
                if flag== False:
                    memory["WhereClause"].append(f"BETWEEN {w2n.word_to_num(values[0])} AND {w2n.word_to_num(values[-1])}")
                if flag == True:
                    memory["WhereClause"].append(f"BETWEEN {values[0]} AND {values[-1]}")
        flag= False
        idx= None
        for x in goldPOS:
            if (x[0]<=memory["idxfirst"]) & (x[1] in ["MAX", "MIN"])& (x[2]=="SQL function"):
                for z in goldPOS:
                    if (z[0]>=x[0]) & (z[1] in ["common", "usual", "frequent", "frequently", "commonly","usually", "widely"]):
                        idx= z[0]
                        flag = True
                        break
            if (x[0]>=memory["idxfirst"]) & (x[1] in ["MAX", "MIN"])& (x[2]=="SQL function"):
                for z in goldPOS:
                    if (z[0]>=x[0]) & (z[1] in ["common", "usual", "frequent", "frequently", "commonly","usually", "widely"]):
                        idx= z[0]
                        flag = True
                        break
        if flag == True:
            flag1=False
            for z in goldPOS:
                if (z[0]>=idx) & (z[2].startswith("Column:")):
                    table = z[1][6:]
                    column = z[2][7:]
                    memory["orderBy"].append(f"ORDER BY COUNT(*) DESC LIMIT 1")
                    memory["groupByClause"]= f"GROUP BY {table}.{str.lower(column)}"
                    memory["specialFlag"]= True
                    flag1= True
                    break
                if (z[0]>=idx) & ((z[2].startswith("Main")) | (z[1].startswith("Reference"))) :
                    for c in goldPOS:
                        if c[2].startswith("Column"):
                            table = c[1][6:]
                            column = c[2][7:]
                            memory["orderBy"].append(f"ORDER BY COUNT(*) DESC LIMIT 1")
                            memory["groupByClause"]= f"GROUP BY {table}.{str.lower(column)}"
                            memory["specialFlag"]= True
                            flag1= True
                            break
            if flag1==False:
                for z in goldPOS:
                    if (z[0]<=idx) & (z[2].startswith("Column")):
                        table = z[1][6:]
                        column = z[2][7:]
                        memory["orderBy"].append(f"ORDER BY COUNT(*) DESC LIMIT 1")
                        memory["groupByClause"]= f"GROUP BY {table}.{str.lower(column)}"
                        memory["specialFlag"]= True
                        flag1= True
                        break
            if len(memory["orderBy"])>0:
                memory["orderBy"]= " ".join(memory["orderBy"])
    return


def propnWhere(goldPOS, memory):
    if memory["ofCount"]>0:
        propn ={}
        flag = False
        internalFlag= False
        for x in goldPOS:
            if (x[0]>memory["idxfirst"]) & (x[2] in ["propnPattern"]):
                function= None
                for z in goldPOS:
                    if (z[0]>memory["idxfirst"])  & (z[0]<=x[0]) & (z[1] in ["!=", "="]) :
                        function = z[1]
                    if (z[0]>memory["idxfirst"]) & (z[0]<=x[0]) & (z[1] in notIN):
                        function = "!="
                    if (z[0]<=x[0]) & (z[0]> memory["idxfirst"]) & (z[2].startswith("Column:")):
                        for c in goldPOS:
                            if (c[0]>=z[0]) &(c[0]<=x[0])& (c[1] in ["<", ">"]) & (c[2]== "SQL function"):
                                internalFlag= True
                                break
                            if (c[0]>memory["idxfirst"])  & (c[0]<=x[0]) & (c[1] in ["!=", "="]) :
                                function = c[1]
                            if (c[0]>memory["idxfirst"]) & (c[0]<=x[0]) & (c[1] in notIN):
                                function = "!="
                        if internalFlag==True:
                            break
                        if function == None:
                            propn[f"{z[1][6:]}.{z[2][7:]}"]= x[1]
                            memory["WhereClause"].append(f"{z[1][6:]}.{z[2][7:]}")
                            memory["WhereClause"].append(f""" ='{x[1]}'""")
                            flag=True
                        if function!= None:
                            propn[f"{z[1][6:]}.{z[2][7:]} {function}"]= x[1]
                            memory["WhereClause"].append(f"{z[1][6:]}.{z[2][7:]}")
                            memory["WhereClause"].append(f""" {function} '{x[1]}' """)
                            flag= True
        if flag==False:
            for x in goldPOS:
                if (x[0]>memory["idxfirst"]) & (x[2]in ["propnPattern"]):
                    for z in goldPOS:
                        if (z[0]>memory["idxfirst"])  & (z[0]<=x[0]) & (z[1] in ["!=", "="]):
                            function = z[1]
                        if (z[0]>memory["idxfirst"])  & (z[0]<=x[0]) & (z[1] in notIN):
                            function = z[1]
                        if (z[0]>= x[0]) & ((z[1] in ["which", "whose", "where","with", "and", "or"]) | (z[2].startswith("SQL function"))):
                            break
                        if (z[0]>=x[0]) & (z[2].startswith("Column:")):
                            if function == None:
                                propn[f"{z[1][6:]}.{z[2][7:]}"]= x[1]
                                memory["WhereClause"].append(f"{z[1][6:]}.{z[2][7:]}")
                                memory["WhereClause"].append(f" = '{x[1]}'")
                                flag=True
                            if function!= None:
                                propn[f"{z[1][6:]}.{z[2][7:]} {function}"]= x[1]
                                memory["WhereClause"].append(f"{z[1][6:]}.{z[2][7:]} ")
                                memory["WhereClause"].append(f""" {function}  '{x[1]}'""")
                                flag= True
                            break
        if flag == False:
            internalFlag= False
            function = None
            for x in goldPOS:
                if (x[0]>memory["idxfirst"]) & (x[2]in ["propnPattern"]):
                    for z in goldPOS:
                        if (z[0]>memory["idxfirst"]) & (z[0]<x[0]) & ((z[1] in notIN )| (z[1]=="!=")):
                            function= "!=" 
                    if memory["idxSim1Table"]!= None:
                        embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                        embedAll= SimilarityModel.encode(tableColumns[memory["idxSim1Table"]][1:],convert_to_tensor=True, normalize_embeddings=True)
                        sim =util.dot_score(embed, embedAll)
                        if torch.max(sim)>=0.4:
                            memory["WhereClause"].append(f"""{memory["idxSim1Table"]}.{tableColumns[memory["idxSim1Table"]][1:][torch.argmax(sim)]}""")
                            if function == None:
                                memory["WhereClause"].append(f"""= '{x[1]}'""")
                            if function != None:
                                memory["WhereClause"].append(f"""!= '{x[1]}'""")
                        if torch.max(sim)<=0.4:
                            internalFlag= True
                    if (memory["idxSim1Table"]== None) | (internalFlag==True):
                        embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                        embedTables = SimilarityModel.encode(tables[0], convert_to_tensor=True, normalize_embeddings=True)
                        tableSim= util.dot_score(embed, embedTables)
                        table = tables[0][torch.argmax(tableSim)]
                        embedTableCols= SimilarityModel.encode(tableColumns[table][1:],convert_to_tensor=True,normalize_embeddings=True)
                        sim =  util.dot_score(embed, embedTableCols)
                        memory["WhereClause"].append(f"""{table}.{tableColumns[table][1:][torch.argmax(sim)]}""")
                        if function == None:
                                memory["WhereClause"].append(f"""= '{x[1]}'""")
                        if function != None:
                                memory["WhereClause"].append(f"""!= '{x[1]}'""")
    return


def joinTables(goldPOS, memory):
    memory["aliases"]= {}
    if (memory["ofCount"]>0) & (len(set(memory["Tables"]))>1):
        pattern = re.compile('.*id$')
        ### Select the ids from the respective tables:
        ids = {}
        AllIDS= []
        for x in AllTableColumns:
            value= []
            for id in AllTableColumns[x]:
                if ('_id' in str.lower(id)) or ('id_' in str.lower(id)) or (("ID" in (id)))or (("id" in str.lower(id)) & (len(id)==2)) or (("id " in str.lower(id))) or(pattern.match(str.lower(id))):
                    value.append(id)
            AllIDS.append(value)
            ids[x]=value
        
        for key, value in ids.items():
            if not value:
                for column_key, column_values in AllTableColumns.items():
                    for z in column_values:
                        if (key != column_key) & (z in AllTableColumns[key]):
                            ids[key].append(z)
                            ids[column_key].append(z)
        for z in range(len(AllIDS)):
            if len(AllIDS[z])==1:
                AllIDS[z]= AllIDS[z][0]

        for key , value in ids.items():
            if not value:
                for all_dict_key, all_dict_values in AllTableColumns.items():        
                    if all_dict_key == key:
                        for values in  all_dict_values:
                            if (values in num for num in ["1","2","3","4","5","6","7","8","9","0"]):
                                ids[key].append(values)
        idsToJoin= {}
        for x in ids:
            if x in memory["Tables"]:
                idsToJoin[x]= str.lower(ids[x][0])

        if len(AllIDS)>2:
            embedAllIDS= SimilarityModel.encode(AllIDS, convert_to_tensor=True, normalize_embeddings=True)
            embedIdsToJoin= SimilarityModel.encode([list(idsToJoin.values())], convert_to_tensor=True, normalize_embeddings=True)
            simIDS= util.dot_score(embedIdsToJoin, embedAllIDS)
            specificIDembed= SimilarityModel.encode(AllIDS[torch.argmax(simIDS)], convert_to_tensor=True, normalize_embeddings=True)
            keyTable = None
            for key, value in ids.items():
                if AllIDS[torch.argmax(simIDS)] == value:
                    keyTable = key
                    if keyTable not in memory["Tables"]:
                        memory["Tables"].append(keyTable)
            indexes ={}
            for x in idsToJoin:
                tableIDS= {}
                embed= SimilarityModel.encode(idsToJoin[x], convert_to_tensor=True, normalize_embeddings=True)
                sim= util.dot_score(embed, specificIDembed)
                if type(AllIDS[torch.argmax(simIDS)])!=str:
                    tableIDS[idsToJoin[x]]= AllIDS[torch.argmax(simIDS)][torch.argmax(sim)]
                    indexes[x]= tableIDS
                if type(AllIDS[torch.argmax(simIDS)])==str:
                    tableIDS[idsToJoin[x]]= AllIDS[torch.argmax(simIDS)]
                    indexes[x]= tableIDS        
            
        if len(AllIDS)==2:
            embedAllIDS= SimilarityModel.encode(AllIDS, convert_to_tensor=True, normalize_embeddings=True)
            embedIdsToJoin= SimilarityModel.encode([list(idsToJoin.values())], convert_to_tensor=True, normalize_embeddings=True)
            simIDS= util.dot_score(embedIdsToJoin, embedAllIDS)
            vals= AllIDS[torch.argmax(simIDS)]
            if (type(vals)==list):
                specificIDembed= SimilarityModel.encode(AllIDS[torch.argmax(simIDS)], convert_to_tensor=True, normalize_embeddings=True)
                for key, value in ids.items():
                    if AllIDS[torch.argmax(simIDS)] == value:
                        keyTable = key
                indexes = {}
                for x in idsToJoin:
                    tableIDS= {}
                    embed= SimilarityModel.encode(idsToJoin[x], convert_to_tensor=True, normalize_embeddings=True)
                    sim= util.dot_score(embed, specificIDembed)
                    if type(AllIDS[torch.argmax(simIDS)])!=str:
                        tableIDS[idsToJoin[x]]= AllIDS[torch.argmax(simIDS)][torch.argmax(sim)]
                        indexes[x]= tableIDS
                    if type(AllIDS[torch.argmax(simIDS)])==str:
                        tableIDS[idsToJoin[x]]= AllIDS[torch.argmax(simIDS)]
                        indexes[x]= tableIDS  
            if type(vals)== str:
                indexes ={}
                keyTable = list(idsToJoin.keys())[0]
                for x in idsToJoin:
                    if x != keyTable:
                        indexes[x]= {idsToJoin[x]: idsToJoin[keyTable]}
        i=1
        for z in set(memory["Tables"]):
            memory["aliases"][z]= i
            i+=1
        if len(set(memory["Tables"]))==2:
            if keyTable== None:
                keyTable= list(indexes.keys())[1]
            for x in indexes:
                if x != keyTable:
                    memory["SelectClause"].append(f"""FROM {x} AS T{memory["aliases"][x]} JOIN {keyTable} AS T{memory["aliases"][keyTable]} ON T{memory["aliases"][x]}.{ids[x][0]} = T{memory["aliases"][keyTable]}.{indexes[x][str.lower(ids[x][0])]}""")
        if len(set(memory["Tables"]))>2:
            if keyTable== None:
                keyTable= list(indexes.keys())[1]
            for x in list(indexes.keys())[0:1]:
                if (x != keyTable):
                    memory["SelectClause"].append(f"""FROM {x} AS T{memory["aliases"][x]} JOIN {keyTable} AS T{memory["aliases"][keyTable]} ON T{memory["aliases"][x]}.{ids[x][0]} = T{memory["aliases"][keyTable]}.{indexes[x][str.lower(ids[x][0])]}  """)
                    break
            for x in list(indexes.keys())[1:]:
                if x != keyTable:
                    memory["SelectClause"].append(f""" JOIN {x} AS T{memory["aliases"][x]} ON T{memory["aliases"][x]}.{ids[x][0]} = T{memory["aliases"][keyTable]}.{indexes[x][str.lower(ids[x][0])]}  """)
            
            from_index= None
            for x in memory["SelectClause"]:
                if x.startswith("FROM"):
                    from_index= memory["SelectClause"].index(x)
            scnd= " ".join(memory["SelectClause"][from_index:])
            del memory["SelectClause"][from_index:]
            memory["SelectClause"].append(scnd)

    return


def CountClause(goldPOS, memory):
    if memory["ofCount"]>=1:
        c= None
        for x in goldPOS:
            if (x[0]<= memory["idxfirst"]) & (str.lower(x[1]) in count):
                memory["SelectClause"].insert(0,"COUNT(*)")
                for z in goldPOS:
                    if (z[0]>x[0]) & (z[0]<=memory["idxfirst"]):
                        if z[1] == "DISTINCT":
                            memory["SelectClause"].pop(memory["SelectClause"].index(f"COUNT(*)"))
                            break
                        if z[1] in ["and", "or"]:
                            break
                        if (z[2].startswith("Column")) :
                            memory["SelectClause"].pop(memory["SelectClause"].index(f"{z[1][6:]}.{z[2][7:]}"))
                            break
        if len(memory["Tables"])<1:
            for x in goldPOS:
                if (x[2].startswith("Main table")) | (x[2].startswith("Reference")):
                    if x[1] not in memory["Tables"]:
                        memory["Tables"].append(x[1])
        for x in goldPOS:
            if (x[0]<=memory["idxfirst"]) & (x[1] == "number"):
                for z in goldPOS:
                    if (z[0]>x[0]) & (z[1]=="DISTINCT") :
                        for y in memory["SelectClause"]:
                            if "DISTINCT" in y:
                                idx= memory["SelectClause"].index(y)
                                tc= memory["SelectClause"][idx][9:-1]
                                memory["SelectClause"].insert(idx, f"COUNT(DISTINCT({tc}))")
                                memory["SelectClause"].pop(idx+1)
                                c=True
                                for y in memory["SelectClause"]:
                                    if "COUNT(*)" in y:
                                        idx= memory["SelectClause"].index(y)
                                        memory["SelectClause"].pop(idx)
                                        break   
                                break
                        
        for x in goldPOS:
            if (x[0]<=memory["idxfirst"]) & (x[1] == "number"):
                for z in goldPOS:
                    if ((z[0]>x[0]) & (z[2].startswith("Column"))) | ((z[0]>x[0])& (z[1].startswith("Table"))):
                        if any(str.lower(value) in str.lower(z[2][7:]) for value in ["number", "num_", "total","sum"]):
                            memory["SelectClause"].insert(0,f"{z[1][6:]}.{z[2][7:]}")
                            break
                        else:
                            if (c== None):
                                for y in memory["SelectClause"]:
                                    if str.lower(y).startswith("count"):
                                        pass
                                    else:
                                        if goldPOS[x[0]-2][1] in ["AVG", "SUM", "MIN", "MAX"]:
                                            pass
                                        else:
                                            memory["SelectClause"].insert(0,"COUNT(*)")
                                            break
                            if (c== None) & (len(memory["SelectClause"])==0):
                                if goldPOS[x[0]-2][1] in ["AVG","SUM"]:
                                    pass
                                else:
                                    memory["SelectClause"].insert(0, "COUNT(*)")
                        break
        flag= False
        column = None
        for x in goldPOS:
            if (x[0]> memory["idxfirst"]) & (x[1] == "number"):
                for z in goldPOS:
                    if (z[0]>memory["idxfirst"]) & (z[0]<=x[0]) & (z[1] in having):
                        break
                    if (z[0]>memory["idxfirst"]) & (z[0]>=x[0]) & (z[1].startswith("Table")) & (not z[2].startswith("Column")):
                        for c in goldPOS:
                            if c[2].startswith("Column"):
                                table= c[1][6:]
                                column = c[2][7:]
                                break
                        if column != None:
                            memory["groupByClause"]= f"GROUP BY {table}.{str.lower(column)}"
                            memory["specialFlag"]= True
                        if column == None:
                            memory["groupByClause"]= f"GROUP BY {table}.{str.lower(AllTableColumns[table][0])}"
                            memory["specialFlag"]= True
                        for c in goldPOS:
                            if (c[0]>=memory["idxfirst"]) & (c[0]<x[0]) & (c[1] in ["and"]):
                                if "COUNT(*)" not in memory["SelectClause"]:
                                    memory["SelectClause"].insert(0,"COUNT(*)")
                                break
                            else:
                                pass
                        flag = True
                    if (z[0]>=x[0]) & (z[1].startswith("Table")) & (z[2].startswith("Column")) & (flag==False):
                        pass
    return


def GroupByClause(goldPOS, memory):
    if (memory["specialFlag"]!= False) | (memory["specialFlagOrder"]!= False):
        pass
    if (memory["specialFlag"]== False) & (memory["specialFlagOrder"]==False):
        memory["groupByClause"] = []
    flag= False
    if memory["ofCount"]>=1:
        for x in goldPOS:
            if (x[0]>memory["idxfirst"]) & (x[1] in groupBy):
                for z in goldPOS:
                    if (z[0]>=x[0]) & (z[2].startswith("Column:")):
                        table= z[1][6:]
                        column = z[2][7:]
                        memory["groupByClause"]= f"GROUP BY {table}.{str.lower(column)}"
                        break
                break
        for x in goldPOS:
            if (x[0]<=memory["idxfirst"]) & (x[1] in groupBy):
                for z in goldPOS:
                    if (z[0]>=x[0]) & (z[2].startswith("Column:")):
                        table= z[1][6:]
                        column = z[2][7:]
                        memory["groupByClause"]= f"GROUP BY {table}.{str.lower(column)}"
                        break
                break
        ### Para "each" no caso de o proximo token ser coluna 
        for x in goldPOS:
            if (x[1] in ["each"]) & (len(memory["groupByClause"])==0):
                if (goldPOS[x[0]][2].startswith("Column:")):
                    table = goldPOS[x[0]][1][6:]
                    column = goldPOS[x[0]][2][7:]
                    if f"{table}.{column}" not in memory["groupByClause"] :
                        memory["groupByClause"]= f"GROUP BY {table}.{str.lower(column)}"
                        if AllTableColumns[table][0]!= column:
                            if (str.lower(f"{table}.{column}")  in memory["SelectClause"]) | (str.lower(column) in memory["SelectClause"]):
                                pass
                            else:
                                memory["SelectClause"].insert(0,f"{table}.{str.lower(column)}")
                        flag= True
                if (goldPOS[x[0]][2].startswith("Reference")) | (goldPOS[x[0]][2].startswith("Main")):
                    table= goldPOS[x[0]][1][6:]
                    column = AllTableColumns[table][0]
                    memory["groupByClause"] = f"GROUP BY {table}.{str.lower(column)}"
                    if AllTableColumns[table][0]!= column:
                        if (str.lower(f"{table}.{column}")  in memory["SelectClause"]) | (str.lower(column) in memory["SelectClause"]):
                            pass
                        else:
                            memory["SelectClause"].insert(0,f"{table}.{str.lower(column)}")
        if flag == False:
            for x in goldPOS:
                if (x[1] in ["each"]) & (len(memory["groupByClause"])==0):
                    for c in goldPOS:
                        if (c[0]<x[0]) & (c[2].startswith("Column")):
                            memory["groupByClause"]= f"GROUP BY {str.lower(c[2][7:])}"
                            if AllTableColumns[c[1][6:]][0]!= c[2][7:]:
                                if (str.lower(f"{c[1][6:]}.{c[2][7:]}")  in memory["SelectClause"]) | ( str.lower(f"{c[2][7:]}") in memory["SelectClause"]):
                                    pass
                                else:
                                    memory["SelectClause"].insert(0,f"{c[1][6:]}.{str.lower(c[2][7:])}")

    return


def havingClause(goldPOS, memory):
    if memory["ofCount"]>0:
        memory["having"]= []
        generalFlag= False
        flag= False
        memory["numFlagHaving"]= False
        try:
            for x in goldPOS:
                if (x[0]>memory["idxfirst"]) & (x[1] in having):
                    for c in goldPOS:
                        if (c[0]>=x[0]) & (c[2].startswith("Column")):
                            break
                        if (c[0]>=x[0]) & (c[2] == "SQL function"):
                            if (c[1] =="MIN") & (goldPOS[c[0]-2][1] == "at"):
                                for z in goldPOS:
                                    if (z[0]>c[0]) & (z[2] =="NUM"):
                                        memory["having"] = [f"HAVING COUNT(*) >= {w2n.word_to_num(z[1])}"]
                                        if len(memory["Tables"])>0:
                                            for c in goldPOS:
                                                if (c[0]>z[0])&(c[1] in to_ignore):
                                                    break
                                                if (c[0]>z[0]) & (c[1].startswith("Table")):
                                                    if c[2].startswith("Column"):
                                                        table= c[1][6:]
                                                        col= c[2][7:]
                                                        memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                        flag= True
                                                    if not c[2].startswith("Column"):
                                                        for d in goldPOS:
                                                            if (d[2].startswith("Column")) & (d[1]== c[1]):
                                                                table= d[1][6:]
                                                                col= d[2][7:]
                                                                memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                                flag=True
                                                                break
                                            if len(memory["groupByClause"])<1:
                                                for d in goldPOS:
                                                    if (d[2].startswith("Column")):
                                                        table= d[1][6:]
                                                        col= d[2][7:]
                                                        memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                        flag=True
                                                        break
                                            if len(memory["groupByClause"])<1:
                                                table = memory["Tables"][0]
                                                col = AllTableColumns[table][0]
                                                memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                flag =True
                                                break  
                                        break
                            if (c[1] =="MIN") & (goldPOS[c[0]-2][1] != "at"):
                                for z in goldPOS:
                                    if (z[0]>c[0]) & (z[2] =="NUM"):
                                        memory["having"] = [f"HAVING COUNT(*) <= {w2n.word_to_num(z[1])}"]
                                        if len(memory["Tables"])>0:
                                            for c in goldPOS:
                                                if (c[0]>z[0])&(c[1] in to_ignore):
                                                    break
                                                if (c[0]>z[0]) & (c[1].startswith("Table")):
                                                    if c[2].startswith("Column"):
                                                        table= c[1][6:]
                                                        col= c[2][7:]
                                                        memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                        flag= True
                                                    if not c[2].startswith("Column"):
                                                        for d in goldPOS:
                                                            if (d[2].startswith("Column")) & (d[1]== c[1]):
                                                                table= d[1][6:]
                                                                col= d[2][7:]
                                                                memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                                flag=True
                                                                break
                                            if len(memory["groupByClause"])<1:
                                                for d in goldPOS:
                                                    if (d[2].startswith("Column")):
                                                        table= d[1][6:]
                                                        col= d[2][7:]
                                                        memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                        flag=True
                                                        break
                                            if len(memory["groupByClause"])<1:
                                                table = memory["Tables"][0]
                                                col = AllTableColumns[table][0]
                                                memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                flag =True
                                                break  
                                        break
                            if c[1] =="MAX":
                                for z in goldPOS:
                                    if (z[0]>c[0]) & (z[2] =="NUM"):
                                        memory["numFlagHaving"]= True
                                        memory["having"] = [f"HAVING COUNT(*) >= {w2n.word_to_num(z[1])}"]
                                        if len(memory["Tables"])>0:
                                            for c in goldPOS:
                                                if (c[0]>z[0])&(c[1] in to_ignore):
                                                    break
                                                if (c[0]>z[0]) & (c[1].startswith("Table")):
                                                    if c[2].startswith("Column"):
                                                        table= c[1][6:]
                                                        col= c[2][7:]
                                                        memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                        flag= True
                                                    if not c[2].startswith("Column"):
                                                        for d in goldPOS:
                                                            if (d[2].startswith("Column")) & (d[1]== c[1]):
                                                                table= d[1][6:]
                                                                col= d[2][7:]
                                                                memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                                flag=True
                                                                break
                                            if len(memory["groupByClause"])<1:
                                                for d in goldPOS:
                                                    if (d[2].startswith("Column")):
                                                        table= d[1][6:]
                                                        col= d[2][7:]
                                                        memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                        flag=True
                                                        break
                                            if len(memory["groupByClause"])<1:
                                                table = memory["Tables"][0]
                                                col = AllTableColumns[table][0]
                                                memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                flag =True
                                                break  
                                        break
                            if c[1]== ">":
                                for z in goldPOS:
                                    if (z[0]>= c[0]) &(z[1].startswith("Table")):
                                        break
                                    if (z[0]>=c[0])&(z[2]== "NUM"):
                                        memory["having"]= [f"HAVING COUNT(*)> {w2n.word_to_num(z[1])}"]
                                        if len(memory["Tables"])>0:
                                            for c in goldPOS:
                                                if (c[0]>z[0])&(c[1] in to_ignore):
                                                    break
                                                if (c[0]>z[0]) & (c[1].startswith("Table")):
                                                    if c[2].startswith("Column"):
                                                        table= c[1][6:]
                                                        col= c[2][7:]
                                                        memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                        flag= True
                                                    if not c[2].startswith("Column"):
                                                        for d in goldPOS:
                                                            if (d[2].startswith("Column")) & (d[1] == c[1]):
                                                                table= d[1][6:]
                                                                col= d[2][7:]
                                                                memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                                flag=True
                                                                break
                                            if len(memory["groupByClause"])<1:
                                                for d in goldPOS:
                                                    if (d[2].startswith("Column")):
                                                        table= d[1][6:]
                                                        col= d[2][7:]
                                                        memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                        flag=True
                                                        break
                                            if len(memory["groupByClause"])<1:
                                                table = memory["Tables"][0]
                                                col = AllTableColumns[table][0]
                                                memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                flag =True
                                                break   
                                        break
                                if flag == False:
                                    for z in goldPOS:
                                        if (z[0]<=c[0]) & (z[2]=="NUM"):
                                            if goldPOS[c[0]-2][1]=="or":
                                                memory["having"]= [f"HAVING COUNT(*)>= {w2n.word_to_num(z[1])}"]
                                            if goldPOS[c[0]-2][1]!="or":
                                                memory["having"]= [f"HAVING COUNT(*)> {w2n.word_to_num(z[1])}"]
                                            if (len(memory["idxSim1Table"])>0) & (len(memory["SelectClause"])==0):
                                                table = memory["idxSim1Table"]
                                                col = AllTableColumns[table][0]
                                                memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                break
                                            if (len(memory["SelectClause"])!=0):
                                                for z in goldPOS:
                                                    if (z[0]<=memory["idxfirst"]) & (z[2].startswith("Column")):
                                                        table= z[1][6:]
                                                        column = z[2][7:]
                                                        memory["groupByClause"] = f"GROUP BY {table}.{str.lower(column)}"
                                                        break
                                            break
                                            
                            if c[1]== "<":
                                for z in goldPOS:
                                    if (z[0]>= c[0]) &(z[1].startswith("Table")):
                                        break
                                    if (z[0]>=c[0])&(z[2]== "NUM"):
                                        memory["having"]= [f"HAVING COUNT(*)< {w2n.word_to_num(z[1])}"]
                                        if len(memory["Tables"])>0:
                                            for c in goldPOS:
                                                if (c[0]>z[0])&(c[1] in to_ignore):
                                                    break
                                                if (c[0]>z[0]) & (c[1].startswith("Table")):
                                                    if c[2].startswith("Column"):
                                                        table= c[1][6:]
                                                        col= c[2][7:]
                                                        memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                        flag= True
                                                    if not c[2].startswith("Column"):
                                                        for d in goldPOS:
                                                            if (d[2].startswith("Column")) & (d[1]== c[1]):
                                                                table= d[1][6:]
                                                                col= d[2][7:]
                                                                memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                                flag=True
                                                                break
                                            if len(memory["groupByClause"])<1:
                                                for d in goldPOS:
                                                    if (d[2].startswith("Column")):
                                                        table= d[1][6:]
                                                        col= d[2][7:]
                                                        memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                        flag=True
                                                        break
                                            if len(memory["groupByClause"])<1:
                                                table = memory["Tables"][0]
                                                col = AllTableColumns[table][0]
                                                memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                flag =True
                                                break  
                                        break
                                if flag == False:
                                    for z in goldPOS:
                                        if (z[0]<=c[0]) & (z[2]=="NUM"):
                                            if goldPOS[c[0]-2][1]=="or":
                                                memory["having"]= [f"HAVING COUNT(*)<= {w2n.word_to_num(z[1])}"]
                                            if goldPOS[c[0]-2][1]!= "or":
                                                memory["having"]= [f"HAVING COUNT(*)< {w2n.word_to_num(z[1])}"]
                                            if (len(memory["idxSim1Table"])>0) & (len(memory["SelectClause"])==0):
                                                table = memory["idxSim1Table"]
                                                col = AllTableColumns[table][0]
                                                memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                break
                                            if (len(memory["SelectClause"])!=0):
                                                for z in goldPOS:
                                                    if (z[0]<=memory["idxfirst"]) & (z[2].startswith("Column")):
                                                        table= z[1][6:]
                                                        column = z[2][7:]
                                                        memory["groupByClause"] = f"GROUP BY {table}.{str.lower(column)}"
                                                        break
                                            break
                        
                            generalFlag= True   
                    if generalFlag== False:
                        for c in goldPOS:
                            if (c[0]>x[0]) & (c[2].startswith("Column")):
                                break
                            if (c[0]>x[0]) & (c[2]=='NUM'):
                                memory["having"]= [f"HAVING COUNT (*) = {w2n.word_to_num(c[1])}"]
                                if (len(memory["idxSim1Table"])>0) & (len(memory["SelectClause"])==0):
                                    table = memory["idxSim1Table"]
                                    col = AllTableColumns[table][0]
                                    memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                    break
                                if (len(memory["SelectClause"])!=0):
                                    for z in goldPOS:
                                        if (z[0]<=memory["idxfirst"]) & (z[2].startswith("Column")):
                                            table= z[1][6:]
                                            column = z[2][7:]
                                            memory["groupByClause"] = f"GROUP BY {table}.{str.lower(column)}"
                                            break
                                    break

            if memory["having"]== []:
                for x in goldPOS:
                    if (x[0]>memory["idxfirst"]) & (x[1] in ["MIN", "MAX"]):
                        if (goldPOS[x[0]-2][1] == "at") & (not goldPOS[x[0]-3][2].startswith("Column")):
                            for c in goldPOS:
                                if (c[0]>x[0]) & (c[2] in "NUM"):
                                    if x[1] == "MIN":
                                        memory["having"]= [f"HAVING COUNT(*)>= {w2n.word_to_num(c[1])}"]
                                        break
                                    if x[1]== "MAX":
                                        memory["having"]= [f"HAVING COUNT(*)<= {w2n.word_to_num(c[1])}"]
                            for c in goldPOS:
                                if (c[0]<x[0]) & (c[2].startswith("Column")):
                                    tabel = c[1][6:]
                                    column = c[2][7:]
                                    memory["groupByClause"]= f"GROUP BY {tabel}.{str.lower(column)}"
                                    break
        except:pass
    
                                    
    return

def subselectClause(goldPOS, memory):
    if memory["ofCount"]>0:
        if (len(memory["having"])==0) & (len(memory["orderBy"])== 0) & (len(memory["WhereClause"])<2) :
            haveFlag= False
            internalCol= None
            internalTable=None
            internalsubFunc=None
            subFunc=None
            internalFuncSubFlag=False
            internalColSubFlag=False
            flag= False
            for x in goldPOS:
                if (x[0]>=memory["idxfirst"]) & (x[1] in having) & (haveFlag==False):
                    for c in goldPOS:
                        if (c[0]>memory["idxfirst"]) & (c[1] in notIN):
                            flag=True
                        if (c[0]>=x[0]) & (c[2].startswith("Column")):
                            internalCol= c[2][7:]
                            internalTable= c[1][6:]
                            for z in goldPOS:
                                if (z[0]>c[0]) & (z[1] in ["<", ">"]):
                                    internalsubFunc= z[1]
                                if (z[0]>c[0]) & (z[1] in ["MIN", "MAX", "SUM", "AVG"]):
                                    subFunc= z[1]
                                    internalFuncSubFlag= True
                                if (z[0]>c[0]) & (z[2].startswith("Column")):
                                    column = z[2][7:]
                                    table= z[1][6:]
                                    internalColSubFlag=True
                            break
                            
                        if (c[0]>=x[0]) & (c[1] in notIN):
                            flag=True
                        if (c[0]>=x[0]) & ((c[2].startswith("Reference")) | (c[2].startswith("Main"))):
                            for value in memory["SelectClause"]:
                                if " AS " in value:
                                    memory["SelectClause"].pop(memory["SelectClause"].index(value))
                                    memory["SelectClause"].append(value.split(" AS ")[0])
                            if flag==False:
                                if memory["idxSim1Table"]!= None:
                                    idCol= AllTableColumns[memory["idxSim1Table"]][0]
                                    memory["WhereClause"].append(f" {idCol} IN (SELECT {idCol} ")
                                    memory["WhereClause"].append(f"FROM {c[1][6:]})")
                                    haveFlag=True
                                if memory["idxSim1Table"]== None:
                                    table = memory["Tables"][0]
                                    idCol= AllTableColumns[table][0]
                                    memory["WhereClause"].append(f" {idCol} IN (SELECT {idCol} ")
                                    memory["WhereClause"].append(f"FROM {c[1][6:]})")
                                    haveFlag=True
                            if flag==True:
                                if memory["idxSim1Table"]!=None:
                                    idCol= AllTableColumns[memory["idxSim1Table"]][0]
                                    memory["WhereClause"].append(f" {idCol} NOT IN (SELECT {idCol} ")
                                    memory["WhereClause"].append(f"FROM {c[1][6:]})")
                                    haveFlag= True
                                if memory["idxSim1Table"]== None:
                                    table = memory["Tables"][0]
                                    idCol = AllTableColumns[table][0]
                                    memory["WhereClause"].append(f" {idCol} NOT IN (SELECT {idCol} ")
                                    memory["WhereClause"].append(f"FROM {c[1][6:]})")
                                    haveFlag=True
            if (internalColSubFlag==True) & (internalFuncSubFlag==True):
                for value in memory["SelectClause"]:
                    if " AS " in value:
                        memory["SelectClause"].pop(memory["SelectClause"].index(value))
                        memory["SelectClause"].append(value.split(" AS ")[0])
                if (internalsubFunc!=None )& (column !=None):
                    memory["WhereClause"].append(f" {internalCol} {internalsubFunc} (SELECT {subFunc}({column}) ")
                    memory["WhereClause"].append(f"FROM {table})")
            wherecol= None
            subCol= None
            for x in goldPOS:
                if (x[0]>memory["idxfirst"]) & (x[2] == "SQL function"):
                    for c in goldPOS:
                        if (c[0] > x[0]) & (c[2] == "SQL function")& (c[1] in ["MIN", "MAX", "SUM", "AVG"]):
                            if goldPOS[x[0]-2][2].startswith("Column"):
                                table= goldPOS[x[0]-2][1][6:]
                                wherecol= f"{goldPOS[x[0]-2][1][6:]}.{goldPOS[x[0]-2][2][7:]}"
                            if goldPOS[c[0]-2][2].startswith("Column"):
                                subCol= f"{goldPOS[c[0]-2][1][6:]}.{goldPOS[c[0]-2][2][7:]}"
                            if not goldPOS[c[0]-2][2].startswith("Column"):
                                subCol= f"{goldPOS[x[0]-2][1][6:]}.{goldPOS[x[0]-2][2][7:]}"
                            
                            if (wherecol!= None) & (subCol!= None):
                                memory["WhereClause"].append(f" {wherecol} {x[1]} ")
                                memory["WhereClause"].append(f"(SELECT {c[1]}({subCol}) ")
                                memory["WhereClause"].append(f"FROM {table})")
                    break
                        
    return

def StructureQueryOF(goldPOS,memory):
    if memory["ofCount"]>0:
        ######## PARA UMA TABELA
        if len(set(memory["Tables"]))==1:
            if len(memory["SelectClause"])>1:
                memory["SelectClause"]= [memory["SelectClause"][0]] + ["," + item for item in memory["SelectClause"][1:]]
                memory["SelectClause"].insert(0, "SELECT")
                memory["SelectClause"].insert(len(memory["SelectClause"]), f'FROM {memory["Tables"][0]}')
            if len(memory["SelectClause"])== 1:
                memory["SelectClause"].insert(0, "SELECT")
                memory["SelectClause"].insert(len(memory["SelectClause"]), f'FROM {memory["Tables"][0]}')
            if len(memory["WhereClause"])>3:
                idx = None
                for x in memory["WhereClause"]:
                    if ('.' in x):
                        col= x
                        idx= memory["WhereClause"].index(x)
                        if memory["WhereClause"][idx+1].startswith("(SELECT"):
                            idx= None
                            break
                        break
                if idx!= None:
                    if memory["or"]==True:
                        memory["WhereClause"].insert(idx+2, "OR")
                    if memory["or"]==False:
                        for x in memory["WhereClause"][idx+1:]:
                            if col in x:
                                memory["WhereClause"].insert(idx+2, "OR")
                            if (col not in x) & ("." in x):
                                memory["WhereClause"].insert(memory["WhereClause"].index(x), "AND")
                where= memory["WhereClause"]
            if len(memory["WhereClause"])<=3:
                where = memory["WhereClause"]
            if len(memory["WhereClause"])==2:
                where = memory["WhereClause"]
                for c in memory["WhereClause"]:
                    if "BETWEEN" in c:
                        where = " ".join(memory["WhereClause"])
            select= " ".join(memory["SelectClause"])
            try:
                if len(where)>=3:
                    if len(where)==3:
                        where= " ".join(memory["WhereClause"])
                        final= select +" "+ where
                    if len(where)>=3:
                        where= " ".join(memory["WhereClause"])
                        final = select+ " " +where
            except:pass
            try:
                if len(where)<3:
                    final= select
            except:pass
            try:
                if len(memory["groupByClause"])>0:
                    final = final +" "+ memory["groupByClause"]
            except:pass
            try:
                if len(memory["orderBy"])>0:
                    if type(memory["orderBy"])== list:
                        final = final+" "+memory["orderBy"][0]
                    else:
                        final = final+ " "+ memory["orderBy"]
            except:pass
            if len (memory["having"])>0:
                final = final + " " + memory["having"][0]
        ##### PARA mais que uma TABELA
        if len(set(memory["Tables"]))>1:
            flag = False
            flag1= False
            for i in range(len(memory["SelectClause"])):
                if len(memory["WhereClause"])>0:
                    for x in memory["WhereClause"]:
                        if "(SELECT " in x:
                            flag1= True
                            break
                if flag1==False:
                    for key in memory["aliases"]:
                        if memory["SelectClause"][i].startswith("FROM"):
                            flag= True
                            break
                        if (key in memory["SelectClause"][i]): 
                            if ("MAX" in memory["SelectClause"][i]) | ("MIN" in memory["SelectClause"][i])| ("SUM" in memory["SelectClause"][i]) | ("AVG" in memory["SelectClause"][i]) | ("DISTINCT" in memory["SelectClause"][i]):
                                memory["SelectClause"][i] =  memory["SelectClause"][i].replace(key, str(f'T{memory["aliases"][key]}')) 
                            else:
                                splited= memory["SelectClause"][i].split(".")
                                if len(key)== len(splited[0]):
                                    splited[0]= splited[0].replace(key, str(f'T{memory["aliases"][key]}')) 
                                    splited = ".".join(splited)
                                    memory["SelectClause"][i] = splited  
                    if flag== True:
                        break
                
            result = []
            for index, item in enumerate(memory["SelectClause"]):
                if item.startswith('FROM'):
                    result.append(item)
                    break
                if index > 0 and not memory["SelectClause"][index-1].startswith('FROM'):
                    result.append(', ')
                result.append(item)
            memory["SelectClause"]= result            

                
            memory["SelectClause"].insert(0, "SELECT")  
            select= " ".join(memory["SelectClause"])
            if len(memory["WhereClause"])>3:
                idx = None
                for x in memory["WhereClause"]:
                    if ('.' in x):
                        col= x
                        idx= memory["WhereClause"].index(x)
                        break
                if memory["or"]==True:
                    memory["WhereClause"].insert(idx+2, "OR")
                if memory["or"]==False:
                    for x in memory["WhereClause"][idx+1:]:
                        if col in x:
                            memory["WhereClause"].insert(idx+2, "OR")
                        if (col not in x) & ("." in x):
                            memory["WhereClause"].insert(memory["WhereClause"].index(x), "AND")
                where= memory["WhereClause"]
            if len(memory["WhereClause"])>=2:
                for i in range(len(memory["WhereClause"])):
                    if "(SELECT " in memory["WhereClause"][i]:
                        break
                    for key in memory["aliases"]:
                        if key in memory["WhereClause"][i]:
                            splited= memory["WhereClause"][i].split(".")
                            if ("MAX" in memory["WhereClause"][i]) | ("MIN" in memory["WhereClause"][i])| ("SUM" in memory["WhereClause"][i]) | ("AVG" in memory["WhereClause"][i]) | ("DISTINCT" in memory["WhereClause"][i]):
                                memory["WhereClause"][i] = memory["WhereClause"][i].replace(key, str(f'T{memory["aliases"][key]}'))
                            else:
                                if len(key)== len(splited[0]):
                                    splited[0]= splited[0].replace(key, str(f'T{memory["aliases"][key]}')) 
                                    splited = ".".join(splited)
                                    memory["WhereClause"][i] = splited 
                where= " ".join(memory["WhereClause"])  
                final = select+ " "+ where
            if len(memory["WhereClause"])<2:
                final = select
            try: 
                if len(memory["groupByClause"])>0:
                    for i in (memory["groupByClause"]):
                        for key in memory["aliases"]:
                            if key in memory["groupByClause"]:
                                splitted= memory["groupByClause"].split(".")
                                group1= splitted[0].replace("GROUP BY", "")
                                group2 = group1.replace(" ", "")
                                group2 = group2.replace(key, str(f'T{memory["aliases"][key]}'))
                                grouped = group2+"."+ splitted[-1]
                                memory["groupByClause"] = f"GROUP BY {grouped}"
                final = final +" "+ memory["groupByClause"]
            except:pass
            if len(memory["orderBy"])>0:
                for i in (memory["orderBy"]):
                    for key in memory["aliases"]:
                        if key in memory["orderBy"]:
                            memory["orderBy"] = memory["orderBy"].replace(key, str(f'T{memory["aliases"][key]}'))
                if type(memory["orderBy"])== list:
                    final = final +" "+ memory["orderBy"][0]
                else:
                    final = final +" "+ memory["orderBy"]
            
            if len(memory["having"])>0:
                final = final + " " + memory["having"][0]
            
            

    return final





######################################################################################################################################################################################################
################################################################## WITHOUT OF ########################################################################################################################
######################################################################################################################################################################################################

def phraseBroker(goldPOS,memory):
    if memory["ofCount"]==0:
        memory["phraseBroke"]=None
        for x in goldPOS:
            if ((x[1] in phraseBroke) & (x[0]  not in [1,2])) | ((x[2] in ["VERB"]) & (x[0] not in [1,2])) | ((x[1] in ["which", "whose","where", "what", "when", "with"]) & (x[0] not in [1, 2])) | ((x[1] in ["for", "from"]) & (x[0] not in [1,2])):
                memory["phraseBroke"]= x[0]-1
                break
        if memory["phraseBroke"]== None:
            memory["phraseBroke"]= goldPOS[-1][0]-1
    return

def specialCaseWordsWithoutOf(goldPOS, memory):
    if memory["ofCount"]==0:
        for x in goldPOS:
            if x[1] in special_cases:
                if x[1] in ["older"]:
                    embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                    embedAllCols= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                    sim2 =util.dot_score(embed, embedAllCols)
                    column = AllColumns[torch.argmax(sim2)]
                    table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                    goldPOS[x[0]-1]= [x[0], f'Table:{table}', f'Column:{column}']
                    goldPOS.insert(x[0],[x[0], '>', "SQL function"])
                    i=1
                    for z in goldPOS:
                        goldPOS[i-1]= [i, z[1], z[2]]
                        i+=1
                if x[1] in ["younger"]:
                    embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                    embedAllCols= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                    sim2 =util.dot_score(embed, embedAllCols)
                    column = AllColumns[torch.argmax(sim2)]
                    table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                    goldPOS[x[0]-1]= [x[0], f'Table:{table}', f'Column:{column}']
                    goldPOS.insert(x[0],[x[0]+1, '<', "SQL function"])
                    i=1
                    for z in goldPOS:
                        goldPOS[i-1]= [i, z[1], z[2]]
                        i+=1
                if x[1] in ["youngest"]:
                    embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                    embedAllCols= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                    sim2 =util.dot_score(embed, embedAllCols)
                    column = AllColumns[torch.argmax(sim2)]
                    table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                    goldPOS[x[0]-1]= [x[0], f'Table:{table}', f'Column:{column}']
                    goldPOS.insert(x[0],[x[0]+1, 'MIN', "SQL function"])
                    i=1
                    for z in goldPOS:
                        goldPOS[i-1]= [i, z[1], z[2]]
                    
                if x[1] in ["oldest"]:
                    embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                    embedAllCols= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                    sim2 =util.dot_score(embed, embedAllCols)
                    column = AllColumns[torch.argmax(sim2)]
                    table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                    goldPOS[x[0]-1]= [x[0], f'Table:{table}', f'Column:{column}']
                    goldPOS.insert(x[0],[x[0]+1, 'MAX', "SQL function"])
                    i=1
                    for z in goldPOS:
                        goldPOS[i-1]= [i, z[1], z[2]]
                        i+=1
                    

    return



def afterBroke(goldPOS,memory):
    if (memory["ofCount"]==0):
        memory["MainTableAfterBreak"]= None
        memory["idxMainTableAfterBreak"]= goldPOS[-1][0]-1
        for x in goldPOS:
            if (x[0] > memory["phraseBroke"]) & (x[2] in ["NOUN", "PatternEmbed"]) &(x[1] not in to_ignore):
                embed = SimilarityModel.encode(x[1], convert_to_tensor=True,  normalize_embeddings=True)
                embedTables = SimilarityModel.encode((list(tableColumns.keys())), convert_to_tensor=True, normalize_embeddings=True)
                simTable= util.dot_score(embed,embedTables)
                embedAllcols= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                simAllcols= util.dot_score(embed, embedAllcols)
                if (torch.max(simTable)>=0.75) & (torch.max(simAllcols)<0.81):
                    table= tables[0][torch.argmax(simTable)]
                    goldPOS[x[0]-1]= [x[0], f"Table:{table}", "Reference Table after break pattern"]
                    if memory["MainTableAfterBreak"]== None:
                        memory["MainTableAfterBreak"]= table
                        memory["idxMainTableAfterBreak"]= x[0]-1
                if memory["MainTableAfterBreak"]== None:
                    if (torch.max(simAllcols)>=0.81):
                        column = AllColumns[torch.argmax(simAllcols)]
                        table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                        goldPOS[x[0]-1]= [x[0], f'Table:{table}', f"Column:{column}"]
                    if (torch.max(simTable)<0.80) & (torch.max(simAllcols)<0.81):
                        column = AllColumns[torch.argmax(simAllcols)]
                        table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                        goldPOS[x[0]-1]= [x[0], f'Table:{table}', f"Column:{column}"]
                if (memory["MainTableAfterBreak"]!= None) &(x[0]-1 != memory["idxMainTableAfterBreak"]) :
                    embedTableCols= SimilarityModel.encode(tableColumns[memory["MainTableAfterBreak"]], convert_to_tensor=True, normalize_embeddings=True)
                    sim1 = util.dot_score(embed, embedTableCols)
                    if torch.max(sim1)>=0.75:
                        goldPOS[x[0]-1]= [x[0], f'Table:{memory["MainTableAfterBreak"]}', f'Column:{tableColumns[memory["MainTableAfterBreak"]][torch.argmax(sim1)]}']
                    if torch.max(sim1)<0.75:
                        column = AllColumns[torch.argmax(simAllcols)]
                        table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                        goldPOS[x[0]-1]= [x[0], f'Table:{table}', f"Column:{column}"]

                
    return

def untilBroke(goldPOS, memory):
    if (memory["ofCount"]==0):
        for x in goldPOS:
            if (x[0]<=memory["phraseBroke"]) & (x[2] in ["NOUN", "PatternEmbed"]) & (x[1] not in to_ignore):
                embed = SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                embedTables = SimilarityModel.encode(list(tableColumns.keys()), convert_to_tensor=True,normalize_embeddings=True)
                simTable= util.dot_score(embed,embedTables)
                embedAllcols= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                simAllcols= util.dot_score(embed, embedAllcols)
                if memory["MainTableAfterBreak"] != None:
                    embedTableCols= SimilarityModel.encode(tableColumns[memory["MainTableAfterBreak"]][1:], convert_to_tensor=True, normalize_embeddings=True)
                    sim1 = util.dot_score(embed,embedTableCols)
                    if torch.max(sim1)>0.68:
                        goldPOS[x[0]-1]= [x[0], f'Table:{memory["MainTableAfterBreak"]}', f'Column:{tableColumns[memory["MainTableAfterBreak"]][1:][torch.argmax(sim1)]}']
                    if torch.max(sim1)<=0.68:
                        if (torch.max(simTable)>0.82) & (torch.max(simAllcols)<0.75):
                            table= tables[0][torch.argmax(simTable)]
                            goldPOS[x[0]-1]= [x[0], f"Table:{table}", "Reference Table before break pattern"]
                        if (torch.max(simTable)<=0.82):
                            column = AllColumns[torch.argmax(simAllcols)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[x[0]-1]= [x[0], f'Table:{table}', f"Column:{column}"]
                        if (torch.max(simAllcols)>=0.75):
                            column = AllColumns[torch.argmax(simAllcols)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[x[0]-1]= [x[0], f'Table:{table}', f"Column:{column}"]
                if memory["MainTableAfterBreak"]== None:
                        if (torch.max(simTable)>0.82) & (torch.max(simAllcols)<0.81):
                            table= tables[0][torch.argmax(simTable)]
                            goldPOS[x[0]-1]= [x[0], f"Table:{table}", "Reference Table before break pattern"]
                        if (torch.max(simTable)<=0.82):
                            column = AllColumns[torch.argmax(simAllcols)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[x[0]-1]= [x[0], f'Table:{table}', f"Column:{column}"]
                        if (torch.max(simAllcols)>=0.81):
                            column = AllColumns[torch.argmax(simAllcols)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[x[0]-1]= [x[0], f'Table:{table}', f"Column:{column}"]
    return

def notIdentifiedWithutOf(goldPOS, memory):
    if memory["ofCount"]==0:
        for x in goldPOS:
            if x[1].startswith("Table"):
                memory["not_identified"]= 0
                break
            else: 
                memory["not_identified"]=1
    return

def yearWithoutof(goldPOS, memory):
    if memory["ofCount"]==0:
        for x in goldPOS:
            if (x[0]>memory["phraseBroke"]) & (x[2].startswith("Column")):
                break
            if (x[0]>memory["phraseBroke"]) & (x[2]== "NUM") & (len(x[1])==4):
                match = re.match(r'.*([1-9][0-9]{3})', x[1])
                if match != None:
                    num = match.group(1)
                    embed = SimilarityModel.encode(num, convert_to_tensor=True, normalize_embeddings=True)
                    embedAllCols= SimilarityModel.encode(AllColumns,  convert_to_tensor=True, normalize_embeddings=True)
                    sim = util.dot_score(embed, embedAllCols)
                    if torch.max(sim)>0.45:
                        column = AllColumns[torch.argmax(sim)]
                        table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                        goldPOS.insert(x[0]-1,[x[0], f'Table:{table}', f"Column:{column}"])
                        i=1
                        for z in goldPOS:
                            goldPOS[i-1]= [i, z[1], z[2]]
                            i+=1
                        break
    return



def TruncateAggColumnsWithoutOF(memory, goldPOS):
    if memory["ofCount"]==0:
        idx= []
        for x in goldPOS:
            if (x[2] =="SQL function") & (x[1] in ["AVG", "MAX", "MIN", "SUM"]) :
                idx.append(x[0]-1)
        for index in idx:
            for x in goldPOS:
                if ((x[0]-1)>=index ) & (x[2] in ["NOUN", "PatternEmbed"]) & (x[1]  not in to_ignore):
                    if goldPOS[index][1]== "AVG":
                        new_word= f"average_{x[1]}"
                        new_word2= f"mean_{x[1]}"
                    if goldPOS[index][1]== "SUM":
                        new_word= f"total_{x[1]}"
                        new_word2= f"sum_{x[1]}"
                    if goldPOS[index][1] not in ["AVG", "SUM"]: 
                        break
                    embedingOriginalWord= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                    embedNewWord=SimilarityModel.encode(new_word, convert_to_tensor=True, normalize_embeddings=True)
                    embedAllcolumns= SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                    simOriginal= util.dot_score(embedingOriginalWord, embedAllcolumns)
                    simNewWord= util.dot_score(embedNewWord, embedAllcolumns)
                    if torch.max(simOriginal)> torch.max(simNewWord) :
                        pass
                    if torch.max(simNewWord)> torch.max(simOriginal):
                        embedNewWord2= SimilarityModel.encode(new_word2, convert_to_tensor=True, normalize_embeddings=True)
                        simNewWord2= util.dot_score(embedAllcolumns, embedNewWord2)
                        if torch.max(simNewWord2)> torch.max(simNewWord):
                            goldPOS[index]= [index, "ignore", "Delted by Aggregation Truncation"]
                            column = AllColumns[torch.argmax(simNewWord2)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[x[0]-1]= [x[0], f"Table:{table}", f"Column:{column}"]
                        if torch.max(simNewWord)> torch.max(simNewWord2):
                            goldPOS[index]= [index, "ignore", "Delted by Aggregation Truncation"]
                            column = AllColumns[torch.argmax(simNewWord)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[x[0]-1]= [x[0], f"Table:{table}", f"Column:{column}"]
                    break
                if ((x[0]-1)>= index ) & (x[2].startswith("Column:")):
                    originalWord= memory["goldPOScopy"][x[0]-1][1]
                    if goldPOS[index][1]== "AVG":
                        new_word= f"average_{originalWord}"
                        new_word2= f"mean_{originalWord}"
                    if goldPOS[index][1]== "SUM":
                        new_word= f"total_{originalWord}"
                        new_word2= f"sum_{originalWord}"
                    if goldPOS[index][1] not in ["SUM", "AVG"]: 
                        break
                    embedAllcolumns = SimilarityModel.encode(AllColumns, convert_to_tensor=True, normalize_embeddings=True)
                    embedingOriginalWord= SimilarityModel.encode(originalWord, convert_to_tensor=True, normalize_embeddings=True)
                    embedingNewWord= SimilarityModel.encode(new_word, convert_to_tensor=True, normalize_embeddings=True)
                    simOriginalWord= util.dot_score(embedingOriginalWord, embedAllcolumns)
                    simNewWord= util.dot_score(embedingNewWord, embedAllcolumns)
                    if torch.max(simOriginalWord)> torch.max(simNewWord):
                        pass
                    if torch.max(simNewWord)> torch.max(simOriginalWord):
                        embedNewWord2= SimilarityModel.encode(new_word2, convert_to_tensor=True, normalize_embeddings=True)
                        simNewWord2= util.dot_score(embedAllcolumns, embedNewWord2)
                        if torch.max(simNewWord2)> torch.max(simNewWord):
                            goldPOS[index]= [index, "ignore", "Delted by Aggregation Truncation"]
                            column = AllColumns[torch.argmax(simNewWord2)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[x[0]-1]= [x[0], f"Table:{table}", f"Column:{column}"]
                        if torch.max(simNewWord)> torch.max(simNewWord2):
                            goldPOS[index]= [index, "ignore", "Delted by Aggregation Truncation"]
                            column = AllColumns[torch.argmax(simNewWord)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            goldPOS[x[0]-1]= [x[0], f"Table:{table}", f"Column:{column}"]
                    break
    return





def SelectClauseWithoutOf(goldPOS, memory):
    memory["SelectClause"]= []
    memory["Tables"]= []
    cols= []
    if memory["ofCount"]==0:
        for x in goldPOS:
            if (x[0]<= memory["phraseBroke"]):
                if (x[2].startswith("Column:")) & (x[2][7:] not in cols):
                    column= str.lower(x[2][7:])
                    table= x[1][6:]
                    cols.append(column)
                    if (column in  memory["SelectClause"]) | (f"{table}.{column}" in memory["SelectClause"]):
                        pass
                    else: memory["SelectClause"].append(column)
                    if table not in memory["Tables"]:
                        memory["Tables"].append(table)
            if (x[0]> memory["phraseBroke"]):
                if (x[2] in ["VERB"]) | (x[1] in ["which", "whose", "where", "what", "with", "from"]):
                    break
                if x[1]== "and":
                    for c in goldPOS:
                        if (c[0]>x[0])&(c[2].startswith("Column")):
                            if (str.lower(f"{c[2][7:]}") in memory["SelectClause"]) | (f"{c[1][6:]}.{str.lower(c[2][7:])}" in memory["SelectClause"]):
                                pass
                            else:
                                memory["SelectClause"].append(f"{c[1][6:]}.{str.lower(c[2][7:])}")
                            break  
    
        if len(memory["Tables"])>1:
            i = 0
            while i<len(memory["Tables"]):
                memory["SelectClause"][i]= f'{memory["Tables"][i]}.{memory["SelectClause"][i]}'
                i+=1

        for x in goldPOS:
            if (x[2].startswith("Column:") | (x[2].startswith("Reference")) | ((x[2].startswith("Main")))):
                if x[1][6:] not in memory["Tables"]:
                    memory["Tables"].append(x[1][6:])
            
    return

def andOrWhereWithoutOF(goldPOS, memory):
    if memory["ofCount"]==0:
        memory["or"] = False
        for x in goldPOS:
            if (x[0]>memory["phraseBroke"]) & (x[1].startswith("Table")| (x[2] == "PROPN")) :
                for c in goldPOS:
                    if (c[0]>=x[0]) & (c[1]in ["or", "either"]):
                        memory["or"] = True
                        break

    return


def WhereClauseWithoutOF(goldPOS, memory):
    if memory["ofCount"]==0:
        memory["WhereClause"] = ["WHERE"]
        for x in goldPOS:
            where= None
            values= []
            if (x[0] >= memory["phraseBroke"]+1) & (x[2] == "SQL function") & (x[1] not in ["=", "!="]):
                if goldPOS[x[0]][1]in["!", "?", "."]:
                    break
                if goldPOS[x[0]-2][1]== "at":
                    continue
                if goldPOS[x[0]-2][1] in having:
                    break
                if goldPOS[x[0]-2][2] =="NUM":
                    break
                if x[1] == "BETWEEN":
                    break
                for z in goldPOS:
                    # if (z[0]>memory["phraseBroke"]+1) & (z[0]< x[0]) &(z[1] in ["which", "what", "where", "with", "whose"]):
                    #     break
                    if (z[0]>memory["phraseBroke"]+1) & (z[0]< x[0]) & (z[1].startswith("Table:")) & (z[2].startswith("Column:")) & (f"{z[1][6:]}.{z[2][7:]}" not in memory["WhereClause"]):
                        table= z[1][6:]
                        column = z[2][7:]
                        where= f"{table}.{column}"
                        for c in goldPOS:
                            if (c[0]>= z[0])& (c[0]<=x[0]):
                                if c[2] == "NUM":
                                    values=[c[1]]
                    if (z[0]>x[0]) :
                        if z[2] == "SQL function":
                            break
                    if (z[0]>x[0])& (z[2] in ["NUM", "Value"]):
                        values.append(z[1])
                if (len(values)==1) & (where != None):
                    memory["WhereClause"].append(f"{where}")
                    memory["WhereClause"].append(f"{x[1]} '{values[0]}' ")
                    values.pop(0)

                if where == None:
                    for z in goldPOS:
                        if (z[0]>=x[0]) & (z[1].startswith("Table:")) & (z[2].startswith("Column:")):
                            table= z[1][6:]
                            column = z[2][7:]
                            where= f"{table}.{column}"
                            break
                    if where is not None:
                        memory["WhereClause"].append(where)
                if (len(memory["WhereClause"])==2) & (len(values)>=1):
                    flag= False
                    for z in memory["WhereClause"]:
                        if ("MAX" in z) or ("MIN" in z) or ("SUM" in z):
                            flag=True
                            break
                    if flag== False:
                        memory["WhereClause"].append(x[1])
                        memory["WhereClause"].append(values[0])
                if len(memory["WhereClause"])<3:
                    if (where !=None) & (x[1] in ["MAX", "MIN","AVG", "SUM"]) & (where != "WHERE"):
                        memory["SelectClause"].append(f"{x[1]}({where})")
                        memory["WhereClause"]= ["WHERE"]
                    if where== None:
                        pass
                                
        if len(memory["WhereClause"])==1:
            func = None
            internalFlagValue= False
            internalFlagColumn= False
            for c in goldPOS:
                where= None
                values= []
                if (c[0]>memory["phraseBroke"]) & (c[2] == "Value"):
                    internalFlagValue= True
                    val= c[1]
                    validx= c[0]
                if (c[0]>memory["phraseBroke"]) & (c[2]== "SQL function"):
                    break
                if (c[0]> memory["phraseBroke"]) & (c[2].startswith("Column:")):
                    table= c[1][6:]
                    column = c[2][7:]
                    where= f"{table}.{column}"
                    whereInternalFlag= f"{table}.{column}"
                    internalFlagColumn= True
                    for y in goldPOS:
                        if (y[0]>c[0]) & (y[1]=="BETWEEN"):
                            break
                        if (y[0]>c[0])&(y[2].startswith("Column")):
                            break
                        if(y[0]>c[0])& (y[2] in ["NUM", "Value"]):
                            values.append(y[1])
                            if goldPOS[y[0]-2][1] in having:
                                break
                            for z in goldPOS:
                                if ((z[0]>memory["phraseBroke"]) & (z[0]<y[0]) & (z[1] in notIN)):
                                    func= "!="
                                if (z[0]>memory["phraseBroke"])  & (z[0]<y[0]) & (z[1] in[">", "<"]):
                                    func=f"{z[1]}"
                    if len(values)>=1:
                        for value in values:
                            if func==None:
                                memory["WhereClause"].append(f"{where}")
                                memory["WhereClause"].append(f" = '{value}' ")
                            if func!=None:
                                memory["WhereClause"].append(f"{where}")
                                memory["WhereClause"].append(f" {func} '{value}' ")
            if (internalFlagValue==True) & (internalFlagColumn==False):
                    for z in goldPOS:
                        if (z[0]<= validx) &( z[0] > memory["phraseBroke"]) & (z[1] in notIN):
                            func= "!="
                    if memory["MainTableAfterBreak"]!= None:
                        embed = SimilarityModel.encode(val, convert_to_tensor=True, normalize_embeddings=True)
                        embedMainTable = SimilarityModel.encode(tableColumns[memory["MainTableAfterBreak"]], convert_to_tensor=True, normalize_embeddings=True)
                        sim= util.dot_score(embed, embedMainTable)
                        if torch.max(sim)>0.3:
                            table = memory["MainTableAfterBreak"]
                            column = tableColumns[memory["MainTableAfterBreak"]][torch.argmax(sim)]
                            where= f"{table}.{column}"
                            if func== None:
                                memory["WhereClause"].append(f"{where}")
                                memory["WhereClause"].append(f" = {val}")
                            if func!= None:
                                memory["WhereClause"].append(f"{where}")
                                memory["WhereClause"].append(f" {func} {val}")
                    if memory["MainTableAfterBreak"]== None:
                        embed = SimilarityModel.encode(val, convert_to_tensor=True, normalize_embeddings=True)
                        embedAllColumns= SimilarityModel.encode(AllColumns,convert_to_tensor=True, normalize_embeddings=True)
                        sim= util.dot_score(embed, embedAllColumns)
                        if torch.max(sim)>0.3:
                            column = AllColumns[torch.argmax(sim)]
                            table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                            where= f"{table}.{column}"
                            if func== None:
                                memory["WhereClause"].append(f"{where}")
                                memory["WhereClause"].append(f" = {val}")
                            if func!= None:
                                memory["WhereClause"].append(f"{where}")
                                memory["WhereClause"].append(f" {func} {val}")
            if (internalFlagColumn==True) & (internalFlagValue==False):
                flag= False
                for z in goldPOS:
                    if (z[0]>memory["phraseBroke"]) & (z[2]== "SQL function"):
                        break
                    if (z[0]>memory["phraseBroke"]) & (z[2] =="NUM"):
                        if goldPOS[z[0]-2][1] in having:
                            break
                        val = z[1]
                        validx= z[0]
                        internalFlagValue=True
                if internalFlagValue==True:
                    for z in goldPOS:
                            if (z[0]<= validx) &( z[0] > memory["phraseBroke"]) & (z[1] in notIN):
                                func= "!="
                    for value in memory["WhereClause"]:
                        if val in value:
                            flag= True
                    if (func== None) & (flag==False):
                        memory["WhereClause"].append(f"{whereInternalFlag}")
                        memory["WhereClause"].append(f" = {val}")
                    if (func!= None )& (flag==False):
                        memory["WhereClause"].append(f"{whereInternalFlag}")
                        memory["WhereClause"].append(f" {func} {val}")
                            
                            
                    
                

    return



def OrderByWithoutOf(goldPOS,memory):
    if memory["ofCount"]==0:
        orderBy= None
        memory["orderBy"]= [] 
        memory["orderAggFlag"] = False
        memory["specialFlagOrder"]=False
        flag= False   
        numIDX= None  
        for x in goldPOS:
            if (x[0]>=memory["phraseBroke"]) & (x[1] in order):
                orderBy= x[0]-1
                for z in goldPOS:
                    if (z[0]>x[0]) & (z[2]== "SQL function"):
                        memory["orderAggFlag"] = True
                        break
                    if (z[0]>x[0]) & ((z[1] in count) | (z[1] in ["number"])):
                        flag= True
                        numIDX=z[0]
                        break
                    if (z[0]>x[0]) & (z[1].startswith("Table:"))&(z[2].startswith("Column:")):
                        table= z[1][6:]
                        column= z[2][7:]
                        memory["orderBy"].append(f"ORDER BY {table}.{column}")
        if flag== True:
            for x in goldPOS:
                if (x[0]>=numIDX): 
                    if x[1].startswith("Table:") & (not x[2].startswith("Column:")):
                        memory["orderBy"].append(f"ORDER BY COUNT(*) ASC")
                        break
                    if x[1].startswith("Table:") & x[2].startswith("Column:"):
                        table= z[1][6:]
                        column= z[2][7:] 
                        memory["orderBy"].append(f"ORDER BY COUNT({table}.{column}) ASC")

        if len(memory["orderBy"])>0:
            for z in goldPOS:
                if (z[0]>memory["phraseBroke"]) & (z[1] in ["ascending", "descending"]):
                    if z[1] == "ascending":
                        memory["orderBy"].append(f"ASC")
                    if z[1] == "descending":
                        memory["orderBy"].append(f"DESC")
        table = None
        column= None
        if (bool(orderBy) ==True ) & (len(memory["orderBy"])==0)  & (memory["orderAggFlag"] == False):
            for x in goldPOS:
                if (x[2].startswith("Column:")) & (x[0]<=orderBy):
                    table = x[1][6:]
                    column = x[2][7:]
            for z in goldPOS:
                if memory["phraseBroke"]!= goldPOS[-1][0]-1:
                    if  (z[0]>= memory["phraseBroke"])  & (z[1] in ["alphabetical", "alphabetic", "alphabetically"]) :
                        memory["orderBy"]=[f"ORDER BY {table}.{column}"]
                    if (z[0]>= memory["phraseBroke"])  & (z[1] in ["ascending", "descending"]):
                        if z[1] in [ "ascending"]:
                            memory["orderBy"].append(f"ORDER BY  {table}.{column} ASC")
                        if z[1] in ["descending"]:
                            memory["orderBy"].append(f"ORDER BY {table}.{column} DESC")
                if memory["phraseBroke"]== goldPOS[-1][0]-1:
                    if  (z[0]<= memory["phraseBroke"])  & (z[1] in ["alphabetical", "alphabetic", "alphabetically"]) :
                        memory["orderBy"]=[f"ORDER BY {table}.{column}"]
                    if (z[0]<= memory["phraseBroke"])  & (z[1] in ["ascending", "descending"]):
                        if z[1] in [ "ascending"]:
                            memory["orderBy"].append(f"ORDER BY  {table}.{column} ASC")
                        if z[1] in ["descending"]:
                            memory["orderBy"].append(f"ORDER BY {table}.{column} DESC")
            if len(memory["orderBy"])==0:
                for x in goldPOS:
                    if ((x[1] in count) or (x[1] in "number")):
                        flag= True
                        numIDX= x[0]
                    if ((x[2].startswith("Column:")) & (x[0]<=memory["phraseBroke"])):
                        table=x[1][6:]
                        column= x[2][7:]
                if flag== True:
                    for x in goldPOS:
                        if (x[0]>=numIDX): 
                            if (x[1].startswith("Table:") & (not x[2].startswith("Column:"))):
                                memory["orderBy"].append(f"ORDER BY COUNT(*) ASC")
                                break
                            if (x[2].startswith("Column:")):
                                table= z[1][6:]
                                column= z[2][7:] 
                                memory["orderBy"].append(f"ORDER BY COUNT({table}.{column}) ASC")
            
                for z in goldPOS:
                    if (z[0]>= memory["phraseBroke"])  & (z[1] in ["ascending", "descending"]):
                        if z[1] == "ascending":
                            memory["orderBy"].append(f"ORDER BY  {table}.{column} ASC")
                        if z[1] == "descending":
                            memory["orderBy"].append(f"ORDER BY {table}.{column} DESC")
        
        
        if (bool(orderBy) == False ) & (len(memory["orderBy"])==0) & (memory["orderAggFlag"] == False):
            for x in goldPOS:
                if (x[1] in  order):
                    orderBy = x[0]
                    break
            if bool(orderBy) == True:
                numIDX= []
                for x in goldPOS:
                    if (x[0]>=orderBy) & (x[1] in ["number"]):
                        numIDX.append(x[0])
                        break
                if len(numIDX)>0:
                    for x in goldPOS:
                        if (x[0]>numIDX[0]) & ((x[2].startswith("Reference")) | (x[2].startswith("Main "))):
                            memory["orderBy"].append(f"ORDER BY COUNT(*)ASC")
                            break
                        if (x[0]>numIDX[0])&(x[1].startswith("Table:") & x[2].startswith("Column:")):
                            table= x[1][6:]
                            column= x[2][7:] 
                            memory["orderBy"].append(f"ORDER BY COUNT({table}.{column}) ASC")
                            break
        if (bool(orderBy) == True) & (memory["orderAggFlag"] == True):
            for c in goldPOS:
                if c[2] in "SQL function":
                    if c[1] in ["MAX", "MIN"]:
                        for z in goldPOS:
                            if (z[0]>=c[0]) & (z[2].startswith("Column:")):
                                table = z[1][6:]
                                column= z[2][7:]
                                break
                        if c[1]== "MAX":
                            memory["orderBy"]= []
                            memory["orderBy"].append(f"ORDER BY MAX({table}.{column})")
                        if c[1]== "MIN":
                            memory["orderBy"]= []
                            memory["orderBy"].append(f"ORDER BY MIN({table}.{column})")
                        for z in goldPOS:
                            if z[1] in ["ascending", "descending"]:
                                if z[1] == "ascending":
                                    memory["orderBy"].append("ASC")
                                if z[1]== "descending":
                                    memory["orderBy"].append("DESC")
        if (bool(orderBy)== False) & (memory["orderAggFlag"]== False):
            groupTable=None
            groupCol=None
            func= None
            flagMAXMIN= False
            for x in goldPOS:
                if x[2] == "SQL function":
                    if x[1] in ["MAX", "MIN"]:
                        func=x[1]
                        for z in goldPOS:
                            if (z[0]>= x[0]) & (z[1] == "number"):
                                for c in goldPOS:
                                    if (c[0]>z[0])&(c[1].startswith("Table")):
                                        table= c[1][6:]
                                        break
                                    if (c[0]>z[0])&(c[1].startswith("Table")) & (c[2].startswith("Column")):
                                        groupTable= c[1][6:]
                                        groupCol= c[2][7:]
                                        flagMAXMIN= True
                                        break
                                if flagMAXMIN==False:
                                    for c in goldPOS:
                                        if(c[0]<z[0])& (c[2].startswith("Column")):
                                            groupTable= c[1][6:]
                                            groupCol= c[2][7:]
                                            break
                                break
            if(groupCol!=None)& (groupTable!=None) & (func!=None):
                if func== "MAX":
                    memory["orderBy"].append(f"ORDER BY COUNT(*)")
                    memory["orderBy"].append(f"DESC LIMIT 1")
                    memory["specialFlagOrder"]= True
                    memory["groupByClause"]= f"GROUP BY {groupTable}.{str.lower(groupCol)}"
                if func== "MIN":
                    memory["orderBy"].append(f"ORDER BY COUNT(*)")
                    memory["orderBy"].append(f"ASC LIMI 1")
                    memory["specialFlagOrder"]= True
                    memory["groupByClause"]= f"GROUP BY {groupTable}.{str.lower(groupCol)}"
        
        if len(memory["orderBy"])>=1:
            memory["orderBy"] = " ".join(memory["orderBy"])
            

    return




def aggregationsWithoutOF(goldPOS,memory):
    if memory["ofCount"]==0:
        memory["specialFlag"]= False
    ############# Before of 
        memory["aggregations"]= {}
        specialFlag = False
        for x in goldPOS:
            col = None
            colonly= None
            if (x[2] == "SQL function") & (x[0]<= memory["phraseBroke"]):
                if (x[1]  in ["MIN", "MAX", "AVG", "DISTINCT", "SUM"]):
                    for z in goldPOS:
                        if (z[1] in ["common", "usual", "frequent", "frequently", "commonly","usually", "widely"]):
                            break
                        if(z[0]>=x[0]) & (z[2].startswith("Column:")):
                            col =f"{z[1][6:]}.{z[2][7:]}"
                            colonly = f"{z[2][7:]}"
                            memory["aggregations"][x[1]]= f"{col}"
                            break
                    if (col!= None) & (colonly!=None):
                        if( str.lower(col) in memory["SelectClause"]):
                            memory["SelectClause"].pop(memory["SelectClause"].index(str.lower(col)))   
                        if  (str.lower(colonly) in memory["SelectClause"]):
                            memory["SelectClause"].pop(memory["SelectClause"].index(str.lower(colonly)))
        if len(memory["aggregations"])>=1:
            select_clause = memory["SelectClause"]
            aggregations = memory.get('aggregations', {})
            agg_values = []
            for agg_func, agg_key in aggregations.items():
                if (agg_key in select_clause):
                    select_clause[select_clause.index(agg_key)] = f"{agg_func}({agg_key})"
                    specialFlag= True
                if ((list(agg_key.split("."))[1]) in select_clause) & (specialFlag==False):
                    select_clause[select_clause.index(list(agg_key.split("."))[1])] = f"{agg_func}({agg_key})"
                else:
                    agg_values.append(f"{agg_func}({agg_key})")
            select_clause.extend(agg_values)
    ################# After OF
        memory["aggregationsAfterOf"]= {}
        num= None
        internalFlag= False
        for x in goldPOS:
            if (x[2] == "SQL function") & (x[0]>memory["phraseBroke"]) & ((x[1] in ["MIN", "MAX", "AVG", "DISTINCT"])):
                if (goldPOS[x[0]-2][1]== "at"):
                    break
                if (goldPOS[x[0]][2].startswith("Reference")) & (x[1] in ["MIN", "MAX"]):
                        for c in goldPOS:
                            if (c[0]<x[0]) & (c[2].startswith("Column")):
                                if x[1] == "MAX":
                                    table = c[1][6:]
                                    column= c[2][7:]
                                    memory["groupByClause"] =f"GROUP BY {table}.{str.lower(column)}"
                                    memory["orderBy"].append("ORDER BY COUNT(*) DESC LIMIT 1")
                                    memory["specialFlag"] = True
                                if x[1] == "MIN":
                                    table = c[1][6:]
                                    column = c[2][7:]
                                    memory["groupByClause"]= f"GROUP BY {table}.{str.lower(column)}"
                                    memory["orderBy"].append("ORDER BY COUNT(*) ASC LIMIT 1")
                                    memory["specialFlag"] = True
                        break
                for z in goldPOS:
                    if (z[0]>x[0]) & (z[1] in ["MIN", "MAX", "AVG","SUM"]):
                        internalFlag= True
                        func= x[1]
                        internalfunc= z[1]
                        break
                    if (z[0]<x[0]) & (z[2] in ["NUM"]):
                        num= z[1]
                    if (z[0]>=x[0]) & (z[2] in ["common", "usual", "frequent", "frequently", "commonly","usually", "widely"]):
                        break
                    if(z[0]>=x[0]) & (z[2].startswith("Column:")):
                        memory["aggregationsAfterOf"][x[1]]= f"{z[1][6:]}.{z[2][7:]}"
                        break
                    
        if (len(memory["aggregationsAfterOf"])>=1) & (memory["orderAggFlag"]==False) :
            memory["orderBy"]= []
            oderby_clause = memory["orderBy"]
            aggregations = memory.get("aggregationsAfterOf", {})
            agg_values = []
            for agg_func, agg_key in aggregations.items():
                if (agg_func in["MAX", "MIN"]) & (internalFlag== False):                            
                    if (agg_func == "MAX"):
                        if num == None:
                            agg_values.append(f"ORDER BY {agg_key}")
                            agg_values.append(f"DESC LIMIT 1")
                        if num != None:
                            agg_values.append(f"ORDER BY {agg_key}")
                            agg_values.append(f"DESC LIMIT {w2n.word_to_num(num)}")
                        for v in memory["SelectClause"]:
                            if v.startswith("MAX"):
                                idx = memory["SelectClause"].index(v)
                                memory["SelectClause"].pop(idx)
                    if agg_func == "MIN":
                        if num == None:
                            agg_values.append(f"ORDER BY {agg_key}")
                            agg_values.append(f"ASC LIMIT 1")
                        if num != None:
                            agg_values.append(f"ORDER BY {agg_key}")
                            agg_values.append(f"ASC LIMIT {w2n.word_to_num(num)}")
                        for v in memory["SelectClause"]:
                            if v.startswith("MIN"):
                                idx = memory["SelectClause"].index(v)
                                memory["SelectClause"].pop(idx)
                if (agg_func in ["AVG", "DISTINCT"]) & (internalFlag==True):
                    if func == "MAX":
                        if num == None:
                            agg_values.append(f"ORDER BY {internalfunc}({agg_key})")
                            agg_values.append(f"DESC LIMIT 1")
                        if num != None:
                            agg_values.append(f"ORDER BY {internalfunc}({agg_key})")
                            agg_values.append(f"DESC LIMIT {w2n.word_to_num(num)}")
                    if func == "MIN":
                        if num == None:
                            agg_values.append(f"ORDER BY {internalfunc}({agg_key})")
                            agg_values.append(f"ASC LIMIT 1")
                        if num != None:
                            agg_values.append(f"ORDER BY {internalfunc}({agg_key})")
                            agg_values.append(f"ASC LIMIT {w2n.word_to_num(num)}")
            oderby_clause.extend(agg_values)
            if len(memory["orderBy"])>1:
                memory["orderBy"] = " ".join(memory["orderBy"])


    return



def specialAggregationsWithoutOF(goldPOS,memory):
    if (memory["ofCount"]==0) & (len(memory["WhereClause"])==1):
        values= []
        where= None
        for x in goldPOS:
            if (x[0]>memory["phraseBroke"])& (x[1]=="BETWEEN") & (x[2]=="SQL function"):
                betweenIDX=x[0]
                for z in goldPOS:
                    if (z[0]<x[0]) & (z[2].startswith("Column")):
                        table= z[1][6:]
                        column= z[2][7:]
                        where= f"{table}.{column}"
                    if (z[0]>x[0]) & (z[2] in ["NUM"]):
                        values.append(z[1])
        if (len(values)>0) & (where != None) :
            memory["WhereClause"].append(f"{where} ")
            for c in values:
                if "." in c:
                    flag = True  
                    break
            if flag== False:
                memory["WhereClause"].append(f"BETWEEN {w2n.word_to_num(values[0])} AND {w2n.word_to_num(values[-1])}")
            if flag == True:
                memory["WhereClause"].append(f"BETWEEN {values[0]} AND {values[-1]}")
        if (len(values)>0) & (where == None) :
            for z in goldPOS:
                if (z[0]>betweenIDX) & z[2].startswith("Column"):
                    table= z[1][6:]
                    column= z[2][7:]
                    where= f"{table}.{column}"
            if where!=None:
                memory["WhereClause"].append(f"{where} ")
                for c in values:
                    if "." in c:
                        flag = True  
                        break
                if flag== False:
                    memory["WhereClause"].append(f"BETWEEN {w2n.word_to_num(values[0])} AND {w2n.word_to_num(values[-1])}")
                if flag == True:
                    memory["WhereClause"].append(f"BETWEEN {values[0]} AND {values[-1]}")

        flag= False
        idx= None
        for x in goldPOS:
            if (x[0]<=memory["phraseBroke"]) & (x[1] in ["MAX", "MIN"])& (x[2]=="SQL function"):
                for z in goldPOS:
                    if (z[0]>=x[0]) & (z[1] in ["common", "usual", "frequent", "frequently", "commonly","usually", "widely"]):
                        idx= z[0]
                        flag = True
                        break
            if (x[0]>=memory["phraseBroke"]) & (x[1] in ["MAX", "MIN"])& (x[2]=="SQL function"):
                for z in goldPOS:
                    if (z[0]>=x[0]) & (z[1] in ["common", "usual", "frequent", "frequently", "commonly","usually", "widely"]):
                        idx= z[0]
                        flag = True
                        break
        if flag == True:
            flag1= False
            for z in goldPOS:
                if (z[0]>=idx) & (z[2].startswith("Column:")):
                    table = z[1][6:]
                    column = z[2][7:]
                    memory["orderBy"].append(f"ORDER BY COUNT(*) DESC LIMIT 1")
                    memory["groupByClause"]= f"GROUP BY {table}.{str.lower(column)}"
                    memory["specialFlag"]= True
                    flag1= True
                    break
                if (z[0]>=idx) & ((z[2].startswith("Main")) | (z[2].startswith("Reference"))) :
                    for c in goldPOS:
                        if c[2].startswith("Column"):
                            table = c[1][6:]
                            column = c[2][7:]
                            memory["orderBy"].append(f"ORDER BY COUNT(*) DESC LIMIT 1")
                            memory["groupByClause"]= f"GROUP BY {table}.{str.lower(column)}"
                            memory["specialFlag"]= True
                            flag1= True
                            break
            if flag1== False:
                for z in goldPOS:
                    if (z[0]>=idx) & (z[2].startswith("Column")):
                        table = z[1][6:]
                        column = z[2][7:]
                        memory["orderBy"].append(f"ORDER BY COUNT(*) DESC LIMIT 1")
                        memory["groupByClause"]= f"GROUP BY {table}.{str.lower(column)}"
                        memory["specialFlag"]= True
                        flag1= True
                        break
                    if (z[0]<=idx) & ((z[2].startswith("Main")) | (z[1].startswith("Reference"))) :
                        break
            if len(memory["orderBy"])>0:
                memory["orderBy"]= " ".join(memory["orderBy"])
    return



def propnWhereWithoutOF(goldPOS, memory):
    if memory["ofCount"]==0:
        propn ={}
        flag = False
        internalFlag= False
        for x in goldPOS:
            if (x[0]>memory["phraseBroke"]) & (x[2] in ["propnPattern"]):
                function= None
                for z in goldPOS:
                    if (z[0]>memory["phraseBroke"])  & (z[0]<=x[0]) & (z[1] in ["!=", "="]):
                        function = z[1]
                    if (z[0]>memory["phraseBroke"]) & (z[0]<=x[0]) & (z[1] in notIN):
                        function = "!="
                    if (z[0]<=x[0]) & (z[0]> memory["phraseBroke"]) & (z[2].startswith("Column:")) :
                        for c in goldPOS:
                            if (c[0]>=z[0]) &(c[0]<=x[0])& (c[1] in ["<", ">"]) & (c[2]== "SQL function"):
                                internalFlag= True
                                break
                            if (c[0]>memory["phraseBroke"])  & (c[0]<=x[0]) & (c[1] in ["!=", "="]):
                                function = c[1]
                            if (c[0]>memory["phraseBroke"]) & (c[0]<=x[0]) & (c[1] in notIN):
                                function = "!="
                        if internalFlag==True:
                            break
                        if function == None:
                            propn[f"{z[1][6:]}.{z[2][7:]}"]= x[1]
                            memory["WhereClause"].append(f"{z[1][6:]}.{z[2][7:]}")
                            memory["WhereClause"].append(f"""' = {x[1]}'""")
                            flag=True
                        if function!= None:
                            propn[f"{z[1][6:]}.{z[2][7:]} {function}"]= x[1]
                            memory["WhereClause"].append(f"{z[1][6:]}.{z[2][7:]}")
                            memory["WhereClause"].append(f""" {function}  '{x[1]}' """)
                            flag= True
        if flag==False:
            function= None
            for x in goldPOS:
                if (x[0]>memory["phraseBroke"]) & (x[2]in ["propnPattern"]):
                    for z in goldPOS:
                        if (z[0]>memory["phraseBroke"])  & (z[0]<=x[0]) & (z[1] in ["!=", "="]):
                            function = z[1]
                        if (z[0]>memory["phraseBroke"]) & (z[0]<= x[0]) & (z[1] in notIN):
                            function ="!="
                        if (z[0]>= x[0]) & ((z[1] in ["which", "whose", "where","with", "and", "or"]) | (z[2].startswith("SQL function"))):
                            break
                        if (z[0]>=x[0]) & (z[2].startswith("Column:")):
                            if function == None:
                                propn[f"{z[1][6:]}.{z[2][7:]}"]= x[1]
                                memory["WhereClause"].append(f"{z[1][6:]}.{z[2][7:]} =")
                                memory["WhereClause"].append(f" '{x[1]}'")
                                flag=True
                            if function!= None:
                                propn[f"{z[1][6:]}.{z[2][7:]} {function}"]= x[1]
                                memory["WhereClause"].append(f"{z[1][6:]}.{z[2][7:]} {function}  ")
                                memory["WhereClause"].append(f"""'{x[1]}'""")
                                flag= True
                            break
        if flag == False:
            for x in goldPOS:
                if (x[0]>memory["phraseBroke"]) & (x[2]in ["propnPattern"]):
                        embed= SimilarityModel.encode(x[1], convert_to_tensor=True, normalize_embeddings=True)
                        embedAll= SimilarityModel.encode(AllColumns,convert_to_tensor=True, normalize_embeddings=True)
                        sim =util.dot_score(embed, embedAll)
                        column = AllColumns[torch.argmax(sim)]
                        table= [key for key, value_list in tableColumns.items() if column in value_list][0]
                        memory["WhereClause"].append(f"""{table}.{column}""")
                        if function == None:
                            memory["WhereClause"].append(f"""= '{x[1]}'""")
                        if function != None:
                            memory["WhereClause"].append(f"""!= '{x[1]}'""")
                
    return




def joinTablesWithoutOF(goldPOS, memory):
    memory["aliases"]= {}
    if (memory["ofCount"]==0) & (len(set(memory["Tables"]))>1):
        ### Select the ids from the respective tables:
        ids = {}
        AllIDS= []
        pattern = re.compile('.*id$')
        for x in AllTableColumns:
            value= []
            for id in AllTableColumns[x]:
                if ('_id' in str.lower(id)) or ('id_' in str.lower(id)) or (("ID" in (id)))or (("id" in str.lower(id)) & (len(id)==2)) or (("id " in str.lower(id))) or (pattern.match(str.lower(id))):
                    value.append(id)
            AllIDS.append(value)
            ids[x]=value
        
  
        for key, value in ids.items():
            if not value:
                for column_key, column_values in AllTableColumns.items():
                    for z in column_values:
                        if (key != column_key) & (z in AllTableColumns[key]):
                            ids[key].append(z)
                            ids[column_key].append(z)
        for z in range(len(AllIDS)):
            if len(AllIDS[z])==1:
                AllIDS[z]= AllIDS[z][0]
        
        for key , value in ids.items():
            if not value:
                for all_dict_key, all_dict_values in AllTableColumns.items():        
                    if all_dict_key == key:
                        for values in  all_dict_values:
                            if (values in num for num in ["1","2","3","4","5","6","7","8","9","0"]):
                                ids[key].append(values)

        idsToJoin= {}
        for x in ids:
            if x in memory["Tables"]:
                idsToJoin[x]= str.lower(ids[x][0])
        if len(AllIDS)>2:
            embedAllIDS= SimilarityModel.encode(AllIDS, convert_to_tensor=True, normalize_embeddings=True)
            embedIdsToJoin= SimilarityModel.encode([list(idsToJoin.values())], convert_to_tensor=True, normalize_embeddings=True)
            simIDS= util.dot_score(embedIdsToJoin, embedAllIDS)
            specificIDembed= SimilarityModel.encode(AllIDS[torch.argmax(simIDS)], convert_to_tensor=True, normalize_embeddings=True)
            keyTable = None
            for key, value in ids.items():
                if AllIDS[torch.argmax(simIDS)] == value:
                    keyTable = key
                    if keyTable not in memory["Tables"]:
                        memory["Tables"].append(keyTable)
            indexes ={}
            for x in idsToJoin:
                tableIDS= {}
                embed= SimilarityModel.encode(idsToJoin[x], convert_to_tensor=True, normalize_embeddings=True)
                sim= util.dot_score(embed, specificIDembed)
                if type(AllIDS[torch.argmax(simIDS)])!=str:
                    tableIDS[idsToJoin[x]]= AllIDS[torch.argmax(simIDS)][torch.argmax(sim)]
                    indexes[x]= tableIDS
                if type(AllIDS[torch.argmax(simIDS)])==str:
                    tableIDS[idsToJoin[x]]= AllIDS[torch.argmax(simIDS)]
                    indexes[x]= tableIDS
        if len (AllIDS)==2:
            embedAllIDS= SimilarityModel.encode(AllIDS, convert_to_tensor=True, normalize_embeddings=True)
            embedIdsToJoin= SimilarityModel.encode([list(idsToJoin.values())], convert_to_tensor=True, normalize_embeddings=True)
            simIDS= util.dot_score(embedIdsToJoin, embedAllIDS)
            vals= AllIDS[torch.argmax(simIDS)]
            if (type(vals)==list):
                specificIDembed= SimilarityModel.encode(AllIDS[torch.argmax(simIDS)], convert_to_tensor=True, normalize_embeddings=True)
                for key, value in ids.items():
                    if AllIDS[torch.argmax(simIDS)] == value:
                        keyTable = key
                indexes = {}
                for x in idsToJoin:
                    tableIDS= {}
                    embed= SimilarityModel.encode(idsToJoin[x], convert_to_tensor=True, normalize_embeddings=True)
                    sim= util.dot_score(embed, specificIDembed)
                    if type(AllIDS[torch.argmax(simIDS)])!=str:
                        tableIDS[idsToJoin[x]]= AllIDS[torch.argmax(simIDS)][torch.argmax(sim)]
                        indexes[x]= tableIDS
                    if type(AllIDS[torch.argmax(simIDS)])==str:
                        tableIDS[idsToJoin[x]]= AllIDS[torch.argmax(simIDS)]
                        indexes[x]= tableIDS  
            if type(vals)== str:
                indexes ={}
                keyTable = list(idsToJoin.keys())[0]
                for x in idsToJoin:
                    if x != keyTable:
                        indexes[x]= {idsToJoin[x]: idsToJoin[keyTable]}

        i=1
        for z in set(memory["Tables"]):
            memory["aliases"][z]= i
            i+=1
        if len(set(memory["Tables"]))==2:
            if keyTable== None:
                keyTable= list(indexes.keys())[1]
            for x in indexes:
                if x != keyTable:
                    memory["SelectClause"].append(f"""FROM {x} AS T{memory["aliases"][x]} JOIN {keyTable} AS T{memory["aliases"][keyTable]} ON T{memory["aliases"][x]}.{ids[x][0]} = T{memory["aliases"][keyTable]}.{indexes[x][str.lower(ids[x][0])]}""")
        if len(set(memory["Tables"]))>2:
            if keyTable== None:
                keyTable=list(indexes.keys())[1]
            for x in list(indexes.keys())[0:1]:
                if x != keyTable:
                    memory["SelectClause"].append(f"""FROM {x} AS T{memory["aliases"][x]} JOIN {keyTable} AS T{memory["aliases"][keyTable]} ON T{memory["aliases"][x]}.{ids[x][0]} = T{memory["aliases"][keyTable]}.{indexes[x][str.lower(ids[x][0])]}  """)
                    break
            for x in list(indexes.keys())[1:]:
                if x != keyTable:
                    memory["SelectClause"].append(f""" JOIN {x} AS T{memory["aliases"][x]} ON T{memory["aliases"][x]}.{ids[x][0]} = T{memory["aliases"][keyTable]}.{indexes[x][str.lower(ids[x][0])]}  """)
            
            from_index= None
            for x in memory["SelectClause"]:
                if x.startswith("FROM"):
                    from_index= memory["SelectClause"].index(x)
            scnd= " ".join(memory["SelectClause"][from_index:])
            del memory["SelectClause"][from_index:]
            memory["SelectClause"].append(scnd)

    return



def CountClauseWithoutOF(goldPOS, memory):
    if memory["ofCount"]==0:
        c= None
        for x in goldPOS:
            if (x[0]<= memory["phraseBroke"]) & (str.lower(x[1]) in count):
                memory["SelectClause"].insert(0,"COUNT(*)")
                for z in goldPOS:
                    if (z[0]>x[0]) & (z[0]<=memory["phraseBroke"]):
                        if z[1] == "DISTINCT":
                            memory["SelectClause"].pop(memory["SelectClause"].index(f"COUNT(*)"))
                            break
                        if z[1] in ["and", "or"]:
                            break
                        if (z[2].startswith("Column")) :
                            if ((f"{z[1][6:]}.{z[2][7:]}" )in memory["SelectClause"]):
                                memory["SelectClause"].pop(memory["SelectClause"].index(f"{z[1][6:]}.{z[2][7:]}"))
                                break
                            if (f"{z[2][7:]}" in memory["SelectClause"]):
                                memory["SelectClause"].pop(memory["SelectClause"].index(f"{z[2][7:]}"))
                                break

        if len(memory["Tables"])<1:
            for x in goldPOS:
                if (x[2].startswith("Main table")) | (x[2].startswith("Reference")):
                    if x[1] not in memory["Tables"]:
                        memory["Tables"].append(x[1][6:])
        for x in goldPOS:
            if (x[0]<=memory["phraseBroke"]) & ((x[1] == "number") | (x[1] in count)):
                for z in goldPOS:
                    if (z[0]>x[0]) & (z[1]=="DISTINCT") :
                        for y in memory["SelectClause"]:
                            if "DISTINCT" in y:
                                idx= memory["SelectClause"].index(y)
                                tc= memory["SelectClause"][idx][9:-1]
                                memory["SelectClause"].insert(idx, f"COUNT(DISTINCT({tc}))")
                                memory["SelectClause"].pop(idx+1)
                                c=True
                                for y in memory["SelectClause"]:
                                    if "COUNT(*)" in y:
                                        idx= memory["SelectClause"].index(y)
                                        memory["SelectClause"].pop(idx)
                                        break   
                                break
        for x in goldPOS:
            if (x[0]<=memory["phraseBroke"]) & (x[1] == "number"):
                for z in goldPOS:
                    if ((z[0]>x[0]) & (z[2].startswith("Column"))) | ((z[0]>x[0])& (z[1].startswith("Table"))):
                        if any(str.lower(value) in str.lower(z[2][7:]) for value in ["number", "num_", "total", "sum"]):
                            memory["SelectClause"].insert(0,f"{z[1][6:]}.{z[2][7:]}")
                            break
                        else:
                            if (c== None):
                                for y in memory["SelectClause"]:
                                    if str.lower(y).startswith("count"):
                                        pass
                                    else:
                                        memory["SelectClause"].insert(0,"COUNT(*)")
                                        break
                            if (c== None) & (len(memory["SelectClause"])==0):
                                if goldPOS[x[0]-2][1] in ["AVG","SUM", "MAX", "MIN"]:
                                    pass
                                else:
                                    memory["SelectClause"].insert(0, "COUNT(*)")
                            break
        flag= False
        column = None
        for x in goldPOS:
            if (x[0]>= memory["phraseBroke"]) & (x[1] == "number"):
                for z in goldPOS:
                    if (z[0]>memory["phraseBroke"]) & (z[0]<=x[0]) & (z[1] in having):
                        break
                    if (z[0]>memory["phraseBroke"]) & (z[0]>=x[0]) & (z[1].startswith("Table")) & (not z[2].startswith("Column")):
                        for c in goldPOS:
                            if c[2].startswith("Column"):
                                table= c[1][6:]
                                column = c[2][7:]
                                break
                        if column != None:
                            memory["groupByClause"]= f"GROUP BY {table}.{str.lower(column)}"
                            memory["specialFlag"]= True
                        if column == None:
                            memory["groupByClause"]= f"GROUP BY {table}.{str.lower(AllTableColumns[table][0])}"
                            memory["specialFlag"]= True
                        for c in goldPOS:
                            if (c[0]>=memory["phraseBroke"]) & (c[0]<x[0]) & (c[1] in ["and"]):
                                if "COUNT(*)" not in memory["SelectClause"]:
                                    memory["SelectClause"].insert(0,"COUNT(*)")
                                break
                            else:
                                pass
                        flag = True
                    if (z[0]>=x[0]) & (z[1].startswith("Table")) & (z[2].startswith("Column")) & (flag==False):
                        pass



def GroupByClauseWithoutOF(goldPOS, memory):
    if memory["specialFlag"]!=False:
        pass
    if memory["specialFlag"]== False:
        memory["groupByClause"]= []
    flag= False
    if memory["ofCount"]==0:
        for x in goldPOS:
            if (x[0]>memory["phraseBroke"]) & (x[1] in groupBy):
                for z in goldPOS:
                    if (z[0]>=x[0]) & (z[2].startswith("Column:")):
                        table= z[1][6:]
                        column = z[2][7:]
                        memory["groupByClause"]= f"GROUP BY {table}.{str.lower(column)}"
                        break
                break
        for x in goldPOS:
            if (x[0]<=memory["phraseBroke"]) & (x[1] in groupBy):
                for z in goldPOS:
                    if (z[0]>=x[0]) & (z[2].startswith("Column:")):
                        table= z[1][6:]
                        column = z[2][7:]
                        memory["groupByClause"]= f"GROUP BY {table}.{str.lower(column)}"
                        break
                break
        ### Para "each" no caso de o proximo token ser coluna 
        for x in goldPOS:
            if (x[1] in ["each"]) :
                if (goldPOS[x[0]][2].startswith("Column:")):
                    table = goldPOS[x[0]][1][6:]
                    column = goldPOS[x[0]][2][7:]
                    if f"{table}.{column}" not in memory["groupByClause"] :
                        memory["groupByClause"]= f"GROUP BY {table}.{str.lower(column)}"
                        if AllTableColumns[table][0]!= column:
                            if( str.lower(f"{table}.{column}") in memory["SelectClause"]) | (str.lower(column) in memory["SelectClause"]):
                                pass
                            else:
                                memory["SelectClause"].insert(0,f"{table}.{column}")
                        flag= True
                if (goldPOS[x[0]][2].startswith("Reference")) | (goldPOS[x[0]][2].startswith("Main")):
                    table= goldPOS[x[0]][1][6:]
                    column = AllTableColumns[table][0]
                    memory["groupByClause"] = f"GROUP BY {table}.{str.lower(column)}"
                    if AllTableColumns[table][0]!= column:
                        if( str.lower(f"{table}.{column}") in memory["SelectClause"]) | (str.lower(column) in memory["SelectClause"]):
                            pass
                        else:
                            memory["SelectClause"].insert(0,f"{table}.{column}")
                    flag= True
        ### Para "each" no caso de nao existir proximo token que seja coluna
        if flag == False:
            for x in goldPOS:
                if (x[1] in ["each"]):
                    for c in goldPOS:
                        if (c[0]<x[0]) & (c[2].startswith("Column")):
                            memory["groupByClause"]= f"GROUP BY {str.lower(c[2][7:])}"
                            if AllTableColumns[c[1][6:]][0]!= c[2][7:]:
                                if ( str.lower(f"{c[1][6:]}.{c[2][7:]}") in memory["SelectClause"])| (str.lower(f"{c[2][7:]}") in memory["SelectClause"]):
                                    pass
                                else:
                                    memory["SelectClause"].insert(0,f"{c[1][6:]}.{c[2][7:]}")

    return


def havingClauseWithoutOF(goldPOS, memory):
    if memory["ofCount"]==0:
        memory["having"]= []
        generalFlag= False
        flag= False
        for x in goldPOS:
            if (x[0]>memory["phraseBroke"]) & (x[1] in having):
                for c in goldPOS:
                    if (c[0]>=x[0]) & (c[2].startswith("Column")):
                        break
                    if (c[0]>=x[0]) & (c[2] == "SQL function"):
                        if (c[1] =="MIN") & (goldPOS[c[0]-2][1] == "at"):
                            for z in goldPOS:
                                if (z[0]>c[0]) & (z[2] == "NUM"):
                                    memory["having"] = [f"HAVING COUNT(*) >= {w2n.word_to_num(z[1])}"]
                                    if len(memory["Tables"])>0:
                                        for c in goldPOS:
                                            if (c[0]>z[0])&(c[1] in to_ignore):
                                                break
                                            if (c[0]>z[0]) & (c[1].startswith("Table")):
                                                if c[2].startswith("Column"):
                                                    table= c[1][6:]
                                                    col= c[2][7:]
                                                    memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                    flag= True
                                                if not c[2].startswith("Column"):
                                                    for d in goldPOS:
                                                        if (d[2].startswith("Column")) & (d[1]== c[1]):
                                                            table= d[1][6:]
                                                            col= d[2][7:]
                                                            memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                            flag=True
                                                            break
                                        if len(memory["groupByClause"])<1:
                                            table = memory["Tables"][0]
                                            col = AllTableColumns[table][0]
                                            memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                            flag =True
                                    break
                        if (c[1] =="MIN") & (goldPOS[c[0]-2][1] != "at"):
                            for z in goldPOS:
                                if (z[0]>c[0]) & (z[2] =="NUM"):
                                    memory["having"] = [f"HAVING COUNT(*) <= {w2n.word_to_num(z[1])}"]
                                    if len(memory["Tables"])>0:
                                        for c in goldPOS:
                                            if (c[0]>z[0])&(c[1] in to_ignore):
                                                break
                                            if (c[0]>z[0]) & (c[1].startswith("Table")):
                                                if c[2].startswith("Column"):
                                                    table= c[1][6:]
                                                    col= c[2][7:]
                                                    memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                    flag= True
                                                if not c[2].startswith("Column"):
                                                    for d in goldPOS:
                                                        if (d[2].startswith("Column")) & (d[1]== c[1]):
                                                            table= d[1][6:]
                                                            col= d[2][7:]
                                                            memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                            flag=True
                                                            break
                                        if len(memory["groupByClause"])<1:
                                            table = memory["Tables"][0]
                                            col = AllTableColumns[table][0]
                                            memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                            flag =True
                                    break
                        if c[1] =="MAX":
                            for z in goldPOS:
                                if (z[0]>c[0]) & (z[2] =="NUM"):
                                    memory["having"] = [f"HAVING COUNT(*) >= {w2n.word_to_num(z[1])}"]
                                    if len(memory["Tables"])>0:
                                        for c in goldPOS:
                                            if (c[0]>z[0])&(c[1] in to_ignore):
                                                break
                                            if (c[0]>z[0]) & (c[1].startswith("Table")):
                                                if c[2].startswith("Column"):
                                                    table= c[1][6:]
                                                    col= c[2][7:]
                                                    memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                    flag= True
                                                if not c[2].startswith("Column"):
                                                    for d in goldPOS:
                                                        if (d[2].startswith("Column")) & (d[1]== c[1]):
                                                            table= d[1][6:]
                                                            col= d[2][7:]
                                                            memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                            flag=True
                                                            break
                                        if len(memory["groupByClause"])<1:
                                                for d in goldPOS:
                                                    if (d[2].startswith("Column")):
                                                        table= d[1][6:]
                                                        col= d[2][7:]
                                                        memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                        flag=True
                                                        break
                                        if len(memory["groupByClause"])<1:
                                            table = memory["Tables"][0]
                                            col = AllTableColumns[table][0]
                                            memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                            flag =True
                                            break  
                                    break
                        if c[1]== ">":
                            for z in goldPOS:
                                if (z[0]>= c[0]) &(z[1].startswith("Table")):
                                    break
                                if (z[0]>=c[0])&(z[2]== "NUM"):
                                    memory["having"]= [f"HAVING COUNT(*)> {w2n.word_to_num(z[1])}"]
                                    if len(memory["Tables"])>0:
                                        for c in goldPOS:
                                            if (c[0]>z[0])&(c[1] in to_ignore):
                                                break
                                            if (c[0]>z[0]) & (c[1].startswith("Table")):
                                                if c[2].startswith("Column"):
                                                    table= c[1][6:]
                                                    col= c[2][7:]
                                                    memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                    flag= True
                                                if not c[2].startswith("Column"):
                                                    for d in goldPOS:
                                                        if (d[2].startswith("Column")) & (d[1]== c[1]):
                                                            table= d[1][6:]
                                                            col= d[2][7:]
                                                            memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                            flag=True
                                                            break
                                        if len(memory["groupByClause"])<1:
                                                for d in goldPOS:
                                                    if (d[2].startswith("Column")):
                                                        table= d[1][6:]
                                                        col= d[2][7:]
                                                        memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                        flag=True
                                                        break
                                        if len(memory["groupByClause"])<1:
                                            table = memory["Tables"][0]
                                            col = AllTableColumns[table][0]
                                            memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                            flag =True
                                            break
                                    break
                            if flag == False:
                                for z in goldPOS:
                                    if (z[0]<=c[0]) & (z[2]=="NUM"):
                                        if goldPOS[c[0]-2][1]=="or":
                                                memory["having"]= [f"HAVING COUNT(*)>= {w2n.word_to_num(z[1])}"]
                                        if goldPOS[c[0]-2][1]!="or":
                                            memory["having"]= [f"HAVING COUNT(*)> {w2n.word_to_num(z[1])}"]
                                        if len(memory["Tables"])>0:
                                            table = memory["Tables"][0]
                                            col = AllTableColumns[table][0]
                                            memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                        break
                        if c[1]== "<":
                            for z in goldPOS:
                                if (z[0]>= c[0]) &(z[1].startswith("Table")):
                                    break
                                if (z[0]>=c[0])&(z[2]== "NUM"):
                                    memory["having"]= [f"HAVING COUNT(*)< {w2n.word_to_num(z[1])}"]
                                    if len(memory["Tables"])>0:
                                        for c in goldPOS:
                                            if (c[0]>z[0])&(c[1] in to_ignore):
                                                break
                                            if (c[0]>z[0]) & (c[1].startswith("Table")):
                                                if c[2].startswith("Column"):
                                                    table= c[1][6:]
                                                    col= c[2][7:]
                                                    memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                    flag= True
                                                if not c[2].startswith("Column"):
                                                    for d in goldPOS:
                                                        if (d[2].startswith("Column")) & (d[1]== c[1]):
                                                            table= d[1][6:]
                                                            col= d[2][7:]
                                                            memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                            flag=True
                                                            break
                                        if len(memory["groupByClause"])<1:
                                                for d in goldPOS:
                                                    if (d[2].startswith("Column")):
                                                        table= d[1][6:]
                                                        col= d[2][7:]
                                                        memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                                        flag=True
                                                        break
                                        if len(memory["groupByClause"])<1:
                                            table = memory["Tables"][0]
                                            col = AllTableColumns[table][0]
                                            memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                            flag =True
                                            break
                                    break
                            if flag == False:
                                for z in goldPOS:
                                    if (z[0]<=c[0]) & (z[2]=="NUM"):
                                        if goldPOS[c[0]-2][1]=="or":
                                                memory["having"]= [f"HAVING COUNT(*)<= {w2n.word_to_num(z[1])}"]
                                        if goldPOS[c[0]-2][1]!= "or":
                                            memory["having"]= [f"HAVING COUNT(*)< {w2n.word_to_num(z[1])}"]
                                        if len(memory["Tables"])>0:
                                            table = memory["Tables"][0]
                                            col = AllTableColumns[table][0]
                                            memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                        break
                    
                        generalFlag= True   
                if generalFlag== False:
                    for c in goldPOS:
                        if (c[0]>x[0]) & (c[2].startswith("Column")):
                            break
                        if (c[0]>x[0]) & (c[2]=='NUM'):
                            memory["having"]= [f"HAVING COUNT (*) = {w2n.word_to_num(c[1])}"]
                            if (len(memory["Tables"])>0) & (len(memory["SelectClause"])==0):
                                table = memory["Tables"][0]
                                col = AllTableColumns[table][0]
                                memory["groupByClause"] = f"GROUP BY {table}.{str.lower(col)}"
                                break
                            if (len(memory["SelectClause"])!=0):
                                for z in goldPOS:
                                    if (z[0]<=memory["phraseBroke"]) & (z[2].startswith("Column")):
                                        table= z[1][6:]
                                        column = z[2][7:]
                                        memory["groupByClause"] = f"GROUP BY {table}.{str.lower(column)}"
                                        break
                            break
                            
        if memory["having"]== []:
            for x in goldPOS:
                if (x[0]>memory["phraseBroke"]) & (x[1] in ["MIN", "MAX"]):
                    if (goldPOS[x[0]-2][1] == "at") & (not goldPOS[x[0]-3][2].startswith("Column")):
                        for c in goldPOS:
                            if (c[0]>x[0]) & (c[2] in "NUM"):
                                if x[1] == "MIN":
                                    memory["having"]= [f"HAVING COUNT(*)>= {w2n.word_to_num(c[1])}"]
                                    break
                                if x[1]== "MAX":
                                    memory["having"]= [f"HAVING COUNT(*)<= {w2n.word_to_num(c[1])}"]
                                    break
                        for c in goldPOS:
                            if (c[0]<x[0]) & (c[2].startswith("Column")):
                                tabel = c[1][6:]
                                column = c[2][7:]
                                memory["groupByClause"]= f"GROUP BY {tabel}.{str.lower(column)}"
                                break



                
    return



def subselectClauseWithoutOF(goldPOS, memory):
    if memory["ofCount"]==0:
        if (len(memory["having"])==0) & (len(memory["WhereClause"])<2) & (len(memory["orderBy"])==0):
            haveFlag = False
            flag= False
            internalColSubFlag= False
            internalFuncSubFlag= False
            internalsubFunc=None
            column=None
            for x in goldPOS:
                if (x[0]>=memory["phraseBroke"]) & (x[1] in having) & (haveFlag==False):
                    for c in goldPOS:
                        if (c[0]>memory["phraseBroke"]) & (c[1] in notIN):
                            flag=True
                        if (c[0]>=x[0]) & (c[2].startswith("Column")):
                            internalCol= c[2][7:]
                            internalTable= c[1][6:]
                            for z in goldPOS:
                                if (z[0]>c[0]) & (z[1] in ["<", ">"]):
                                    internalsubFunc= z[1]
                                if (z[0]>c[0]) & (z[1] in ["MIN", "MAX", "SUM", "AVG"]):
                                    subFunc= z[1]
                                    internalFuncSubFlag= True
                                if (z[0]>c[0]) & (z[2].startswith("Column")):
                                    column = z[2][7:]
                                    table= z[1][6:]
                                    internalColSubFlag=True
                            break
                        if (c[0]>=x[0]) & (c[1] in notIN):
                            flag=True
                        if (c[0]>=x[0]) & ((c[2].startswith("Reference")) | (c[2].startswith("Main"))):
                            for value in memory["SelectClause"]:
                                if " AS " in value:
                                    memory["SelectClause"].pop(memory["SelectClause"].index(value))
                                    memory["SelectClause"].append(value.split(" AS ")[0])
                            if flag==False:
                                idCol= AllTableColumns[memory["Tables"][0]][0]
                                memory["WhereClause"].append(f" {idCol} IN (SELECT {idCol} ")
                                memory["WhereClause"].append(f"FROM {c[1][6:]})")
                                haveFlag = True
                                break
                            if flag==True:
                                idCol= AllTableColumns[memory["Tables"][0]][0]
                                memory["WhereClause"].append(f" {idCol} NOT IN (SELECT {idCol} ")
                                memory["WhereClause"].append(f"FROM {c[1][6:]})")
                                haveFlag = True
                                break
            if (internalColSubFlag==True) & (internalFuncSubFlag==True):
                for value in memory["SelectClause"]:
                    if " AS " in value:
                        memory["SelectClause"].pop(memory["SelectClause"].index(value))
                        memory["SelectClause"].append(value.split(" AS ")[0])
                if (internalsubFunc!=None )& (column !=None):
                    memory["WhereClause"].append(f" {internalCol} {internalsubFunc} (SELECT {subFunc}({column}) ")
                    memory["WhereClause"].append(f"FROM {table})")
                    haveFlag = True
            wherecol= None
            subCol= None
            for x in goldPOS:
                if (x[0]>memory["phraseBroke"]) & (x[2] == "SQL function"):
                    for c in goldPOS:
                        if (c[0] > x[0]) & (c[2] == "SQL function")& (c[1] in ["MIN", "MAX", "SUM", "AVG"]):
                            if goldPOS[x[0]-2][2].startswith("Column"):
                                table= goldPOS[x[0]-2][1][6:]
                                wherecol= f"{goldPOS[x[0]-2][1][6:]}.{goldPOS[x[0]-2][2][7:]}"
                            if goldPOS[c[0]-2][2].startswith("Column"):
                                subCol= f"{goldPOS[c[0]-2][1][6:]}.{goldPOS[c[0]-2][2][7:]}"
                            if not goldPOS[c[0]-2][2].startswith("Column"):
                                subCol= f"{goldPOS[x[0]-2][1][6:]}.{goldPOS[x[0]-2][2][7:]}"
                            
                            if (wherecol!= None) & (subCol!= None):
                                memory["WhereClause"].append(f" {wherecol} {x[1]} ")
                                memory["WhereClause"].append(f"(SELECT {c[1]}({subCol}) FROM {table})")
                                haveFlag = True
                    break


                        
    return



def StructureQueryWithoutOF(goldPOS,memory):
    if memory["ofCount"]==0:

        ######## PARA UMA TABELA
        if len(set(memory["Tables"]))==1:
            if len(memory["SelectClause"])>1:
                memory["SelectClause"]= [memory["SelectClause"][0]] + ["," + item for item in memory["SelectClause"][1:]]
                memory["SelectClause"].insert(0, "SELECT")
                memory["SelectClause"].insert(len(memory["SelectClause"]), f'FROM {memory["Tables"][0]}')
            if len(memory["SelectClause"])== 1:
                memory["SelectClause"].insert(0, "SELECT")
                memory["SelectClause"].insert(len(memory["SelectClause"]), f'FROM {memory["Tables"][0]}')
            if len(memory["WhereClause"])>3:
                idx = None
                for x in memory["WhereClause"]:
                    if ('.' in x):
                        col= x
                        idx= memory["WhereClause"].index(x)
                        if memory["WhereClause"][idx+1].startswith("(SELECT"):
                            idx= None
                            break
                        break
                if idx!= None:
                    if memory["or"]==True:
                        memory["WhereClause"].insert(idx+2, "OR")
                    if memory["or"]==False:
                        for x in memory["WhereClause"][idx+1:]:
                            if (col in x):
                                memory["WhereClause"].insert(idx+2, "OR")
                            if (col not in x) & ("." in x) & (memory["or"]==False):
                                memory["WhereClause"].insert(memory["WhereClause"].index(x), "AND")
                where= memory["WhereClause"]
            if len(memory["WhereClause"])<=3:
                where = memory["WhereClause"]
            if len(memory["WhereClause"])==2:
                where = memory["WhereClause"]
            select= " ".join(memory["SelectClause"])
            try:
                if len(where)>=3:
                    if len(where)==3:
                        where= " ".join(memory["WhereClause"])
                        final= select +" "+ where
                    if len(where)>=3:
                        where= " ".join(memory["WhereClause"])
                        final = select+ " " +where
            except:pass
            try:
                if len(where)<3:
                    final= select
            except:pass
            try:
                if len(memory["groupByClause"])>0:
                    final = final +" "+ memory["groupByClause"]
            except:pass
            try:
                if len(memory["orderBy"])>0:
                    if type(memory["orderBy"])==  list:
                        final = final+" "+ memory["orderBy"][0]
                    else:
                        final = final+ " "+ memory["orderBy"]
            except:pass
            if len(memory["having"])>0:
                final = final + " " + memory["having"][0]
        ##### PARA mais que uma TABELA
        if len(set(memory["Tables"]))>1:
            flag = False
            flag1= False
            for i in range(len(memory["SelectClause"])):
                if len(memory["WhereClause"])>0:
                    for x in memory["WhereClause"]:
                        if "(SELECT " in x:
                            flag1= True
                            break
                if flag1==False:
                    for key in memory["aliases"]:
                        if memory["SelectClause"][i].startswith("FROM"):
                            flag= True
                            break
                        if (key in memory["SelectClause"][i]):
                            if ("MAX" in memory["SelectClause"][i]) | ("MIN" in memory["SelectClause"][i])| ("SUM" in memory["SelectClause"][i]) | ("AVG" in memory["SelectClause"][i]) | ("DISTINCT" in memory["SelectClause"][i]):
                                memory["SelectClause"][i] =  memory["SelectClause"][i].replace(key, str(f'T{memory["aliases"][key]}')) 
                            else:
                                splited= memory["SelectClause"][i].split(".")
                                if len(key)== len(splited[0]):
                                    splited[0]= splited[0].replace(key, str(f'T{memory["aliases"][key]}')) 
                                    splited = ".".join(splited)
                                    memory["SelectClause"][i] = splited   
                    if flag== True:
                        break
                
            result = []
            for index, item in enumerate(memory["SelectClause"]):
                if item.startswith('FROM'):
                    result.append(item)
                    break
                if index > 0 and not memory["SelectClause"][index-1].startswith('FROM'):
                    result.append(', ')
                result.append(item)
            memory["SelectClause"]= result            

                
            memory["SelectClause"].insert(0, "SELECT")  
            select= " ".join(memory["SelectClause"])
            if len(memory["WhereClause"])>3:
                idx = None
                for x in memory["WhereClause"]:
                    if ('.' in x):
                        col= x
                        idx= memory["WhereClause"].index(x)
                        break
                if memory["or"]==True:
                    memory["WhereClause"].insert(idx+2, "OR")
                if memory["or"]==False:
                    for x in memory["WhereClause"][idx+1:]:
                        if col in x:
                            memory["WhereClause"].insert(idx+2, "OR")
                        if (col not in x) & ("." in x):
                            memory["WhereClause"].insert(memory["WhereClause"].index(x), "AND")
                where= memory["WhereClause"]
            if len(memory["WhereClause"])>=2:
                for i in range(len(memory["WhereClause"])):
                    if "(SELECT " in memory["WhereClause"][i]:
                        break
                    for key in memory["aliases"]:
                        if key in memory["WhereClause"][i]:
                            splited= memory["WhereClause"][i].split(".")
                            if ("MAX" in memory["WhereClause"][i]) | ("MIN" in memory["WhereClause"][i])| ("SUM" in memory["WhereClause"][i]) | ("AVG" in memory["WhereClause"][i]) | ("DISTINCT" in memory["WhereClause"][i]):
                                memory["WhereClause"][i] = memory["WhereClause"][i].replace(key, str(f'T{memory["aliases"][key]}'))
                            else:
                                if len(key)== len(splited[0]):
                                    splited[0]= splited[0].replace(key, str(f'T{memory["aliases"][key]}')) 
                                    splited = ".".join(splited)
                                    memory["WhereClause"][i] = splited 
                where= " ".join(memory["WhereClause"])  
                final = select+ " "+ where
            if len(memory["WhereClause"])<2:
                final = select
            try: 
                if len(memory["groupByClause"])>0:
                    for i in (memory["groupByClause"]):
                        for key in memory["aliases"]:
                            if key in memory["groupByClause"]:
                                splitted= memory["groupByClause"].split(".")
                                group1= splitted[0].replace("GROUP BY", "")
                                group2 = group1.replace(" ", "")
                                group2 = group2.replace(key, str(f'T{memory["aliases"][key]}'))
                                grouped = group2+"."+ splitted[-1]
                                memory["groupByClause"] = f"GROUP BY {grouped}"
                final = final +" "+ memory["groupByClause"]
            except:pass
            if len(memory["orderBy"])>0:
                for i in (memory["orderBy"]):
                    for key in memory["aliases"]:
                        if key in memory["orderBy"]:
                            memory["orderBy"] = memory["orderBy"].replace(key, str(f'T{memory["aliases"][key]}'))
                final = final +" "+ memory["orderBy"]
            
            if len(memory["having"])>0:
                final = final + " " + memory["having"][0]
    return final


datasetTotest= pd.read_csv("/path/toTheData/ofQuestions/andDatabasesNames")



i=0
connecterror= 0
test_results = pd.DataFrame(columns=["database", "question", "verified_output", "correct"])
start = time.time()
for index, row in datasetTotest[:].iterrows():
    try:
        goldPOS = question(f"{row['question']}")
        ColumnEmbeddings=[]
        tableColumnEmbeddings=[] 
        names={}
        AllColumns= []
        AllTableColumns= {}
        tables=[]
        memory={"goldPOScopy":goldPOS.copy(), "idxfirst" : None, "interIDX" : None, "idxSim1": None, "idxSim2": None, "exceptionSimilarity":None, "idxSim1Table": None, "idxSim2Table": None}
        tableColumns={}
        sqlDatabase(f"/Users/yuriyperezhohin2/Desktop/spiderdatabases/database/{row['database']}/{row['database']}.sqlite")
        sqlEmbeddings(names)
        phraseCouter(goldPOS, memory)
        truncateSpecialChars(goldPOS)
        propnTruncation(goldPOS)
        memory["tripleCompound"]=None
        tripleCompoundDepencies(text1, goldPOS)
        compoundDependencies(text1,goldPOS)
        amodDependencies(text1,goldPOS)
        conjuctionDependencies(text1, goldPOS)
        mixedDependencies(text1, goldPOS)
        CountTruncate(goldPOS)
        ofCountPattern(goldPOS,memory)
        print(text1)
    except:sql= "Gave error"
    try:
        if memory["ofCount"]>0:
            memory["idxAdpositionInclusiveLast"]= None
            memory["idxAdpositionInclusive"]= None
            OfexceptionFirst(memory, goldPOS,tables)
            OfFirstexceptionSimilarity(memory, goldPOS)
            bothExceptionOf(memory,goldPOS,tableColumns)
            secondExceptionOf(memory,goldPOS,tableColumns)
            firstExceptionOf(memory, goldPOS)
            neitherExceptionOf(memory, goldPOS)
            memory["interOfidx"]=None
            neitherSimilarityException(memory,goldPOS)
            memory["doubleOfExceptionBoth"]= None
            doubleOfExceptionBoth(goldPOS, memory)
            memory["doubleOfExceptionSecond"]=None
            doubleOfExceptionSecond(goldPOS, memory)
            memory["doubleOfExceptionFirst"]= None
            doubleOfExceptionFirst(goldPOS, memory)
            doubleOfExceptionNeither(goldPOS, memory)
            vocabularyInsert(goldPOS, memory)
            notIdentifiedOF(goldPOS, memory)
            specialCaseWords(goldPOS, memory)
            TruncateAggColumns(memory=memory, goldPOS=goldPOS)
            continuedPatternsAfterOf(goldPOS, memory)
            continuedPatternsBeforeOf(goldPOS,memory)
            continuedPatternsAfterDoubleOf(goldPOS, memory)
            continuedPatternsBeforeDoubleOf(goldPOS, memory)
            yearOF(goldPOS, memory)
            swapNumberTable(goldPOS, memory)
            idRecognizer(goldPOS, memory)
            SelectClauseOF(goldPOS, memory)
            andOrWhere(goldPOS,memory)
            WhereClauseOF(goldPOS,memory)
            OrderByOF(goldPOS,memory)
            aggregationsOF(goldPOS,memory)
            specialAggregations(goldPOS,memory)
            propnWhere(goldPOS, memory)
            joinTables(goldPOS, memory)
            CountClause(goldPOS, memory)
            GroupByClause(goldPOS, memory)
            havingClause(goldPOS, memory)
            subselectClause(goldPOS, memory)
            sql=StructureQueryOF(goldPOS, memory)
        if memory["ofCount"]==0:
            phraseBroker(goldPOS, memory)
            afterBroke(goldPOS, memory)
            untilBroke(goldPOS, memory)
            vocabularyInsert(goldPOS, memory)
            notIdentifiedWithutOf(goldPOS, memory)
            specialCaseWordsWithoutOf(goldPOS, memory)
            yearWithoutof(goldPOS, memory)
            TruncateAggColumnsWithoutOF(memory=memory, goldPOS=goldPOS)
            idRecognizer(goldPOS, memory)
            SelectClauseWithoutOf(goldPOS, memory)
            andOrWhereWithoutOF(goldPOS,memory)
            WhereClauseWithoutOF(goldPOS,memory)
            OrderByWithoutOf(goldPOS,memory)
            aggregationsWithoutOF(goldPOS,memory)
            specialAggregationsWithoutOF(goldPOS,memory)
            propnWhereWithoutOF(goldPOS, memory)
            joinTablesWithoutOF(goldPOS, memory)
            CountClauseWithoutOF(goldPOS, memory)
            GroupByClauseWithoutOF(goldPOS, memory)
            havingClauseWithoutOF(goldPOS, memory)
            subselectClauseWithoutOF(goldPOS, memory)
            sql= StructureQueryWithoutOF(goldPOS, memory)
    except:sql = "Gave error"
    if len(sql)>0:
        if sql != "Gave error":
            try:
                connection= sqlite3.connect(f'/Users/yuriyperezhohin2/Desktop/spiderdatabases/database/{row["database"]}/{row["database"]}.sqlite')
                cursor= connection.cursor()
            except:
                connecterror+=1
            try:
                df = pd.read_sql(f"{sql}", con=connection)
            except:pass
            try:
                new=[]
                for value in df.columns: 
                    if ("MAX" in value) or ("DISTINCT" in value) or ("MIN" in value) or ("SUM"in value) or ("COUNT"in value) or ("AVG" in value):
                        if ("COUNT" in value)  & ("DISTINCT" in value):
                            value = value.replace(" ", "")
                            z=value.split("(")
                            newval = []
                            scndpart= []
                            for v in z :
                                val = v.split('.')[-1]
                                if "DISTINCT" in v:
                                    newval.append(f"({val}")
                                elif ")" in v:
                                    scndpart.append(f"{val[:-1]}")
                                elif "COUNT" not in v:
                                    newval.append(f" {val}")
                                else:
                                    newval.append(f"{val}")
                            newval= "".join(newval)
                            scndpart.insert(0, newval)
                            scndpart="".join(scndpart)
                            new.append(scndpart)
                        else:
                            z=value.split("(")
                            newval = []
                            for v in z :
                                val = v.split('.')[-1]
                                newval.append(f"({val}")
                            newval[0]= newval[0][1:]
                            newval= "".join(newval)
                            new.append(newval)
                            
                    else:
                        new.append(value)
                df.columns= new
                df.columns = [x.lower() for x in df.columns]
                df = df[sorted(df.columns)]
                dforiginal= pd.read_sql(f"{row['sql']}", con=connection)
                new=[]
                for value in dforiginal.columns: 
                    if ("max" in str.lower(value)) or ("distinct" in str.lower(value)) or ("min" in str.lower(value)) or ("sum"in str.lower(value)) or ("count"in str.lower(value)) or ("avg" in str.lower(value)):
                        if ("count" in str.lower(value))  & ("distinct" in str.lower(value)):
                            value = value.replace(" ", "")
                            z=value.split("(")
                            newval = []
                            scndpart= []
                            for v in z :
                                val = v.split('.')[-1]
                                if "distinct" in str.lower(v):
                                    newval.append(f"({val}")
                                elif ")" in v:
                                    scndpart.append(f"{val[:-1]}")
                                elif "count" not in str.lower(v):
                                    newval.append(f" {val}")
                                else:
                                    newval.append(f"{val}")
                            newval= "".join(newval)
                            scndpart.insert(0, newval)
                            scndpart="".join(scndpart)
                            new.append(scndpart)
                        else:
                            z=value.split("(")
                            newval = []
                            for v in z :
                                val = v.split('.')[-1]
                                newval.append(f"({val}")
                            newval[0]= newval[0][1:]
                            newval= "".join(newval)
                            new.append(newval)
                            
                    else:
                        new.append(value)
                dforiginal.columns= new
                dforiginal.columns = [x.lower() for x in dforiginal.columns]
                dforiginal= dforiginal[sorted(dforiginal.columns)]
                del new
                if df.equals(dforiginal)==True:
                    test_results.loc[len(test_results.index)] = [row["database"], row["question"], sql, 1] 
                    i+=1
                    print(i)
                if df.equals(dforiginal)==False:
                    test_results.loc[len(test_results.index)] = [row["database"], row["question"], sql, 0] 
                del df
                del dforiginal
            except:
                pass

end = time.time()
print(end - start)

print(i)
test_results.to_csv("devALLSecondModelDotProduct.csv")




