"""
=== Script to perform evaluation of Probabilistic Random Forests ===

This script requires the ChEMLBL v27 and PubChem datasets as described in the paper.
To obtain the ChEMBL dataset the sql command is first performed to generate the file:

mysql -u <user> -p <password> chembl_27 < ChEMBL_data_extract_5cs.sql > data_5cs_smiles.txt

This requires chembl version 27 installed and will output the active dataset to the file data_5cs_smiles
Also run the following to generate inchi > smile mappings:

mysql -u <user> -p <password> chembl_27 < InchiKey_to_SMILES.sql > InchiKey_to_SMILES.txt


Scaffold-based splitting can be enabled but this is not recommended due to complexity 
in obtaining viable splits (splits already require intermediate y-probabilities 
around the near to the decision threshold of the bioactivity threshold)

Input args:
	N_cores [number of cores to use]
	thresh [the bioactivity threshold (pChEMBL)]
"""

import sys
import scipy
import numpy as np
from multiprocessing import Pool
import multiprocessing
multiprocessing.freeze_support()
import numpy as np
import random
import math
import os
import glob
import pandas as pd
import zipfile
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from standardiser import standardise
from rdkit import RDLogger
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
import sklearn
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit, cross_val_predict, GridSearchCV
from sklearn.calibration import calibration_curve
from sklearn.utils import class_weight
from sklearn.metrics import mean_squared_error, r2_score, auc
from sklearn.svm import SVR, SVC, LinearSVR, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from PRF import prf
import warnings
from statsmodels import regression
import scipy
from scipy.stats import entropy, median_absolute_deviation, chisquare
from math import sqrt
import statsmodels.api as sm
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


####
#you must set the path to the pidgin inactives from PIDGIN here
path_to_pidgin_inactives=''
N_cores = int(sys.argv[1])
thresh=int(sys.argv[2])
###

def calcFingerprints(smiles,preproc=True):
	global global_mol_dict
	try:
		if preproc: precalculated = global_mol_dict[smiles]
		else: precalculated = global_mol_dict_no_preproc[smiles]
		if precalculated is not None: return precalculated
		else: raise PreprocessViolation(' Molecule preprocessing violation')
	except KeyError:
		m1 = Chem.MolFromSmiles(smiles)
		if preproc: m1 = preprocessMolecule(m1)
		if not m1: 
			global_mol_dict[smiles] = None
			raise PreprocessViolation(' Molecule preprocessing violation') 
		scaf = Chem.MolToSmiles(MakeScaffoldGeneric(GetScaffoldForMol(m1)))
		fp = AllChem.GetMorganFingerprintAsBitVect(m1,2, nBits=2048)
		bitstring = list(map(int,list(fp.ToBitString())))
	if preproc: global_mol_dict[smiles] = [bitstring, scaf]
	else: global_mol_dict_no_preproc[smiles] = [bitstring, scaf]
	return bitstring, scaf
	
def preprocessMolecule(inp):
	if not inp: return False
	def checkC(mm):
		mwt = Descriptors.MolWt(mm)
		for atom in mm.GetAtoms():
			if atom.GetAtomicNum() == 6 and 100 <= mwt <= 1000: return True
		return False
	def checkHm(mm):
		for atom in mm.GetAtoms():
			if atom.GetAtomicNum() in [2,10,13,18]: return False
			if 21 <= atom.GetAtomicNum() <= 32: return False
			if 36 <= atom.GetAtomicNum() <= 52: return False
			if atom.GetAtomicNum() >= 54: return False
		return True
	try: std_mol = standardise.run(inp)
	except standardise.StandardiseException: return None
	if not std_mol or checkHm(std_mol) == False or checkC(std_mol) == False: return None
	else: return std_mol
	
class PreprocessViolation(Exception):
	'raise due to preprocess violation'

def calcFingerprints_array(inp, file=False, preproc=True, act_chk=0):
	outp = []
	outscaf = []
	otm = []
	pactivity = []
	for i in inp:
		if i[0] == "": continue
		try:
			if file:
				oo = calcFingerprints(i[0],preproc)
				try:
					if float(i[1]) < act_chk: continue
					pactivity.append(float(i[1]))
				except TypeError: continue
			else: oo = calcFingerprints(i,preproc)
			outscaf.append(oo[1])
			outp.append(oo[0])
		except KeyboardInterrupt: quit()
		except PreprocessViolation: pass
		except: pass
	if file: return outp, outscaf, pactivity
	else: return outp, outscaf

def getfp(needed):
	random.seed(2)
	smiles = random.sample(pooled_smi,needed)
	fps = calcFingerprints_array(smiles,preproc=False)
	return fps

def processfile(infile,file=False,act_chk=0,preproc=True):
	return calcFingerprints_array(infile,file=file,act_chk=act_chk,preproc=preproc)

def convertPvalue(pactivity,activity_threshold,standard_dev):
    ret=[]
    for pvalue in pactivity:
        try:
            pvalue=float(pvalue)
            ret.append(scipy.stats.norm.cdf(pvalue, activity_threshold, standard_dev).round(3))
        except ValueError: ret.append('')
    return ret

def processtarget(inp):
	global thresh
	activity_threshold = thresh
	sdict = {idx:i for idx, i in enumerate([round(float(i),2) for i in np.arange(0,.9,0.1)])}
	uniprot,infile = inp
	try: matrix,active_scaf,pactivity = processfile(infile.groupby('smiles').mean().reset_index()[['smiles','pchembl_value']].values,file=True)
	except TypeError: return
	if len(matrix) < 100: return
	vector = [1 if x >= activity_threshold else 0 for x in pactivity]
	sfvector = []
	#set up cdf for bioactivity scale
	for standard_deviation_threshold in sorted(sdict.values()):
		if standard_deviation_threshold == 0.0:
			sfvector.append(vector)
		else:
			reweighted = convertPvalue(pactivity,activity_threshold,standard_deviation_threshold)
			sfvector.append(reweighted)
	#process the inactive set
	if sum(vector) < 100: return
	print(uniprot)
	nact = sum(vector)
	ninact = len(vector)-sum(vector)
	conf_smiles = []
	egids = uniprot_egid.get(uniprot)
	if egids != None:
		for egid in egids:
			try:
				with zipfile.ZipFile(path_to_pidgin_inactives + egid + '.smi.zip') as z:
					conf_smiles += [i.split(' ')[0] for i in z.open(egid + '.smi').read().decode('UTF-8').splitlines()]		
			except: pass
	req = nact * 2
	if req < 1000: req = 1000
	if req > 2000: req = 2000
	req -= ninact
	if req < 0: req = 0
	conf_inactives, inactive_scaf = [], []
	#sample inactives if necessary
	if len(conf_smiles) > 0:
		random.seed(2)
		random.shuffle(conf_smiles)
		try:
			random.seed(2)
			conf_inactives,inactive_scaf = calcFingerprints_array(random.sample(conf_smiles,req))
		except ValueError: conf_inactives,inactive_scaf = calcFingerprints_array(conf_smiles)
	conf_smiles = []
	vector2 = []
	for i in conf_inactives:
		if req > 0:
			matrix.append(i)
			vector2.append(0)
			req-=1
	conf_inactives = None
	ninact += len(vector2)
	nse = 0
	if req > 0:
		vector2 += [0] * req
		random_bg, random_scaf = getfp(req)
		nse = len(random_bg)
		matrix += random_bg
		inactive_scaf += random_scaf
	del random_bg, random_scaf
	all_scafs = active_scaf+inactive_scaf
	del active_scaf, inactive_scaf
	scaf_dict = {s[0]:s[1] for s in zip(set(all_scafs),range(0,len(set(all_scafs)),1))}
	all_scafs = [scaf_dict[sca] for sca in all_scafs]
	nscaf = len(scaf_dict.keys())
	vector += vector2
	pactivity = np.array(pactivity + [0] * len(vector2), dtype=np.float32)
	sfvector = [s+vector2 for s in sfvector]
	vector2 = None
	matrix = np.array(matrix, dtype=np.uint8)
	vector = np.array(vector, dtype=np.uint8)
	sfvector = [np.array(s) for s in sfvector]
	skf = StratifiedShuffleSplit(n_splits=3, random_state=2, test_size=0.75, train_size=0.25)
	lso = GroupShuffleSplit(n_splits=3, random_state=2, test_size=0.75, train_size=0.25)
	base_predicted1, base_predicted2, base_predicted3 = [], [], []
	y_lab, y_lab_raw, y_binary = [], [], []
	per_fold=[]
	try:
		#remove '[:1]' to enable scaffold splitting
		for split_method, split_name in [(skf,0),(lso,1)][:1]:
			#for each splitting method, perform the evaluation
			for train, test in split_method.split(matrix,vector,groups=all_scafs):
				x, y, X_test,Y_binary, Y_raw = matrix[train], vector[train], matrix[test], vector[test], pactivity[test]
				class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
				sw = np.array([class_weights[1] if i == 1 else class_weights[0] for i in y])				
				rfc = RandomForestClassifier(n_jobs = 1, n_estimators=200, class_weight='balanced', random_state=2)
				###### ###### ###### ###### ###### ###### ###### ###### ###### 
				brfc=sklearn.base.clone(rfc)
				brfc.fit(x,y,sample_weight=sw)
				#for each emulated experimental error, generate predictions
				for sidx,ystrain in enumerate(sfvector):
					sw2 = ystrain[train]
					py=np.zeros([len(sw2),2])
					py[:,1] = sw2
					py[:,0] = 1-py[:,1]
					prfc = prf(n_estimators=200, bootstrap=True, keep_proba=0.05)
					prfc.fit(X=x.astype(float), py=py.astype(float))
					rfr = RandomForestRegressor(n_jobs = 1, n_estimators=200, random_state=2)
					rfr.fit(x,sw2)
					p_prfc = [round(pr,3) for pr in list(np.array(prfc.predict_proba(X=X_test.astype(float)))[:,1])]
					p_brfc = [round(pr,3) for pr in list(brfc.predict_proba(X_test)[:,1])]
					p_rfr = [round(pr,3) for pr in list(np.array(rfr.predict(X_test)))]
					for sidx2, ystest in enumerate(sfvector):
						y_test=list(ystest[test])
						#add base rf method output
						base_predicted1 += p_brfc
						#add base prf method output (when stdev = 0)
						base_predicted2 += p_prfc
						#add prf method output
						base_predicted3 += p_rfr
						y_lab_raw += list(Y_raw)
						y_lab += list(y_test)
						y_binary += list(Y_binary)
						per_fold.append([len(y_test),[split_name,sdict[sidx],sdict[sidx2]]])
	except ValueError: return
	return [uniprot,nact,ninact,nse,nscaf], [y_binary,y_lab_raw,y_lab,base_predicted1,base_predicted2,base_predicted3], per_fold


#main
global_mol_dict = dict()
global_mol_dict_no_preproc = dict()

uniprot_egid=dict()
for i in open('uniprot_egid.dat'):
	i = i.strip().split('\t')
	try: uniprot_egid[i[0]].append(i[1])
	except KeyError: uniprot_egid[i[0]] = [i[1]]
	
pooled_smi = open('pool_atfilter.smi').read().splitlines()
df2 = pd.read_csv('InchiKey_to_SMILES.txt',sep='\t',header=None)
df2.columns = ['standard_inchi_key','smiles']
df = pd.read_csv('data_5cs_smiles.txt',sep='\t')
df_chembl = df.merge(df2,left_on='standard_inchi_key',right_on='standard_inchi_key',how='inner')
df_chembl = df_chembl[df_chembl['potential_duplicate']==0]
df_chembl = df_chembl[df_chembl['Uniprot_Accession'].map(df_chembl['Uniprot_Accession'].value_counts()) > 100]

mod_log = open('output/'+str(thresh)+'short_oneline.txt','w')
mod_log.write('\t'.join(map(str,['egid','nact','ninact','nse','nscaf','cv','sd_train','sd_test','ylabel','y_lab_raw','ylab','rfc','prfc','rfr'])) + '\n')
mod_log.close()
print('Created files')

pool = Pool(processes=N_cores)  # set up resources
jobs = pool.imap_unordered(processtarget, df_chembl.sort_values('Uniprot_Accession').groupby('Uniprot_Accession'))


for i, result in enumerate(jobs):
	upto = open('output/progress'+str(thresh)+'.csv','a')
	upto.write(str(len(df_chembl.Uniprot_Accession.unique())-i) + '\n')
	upto.close()
	if result == None: continue
	df=[[],[],[]]
	for fold in result[2]:
		for idx1,f in enumerate(fold[1]):
			df[idx1] += [f]*fold[0]
	df+=result[1]
	df = pd.DataFrame(df).transpose()
	df.columns =['cv','sd_train','sd_test','ylabel','y_lab_raw','ylab','rfc','prfc','rfr']
	df=df[df.ylab.between(0.001,0.999)]
	for filt,vals in df.groupby(['cv','sd_train','sd_test']):
		rr = result[0][:]
		mod_log = open('output/'+str(thresh)+'short_oneline.txt','a')
		for row in vals.values:
			mod_log.write('\t'.join(map(str,rr)) + '\t' + '\t'.join(map(str,list(row))) + '\n')
		mod_log.close()
