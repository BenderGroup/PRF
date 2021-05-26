# Probabilistic Random Forest Improves Bioactivity Predictions Close to the Classification Threshold by Taking into Account Experimental Uncertainty
Authors: Lewis Mervin, Maria-Anna Trapotsi

pRF_evaluation.py -> Script to perform evaluation of Probabilistic Random Forests
- This script requires the ChEMLBL v27 and PubChem datasets as described in the paper.
- To obtain the ChEMBL dataset the sql command is first performed to generate the file:


mysql -u <user> -p <password> chembl_27 < ChEMBL_data_extract_5cs.sql > data_5cs_smiles.txt

  
  (This requires chembl version 27 installed and will output the active dataset to the file data_5cs_smiles)
- Also run the following to generate inchi > smile mappings:
  
mysql -u <user> -p <password> chembl_27 < InchiKey_to_SMILES.sql > InchiKey_to_SMILES.txt

  References
----------
Mervin, L., Trapotsi, M. A., Afzal, A. M., Barrett, I., Bender, A., & Engkvist, O. (2021). Probabilistic Random Forest improves bioactivity predictions close to the classification threshold by taking into account experimental uncertainty.
https://chemrxiv.org/articles/preprint/Probabilistic_Random_Forest_Improves_Bioactivity_Predictions_Close_to_the_Classification_Threshold_by_Taking_into_Account_Experimental_Uncertainty/14544291
