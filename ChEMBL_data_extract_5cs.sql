SELECT`component_sequences`.`accession` AS 'Uniprot_Accession',`compound_structures`.`standard_inchi_key`,`activities`.`pchembl_value`,`activities`.`potential_duplicate`, `activities`.`standard_type`,`assays`.`doc_id`,`source`.`src_id`,`source`.`src_description`,`source`.`src_short_name`FROM`compound_properties`,`compound_structures`,`compound_records`,`molecule_dictionary`,`activities`,`assays`,`target_dictionary`,`target_components`,`component_sequences`,`source`WHERE`molecule_dictionary`.`molregno` = `compound_records`.`molregno` AND`compound_properties`.`molregno` = `molecule_dictionary`.`molregno` AND`compound_structures`.`molregno` = `molecule_dictionary`.`molregno` AND`molecule_dictionary`.`molregno` = `activities`.`molregno` AND`activities`.`assay_ID` = `assays`.`assay_ID` AND`assays`.`tid` = `target_dictionary`.`tid` AND`target_dictionary`.`tid` = `target_components`.`tid` AND`target_components`.`component_id` = `component_sequences`.`component_id` AND`activities`.`src_id` = `source`.`src_id`AND`activities`.`standard_relation` IN ('=', '==', '===', '<', '<=', '<<', '<<<', '~') AND`target_dictionary`.`organism` IN ('Homo sapiens') AND`activities`.`standard_type` IN ('IC50', 'Ki', 'Kd', 'EC50', 'AC50') AND`activities`.`standard_value` <=100 AND`assays`.`confidence_score` >= 5 AND`assays`.`assay_type` IN ('B') AND`activities`.`pchembl_value` IS NOT NULLorder by `component_sequences`.`accession` ASC, `molecule_dictionary`.`chembl_id` ASC,`activities`.`standard_value` DESC, `assays`.`confidence_score` ASC;