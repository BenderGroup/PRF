SELECT
`compound_structures`.`standard_inchi_key`,
`compound_structures`.`canonical_smiles`

FROM
`compound_structures`

WHERE
`compound_structures`.`standard_inchi_key` IS NOT NULL AND
`compound_structures`.`canonical_smiles` IS NOT NULL