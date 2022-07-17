MSH WSD Corpus
--------------

The MSH WSD data set contains 203 ambiguous words. Each instance containing the ambiguous word was assigned a CUI from the 2009AB version of the UMLS.

PACKAGE ORGANIZATION

  The available files are:

  * README - this file

  * benchmark_mesh.txt -  The first field is the ambiguous word and the
  subsequent fields are the candidate CUIS considered in this data set.
  The fields are separated by a tab ("\t"). The order of the CUIs denotes
  the name assigned to this CUI in the annotation set (M + CUI order), for
  instance:

   AA          C0002520             C0001972

   M1 = C0002520
   M2 = C0001972

  * <target word>.arff - There is one file per ambiguous word, which
  contains examples for the available senses. The files are in arff
  format(http://www.cs.waikato.ac.nz/ml/weka/arff.html). Each instance
  contains three fields, the pmid, the citation text (title + abstract)
  and the sense based on the name derived from the benchmark file (M1,
  M2, ...). In the citation text, the instance of the ambiguous word
  considered for disambiguation is denoted by the e tag (e.g.<e>AA</e>).

CONTACT INFORMATION

  Please contact us if you have any questions.

  Jim Mork, jmork@nlm.nih.gov
  Alan Aronson, alan@nlm.nih.gov
  Antonio Jimeno-Yepes, antonio.jimenoyepes@nih.gov
  Bridget McInness, bthomson@umn.edu

REFERENCING

  If you write a paper that has used the MSH WSD corpus, we'd
  certainly be grateful if you sent us a copy and referenced us.

  We have a published paper that provides a suitable reference:

   @article{Jimeno2011,
     title={Exploiting MeSH indexing in MEDLINE to generate a data set for
     word sense disambiguation},
     author={Jimeno-Yepes, A.J. and McInnes, B.T. and Aronson, A.R.},
     journal={BMC bioinformatics},
     volume={12},
     pages={223},
     year={2011},
     publisher={BioMed Central}
   }
