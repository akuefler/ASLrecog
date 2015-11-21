The file transposed_results.csv provides a frame-by-frame listing of each explicitly human-collected feature on our collected utterance data.

The first row labels the features collected.  This includes all the non-gloss, non-translation features available within the data set.  (A side research project could be fun to look at the gloss and/or translation features -- additional information is contained there that has the potential to help the learning problem.)

Each content row gives the filename_frameNum, where frameNum begins at 0 and goes to the last frame in the given filename.  There is a distinct filename for each utterance, as per the original data.

At the intersection of each labeled row/column pair, we have:
-- 1: If that feature was listed as occuring in that frame of that filename
-- 0: If that feature was not listed as occuring in that frame of that filename
-- ?: If that feature was not included at all in that filename/that utterance.  ? is a slightly smaller variant of NaN.  For these instances, we have no data about that feature.


Original data and human-collected features with thanks to:
Neidle, C. & Vogler, C. (2012). A New Web Interface to Facilitate Access to Corpora: Development of the ASLLRP Data Access Interface. Proceedings of the 5th Workshop on the Representation and Processing of Sign Languages: Interactions between Corpus and Lexicon, LREC 2012, Istanbul, Turkey. Data available at http://www.bu.edu/asllrp/ and http://secrets.rutgers.edu/dai/queryPages/.